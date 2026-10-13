//! ALU, fetch and control flow semantics checked through the reference interpreter (which mirrors
//! the generated WGSL helper by helper).

mod common;

use common::*;
use xenos_shader::interpreter::{Environment, Exports, Interpreter, State};
use xenos_shader::{BoolLoopConstants, Program, ShaderStage};

struct Setup {
    constants: Vec<[f32; 4]>,
    bool_loop: BoolLoopConstants,
    fetch: Vec<u32>,
    memory: Vec<u32>,
    registers: Vec<(usize, [f32; 4])>,
}

impl Default for Setup {
    fn default() -> Self {
        Setup {
            constants: vec![[0.0; 4]; 256],
            bool_loop: BoolLoopConstants::default(),
            fetch: vec![0; 192],
            memory: vec![0; 256],
            registers: Vec::new(),
        }
    }
}

fn run_stage(stage: ShaderStage, cfs: &[Cf], setup: &Setup) -> (State, Exports) {
    let ucode = assemble(cfs);
    let program = Program::decode(stage, &ucode).unwrap();
    let env = Environment {
        float_constants: &setup.constants,
        bool_loop: &setup.bool_loop,
        fetch_constants: &setup.fetch,
        shared_memory: &setup.memory,
    };
    let mut interpreter = Interpreter::new(&program, env);
    for (i, v) in &setup.registers {
        interpreter.state.registers[*i] = *v;
    }
    interpreter.run();
    (interpreter.state, interpreter.exports)
}

fn run(cfs: &[Cf], setup: &Setup) -> State {
    run_stage(ShaderStage::Vertex, cfs, setup).0
}

/// Runs one vector op `op r1, c0, c1, c2` and returns r1.
fn vector_op(op: u32, a: [f32; 4], b: [f32; 4], c2: [f32; 4]) -> [f32; 4] {
    let mut setup = Setup::default();
    setup.constants[0] = a;
    setup.constants[1] = b;
    setup.constants[2] = c2;
    let state = run(&[exece(vec![Alu::vector(op, 1, 0b1111, &[c(0), c(1), c(2)]).into()])], &setup);
    state.registers[1]
}

/// Runs a scalar op on c0 (`a` = c0.w, `b` = c0.x) after setting ps, returns (r1.x, state).
fn scalar_op(op: u32, a: f32, b: f32, ps: f32) -> (f32, State) {
    let mut setup = Setup::default();
    setup.constants[0] = [b, 0.0, 0.0, a];
    setup.constants[1] = [0.0, 0.0, 0.0, ps];
    let state = run(
        &[exece(vec![
            Alu::scalar(1, 9, 0b0000, c(1)).into(), // adds_prev: ps = c1.w + ps(0)
            Alu::scalar(op, 1, 0b0001, c(0)).into(),
        ])],
        &setup,
    );
    (state.registers[1][0], state)
}

fn bits(v: [f32; 4]) -> [u32; 4] {
    v.map(f32::to_bits)
}

#[test]
fn legacy_multiplication() {
    let inf = f32::INFINITY;
    let r = vector_op(1, [0.0, inf, f32::NAN, -0.0], [inf, 0.0, 0.0, 5.0], [0.0; 4]);
    assert_eq!(bits(r), bits([0.0, 0.0, 0.0, 0.0]));
    // mad: +0 product plus -0 is +0, not -0.
    let r = vector_op(11, [0.0, 2.0, inf, 1.0], [inf, 3.0, 0.0, 1.0], [-0.0, 1.0, 7.0, -1.0]);
    assert_eq!(bits(r), bits([0.0, 7.0, 7.0, 0.0]));
    // dp4 with 0 * inf terms.
    let r = vector_op(15, [0.0, 1.0, 2.0, 3.0], [inf, 4.0, 5.0, 6.0], [0.0; 4]);
    assert_eq!(r, [32.0; 4]);
    let r = vector_op(16, [1.0, 2.0, 3.0, 100.0], [4.0, 5.0, 6.0, 100.0], [0.0; 4]);
    assert_eq!(r, [32.0; 4]);
    let r = vector_op(17, [1.0, 2.0, 50.0, 50.0], [3.0, 4.0, 50.0, 50.0], [5.0, 9.0, 9.0, 9.0]);
    assert_eq!(r, [16.0; 4]);
    // muls / mulsc.
    assert_eq!(scalar_op(2, 0.0, f32::NAN, 0.0).0.to_bits(), 0);
    assert_eq!(scalar_op(3, f32::INFINITY, 0.0, 0.0).0.to_bits(), 0);
}

#[test]
fn min_max_nan_rules() {
    let nan = f32::NAN;
    let r = vector_op(2, [nan, 1.0, 3.0, -1.0], [1.0, nan, 2.0, -1.0], [0.0; 4]);
    assert_eq!(r[0], 1.0);
    assert!(r[1].is_nan());
    assert_eq!(&r[2..], &[3.0, -1.0]);
    let r = vector_op(3, [nan, 1.0, 3.0, -2.0], [1.0, nan, 2.0, -1.0], [0.0; 4]);
    assert_eq!(r[0], 1.0);
    assert!(r[1].is_nan());
    assert_eq!(&r[2..], &[2.0, -2.0]);
    assert_eq!(scalar_op(5, nan, 4.0, 0.0).0, 4.0);
    assert!(scalar_op(6, 4.0, nan, 0.0).0.is_nan());
}

#[test]
fn comparisons_and_conditional_moves() {
    let a = [1.0, 2.0, -0.0, f32::NAN];
    let b = [1.0, 3.0, 0.0, f32::NAN];
    assert_eq!(vector_op(4, a, b, [0.0; 4]), [1.0, 0.0, 1.0, 0.0]);
    assert_eq!(vector_op(5, a, b, [0.0; 4]), [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(vector_op(6, a, b, [0.0; 4]), [1.0, 0.0, 1.0, 0.0]);
    assert_eq!(vector_op(7, a, b, [0.0; 4]), [0.0, 1.0, 0.0, 1.0]);
    let sel = [0.0, -1.0, 2.0, f32::NAN];
    let (t, f) = ([10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]);
    assert_eq!(vector_op(12, sel, t, f), [10.0, 21.0, 22.0, 23.0]);
    assert_eq!(vector_op(13, sel, t, f), [10.0, 21.0, 12.0, 23.0]);
    assert_eq!(vector_op(14, sel, t, f), [20.0, 21.0, 12.0, 23.0]);
    assert_eq!(vector_op(8, [1.25, -1.25, 3.0, -0.5], [0.0; 4], [0.0; 4]), [0.25, 0.75, 0.0, 0.5]);
    assert_eq!(vector_op(9, [1.75, -1.75, 3.0, -0.5], [0.0; 4], [0.0; 4]), [1.0, -1.0, 3.0, -0.0]);
    assert_eq!(vector_op(10, [1.75, -1.75, 3.0, -0.5], [0.0; 4], [0.0; 4]), [1.0, -2.0, 3.0, -1.0]);
}

#[test]
fn special_vector_ops() {
    // max4 picks the first maximum.
    assert_eq!(vector_op(19, [1.0, 5.0, 3.0, 5.0], [0.0; 4], [0.0; 4]), [5.0; 4]);
    assert_eq!(vector_op(19, [-1.0, -5.0, -3.0, -0.5], [0.0; 4], [0.0; 4]), [-0.5; 4]);
    // dst.
    assert_eq!(vector_op(28, [9.0, 2.0, 3.0, 9.0], [9.0, 4.0, 9.0, 5.0], [0.0; 4]), [1.0, 8.0, 3.0, 5.0]);
    // cube with the canonical src0.zzxy swizzle: direction (x, y, z) = (0.25, 0.5, -2).
    let mut setup = Setup::default();
    setup.constants[0] = [0.25, 0.5, -2.0, 0.0];
    let state = run(&[exece(vec![Alu::vector(18, 1, 0b1111, &[c(0).swz("zzxy"), c(0).swz("yxzz")]).into()])], &setup);
    // Major axis -Z: tc = -y, sc = -x, ma = 2z, face 5.
    assert_eq!(state.registers[1], [-0.5, -0.25, -4.0, 5.0]);
    setup.constants[0] = [3.0, -1.0, 0.5, 0.0];
    let state = run(&[exece(vec![Alu::vector(18, 1, 0b1111, &[c(0).swz("zzxy"), c(0).swz("yxzz")]).into()])], &setup);
    // Major axis +X: tc = -y, sc = -z, face 0.
    assert_eq!(state.registers[1], [1.0, -0.5, 6.0, 0.0]);
    setup.constants[0] = [0.5, -3.0, 1.0, 0.0];
    let state = run(&[exece(vec![Alu::vector(18, 1, 0b1111, &[c(0).swz("zzxy"), c(0).swz("yxzz")]).into()])], &setup);
    // Major axis -Y: tc = -z, sc = x, face 3.
    assert_eq!(state.registers[1], [-1.0, 0.5, -6.0, 3.0]);
}

#[test]
fn transcendental_variants() {
    let max = f32::MAX;
    assert_eq!(scalar_op(19, 0.0, 0.0, 0.0).0, f32::INFINITY); // rcp
    assert_eq!(scalar_op(17, 0.0, 0.0, 0.0).0, max); // rcpc
    assert_eq!(scalar_op(17, -0.0, 0.0, 0.0).0, -max);
    assert_eq!(scalar_op(18, 0.0, 0.0, 0.0).0.to_bits(), 0); // rcpf
    assert_eq!(scalar_op(18, -0.0, 0.0, 0.0).0.to_bits(), 0x8000_0000);
    assert_eq!(scalar_op(19, 1.0, 0.0, 0.0).0, 1.0);
    assert_eq!(scalar_op(19, 3.0, 0.0, 0.0).0.to_bits() & 3, 0, "21-bit precision");
    assert_eq!(scalar_op(22, 4.0, 0.0, 0.0).0, 0.5); // rsq
    assert_eq!(scalar_op(20, 0.0, 0.0, 0.0).0, max); // rsqc
    assert_eq!(scalar_op(21, 0.0, 0.0, 0.0).0, 0.0); // rsqf
    assert!(scalar_op(22, -1.0, 0.0, 0.0).0.is_nan());
    assert_eq!(scalar_op(16, 0.0, 0.0, 0.0).0, f32::NEG_INFINITY); // log
    assert_eq!(scalar_op(15, 0.0, 0.0, 0.0).0, -max); // logc
    assert_eq!(scalar_op(16, 8.0, 0.0, 0.0).0, 3.0);
    assert_eq!(scalar_op(14, 0.0, 0.0, 0.0).0, 1.0); // exp
    assert_eq!(scalar_op(14, -2.0, 0.0, 0.0).0, 0.25);
    assert_eq!(scalar_op(40, 16.0, 0.0, 0.0).0, 4.0); // sqrt
    assert_eq!(scalar_op(48, 0.0, 0.0, 0.0).0, 0.0); // sin
    assert_eq!(scalar_op(49, 0.0, 0.0, 0.0).0, 1.0); // cos
}

#[test]
fn previous_scalar_ops() {
    assert_eq!(scalar_op(1, 2.0, 0.0, 5.0).0, 7.0); // adds_prev
    assert_eq!(scalar_op(26, 2.0, 0.0, 5.0).0, -3.0); // subs_prev
    assert_eq!(scalar_op(3, 2.0, 0.0, 5.0).0, 10.0); // muls_prev
    assert_eq!(scalar_op(50, 2.0, 0.0, 5.0).0, 5.0); // retain_prev
    // muls_prev2 (LIT emulation).
    assert_eq!(scalar_op(4, 2.0, 1.0, 5.0).0, 10.0);
    assert_eq!(scalar_op(4, 2.0, 0.0, 5.0).0, -f32::MAX);
    assert_eq!(scalar_op(4, 2.0, -1.0, 5.0).0, -f32::MAX);
    assert_eq!(scalar_op(4, 2.0, f32::INFINITY, 5.0).0, -f32::MAX);
    assert_eq!(scalar_op(4, 2.0, 1.0, -f32::MAX).0, -f32::MAX);
    // Two-operand ops use a = .w and b = .x.
    assert_eq!(scalar_op(0, 2.0, 3.0, 0.0).0, 5.0);
    assert_eq!(scalar_op(25, 2.0, 3.0, 0.0).0, -1.0);
    assert_eq!(scalar_op(2, 2.0, 3.0, 0.0).0, 6.0);
}

#[test]
fn predicate_set_ops() {
    let check = |op: u32, a: f32, value: f32, predicate: bool| {
        let (r, state) = scalar_op(op, a, 0.0, 0.0);
        assert_eq!((r, state.predicate), (value, predicate), "op {op} a {a}");
    };
    check(27, 0.0, 0.0, true);
    check(27, 2.0, 1.0, false);
    check(28, 2.0, 0.0, true);
    check(29, -1.0, 1.0, false);
    check(30, 0.0, 0.0, true);
    check(31, 1.0, 0.0, true); // setp_inv
    check(31, 0.0, 1.0, false);
    check(31, 5.0, 5.0, false);
    check(32, 3.0, 2.0, false); // setp_pop
    check(32, 1.0, 0.0, true);
    check(33, 7.0, f32::MAX, false); // setp_clr
    check(34, 0.0, 0.0, true); // setp_rstr
    check(34, 4.0, 4.0, false);
    // Vector predicate push: p0 from W, result from X.
    let r = vector_op(20, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0; 4]);
    assert_eq!(r, [0.0; 4]);
    let r = vector_op(22, [2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0; 4]);
    assert_eq!(r, [3.0; 4]);
    let mut setup = Setup::default();
    setup.constants[0] = [0.0, 0.0, 0.0, 0.0];
    setup.constants[1] = [0.0, 0.0, 0.0, -1.0];
    let state = run(&[exece(vec![Alu::vector(23, 1, 0b1111, &[c(0), c(1)]).into()])], &setup);
    assert!(!state.predicate);
    let state = run(&[exece(vec![Alu::vector(21, 1, 0b1111, &[c(0), c(1)]).into()])], &setup);
    assert!(state.predicate);
}

#[test]
fn predication_and_kill() {
    let mut setup = Setup::default();
    setup.constants[0] = [0.0, 0.0, 0.0, 1.0];
    setup.constants[1] = [5.0, 6.0, 7.0, 8.0];
    let state = run(
        &[exece(vec![
            Alu::scalar(27, 0, 0, c(0)).into(), // setp_eq c0.w (1) -> p0 false
            Alu::vector(2, 1, 0b1111, &[c(1), c(1)]).pred(true).into(),
            Alu::vector(2, 2, 0b1111, &[c(1), c(1)]).pred(false).into(),
        ])],
        &setup,
    );
    assert_eq!(state.registers[1], [0.0; 4]);
    assert_eq!(state.registers[2], [5.0, 6.0, 7.0, 8.0]);
    // Predicated exec evaluated at its start.
    let state = run(
        &[
            exec(vec![Alu::scalar(28, 0, 0, c(0)).into()]), // p0 = (1 != 0)
            cexec(
                Cond::Pred(true),
                false,
                vec![Alu::scalar(33, 3, 0b0001, r(0)).into(), Alu::vector(2, 4, 0b0001, &[c(1), c(1)]).into()],
            ),
            exece(vec![]),
        ],
        &setup,
    );
    assert_eq!(state.registers[3][0], f32::MAX);
    assert_eq!(state.registers[4][0], 5.0);
    // Kill only in pixel shaders.
    let program = [exece(vec![Alu::vector(25, 0, 0, &[c(1), c(0)]).into()])];
    assert!(run_stage(ShaderStage::Pixel, &program, &setup).1.killed);
    let program = [exece(vec![Alu::scalar(39, 1, 0b0001, c(0)).into()])];
    let (state, exports) = run_stage(ShaderStage::Pixel, &program, &setup);
    assert!(exports.killed);
    assert_eq!(state.registers[1][0], 1.0);
    let program = [exece(vec![Alu::scalar(36, 1, 0b0001, c(0).swz("xxxx")).into()])];
    assert!(!run_stage(ShaderStage::Pixel, &program, &setup).1.killed);
}

#[test]
fn address_register_and_relative_constants() {
    let mut setup = Setup::default();
    for (i, constant) in setup.constants.iter_mut().enumerate() {
        *constant = [i as f32; 4];
    }
    setup.constants[0] = [0.0, 0.0, 0.0, 2.6];
    setup.constants[1] = [0.0, 0.0, 0.0, -300.0];
    // maxa rounds src0.w to the nearest, then c[a0 + 10] = c13.
    let mut read = Alu::vector(0, 2, 0b1111, &[c(10), c(200)]);
    read.const_0_rel = true;
    read.const_a0 = true;
    let state = run(&[exece(vec![Alu::vector(29, 1, 0b0001, &[c(0), c(0)]).into(), read.into()])], &setup);
    assert_eq!(state.address_register, 3);
    assert_eq!(state.registers[2], [213.0; 4]);
    // Vector maxa clamps to -256, scalar maxas to 0; maxasf floors.
    let state = run(&[exece(vec![Alu::vector(29, 1, 0b0001, &[c(1), c(1)]).into()])], &setup);
    assert_eq!(state.address_register, -256);
    let state = run(&[exece(vec![Alu::scalar(23, 1, 0b0001, c(1)).into()])], &setup);
    assert_eq!(state.address_register, 0);
    let state = run(&[exece(vec![Alu::scalar(24, 1, 0b0001, c(0)).into()])], &setup);
    assert_eq!(state.address_register, 2);
    // Out of range relative constants read zero.
    let mut far = Alu::vector(0, 3, 0b1111, &[c(254), c(0)]);
    far.const_0_rel = true;
    far.const_a0 = true;
    let state = run(&[exece(vec![Alu::vector(29, 1, 0b0001, &[c(0), c(0)]).into(), far.into()])], &setup);
    assert_eq!(state.registers[3], [0.0, 0.0, 0.0, 2.6]);
}

#[test]
fn constant_scalar_ops_register_encoding() {
    let mut setup = Setup::default();
    setup.constants[12] = [0.0, 0.0, 0.0, 3.0];
    setup.registers.push((11, [4.0, 0.0, 0.0, 0.0]));
    let mut mulsc = Alu::scalar(43, 9, 0b0001, c(12));
    mulsc.src[2].temp = true;
    mulsc.src[2].swizzle = 0b10 << 2;
    let mut subsc = mulsc;
    subsc.scalar_op = 47;
    subsc.scalar_dest = 8;
    subsc.src[2].negate = true;
    let state = run(&[exece(vec![mulsc.into(), subsc.into()])], &setup);
    assert_eq!(state.registers[9][0], 12.0);
    // Negation applies to both operands: -3 - (-4).
    assert_eq!(state.registers[8][0], 1.0);
}

#[test]
fn results_clamp_and_export_constants() {
    let mut setup = Setup::default();
    setup.constants[0] = [-2.0, 0.5, 3.0, f32::NAN];
    let mut sat = Alu::vector(2, 1, 0b1111, &[c(0), c(0)]);
    sat.vector_clamp = true;
    sat.scalar_op = 0;
    sat.scalar_dest = 2;
    sat.scalar_mask = 0b0001;
    sat.scalar_clamp = true;
    sat.src[2] = c(0).swz("xyzy"); // adds: a = .y (0.5), b = .x (-2)
    let state = run(&[exece(vec![sat.into()])], &setup);
    assert_eq!(&state.registers[1][..3], &[0.0, 0.5, 1.0]);
    assert_eq!(state.registers[2][0], 0.0);
    assert_eq!(state.previous_scalar, -1.5, "ps is not clamped");
    // Export: vector XY, scalar Z (ps of the previous instruction), constant 1 in W.
    let mut e = Alu::vector(2, 0, 0b1011, &[c(0), c(0)]).export();
    e.scalar_op = 50;
    e.scalar_mask = 0b1100;
    let set_ps = Alu::scalar(0, 0, 0, c(0).swz("xyzy"));
    let (_, exports) = run_stage(ShaderStage::Pixel, &[exece(vec![set_ps.into(), e.into()])], &setup);
    assert_eq!(exports.colors[0], [-2.0, 0.5, -1.5, 1.0]);
}

#[test]
fn loops_and_relative_addressing() {
    let mut setup = Setup::default();
    for (i, constant) in setup.constants.iter_mut().enumerate() {
        *constant = [i as f32, 0.0, 0.0, 0.0];
    }
    // i0: count 3, start 4, step 2 -> aL = 4, 6, 8.
    setup.bool_loop.loops[0] = 3 | 4 << 8 | 2 << 16;
    // i1: count 2, start 1, step -1.
    setup.bool_loop.loops[1] = 2 | 1 << 8 | 0xFF << 16;
    let mut acc = Alu::vector(0, 1, 0b0001, &[r(1), c(0)]);
    acc.const_0_rel = true;
    let mut inner = Alu::vector(0, 2, 0b0001, &[r(2), c(100)]);
    inner.const_0_rel = true;
    let state = run(
        &[
            Cf::LoopStart { loop_id: 0, skip: 6, repeat: false },
            exec(vec![acc.into()]),
            Cf::LoopStart { loop_id: 1, skip: 5, repeat: false },
            exec(vec![inner.into()]),
            Cf::LoopEnd { loop_id: 1, body: 3, pred_break: None },
            Cf::LoopEnd { loop_id: 0, body: 1, pred_break: None },
            exece(vec![]),
        ],
        &setup,
    );
    assert_eq!(state.registers[1][0], 4.0 + 6.0 + 8.0);
    // Inner loop: aL = 1, 0 three times -> c101 + c100 per outer iteration.
    assert_eq!(state.registers[2][0], 3.0 * (101.0 + 100.0));
    // Predicated break after the second iteration; zero-count loops are skipped.
    setup.bool_loop.loops[2] = 10;
    setup.bool_loop.loops[3] = 0;
    setup.constants[50] = [1.0, 0.0, 0.0, 1.0];
    let state = run(
        &[
            Cf::LoopStart { loop_id: 2, skip: 3, repeat: false },
            exec(vec![
                Alu::vector(0, 3, 0b0001, &[r(3), c(50)]).into(),
                Alu::scalar(30, 0, 0, r(3).swz("xxxx")).into(), // p0 = r3.x - 2 >= 0 -> uses r3.x
            ]),
            Cf::LoopEnd { loop_id: 2, body: 1, pred_break: Some(true) },
            Cf::LoopStart { loop_id: 3, skip: 6, repeat: false },
            exec(vec![Alu::vector(0, 4, 0b0001, &[r(4), c(50)]).into()]),
            Cf::LoopEnd { loop_id: 3, body: 4, pred_break: None },
            exece(vec![]),
        ],
        &setup,
    );
    assert_eq!(state.registers[3][0], 1.0, "breaks after the first iteration since r3.x >= 0");
    assert_eq!(state.registers[4][0], 0.0);
    assert_eq!(state.loop_count, [0; 4]);
}

#[test]
fn calls_and_jumps() {
    let mut setup = Setup::default();
    setup.constants[0] = [1.0, 0.0, 0.0, 0.0];
    setup.bool_loop.bools[0] = 1 << 3;
    let inc: Ins = Alu::vector(0, 1, 0b0001, &[r(1), c(0)]).into();
    let state = run(
        &[
            Cf::Call { target: 5, cond: Cond::Always },
            Cf::Call { target: 5, cond: Cond::Bool(3, true) },
            Cf::Call { target: 5, cond: Cond::Bool(3, false) },
            Cf::Jmp { target: 6, cond: Cond::Always },
            exec(vec![inc, inc, inc]),
            exec(vec![inc]),
            Cf::Ret,
            exece(vec![inc]),
        ],
        &setup,
    );
    // The first two calls run the subroutine at 5 (one add, then ret), the third is skipped, jmp
    // skips the exec at 4, the ret at 6 is a no-op outside a call, and 7 adds once more.
    assert_eq!(state.registers[1][0], 3.0);
}

#[test]
fn vertex_fetch_formats() {
    let mut setup = Setup::default();
    // Vertex buffer at dword 16, 8 dwords, endianness 8in32 (big-endian dwords in memory).
    setup.fetch[0] = (16 << 2) | 3;
    setup.fetch[1] = (8 << 2) | 2;
    let be = |v: u32| v.swap_bytes();
    setup.memory[16] = be(1.5f32.to_bits());
    setup.memory[17] = be((-2.0f32).to_bits());
    setup.memory[18] = be(0x7F80_01FF); // 8_8_8_8: x = 0xFF, y = 0x01, z = 0x80, w = 0x7F
    setup.memory[19] = be(0x7C00_3C00); // half: x = 1.0, y = 65536 (Xenos extended range)
    setup.memory[20] = be(0xFFFF_8000); // 16_16 signed: x = -32768, y = -1
    setup.registers.push((0, [0.0; 4]));
    let fetch = |format: u32, offset: i32, signed: bool, integer: bool| VFetch {
        dest: 1,
        fetch_constant: 0,
        format,
        offset,
        signed,
        integer,
        stride: 0,
        ..Default::default()
    };
    let get = |setup: &Setup, f: VFetch| run(&[exece(vec![f.into()])], setup).registers[1];
    assert_eq!(get(&setup, fetch(37, 0, false, false))[..2], [1.5, -2.0]);
    assert_eq!(get(&setup, fetch(6, 2, false, false)), [1.0, 1.0 / 255.0, 128.0 / 255.0, 127.0 / 255.0]);
    assert_eq!(get(&setup, fetch(6, 2, true, false)), [-1.0 / 127.0, 1.0 / 127.0, -1.0, 1.0]);
    assert_eq!(get(&setup, fetch(6, 2, true, true)), [-1.0, 1.0, -128.0, 127.0]);
    assert_eq!(get(&setup, fetch(31, 3, false, false))[..2], [1.0, 65536.0]);
    assert_eq!(get(&setup, fetch(25, 4, true, true))[..2], [-32768.0, -1.0]);
    assert_eq!(get(&setup, fetch(25, 4, true, false))[..2], [-1.0, -1.0 / 32767.0]);
    // Reads past the end of the buffer return zero.
    setup.memory[24] = be(1.0f32.to_bits());
    assert_eq!(get(&setup, fetch(36, 8, false, false))[0], 0.0);
    assert_eq!(get(&setup, fetch(36, 7, false, false))[0], 0.0);
    // Indexed fetch with stride and index rounding, and a mini fetch reusing the address.
    setup.registers[0] = (0, [0.6, 0.0, 0.0, 0.0]);
    let full = VFetch {
        dest: 1,
        dest_swizzle: "x___",
        fetch_constant: 0,
        format: 36,
        stride: 2,
        round_index: true,
        ..Default::default()
    };
    let mini = VFetch { dest: 2, dest_swizzle: "1x0_", mini: true, format: 36, offset: 1, ..full };
    let state = run(&[exece(vec![full.into(), mini.into()])], &setup);
    assert_eq!(state.registers[1][0].to_bits(), 0x7F80_01FF, "floor(0.6 + 0.5) * 2 + 16 = dword 18");
    assert_eq!(state.registers[2][0], 1.0);
    assert_eq!(state.registers[2][1].to_bits(), 0x7C00_3C00);
    assert_eq!(state.registers[2][2], 0.0);
    // Exponent adjust.
    let f = VFetch { exp_adjust: -1, ..fetch(36, 0, false, false) };
    assert_eq!(get(&setup, f)[0], 0.75);
}
