mod common;

use common::*;
use xenos_shader::{ShaderStage, disassemble};

fn check(stage: ShaderStage, cfs: &[Cf], expected: &str) {
    let ucode = assemble(cfs);
    let text = disassemble(stage, &ucode).unwrap();
    if text != expected {
        panic!("disassembly mismatch\n--- expected\n{expected}--- actual\n{text}");
    }
}

#[test]
fn vertex_fetch_and_exports() {
    check(
        ShaderStage::Vertex,
        &[
            exec(vec![
                VFetch {
                    dest: 1,
                    dest_swizzle: "xyz1",
                    fetch_constant: 0,
                    format: 57,
                    stride: 3,
                    ..Default::default()
                }
                .into(),
                VFetch { dest: 2, dest_swizzle: "xy__", mini: true, format: 37, offset: 3, ..Default::default() }
                    .into(),
                Alu::vector(15, 3, 0b0001, &[r(1), c(0)]).into(),
            ]),
            Cf::Alloc { kind: 1, size: 0 },
            exece(vec![
                Alu::vector(2, 62, 0b1111, &[r(1), r(1)]).export().into(),
                Alu::vector(1, 0, 0b1111, &[r(2), c(1)]).export().into(),
            ]),
        ],
        "/*    0.0 */       exec
/*    2   */          vfetch_full r1.xyz1, r0.x, vf95, DataFormat=FMT_32_32_32_FLOAT, Stride=3
/*    3   */          vfetch_mini r2.xy__, Offset=3, DataFormat=FMT_32_32_FLOAT
/*    4   */          dp4 r3.x___, r1, c0
/*    0.1 */       alloc position
/*    1.0 */       exece
/*    5   */          max oPos, r1, r1
/*    6   */          mul o0, r2, c1
/*    1.1 */       cnop
",
    );
}

#[test]
fn vertex_fetch_attributes() {
    check(
        ShaderStage::Vertex,
        &[exece(vec![
            VFetch {
                dest: 4,
                dest_swizzle: "zyx0",
                src: 3,
                src_component: 2,
                fetch_constant: 7,
                format: 6,
                signed: true,
                integer: true,
                round_index: true,
                exp_adjust: -3,
                offset: -2,
                stride: 12,
                prefetch: 3,
                predicated: Some(false),
                ..Default::default()
            }
            .into(),
            VFetch { dest: 5, dest_swizzle: "____", fetch_constant: 9, format: 26, stride: 2, ..Default::default() }
                .into(),
        ])],
        "/*    0.0 */       exece
/*    1   */    (!p0) vfetch_full r4.zyx0, r3.z, vf88, RoundIndex=true, ExpAdjust=-3, Offset=-2, DataFormat=FMT_8_8_8_8, Stride=12, Signed=true, NumFormat=integer, PrefetchCount=4
/*    2   */          vfetch_full r5._, r0.x, vf86, DataFormat=FMT_16_16_16_16, Stride=2
/*    0.1 */       cnop
",
    );
}

#[test]
fn control_flow_instructions() {
    check(
        ShaderStage::Pixel,
        &[
            cexec(Cond::Bool(5, true), false, vec![Alu::vector(0, 1, 0b1111, &[r(0), r(0)]).into()]),
            Cf::Exec { cond: Cond::Bool(6, false), end: false, instrs: vec![], yield_: true },
            cexec(Cond::Pred(true), false, vec![]),
            Cf::Exec { cond: Cond::Pred(false), end: false, instrs: vec![], yield_: true },
            Cf::LoopStart { loop_id: 2, skip: 7, repeat: false },
            Cf::LoopEnd { loop_id: 2, body: 5, pred_break: None },
            Cf::LoopEnd { loop_id: 3, body: 5, pred_break: Some(true) },
            Cf::LoopStart { loop_id: 4, skip: 7, repeat: true },
            Cf::Call { target: 12, cond: Cond::Always },
            Cf::Call { target: 12, cond: Cond::Bool(3, false) },
            Cf::Call { target: 12, cond: Cond::Pred(true) },
            Cf::Ret,
            Cf::Jmp { target: 14, cond: Cond::Always },
            Cf::Jmp { target: 15, cond: Cond::Bool(1, true) },
            Cf::Jmp { target: 15, cond: Cond::Pred(false) },
            Cf::Alloc { kind: 3, size: 2 },
            Cf::Alloc { kind: 2, size: 0 },
            Cf::MarkVsFetchDone,
            cexec(Cond::BoolClean(9, true), true, vec![]),
            cexec(Cond::Bool(10, false), true, vec![]),
            cexec(Cond::Pred(true), true, vec![]),
            exece(vec![]),
        ],
        "/*    0.0 */       cexec b5    // PredicateClean=false
/*   11   */          add r1, r0, r0
/*    0.1 */       cexec !b6, Yield=true    // PredicateClean=false
/*    1.0 */  (p0) exec
/*    1.1 */ (!p0) exec Yield=true
/*    2.0 */       loop i2, L7
                label L5
/*    2.1 */       endloop i2, L5
/*    3.0 */  (p0) endloop i3, L5
                label L7
/*    3.1 */       loop i4, L7, Repeat=true
/*    4.0 */       call L12
/*    4.1 */       ccall !b3, L12
/*    5.0 */  (p0) call L12
/*    5.1 */       ret
                label L12
/*    6.0 */       jmp L14
/*    6.1 */       cjmp b1, L15
                label L14
/*    7.0 */ (!p0) jmp L15
                label L15
/*    7.1 */       alloc export = 2
/*    8.0 */       alloc colors
/*    8.1 */       mark_vs_fetch_done
/*    9.0 */       cexece b9
/*    9.1 */       cexece !b10    // PredicateClean=false
/*   10.0 */  (p0) exece
/*   10.1 */       exece
",
    );
}

#[test]
fn alu_operands_and_modifiers() {
    let mut rel = Alu::vector(11, 5, 0b0111, &[r(1).rel().neg(), c(3), c(4)]);
    rel.const_0_rel = true;
    rel.const_1_rel = true;
    rel.const_a0 = true;
    rel.vector_clamp = true;
    rel.scalar_op = 0; // adds
    rel.scalar_dest = 6;
    rel.scalar_mask = 0b1000;
    rel.scalar_clamp = true;
    let mut abs_consts = Alu::vector(16, 7, 0b1111, &[r(2).abs().swz("zyxw"), c(8).swz("wwww").neg()]);
    abs_consts.abs_constants = true;
    abs_consts.vector_dest_rel = true;
    let mut mulsc = Alu::scalar(43, 9, 0b0011, c(12));
    mulsc.src[2].temp = true; // selects odd register bits
    mulsc.src[2].swizzle = 0b10 << 2;
    let mut maxa_loop = Alu::vector(29, 0, 0b0001, &[c(20), c(21)]);
    maxa_loop.const_0_rel = true;
    maxa_loop.const_1_rel = false;
    check(
        ShaderStage::Pixel,
        &[exece(vec![
            rel.into(),
            abs_consts.pred(true).into(),
            mulsc.into(),
            Alu::scalar(33, 0, 0b0001, r(0)).pred(false).into(),
            Alu::scalar(50, 0, 0, r(0)).into(),
            maxa_loop.into(),
        ])],
        "/*    0.0 */       exece
/*    1   */          mad_sat r5.xyz_, -r[1+aL], c[3+a0], c[4+a0]
              +       adds_sat r6.___w, c[4+a0].wz
/*    2   */     (p0) dp3 r[7+aL], r_abs[2].zyxw, -c_abs[8].wwww
/*    3   */          mulsc r9.xy__, c12.w, r11.x
/*    4   */    (!p0) setp_clr r0.x___
/*    5   */          nop
/*    6   */          maxa r0.x___, c[20+aL], c21
/*    0.1 */       cnop
",
    );
}

#[test]
fn exports_with_constants() {
    // Export writes: vector XY, scalar Z, constant 1 in W (both masks), relative scalar dest =
    // constant 0 in unwritten components.
    let mut a = Alu::vector(0, 0, 0b1011, &[r(1), r(2)]).export();
    a.scalar_op = 22; // rsq
    a.scalar_mask = 0b1100;
    a.src[2] = r(3);
    let mut b = Alu::vector(2, 1, 0b0001, &[r(4), r(4)]).export();
    b.scalar_dest_rel = true;
    let mut depth = Alu::vector(2, 61, 0b0001, &[r(5), r(5)]).export();
    depth.vector_clamp = true;
    check(
        ShaderStage::Pixel,
        &[exece(vec![a.into(), b.into(), depth.into()])],
        "/*    0.0 */       exece
/*    1   */          add oC0.xy_1, r1, r2
              +       rsq oC0.__z_, r3.w
/*    2   */          max oC1.x000, r4, r4
/*    3   */          max_sat oDepth.x___, r5, r5
/*    0.1 */       cnop
",
    );
}

#[test]
fn texture_fetch_variants() {
    check(
        ShaderStage::Pixel,
        &[exece(vec![
            TFetch {
                dest: 1,
                dest_swizzle: "xyzw",
                src: 2,
                src_swizzle: "xx",
                dimension: 0,
                fetch_constant: 3,
                ..Default::default()
            }
            .into(),
            TFetch {
                dest: 2,
                dest_swizzle: "wzy_",
                src: 0,
                src_swizzle: "yzx",
                dimension: 3,
                fetch_constant: 5,
                unnormalized: true,
                fetch_valid_only: false,
                mag: 0,
                min: 1,
                mip: 2,
                aniso: 3,
                computed_lod: false,
                register_lod: true,
                lod_bias_16ths: -24,
                offset_halves: [1, -2, 3],
                predicated: Some(true),
                ..Default::default()
            }
            .into(),
            TFetch {
                dest: 3,
                dimension: 2,
                fetch_constant: 1,
                vol_mag: 1,
                vol_min: 0,
                register_gradients: true,
                offset_halves: [0, 0, -1],
                ..Default::default()
            }
            .into(),
            TFetch { opcode: 24, src: 4, src_swizzle: "w", ..Default::default() }.into(),
            TFetch { opcode: 25, src: 5, src_swizzle: "xyz", ..Default::default() }.into(),
            TFetch { opcode: 18, dest: 6, src: 7, src_swizzle: "xy", dest_swizzle: "xyzw", ..Default::default() }.into(),
        ])],
        "/*    0.0 */       exece
/*    1   */          tfetch1D r1, r2.x, tf3
/*    2   */     (p0) tfetchCube r2.wzy_, r0.yzx, tf5, FetchValidOnly=false, UnnormalizedTextureCoords=true, MagFilter=point, MinFilter=linear, MipFilter=basemap, AnisoFilter=max4to1, UseComputedLOD=false, UseRegisterLOD=true, LODBias=-1.5, OffsetX=0.5, OffsetY=-1, OffsetZ=1.5
/*    3   */          tfetch3D r3, r0.xyz, tf1, VolMagFilter=linear, VolMinFilter=point, UseRegisterGradients=true, OffsetZ=-0.5
/*    4   */          setTexLOD r4.w
/*    5   */          setGradientH r5.xyz
/*    6   */          getGradients r6, r7.xy, tf0
/*    0.1 */       cnop
",
    );
    // Other getters.
    check(
        ShaderStage::Pixel,
        &[exece(vec![
            TFetch { opcode: 16, dimension: 1, dest: 1, src_swizzle: "xy", ..Default::default() }.into(),
            TFetch { opcode: 17, dimension: 2, dest: 2, src_swizzle: "xyz", ..Default::default() }.into(),
            TFetch { opcode: 19, dimension: 3, dest: 3, src_swizzle: "xyz", ..Default::default() }.into(),
            TFetch { opcode: 26, src: 9, src_swizzle: "zyx", ..Default::default() }.into(),
            TFetch { dimension: 0, dest: 4, src_swizzle: "xy", fetch_constant: 2, ..Default::default() }.into(),
        ])],
        "/*    0.0 */       exece
/*    1   */          getBCF2D r1, r0.xy, tf0
/*    2   */          getCompTexLOD3D r2, r0.xyz, tf0
/*    3   */          getWeightsCube r3, r0.xyz, tf0
/*    4   */          setGradientV r9.zyx
/*    5   */          tfetch1D r4, r0.xy, tf2
/*    0.1 */       cnop
",
    );
}

#[test]
fn serialize_and_scalar_only_lines() {
    let mut both = Alu::vector(4, 1, 0b0001, &[r(1), c(2).neg()]);
    both.scalar_op = 14; // exp
    both.scalar_dest = 2;
    both.scalar_mask = 0b0010;
    both.src[2] = r(3).swz("xxxz");
    check(
        ShaderStage::Vertex,
        &[exece(vec![
            Ins::SerializedFetch(
                VFetch { fetch_constant: 1, format: 36, dest_swizzle: "x___", ..Default::default() }.encode(),
            ),
            both.into(),
            Alu::scalar(16, 3, 0b0001, c(4).swz("yyyy")).into(),
        ])],
        "/*    0.0 */       exece
/*    1   */          serialize
                      vfetch_full r1.x___, r0.x, vf94, DataFormat=FMT_32_FLOAT, Stride=4
/*    2   */          seq r1.x___, r1, -c2
              +       exp r2._y__, r3.z
/*    3   */          log r3.x___, c4.y
/*    0.1 */       cnop
",
    );
}
