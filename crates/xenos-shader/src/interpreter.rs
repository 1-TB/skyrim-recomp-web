//! CPU reference interpreter of Xenos microcode with the same semantics as the generated WGSL
//! (control flow, every ALU operation, vertex fetch from shared memory and exports). Texture
//! fetches produce zero. Useful for testing, debugging and CPU-side evaluation of vertex shaders.

use crate::ShaderStage;
use crate::bindings::BoolLoopConstants;
use crate::parsed::{
    AddressingMode, CfCondition, InstructionOperand, InstructionResult, ParsedAlu, ParsedCf, ParsedTextureFetch,
    ParsedVertexFetch, StorageSource, StorageTarget, SwizzleSource,
};
use crate::program::{Instruction, Program};
use crate::ucode::{self, FetchOpcode, scalar_op as s, vector_op as v, vertex_format as vfmt};

/// Guest physical memory as vertex fetches read it.
pub trait SharedMemoryWords {
    /// Dword `index`: bytes `4 * index..4 * index + 3` as stored (little-endian), 0 past the end.
    fn word(&self, index: u32) -> u32;
}

impl SharedMemoryWords for [u32] {
    fn word(&self, index: u32) -> u32 {
        self.get(index as usize).copied().unwrap_or(0)
    }
}

impl SharedMemoryWords for Vec<u32> {
    fn word(&self, index: u32) -> u32 {
        self.as_slice().word(index)
    }
}

/// Inputs of an invocation.
pub struct Environment<'a> {
    /// The 256 float constants of the shader's stage.
    pub float_constants: &'a [[f32; 4]],
    pub bool_loop: &'a BoolLoopConstants,
    pub fetch_constants: &'a [u32],
    /// Guest physical memory.
    pub shared_memory: &'a dyn SharedMemoryWords,
}

/// Values written by the shader.
#[derive(Clone, Debug, PartialEq)]
pub struct Exports {
    pub position: [f32; 4],
    pub interpolators: [[f32; 4]; 16],
    pub point_size_edge_flag_kill_vertex: [f32; 3],
    pub colors: [[f32; 4]; 4],
    pub depth: Option<f32>,
    pub killed: bool,
    pub export_address: [f32; 4],
    pub export_data: [[f32; 4]; 5],
}

impl Default for Exports {
    fn default() -> Self {
        Exports {
            position: [0.0; 4],
            interpolators: [[0.0; 4]; 16],
            point_size_edge_flag_kill_vertex: [-1.0, 0.0, 0.0],
            colors: [[0.0; 4]; 4],
            depth: None,
            killed: false,
            export_address: [0.0; 4],
            export_data: [[0.0; 4]; 5],
        }
    }
}

/// Architectural state of an invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub registers: [[f32; 4]; 64],
    pub previous_scalar: f32,
    pub predicate: bool,
    pub address_register: i32,
    /// aL stack, `[0]` is the innermost loop.
    pub loop_address: [i32; 4],
    /// Remaining iterations stack.
    pub loop_count: [u32; 4],
    pub call_stack: Vec<u32>,
    pub vfetch_address: i32,
    pub vfetch_end: i32,
    pub tfetch_lod: f32,
    pub tfetch_gradients: [[f32; 3]; 2],
}

impl Default for State {
    fn default() -> Self {
        State {
            registers: [[0.0; 4]; 64],
            previous_scalar: 0.0,
            predicate: false,
            address_register: 0,
            loop_address: [0; 4],
            loop_count: [0; 4],
            call_stack: Vec::new(),
            vfetch_address: 0,
            vfetch_end: 0,
            tfetch_lod: 0.0,
            tfetch_gradients: [[0.0; 3]; 2],
        }
    }
}

pub const FLT_MAX: f32 = f32::MAX;

pub fn mul(a: f32, b: f32) -> f32 {
    if a == 0.0 || b == 0.0 { 0.0 } else { a * b }
}

pub fn max(a: f32, b: f32) -> f32 {
    if a >= b { a } else { b }
}

pub fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Rounds a finite value to 21 mantissa bits, halfway away from zero.
pub fn reduce_precision(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7F80_0000 == 0x7F80_0000 {
        return value;
    }
    let truncated = bits & !3;
    let rounded = truncated.wrapping_add(4);
    let result =
        if bits & 3 >= 2 { if rounded & 0x7F80_0000 == 0x7F80_0000 { truncated } else { rounded } } else { truncated };
    f32::from_bits(result)
}

fn inf_to_max(v: f32) -> f32 {
    if v == f32::INFINITY {
        FLT_MAX
    } else if v == f32::NEG_INFINITY {
        -FLT_MAX
    } else {
        v
    }
}

fn inf_to_zero(v: f32) -> f32 {
    if v == f32::INFINITY {
        0.0
    } else if v == f32::NEG_INFINITY {
        -0.0
    } else {
        v
    }
}

/// `cube` with the operand already swizzled (src0.zzxy).
pub fn cube(s: [f32; 4]) -> [f32; 4] {
    let (x, y, z) = (s[2], s[3], s[0]);
    let mut r = if z.abs() >= x.abs() && z.abs() >= y.abs() {
        let n = z < 0.0;
        [-y, if n { -x } else { x }, z, if n { 5.0 } else { 4.0 }]
    } else if y.abs() >= x.abs() {
        let n = y < 0.0;
        [if n { -z } else { z }, x, y, if n { 3.0 } else { 2.0 }]
    } else {
        let n = x < 0.0;
        [-y, if n { z } else { -z }, x, if n { 1.0 } else { 0.0 }]
    };
    r[2] *= 2.0;
    r
}

pub fn max4(v: [f32; 4]) -> f32 {
    if v[0] >= v[1] && v[0] >= v[2] && v[0] >= v[3] {
        v[0]
    } else if v[1] >= v[2] && v[1] >= v[3] {
        v[1]
    } else if v[2] >= v[3] {
        v[2]
    } else {
        v[3]
    }
}

/// Xenos float16 (no infinity or NaN).
pub fn half_to_f32(h: u32) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (h >> 10) & 0x1F;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return sign * m * (2.0f32).powi(-24);
    }
    sign * (1.0 + m / 1024.0) * (2.0f32).powi(e as i32 - 15)
}

pub fn swap32(v: u32, endian: u32) -> u32 {
    let mut r = v;
    if endian == 1 || endian == 2 {
        r = ((r >> 8) & 0x00FF_00FF) | ((r & 0x00FF_00FF) << 8);
    }
    if endian == 2 || endian == 3 {
        r = r.rotate_left(16);
    }
    r
}

fn compare(op: &str, a: f32, b: f32) -> bool {
    match op {
        "==" => a == b,
        ">" => a > b,
        ">=" => a >= b,
        _ => a != b,
    }
}

fn vector_comparison(op: u32) -> &'static str {
    match op {
        v::SEQ | v::CNDEQ | v::SETP_EQ_PUSH | v::KILL_EQ => "==",
        v::SGT | v::CNDGT | v::SETP_GT_PUSH | v::KILL_GT => ">",
        v::SGE | v::CNDGE | v::SETP_GE_PUSH | v::KILL_GE => ">=",
        _ => "!=",
    }
}

const DISPATCH_LIMIT: u32 = 1 << 20;

pub struct Interpreter<'a> {
    pub program: &'a Program,
    pub env: Environment<'a>,
    pub state: State,
    pub exports: Exports,
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program, env: Environment<'a>) -> Self {
        Interpreter { program, env, state: State::default(), exports: Exports::default() }
    }

    fn bool_constant(&self, index: u32) -> bool {
        self.env.bool_loop.bools[(index >> 5) as usize] >> (index & 31) & 1 != 0
    }

    fn condition(&self, c: &CfCondition) -> bool {
        match c {
            CfCondition::Always => true,
            CfCondition::Bool { index, value } => self.bool_constant(*index) == *value,
            CfCondition::Predicate(value) => self.state.predicate == *value,
        }
    }

    /// Runs the program from the first control flow instruction.
    pub fn run(&mut self) {
        let cf_count = self.program.cf.len() as u32;
        let mut pc = 0u32;
        let mut iterations = 0;
        while pc < cf_count && iterations < DISPATCH_LIMIT {
            iterations += 1;
            let entry = &self.program.cf[pc as usize];
            let mut next = pc + 1;
            match entry.cf {
                ParsedCf::Nop | ParsedCf::Alloc { .. } | ParsedCf::MarkVsFetchDone => {}
                ParsedCf::Exec(exec) => {
                    if self.condition(&exec.condition) {
                        for instruction in &entry.instructions {
                            match &instruction.instruction {
                                Instruction::Alu(alu) => self.alu(alu),
                                Instruction::VertexFetch(vf) => self.vertex_fetch(vf),
                                Instruction::TextureFetch(tf) => self.texture_fetch(tf),
                            }
                        }
                        if exec.is_end {
                            return;
                        }
                    }
                }
                ParsedCf::LoopStart { loop_constant, is_repeat, skip_address } => {
                    let lc = self.env.bool_loop.loops[loop_constant as usize];
                    let count = lc & 0xFF;
                    if count == 0 {
                        next = skip_address;
                    } else {
                        let st = &mut self.state;
                        st.loop_count = [count, st.loop_count[0], st.loop_count[1], st.loop_count[2]];
                        let start = if is_repeat { st.loop_address[0] } else { ((lc >> 8) & 0xFF) as i32 };
                        st.loop_address = [start, st.loop_address[0], st.loop_address[1], st.loop_address[2]];
                    }
                }
                ParsedCf::LoopEnd { loop_constant, body_address, predicated_break } => {
                    let st = &mut self.state;
                    let remaining = st.loop_count[0].wrapping_sub(1);
                    let mut cont = remaining != 0;
                    if let Some(c) = predicated_break {
                        cont = cont && st.predicate != c;
                    }
                    if cont {
                        st.loop_count[0] = remaining;
                        let step = ((self.env.bool_loop.loops[loop_constant as usize] << 8) as i32) >> 24;
                        st.loop_address[0] += step;
                        next = body_address;
                    } else {
                        st.loop_count = [st.loop_count[1], st.loop_count[2], st.loop_count[3], 0];
                        st.loop_address = [st.loop_address[1], st.loop_address[2], st.loop_address[3], 0];
                    }
                }
                ParsedCf::Call { target, condition } => {
                    if self.condition(&condition) && self.state.call_stack.len() < 4 {
                        self.state.call_stack.push(pc + 1);
                        next = target;
                    }
                }
                ParsedCf::Return => {
                    if let Some(r) = self.state.call_stack.pop() {
                        next = r;
                    }
                }
                ParsedCf::Jump { target, condition } => {
                    if self.condition(&condition) {
                        next = target;
                    }
                }
            }
            pc = next;
        }
    }

    fn register_index(&self, index: u32, addressing: AddressingMode) -> usize {
        (match addressing {
            AddressingMode::Absolute => index as i32,
            AddressingMode::LoopRelative => index as i32 + self.state.loop_address[0],
            AddressingMode::AddressRegisterRelative => index as i32 + self.state.address_register,
        } & 63) as usize
    }

    fn storage(&self, op: &InstructionOperand) -> [f32; 4] {
        match op.source {
            StorageSource::Register => self.state.registers[self.register_index(op.index, op.addressing)],
            StorageSource::ConstantFloat => {
                let index = match op.addressing {
                    AddressingMode::Absolute => op.index as i32,
                    AddressingMode::AddressRegisterRelative => op.index as i32 + self.state.address_register,
                    AddressingMode::LoopRelative => op.index as i32 + self.state.loop_address[0],
                };
                if (0..256).contains(&index) {
                    self.env.float_constants.get(index as usize).copied().unwrap_or([0.0; 4])
                } else {
                    [0.0; 4]
                }
            }
            _ => [0.0; 4],
        }
    }

    fn modify(op: &InstructionOperand, mut x: f32) -> f32 {
        if op.is_absolute_value {
            x = x.abs();
        }
        if op.is_negated {
            x = -x;
        }
        x
    }

    fn operand_component(&self, op: &InstructionOperand, i: u32) -> f32 {
        let storage = self.storage(op);
        let x = match op.component(i) {
            SwizzleSource::Zero => 0.0,
            SwizzleSource::One => 1.0,
            c => storage[c.component().unwrap() as usize],
        };
        Self::modify(op, x)
    }

    fn operand_vec4(&self, op: &InstructionOperand) -> [f32; 4] {
        std::array::from_fn(|i| self.operand_component(op, i as u32))
    }

    fn store(&mut self, result: &InstructionResult, value: Option<[f32; 4]>) {
        let mask = result.used_write_mask();
        if mask == 0 {
            return;
        }
        let mut value = value.unwrap_or([0.0; 4]);
        if result.is_clamped {
            value = value.map(|x| x.clamp(0.0, 1.0));
        }
        let component = |i: usize| match result.components[i] {
            SwizzleSource::Zero => 0.0,
            SwizzleSource::One => 1.0,
            c => value[c.component().unwrap() as usize],
        };
        let target: &mut [f32] = match result.target {
            StorageTarget::None => return,
            StorageTarget::Register => {
                let index = self.register_index(result.index, result.addressing);
                &mut self.state.registers[index]
            }
            StorageTarget::Interpolator => &mut self.exports.interpolators[(result.index & 15) as usize],
            StorageTarget::Position => &mut self.exports.position,
            StorageTarget::PointSizeEdgeFlagKillVertex => &mut self.exports.point_size_edge_flag_kill_vertex,
            StorageTarget::ExportAddress => &mut self.exports.export_address,
            StorageTarget::ExportData => &mut self.exports.export_data[result.index.min(4) as usize],
            StorageTarget::Color => &mut self.exports.colors[(result.index & 3) as usize],
            StorageTarget::Depth => {
                let d = component(0);
                self.exports.depth = Some(if result.is_clamped { d } else { d.clamp(0.0, 1.0) });
                return;
            }
        };
        for (i, t) in target.iter_mut().enumerate() {
            if mask & (1 << i) != 0 {
                *t = component(i);
            }
        }
    }

    pub fn alu(&mut self, alu: &ParsedAlu) {
        if alu.is_nop() {
            return;
        }
        if alu.is_predicated && alu.predicate_condition != self.state.predicate {
            return;
        }
        let vop = alu.vector_opcode;
        let vector_needed = alu.vector_result.used_result_components() != 0 || ucode::vector_changes_state(vop);
        let vo: [[f32; 4]; 3] = std::array::from_fn(|i| {
            if vector_needed && (i as u32) < alu.vector_operand_count {
                self.operand_vec4(&alu.vector_operands[i])
            } else {
                [0.0; 4]
            }
        });
        let (sa, sb) = match alu.scalar_operand_count {
            1 => {
                let op = &alu.scalar_operands[0];
                (
                    self.operand_component(op, 0),
                    if op.component_count == 2 { self.operand_component(op, 1) } else { 0.0 },
                )
            }
            2 => {
                (self.operand_component(&alu.scalar_operands[0], 0), self.operand_component(&alu.scalar_operands[1], 0))
            }
            _ => (0.0, 0.0),
        };

        let mut vector_result = None;
        if vector_needed {
            let (a, b, c) = (vo[0], vo[1], vo[2]);
            let cmp = vector_comparison(vop);
            let per = |f: &dyn Fn(usize) -> f32| -> [f32; 4] { std::array::from_fn(f) };
            let r = match vop {
                v::ADD => per(&|i| a[i] + b[i]),
                v::MUL => per(&|i| mul(a[i], b[i])),
                v::MAX => per(&|i| max(a[i], b[i])),
                v::MIN => per(&|i| min(a[i], b[i])),
                v::SEQ | v::SGT | v::SGE | v::SNE => per(&|i| compare(cmp, a[i], b[i]) as u32 as f32),
                v::FRC => per(&|i| a[i] - a[i].floor()),
                v::TRUNC => per(&|i| a[i].trunc()),
                v::FLOOR => per(&|i| a[i].floor()),
                v::MAD => per(&|i| mul(a[i], b[i]) + c[i]),
                v::CNDEQ | v::CNDGE | v::CNDGT => per(&|i| if compare(cmp, a[i], 0.0) { b[i] } else { c[i] }),
                v::DP4 => [0.0 + mul(a[0], b[0]) + mul(a[1], b[1]) + mul(a[2], b[2]) + mul(a[3], b[3]); 4],
                v::DP3 => [0.0 + mul(a[0], b[0]) + mul(a[1], b[1]) + mul(a[2], b[2]); 4],
                v::DP2ADD => [0.0 + mul(a[0], b[0]) + mul(a[1], b[1]) + c[0]; 4],
                v::CUBE => cube(a),
                v::MAX4 => [max4(a); 4],
                v::SETP_EQ_PUSH | v::SETP_NE_PUSH | v::SETP_GT_PUSH | v::SETP_GE_PUSH => {
                    self.state.predicate = a[3] == 0.0 && compare(cmp, b[3], 0.0);
                    [if a[0] == 0.0 && compare(cmp, b[0], 0.0) { 0.0 } else { a[0] + 1.0 }; 4]
                }
                v::KILL_EQ | v::KILL_GT | v::KILL_GE | v::KILL_NE => {
                    let kill = (0..4).any(|i| compare(cmp, a[i], b[i]));
                    if kill && self.program.stage == ShaderStage::Pixel {
                        self.exports.killed = true;
                    }
                    [kill as u32 as f32; 4]
                }
                v::DST => [1.0, mul(a[1], b[1]), a[2], b[3]],
                v::MAXA => {
                    self.state.address_register = (a[3] + 0.5).floor().clamp(-256.0, 255.0) as i32;
                    per(&|i| max(a[i], b[i]))
                }
                _ => [0.0; 4],
            };
            vector_result = Some(r);
        }

        let ps = self.state.previous_scalar;
        let sop = alu.scalar_opcode;
        let mut write_ps = true;
        let sr = match sop {
            s::ADDS | s::ADDSC0 | s::ADDSC1 => sa + sb,
            s::ADDS_PREV => sa + ps,
            s::MULS | s::MULSC0 | s::MULSC1 => mul(sa, sb),
            s::MULS_PREV => mul(sa, ps),
            s::MULS_PREV2 => {
                if ps == -FLT_MAX || !ps.is_finite() || !sb.is_finite() || sb <= 0.0 {
                    -FLT_MAX
                } else {
                    mul(sa, ps)
                }
            }
            s::MAXS => max(sa, sb),
            s::MINS => min(sa, sb),
            s::SEQS => (sa == 0.0) as u32 as f32,
            s::SGTS => (sa > 0.0) as u32 as f32,
            s::SGES => (sa >= 0.0) as u32 as f32,
            s::SNES => (sa != 0.0) as u32 as f32,
            s::FRCS => sa - sa.floor(),
            s::TRUNCS => sa.trunc(),
            s::FLOORS => sa.floor(),
            s::EXP => reduce_precision(sa.exp2()),
            s::LOGC => inf_to_max(reduce_precision(sa.log2())),
            s::LOG => reduce_precision(sa.log2()),
            s::RCPC => inf_to_max(reduce_precision(1.0 / sa)),
            s::RCPF => inf_to_zero(reduce_precision(1.0 / sa)),
            s::RCP => reduce_precision(1.0 / sa),
            s::RSQC => inf_to_max(reduce_precision(1.0 / sa.sqrt())),
            s::RSQF => inf_to_zero(reduce_precision(1.0 / sa.sqrt())),
            s::RSQ => reduce_precision(1.0 / sa.sqrt()),
            s::MAXAS | s::MAXASF => {
                let rounded = if sop == s::MAXAS { sa + 0.5 } else { sa };
                self.state.address_register = rounded.floor().clamp(0.0, 255.0) as i32;
                max(sa, sb)
            }
            s::SUBS | s::SUBSC0 | s::SUBSC1 => sa - sb,
            s::SUBS_PREV => sa - ps,
            s::SETP_EQ | s::SETP_NE | s::SETP_GT | s::SETP_GE => {
                let p = match sop {
                    s::SETP_EQ => sa == 0.0,
                    s::SETP_NE => sa != 0.0,
                    s::SETP_GT => sa > 0.0,
                    _ => sa >= 0.0,
                };
                self.state.predicate = p;
                if p { 0.0 } else { 1.0 }
            }
            s::SETP_INV => {
                self.state.predicate = sa == 1.0;
                if self.state.predicate {
                    0.0
                } else if sa == 0.0 {
                    1.0
                } else {
                    sa
                }
            }
            s::SETP_POP => {
                let counter = sa - 1.0;
                self.state.predicate = counter <= 0.0;
                if self.state.predicate { 0.0 } else { counter }
            }
            s::SETP_CLR => {
                self.state.predicate = false;
                FLT_MAX
            }
            s::SETP_RSTR => {
                self.state.predicate = sa == 0.0;
                if self.state.predicate { 0.0 } else { sa }
            }
            s::KILLS_EQ | s::KILLS_GT | s::KILLS_GE | s::KILLS_NE | s::KILLS_ONE => {
                let kill = match sop {
                    s::KILLS_EQ => sa == 0.0,
                    s::KILLS_GT => sa > 0.0,
                    s::KILLS_GE => sa >= 0.0,
                    s::KILLS_NE => sa != 0.0,
                    _ => sa == 1.0,
                };
                if kill && self.program.stage == ShaderStage::Pixel {
                    self.exports.killed = true;
                }
                kill as u32 as f32
            }
            s::SQRT => reduce_precision(sa.sqrt()),
            s::SIN => sa.sin(),
            s::COS => sa.cos(),
            _ => {
                write_ps = false;
                ps
            }
        };
        if write_ps {
            self.state.previous_scalar = sr;
        }
        self.store(&alu.vector_result, vector_result);
        self.store(&alu.scalar_result, Some([sr; 4]));
    }

    fn vfetch_word(&self, address: i32, end: i32, endian: u32) -> u32 {
        if (address as u32) >= (end as u32) {
            return 0;
        }
        swap32(self.env.shared_memory.word(address as u32), endian)
    }

    pub fn vertex_fetch(&mut self, vf: &ParsedVertexFetch) {
        if vf.is_predicated && vf.predicate_condition != self.state.predicate {
            return;
        }
        let a = vf.attributes;
        let fc = |i: u32| self.env.fetch_constants.get(i as usize).copied().unwrap_or(0);
        let (word_0, word_1) = (fc(vf.fetch_constant * 2), fc(vf.fetch_constant * 2 + 1));
        if !vf.is_mini_fetch {
            self.state.vfetch_address = ((word_0 >> 2) & 0x07FF_FFFF) as i32;
            self.state.vfetch_end = self.state.vfetch_address + ((word_1 >> 2) & 0xFF_FFFF) as i32;
            if a.stride != 0 {
                let mut index = self.operand_component(&vf.index_operand, 0);
                if a.is_index_rounded {
                    index += 0.5;
                }
                self.state.vfetch_address += (index.floor() as i32).wrapping_mul(a.stride as i32);
            }
        }
        let used = vf.result.used_result_components();
        let format_components = vfmt::component_count(a.data_format);
        let used_format = used & ((1u32 << format_components) - 1);
        let needed = vfmt::needed_words(a.data_format, used);
        if needed == 0 || used_format == 0 {
            self.store(&vf.result, Some([0.0; 4]));
            return;
        }
        let endian = word_1 & 3;
        let words: [u32; 4] = std::array::from_fn(|w| {
            if needed & (1 << w) != 0 {
                self.vfetch_word(self.state.vfetch_address + a.offset + w as i32, self.state.vfetch_end, endian)
            } else {
                0
            }
        });
        let packed: Option<&[(usize, u32, u32)]> = match a.data_format {
            vfmt::K_8_8_8_8 => Some(&[(0, 0, 8), (0, 8, 8), (0, 16, 8), (0, 24, 8)]),
            vfmt::K_2_10_10_10 => Some(&[(0, 0, 10), (0, 10, 10), (0, 20, 10), (0, 30, 2)]),
            vfmt::K_10_11_11 => Some(&[(0, 0, 11), (0, 11, 11), (0, 22, 10)]),
            vfmt::K_11_11_10 => Some(&[(0, 0, 10), (0, 10, 11), (0, 21, 11)]),
            vfmt::K_16_16 => Some(&[(0, 0, 16), (0, 16, 16)]),
            vfmt::K_16_16_16_16 => Some(&[(0, 0, 16), (0, 16, 16), (1, 0, 16), (1, 16, 16)]),
            _ => None,
        };
        let mut result = [0.0f32; 4];
        for (i, r) in result.iter_mut().enumerate() {
            if used_format & (1 << i) == 0 {
                continue;
            }
            *r = if let Some(packed) = packed {
                let (word, offset, width) = packed[i];
                let raw = if a.is_signed {
                    (((words[word] << (32 - offset - width)) as i32) >> (32 - width)) as f32
                } else {
                    ((words[word] >> offset) & ((1u64 << width) - 1) as u32) as f32
                };
                if a.is_integer {
                    raw
                } else if a.is_signed {
                    let half = ((1u64 << (width - 1)) - 1) as f32;
                    if a.signed_rf_mode_no_zero { (raw + 0.5) / (half + 0.5) } else { (raw / half).max(-1.0) }
                } else {
                    raw / ((1u64 << width) - 1) as f32
                }
            } else {
                match a.data_format {
                    vfmt::K_16_16_FLOAT | vfmt::K_16_16_16_16_FLOAT => {
                        let w = words[i / 2];
                        half_to_f32(if i & 1 == 0 { w & 0xFFFF } else { w >> 16 })
                    }
                    vfmt::K_32 | vfmt::K_32_32 | vfmt::K_32_32_32_32 => {
                        let raw = if a.is_signed { words[i] as i32 as f32 } else { words[i] as f32 };
                        if a.is_integer {
                            raw
                        } else if a.is_signed {
                            if a.signed_rf_mode_no_zero { (raw + 0.5) / 2147483647.5 } else { raw / 2147483647.0 }
                        } else {
                            raw / 4294967295.0
                        }
                    }
                    _ => f32::from_bits(words[i]),
                }
            };
        }
        if a.exp_adjust != 0 {
            let f = 2f32.powi(a.exp_adjust);
            result = result.map(|x| x * f);
        }
        self.store(&vf.result, Some(result));
    }

    pub fn texture_fetch(&mut self, tf: &ParsedTextureFetch) {
        if tf.is_predicated && tf.predicate_condition != self.state.predicate {
            return;
        }
        let c = |i: u32| self.operand_component(&tf.src, i);
        match tf.opcode {
            FetchOpcode::SetTextureLod => self.state.tfetch_lod = c(0),
            FetchOpcode::SetTextureGradientsHorz => self.state.tfetch_gradients[0] = [c(0), c(1), c(2)],
            FetchOpcode::SetTextureGradientsVert => self.state.tfetch_gradients[1] = [c(0), c(1), c(2)],
            _ => self.store(&tf.result, Some([0.0; 4])),
        }
    }
}
