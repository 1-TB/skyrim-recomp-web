//! Raw Xenos microcode encodings: control flow (two 48-bit instructions per three dwords), ALU and
//! fetch instructions (three dwords each). Bit layouts follow Xenia's `ucode.h`; dwords are the
//! host-order values of the big-endian words uploaded by `IM_LOAD`.

#[inline]
fn bits(value: u32, shift: u32, count: u32) -> u32 {
    (value >> shift) & ((1u32 << count) - 1)
}

#[inline]
fn sbits(value: u32, shift: u32, count: u32) -> i32 {
    ((value << (32 - shift - count)) as i32) >> (32 - count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CfOpcode {
    Nop,
    Exec,
    ExecEnd,
    CondExec,
    CondExecEnd,
    CondExecPred,
    CondExecPredEnd,
    LoopStart,
    LoopEnd,
    CondCall,
    Return,
    CondJmp,
    Alloc,
    CondExecPredClean,
    CondExecPredCleanEnd,
    MarkVsFetchDone,
}

impl CfOpcode {
    pub fn from_bits(value: u32) -> CfOpcode {
        use CfOpcode::*;
        [
            Nop,
            Exec,
            ExecEnd,
            CondExec,
            CondExecEnd,
            CondExecPred,
            CondExecPredEnd,
            LoopStart,
            LoopEnd,
            CondCall,
            Return,
            CondJmp,
            Alloc,
            CondExecPredClean,
            CondExecPredCleanEnd,
            MarkVsFetchDone,
        ][(value & 15) as usize]
    }

    pub fn is_exec(self) -> bool {
        matches!(
            self,
            CfOpcode::Exec
                | CfOpcode::ExecEnd
                | CfOpcode::CondExec
                | CfOpcode::CondExecEnd
                | CfOpcode::CondExecPred
                | CfOpcode::CondExecPredEnd
                | CfOpcode::CondExecPredClean
                | CfOpcode::CondExecPredCleanEnd
        )
    }

    /// Whether the exec terminates the shader (only if its condition passes, for conditional ones).
    pub fn ends_shader(self) -> bool {
        matches!(
            self,
            CfOpcode::ExecEnd | CfOpcode::CondExecEnd | CfOpcode::CondExecPredEnd | CfOpcode::CondExecPredCleanEnd
        )
    }
}

/// One 48-bit control flow instruction: `dword0` holds bits 0:31, `dword1` bits 32:47.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawCf {
    pub dword0: u32,
    pub dword1: u32,
}

impl RawCf {
    pub fn opcode(&self) -> CfOpcode {
        CfOpcode::from_bits(self.dword1 >> 12)
    }
    pub fn addressing_absolute(&self) -> bool {
        bits(self.dword1, 11, 1) != 0
    }
    // Exec family.
    pub fn exec_address(&self) -> u32 {
        bits(self.dword0, 0, 12)
    }
    pub fn exec_count(&self) -> u32 {
        bits(self.dword0, 12, 3)
    }
    pub fn exec_is_yield(&self) -> bool {
        bits(self.dword0, 15, 1) != 0
    }
    pub fn exec_sequence(&self) -> u32 {
        bits(self.dword0, 16, 12)
    }
    pub fn exec_vertex_cache(&self) -> u32 {
        bits(self.dword0, 28, 4) | (bits(self.dword1, 0, 2) << 4)
    }
    pub fn exec_is_predicate_clean(&self) -> bool {
        bits(self.dword1, 9, 1) != 0
    }
    pub fn cond_bool_address(&self) -> u32 {
        bits(self.dword1, 2, 8)
    }
    pub fn condition(&self) -> bool {
        bits(self.dword1, 10, 1) != 0
    }
    // Loops.
    pub fn loop_address(&self) -> u32 {
        bits(self.dword0, 0, 13)
    }
    pub fn loop_is_repeat(&self) -> bool {
        bits(self.dword0, 13, 1) != 0
    }
    pub fn loop_id(&self) -> u32 {
        bits(self.dword0, 16, 5)
    }
    pub fn loop_is_predicated_break(&self) -> bool {
        bits(self.dword0, 21, 1) != 0
    }
    // Calls and jumps.
    pub fn target_address(&self) -> u32 {
        bits(self.dword0, 0, 13)
    }
    pub fn is_unconditional(&self) -> bool {
        bits(self.dword0, 13, 1) != 0
    }
    pub fn is_predicated(&self) -> bool {
        bits(self.dword0, 14, 1) != 0
    }
    pub fn jump_direction(&self) -> bool {
        bits(self.dword1, 1, 1) != 0
    }
    // Alloc.
    pub fn alloc_size(&self) -> u32 {
        bits(self.dword0, 0, 3)
    }
    pub fn alloc_type(&self) -> u32 {
        bits(self.dword1, 9, 2)
    }
    pub fn alloc_is_unserialized(&self) -> bool {
        bits(self.dword1, 8, 1) != 0
    }
}

/// Splits three dwords into the two control flow instructions they contain.
pub fn unpack_cf_pair(dwords: [u32; 3]) -> [RawCf; 2] {
    [
        RawCf { dword0: dwords[0], dword1: dwords[1] & 0xFFFF },
        RawCf { dword0: (dwords[1] >> 16) | (dwords[2] << 16), dword1: dwords[2] >> 16 },
    ]
}

/// Inverse of [`unpack_cf_pair`].
pub fn pack_cf_pair(a: RawCf, b: RawCf) -> [u32; 3] {
    [a.dword0, (a.dword1 & 0xFFFF) | (b.dword0 << 16), (b.dword0 >> 16) | (b.dword1 << 16)]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FetchOpcode {
    VertexFetch,
    TextureFetch,
    GetTextureBorderColorFrac,
    GetTextureComputedLod,
    GetTextureGradients,
    GetTextureWeights,
    SetTextureLod,
    SetTextureGradientsHorz,
    SetTextureGradientsVert,
    /// Unassigned opcode values.
    Unknown(u32),
}

impl FetchOpcode {
    pub fn from_bits(value: u32) -> FetchOpcode {
        match value & 31 {
            0 => FetchOpcode::VertexFetch,
            1 => FetchOpcode::TextureFetch,
            16 => FetchOpcode::GetTextureBorderColorFrac,
            17 => FetchOpcode::GetTextureComputedLod,
            18 => FetchOpcode::GetTextureGradients,
            19 => FetchOpcode::GetTextureWeights,
            24 => FetchOpcode::SetTextureLod,
            25 => FetchOpcode::SetTextureGradientsHorz,
            26 => FetchOpcode::SetTextureGradientsVert,
            other => FetchOpcode::Unknown(other),
        }
    }
}

/// Fetch instruction (vertex or texture), three dwords.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawFetch(pub [u32; 3]);

impl RawFetch {
    pub fn opcode(&self) -> FetchOpcode {
        FetchOpcode::from_bits(self.0[0])
    }
    pub fn src(&self) -> u32 {
        bits(self.0[0], 5, 6)
    }
    pub fn is_src_relative(&self) -> bool {
        bits(self.0[0], 11, 1) != 0
    }
    pub fn dest(&self) -> u32 {
        bits(self.0[0], 12, 6)
    }
    pub fn is_dest_relative(&self) -> bool {
        bits(self.0[0], 18, 1) != 0
    }
    pub fn dest_swizzle(&self) -> u32 {
        bits(self.0[1], 0, 12)
    }
    pub fn is_predicated(&self) -> bool {
        bits(self.0[1], 31, 1) != 0
    }
    pub fn predicate_condition(&self) -> bool {
        bits(self.0[2], 31, 1) != 0
    }

    // Vertex fetch fields.
    pub fn vf_fetch_constant_index(&self) -> u32 {
        bits(self.0[0], 20, 5) * 3 + bits(self.0[0], 25, 2)
    }
    pub fn vf_prefetch_count(&self) -> u32 {
        bits(self.0[0], 27, 3)
    }
    pub fn vf_src_swizzle(&self) -> u32 {
        bits(self.0[0], 30, 2)
    }
    pub fn vf_is_signed(&self) -> bool {
        bits(self.0[1], 12, 1) != 0
    }
    pub fn vf_is_normalized(&self) -> bool {
        bits(self.0[1], 13, 1) == 0
    }
    pub fn vf_signed_rf_mode_no_zero(&self) -> bool {
        bits(self.0[1], 14, 1) != 0
    }
    pub fn vf_is_index_rounded(&self) -> bool {
        bits(self.0[1], 15, 1) != 0
    }
    pub fn vf_format(&self) -> u32 {
        bits(self.0[1], 16, 6)
    }
    pub fn vf_exp_adjust(&self) -> i32 {
        sbits(self.0[1], 24, 6)
    }
    pub fn vf_is_mini_fetch(&self) -> bool {
        bits(self.0[1], 30, 1) != 0
    }
    pub fn vf_stride(&self) -> u32 {
        bits(self.0[2], 0, 8)
    }
    pub fn vf_offset(&self) -> i32 {
        sbits(self.0[2], 8, 23)
    }

    // Texture fetch fields.
    pub fn tf_fetch_valid_only(&self) -> bool {
        bits(self.0[0], 19, 1) != 0
    }
    pub fn tf_fetch_constant_index(&self) -> u32 {
        bits(self.0[0], 20, 5)
    }
    pub fn tf_unnormalized_coordinates(&self) -> bool {
        bits(self.0[0], 25, 1) != 0
    }
    pub fn tf_src_swizzle(&self) -> u32 {
        bits(self.0[0], 26, 6)
    }
    pub fn tf_mag_filter(&self) -> u32 {
        bits(self.0[1], 12, 2)
    }
    pub fn tf_min_filter(&self) -> u32 {
        bits(self.0[1], 14, 2)
    }
    pub fn tf_mip_filter(&self) -> u32 {
        bits(self.0[1], 16, 2)
    }
    pub fn tf_aniso_filter(&self) -> u32 {
        bits(self.0[1], 18, 3)
    }
    pub fn tf_arbitrary_filter(&self) -> u32 {
        bits(self.0[1], 21, 3)
    }
    pub fn tf_vol_mag_filter(&self) -> u32 {
        bits(self.0[1], 24, 2)
    }
    pub fn tf_vol_min_filter(&self) -> u32 {
        bits(self.0[1], 26, 2)
    }
    pub fn tf_use_computed_lod(&self) -> bool {
        bits(self.0[1], 28, 1) != 0
    }
    pub fn tf_use_register_lod(&self) -> bool {
        bits(self.0[1], 29, 1) != 0
    }
    pub fn tf_use_register_gradients(&self) -> bool {
        bits(self.0[2], 0, 1) != 0
    }
    pub fn tf_sample_location_center(&self) -> bool {
        bits(self.0[2], 1, 1) != 0
    }
    /// LOD bias in 1/16 units.
    pub fn tf_lod_bias_raw(&self) -> i32 {
        sbits(self.0[2], 2, 7)
    }
    pub fn tf_dimension(&self) -> u32 {
        bits(self.0[2], 14, 2)
    }
    /// Offsets in 1/2 texel units.
    pub fn tf_offset_raw(&self, axis: u32) -> i32 {
        sbits(self.0[2], 16 + 5 * axis, 5)
    }
}

/// ALU instruction (paired vector + scalar operation), three dwords.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawAlu(pub [u32; 3]);

impl RawAlu {
    pub fn vector_dest(&self) -> u32 {
        bits(self.0[0], 0, 6)
    }
    pub fn is_vector_dest_relative(&self) -> bool {
        bits(self.0[0], 6, 1) != 0
    }
    pub fn abs_constants(&self) -> bool {
        bits(self.0[0], 7, 1) != 0
    }
    pub fn scalar_dest(&self) -> u32 {
        bits(self.0[0], 8, 6)
    }
    pub fn is_scalar_dest_relative(&self) -> bool {
        bits(self.0[0], 14, 1) != 0
    }
    pub fn is_export(&self) -> bool {
        bits(self.0[0], 15, 1) != 0
    }
    pub fn vector_write_mask(&self) -> u32 {
        bits(self.0[0], 16, 4)
    }
    pub fn scalar_write_mask(&self) -> u32 {
        bits(self.0[0], 20, 4)
    }
    pub fn vector_clamp(&self) -> bool {
        bits(self.0[0], 24, 1) != 0
    }
    pub fn scalar_clamp(&self) -> bool {
        bits(self.0[0], 25, 1) != 0
    }
    pub fn scalar_opcode_bits(&self) -> u32 {
        bits(self.0[0], 26, 6)
    }
    pub fn src_swizzle(&self, i: u32) -> u32 {
        match i {
            1 => bits(self.0[1], 16, 8),
            2 => bits(self.0[1], 8, 8),
            _ => bits(self.0[1], 0, 8),
        }
    }
    pub fn src_negate(&self, i: u32) -> bool {
        match i {
            1 => bits(self.0[1], 26, 1) != 0,
            2 => bits(self.0[1], 25, 1) != 0,
            _ => bits(self.0[1], 24, 1) != 0,
        }
    }
    pub fn predicate_condition(&self) -> bool {
        bits(self.0[1], 27, 1) != 0
    }
    pub fn is_predicated(&self) -> bool {
        bits(self.0[1], 28, 1) != 0
    }
    pub fn is_const_address_register_relative(&self) -> bool {
        bits(self.0[1], 29, 1) != 0
    }
    pub fn const_1_rel_abs(&self) -> bool {
        bits(self.0[1], 30, 1) != 0
    }
    pub fn const_0_rel_abs(&self) -> bool {
        bits(self.0[1], 31, 1) != 0
    }
    pub fn src_reg(&self, i: u32) -> u32 {
        match i {
            1 => bits(self.0[2], 16, 8),
            2 => bits(self.0[2], 8, 8),
            _ => bits(self.0[2], 0, 8),
        }
    }
    pub fn vector_opcode_bits(&self) -> u32 {
        bits(self.0[2], 24, 5)
    }
    pub fn src_is_temp(&self, i: u32) -> bool {
        match i {
            1 => bits(self.0[2], 31, 1) != 0,
            2 => bits(self.0[2], 30, 1) != 0,
            _ => bits(self.0[2], 29, 1) != 0,
        }
    }
    /// Whether the constant in operand `i` uses relative addressing (see Xenia's
    /// `src_const_is_addressed` for why the third operand depends on the first two).
    pub fn src_const_is_addressed(&self, i: u32) -> bool {
        match i {
            1 => self.const_0_rel_abs(),
            2 => {
                if self.src_is_temp(1) {
                    self.const_0_rel_abs()
                } else {
                    self.const_1_rel_abs()
                }
            }
            _ => {
                if self.src_is_temp(1) && self.src_is_temp(2) {
                    self.const_0_rel_abs()
                } else {
                    self.const_1_rel_abs()
                }
            }
        }
    }
    /// Temporary register index of the second operand of `mulsc`/`addsc`/`subsc`.
    pub fn scalar_const_reg_op_src_temp_reg(&self) -> u32 {
        (self.scalar_opcode_bits() & 1) | ((self.src_is_temp(3) as u32) << 1) | (self.src_swizzle(3) & 0x3C)
    }

    pub fn vector_op_result_write_mask(&self) -> u32 {
        let mut mask = self.vector_write_mask();
        if self.is_export() {
            mask &= !self.scalar_write_mask();
        }
        mask
    }
    pub fn scalar_op_result_write_mask(&self) -> u32 {
        let mut mask = self.scalar_write_mask();
        if self.is_export() {
            mask &= !self.vector_write_mask();
        }
        mask
    }
    pub fn constant_0_write_mask(&self) -> u32 {
        if !self.is_export() || !self.is_scalar_dest_relative() {
            return 0;
        }
        0b1111 & !(self.vector_write_mask() | self.scalar_write_mask())
    }
    pub fn constant_1_write_mask(&self) -> u32 {
        if !self.is_export() {
            return 0;
        }
        self.vector_write_mask() & self.scalar_write_mask()
    }
}

/// Absolute component index selected by a relative ALU source swizzle.
#[inline]
pub fn swizzled_component_index(swizzle: u32, component: u32) -> u32 {
    ((swizzle >> (2 * component)) + component) & 3
}

pub const SCALAR_NAMES: [&str; 64] = [
    "adds",
    "adds_prev",
    "muls",
    "muls_prev",
    "muls_prev2",
    "maxs",
    "mins",
    "seqs",
    "sgts",
    "sges",
    "snes",
    "frcs",
    "truncs",
    "floors",
    "exp",
    "logc",
    "log",
    "rcpc",
    "rcpf",
    "rcp",
    "rsqc",
    "rsqf",
    "rsq",
    "maxas",
    "maxasf",
    "subs",
    "subs_prev",
    "setp_eq",
    "setp_ne",
    "setp_gt",
    "setp_ge",
    "setp_inv",
    "setp_pop",
    "setp_clr",
    "setp_rstr",
    "kills_eq",
    "kills_gt",
    "kills_ge",
    "kills_ne",
    "kills_one",
    "sqrt",
    "opcode_41",
    "mulsc",
    "mulsc",
    "addsc",
    "addsc",
    "subsc",
    "subsc",
    "sin",
    "cos",
    "retain_prev",
    "opcode_51",
    "opcode_52",
    "opcode_53",
    "opcode_54",
    "opcode_55",
    "opcode_56",
    "opcode_57",
    "opcode_58",
    "opcode_59",
    "opcode_60",
    "opcode_61",
    "opcode_62",
    "opcode_63",
];

pub const VECTOR_NAMES: [&str; 32] = [
    "add",
    "mul",
    "max",
    "min",
    "seq",
    "sgt",
    "sge",
    "sne",
    "frc",
    "trunc",
    "floor",
    "mad",
    "cndeq",
    "cndge",
    "cndgt",
    "dp4",
    "dp3",
    "dp2add",
    "cube",
    "max4",
    "setp_eq_push",
    "setp_ne_push",
    "setp_gt_push",
    "setp_ge_push",
    "kill_eq",
    "kill_gt",
    "kill_ge",
    "kill_ne",
    "dst",
    "maxa",
    "opcode_30",
    "opcode_31",
];

pub mod scalar_op {
    pub const ADDS: u32 = 0;
    pub const ADDS_PREV: u32 = 1;
    pub const MULS: u32 = 2;
    pub const MULS_PREV: u32 = 3;
    pub const MULS_PREV2: u32 = 4;
    pub const MAXS: u32 = 5;
    pub const MINS: u32 = 6;
    pub const SEQS: u32 = 7;
    pub const SGTS: u32 = 8;
    pub const SGES: u32 = 9;
    pub const SNES: u32 = 10;
    pub const FRCS: u32 = 11;
    pub const TRUNCS: u32 = 12;
    pub const FLOORS: u32 = 13;
    pub const EXP: u32 = 14;
    pub const LOGC: u32 = 15;
    pub const LOG: u32 = 16;
    pub const RCPC: u32 = 17;
    pub const RCPF: u32 = 18;
    pub const RCP: u32 = 19;
    pub const RSQC: u32 = 20;
    pub const RSQF: u32 = 21;
    pub const RSQ: u32 = 22;
    pub const MAXAS: u32 = 23;
    pub const MAXASF: u32 = 24;
    pub const SUBS: u32 = 25;
    pub const SUBS_PREV: u32 = 26;
    pub const SETP_EQ: u32 = 27;
    pub const SETP_NE: u32 = 28;
    pub const SETP_GT: u32 = 29;
    pub const SETP_GE: u32 = 30;
    pub const SETP_INV: u32 = 31;
    pub const SETP_POP: u32 = 32;
    pub const SETP_CLR: u32 = 33;
    pub const SETP_RSTR: u32 = 34;
    pub const KILLS_EQ: u32 = 35;
    pub const KILLS_GT: u32 = 36;
    pub const KILLS_GE: u32 = 37;
    pub const KILLS_NE: u32 = 38;
    pub const KILLS_ONE: u32 = 39;
    pub const SQRT: u32 = 40;
    pub const MULSC0: u32 = 42;
    pub const MULSC1: u32 = 43;
    pub const ADDSC0: u32 = 44;
    pub const ADDSC1: u32 = 45;
    pub const SUBSC0: u32 = 46;
    pub const SUBSC1: u32 = 47;
    pub const SIN: u32 = 48;
    pub const COS: u32 = 49;
    pub const RETAIN_PREV: u32 = 50;
}

pub mod vector_op {
    pub const ADD: u32 = 0;
    pub const MUL: u32 = 1;
    pub const MAX: u32 = 2;
    pub const MIN: u32 = 3;
    pub const SEQ: u32 = 4;
    pub const SGT: u32 = 5;
    pub const SGE: u32 = 6;
    pub const SNE: u32 = 7;
    pub const FRC: u32 = 8;
    pub const TRUNC: u32 = 9;
    pub const FLOOR: u32 = 10;
    pub const MAD: u32 = 11;
    pub const CNDEQ: u32 = 12;
    pub const CNDGE: u32 = 13;
    pub const CNDGT: u32 = 14;
    pub const DP4: u32 = 15;
    pub const DP3: u32 = 16;
    pub const DP2ADD: u32 = 17;
    pub const CUBE: u32 = 18;
    pub const MAX4: u32 = 19;
    pub const SETP_EQ_PUSH: u32 = 20;
    pub const SETP_NE_PUSH: u32 = 21;
    pub const SETP_GT_PUSH: u32 = 22;
    pub const SETP_GE_PUSH: u32 = 23;
    pub const KILL_EQ: u32 = 24;
    pub const KILL_GT: u32 = 25;
    pub const KILL_GE: u32 = 26;
    pub const KILL_NE: u32 = 27;
    pub const DST: u32 = 28;
    pub const MAXA: u32 = 29;
}

/// Number of operands (0 = none, 1 = one register with 1 or 2 components, 2 = constant and temp).
pub fn scalar_operand_count(op: u32) -> u32 {
    match op {
        scalar_op::SETP_CLR | scalar_op::RETAIN_PREV | 41 | 51..=63 => 0,
        scalar_op::MULSC0..=scalar_op::SUBSC1 => 2,
        _ => 1,
    }
}

pub fn scalar_operand_is_two_component(op: u32) -> bool {
    matches!(
        op,
        scalar_op::ADDS
            | scalar_op::MULS
            | scalar_op::MULS_PREV2
            | scalar_op::MAXS
            | scalar_op::MINS
            | scalar_op::MAXAS
            | scalar_op::MAXASF
            | scalar_op::SUBS
    )
}

pub fn scalar_changes_predicate(op: u32) -> bool {
    (scalar_op::SETP_EQ..=scalar_op::SETP_RSTR).contains(&op)
}

pub fn scalar_kills(op: u32) -> bool {
    (scalar_op::KILLS_EQ..=scalar_op::KILLS_ONE).contains(&op)
}

pub fn scalar_changes_address(op: u32) -> bool {
    op == scalar_op::MAXAS || op == scalar_op::MAXASF
}

/// Components of each vector operand read by the opcode (from Xenia's `kAluVectorOpcodeInfos`).
pub fn vector_operand_components(op: u32) -> [u32; 3] {
    use vector_op::*;
    match op {
        ADD | MUL | MAX | MIN | SEQ | SGT | SGE | SNE | DP4 | CUBE | KILL_EQ | KILL_GT | KILL_GE | KILL_NE | MAXA => {
            [0b1111, 0b1111, 0]
        }
        FRC | TRUNC | FLOOR | MAX4 => [0b1111, 0, 0],
        MAD | CNDEQ | CNDGE | CNDGT => [0b1111, 0b1111, 0b1111],
        DP3 => [0b0111, 0b0111, 0],
        DP2ADD => [0b0011, 0b0011, 0b0001],
        SETP_EQ_PUSH | SETP_NE_PUSH | SETP_GT_PUSH | SETP_GE_PUSH => [0b1001, 0b1001, 0],
        DST => [0b0110, 0b1010, 0],
        _ => [0, 0, 0],
    }
}

pub fn vector_operand_count(op: u32) -> u32 {
    let c = vector_operand_components(op);
    if c[2] != 0 {
        3
    } else if c[1] != 0 {
        2
    } else if c[0] != 0 {
        1
    } else {
        0
    }
}

pub fn vector_changes_predicate(op: u32) -> bool {
    (vector_op::SETP_EQ_PUSH..=vector_op::SETP_GE_PUSH).contains(&op)
}

pub fn vector_kills(op: u32) -> bool {
    (vector_op::KILL_EQ..=vector_op::KILL_NE).contains(&op)
}

pub fn vector_changes_address(op: u32) -> bool {
    op == vector_op::MAXA
}

pub fn vector_changes_state(op: u32) -> bool {
    vector_changes_predicate(op) || vector_kills(op) || vector_changes_address(op)
}

/// Export register numbers (ALU `vector_dest` when exporting).
pub mod export_reg {
    pub const VS_POSITION: u32 = 62;
    pub const VS_POINT_SIZE_EDGE_FLAG_KILL_VERTEX: u32 = 63;
    pub const PS_DEPTH: u32 = 61;
    pub const EXPORT_ADDRESS: u32 = 32;
    pub const EXPORT_DATA_0: u32 = 33;
    pub const EXPORT_DATA_4: u32 = 37;
}

/// Vertex fetch data formats (`xenos::VertexFormat`).
pub mod vertex_format {
    pub const K_8_8_8_8: u32 = 6;
    pub const K_2_10_10_10: u32 = 7;
    pub const K_10_11_11: u32 = 16;
    pub const K_11_11_10: u32 = 17;
    pub const K_16_16: u32 = 25;
    pub const K_16_16_16_16: u32 = 26;
    pub const K_16_16_FLOAT: u32 = 31;
    pub const K_16_16_16_16_FLOAT: u32 = 32;
    pub const K_32: u32 = 33;
    pub const K_32_32: u32 = 34;
    pub const K_32_32_32_32: u32 = 35;
    pub const K_32_FLOAT: u32 = 36;
    pub const K_32_32_FLOAT: u32 = 37;
    pub const K_32_32_32_32_FLOAT: u32 = 38;
    pub const K_32_32_32_FLOAT: u32 = 57;

    pub fn name(format: u32) -> Option<&'static str> {
        Some(match format {
            K_8_8_8_8 => "FMT_8_8_8_8",
            K_2_10_10_10 => "FMT_2_10_10_10",
            K_10_11_11 => "FMT_10_11_11",
            K_11_11_10 => "FMT_11_11_10",
            K_16_16 => "FMT_16_16",
            K_16_16_16_16 => "FMT_16_16_16_16",
            K_16_16_FLOAT => "FMT_16_16_FLOAT",
            K_16_16_16_16_FLOAT => "FMT_16_16_16_16_FLOAT",
            K_32 => "FMT_32",
            K_32_32 => "FMT_32_32",
            K_32_32_32_32 => "FMT_32_32_32_32",
            K_32_FLOAT => "FMT_32_FLOAT",
            K_32_32_FLOAT => "FMT_32_32_FLOAT",
            K_32_32_32_32_FLOAT => "FMT_32_32_32_32_FLOAT",
            K_32_32_32_FLOAT => "FMT_32_32_32_FLOAT",
            _ => return None,
        })
    }

    pub fn component_count(format: u32) -> u32 {
        match format {
            K_32 | K_32_FLOAT => 1,
            K_16_16 | K_16_16_FLOAT | K_32_32 | K_32_32_FLOAT => 2,
            K_10_11_11 | K_11_11_10 | K_32_32_32_FLOAT => 3,
            K_8_8_8_8 | K_2_10_10_10 | K_16_16_16_16 | K_16_16_16_16_FLOAT | K_32_32_32_32 | K_32_32_32_32_FLOAT => 4,
            _ => 0,
        }
    }

    /// Dwords of the element needed to produce `used_components` (bit per dword).
    pub fn needed_words(format: u32, used_components: u32) -> u32 {
        if used_components == 0 {
            return 0;
        }
        match format {
            K_8_8_8_8 | K_2_10_10_10 => 0b0001,
            K_10_11_11 | K_11_11_10 => u32::from(used_components & 0b0111 != 0),
            K_16_16 | K_16_16_FLOAT => u32::from(used_components & 0b0011 != 0),
            K_16_16_16_16 | K_16_16_16_16_FLOAT => {
                u32::from(used_components & 0b0011 != 0) | (u32::from(used_components & 0b1100 != 0) << 1)
            }
            K_32 | K_32_FLOAT => used_components & 0b0001,
            K_32_32 | K_32_32_FLOAT => used_components & 0b0011,
            K_32_32_32_32 | K_32_32_32_32_FLOAT => used_components,
            K_32_32_32_FLOAT => used_components & 0b0111,
            _ => 0,
        }
    }

    /// Size of one element in dwords.
    pub fn size_dwords(format: u32) -> u32 {
        match format {
            K_8_8_8_8 | K_2_10_10_10 | K_10_11_11 | K_11_11_10 | K_16_16 | K_16_16_FLOAT | K_32 | K_32_FLOAT => 1,
            K_16_16_16_16 | K_16_16_16_16_FLOAT | K_32_32 | K_32_32_FLOAT => 2,
            K_32_32_32_FLOAT => 3,
            K_32_32_32_32 | K_32_32_32_32_FLOAT => 4,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cf_pair_round_trip() {
        let a = RawCf { dword0: 0x1234_5678, dword1: 0xABCD };
        let b = RawCf { dword0: 0x9876_5432, dword1: 0x1F2E };
        let packed = pack_cf_pair(a, b);
        assert_eq!(unpack_cf_pair(packed), [a, b]);
    }

    #[test]
    fn signed_fields() {
        let f = RawFetch([0, 0x3F << 24, 0xFF << 8]);
        assert_eq!(f.vf_exp_adjust(), -1);
        assert_eq!(f.vf_offset(), 255);
        let f = RawFetch([0, 0, 0x7FFFFF << 8]);
        assert_eq!(f.vf_offset(), -1);
    }
}
