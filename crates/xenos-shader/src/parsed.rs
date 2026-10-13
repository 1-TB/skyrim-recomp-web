//! Decoded instruction representations shared by the disassembler, analysis, the WGSL generator
//! and the reference interpreter (a port of Xenia's `Parsed*Instruction` model).

use crate::ShaderStage;
use crate::ucode::{self, CfOpcode, FetchOpcode, RawAlu, RawCf, RawFetch, export_reg};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageTarget {
    None,
    Register,
    Interpolator,
    Position,
    PointSizeEdgeFlagKillVertex,
    ExportAddress,
    ExportData,
    Color,
    Depth,
}

impl StorageTarget {
    pub fn used_component_count(self) -> u32 {
        match self {
            StorageTarget::None => 0,
            StorageTarget::PointSizeEdgeFlagKillVertex => 3,
            StorageTarget::Depth => 1,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressingMode {
    Absolute,
    /// `[n + a0]`.
    AddressRegisterRelative,
    /// `[n + aL]`.
    LoopRelative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SwizzleSource {
    X,
    Y,
    Z,
    W,
    Zero,
    One,
}

impl SwizzleSource {
    pub fn from_component(i: u32) -> SwizzleSource {
        [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W][(i & 3) as usize]
    }
    pub fn char(self) -> char {
        ['x', 'y', 'z', 'w', '0', '1'][self as usize]
    }
    pub fn component(self) -> Option<u32> {
        match self {
            SwizzleSource::X => Some(0),
            SwizzleSource::Y => Some(1),
            SwizzleSource::Z => Some(2),
            SwizzleSource::W => Some(3),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionResult {
    pub target: StorageTarget,
    pub index: u32,
    pub addressing: AddressingMode,
    pub is_clamped: bool,
    pub original_write_mask: u32,
    pub components: [SwizzleSource; 4],
}

impl Default for InstructionResult {
    fn default() -> Self {
        InstructionResult {
            target: StorageTarget::None,
            index: 0,
            addressing: AddressingMode::Absolute,
            is_clamped: false,
            original_write_mask: 0,
            components: [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W],
        }
    }
}

impl InstructionResult {
    /// Write mask limited to the components that exist in the target.
    pub fn used_write_mask(&self) -> u32 {
        self.original_write_mask & ((1u32 << self.target.used_component_count()) - 1)
    }
    /// Components of the computed value that are needed by the write.
    pub fn used_result_components(&self) -> u32 {
        let mask = self.used_write_mask();
        let mut used = 0;
        for (i, component) in self.components.iter().enumerate() {
            if mask & (1 << i) != 0
                && let Some(c) = component.component()
            {
                used |= 1 << c;
            }
        }
        used
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageSource {
    Register,
    ConstantFloat,
    VertexFetchConstant,
    TextureFetchConstant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionOperand {
    pub source: StorageSource,
    pub index: u32,
    pub addressing: AddressingMode,
    pub is_negated: bool,
    pub is_absolute_value: bool,
    pub component_count: u32,
    pub components: [SwizzleSource; 4],
}

impl Default for InstructionOperand {
    fn default() -> Self {
        InstructionOperand {
            source: StorageSource::Register,
            index: 0,
            addressing: AddressingMode::Absolute,
            is_negated: false,
            is_absolute_value: false,
            component_count: 4,
            components: [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W],
        }
    }
}

impl InstructionOperand {
    pub fn component(&self, index: u32) -> SwizzleSource {
        self.components[index.min(self.component_count.max(1) - 1) as usize]
    }
    pub fn is_standard_swizzle(&self) -> bool {
        self.component_count == 4
            && self.components == [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W]
    }
    /// Bit mask of components that are bitwise identical between two operands.
    pub fn identical_components(&self, other: &InstructionOperand) -> u32 {
        if self.source != other.source
            || self.index != other.index
            || self.addressing != other.addressing
            || self.is_negated != other.is_negated
            || self.is_absolute_value != other.is_absolute_value
        {
            return 0;
        }
        (0..4).filter(|&i| self.component(i) == other.component(i)).fold(0, |m, i| m | (1 << i))
    }
}

/// Condition of a control flow instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CfCondition {
    Always,
    /// Passes if bool constant `index` equals `value`.
    Bool {
        index: u32,
        value: bool,
    },
    /// Passes if the predicate register equals the value.
    Predicate(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParsedExec {
    pub cf_index: u32,
    pub opcode: CfOpcode,
    pub name: &'static str,
    pub address: u32,
    pub count: u32,
    pub condition: CfCondition,
    pub is_end: bool,
    pub is_predicate_clean: bool,
    pub is_yield: bool,
    /// Two bits per instruction: bit 0 = fetch (1) or ALU (0), bit 1 = serialize.
    pub sequence: u32,
}

impl ParsedExec {
    pub fn is_fetch(&self, i: u32) -> bool {
        (self.sequence >> (2 * i)) & 1 != 0
    }
    pub fn is_serialize(&self, i: u32) -> bool {
        (self.sequence >> (2 * i)) & 2 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocType {
    None,
    VsPosition,
    /// Interpolators in vertex shaders, colors in pixel shaders.
    InterpolatorsOrColors,
    Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParsedCf {
    Nop,
    Exec(ParsedExec),
    LoopStart { loop_constant: u32, is_repeat: bool, skip_address: u32 },
    LoopEnd { loop_constant: u32, body_address: u32, predicated_break: Option<bool> },
    Call { target: u32, condition: CfCondition },
    Return,
    Jump { target: u32, condition: CfCondition },
    Alloc { alloc_type: AllocType, size: u32 },
    MarkVsFetchDone,
}

fn call_jump_condition(cf: &RawCf) -> CfCondition {
    if cf.is_unconditional() {
        CfCondition::Always
    } else if cf.is_predicated() {
        CfCondition::Predicate(cf.condition())
    } else {
        CfCondition::Bool { index: cf.cond_bool_address(), value: cf.condition() }
    }
}

pub fn parse_cf(cf: &RawCf, cf_index: u32) -> ParsedCf {
    let opcode = cf.opcode();
    match opcode {
        CfOpcode::Nop => ParsedCf::Nop,
        CfOpcode::Exec | CfOpcode::ExecEnd => ParsedCf::Exec(ParsedExec {
            cf_index,
            opcode,
            name: if opcode == CfOpcode::ExecEnd { "exece" } else { "exec" },
            address: cf.exec_address(),
            count: cf.exec_count(),
            condition: CfCondition::Always,
            is_end: opcode == CfOpcode::ExecEnd,
            is_predicate_clean: cf.exec_is_predicate_clean(),
            is_yield: cf.exec_is_yield(),
            sequence: cf.exec_sequence(),
        }),
        CfOpcode::CondExec | CfOpcode::CondExecEnd | CfOpcode::CondExecPredClean | CfOpcode::CondExecPredCleanEnd => {
            let is_end = matches!(opcode, CfOpcode::CondExecEnd | CfOpcode::CondExecPredCleanEnd);
            ParsedCf::Exec(ParsedExec {
                cf_index,
                opcode,
                name: if is_end { "cexece" } else { "cexec" },
                address: cf.exec_address(),
                count: cf.exec_count(),
                condition: CfCondition::Bool { index: cf.cond_bool_address(), value: cf.condition() },
                is_end,
                is_predicate_clean: !matches!(opcode, CfOpcode::CondExec | CfOpcode::CondExecEnd),
                is_yield: cf.exec_is_yield(),
                sequence: cf.exec_sequence(),
            })
        }
        CfOpcode::CondExecPred | CfOpcode::CondExecPredEnd => ParsedCf::Exec(ParsedExec {
            cf_index,
            opcode,
            name: if opcode == CfOpcode::CondExecPredEnd { "exece" } else { "exec" },
            address: cf.exec_address(),
            count: cf.exec_count(),
            condition: CfCondition::Predicate(cf.condition()),
            is_end: opcode == CfOpcode::CondExecPredEnd,
            is_predicate_clean: cf.exec_is_predicate_clean(),
            is_yield: cf.exec_is_yield(),
            sequence: cf.exec_sequence(),
        }),
        CfOpcode::LoopStart => ParsedCf::LoopStart {
            loop_constant: cf.loop_id(),
            is_repeat: cf.loop_is_repeat(),
            skip_address: cf.loop_address(),
        },
        CfOpcode::LoopEnd => ParsedCf::LoopEnd {
            loop_constant: cf.loop_id(),
            body_address: cf.loop_address(),
            predicated_break: if cf.loop_is_predicated_break() { Some(cf.condition()) } else { None },
        },
        CfOpcode::CondCall => ParsedCf::Call { target: cf.target_address(), condition: call_jump_condition(cf) },
        CfOpcode::Return => ParsedCf::Return,
        CfOpcode::CondJmp => ParsedCf::Jump { target: cf.target_address(), condition: call_jump_condition(cf) },
        CfOpcode::Alloc => ParsedCf::Alloc {
            alloc_type: match cf.alloc_type() {
                0 => AllocType::None,
                1 => AllocType::VsPosition,
                2 => AllocType::InterpolatorsOrColors,
                _ => AllocType::Memory,
            },
            size: cf.alloc_size(),
        },
        CfOpcode::MarkVsFetchDone => ParsedCf::MarkVsFetchDone,
    }
}

fn parse_fetch_result(dest: u32, swizzle: u32, is_relative: bool) -> InstructionResult {
    let mut result = InstructionResult {
        target: StorageTarget::Register,
        index: dest,
        addressing: if is_relative { AddressingMode::LoopRelative } else { AddressingMode::Absolute },
        is_clamped: false,
        original_write_mask: 0b1111,
        ..Default::default()
    };
    for i in 0..4 {
        let s = (swizzle >> (3 * i)) & 7;
        result.components[i] = match s {
            0..=3 => SwizzleSource::from_component(s),
            5 => SwizzleSource::One,
            7 => {
                result.original_write_mask &= !(1 << i);
                SwizzleSource::Zero
            }
            // 4 is 0; 6 is invalid and treated as 0 like Xenia.
            _ => SwizzleSource::Zero,
        };
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexFetchAttributes {
    pub data_format: u32,
    /// Dword offset added to the element address.
    pub offset: i32,
    /// Element stride in dwords (from the vfetch_full this fetch belongs to).
    pub stride: u32,
    pub exp_adjust: i32,
    pub prefetch_count: u32,
    pub signed_rf_mode_no_zero: bool,
    pub is_index_rounded: bool,
    pub is_signed: bool,
    /// Not normalized (NumFormat=integer).
    pub is_integer: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParsedVertexFetch {
    pub is_mini_fetch: bool,
    pub is_predicated: bool,
    pub predicate_condition: bool,
    pub result: InstructionResult,
    /// Index source (single component, from the vfetch_full for vfetch_mini).
    pub index_operand: InstructionOperand,
    /// Vertex fetch constant index 0..95 (from the vfetch_full for vfetch_mini).
    pub fetch_constant: u32,
    pub attributes: VertexFetchAttributes,
}

/// Parses a vertex fetch; `previous_full` supplies the addressing of vfetch_mini.
pub fn parse_vertex_fetch(op: &RawFetch, previous_full: &RawFetch) -> ParsedVertexFetch {
    let full = if op.vf_is_mini_fetch() { previous_full } else { op };
    let mut index_operand = InstructionOperand {
        source: StorageSource::Register,
        index: full.src(),
        addressing: if full.is_src_relative() { AddressingMode::LoopRelative } else { AddressingMode::Absolute },
        component_count: 1,
        ..Default::default()
    };
    index_operand.components[0] = SwizzleSource::from_component(full.vf_src_swizzle());
    ParsedVertexFetch {
        is_mini_fetch: op.vf_is_mini_fetch(),
        is_predicated: op.is_predicated(),
        predicate_condition: op.predicate_condition(),
        result: parse_fetch_result(op.dest(), op.dest_swizzle(), op.is_dest_relative()),
        index_operand,
        fetch_constant: full.vf_fetch_constant_index(),
        attributes: VertexFetchAttributes {
            data_format: op.vf_format(),
            offset: op.vf_offset(),
            stride: full.vf_stride(),
            exp_adjust: op.vf_exp_adjust(),
            prefetch_count: op.vf_prefetch_count(),
            signed_rf_mode_no_zero: op.vf_signed_rf_mode_no_zero(),
            is_index_rounded: full.vf_is_index_rounded(),
            is_signed: op.vf_is_signed(),
            is_integer: !op.vf_is_normalized(),
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FetchDimension {
    D1,
    D2,
    D3OrStacked,
    Cube,
}

impl FetchDimension {
    pub fn from_bits(value: u32) -> FetchDimension {
        [FetchDimension::D1, FetchDimension::D2, FetchDimension::D3OrStacked, FetchDimension::Cube]
            [(value & 3) as usize]
    }
    pub fn component_count(self) -> u32 {
        match self {
            FetchDimension::D1 => 1,
            FetchDimension::D2 => 2,
            FetchDimension::D3OrStacked | FetchDimension::Cube => 3,
        }
    }
}

/// Texture filter override in a fetch instruction (`TextureFilter`); `UseFetchConst` = no override.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TextureFilter {
    Point,
    Linear,
    BaseMap,
    UseFetchConst,
}

impl TextureFilter {
    pub fn from_bits(v: u32) -> TextureFilter {
        [TextureFilter::Point, TextureFilter::Linear, TextureFilter::BaseMap, TextureFilter::UseFetchConst]
            [(v & 3) as usize]
    }
    pub fn name(self) -> &'static str {
        ["point", "linear", "basemap", "keep"][self as usize]
    }
}

/// Anisotropic filter override (`AnisoFilter`); 6 is unassigned, 7 = use fetch constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AnisoFilter {
    Disabled,
    Max1To1,
    Max2To1,
    Max4To1,
    Max8To1,
    Max16To1,
    Invalid6,
    UseFetchConst,
}

impl AnisoFilter {
    pub fn from_bits(v: u32) -> AnisoFilter {
        use AnisoFilter::*;
        [Disabled, Max1To1, Max2To1, Max4To1, Max8To1, Max16To1, Invalid6, UseFetchConst][(v & 7) as usize]
    }
    pub fn name(self) -> &'static str {
        ["disabled", "max1to1", "max2to1", "max4to1", "max8to1", "max16to1", "keep", "keep"][self as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureFetchAttributes {
    pub fetch_valid_only: bool,
    pub unnormalized_coordinates: bool,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mip_filter: TextureFilter,
    pub aniso_filter: AnisoFilter,
    pub vol_mag_filter: TextureFilter,
    pub vol_min_filter: TextureFilter,
    pub use_computed_lod: bool,
    pub use_register_lod: bool,
    pub use_register_gradients: bool,
    pub lod_bias: f32,
    pub offset: [f32; 3],
}

impl Default for TextureFetchAttributes {
    fn default() -> Self {
        TextureFetchAttributes {
            fetch_valid_only: true,
            unnormalized_coordinates: false,
            mag_filter: TextureFilter::UseFetchConst,
            min_filter: TextureFilter::UseFetchConst,
            mip_filter: TextureFilter::UseFetchConst,
            aniso_filter: AnisoFilter::UseFetchConst,
            vol_mag_filter: TextureFilter::UseFetchConst,
            vol_min_filter: TextureFilter::UseFetchConst,
            use_computed_lod: true,
            use_register_lod: false,
            use_register_gradients: false,
            lod_bias: 0.0,
            offset: [0.0; 3],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParsedTextureFetch {
    pub opcode: FetchOpcode,
    pub name: &'static str,
    pub dimension: FetchDimension,
    pub is_predicated: bool,
    pub predicate_condition: bool,
    /// `StorageTarget::None` for the set* instructions.
    pub result: InstructionResult,
    pub src: InstructionOperand,
    /// Texture fetch constant 0..31, `None` for the set* instructions.
    pub fetch_constant: Option<u32>,
    pub attributes: TextureFetchAttributes,
}

impl ParsedTextureFetch {
    pub fn has_result(&self) -> bool {
        self.result.target != StorageTarget::None
    }

    /// Result components that may be non-zero (Xenia's `GetNonZeroResultComponents`).
    pub fn non_zero_result_components(&self) -> u32 {
        let components = match self.opcode {
            FetchOpcode::TextureFetch | FetchOpcode::GetTextureGradients => 0b1111,
            FetchOpcode::GetTextureBorderColorFrac | FetchOpcode::GetTextureComputedLod => 0b0001,
            FetchOpcode::GetTextureWeights => {
                let mut c = match self.dimension {
                    FetchDimension::D1 => 0b1001,
                    FetchDimension::D2 | FetchDimension::Cube => 0b1011,
                    FetchDimension::D3OrStacked => 0b1111,
                };
                if matches!(self.attributes.mip_filter, TextureFilter::BaseMap | TextureFilter::Point) {
                    c &= !0b1000;
                }
                c
            }
            _ => 0,
        };
        self.result.used_result_components() & components
    }
}

pub fn parse_texture_fetch(op: &RawFetch) -> ParsedTextureFetch {
    let dimension = FetchDimension::from_bits(op.tf_dimension());
    let dim_index = dimension as usize;
    // (name, has_dest, has_const, has_attributes, override_component_count)
    let (name, has_dest, has_const, has_attributes, override_count) = match op.opcode() {
        FetchOpcode::TextureFetch => {
            (["tfetch1D", "tfetch2D", "tfetch3D", "tfetchCube"][dim_index], true, true, true, 0)
        }
        FetchOpcode::GetTextureBorderColorFrac => {
            (["getBCF1D", "getBCF2D", "getBCF3D", "getBCFCube"][dim_index], true, true, true, 0)
        }
        FetchOpcode::GetTextureComputedLod => (
            ["getCompTexLOD1D", "getCompTexLOD2D", "getCompTexLOD3D", "getCompTexLODCube"][dim_index],
            true,
            true,
            true,
            0,
        ),
        FetchOpcode::GetTextureGradients => ("getGradients", true, true, true, 2),
        FetchOpcode::GetTextureWeights => {
            (["getWeights1D", "getWeights2D", "getWeights3D", "getWeightsCube"][dim_index], true, true, true, 0)
        }
        FetchOpcode::SetTextureLod => ("setTexLOD", false, false, false, 1),
        FetchOpcode::SetTextureGradientsHorz => ("setGradientH", false, false, false, 3),
        FetchOpcode::SetTextureGradientsVert => ("setGradientV", false, false, false, 3),
        FetchOpcode::VertexFetch | FetchOpcode::Unknown(_) => ("tfetch_unknown", false, false, false, 1),
    };
    let result = if has_dest {
        parse_fetch_result(op.dest(), op.dest_swizzle(), op.is_dest_relative())
    } else {
        InstructionResult::default()
    };
    let mut src = InstructionOperand {
        source: StorageSource::Register,
        index: op.src(),
        addressing: if op.is_src_relative() { AddressingMode::LoopRelative } else { AddressingMode::Absolute },
        component_count: if override_count != 0 { override_count } else { dimension.component_count() },
        ..Default::default()
    };
    let swizzle = op.tf_src_swizzle();
    if op.opcode() == FetchOpcode::TextureFetch && dimension == FetchDimension::D1 {
        let x = swizzle & 3;
        if (swizzle >> 2) & 3 != x || (swizzle >> 4) & 3 != x {
            // 1D fetches may provide XY for a 2D fetch constant.
            src.component_count = 2;
        }
    }
    for j in 0..src.component_count {
        src.components[j as usize] = SwizzleSource::from_component(swizzle >> (2 * j));
    }
    let mut attributes = TextureFetchAttributes::default();
    if has_attributes {
        attributes = TextureFetchAttributes {
            fetch_valid_only: op.tf_fetch_valid_only(),
            unnormalized_coordinates: op.tf_unnormalized_coordinates(),
            mag_filter: TextureFilter::from_bits(op.tf_mag_filter()),
            min_filter: TextureFilter::from_bits(op.tf_min_filter()),
            mip_filter: TextureFilter::from_bits(op.tf_mip_filter()),
            aniso_filter: AnisoFilter::from_bits(op.tf_aniso_filter()),
            vol_mag_filter: TextureFilter::from_bits(op.tf_vol_mag_filter()),
            vol_min_filter: TextureFilter::from_bits(op.tf_vol_min_filter()),
            use_computed_lod: op.tf_use_computed_lod(),
            use_register_lod: op.tf_use_register_lod(),
            use_register_gradients: op.tf_use_register_gradients(),
            lod_bias: op.tf_lod_bias_raw() as f32 * (1.0 / 16.0),
            offset: [
                op.tf_offset_raw(0) as f32 * 0.5,
                op.tf_offset_raw(1) as f32 * 0.5,
                op.tf_offset_raw(2) as f32 * 0.5,
            ],
        };
    }
    ParsedTextureFetch {
        opcode: op.opcode(),
        name,
        dimension,
        is_predicated: op.is_predicated(),
        predicate_condition: op.predicate_condition(),
        result,
        src,
        fetch_constant: if has_const { Some(op.tf_fetch_constant_index()) } else { None },
        attributes,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParsedAlu {
    pub raw: RawAlu,
    pub vector_opcode: u32,
    pub scalar_opcode: u32,
    pub is_predicated: bool,
    pub predicate_condition: bool,
    /// Vector operation result plus the constant 0/1 export writes.
    pub vector_result: InstructionResult,
    pub scalar_result: InstructionResult,
    pub vector_operand_count: u32,
    pub vector_operands: [InstructionOperand; 3],
    pub scalar_operand_count: u32,
    pub scalar_operands: [InstructionOperand; 2],
}

fn parse_alu_operand(op: &RawAlu, i: u32, swizzle_component_count: u32, scalar_b_component: u32) -> InstructionOperand {
    let reg = op.src_reg(i);
    let mut out = InstructionOperand { is_negated: op.src_negate(i), ..Default::default() };
    if op.src_is_temp(i) {
        out.source = StorageSource::Register;
        out.index = reg & 0x3F;
        out.is_absolute_value = reg & 0x80 != 0;
        out.addressing = if reg & 0x40 != 0 { AddressingMode::LoopRelative } else { AddressingMode::Absolute };
    } else {
        out.source = StorageSource::ConstantFloat;
        out.index = reg;
        out.addressing = if op.src_const_is_addressed(i) {
            if op.is_const_address_register_relative() {
                AddressingMode::AddressRegisterRelative
            } else {
                AddressingMode::LoopRelative
            }
        } else {
            AddressingMode::Absolute
        };
        out.is_absolute_value = op.abs_constants();
    }
    out.component_count = swizzle_component_count;
    let swizzle = op.src_swizzle(i);
    let comp = |c: u32| SwizzleSource::from_component(ucode::swizzled_component_index(swizzle, c));
    match swizzle_component_count {
        1 => out.components[0] = comp(3),
        2 => {
            out.components[0] = comp(3);
            out.components[1] = comp(scalar_b_component);
        }
        _ => {
            for j in 0..4 {
                out.components[j as usize] = comp(j);
            }
        }
    }
    out
}

pub fn parse_alu(op: &RawAlu, stage: ShaderStage) -> ParsedAlu {
    let is_export = op.is_export();
    let mut target = StorageTarget::Register;
    let mut export_index = 0;
    if is_export {
        let reg = op.vector_dest();
        target = StorageTarget::None;
        if reg == export_reg::EXPORT_ADDRESS {
            target = StorageTarget::ExportAddress;
        } else if (export_reg::EXPORT_DATA_0..=export_reg::EXPORT_DATA_4).contains(&reg) {
            target = StorageTarget::ExportData;
            export_index = reg - export_reg::EXPORT_DATA_0;
        } else if stage == ShaderStage::Vertex {
            if reg < 16 {
                target = StorageTarget::Interpolator;
                export_index = reg;
            } else if reg == export_reg::VS_POSITION {
                target = StorageTarget::Position;
            } else if reg == export_reg::VS_POINT_SIZE_EDGE_FLAG_KILL_VERTEX {
                target = StorageTarget::PointSizeEdgeFlagKillVertex;
            }
        } else if reg < 4 {
            target = StorageTarget::Color;
            export_index = reg;
        } else if reg == export_reg::PS_DEPTH {
            target = StorageTarget::Depth;
        }
    }

    let vector_opcode = op.vector_opcode_bits();
    let mut vector_result = InstructionResult {
        target,
        index: if is_export { export_index } else { op.vector_dest() },
        addressing: if !is_export && op.is_vector_dest_relative() {
            AddressingMode::LoopRelative
        } else {
            AddressingMode::Absolute
        },
        is_clamped: op.vector_clamp(),
        ..Default::default()
    };
    let constant_0 = op.constant_0_write_mask();
    let constant_1 = op.constant_1_write_mask();
    vector_result.original_write_mask = op.vector_op_result_write_mask() | constant_0 | constant_1;
    for (i, component) in vector_result.components.iter_mut().enumerate() {
        *component = if constant_0 & (1 << i) != 0 {
            SwizzleSource::Zero
        } else if constant_1 & (1 << i) != 0 {
            SwizzleSource::One
        } else {
            SwizzleSource::from_component(i as u32)
        };
    }
    let vector_operand_count = ucode::vector_operand_count(vector_opcode);
    let mut vector_operands = [InstructionOperand::default(); 3];
    for i in 0..vector_operand_count {
        vector_operands[i as usize] = parse_alu_operand(op, i + 1, 4, 0);
    }

    let scalar_opcode = op.scalar_opcode_bits();
    let scalar_result = InstructionResult {
        target,
        index: if is_export { export_index } else { op.scalar_dest() },
        addressing: if !is_export && op.is_scalar_dest_relative() {
            AddressingMode::LoopRelative
        } else {
            AddressingMode::Absolute
        },
        is_clamped: op.scalar_clamp(),
        original_write_mask: op.scalar_op_result_write_mask(),
        ..Default::default()
    };
    let scalar_operand_count = ucode::scalar_operand_count(scalar_opcode);
    let mut scalar_operands = [InstructionOperand::default(); 2];
    if scalar_operand_count == 1 {
        scalar_operands[0] = parse_alu_operand(
            op,
            3,
            if ucode::scalar_operand_is_two_component(scalar_opcode) { 2 } else { 1 },
            if vector_operand_count == 3 { 2 } else { 0 },
        );
    } else if scalar_operand_count == 2 {
        let negate = op.src_negate(3);
        let swizzle = op.src_swizzle(3);
        let mut c = InstructionOperand {
            source: StorageSource::ConstantFloat,
            index: op.src_reg(3),
            addressing: if op.src_const_is_addressed(3) {
                if op.is_const_address_register_relative() {
                    AddressingMode::AddressRegisterRelative
                } else {
                    AddressingMode::LoopRelative
                }
            } else {
                AddressingMode::Absolute
            },
            is_negated: negate,
            is_absolute_value: op.abs_constants(),
            component_count: 1,
            ..Default::default()
        };
        c.components[0] = SwizzleSource::from_component(ucode::swizzled_component_index(swizzle, 3));
        let mut t = InstructionOperand {
            source: StorageSource::Register,
            index: op.scalar_const_reg_op_src_temp_reg(),
            addressing: AddressingMode::Absolute,
            is_negated: negate,
            is_absolute_value: op.abs_constants(),
            component_count: 1,
            ..Default::default()
        };
        t.components[0] = SwizzleSource::from_component(ucode::swizzled_component_index(swizzle, 0));
        scalar_operands = [c, t];
    }
    ParsedAlu {
        raw: *op,
        vector_opcode,
        scalar_opcode,
        is_predicated: op.is_predicated(),
        predicate_condition: op.predicate_condition(),
        vector_result,
        scalar_result,
        vector_operand_count,
        vector_operands,
        scalar_operand_count,
        scalar_operands,
    }
}

impl ParsedAlu {
    pub fn vector_name(&self) -> &'static str {
        ucode::VECTOR_NAMES[self.vector_opcode as usize]
    }
    pub fn scalar_name(&self) -> &'static str {
        ucode::SCALAR_NAMES[self.scalar_opcode as usize]
    }

    pub fn is_vector_op_default_nop(&self) -> bool {
        let standard_r0 = |o: &InstructionOperand| {
            o.source == StorageSource::Register
                && o.index == 0
                && o.addressing == AddressingMode::Absolute
                && !o.is_negated
                && !o.is_absolute_value
                && o.is_standard_swizzle()
        };
        if self.vector_opcode != ucode::vector_op::MAX
            || self.vector_result.original_write_mask != 0
            || self.vector_result.is_clamped
            || !standard_r0(&self.vector_operands[0])
            || !standard_r0(&self.vector_operands[1])
        {
            return false;
        }
        if self.vector_result.target == StorageTarget::Register {
            if self.vector_result.index != 0 || self.vector_result.addressing != AddressingMode::Absolute {
                return false;
            }
        } else if self.is_scalar_op_default_nop() {
            return false;
        }
        true
    }

    pub fn is_scalar_op_default_nop(&self) -> bool {
        if self.scalar_opcode != ucode::scalar_op::RETAIN_PREV
            || self.scalar_result.original_write_mask != 0
            || self.scalar_result.is_clamped
        {
            return false;
        }
        if self.scalar_result.target == StorageTarget::Register
            && (self.scalar_result.index != 0 || self.scalar_result.addressing != AddressingMode::Absolute)
        {
            return false;
        }
        true
    }

    /// Whether the instruction has no effect at all.
    pub fn is_nop(&self) -> bool {
        self.scalar_opcode == ucode::scalar_op::RETAIN_PREV
            && self.scalar_result.used_write_mask() == 0
            && self.vector_result.used_write_mask() == 0
            && !ucode::vector_changes_state(self.vector_opcode)
    }

    /// The memexport stream constant (`mad eA, r#, const0100, c#`), if recognizable.
    pub fn memexport_stream_constant(&self) -> Option<u32> {
        let o = &self.vector_operands[2];
        if self.vector_result.target == StorageTarget::ExportAddress
            && self.vector_opcode == ucode::vector_op::MAD
            && self.vector_result.used_result_components() == 0b1111
            && !self.vector_result.is_clamped
            && o.source == StorageSource::ConstantFloat
            && o.addressing == AddressingMode::Absolute
            && o.is_standard_swizzle()
            && !o.is_negated
            && !o.is_absolute_value
        {
            Some(o.index)
        } else {
            None
        }
    }
}
