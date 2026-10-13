//! Static analysis of a decoded program: resources used and outputs written.

use std::collections::BTreeSet;

use crate::parsed::{
    AddressingMode, CfCondition, FetchDimension, InstructionOperand, InstructionResult, ParsedCf, StorageSource,
    StorageTarget, TextureFilter,
};
use crate::program::{Instruction, Program};
use crate::ucode::{self, FetchOpcode};

/// One vertex attribute fetched by a `vfetch_full`/`vfetch_mini`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    /// Instruction slot of the fetch.
    pub address: u32,
    pub data_format: u32,
    /// Dword offset within the element.
    pub offset: i32,
    pub exp_adjust: i32,
    pub is_signed: bool,
    /// `NumFormat=integer` (not normalized).
    pub is_integer: bool,
    pub signed_rf_mode_no_zero: bool,
    pub is_index_rounded: bool,
    pub is_mini_fetch: bool,
    /// Destination temporary register (absolute index or base for aL-relative).
    pub dest_register: u32,
    /// Format components whose values are used.
    pub used_components: u32,
}

/// Attributes fetched through one vertex fetch constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexBinding {
    /// Vertex fetch constant 0..95 (dwords `2 * index` and `2 * index + 1` of the fetch constants).
    pub fetch_constant: u32,
    pub stride_dwords: u32,
    pub attributes: Vec<VertexAttribute>,
}

/// A texture-related fetch instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureFetchUse {
    pub address: u32,
    pub opcode: FetchOpcode,
    pub fetch_constant: u32,
    pub dimension: FetchDimension,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mip_filter: TextureFilter,
    pub aniso_filter: crate::parsed::AnisoFilter,
    pub unnormalized_coordinates: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderInfo {
    pub cf_pair_bound: u32,
    pub labels: BTreeSet<u32>,
    /// One more than the highest statically addressed temporary register.
    pub register_static_address_bound: u32,
    pub uses_register_dynamic_addressing: bool,
    /// Float constants c0..c255 referenced with absolute addressing (bit per constant).
    pub float_constants: [u64; 4],
    /// Float constants are indexed with a0 or aL (any of the 256 may be read).
    pub float_dynamic_addressing: bool,
    pub bool_constants: [u32; 8],
    pub loop_constants: u32,
    pub vertex_fetch_constants: [u32; 3],
    pub vertex_bindings: Vec<VertexBinding>,
    pub texture_fetches: Vec<TextureFetchUse>,
    pub uses_texture_fetch_results: bool,
    /// Uses register LOD / gradients (`setTexLOD`, `setGradientH/V`).
    pub sets_texture_lod_or_gradients: bool,
    pub kills_pixels: bool,
    pub writes_position: bool,
    pub writes_interpolators: u32,
    /// Bits: 0 point size, 1 edge flag, 2 kill vertex.
    pub writes_point_size_edge_flag_kill_vertex: u32,
    pub writes_color_targets: u32,
    pub writes_depth: bool,
    pub uses_address_register: bool,
    pub uses_loops: bool,
    pub uses_calls: bool,
    pub uses_jumps: bool,
    pub uses_predicate: bool,
    /// eM# written (bit per export data register).
    pub memexport_data_written: u8,
    pub memexport_address_written: bool,
    pub memexport_stream_constants: BTreeSet<u32>,
}

impl ShaderInfo {
    pub fn uses_memexport(&self) -> bool {
        self.memexport_data_written != 0 || self.memexport_address_written
    }

    /// How many float constants from c0 on the shader may read (all 256 with dynamic addressing).
    pub fn float_constants_read(&self) -> u32 {
        if self.float_dynamic_addressing {
            return 256;
        }
        (0..4)
            .rev()
            .find(|&w| self.float_constants[w] != 0)
            .map_or(0, |w| w as u32 * 64 + 64 - self.float_constants[w].leading_zeros())
    }

    pub fn is_float_constant_used(&self, index: u32) -> bool {
        self.float_dynamic_addressing
            || (index < 256 && self.float_constants[(index >> 6) as usize] >> (index & 63) & 1 != 0)
    }

    fn operand(&mut self, op: &InstructionOperand) {
        match op.source {
            StorageSource::Register => {
                if op.addressing == AddressingMode::Absolute {
                    self.register_static_address_bound = self.register_static_address_bound.max(op.index + 1);
                } else {
                    self.uses_register_dynamic_addressing = true;
                }
                if op.addressing == AddressingMode::LoopRelative {
                    self.uses_loops = true;
                }
            }
            StorageSource::ConstantFloat => match op.addressing {
                AddressingMode::Absolute => {
                    self.float_constants[(op.index >> 6) as usize & 3] |= 1u64 << (op.index & 63);
                }
                AddressingMode::AddressRegisterRelative => {
                    self.float_dynamic_addressing = true;
                    self.uses_address_register = true;
                }
                AddressingMode::LoopRelative => self.float_dynamic_addressing = true,
            },
            StorageSource::VertexFetchConstant | StorageSource::TextureFetchConstant => {}
        }
    }

    fn fetch_result(&mut self, result: &InstructionResult) {
        if result.used_write_mask() == 0 {
            return;
        }
        if result.addressing == AddressingMode::Absolute {
            self.register_static_address_bound = self.register_static_address_bound.max(result.index + 1);
        } else {
            self.uses_register_dynamic_addressing = true;
        }
    }

    fn alu_result(&mut self, result: &InstructionResult) {
        let mask = result.used_write_mask();
        if mask == 0 {
            return;
        }
        match result.target {
            StorageTarget::Register => {
                if result.addressing == AddressingMode::Absolute {
                    self.register_static_address_bound = self.register_static_address_bound.max(result.index + 1);
                } else {
                    self.uses_register_dynamic_addressing = true;
                }
            }
            StorageTarget::Interpolator => self.writes_interpolators |= 1 << result.index,
            StorageTarget::Position => self.writes_position = true,
            StorageTarget::PointSizeEdgeFlagKillVertex => self.writes_point_size_edge_flag_kill_vertex |= mask,
            StorageTarget::ExportData => self.memexport_data_written |= 1 << result.index,
            StorageTarget::ExportAddress => self.memexport_address_written = true,
            StorageTarget::Color => self.writes_color_targets |= 1 << result.index,
            StorageTarget::Depth => self.writes_depth = true,
            StorageTarget::None => {}
        }
    }

    pub fn analyze(program: &Program) -> ShaderInfo {
        let mut info =
            ShaderInfo { cf_pair_bound: program.cf_pair_bound, labels: program.labels.clone(), ..Default::default() };
        let use_bool = |info: &mut ShaderInfo, c: &CfCondition| match c {
            CfCondition::Bool { index, .. } => info.bool_constants[(*index >> 5) as usize] |= 1 << (index & 31),
            CfCondition::Predicate(_) => info.uses_predicate = true,
            CfCondition::Always => {}
        };
        for entry in &program.cf {
            match &entry.cf {
                ParsedCf::Exec(e) => use_bool(&mut info, &e.condition),
                ParsedCf::LoopStart { loop_constant, .. } => {
                    info.loop_constants |= 1 << loop_constant;
                    info.uses_loops = true;
                }
                ParsedCf::LoopEnd { loop_constant, predicated_break, .. } => {
                    info.loop_constants |= 1 << loop_constant;
                    info.uses_loops = true;
                    if predicated_break.is_some() {
                        info.uses_predicate = true;
                    }
                }
                ParsedCf::Call { condition, .. } => {
                    use_bool(&mut info, condition);
                    info.uses_calls = true;
                }
                ParsedCf::Jump { condition, .. } => {
                    use_bool(&mut info, condition);
                    info.uses_jumps = true;
                }
                ParsedCf::Return => info.uses_calls = true,
                _ => {}
            }
        }
        for (_, instr) in program.instructions() {
            match &instr.instruction {
                Instruction::Alu(alu) => {
                    if alu.is_predicated {
                        info.uses_predicate = true;
                    }
                    if ucode::vector_kills(alu.vector_opcode) || ucode::scalar_kills(alu.scalar_opcode) {
                        info.kills_pixels = true;
                    }
                    if ucode::vector_changes_predicate(alu.vector_opcode)
                        || ucode::scalar_changes_predicate(alu.scalar_opcode)
                    {
                        info.uses_predicate = true;
                    }
                    if ucode::vector_changes_address(alu.vector_opcode)
                        || ucode::scalar_changes_address(alu.scalar_opcode)
                    {
                        info.uses_address_register = true;
                    }
                    info.alu_result(&alu.vector_result);
                    info.alu_result(&alu.scalar_result);
                    for operand in &alu.vector_operands[..alu.vector_operand_count as usize] {
                        info.operand(operand);
                    }
                    for operand in &alu.scalar_operands[..alu.scalar_operand_count as usize] {
                        info.operand(operand);
                    }
                    if alu.vector_result.target == StorageTarget::ExportAddress
                        && let Some(c) = alu.memexport_stream_constant()
                    {
                        info.memexport_stream_constants.insert(c);
                    }
                    if alu.vector_result.addressing == AddressingMode::LoopRelative
                        || alu.scalar_result.addressing == AddressingMode::LoopRelative
                    {
                        info.uses_loops = true;
                    }
                }
                Instruction::VertexFetch(vf) => {
                    if vf.is_predicated {
                        info.uses_predicate = true;
                    }
                    info.fetch_result(&vf.result);
                    if !vf.is_mini_fetch {
                        info.operand(&vf.index_operand);
                    }
                    let used = vf.result.used_result_components();
                    if used == 0 {
                        continue;
                    }
                    info.vertex_fetch_constants[(vf.fetch_constant >> 5) as usize] |= 1 << (vf.fetch_constant & 31);
                    let a = &vf.attributes;
                    let attribute = VertexAttribute {
                        address: instr.address,
                        data_format: a.data_format,
                        offset: a.offset,
                        exp_adjust: a.exp_adjust,
                        is_signed: a.is_signed,
                        is_integer: a.is_integer,
                        signed_rf_mode_no_zero: a.signed_rf_mode_no_zero,
                        is_index_rounded: a.is_index_rounded,
                        is_mini_fetch: vf.is_mini_fetch,
                        dest_register: vf.result.index,
                        used_components: used & ((1 << ucode::vertex_format::component_count(a.data_format)) - 1),
                    };
                    match info.vertex_bindings.iter_mut().find(|b| b.fetch_constant == vf.fetch_constant) {
                        Some(binding) => binding.attributes.push(attribute),
                        None => info.vertex_bindings.push(VertexBinding {
                            fetch_constant: vf.fetch_constant,
                            stride_dwords: a.stride,
                            attributes: vec![attribute],
                        }),
                    }
                }
                Instruction::TextureFetch(tf) => {
                    if tf.is_predicated {
                        info.uses_predicate = true;
                    }
                    info.fetch_result(&tf.result);
                    info.operand(&tf.src);
                    if tf.result.used_result_components() != 0 {
                        info.uses_texture_fetch_results = true;
                    }
                    match tf.opcode {
                        FetchOpcode::SetTextureLod
                        | FetchOpcode::SetTextureGradientsHorz
                        | FetchOpcode::SetTextureGradientsVert => info.sets_texture_lod_or_gradients = true,
                        _ => {
                            if let Some(fc) = tf.fetch_constant {
                                info.texture_fetches.push(TextureFetchUse {
                                    address: instr.address,
                                    opcode: tf.opcode,
                                    fetch_constant: fc,
                                    dimension: tf.dimension,
                                    mag_filter: tf.attributes.mag_filter,
                                    min_filter: tf.attributes.min_filter,
                                    mip_filter: tf.attributes.mip_filter,
                                    aniso_filter: tf.attributes.aniso_filter,
                                    unnormalized_coordinates: tf.attributes.unnormalized_coordinates,
                                });
                            }
                        }
                    }
                }
            }
        }
        if info.float_dynamic_addressing {
            info.float_constants = [u64::MAX; 4];
        }
        info
    }
}
