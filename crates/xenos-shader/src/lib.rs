//! Xenos (Xbox 360 GPU) shader microcode to WGSL translation.
//!
//! Input is the microcode as uploaded by PM4 `IM_LOAD` / `IM_LOAD_IMMEDIATE`, as host-order dwords.
//! [`Shader::new`] decodes and analyzes it, [`Shader::disassemble`] produces Xenia-compatible
//! disassembly and [`Shader::translate`] generates WGSL using the fixed binding interface described
//! in [`bindings`]. [`interpreter`] is a CPU reference implementation of the instruction semantics.
//!
//! Texture contract: each texture view exposes the guest data components in RGBA order as UNORM
//! (or float for float formats, not sRGB); the shader replicates missing components, applies the
//! fetch constant's per-component signedness / gamma / integer number format, swizzle and exponent
//! bias, and handles offsets, unnormalized coordinates, LOD biases and register LOD / gradients.
//!
//! Not performed by the generated WGSL (reported instead): memory export (`eA`/`eM#` writes, see
//! [`ShaderInfo::uses_memexport`]) since shared memory is bound read-only, `getBCF` (returns 0),
//! point sprite expansion (the point size is optionally output for the host to expand), and
//! wide (> 8192 texel) 1D textures.

pub mod analysis;
pub mod bindings;
pub mod disasm;
pub mod interpreter;
pub mod parsed;
pub mod program;
pub mod ucode;
mod wgsl;

pub use analysis::{ShaderInfo, TextureFetchUse, VertexAttribute, VertexBinding};
pub use bindings::{BoolLoopConstants, SystemConstants};
pub use program::{Program, UcodeError};
pub use wgsl::{TextureBinding, TextureBindingDimension};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Pixel,
}

/// Options affecting the generated WGSL interface.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TranslationOptions {
    /// Vertex shaders: interpolators output at `@location(i)` (bit per interpolator). The default
    /// outputs all 16 so any pixel shader can be paired with any vertex shader; interpolators the
    /// shader doesn't write are zero.
    pub vs_output_interpolators: u32,
    /// Vertex shaders: output the clamped oPts.x point diameter (negative if not exported) as an
    /// `f32` at this location, for host point sprite expansion.
    pub vs_point_size_location: Option<u32>,
    /// Pixel shaders: number of interpolators read into r0.. (`None`: the shader's register count,
    /// at most 16 - which is what SQ_PROGRAM_CNTL::PS_NUM_REG normally implies).
    pub ps_interpolator_count: Option<u32>,
    /// Pixel shaders: support `SystemConstants::flags::PS_PARAM_GEN` (uses `@builtin(front_facing)`).
    pub ps_param_gen: bool,
    /// First `@group(2)` binding number this shader's textures and samplers get. The vertex and
    /// pixel shader of a pipeline share group 2, so their ranges must not overlap.
    pub texture_binding_base: u32,
    /// Vertex shaders: primitive expansion performed in place of a geometry shader.
    pub vs_expansion: VertexExpansion,
}

/// Vertex shader primitive expansion (WebGPU has no geometry shaders). In both expanded modes the
/// host draws two-triangle strips with host vertex index `primitive * 4 + corner`, and the guest
/// vertex index is taken from index position `primitive` (point sprites) or `3 * primitive + k`
/// (rectangles) - through `SystemConstants::vertex_index_load_address` with
/// `flags::VERTEX_INDEX_LOAD`, or directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VertexExpansion {
    #[default]
    None,
    /// Each point becomes a screen-aligned square of the point diameter (oPts.x when exported,
    /// otherwise `SystemConstants::point_constant_diameter`).
    PointSprite,
    /// Each rectangle (3 guest vertices) becomes a quad: like Xenia's geometry shader, the vertex
    /// opposite the longest edge is mirrored across it to make the fourth vertex, so the shader
    /// runs the microcode for all three guest vertices.
    RectangleList,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        TranslationOptions {
            vs_output_interpolators: 0xFFFF,
            vs_point_size_location: None,
            ps_interpolator_count: None,
            ps_param_gen: true,
            texture_binding_base: 0,
            vs_expansion: VertexExpansion::None,
        }
    }
}

/// A translated shader and the metadata needed to bind it.
#[derive(Clone, Debug)]
pub struct Translation {
    pub stage: ShaderStage,
    pub wgsl: String,
    pub entry_point: &'static str,
    /// Textures and samplers in `@group(2)`, in binding order.
    pub texture_bindings: Vec<TextureBinding>,
    /// Vertex fetches (vertex data is read from shared memory by the shader itself).
    pub vertex_bindings: Vec<VertexBinding>,
    /// Size of the temporary register file.
    pub register_count: u32,
    /// Vertex shaders: interpolators output (bit per `@location`).
    pub vs_output_interpolators: u32,
    /// Vertex shaders: interpolators actually written by the shader.
    pub interpolators_written: u32,
    /// Pixel shaders: interpolators read from `@location(i)`.
    pub ps_input_interpolators: u32,
    /// Pixel shaders: color targets output at `@location(i)`.
    pub color_targets_written: u32,
    pub writes_depth: bool,
    pub kills_pixels: bool,
    /// The vertex shader exports a point size (oPts.x).
    pub writes_point_size: bool,
    /// The vertex shader exports the vertex kill flag (oPts.z), handled by a NaN position W.
    pub writes_vertex_kill: bool,
    /// The shader reads `@group(1)` shared memory (vertex shaders always may, for the vertex index).
    pub uses_shared_memory: bool,
    /// Control flow was emitted as structured WGSL (otherwise as a dispatch loop over the control
    /// flow instruction index).
    pub structured_control_flow: bool,
    /// Memory export is used but not performed (see the crate documentation).
    pub uses_memexport: bool,
    /// Non-fatal translation issues (unsupported or invalid instructions).
    pub warnings: Vec<String>,
    pub info: ShaderInfo,
}

/// A decoded and analyzed shader program.
#[derive(Clone, Debug)]
pub struct Shader {
    program: Program,
    info: ShaderInfo,
}

impl Shader {
    pub fn new(stage: ShaderStage, ucode: &[u32]) -> Result<Shader, UcodeError> {
        let program = Program::decode(stage, ucode)?;
        let info = ShaderInfo::analyze(&program);
        Ok(Shader { program, info })
    }

    pub fn stage(&self) -> ShaderStage {
        self.program.stage
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn info(&self) -> &ShaderInfo {
        &self.info
    }

    pub fn disassemble(&self) -> String {
        disasm::program(&self.program)
    }

    pub fn translate(&self, options: &TranslationOptions) -> Translation {
        let output = wgsl::Generator::new(&self.program, &self.info, options).generate();
        let info = &self.info;
        Translation {
            stage: self.program.stage,
            wgsl: output.wgsl,
            entry_point: output.entry_point,
            texture_bindings: output.texture_bindings,
            vertex_bindings: info.vertex_bindings.clone(),
            register_count: output.register_count,
            vs_output_interpolators: output.vs_output_interpolators,
            interpolators_written: info.writes_interpolators,
            ps_input_interpolators: output.ps_input_interpolators,
            color_targets_written: if self.program.stage == ShaderStage::Pixel { info.writes_color_targets } else { 0 },
            writes_depth: self.program.stage == ShaderStage::Pixel && info.writes_depth,
            kills_pixels: self.program.stage == ShaderStage::Pixel && info.kills_pixels,
            writes_point_size: info.writes_point_size_edge_flag_kill_vertex & 1 != 0,
            writes_vertex_kill: info.writes_point_size_edge_flag_kill_vertex & 4 != 0,
            uses_shared_memory: output.uses_shared_memory,
            structured_control_flow: output.structured_control_flow,
            uses_memexport: info.uses_memexport(),
            warnings: {
                let mut w = output.warnings;
                if info.uses_memexport() {
                    w.push("memory export (eA/eM#) is not performed".to_string());
                }
                w
            },
            info: info.clone(),
        }
    }
}

/// Disassembles microcode in Xenia's format.
pub fn disassemble(stage: ShaderStage, ucode: &[u32]) -> Result<String, UcodeError> {
    Ok(Shader::new(stage, ucode)?.disassemble())
}

/// Decodes, analyzes and translates microcode to WGSL.
pub fn translate(stage: ShaderStage, ucode: &[u32], options: &TranslationOptions) -> Result<Translation, UcodeError> {
    Ok(Shader::new(stage, ucode)?.translate(options))
}
