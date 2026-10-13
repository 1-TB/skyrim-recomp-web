//! Shader program setup: SQ_* registers and the PM4 packets that select the active shaders.
//!
//! On the Xbox 360, Direct3D loads microcode with PM4 `IM_LOAD` (pointer to guest memory) or
//! `IM_LOAD_IMMEDIATE` (code embedded in the packet). Like Xenia, the most recently loaded vertex and
//! pixel shaders are the active ones for subsequent draws; SQ_VS_PROGRAM / SQ_PS_PROGRAM are not used
//! to locate them.

use crate::RegisterSource;
use crate::bitfields::*;
use crate::registers as r;
use crate::xenos::*;

/// PM4 type 3 opcode of IM_LOAD.
pub const PM4_IM_LOAD: u32 = 0x27;
/// PM4 type 3 opcode of IM_LOAD_IMMEDIATE.
pub const PM4_IM_LOAD_IMMEDIATE: u32 = 0x2B;

/// Where shader microcode comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderCodeLocation {
    /// Guest physical address of the microcode (IM_LOAD).
    Guest(u32),
    /// Embedded in the command buffer; the code dwords follow the two header dwords (IM_LOAD_IMMEDIATE).
    Immediate,
}

/// A decoded IM_LOAD / IM_LOAD_IMMEDIATE packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderLoad {
    pub shader_type: ShaderType,
    pub location: ShaderCodeLocation,
    /// Start offset within instruction memory (always 0 in practice).
    pub start: u32,
    /// Microcode size in dwords.
    pub size_dwords: u32,
}

impl ShaderLoad {
    /// Parses an IM_LOAD payload (host-order dwords): `addr | type`, `start << 16 | size`.
    pub fn parse_im_load(payload: &[u32]) -> Option<Self> {
        let (&addr_type, &start_size) = (payload.first()?, payload.get(1)?);
        Some(Self {
            shader_type: ShaderType::from_raw(addr_type & 3),
            location: ShaderCodeLocation::Guest(addr_type & !3),
            start: start_size >> 16,
            size_dwords: start_size & 0xFFFF,
        })
    }

    /// Parses an IM_LOAD_IMMEDIATE payload: `type`, `start << 16 | size`, then the code. Returns the
    /// load and the embedded code (`None` if the payload is shorter than the declared size).
    pub fn parse_im_load_immediate(payload: &[u32]) -> Option<(Self, &[u32])> {
        let (&ty, &start_size) = (payload.first()?, payload.get(1)?);
        let load = Self {
            shader_type: ShaderType::from_raw(ty & 3),
            location: ShaderCodeLocation::Immediate,
            start: start_size >> 16,
            size_dwords: start_size & 0xFFFF,
        };
        let code = payload.get(2..2 + load.size_dwords as usize)?;
        Some((load, code))
    }
}

/// The currently bound shaders, updated by the command processor on each IM_LOAD packet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActiveShaders {
    pub vertex: Option<ShaderLoad>,
    pub pixel: Option<ShaderLoad>,
}

impl ActiveShaders {
    /// Records a shader load as the active shader of its stage.
    pub fn apply(&mut self, load: ShaderLoad) {
        match load.shader_type {
            ShaderType::Vertex => self.vertex = Some(load),
            ShaderType::Pixel => self.pixel = Some(load),
        }
    }
}

/// Shader-related draw state decoded from the SQ registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderProgramSetup {
    pub program_cntl: SqProgramCntl,
    pub context_misc: SqContextMisc,
    pub interpolator_cntl: SqInterpolatorCntl,
    /// Float constant window of the vertex shader (SQ_VS_CONST).
    pub vs_const: SqConstRange,
    /// Float constant window of the pixel shader (SQ_PS_CONST).
    pub ps_const: SqConstRange,
    /// Raw SQ_VS_PROGRAM / SQ_PS_PROGRAM (informational).
    pub vs_program: SqProgramBase,
    pub ps_program: SqProgramBase,
    /// Vertex shader temporary register count.
    pub vs_register_count: u32,
    /// Pixel shader temporary register count.
    pub ps_register_count: u32,
    /// Interpolators exported by the vertex shader (1..16).
    pub interpolator_count: u32,
    /// Interpolator receiving PsParamGen, if enabled.
    pub param_gen_interpolator: Option<u32>,
    /// Whether the vertex index is preloaded into r0.x of the vertex shader.
    pub vertex_index_in_r0: bool,
    /// Per-interpolator sample location bits (1 = center, 0 = centroid), resolved for the MSAA mode.
    pub sampling_pattern: u32,
    /// Whether the pixel shader runs at all (EDRAM mode must be color + depth).
    pub pixel_shader_enabled: bool,
    /// Whether the draw rasterizes anything (false for memexport-only multipass vertex shaders).
    pub rasterization_enabled: bool,
}

/// Resolves the per-interpolator sampling locations (GetInterpolatorSamplingPattern).
pub fn interpolator_sampling_pattern(msaa: MsaaSamples, control: SampleControl, pattern: u32) -> u32 {
    let all = (1u32 << MAX_INTERPOLATORS) - 1;
    if msaa == MsaaSamples::X1 || control == SampleControl::CentersOnly {
        return all;
    }
    if control == SampleControl::CentroidsOnly {
        return 0;
    }
    pattern & all
}

impl ShaderProgramSetup {
    /// Decodes the shader setup from the registers.
    pub fn from_registers(regs: &(impl RegisterSource + ?Sized)) -> Self {
        let program_cntl = SqProgramCntl::read(regs);
        let context_misc = SqContextMisc::read(regs);
        let interpolator_cntl = SqInterpolatorCntl::read(regs);
        let msaa = RbSurfaceInfo::read(regs).msaa_samples;
        let edram_mode = RbModecontrol::read(regs).edram_mode;
        Self {
            program_cntl,
            context_misc,
            interpolator_cntl,
            vs_const: SqConstRange::decode(regs.reg(r::SQ_VS_CONST)),
            ps_const: SqConstRange::decode(regs.reg(r::SQ_PS_CONST)),
            vs_program: SqProgramBase::decode(regs.reg(r::SQ_VS_PROGRAM)),
            ps_program: SqProgramBase::decode(regs.reg(r::SQ_PS_PROGRAM)),
            vs_register_count: program_cntl.vs_num_reg + 1,
            ps_register_count: program_cntl.ps_num_reg + 1,
            interpolator_count: program_cntl.vs_export_count + 1,
            param_gen_interpolator: program_cntl.param_gen.then_some(context_misc.param_gen_pos),
            vertex_index_in_r0: program_cntl.gen_index_vtx,
            sampling_pattern: interpolator_sampling_pattern(
                msaa,
                context_misc.sc_sample_cntl,
                interpolator_cntl.sampling_pattern,
            ),
            pixel_shader_enabled: edram_mode == EdramMode::ColorDepth,
            rasterization_enabled: program_cntl.vs_export_mode != VertexShaderExportMode::Multipass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn im_load_packets() {
        let l = ShaderLoad::parse_im_load(&[0x0012_3401, 0x0000_0040]).unwrap();
        assert_eq!(l.shader_type, ShaderType::Pixel);
        assert_eq!(l.location, ShaderCodeLocation::Guest(0x0012_3400));
        assert_eq!((l.start, l.size_dwords), (0, 0x40));
        assert!(ShaderLoad::parse_im_load(&[1]).is_none());

        let payload = [0, 3, 0xA, 0xB, 0xC, 0xD];
        let (l, code) = ShaderLoad::parse_im_load_immediate(&payload).unwrap();
        assert_eq!(l.shader_type, ShaderType::Vertex);
        assert_eq!(l.location, ShaderCodeLocation::Immediate);
        assert_eq!(code, &[0xA, 0xB, 0xC]);
        assert!(ShaderLoad::parse_im_load_immediate(&[0, 5, 1]).is_none());

        let mut active = ActiveShaders::default();
        active.apply(l);
        active.apply(ShaderLoad::parse_im_load(&[0x1000_0001, 2]).unwrap());
        assert_eq!(active.vertex.unwrap().location, ShaderCodeLocation::Immediate);
        assert_eq!(active.pixel.unwrap().location, ShaderCodeLocation::Guest(0x1000_0000));
    }

    #[test]
    fn setup_from_registers() {
        let mut regs = vec![0u32; 0x5003];
        regs[r::SQ_PROGRAM_CNTL as usize] = 3 | (7 << 8) | (1 << 18) | (5 << 20) | (1 << 31);
        regs[r::SQ_CONTEXT_MISC as usize] = (2 << 2) | (6 << 8);
        regs[r::SQ_INTERPOLATOR_CNTL as usize] = 0x00F0_0000;
        regs[r::RB_SURFACE_INFO as usize] = 1280 | (1 << 16);
        regs[r::RB_MODECONTROL as usize] = 4;
        regs[r::SQ_VS_CONST as usize] = 0xFF << 12;
        regs[r::SQ_PS_CONST as usize] = 0x100 | (0xFF << 12);
        let s = ShaderProgramSetup::from_registers(&regs);
        assert_eq!((s.vs_register_count, s.ps_register_count), (4, 8));
        assert_eq!(s.interpolator_count, 6);
        assert_eq!(s.param_gen_interpolator, Some(6));
        assert!(s.vertex_index_in_r0 && s.pixel_shader_enabled && s.rasterization_enabled);
        assert_eq!(s.sampling_pattern, 0x00F0);
        assert_eq!((s.ps_const.base, s.ps_const.size), (0x100, 0xFF));

        regs[r::SQ_PROGRAM_CNTL as usize] = 7 << 24;
        regs[r::RB_MODECONTROL as usize] = 5;
        regs[r::RB_SURFACE_INFO as usize] = 1280;
        let s = ShaderProgramSetup::from_registers(&regs);
        assert!(!s.rasterization_enabled && !s.pixel_shader_enabled);
        assert_eq!(s.param_gen_interpolator, None);
        assert_eq!(s.sampling_pattern, 0xFFFF);
    }

    #[test]
    fn sampling_patterns() {
        assert_eq!(interpolator_sampling_pattern(MsaaSamples::X4, SampleControl::CentroidsOnly, 0x1234), 0);
        assert_eq!(interpolator_sampling_pattern(MsaaSamples::X4, SampleControl::CentersOnly, 0), 0xFFFF);
        assert_eq!(interpolator_sampling_pattern(MsaaSamples::X2, SampleControl::CentroidsAndCenters, 0x1234), 0x1234);
        assert_eq!(interpolator_sampling_pattern(MsaaSamples::X1, SampleControl::CentroidsOnly, 0), 0xFFFF);
    }
}
