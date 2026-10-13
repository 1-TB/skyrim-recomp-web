//! Depth/stencil state (RB_DEPTHCONTROL, RB_STENCILREFMASK, RB_STENCILREFMASK_BF) mapped to
//! WebGPU-style descriptions.

use crate::RegisterSource;
use crate::bitfields::*;
use crate::registers as r;
use crate::viewport::is_viz_survey_draw;
use crate::xenos::*;

impl CompareFunction {
    /// WebGPU `GPUCompareFunction` name.
    pub fn webgpu_name(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::Less => "less",
            Self::Equal => "equal",
            Self::LessEqual => "less-equal",
            Self::Greater => "greater",
            Self::NotEqual => "not-equal",
            Self::GreaterEqual => "greater-equal",
            Self::Always => "always",
        }
    }

    /// Evaluates `reference <op> value` like the hardware alpha / copy tests (`value` is the incoming
    /// value, `reference` the register reference).
    pub fn passes(self, value: f32, reference: f32) -> bool {
        match self {
            Self::Never => false,
            Self::Less => value < reference,
            Self::Equal => value == reference,
            Self::LessEqual => value <= reference,
            Self::Greater => value > reference,
            Self::NotEqual => value != reference,
            Self::GreaterEqual => value >= reference,
            Self::Always => true,
        }
    }
}

impl StencilOp {
    /// WebGPU `GPUStencilOperation` name.
    pub fn webgpu_name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Zero => "zero",
            Self::Replace => "replace",
            Self::IncrementClamp => "increment-clamp",
            Self::DecrementClamp => "decrement-clamp",
            Self::Invert => "invert",
            Self::IncrementWrap => "increment-wrap",
            Self::DecrementWrap => "decrement-wrap",
        }
    }
}

/// RB_DEPTHCONTROL with depth/stencil forced off when the EDRAM mode doesn't use them, VIZ survey
/// draws made test-only, and always-passing read-only depth disabled (GetNormalizedDepthControl).
pub fn normalized_depth_control(regs: &(impl RegisterSource + ?Sized), viz_queries_emulated: bool) -> RbDepthcontrol {
    let mode = RbModecontrol::read(regs).edram_mode;
    if mode != EdramMode::ColorDepth && mode != EdramMode::DepthOnly {
        return RbDepthcontrol::decode(0);
    }
    let mut dc = RbDepthcontrol::read(regs);
    if is_viz_survey_draw(regs, viz_queries_emulated) {
        dc.z_write_enable = false;
        dc.stencil_enable = false;
        if !RbHizcontrol::read(regs).hiz_enable {
            dc.z_enable = false;
        }
    }
    if dc.z_enable && !dc.z_write_enable && dc.zfunc == CompareFunction::Always {
        dc.z_enable = false;
    }
    dc
}

/// Per-face stencil state (WebGPU `GPUStencilFaceState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
}

impl StencilFaceState {
    const DISABLED: Self = Self {
        compare: CompareFunction::Always,
        fail_op: StencilOp::Keep,
        depth_fail_op: StencilOp::Keep,
        pass_op: StencilOp::Keep,
    };
}

/// Stencil reference and masks of one face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilRefMask {
    pub reference: u8,
    pub read_mask: u8,
    pub write_mask: u8,
}

impl From<RbStencilrefmask> for StencilRefMask {
    fn from(v: RbStencilrefmask) -> Self {
        Self { reference: v.stencilref as u8, read_mask: v.stencilmask as u8, write_mask: v.stencilwritemask as u8 }
    }
}

/// Host limitation that prevents exact reproduction of the guest depth/stencil state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthStencilIssue {
    /// Back faces use a different stencil reference or masks than front faces; WebGPU has one
    /// reference and one pair of masks for both faces (the front values are used).
    SeparateBackFaceRefMask,
}

/// Depth/stencil state for the draw, WebGPU `GPUDepthStencilState` + stencil reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthStencilState {
    /// Whether the depth/stencil attachment is used at all.
    pub enabled: bool,
    pub depth_format: DepthRenderTargetFormat,
    /// `Always` when depth testing is disabled.
    pub depth_compare: CompareFunction,
    pub depth_write_enabled: bool,
    pub stencil_enabled: bool,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub front_ref_mask: StencilRefMask,
    pub back_ref_mask: StencilRefMask,
    /// Values for WebGPU: `stencilReadMask`, `stencilWriteMask` and `setStencilReference`.
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,
    pub stencil_reference: u8,
    pub issues: Vec<DepthStencilIssue>,
}

impl DepthStencilState {
    /// Builds the state from a normalized depth control. `back_faces_visible` should be false for
    /// non-polygonal primitives or when back faces are culled (back state is then irrelevant).
    pub fn from_registers(
        regs: &(impl RegisterSource + ?Sized),
        normalized: RbDepthcontrol,
        back_faces_visible: bool,
    ) -> Self {
        let front_ref_mask = StencilRefMask::from(RbStencilrefmask::read(regs));
        let bf_raw = RbStencilrefmask::decode(regs.reg(r::RB_STENCILREFMASK_BF));
        let mut s = Self {
            enabled: normalized.z_enable || normalized.stencil_enable,
            depth_format: RbDepthInfo::read(regs).depth_format,
            depth_compare: if normalized.z_enable { normalized.zfunc } else { CompareFunction::Always },
            depth_write_enabled: normalized.z_enable && normalized.z_write_enable,
            stencil_enabled: normalized.stencil_enable,
            stencil_front: StencilFaceState::DISABLED,
            stencil_back: StencilFaceState::DISABLED,
            front_ref_mask,
            back_ref_mask: front_ref_mask,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            stencil_reference: 0,
            issues: Vec::new(),
        };
        if normalized.stencil_enable {
            s.stencil_front = StencilFaceState {
                compare: normalized.stencilfunc,
                fail_op: normalized.stencilfail,
                depth_fail_op: normalized.stencilzfail,
                pass_op: normalized.stencilzpass,
            };
            if normalized.backface_enable {
                s.stencil_back = StencilFaceState {
                    compare: normalized.stencilfunc_bf,
                    fail_op: normalized.stencilfail_bf,
                    depth_fail_op: normalized.stencilzfail_bf,
                    pass_op: normalized.stencilzpass_bf,
                };
                s.back_ref_mask = StencilRefMask::from(bf_raw);
            } else {
                s.stencil_back = s.stencil_front;
            }
            s.stencil_read_mask = front_ref_mask.read_mask;
            s.stencil_write_mask = front_ref_mask.write_mask;
            s.stencil_reference = front_ref_mask.reference;
            if back_faces_visible && s.back_ref_mask != front_ref_mask {
                s.issues.push(DepthStencilIssue::SeparateBackFaceRefMask);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regs_with(depthcontrol: RbDepthcontrol, mode: u32) -> Vec<u32> {
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_DEPTHCONTROL as usize] = depthcontrol.encode();
        regs[r::RB_MODECONTROL as usize] = mode;
        regs
    }

    #[test]
    fn normalization() {
        let dc = RbDepthcontrol {
            z_enable: true,
            z_write_enable: true,
            zfunc: CompareFunction::LessEqual,
            ..Default::default()
        };
        assert_eq!(normalized_depth_control(&regs_with(dc, 4), false), dc);
        assert_eq!(normalized_depth_control(&regs_with(dc, 5), false), dc);
        assert_eq!(normalized_depth_control(&regs_with(dc, 6), false), RbDepthcontrol::default());
        let ro_always = RbDepthcontrol { z_write_enable: false, zfunc: CompareFunction::Always, ..dc };
        assert!(!normalized_depth_control(&regs_with(ro_always, 4), false).z_enable);
        // VIZ survey: test-only, depth disabled without hi-Z.
        let mut regs = regs_with(RbDepthcontrol { stencil_enable: true, ..dc }, 4);
        regs[r::PA_SC_VIZ_QUERY as usize] = 1 | (1 << 7);
        let n = normalized_depth_control(&regs, true);
        assert!(!n.z_write_enable && !n.stencil_enable && !n.z_enable);
        regs[r::RB_HIZCONTROL as usize] = 2;
        assert!(normalized_depth_control(&regs, true).z_enable);
    }

    #[test]
    fn stencil_state() {
        let dc = RbDepthcontrol {
            stencil_enable: true,
            z_enable: true,
            z_write_enable: false,
            zfunc: CompareFunction::Greater,
            backface_enable: true,
            stencilfunc: CompareFunction::Equal,
            stencilfail: StencilOp::Keep,
            stencilzpass: StencilOp::IncrementWrap,
            stencilzfail: StencilOp::Zero,
            stencilfunc_bf: CompareFunction::Always,
            stencilfail_bf: StencilOp::Invert,
            stencilzpass_bf: StencilOp::DecrementWrap,
            stencilzfail_bf: StencilOp::Replace,
        };
        let mut regs = regs_with(dc, 4);
        regs[r::RB_STENCILREFMASK as usize] = 0x00FF_F005;
        regs[r::RB_STENCILREFMASK_BF as usize] = 0x00FF_F005;
        let s = DepthStencilState::from_registers(&regs, normalized_depth_control(&regs, false), true);
        assert!(s.enabled && s.stencil_enabled && !s.depth_write_enabled);
        assert_eq!(s.depth_compare.webgpu_name(), "greater");
        assert_eq!(s.stencil_front.pass_op, StencilOp::IncrementWrap);
        assert_eq!(s.stencil_front.depth_fail_op, StencilOp::Zero);
        assert_eq!(s.stencil_back.fail_op.webgpu_name(), "invert");
        assert_eq!(s.stencil_back.depth_fail_op, StencilOp::Replace);
        assert_eq!((s.stencil_reference, s.stencil_read_mask, s.stencil_write_mask), (5, 0xF0, 0xFF));
        assert!(s.issues.is_empty());

        regs[r::RB_STENCILREFMASK_BF as usize] = 0x00FF_F006;
        let s = DepthStencilState::from_registers(&regs, normalized_depth_control(&regs, false), true);
        assert_eq!(s.issues, vec![DepthStencilIssue::SeparateBackFaceRefMask]);
        assert_eq!(s.back_ref_mask.reference, 6);
        let s = DepthStencilState::from_registers(&regs, normalized_depth_control(&regs, false), false);
        assert!(s.issues.is_empty());

        // Without backface_enable the back face mirrors the front.
        let regs = regs_with(RbDepthcontrol { backface_enable: false, ..dc }, 4);
        let s = DepthStencilState::from_registers(&regs, normalized_depth_control(&regs, false), true);
        assert_eq!(s.stencil_back, s.stencil_front);

        // Disabled depth/stencil.
        let regs = regs_with(RbDepthcontrol::default(), 4);
        let s = DepthStencilState::from_registers(&regs, normalized_depth_control(&regs, false), true);
        assert!(!s.enabled);
        assert_eq!(s.depth_compare, CompareFunction::Always);
        assert_eq!(s.stencil_front, StencilFaceState::DISABLED);
    }

    #[test]
    fn compare_eval() {
        assert!(CompareFunction::Less.passes(0.2, 0.5));
        assert!(!CompareFunction::Greater.passes(0.2, 0.5));
        assert!(CompareFunction::GreaterEqual.passes(0.5, 0.5));
        assert!(!CompareFunction::Never.passes(0.0, 0.0));
    }
}
