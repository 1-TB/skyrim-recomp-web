//! Color output state: write masks, blending (RB_BLENDCONTROL0..3, RB_BLEND_RED..ALPHA), alpha test
//! and alpha to coverage (RB_COLORCONTROL, RB_ALPHA_REF), mapped to WebGPU-style enums.

use crate::RegisterSource;
use crate::bitfields::*;
use crate::registers as r;
use crate::viewport::is_viz_survey_draw;
use crate::xenos::*;

/// WebGPU `GPUBlendFactor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostBlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

impl HostBlendFactor {
    /// WebGPU enum string.
    pub fn webgpu_name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::One => "one",
            Self::Src => "src",
            Self::OneMinusSrc => "one-minus-src",
            Self::SrcAlpha => "src-alpha",
            Self::OneMinusSrcAlpha => "one-minus-src-alpha",
            Self::Dst => "dst",
            Self::OneMinusDst => "one-minus-dst",
            Self::DstAlpha => "dst-alpha",
            Self::OneMinusDstAlpha => "one-minus-dst-alpha",
            Self::SrcAlphaSaturated => "src-alpha-saturated",
            Self::Constant => "constant",
            Self::OneMinusConstant => "one-minus-constant",
        }
    }
}

/// WebGPU `GPUBlendOperation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostBlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl HostBlendOperation {
    /// WebGPU enum string.
    pub fn webgpu_name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::ReverseSubtract => "reverse-subtract",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

/// WebGPU `GPUBlendComponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    pub operation: HostBlendOperation,
    pub src_factor: HostBlendFactor,
    pub dst_factor: HostBlendFactor,
}

impl BlendComponent {
    /// Source replaces destination.
    pub const REPLACE: Self = Self {
        operation: HostBlendOperation::Add,
        src_factor: HostBlendFactor::One,
        dst_factor: HostBlendFactor::Zero,
    };
}

/// State of one color target (WebGPU `GPUColorTargetState` without the format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorTargetState {
    pub format: ColorRenderTargetFormat,
    /// `None` when blending is a plain replace.
    pub blend: Option<(BlendComponent, BlendComponent)>,
    /// RGBA write mask, bit 0 red .. bit 3 alpha (WebGPU `GPUColorWrite` uses the same bits).
    pub write_mask: u32,
}

/// Alpha test (performed in the pixel shader on hosts without fixed-function alpha test).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaTest {
    /// Fragment passes if `alpha <func> reference`.
    pub func: CompareFunction,
    pub reference: f32,
}

/// Alpha to coverage configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlphaToCoverage {
    /// Dither threshold offsets for the pixels (0,0), (0,1), (1,0), (1,1) of each quad.
    pub offsets: [u32; 4],
}

/// Guest blending that can't be represented exactly on a WebGPU host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendIssue {
    /// A blend factor value with no known meaning (treated as zero).
    UnknownFactor { rt: u32, raw: u32 },
    /// A blend operation value with no known meaning (treated as add).
    UnknownOperation { rt: u32, raw: u32 },
    /// Both constant-color and constant-alpha factors are used in color slots with a blend color whose
    /// RGB differs from its alpha; WebGPU has a single blend constant, so constant alpha in color slots
    /// is approximated with the constant color.
    ConflictingBlendConstant,
}

/// Output merger state for the draw.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputMergerState {
    /// Color targets that are written (per the normalized color mask); `None` for unused slots.
    pub targets: [Option<ColorTargetState>; 4],
    /// Value for WebGPU `setBlendConstant`.
    pub blend_constant: [f32; 4],
    /// Guest RB_BLEND_RED..ALPHA.
    pub guest_blend_color: [f32; 4],
    pub alpha_test: Option<AlphaTest>,
    pub alpha_to_coverage: Option<AlphaToCoverage>,
    pub issues: Vec<BlendIssue>,
}

/// Color write mask restricted to render targets the pixel shader writes, with components absent
/// from the formats forced on for targets that are written at all (GetNormalizedColorMask). Returns 4
/// bits per render target.
pub fn normalized_color_mask(
    regs: &(impl RegisterSource + ?Sized),
    pixel_shader_writes_color_targets: u32,
    viz_queries_emulated: bool,
) -> u32 {
    if RbModecontrol::read(regs).edram_mode != EdramMode::ColorDepth || is_viz_survey_draw(regs, viz_queries_emulated) {
        return 0;
    }
    let mask = RbColorMask::read(regs);
    let mut normalized = 0;
    for rt in 0..MAX_COLOR_RENDER_TARGETS as usize {
        if pixel_shader_writes_color_targets & (1 << rt) == 0 {
            continue;
        }
        let components = RbColorInfo::read_rt(regs, rt).color_format.component_count();
        let format_mask = (1u32 << components) - 1;
        let rt_mask = mask.rt(rt) & format_mask;
        if rt_mask == 0 {
            continue;
        }
        normalized |= (rt_mask | (0xF & !format_mask)) << (4 * rt);
    }
    normalized
}

fn map_factor(f: BlendFactor, alpha_slot: bool) -> Option<HostBlendFactor> {
    use HostBlendFactor as H;
    Some(match (f, alpha_slot) {
        (BlendFactor::Zero, _) => H::Zero,
        (BlendFactor::One, _) => H::One,
        (BlendFactor::SrcColor, false) => H::Src,
        (BlendFactor::OneMinusSrcColor, false) => H::OneMinusSrc,
        (BlendFactor::SrcColor, true) | (BlendFactor::SrcAlpha, _) => H::SrcAlpha,
        (BlendFactor::OneMinusSrcColor, true) | (BlendFactor::OneMinusSrcAlpha, _) => H::OneMinusSrcAlpha,
        (BlendFactor::DstColor, false) => H::Dst,
        (BlendFactor::OneMinusDstColor, false) => H::OneMinusDst,
        (BlendFactor::DstColor, true) | (BlendFactor::DstAlpha, _) => H::DstAlpha,
        (BlendFactor::OneMinusDstColor, true) | (BlendFactor::OneMinusDstAlpha, _) => H::OneMinusDstAlpha,
        (BlendFactor::ConstantColor | BlendFactor::ConstantAlpha, _) => H::Constant,
        (BlendFactor::OneMinusConstantColor | BlendFactor::OneMinusConstantAlpha, _) => H::OneMinusConstant,
        (BlendFactor::SrcAlphaSaturate, _) => H::SrcAlphaSaturated,
        (BlendFactor::Unknown(_), _) => return None,
    })
}

fn map_op(op: BlendOp) -> Option<HostBlendOperation> {
    Some(match op {
        BlendOp::Add => HostBlendOperation::Add,
        BlendOp::Subtract => HostBlendOperation::Subtract,
        BlendOp::Min => HostBlendOperation::Min,
        BlendOp::Max => HostBlendOperation::Max,
        BlendOp::RevSubtract => HostBlendOperation::ReverseSubtract,
        BlendOp::Unknown(_) => return None,
    })
}

/// Maps one blend component. Min/max ignore the factors on the guest; WebGPU requires them to be
/// `one`.
fn map_component(
    rt: u32,
    src: BlendFactor,
    op: BlendOp,
    dst: BlendFactor,
    alpha_slot: bool,
    issues: &mut Vec<BlendIssue>,
) -> BlendComponent {
    let mut factor = |f: BlendFactor| {
        map_factor(f, alpha_slot).unwrap_or_else(|| {
            issues.push(BlendIssue::UnknownFactor { rt, raw: f.raw() });
            HostBlendFactor::Zero
        })
    };
    let (src_factor, dst_factor) = (factor(src), factor(dst));
    let operation = map_op(op).unwrap_or_else(|| {
        issues.push(BlendIssue::UnknownOperation { rt, raw: op.raw() });
        HostBlendOperation::Add
    });
    if matches!(operation, HostBlendOperation::Min | HostBlendOperation::Max) {
        return BlendComponent { operation, src_factor: HostBlendFactor::One, dst_factor: HostBlendFactor::One };
    }
    BlendComponent { operation, src_factor, dst_factor }
}

impl OutputMergerState {
    /// Builds the output state from a normalized color mask (see [`normalized_color_mask`]).
    pub fn from_registers(regs: &(impl RegisterSource + ?Sized), normalized_color_mask: u32) -> Self {
        let guest_blend_color = [
            regs.reg_f32(r::RB_BLEND_RED),
            regs.reg_f32(r::RB_BLEND_GREEN),
            regs.reg_f32(r::RB_BLEND_BLUE),
            regs.reg_f32(r::RB_BLEND_ALPHA),
        ];
        let mut issues = Vec::new();
        let mut targets = [None; 4];
        let (mut uses_constant_color, mut uses_constant_alpha_in_color) = (false, false);
        for (rt, target) in targets.iter_mut().enumerate() {
            let write_mask = (normalized_color_mask >> (4 * rt)) & 0xF;
            if write_mask == 0 {
                continue;
            }
            let bc = RbBlendcontrol::read_rt(regs, rt);
            let rti = rt as u32;
            let color =
                map_component(rti, bc.color_srcblend, bc.color_comb_fcn, bc.color_destblend, false, &mut issues);
            let alpha = map_component(rti, bc.alpha_srcblend, bc.alpha_comb_fcn, bc.alpha_destblend, true, &mut issues);
            let blend =
                (color != BlendComponent::REPLACE || alpha != BlendComponent::REPLACE).then_some((color, alpha));
            if blend.is_some() && !matches!(color.operation, HostBlendOperation::Min | HostBlendOperation::Max) {
                for f in [bc.color_srcblend, bc.color_destblend] {
                    match f {
                        BlendFactor::ConstantColor | BlendFactor::OneMinusConstantColor => uses_constant_color = true,
                        BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha => {
                            uses_constant_alpha_in_color = true
                        }
                        _ => {}
                    }
                }
            }
            *target = Some(ColorTargetState { format: RbColorInfo::read_rt(regs, rt).color_format, blend, write_mask });
        }
        let [cr, cg, cb, ca] = guest_blend_color;
        let blend_constant = if uses_constant_alpha_in_color && !uses_constant_color {
            [ca; 4]
        } else {
            if uses_constant_alpha_in_color && (cr != ca || cg != ca || cb != ca) {
                issues.push(BlendIssue::ConflictingBlendConstant);
            }
            guest_blend_color
        };

        let cc = RbColorcontrol::read(regs);
        let alpha_test = (cc.alpha_test_enable && cc.alpha_func != CompareFunction::Always)
            .then(|| AlphaTest { func: cc.alpha_func, reference: regs.reg_f32(r::RB_ALPHA_REF) });
        let alpha_to_coverage = cc.alpha_to_mask_enable.then_some(AlphaToCoverage {
            offsets: [
                cc.alpha_to_mask_offset0,
                cc.alpha_to_mask_offset1,
                cc.alpha_to_mask_offset2,
                cc.alpha_to_mask_offset3,
            ],
        });
        Self { targets, blend_constant, guest_blend_color, alpha_test, alpha_to_coverage, issues }
    }

    /// Whether fragment coverage depends on alpha (DoesCoverageDependOnAlpha).
    pub fn coverage_depends_on_alpha(&self) -> bool {
        self.alpha_test.is_some() || self.alpha_to_coverage.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use HostBlendFactor as H;

    fn base_regs() -> Vec<u32> {
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_MODECONTROL as usize] = 4;
        regs[r::RB_COLOR_MASK as usize] = 0xFFFF;
        regs
    }

    fn blend(
        src: BlendFactor,
        op: BlendOp,
        dst: BlendFactor,
        asrc: BlendFactor,
        aop: BlendOp,
        adst: BlendFactor,
    ) -> u32 {
        RbBlendcontrol {
            color_srcblend: src,
            color_comb_fcn: op,
            color_destblend: dst,
            alpha_srcblend: asrc,
            alpha_comb_fcn: aop,
            alpha_destblend: adst,
        }
        .encode()
    }

    #[test]
    fn color_mask_normalization() {
        let mut regs = base_regs();
        regs[r::RB_COLOR_MASK as usize] = 0x0FC1;
        regs[r::RB_COLOR_INFO as usize] = (ColorRenderTargetFormat::K32Float.raw()) << 16;
        regs[r::RB_COLOR1_INFO as usize] = (ColorRenderTargetFormat::K1616.raw()) << 16;
        regs[r::RB_COLOR2_INFO as usize] = (ColorRenderTargetFormat::K8888.raw()) << 16;
        // RT0: R written of a 1-component format -> all forced on. RT1: only BA written of RG -> skipped.
        assert_eq!(normalized_color_mask(&regs, 0b1111, false), 0x0F0F);
        // RT2 not written by the shader.
        assert_eq!(normalized_color_mask(&regs, 0b0011, false), 0x000F);
        regs[r::RB_MODECONTROL as usize] = 5;
        assert_eq!(normalized_color_mask(&regs, 0b1111, false), 0);
    }

    #[test]
    fn alpha_blend_mapping() {
        let mut regs = base_regs();
        regs[r::RB_BLENDCONTROL0 as usize] = blend(
            BlendFactor::SrcAlpha,
            BlendOp::Add,
            BlendFactor::OneMinusSrcAlpha,
            BlendFactor::SrcColor,
            BlendOp::RevSubtract,
            BlendFactor::OneMinusDstColor,
        );
        regs[r::RB_BLENDCONTROL1 as usize] =
            blend(BlendFactor::One, BlendOp::Add, BlendFactor::Zero, BlendFactor::One, BlendOp::Add, BlendFactor::Zero);
        let s = OutputMergerState::from_registers(&regs, 0x00FF);
        let (c, a) = s.targets[0].unwrap().blend.unwrap();
        assert_eq!(
            (c.src_factor, c.operation, c.dst_factor),
            (H::SrcAlpha, HostBlendOperation::Add, H::OneMinusSrcAlpha)
        );
        assert_eq!(
            (a.src_factor, a.operation, a.dst_factor),
            (H::SrcAlpha, HostBlendOperation::ReverseSubtract, H::OneMinusDstAlpha)
        );
        assert_eq!(c.dst_factor.webgpu_name(), "one-minus-src-alpha");
        assert_eq!(a.operation.webgpu_name(), "reverse-subtract");
        assert_eq!(s.targets[1].unwrap().blend, None);
        assert_eq!(s.targets[1].unwrap().write_mask, 0xF);
        assert!(s.targets[2].is_none());
        assert!(s.issues.is_empty());
    }

    #[test]
    fn min_max_and_unknown() {
        let mut regs = base_regs();
        regs[r::RB_BLENDCONTROL0 as usize] = blend(
            BlendFactor::SrcColor,
            BlendOp::Max,
            BlendFactor::DstColor,
            BlendFactor::Unknown(2),
            BlendOp::Unknown(6),
            BlendFactor::One,
        );
        let s = OutputMergerState::from_registers(&regs, 0xF);
        let (c, a) = s.targets[0].unwrap().blend.unwrap();
        assert_eq!((c.src_factor, c.operation, c.dst_factor), (H::One, HostBlendOperation::Max, H::One));
        assert_eq!((a.src_factor, a.operation, a.dst_factor), (H::Zero, HostBlendOperation::Add, H::One));
        assert_eq!(
            s.issues,
            vec![BlendIssue::UnknownFactor { rt: 0, raw: 2 }, BlendIssue::UnknownOperation { rt: 0, raw: 6 }]
        );
    }

    #[test]
    fn blend_constants() {
        let mut regs = base_regs();
        for (i, v) in [0.25f32, 0.5, 0.75, 0.125].into_iter().enumerate() {
            regs[r::RB_BLEND_RED as usize + i] = v.to_bits();
        }
        let ca = blend(
            BlendFactor::ConstantAlpha,
            BlendOp::Add,
            BlendFactor::OneMinusConstantAlpha,
            BlendFactor::ConstantColor,
            BlendOp::Add,
            BlendFactor::Zero,
        );
        regs[r::RB_BLENDCONTROL0 as usize] = ca;
        let s = OutputMergerState::from_registers(&regs, 0xF);
        let (c, a) = s.targets[0].unwrap().blend.unwrap();
        assert_eq!((c.src_factor, c.dst_factor, a.src_factor), (H::Constant, H::OneMinusConstant, H::Constant));
        assert_eq!(s.blend_constant, [0.125; 4]);
        assert!(s.issues.is_empty());

        regs[r::RB_BLENDCONTROL1 as usize] = blend(
            BlendFactor::ConstantColor,
            BlendOp::Add,
            BlendFactor::One,
            BlendFactor::One,
            BlendOp::Add,
            BlendFactor::Zero,
        );
        let s = OutputMergerState::from_registers(&regs, 0xFF);
        assert_eq!(s.blend_constant, [0.25, 0.5, 0.75, 0.125]);
        assert_eq!(s.issues, vec![BlendIssue::ConflictingBlendConstant]);
        // Not an issue if the RGB channels equal alpha.
        for i in 0..4 {
            regs[r::RB_BLEND_RED as usize + i] = 0.5f32.to_bits();
        }
        assert!(OutputMergerState::from_registers(&regs, 0xFF).issues.is_empty());
    }

    #[test]
    fn alpha_test_and_coverage() {
        let mut regs = base_regs();
        regs[r::RB_COLORCONTROL as usize] =
            RbColorcontrol { alpha_func: CompareFunction::GreaterEqual, alpha_test_enable: true, ..Default::default() }
                .encode();
        regs[r::RB_ALPHA_REF as usize] = 0.5f32.to_bits();
        let s = OutputMergerState::from_registers(&regs, 0xF);
        assert_eq!(s.alpha_test, Some(AlphaTest { func: CompareFunction::GreaterEqual, reference: 0.5 }));
        assert!(s.alpha_to_coverage.is_none() && s.coverage_depends_on_alpha());
        regs[r::RB_COLORCONTROL as usize] = RbColorcontrol {
            alpha_func: CompareFunction::Always,
            alpha_test_enable: true,
            alpha_to_mask_enable: true,
            alpha_to_mask_offset0: 3,
            alpha_to_mask_offset1: 1,
            alpha_to_mask_offset2: 0,
            alpha_to_mask_offset3: 2,
        }
        .encode();
        let s = OutputMergerState::from_registers(&regs, 0xF);
        assert_eq!(s.alpha_test, None);
        assert_eq!(s.alpha_to_coverage, Some(AlphaToCoverage { offsets: [3, 1, 0, 2] }));
    }
}
