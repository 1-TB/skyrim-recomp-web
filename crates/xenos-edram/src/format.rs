//! Host formats standing in for EDRAM formats, and what the draw pipeline must do with them.
//!
//! Every host render target keeps values the pixel shader can write and blend directly; its EDRAM
//! bit pattern is produced when contents leave the host texture (ownership transfers, resolves).
//!
//! | EDRAM format                | Host format            | Notes                                               |
//! |-----------------------------|------------------------|-----------------------------------------------------|
//! | k_8_8_8_8                   | rgba8unorm             | exact                                               |
//! | k_8_8_8_8_GAMMA             | rgba16float (linear)   | blended in linear space; PWL-encoded on the way out |
//! | k_2_10_10_10 (+AS_10_10_10_10) | rgb10a2unorm        | exact                                               |
//! | k_2_10_10_10_FLOAT (7e3)    | rgba16float            | 7e3 values are exact in f16; more precise between   |
//! | k_16_16 / k_16_16_16_16     | rg16float / rgba16float | real -32..32 values; f16 keeps 11 of the 16 bits   |
//! | k_16_16_FLOAT / _16_16_16_16_FLOAT | rg16float / rgba16float | exact                                  |
//! | k_32_FLOAT / k_32_32_FLOAT  | r32float / rg32float   | exact; not blendable in core WebGPU                 |
//! | D24S8                       | depth24plus-stencil8 (or depth32float-stencil8, see `EdramOptions`) | guest z 0..1 |
//! | D24FS8 (20e4)               | depth32float-stencil8 when available, else depth24plus-stencil8 | |

use gpu_webgpu::TextureFormat;
use xenos_state::xenos::{ColorRenderTargetFormat, DepthRenderTargetFormat};

/// Host texture format for an EDRAM color format (aliases collapsed to the storage format).
pub fn host_color_format(format: ColorRenderTargetFormat) -> Option<TextureFormat> {
    use ColorRenderTargetFormat as C;
    Some(match format.storage_format() {
        C::K8888 => TextureFormat::RGBA8Unorm,
        C::K8888Gamma | C::K2101010Float | C::K16161616 | C::K16161616Float => TextureFormat::RGBA16Float,
        C::K2101010 => TextureFormat::RGB10A2Unorm,
        C::K1616 | C::K1616Float => TextureFormat::RG16Float,
        C::K32Float => TextureFormat::R32Float,
        C::K3232Float => TextureFormat::RG32Float,
        _ => return None,
    })
}

/// Host texture format for an EDRAM depth format, given whether depth32float-stencil8 is
/// available and whether D24S8 should use it too (see [`EdramOptions::d24s8_float_host`]).
pub fn host_depth_format(format: DepthRenderTargetFormat, depth32_stencil8: bool, d24s8_float: bool) -> TextureFormat {
    match format {
        DepthRenderTargetFormat::D24FS8 if depth32_stencil8 => TextureFormat::Depth32FloatStencil8,
        DepthRenderTargetFormat::D24S8 if depth32_stencil8 && d24s8_float => TextureFormat::Depth32FloatStencil8,
        _ => TextureFormat::Depth24PlusStencil8,
    }
}

/// Host choices of an [`crate::EdramCache`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdramOptions {
    /// Host D24S8 in depth32float-stencil8 (when available) instead of depth24plus-stencil8. The
    /// default keeps the unorm24 quantization of depth tests; where the implementation's unorm24
    /// loads aren't exact (lavapipe), depth leaving a depth24plus target (transfers, resolves) may
    /// be off by one unit in the last place. With float storage every EDRAM depth value survives.
    pub d24s8_float_host: bool,
}

/// How the draw's pixel shader output for a color target maps to the host target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTargetInfo {
    /// Format as written in RB_COLOR_INFO.
    pub guest_format: ColorRenderTargetFormat,
    pub host_format: TextureFormat,
    /// Multiply the shader output by this (2^RB_COLOR_INFO::COLOR_EXP_BIAS) before writing.
    pub exp_bias_factor: f32,
    /// Clamp the (biased) output to this range per component to reproduce the EDRAM format's range
    /// (unorm hosts clamp by themselves; float hosts don't). `None`: no clamping.
    pub output_range: Option<([f32; 4], [f32; 4])>,
    /// Whether the host format can blend (r32float / rg32float can't in core WebGPU; the Xenos
    /// doesn't blend 32-bit float targets either).
    pub blendable: bool,
}

impl ColorTargetInfo {
    pub fn new(guest_format: ColorRenderTargetFormat, exp_bias: i32) -> Option<Self> {
        use ColorRenderTargetFormat as C;
        let host_format = host_color_format(guest_format)?;
        let output_range = match guest_format.storage_format() {
            C::K8888 | C::K8888Gamma | C::K2101010 => Some(([0.0; 4], [1.0; 4])),
            C::K2101010Float => Some(([0.0; 4], [31.875, 31.875, 31.875, 1.0])),
            C::K1616 | C::K16161616 => Some(([-32.0; 4], [32.0; 4])),
            _ => None,
        };
        Some(Self {
            guest_format,
            host_format,
            exp_bias_factor: (exp_bias.clamp(-32, 31) as f32).exp2(),
            output_range,
            blendable: !matches!(guest_format, C::K32Float | C::K3232Float),
        })
    }
}

/// How depth maps to the host depth/stencil target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthTargetInfo {
    pub guest_format: DepthRenderTargetFormat,
    pub host_format: TextureFormat,
    /// Guest depth is 20e4 floating point; the host keeps the value as f32 when `host_format` is
    /// depth32float-stencil8 (converted with round-to-nearest-even on the way out).
    pub float24: bool,
}

/// Code identifying an EDRAM format in the WGSL helpers: color formats use their raw value, depth
/// formats `DEPTH_FORMAT_CODE_BASE + raw`.
pub(crate) const DEPTH_FORMAT_CODE_BASE: u32 = 16;

/// Xenos piecewise-linear gamma (8-bit code) to linear, exactly like the resolve hardware.
pub fn pwl_gamma_to_linear(code: u8) -> f32 {
    pwl_code_to_linear10(code) as f32 / 1023.0
}

pub(crate) fn pwl_code_to_linear10(code: u8) -> u32 {
    let b = code as u32;
    match b {
        0..64 => b,
        64..96 => 2 * b - 64,
        96..192 => {
            let l = 4 * b - 256;
            l + (l >> 8)
        }
        _ => {
            let l = 8 * b - 1024;
            l + (l >> 7)
        }
    }
}

/// Linear (0..1) to the Xenos piecewise-linear gamma 8-bit code.
pub fn linear_to_pwl_gamma(value: f32) -> u8 {
    let k = (value.clamp(0.0, 1.0) * 1023.0).round() as u32;
    (match k {
        0..64 => k,
        64..128 => (k >> 1) + 32,
        128..512 => (k >> 2) + 64,
        _ => (k >> 3) + 128,
    }) as u8
}

/// WGSL for `fn xe_linear_to_pwl_gamma(v: f32) -> f32`, for shaders that need gamma-encoded values
/// (EDRAM targets don't: gamma targets are stored linear).
pub const WGSL_LINEAR_TO_PWL_GAMMA: &str = r#"
fn xe_linear_to_pwl_gamma(v: f32) -> f32 {
    let k = u32(round(clamp(v, 0.0, 1.0) * 1023.0));
    var b: u32;
    if (k < 64u) { b = k; } else if (k < 128u) { b = (k >> 1u) + 32u; }
    else if (k < 512u) { b = (k >> 2u) + 64u; } else { b = (k >> 3u) + 128u; }
    return f32(b) / 255.0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pwl_round_trip() {
        for code in 0..=255u8 {
            assert_eq!(linear_to_pwl_gamma(pwl_gamma_to_linear(code)), code, "code {code}");
        }
        assert_eq!(pwl_gamma_to_linear(255), 1.0);
        assert_eq!(linear_to_pwl_gamma(0.5), 192);
    }

    #[test]
    fn host_formats() {
        use ColorRenderTargetFormat as C;
        assert_eq!(host_color_format(C::K2101010As10101010), Some(TextureFormat::RGB10A2Unorm));
        assert_eq!(host_color_format(C::K2101010FloatAs16161616), Some(TextureFormat::RGBA16Float));
        assert_eq!(host_color_format(C::Unknown(9)), None);
        let info = ColorTargetInfo::new(C::K3232Float, -2).unwrap();
        assert!(!info.blendable);
        assert_eq!(info.exp_bias_factor, 0.25);
        assert_eq!(host_depth_format(DepthRenderTargetFormat::D24FS8, false, true), TextureFormat::Depth24PlusStencil8);
        assert_eq!(host_depth_format(DepthRenderTargetFormat::D24S8, true, false), TextureFormat::Depth24PlusStencil8);
    }
}
