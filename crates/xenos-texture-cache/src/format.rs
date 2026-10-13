//! xenos-texture host formats as webgpu.h formats.

use gpu_webgpu::TextureFormat;
use xenos_texture::HostFormat;

pub(crate) fn webgpu_format(format: HostFormat) -> TextureFormat {
    use HostFormat as H;
    use TextureFormat as F;
    match format {
        H::R8Unorm => F::R8Unorm,
        H::R8Snorm => F::R8Snorm,
        H::Rg8Unorm => F::RG8Unorm,
        H::Rg8Snorm => F::RG8Snorm,
        H::Rgba8Unorm => F::RGBA8Unorm,
        H::Rgba8UnormSrgb => F::RGBA8UnormSrgb,
        H::Rgba8Snorm => F::RGBA8Snorm,
        H::Bgra8Unorm => F::BGRA8Unorm,
        H::Bgra8UnormSrgb => F::BGRA8UnormSrgb,
        H::R16Unorm => F::R16Unorm,
        H::R16Snorm => F::R16Snorm,
        H::Rg16Unorm => F::RG16Unorm,
        H::Rg16Snorm => F::RG16Snorm,
        H::Rgba16Unorm => F::RGBA16Unorm,
        H::Rgba16Snorm => F::RGBA16Snorm,
        H::R16Float => F::R16Float,
        H::Rg16Float => F::RG16Float,
        H::Rgba16Float => F::RGBA16Float,
        H::R32Uint => F::R32Uint,
        H::R32Float => F::R32Float,
        H::Rg32Float => F::RG32Float,
        H::Rgba32Float => F::RGBA32Float,
        H::Rgb10a2Unorm => F::RGB10A2Unorm,
        H::Rg11b10Ufloat => F::RG11B10Ufloat,
        H::Depth32Float => F::Depth32Float,
        H::Depth24PlusStencil8 => F::Depth24PlusStencil8,
        H::Bc1RgbaUnorm => F::BC1RGBAUnorm,
        H::Bc1RgbaUnormSrgb => F::BC1RGBAUnormSrgb,
        H::Bc2RgbaUnorm => F::BC2RGBAUnorm,
        H::Bc2RgbaUnormSrgb => F::BC2RGBAUnormSrgb,
        H::Bc3RgbaUnorm => F::BC3RGBAUnorm,
        H::Bc3RgbaUnormSrgb => F::BC3RGBAUnormSrgb,
        H::Bc4RUnorm => F::BC4RUnorm,
        H::Bc4RSnorm => F::BC4RSnorm,
        H::Bc5RgUnorm => F::BC5RGUnorm,
        H::Bc5RgSnorm => F::BC5RGSnorm,
    }
}

/// Whether sampling the host format through a `float` binding can use filtering samplers.
pub(crate) fn is_filterable(format: HostFormat, float32_filterable: bool) -> bool {
    match format {
        HostFormat::R32Float | HostFormat::Rg32Float | HostFormat::Rgba32Float => float32_filterable,
        other => other.is_filterable(),
    }
}

/// Whether the host format can be sampled as `texture_*<f32>` at all (no integer, depth or
/// stencil formats).
pub(crate) fn is_float_sampleable(format: HostFormat) -> bool {
    !matches!(format, HostFormat::R32Uint | HostFormat::Depth32Float | HostFormat::Depth24PlusStencil8)
}
