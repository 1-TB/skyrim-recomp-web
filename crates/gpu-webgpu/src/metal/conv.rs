//! WebGPU enums → Metal enums.

use objc2_metal::*;

use crate::{
    AddressMode, BlendFactor, BlendOperation, ColorWriteMask, CompareFunction, CullMode, FilterMode, FrontFace,
    IndexFormat, LoadOp, MipmapFilterMode, PrimitiveTopology, StencilOperation, StoreOp, TextureFormat,
    TextureViewDimension, VertexFormat, VertexStepMode,
};

/// The Metal pixel format storing `format`. Apple GPUs have no 24-bit depth: `Depth24Plus*`
/// become 32-bit float depth (WebGPU allows it, the formats only promise "at least 24 bits").
pub(crate) fn pixel_format(format: TextureFormat) -> Option<MTLPixelFormat> {
    use MTLPixelFormat as M;
    use TextureFormat as F;
    Some(match format {
        F::R8Unorm => M::R8Unorm,
        F::R8Snorm => M::R8Snorm,
        F::R8Uint => M::R8Uint,
        F::R8Sint => M::R8Sint,
        F::R16Unorm => M::R16Unorm,
        F::R16Snorm => M::R16Snorm,
        F::R16Uint => M::R16Uint,
        F::R16Sint => M::R16Sint,
        F::R16Float => M::R16Float,
        F::RG8Unorm => M::RG8Unorm,
        F::RG8Snorm => M::RG8Snorm,
        F::RG8Uint => M::RG8Uint,
        F::RG8Sint => M::RG8Sint,
        F::R32Float => M::R32Float,
        F::R32Uint => M::R32Uint,
        F::R32Sint => M::R32Sint,
        F::RG16Unorm => M::RG16Unorm,
        F::RG16Snorm => M::RG16Snorm,
        F::RG16Uint => M::RG16Uint,
        F::RG16Sint => M::RG16Sint,
        F::RG16Float => M::RG16Float,
        F::RGBA8Unorm => M::RGBA8Unorm,
        F::RGBA8UnormSrgb => M::RGBA8Unorm_sRGB,
        F::RGBA8Snorm => M::RGBA8Snorm,
        F::RGBA8Uint => M::RGBA8Uint,
        F::RGBA8Sint => M::RGBA8Sint,
        F::BGRA8Unorm => M::BGRA8Unorm,
        F::BGRA8UnormSrgb => M::BGRA8Unorm_sRGB,
        F::RGB10A2Uint => M::RGB10A2Uint,
        F::RGB10A2Unorm => M::RGB10A2Unorm,
        F::RG11B10Ufloat => M::RG11B10Float,
        F::RGB9E5Ufloat => M::RGB9E5Float,
        F::RG32Float => M::RG32Float,
        F::RG32Uint => M::RG32Uint,
        F::RG32Sint => M::RG32Sint,
        F::RGBA16Unorm => M::RGBA16Unorm,
        F::RGBA16Snorm => M::RGBA16Snorm,
        F::RGBA16Uint => M::RGBA16Uint,
        F::RGBA16Sint => M::RGBA16Sint,
        F::RGBA16Float => M::RGBA16Float,
        F::RGBA32Float => M::RGBA32Float,
        F::RGBA32Uint => M::RGBA32Uint,
        F::RGBA32Sint => M::RGBA32Sint,
        F::Stencil8 => M::Stencil8,
        F::Depth16Unorm => M::Depth16Unorm,
        F::Depth24Plus | F::Depth32Float => M::Depth32Float,
        F::Depth24PlusStencil8 | F::Depth32FloatStencil8 => M::Depth32Float_Stencil8,
        F::BC1RGBAUnorm => M::BC1_RGBA,
        F::BC1RGBAUnormSrgb => M::BC1_RGBA_sRGB,
        F::BC2RGBAUnorm => M::BC2_RGBA,
        F::BC2RGBAUnormSrgb => M::BC2_RGBA_sRGB,
        F::BC3RGBAUnorm => M::BC3_RGBA,
        F::BC3RGBAUnormSrgb => M::BC3_RGBA_sRGB,
        F::BC4RUnorm => M::BC4_RUnorm,
        F::BC4RSnorm => M::BC4_RSnorm,
        F::BC5RGUnorm => M::BC5_RGUnorm,
        F::BC5RGSnorm => M::BC5_RGSnorm,
        F::BC6HRGBUfloat => M::BC6H_RGBUfloat,
        F::BC6HRGBFloat => M::BC6H_RGBFloat,
        F::BC7RGBAUnorm => M::BC7_RGBAUnorm,
        F::BC7RGBAUnormSrgb => M::BC7_RGBAUnorm_sRGB,
        F::ETC2RGB8Unorm => M::ETC2_RGB8,
        F::ETC2RGB8UnormSrgb => M::ETC2_RGB8_sRGB,
        F::ETC2RGB8A1Unorm => M::ETC2_RGB8A1,
        F::ETC2RGB8A1UnormSrgb => M::ETC2_RGB8A1_sRGB,
        F::ETC2RGBA8Unorm => M::EAC_RGBA8,
        F::ETC2RGBA8UnormSrgb => M::EAC_RGBA8_sRGB,
        F::EACR11Unorm => M::EAC_R11Unorm,
        F::EACR11Snorm => M::EAC_R11Snorm,
        F::EACRG11Unorm => M::EAC_RG11Unorm,
        F::EACRG11Snorm => M::EAC_RG11Snorm,
        F::ASTC4x4Unorm => M::ASTC_4x4_LDR,
        F::ASTC4x4UnormSrgb => M::ASTC_4x4_sRGB,
        F::ASTC5x4Unorm => M::ASTC_5x4_LDR,
        F::ASTC5x4UnormSrgb => M::ASTC_5x4_sRGB,
        F::ASTC5x5Unorm => M::ASTC_5x5_LDR,
        F::ASTC5x5UnormSrgb => M::ASTC_5x5_sRGB,
        F::ASTC6x5Unorm => M::ASTC_6x5_LDR,
        F::ASTC6x5UnormSrgb => M::ASTC_6x5_sRGB,
        F::ASTC6x6Unorm => M::ASTC_6x6_LDR,
        F::ASTC6x6UnormSrgb => M::ASTC_6x6_sRGB,
        F::ASTC8x5Unorm => M::ASTC_8x5_LDR,
        F::ASTC8x5UnormSrgb => M::ASTC_8x5_sRGB,
        F::ASTC8x6Unorm => M::ASTC_8x6_LDR,
        F::ASTC8x6UnormSrgb => M::ASTC_8x6_sRGB,
        F::ASTC8x8Unorm => M::ASTC_8x8_LDR,
        F::ASTC8x8UnormSrgb => M::ASTC_8x8_sRGB,
        F::ASTC10x5Unorm => M::ASTC_10x5_LDR,
        F::ASTC10x5UnormSrgb => M::ASTC_10x5_sRGB,
        F::ASTC10x6Unorm => M::ASTC_10x6_LDR,
        F::ASTC10x6UnormSrgb => M::ASTC_10x6_sRGB,
        F::ASTC10x8Unorm => M::ASTC_10x8_LDR,
        F::ASTC10x8UnormSrgb => M::ASTC_10x8_sRGB,
        F::ASTC10x10Unorm => M::ASTC_10x10_LDR,
        F::ASTC10x10UnormSrgb => M::ASTC_10x10_sRGB,
        F::ASTC12x10Unorm => M::ASTC_12x10_LDR,
        F::ASTC12x10UnormSrgb => M::ASTC_12x10_sRGB,
        F::ASTC12x12Unorm => M::ASTC_12x12_LDR,
        F::ASTC12x12UnormSrgb => M::ASTC_12x12_sRGB,
        _ => return None,
    })
}

pub(crate) fn has_depth(format: TextureFormat) -> bool {
    use TextureFormat as F;
    matches!(format, F::Depth16Unorm | F::Depth24Plus | F::Depth24PlusStencil8 | F::Depth32Float | F::Depth32FloatStencil8)
}

pub(crate) fn has_stencil(format: TextureFormat) -> bool {
    use TextureFormat as F;
    matches!(format, F::Stencil8 | F::Depth24PlusStencil8 | F::Depth32FloatStencil8)
}

/// Texel block edge (1 for uncompressed formats).
pub(crate) fn block_dim(format: TextureFormat) -> (u32, u32) {
    use TextureFormat as F;
    match format {
        F::ASTC5x4Unorm | F::ASTC5x4UnormSrgb => (5, 4),
        F::ASTC5x5Unorm | F::ASTC5x5UnormSrgb => (5, 5),
        F::ASTC6x5Unorm | F::ASTC6x5UnormSrgb => (6, 5),
        F::ASTC6x6Unorm | F::ASTC6x6UnormSrgb => (6, 6),
        F::ASTC8x5Unorm | F::ASTC8x5UnormSrgb => (8, 5),
        F::ASTC8x6Unorm | F::ASTC8x6UnormSrgb => (8, 6),
        F::ASTC8x8Unorm | F::ASTC8x8UnormSrgb => (8, 8),
        F::ASTC10x5Unorm | F::ASTC10x5UnormSrgb => (10, 5),
        F::ASTC10x6Unorm | F::ASTC10x6UnormSrgb => (10, 6),
        F::ASTC10x8Unorm | F::ASTC10x8UnormSrgb => (10, 8),
        F::ASTC10x10Unorm | F::ASTC10x10UnormSrgb => (10, 10),
        F::ASTC12x10Unorm | F::ASTC12x10UnormSrgb => (12, 10),
        F::ASTC12x12Unorm | F::ASTC12x12UnormSrgb => (12, 12),
        _ => match crate::format_block_info(format) {
            Some((_, b)) => (b, b),
            None if format.0 >= F::ETC2RGB8Unorm.0 && format.0 <= F::ASTC4x4UnormSrgb.0 => (4, 4),
            None => (1, 1),
        },
    }
}

pub(crate) fn view_type(dimension: TextureViewDimension, multisampled: bool) -> MTLTextureType {
    use TextureViewDimension as D;
    match dimension {
        D::D1 => MTLTextureType::Type1D,
        D::D2Array if multisampled => MTLTextureType::Type2DMultisampleArray,
        D::D2Array => MTLTextureType::Type2DArray,
        D::Cube => MTLTextureType::TypeCube,
        D::CubeArray => MTLTextureType::TypeCubeArray,
        D::D3 => MTLTextureType::Type3D,
        _ if multisampled => MTLTextureType::Type2DMultisample,
        _ => MTLTextureType::Type2D,
    }
}

pub(crate) fn compare(f: CompareFunction) -> MTLCompareFunction {
    use CompareFunction as C;
    match f {
        C::Never => MTLCompareFunction::Never,
        C::Less => MTLCompareFunction::Less,
        C::Equal => MTLCompareFunction::Equal,
        C::LessEqual => MTLCompareFunction::LessEqual,
        C::Greater => MTLCompareFunction::Greater,
        C::NotEqual => MTLCompareFunction::NotEqual,
        C::GreaterEqual => MTLCompareFunction::GreaterEqual,
        _ => MTLCompareFunction::Always,
    }
}

pub(crate) fn stencil_op(op: StencilOperation) -> MTLStencilOperation {
    use StencilOperation as S;
    match op {
        S::Zero => MTLStencilOperation::Zero,
        S::Replace => MTLStencilOperation::Replace,
        S::Invert => MTLStencilOperation::Invert,
        S::IncrementClamp => MTLStencilOperation::IncrementClamp,
        S::DecrementClamp => MTLStencilOperation::DecrementClamp,
        S::IncrementWrap => MTLStencilOperation::IncrementWrap,
        S::DecrementWrap => MTLStencilOperation::DecrementWrap,
        _ => MTLStencilOperation::Keep,
    }
}

pub(crate) fn blend_factor(f: BlendFactor) -> MTLBlendFactor {
    use BlendFactor as B;
    match f {
        B::Zero => MTLBlendFactor::Zero,
        B::Src => MTLBlendFactor::SourceColor,
        B::OneMinusSrc => MTLBlendFactor::OneMinusSourceColor,
        B::SrcAlpha => MTLBlendFactor::SourceAlpha,
        B::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        B::Dst => MTLBlendFactor::DestinationColor,
        B::OneMinusDst => MTLBlendFactor::OneMinusDestinationColor,
        B::DstAlpha => MTLBlendFactor::DestinationAlpha,
        B::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        B::SrcAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
        B::Constant => MTLBlendFactor::BlendColor,
        B::OneMinusConstant => MTLBlendFactor::OneMinusBlendColor,
        B::Src1 => MTLBlendFactor::Source1Color,
        B::OneMinusSrc1 => MTLBlendFactor::OneMinusSource1Color,
        B::Src1Alpha => MTLBlendFactor::Source1Alpha,
        B::OneMinusSrc1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
        _ => MTLBlendFactor::One,
    }
}

pub(crate) fn blend_op(op: BlendOperation) -> MTLBlendOperation {
    use BlendOperation as B;
    match op {
        B::Subtract => MTLBlendOperation::Subtract,
        B::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        B::Min => MTLBlendOperation::Min,
        B::Max => MTLBlendOperation::Max,
        _ => MTLBlendOperation::Add,
    }
}

pub(crate) fn write_mask(mask: ColorWriteMask) -> MTLColorWriteMask {
    let mut out = MTLColorWriteMask::None;
    if mask.contains(ColorWriteMask::Red) {
        out |= MTLColorWriteMask::Red;
    }
    if mask.contains(ColorWriteMask::Green) {
        out |= MTLColorWriteMask::Green;
    }
    if mask.contains(ColorWriteMask::Blue) {
        out |= MTLColorWriteMask::Blue;
    }
    if mask.contains(ColorWriteMask::Alpha) {
        out |= MTLColorWriteMask::Alpha;
    }
    out
}

pub(crate) fn address_mode(m: AddressMode) -> MTLSamplerAddressMode {
    match m {
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        _ => MTLSamplerAddressMode::ClampToEdge,
    }
}

pub(crate) fn min_mag_filter(f: FilterMode) -> MTLSamplerMinMagFilter {
    match f {
        FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
        _ => MTLSamplerMinMagFilter::Nearest,
    }
}

pub(crate) fn mip_filter(f: MipmapFilterMode) -> MTLSamplerMipFilter {
    match f {
        MipmapFilterMode::Linear => MTLSamplerMipFilter::Linear,
        _ => MTLSamplerMipFilter::Nearest,
    }
}

pub(crate) fn primitive(t: PrimitiveTopology) -> (MTLPrimitiveType, MTLPrimitiveTopologyClass) {
    use PrimitiveTopology as P;
    match t {
        P::PointList => (MTLPrimitiveType::Point, MTLPrimitiveTopologyClass::Point),
        P::LineList => (MTLPrimitiveType::Line, MTLPrimitiveTopologyClass::Line),
        P::LineStrip => (MTLPrimitiveType::LineStrip, MTLPrimitiveTopologyClass::Line),
        P::TriangleStrip => (MTLPrimitiveType::TriangleStrip, MTLPrimitiveTopologyClass::Triangle),
        _ => (MTLPrimitiveType::Triangle, MTLPrimitiveTopologyClass::Triangle),
    }
}

pub(crate) fn cull(c: CullMode) -> MTLCullMode {
    match c {
        CullMode::Front => MTLCullMode::Front,
        CullMode::Back => MTLCullMode::Back,
        _ => MTLCullMode::None,
    }
}

/// WebGPU and Metal both define winding in framebuffer space with y pointing down after the
/// viewport transform, so the mapping is direct.
pub(crate) fn winding(f: FrontFace) -> MTLWinding {
    match f {
        FrontFace::CW => MTLWinding::Clockwise,
        _ => MTLWinding::CounterClockwise,
    }
}

pub(crate) fn index_type(f: IndexFormat) -> (MTLIndexType, u64) {
    match f {
        IndexFormat::Uint32 => (MTLIndexType::UInt32, 4),
        _ => (MTLIndexType::UInt16, 2),
    }
}

pub(crate) fn load_action(op: LoadOp) -> MTLLoadAction {
    match op {
        LoadOp::Load => MTLLoadAction::Load,
        LoadOp::Clear => MTLLoadAction::Clear,
        _ => MTLLoadAction::DontCare,
    }
}

pub(crate) fn store_action(op: StoreOp, resolve: bool) -> MTLStoreAction {
    match (op, resolve) {
        (StoreOp::Store, false) => MTLStoreAction::Store,
        (StoreOp::Store, true) => MTLStoreAction::StoreAndMultisampleResolve,
        (_, true) => MTLStoreAction::MultisampleResolve,
        _ => MTLStoreAction::DontCare,
    }
}

pub(crate) fn step_function(m: VertexStepMode) -> MTLVertexStepFunction {
    match m {
        VertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
        _ => MTLVertexStepFunction::PerVertex,
    }
}

/// Metal vertex format and size in bytes.
pub(crate) fn vertex_format(f: VertexFormat) -> (MTLVertexFormat, u64) {
    use MTLVertexFormat as M;
    use VertexFormat as V;
    match f {
        V::Uint8 => (M::UChar, 1),
        V::Uint8x2 => (M::UChar2, 2),
        V::Uint8x4 => (M::UChar4, 4),
        V::Sint8 => (M::Char, 1),
        V::Sint8x2 => (M::Char2, 2),
        V::Sint8x4 => (M::Char4, 4),
        V::Unorm8 => (M::UCharNormalized, 1),
        V::Unorm8x2 => (M::UChar2Normalized, 2),
        V::Unorm8x4 => (M::UChar4Normalized, 4),
        V::Snorm8 => (M::CharNormalized, 1),
        V::Snorm8x2 => (M::Char2Normalized, 2),
        V::Snorm8x4 => (M::Char4Normalized, 4),
        V::Uint16 => (M::UShort, 2),
        V::Uint16x2 => (M::UShort2, 4),
        V::Uint16x4 => (M::UShort4, 8),
        V::Sint16 => (M::Short, 2),
        V::Sint16x2 => (M::Short2, 4),
        V::Sint16x4 => (M::Short4, 8),
        V::Unorm16 => (M::UShortNormalized, 2),
        V::Unorm16x2 => (M::UShort2Normalized, 4),
        V::Unorm16x4 => (M::UShort4Normalized, 8),
        V::Snorm16 => (M::ShortNormalized, 2),
        V::Snorm16x2 => (M::Short2Normalized, 4),
        V::Snorm16x4 => (M::Short4Normalized, 8),
        V::Float16 => (M::Half, 2),
        V::Float16x2 => (M::Half2, 4),
        V::Float16x4 => (M::Half4, 8),
        V::Float32 => (M::Float, 4),
        V::Float32x2 => (M::Float2, 8),
        V::Float32x3 => (M::Float3, 12),
        V::Float32x4 => (M::Float4, 16),
        V::Uint32 => (M::UInt, 4),
        V::Uint32x2 => (M::UInt2, 8),
        V::Uint32x3 => (M::UInt3, 12),
        V::Uint32x4 => (M::UInt4, 16),
        V::Sint32 => (M::Int, 4),
        V::Sint32x2 => (M::Int2, 8),
        V::Sint32x3 => (M::Int3, 12),
        V::Sint32x4 => (M::Int4, 16),
        V::Unorm10_10_10_2 => (M::UInt1010102Normalized, 4),
        V::Unorm8x4BGRA => (M::UChar4Normalized_BGRA, 4),
        _ => (M::Invalid, 0),
    }
}
