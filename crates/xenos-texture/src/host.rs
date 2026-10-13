//! Host (WebGPU) texture formats and the mapping of every Xenos format onto them.

use crate::format::TextureFormat;

/// WebGPU texture formats used by this crate (names mirror `GPUTextureFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostFormat {
    R8Unorm,
    R8Snorm,
    Rg8Unorm,
    Rg8Snorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    /// Requires `texture-formats-tier1` (formerly native `norm16`).
    R16Unorm,
    R16Snorm,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Unorm,
    Rgba16Snorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Uint,
    R32Float,
    Rg32Float,
    Rgba32Float,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    Depth32Float,
    Depth24PlusStencil8,
    /// BC formats require `texture-compression-bc`.
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
}

/// Optional WebGPU features that change the host format choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HostCaps {
    /// `texture-compression-bc`: upload DXT/DXN/DXT5A blocks directly.
    pub texture_compression_bc: bool,
    /// 16-bit normalized formats (`r16unorm` etc., `texture-formats-tier1`).
    pub norm16: bool,
}

/// Optional feature a host format depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostFeature {
    TextureCompressionBc,
    Norm16,
}

impl HostFormat {
    /// The WebGPU `GPUTextureFormat` string.
    pub fn webgpu_name(self) -> &'static str {
        use HostFormat::*;
        match self {
            R8Unorm => "r8unorm",
            R8Snorm => "r8snorm",
            Rg8Unorm => "rg8unorm",
            Rg8Snorm => "rg8snorm",
            Rgba8Unorm => "rgba8unorm",
            Rgba8UnormSrgb => "rgba8unorm-srgb",
            Rgba8Snorm => "rgba8snorm",
            Bgra8Unorm => "bgra8unorm",
            Bgra8UnormSrgb => "bgra8unorm-srgb",
            R16Unorm => "r16unorm",
            R16Snorm => "r16snorm",
            Rg16Unorm => "rg16unorm",
            Rg16Snorm => "rg16snorm",
            Rgba16Unorm => "rgba16unorm",
            Rgba16Snorm => "rgba16snorm",
            R16Float => "r16float",
            Rg16Float => "rg16float",
            Rgba16Float => "rgba16float",
            R32Uint => "r32uint",
            R32Float => "r32float",
            Rg32Float => "rg32float",
            Rgba32Float => "rgba32float",
            Rgb10a2Unorm => "rgb10a2unorm",
            Rg11b10Ufloat => "rg11b10ufloat",
            Depth32Float => "depth32float",
            Depth24PlusStencil8 => "depth24plus-stencil8",
            Bc1RgbaUnorm => "bc1-rgba-unorm",
            Bc1RgbaUnormSrgb => "bc1-rgba-unorm-srgb",
            Bc2RgbaUnorm => "bc2-rgba-unorm",
            Bc2RgbaUnormSrgb => "bc2-rgba-unorm-srgb",
            Bc3RgbaUnorm => "bc3-rgba-unorm",
            Bc3RgbaUnormSrgb => "bc3-rgba-unorm-srgb",
            Bc4RUnorm => "bc4-r-unorm",
            Bc4RSnorm => "bc4-r-snorm",
            Bc5RgUnorm => "bc5-rg-unorm",
            Bc5RgSnorm => "bc5-rg-snorm",
        }
    }

    pub fn is_compressed(self) -> bool {
        self.block_size() == 4
    }

    /// Block width/height in texels (4 for BC, 1 otherwise).
    pub fn block_size(self) -> u32 {
        use HostFormat::*;
        match self {
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb
            | Bc4RUnorm | Bc4RSnorm | Bc5RgUnorm | Bc5RgSnorm => 4,
            _ => 1,
        }
    }

    /// Bytes per texel, or per 4x4 block for compressed formats. Depth24PlusStencil8 has no
    /// defined copy size and reports 4.
    pub fn bytes_per_block(self) -> u32 {
        use HostFormat::*;
        match self {
            R8Unorm | R8Snorm => 1,
            Rg8Unorm | Rg8Snorm | R16Unorm | R16Snorm | R16Float => 2,
            Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Bgra8Unorm | Bgra8UnormSrgb | Rg16Unorm | Rg16Snorm
            | Rg16Float | R32Uint | R32Float | Rgb10a2Unorm | Rg11b10Ufloat | Depth32Float | Depth24PlusStencil8 => 4,
            Rgba16Unorm | Rgba16Snorm | Rgba16Float | Rg32Float => 8,
            Rgba32Float => 16,
            Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm => 8,
            Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc5RgUnorm | Bc5RgSnorm => 16,
        }
    }

    pub fn required_feature(self) -> Option<HostFeature> {
        use HostFormat::*;
        match self {
            R16Unorm | R16Snorm | Rg16Unorm | Rg16Snorm | Rgba16Unorm | Rgba16Snorm => Some(HostFeature::Norm16),
            f if f.is_compressed() => Some(HostFeature::TextureCompressionBc),
            _ => None,
        }
    }

    /// Whether the format is filterable without extra features (r32float/rg32float/rgba32float
    /// need `float32-filterable`; integer formats never are).
    pub fn is_filterable(self) -> bool {
        !matches!(
            self,
            HostFormat::R32Uint
                | HostFormat::R32Float
                | HostFormat::Rg32Float
                | HostFormat::Rgba32Float
                | HostFormat::Depth32Float
                | HostFormat::Depth24PlusStencil8
        )
    }

    /// sRGB variant for approximating Xenos gamma (`TextureSign::Gamma`) in hardware. Xenos
    /// uses a piecewise-linear curve, so this is an approximation.
    pub fn srgb(self) -> Option<HostFormat> {
        use HostFormat::*;
        match self {
            Rgba8Unorm => Some(Rgba8UnormSrgb),
            Bgra8Unorm => Some(Bgra8UnormSrgb),
            Bc1RgbaUnorm => Some(Bc1RgbaUnormSrgb),
            Bc2RgbaUnorm => Some(Bc2RgbaUnormSrgb),
            Bc3RgbaUnorm => Some(Bc3RgbaUnormSrgb),
            _ => None,
        }
    }
}

/// CPU work needed to turn (endian-swapped) guest blocks into host data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Conversion {
    /// Bytes are copied unchanged (also BC passthrough).
    Copy,
    /// Packed fixed-point texel with up to four components at (shift, bits), expanded to
    /// 8 bits per component (Rgba8Unorm / Rgba8Snorm). Missing components become 1.
    PackedToRgba8 {
        fields: [(u8, u8); 4],
        count: u8,
        signed: bool,
    },
    /// Packed fixed-point texel expanded to 16-bit components (Rgba16Unorm/Snorm, or
    /// Rgba16Float when `float` is set). Missing components become 1.
    PackedToRgba16 {
        fields: [(u8, u8); 4],
        count: u8,
        signed: bool,
        float: bool,
    },
    /// `n` 16-bit normalized components to half floats.
    Norm16ToFloat16 {
        components: u8,
        signed: bool,
    },
    /// `n` 32-bit normalized components to f32.
    Norm32ToFloat32 {
        components: u8,
        signed: bool,
    },
    /// EDRAM 16-bit fixed point (range -32..32) to half floats.
    Edram16ToFloat16 {
        components: u8,
    },
    /// 7e3 floating-point RGB with 2-bit alpha to Rgba16Float.
    Float7e3ToRgba16Float,
    /// RGB32 float to RGBA32 float with alpha 1.
    Rgb32FloatToRgba32Float,
    /// 24-bit unorm depth in the high bits to R32Float.
    Depth24UnormToFloat32,
    /// 20e4 float depth in the high bits to R32Float.
    Depth20e4ToFloat32,
    /// 2x1 Cr/Y1/Cb/Y0 (GBGR) or Y1/Cr/Y0/Cb (BGRG) blocks to (Cr, Y, Cb, Cb) RGBA8.
    YuvGbgrToRgba8,
    YuvBgrgToRgba8,
    Bc1ToRgba8,
    Bc2ToRgba8,
    Bc3ToRgba8,
    /// DXT5A: BC4 layout.
    Bc4ToR8 {
        signed: bool,
    },
    /// DXN: BC5 layout.
    Bc5ToRg8 {
        signed: bool,
    },
    /// DXT3A: 4-bit alpha (BC2 alpha half) to R8.
    Dxt3aToR8,
    /// DXT3A_AS_1_1_1_1: each 4-bit alpha is the R/G/B/A bits.
    Dxt3aAs1111ToRgba8,
    /// CTX1: two 8:8 endpoints, 4-color interpolation, to RG8.
    Ctx1ToRg8 {
        signed: bool,
    },
}

/// How a host channel feeds a guest component (`x`, `y`, `z`, `w` before the fetch swizzle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostChannel {
    R,
    G,
    B,
    A,
}

/// Result of mapping a guest format + signedness to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostMapping {
    pub host_format: HostFormat,
    pub conversion: Conversion,
    /// For guest components x/y/z/w, the host channel holding them. Formats with fewer
    /// components replicate the last one (Xenos behavior).
    pub component_map: [HostChannel; 4],
    /// Present when the host representation loses information.
    pub lossy: Option<&'static str>,
}

/// Overall support class of a guest format, for reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Support {
    /// Host format with identical bits exists (possibly behind an optional feature).
    Direct,
    /// CPU conversion is required.
    Converted,
    Unsupported(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MappingError {
    Unsupported(TextureFormat, &'static str),
    /// No host representation for the signed interpretation of this format.
    NoSignedVariant(TextureFormat),
}

impl std::fmt::Display for MappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MappingError::Unsupported(fmt, why) => write!(f, "{} is unsupported: {why}", fmt.name()),
            MappingError::NoSignedVariant(fmt) => write!(f, "{} has no signed host representation", fmt.name()),
        }
    }
}

impl std::error::Error for MappingError {}

use HostChannel::{A, B, G, R};
const RRRR: [HostChannel; 4] = [R, R, R, R];
const RGGG: [HostChannel; 4] = [R, G, G, G];
const RGBB: [HostChannel; 4] = [R, G, B, B];
const RGBA: [HostChannel; 4] = [R, G, B, A];

const UNKNOWN_SEMANTICS: &str = "video/interlaced or packed-as-8 format with unknown sampling semantics";

fn m(host_format: HostFormat, conversion: Conversion, component_map: [HostChannel; 4]) -> HostMapping {
    HostMapping { host_format, conversion, component_map, lossy: None }
}

fn lossy(mut mapping: HostMapping, why: &'static str) -> HostMapping {
    mapping.lossy = Some(why);
    mapping
}

fn fields(spec: &[(u8, u8)]) -> ([(u8, u8); 4], u8) {
    let mut f = [(0u8, 0u8); 4];
    f[..spec.len()].copy_from_slice(spec);
    (f, spec.len() as u8)
}

fn packed8(spec: &[(u8, u8)], signed: bool, map: [HostChannel; 4]) -> HostMapping {
    let (fields, count) = fields(spec);
    let host = if signed { HostFormat::Rgba8Snorm } else { HostFormat::Rgba8Unorm };
    m(host, Conversion::PackedToRgba8 { fields, count, signed }, map)
}

fn packed16(spec: &[(u8, u8)], signed: bool, caps: HostCaps, map: [HostChannel; 4]) -> HostMapping {
    let (fields, count) = fields(spec);
    if caps.norm16 {
        let host = if signed { HostFormat::Rgba16Snorm } else { HostFormat::Rgba16Unorm };
        m(host, Conversion::PackedToRgba16 { fields, count, signed, float: false }, map)
    } else {
        lossy(
            m(HostFormat::Rgba16Float, Conversion::PackedToRgba16 { fields, count, signed, float: true }, map),
            "fixed-point stored as half float (11-bit mantissa)",
        )
    }
}

fn norm16(components: u8, signed: bool, caps: HostCaps, map: [HostChannel; 4]) -> HostMapping {
    use HostFormat::*;
    if caps.norm16 {
        let host = match (components, signed) {
            (1, false) => R16Unorm,
            (1, true) => R16Snorm,
            (2, false) => Rg16Unorm,
            (2, true) => Rg16Snorm,
            (_, false) => Rgba16Unorm,
            (_, true) => Rgba16Snorm,
        };
        m(host, Conversion::Copy, map)
    } else {
        let host = match components {
            1 => R16Float,
            2 => Rg16Float,
            _ => Rgba16Float,
        };
        lossy(
            m(host, Conversion::Norm16ToFloat16 { components, signed }, map),
            "16-bit normalized stored as half float (11-bit mantissa)",
        )
    }
}

/// Whether BC data of this base size can be uploaded as-is (WebGPU requires level 0 of
/// block-compressed textures to be a multiple of the block size).
pub fn bc_size_compatible(width: u32, height: u32) -> bool {
    width.is_multiple_of(4) && height.is_multiple_of(4)
}

/// Chooses the host format and conversion for `format` (after `base_format()`), for either the
/// unsigned view (`signed == false`; also used for biased and gamma components) or the signed
/// view. `width`/`height` are the level 0 size and decide whether BC data can pass through.
pub fn host_mapping(
    format: TextureFormat,
    signed: bool,
    caps: HostCaps,
    width: u32,
    height: u32,
) -> Result<HostMapping, MappingError> {
    use Conversion as C;
    use HostFormat::*;
    use TextureFormat::*;
    let format = format.base_format();
    let bc = caps.texture_compression_bc && bc_size_compatible(width, height);
    let no_signed = Err(MappingError::NoSignedVariant(format));
    Ok(match format {
        k_1_REVERSE | k_1 => {
            return Err(MappingError::Unsupported(format, "1 bpp formats have no known memory layout or usage"));
        }
        k_32_AS_8
        | k_32_AS_8_8
        | k_16_MPEG
        | k_16_16_MPEG
        | k_8_INTERLACED
        | k_32_AS_8_INTERLACED
        | k_32_AS_8_8_INTERLACED
        | k_16_INTERLACED
        | k_16_MPEG_INTERLACED
        | k_16_16_MPEG_INTERLACED => {
            return Err(MappingError::Unsupported(format, UNKNOWN_SEMANTICS));
        }
        k_8 | k_8_A | k_8_B => m(if signed { R8Snorm } else { R8Unorm }, C::Copy, RRRR),
        k_8_8 => m(if signed { Rg8Snorm } else { Rg8Unorm }, C::Copy, RGGG),
        k_8_8_8_8 | k_8_8_8_8_A => m(if signed { Rgba8Snorm } else { Rgba8Unorm }, C::Copy, RGBA),
        k_1_5_5_5 => packed8(&[(0, 5), (5, 5), (10, 5), (15, 1)], signed, RGBA),
        k_5_6_5 => packed8(&[(0, 5), (5, 6), (11, 5)], signed, RGBB),
        k_6_5_5 => packed8(&[(0, 5), (5, 5), (10, 6)], signed, RGBB),
        k_4_4_4_4 => packed8(&[(0, 4), (4, 4), (8, 4), (12, 4)], signed, RGBA),
        k_2_10_10_10 => {
            if signed {
                packed16(&[(0, 10), (10, 10), (20, 10), (30, 2)], true, caps, RGBA)
            } else {
                m(Rgb10a2Unorm, C::Copy, RGBA)
            }
        }
        k_10_11_11 => packed16(&[(0, 11), (11, 11), (22, 10)], signed, caps, RGBB),
        k_11_11_10 => packed16(&[(0, 10), (10, 11), (21, 11)], signed, caps, RGBB),
        k_Cr_Y1_Cb_Y0_REP => m(if signed { Rgba8Snorm } else { Rgba8Unorm }, C::YuvGbgrToRgba8, RGBB),
        k_Y1_Cr_Y0_Cb_REP => m(if signed { Rgba8Snorm } else { Rgba8Unorm }, C::YuvBgrgToRgba8, RGBB),
        k_16_16_EDRAM => lossy(
            m(Rg16Float, C::Edram16ToFloat16 { components: 2 }, RGGG),
            "EDRAM -32..32 fixed point stored as half float",
        ),
        k_16_16_16_16_EDRAM => lossy(
            m(Rgba16Float, C::Edram16ToFloat16 { components: 4 }, RGBA),
            "EDRAM -32..32 fixed point stored as half float",
        ),
        k_DXT1 => {
            if signed {
                return no_signed;
            } else if bc {
                m(Bc1RgbaUnorm, C::Copy, RGBA)
            } else {
                m(Rgba8Unorm, C::Bc1ToRgba8, RGBA)
            }
        }
        k_DXT2_3 => {
            if signed {
                return no_signed;
            } else if bc {
                m(Bc2RgbaUnorm, C::Copy, RGBA)
            } else {
                m(Rgba8Unorm, C::Bc2ToRgba8, RGBA)
            }
        }
        k_DXT4_5 => {
            if signed {
                return no_signed;
            } else if bc {
                m(Bc3RgbaUnorm, C::Copy, RGBA)
            } else {
                m(Rgba8Unorm, C::Bc3ToRgba8, RGBA)
            }
        }
        k_DXT5A => {
            if bc {
                m(if signed { Bc4RSnorm } else { Bc4RUnorm }, C::Copy, RRRR)
            } else {
                m(if signed { R8Snorm } else { R8Unorm }, C::Bc4ToR8 { signed }, RRRR)
            }
        }
        k_DXN => {
            if bc {
                m(if signed { Bc5RgSnorm } else { Bc5RgUnorm }, C::Copy, RGGG)
            } else {
                m(if signed { Rg8Snorm } else { Rg8Unorm }, C::Bc5ToRg8 { signed }, RGGG)
            }
        }
        k_DXT3A => {
            if signed {
                return no_signed;
            }
            m(R8Unorm, C::Dxt3aToR8, RRRR)
        }
        k_DXT3A_AS_1_1_1_1 => {
            if signed {
                return no_signed;
            }
            m(Rgba8Unorm, C::Dxt3aAs1111ToRgba8, RGBA)
        }
        k_CTX1 => m(if signed { Rg8Snorm } else { Rg8Unorm }, C::Ctx1ToRg8 { signed }, RGGG),
        k_24_8 => lossy(
            m(R32Float, C::Depth24UnormToFloat32, RRRR),
            "stencil dropped; depth sampled as r32float (no hardware comparison)",
        ),
        k_24_8_FLOAT => lossy(
            m(R32Float, C::Depth20e4ToFloat32, RRRR),
            "stencil dropped; depth sampled as r32float (no hardware comparison)",
        ),
        k_16 => norm16(1, signed, caps, RRRR),
        k_16_16 => norm16(2, signed, caps, RGGG),
        k_16_16_16_16 => norm16(4, signed, caps, RGBA),
        k_16_FLOAT => m(R16Float, C::Copy, RRRR),
        k_16_16_FLOAT => m(Rg16Float, C::Copy, RGGG),
        k_16_16_16_16_FLOAT => m(Rgba16Float, C::Copy, RGBA),
        k_32 => lossy(
            m(R32Float, C::Norm32ToFloat32 { components: 1, signed }, RRRR),
            "32-bit normalized stored as f32 (24-bit mantissa); not filterable without float32-filterable",
        ),
        k_32_32 => lossy(
            m(Rg32Float, C::Norm32ToFloat32 { components: 2, signed }, RGGG),
            "32-bit normalized stored as f32 (24-bit mantissa); not filterable without float32-filterable",
        ),
        k_32_32_32_32 => lossy(
            m(Rgba32Float, C::Norm32ToFloat32 { components: 4, signed }, RGBA),
            "32-bit normalized stored as f32 (24-bit mantissa); not filterable without float32-filterable",
        ),
        k_32_FLOAT => m(R32Float, C::Copy, RRRR),
        k_32_32_FLOAT => m(Rg32Float, C::Copy, RGGG),
        k_32_32_32_32_FLOAT => m(Rgba32Float, C::Copy, RGBA),
        k_32_32_32_FLOAT => m(Rgba32Float, C::Rgb32FloatToRgba32Float, RGBB),
        // 7e3 values are exactly representable as halves.
        k_2_10_10_10_FLOAT_EDRAM => m(Rgba16Float, C::Float7e3ToRgba16Float, RGBA),
        // Aliases resolved by base_format().
        k_16_EXPAND
        | k_16_16_EXPAND
        | k_16_16_16_16_EXPAND
        | k_8_8_8_8_AS_16_16_16_16
        | k_DXT1_AS_16_16_16_16
        | k_DXT2_3_AS_16_16_16_16
        | k_DXT4_5_AS_16_16_16_16
        | k_2_10_10_10_AS_16_16_16_16
        | k_10_11_11_AS_16_16_16_16
        | k_11_11_10_AS_16_16_16_16
        | k_8_8_8_8_GAMMA_EDRAM => unreachable!("aliases are resolved by base_format"),
    })
}

/// Support classification with all optional features available and a block-aligned size.
pub fn support(format: TextureFormat) -> Support {
    let caps = HostCaps { texture_compression_bc: true, norm16: true };
    match host_mapping(format, false, caps, 4, 4) {
        Ok(mapping) if mapping.conversion == Conversion::Copy => Support::Direct,
        Ok(_) => Support::Converted,
        Err(MappingError::Unsupported(_, why)) => Support::Unsupported(why),
        Err(MappingError::NoSignedVariant(_)) => unreachable!("unsigned mapping requested"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TextureFormat::*;

    #[test]
    fn every_format_classified() {
        let mut direct = 0;
        let mut converted = 0;
        let mut unsupported = 0;
        for f in TextureFormat::ALL {
            match support(f) {
                Support::Direct => direct += 1,
                Support::Converted => converted += 1,
                Support::Unsupported(_) => unsupported += 1,
            }
        }
        assert_eq!(direct + converted + unsupported, 64);
        assert_eq!(unsupported, 12);
    }

    #[test]
    fn bc_passthrough_and_fallback() {
        let bc = HostCaps { texture_compression_bc: true, norm16: false };
        let m = host_mapping(k_DXT1, false, bc, 64, 64).unwrap();
        assert_eq!((m.host_format, m.conversion), (HostFormat::Bc1RgbaUnorm, Conversion::Copy));
        // Not block aligned: decompress.
        let m = host_mapping(k_DXT1, false, bc, 30, 64).unwrap();
        assert_eq!((m.host_format, m.conversion), (HostFormat::Rgba8Unorm, Conversion::Bc1ToRgba8));
        let m = host_mapping(k_DXT4_5_AS_16_16_16_16, false, HostCaps::default(), 64, 64).unwrap();
        assert_eq!((m.host_format, m.conversion), (HostFormat::Rgba8Unorm, Conversion::Bc3ToRgba8));
        let m = host_mapping(k_DXN, true, bc, 64, 64).unwrap();
        assert_eq!(m.host_format, HostFormat::Bc5RgSnorm);
        let m = host_mapping(k_DXN, true, HostCaps::default(), 64, 64).unwrap();
        assert_eq!((m.host_format, m.conversion), (HostFormat::Rg8Snorm, Conversion::Bc5ToRg8 { signed: true }));
        assert_eq!(host_mapping(k_DXT1, true, bc, 64, 64), Err(MappingError::NoSignedVariant(k_DXT1)));
    }

    #[test]
    fn norm16_choice() {
        let m = host_mapping(k_16_16, false, HostCaps::default(), 4, 4).unwrap();
        assert_eq!(m.host_format, HostFormat::Rg16Float);
        assert!(m.lossy.is_some());
        let m = host_mapping(k_16_16, true, HostCaps { norm16: true, ..Default::default() }, 4, 4).unwrap();
        assert_eq!((m.host_format, m.conversion), (HostFormat::Rg16Snorm, Conversion::Copy));
    }

    #[test]
    fn host_format_properties() {
        assert_eq!(HostFormat::Bc1RgbaUnorm.bytes_per_block(), 8);
        assert_eq!(HostFormat::Bc3RgbaUnorm.block_size(), 4);
        assert_eq!(HostFormat::Rgba16Float.bytes_per_block(), 8);
        assert_eq!(HostFormat::Rg11b10Ufloat.webgpu_name(), "rg11b10ufloat");
        assert_eq!(HostFormat::Rgba8Unorm.srgb(), Some(HostFormat::Rgba8UnormSrgb));
        assert_eq!(HostFormat::R16Unorm.required_feature(), Some(HostFeature::Norm16));
        assert_eq!(HostFormat::Bc4RSnorm.required_feature(), Some(HostFeature::TextureCompressionBc));
        assert!(!HostFormat::R32Float.is_filterable());
    }
}
