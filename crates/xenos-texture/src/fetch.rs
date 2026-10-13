//! Texture and vertex fetch constant decoding (`xe_gpu_texture_fetch_t` /
//! `xe_gpu_vertex_fetch_t`). Input dwords are in host order (already register values).

use crate::endian::Endian;
use crate::format::TextureFormat;

fn bits(v: u32, shift: u32, width: u32) -> u32 {
    (v >> shift) & ((1u32 << width) - 1)
}

fn sbits(v: u32, shift: u32, width: u32) -> i32 {
    ((v << (32 - shift - width)) as i32) >> (32 - width)
}

fn put(v: u32, shift: u32, width: u32) -> u32 {
    (v & ((1u32 << width) - 1)) << shift
}

/// `SQ_TEX_VTX_INVALID/VALID_TEXTURE/BUFFER`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FetchConstantType {
    #[default]
    InvalidTexture = 0,
    InvalidVertex = 1,
    Texture = 2,
    Vertex = 3,
}

impl FetchConstantType {
    pub fn from_bits(b: u32) -> Self {
        match b & 3 {
            0 => Self::InvalidTexture,
            1 => Self::InvalidVertex,
            2 => Self::Texture,
            _ => Self::Vertex,
        }
    }
}

/// Per-component data interpretation (GPUSIGN).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureSign {
    #[default]
    Unsigned = 0,
    /// Two's complement data.
    Signed = 1,
    /// Unsigned data remapped as `2 * x - 1` after sampling.
    UnsignedBiased = 2,
    /// Unsigned data linearized (Xenos piecewise-linear gamma) when sampled.
    Gamma = 3,
}

impl TextureSign {
    pub fn from_bits(b: u32) -> Self {
        match b & 3 {
            0 => Self::Unsigned,
            1 => Self::Signed,
            2 => Self::UnsignedBiased,
            _ => Self::Gamma,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClampMode {
    #[default]
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
    MirrorClampToEdge = 3,
    ClampToHalfway = 4,
    MirrorClampToHalfway = 5,
    ClampToBorder = 6,
    MirrorClampToBorder = 7,
}

impl ClampMode {
    pub fn from_bits(b: u32) -> Self {
        match b & 7 {
            0 => Self::Repeat,
            1 => Self::MirroredRepeat,
            2 => Self::ClampToEdge,
            3 => Self::MirrorClampToEdge,
            4 => Self::ClampToHalfway,
            5 => Self::MirrorClampToHalfway,
            6 => Self::ClampToBorder,
            _ => Self::MirrorClampToBorder,
        }
    }

    pub fn uses_border(self) -> bool {
        matches!(self, Self::ClampToBorder | Self::MirrorClampToBorder)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureFilter {
    #[default]
    Point = 0,
    Linear = 1,
    /// Mip filter only: sample the base level without mip filtering.
    BaseMap = 2,
    /// In fetch instructions: take the value from the fetch constant.
    UseFetchConst = 3,
}

impl TextureFilter {
    pub fn from_bits(b: u32) -> Self {
        match b & 3 {
            0 => Self::Point,
            1 => Self::Linear,
            2 => Self::BaseMap,
            _ => Self::UseFetchConst,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AnisoFilter {
    #[default]
    Disabled = 0,
    Max1to1 = 1,
    Max2to1 = 2,
    Max4to1 = 3,
    Max8to1 = 4,
    Max16to1 = 5,
    /// Encoding 6 is undefined; kept distinct so it round-trips.
    Reserved6 = 6,
    UseFetchConst = 7,
}

impl AnisoFilter {
    pub fn from_bits(b: u32) -> Self {
        match b & 7 {
            0 => Self::Disabled,
            1 => Self::Max1to1,
            2 => Self::Max2to1,
            3 => Self::Max4to1,
            4 => Self::Max8to1,
            5 => Self::Max16to1,
            6 => Self::Reserved6,
            _ => Self::UseFetchConst,
        }
    }

    /// Maximum anisotropy ratio, or `None` when disabled.
    pub fn max_anisotropy(self) -> Option<u16> {
        match self {
            Self::Disabled | Self::Reserved6 | Self::UseFetchConst => None,
            other => Some(1 << (other as u16 - 1)),
        }
    }
}

/// Deprecated arbitrary (kernel) filter field, kept for completeness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ArbitraryFilter {
    #[default]
    Sym2x4 = 0,
    Asym2x4 = 1,
    Sym4x2 = 2,
    Asym4x2 = 3,
    Sym4x4 = 4,
    Asym4x4 = 5,
    Reserved6 = 6,
    UseFetchConst = 7,
}

impl ArbitraryFilter {
    pub fn from_bits(b: u32) -> Self {
        match b & 7 {
            0 => Self::Sym2x4,
            1 => Self::Asym2x4,
            2 => Self::Sym4x2,
            3 => Self::Asym4x2,
            4 => Self::Sym4x4,
            5 => Self::Asym4x4,
            6 => Self::Reserved6,
            _ => Self::UseFetchConst,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BorderColor {
    /// (0, 0, 0, 0).
    #[default]
    AbgrBlack = 0,
    /// (1, 1, 1, 1).
    AbgrWhite = 1,
    /// YUV black, likely (Cr, Y, Cb) = (0.5, 0, 0.5).
    AcbycrBlack = 2,
    /// YUV black, likely (Y, Cr, Cb) = (0, 0.5, 0.5).
    AcbcryBlack = 3,
}

impl BorderColor {
    pub fn from_bits(b: u32) -> Self {
        match b & 3 {
            0 => Self::AbgrBlack,
            1 => Self::AbgrWhite,
            2 => Self::AcbycrBlack,
            _ => Self::AcbcryBlack,
        }
    }

    /// RGBA border value as Xenia programs it into host samplers.
    pub fn rgba(self) -> [f32; 4] {
        match self {
            Self::AbgrBlack => [0.0, 0.0, 0.0, 0.0],
            Self::AbgrWhite => [1.0, 1.0, 1.0, 1.0],
            Self::AcbycrBlack => [0.5, 0.0, 0.5, 0.0],
            Self::AcbcryBlack => [0.0, 0.5, 0.5, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DataDimension {
    D1 = 0,
    #[default]
    D2OrStacked = 1,
    D3 = 2,
    Cube = 3,
}

impl DataDimension {
    pub fn from_bits(b: u32) -> Self {
        match b & 3 {
            0 => Self::D1,
            1 => Self::D2OrStacked,
            2 => Self::D3,
            _ => Self::Cube,
        }
    }
}

/// Whether fixed-point data is a [0, 1] / [-1, 1] fraction or a raw integer value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NumFormat {
    #[default]
    Fraction = 0,
    Integer = 1,
}

/// Source of one destination component in the fetch constant swizzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    X,
    Y,
    Z,
    W,
    Zero,
    One,
}

impl SwizzleSource {
    pub fn from_bits(b: u32) -> Self {
        match b & 7 {
            0 => Self::X,
            1 => Self::Y,
            2 => Self::Z,
            3 => Self::W,
            4 | 6 => Self::Zero,
            _ => Self::One,
        }
    }

    pub fn to_bits(self) -> u32 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
            Self::W => 3,
            Self::Zero => 4,
            Self::One => 5,
        }
    }

    /// Component index for X/Y/Z/W.
    pub fn component(self) -> Option<usize> {
        match self {
            Self::X => Some(0),
            Self::Y => Some(1),
            Self::Z => Some(2),
            Self::W => Some(3),
            _ => None,
        }
    }
}

/// Fully decoded texture fetch constant. Sizes are actual sizes (the encoded minus-one values
/// plus one); which of `width`/`height`/`depth` are meaningful depends on `dimension`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureFetchConstant {
    pub fetch_type: FetchConstantType,
    pub sign: [TextureSign; 4],
    pub clamp: [ClampMode; 3],
    /// Base row pitch in texels divided by 32.
    pub pitch_div_32: u32,
    pub tiled: bool,

    pub format: TextureFormat,
    pub endianness: Endian,
    pub request_size: u32,
    pub stacked: bool,
    pub nearest_clamp_policy: bool,
    /// Base address >> 12 (20 bits).
    pub base_page: u32,

    /// 1D: up to 2^24; 2D/cube: up to 8192; 3D: up to 2048.
    pub width: u32,
    pub height: u32,
    /// 2D/cube: stack depth (cube should be 6); 3D: depth.
    pub depth: u32,

    pub num_format: NumFormat,
    pub swizzle: [SwizzleSource; 4],
    pub exp_adjust: i32,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mip_filter: TextureFilter,
    pub aniso_filter: AnisoFilter,
    pub arbitrary_filter: ArbitraryFilter,
    pub border_size: bool,

    pub vol_mag_filter: TextureFilter,
    pub vol_min_filter: TextureFilter,
    pub mip_min_level: u32,
    pub mip_max_level: u32,
    pub mag_aniso_walk: bool,
    pub min_aniso_walk: bool,
    /// LOD bias in 1/32 units.
    pub lod_bias: i32,
    pub grad_exp_adjust_h: i32,
    pub grad_exp_adjust_v: i32,

    pub border_color: BorderColor,
    pub force_bc_w_to_max: bool,
    pub tri_clamp: u32,
    pub aniso_bias: i32,
    pub dimension: DataDimension,
    pub packed_mips: bool,
    /// Mip address >> 12 (20 bits).
    pub mip_page: u32,
}

impl TextureFetchConstant {
    pub fn decode(d: [u32; 6]) -> Self {
        let dimension = DataDimension::from_bits(bits(d[5], 9, 2));
        let (width, height, depth) = match dimension {
            DataDimension::D1 => (bits(d[2], 0, 24) + 1, 1, 1),
            DataDimension::D2OrStacked | DataDimension::Cube => {
                (bits(d[2], 0, 13) + 1, bits(d[2], 13, 13) + 1, bits(d[2], 26, 6) + 1)
            }
            DataDimension::D3 => (bits(d[2], 0, 11) + 1, bits(d[2], 11, 11) + 1, bits(d[2], 22, 10) + 1),
        };
        TextureFetchConstant {
            fetch_type: FetchConstantType::from_bits(d[0]),
            sign: [
                TextureSign::from_bits(bits(d[0], 2, 2)),
                TextureSign::from_bits(bits(d[0], 4, 2)),
                TextureSign::from_bits(bits(d[0], 6, 2)),
                TextureSign::from_bits(bits(d[0], 8, 2)),
            ],
            clamp: [
                ClampMode::from_bits(bits(d[0], 10, 3)),
                ClampMode::from_bits(bits(d[0], 13, 3)),
                ClampMode::from_bits(bits(d[0], 16, 3)),
            ],
            pitch_div_32: bits(d[0], 22, 9),
            tiled: bits(d[0], 31, 1) != 0,
            format: TextureFormat::from_bits(d[1]),
            endianness: Endian::from_bits(bits(d[1], 6, 2)),
            request_size: bits(d[1], 8, 2),
            stacked: bits(d[1], 10, 1) != 0,
            nearest_clamp_policy: bits(d[1], 11, 1) != 0,
            base_page: bits(d[1], 12, 20),
            width,
            height,
            depth,
            num_format: if d[3] & 1 != 0 { NumFormat::Integer } else { NumFormat::Fraction },
            swizzle: [
                SwizzleSource::from_bits(bits(d[3], 1, 3)),
                SwizzleSource::from_bits(bits(d[3], 4, 3)),
                SwizzleSource::from_bits(bits(d[3], 7, 3)),
                SwizzleSource::from_bits(bits(d[3], 10, 3)),
            ],
            exp_adjust: sbits(d[3], 13, 6),
            mag_filter: TextureFilter::from_bits(bits(d[3], 19, 2)),
            min_filter: TextureFilter::from_bits(bits(d[3], 21, 2)),
            mip_filter: TextureFilter::from_bits(bits(d[3], 23, 2)),
            aniso_filter: AnisoFilter::from_bits(bits(d[3], 25, 3)),
            arbitrary_filter: ArbitraryFilter::from_bits(bits(d[3], 28, 3)),
            border_size: bits(d[3], 31, 1) != 0,
            vol_mag_filter: TextureFilter::from_bits(bits(d[4], 0, 1)),
            vol_min_filter: TextureFilter::from_bits(bits(d[4], 1, 1)),
            mip_min_level: bits(d[4], 2, 4),
            mip_max_level: bits(d[4], 6, 4),
            mag_aniso_walk: bits(d[4], 10, 1) != 0,
            min_aniso_walk: bits(d[4], 11, 1) != 0,
            lod_bias: sbits(d[4], 12, 10),
            grad_exp_adjust_h: sbits(d[4], 22, 5),
            grad_exp_adjust_v: sbits(d[4], 27, 5),
            border_color: BorderColor::from_bits(d[5]),
            force_bc_w_to_max: bits(d[5], 2, 1) != 0,
            tri_clamp: bits(d[5], 3, 2),
            aniso_bias: sbits(d[5], 5, 4),
            dimension,
            packed_mips: bits(d[5], 11, 1) != 0,
            mip_page: bits(d[5], 12, 20),
        }
    }

    pub fn encode(&self) -> [u32; 6] {
        let mut d = [0u32; 6];
        d[0] = put(self.fetch_type as u32, 0, 2)
            | put(self.sign[0] as u32, 2, 2)
            | put(self.sign[1] as u32, 4, 2)
            | put(self.sign[2] as u32, 6, 2)
            | put(self.sign[3] as u32, 8, 2)
            | put(self.clamp[0] as u32, 10, 3)
            | put(self.clamp[1] as u32, 13, 3)
            | put(self.clamp[2] as u32, 16, 3)
            | put(self.pitch_div_32, 22, 9)
            | put(self.tiled as u32, 31, 1);
        d[1] = put(self.format as u32, 0, 6)
            | put(self.endianness as u32, 6, 2)
            | put(self.request_size, 8, 2)
            | put(self.stacked as u32, 10, 1)
            | put(self.nearest_clamp_policy as u32, 11, 1)
            | put(self.base_page, 12, 20);
        let m1 = |v: u32| v.saturating_sub(1);
        d[2] = match self.dimension {
            DataDimension::D1 => put(m1(self.width), 0, 24),
            DataDimension::D2OrStacked | DataDimension::Cube => {
                put(m1(self.width), 0, 13) | put(m1(self.height), 13, 13) | put(m1(self.depth), 26, 6)
            }
            DataDimension::D3 => {
                put(m1(self.width), 0, 11) | put(m1(self.height), 11, 11) | put(m1(self.depth), 22, 10)
            }
        };
        d[3] = put(self.num_format as u32, 0, 1)
            | put(self.swizzle_bits(), 1, 12)
            | put(self.exp_adjust as u32, 13, 6)
            | put(self.mag_filter as u32, 19, 2)
            | put(self.min_filter as u32, 21, 2)
            | put(self.mip_filter as u32, 23, 2)
            | put(self.aniso_filter as u32, 25, 3)
            | put(self.arbitrary_filter as u32, 28, 3)
            | put(self.border_size as u32, 31, 1);
        d[4] = put(self.vol_mag_filter as u32, 0, 1)
            | put(self.vol_min_filter as u32, 1, 1)
            | put(self.mip_min_level, 2, 4)
            | put(self.mip_max_level, 6, 4)
            | put(self.mag_aniso_walk as u32, 10, 1)
            | put(self.min_aniso_walk as u32, 11, 1)
            | put(self.lod_bias as u32, 12, 10)
            | put(self.grad_exp_adjust_h as u32, 22, 5)
            | put(self.grad_exp_adjust_v as u32, 27, 5);
        d[5] = put(self.border_color as u32, 0, 2)
            | put(self.force_bc_w_to_max as u32, 2, 1)
            | put(self.tri_clamp, 3, 2)
            | put(self.aniso_bias as u32, 5, 4)
            | put(self.dimension as u32, 9, 2)
            | put(self.packed_mips as u32, 11, 1)
            | put(self.mip_page, 12, 20);
        d
    }

    /// A fetch constant with neutral defaults, mainly for building constants in host code/tests.
    pub fn new_2d(format: TextureFormat, width: u32, height: u32) -> Self {
        TextureFetchConstant {
            fetch_type: FetchConstantType::Texture,
            sign: [TextureSign::Unsigned; 4],
            clamp: [ClampMode::Repeat; 3],
            pitch_div_32: width.div_ceil(32),
            tiled: false,
            format,
            endianness: Endian::None,
            request_size: 0,
            stacked: false,
            nearest_clamp_policy: false,
            base_page: 0,
            width,
            height,
            depth: 1,
            num_format: NumFormat::Fraction,
            swizzle: [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W],
            exp_adjust: 0,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mip_filter: TextureFilter::Point,
            aniso_filter: AnisoFilter::Disabled,
            arbitrary_filter: ArbitraryFilter::Sym2x4,
            border_size: false,
            vol_mag_filter: TextureFilter::Point,
            vol_min_filter: TextureFilter::Point,
            mip_min_level: 0,
            mip_max_level: 0,
            mag_aniso_walk: false,
            min_aniso_walk: false,
            lod_bias: 0,
            grad_exp_adjust_h: 0,
            grad_exp_adjust_v: 0,
            border_color: BorderColor::AbgrBlack,
            force_bc_w_to_max: false,
            tri_clamp: 0,
            aniso_bias: 0,
            dimension: DataDimension::D2OrStacked,
            packed_mips: false,
            mip_page: 0,
        }
    }

    pub fn swizzle_bits(&self) -> u32 {
        self.swizzle.iter().enumerate().fold(0, |acc, (i, s)| acc | (s.to_bits() << (i * 3)))
    }

    pub fn base_address(&self) -> u32 {
        self.base_page << 12
    }

    pub fn mip_address(&self) -> u32 {
        self.mip_page << 12
    }

    pub fn lod_bias_f32(&self) -> f32 {
        self.lod_bias as f32 / 32.0
    }

    /// Clamp modes normalized for the dimension: unused axes and cube maps clamp to edge.
    pub fn clamp_modes_for_dimension(&self) -> [ClampMode; 3] {
        let edge = ClampMode::ClampToEdge;
        match self.dimension {
            DataDimension::D1 => [self.clamp[0], edge, edge],
            DataDimension::D2OrStacked => [self.clamp[0], self.clamp[1], edge],
            DataDimension::D3 => self.clamp,
            DataDimension::Cube => [edge; 3],
        }
    }

    /// Signs after applying the swizzle (Xenia's `SwizzleSigns`): which sign each destination
    /// component needs. Constant 0/1 components are `Unsigned`, unless every non-constant
    /// component is signed (or, with only constants, every format component is signed).
    pub fn swizzled_signs(&self) -> [TextureSign; 4] {
        let mut out = [TextureSign::Unsigned; 4];
        let mut constant = [false; 4];
        let (mut any_signed, mut any_not_signed) = (false, false);
        for (i, s) in self.swizzle.iter().enumerate() {
            match s.component() {
                Some(c) => {
                    out[i] = self.sign[c];
                    if self.sign[c] == TextureSign::Signed {
                        any_signed = true;
                    } else {
                        any_not_signed = true;
                    }
                }
                None => constant[i] = true,
            }
        }
        let constants_signed = if constant.iter().all(|&c| c) {
            self.sign.iter().all(|&s| s == TextureSign::Signed)
        } else {
            any_signed && !any_not_signed
        };
        if constants_signed {
            for i in 0..4 {
                if constant[i] {
                    out[i] = TextureSign::Signed;
                }
            }
        }
        out
    }

    /// Which host views the swizzled components need: (unsigned view, signed view).
    pub fn needed_views(&self) -> (bool, bool) {
        let signs = self.swizzled_signs();
        let signed = signs.contains(&TextureSign::Signed);
        let unsigned = signs.iter().any(|&s| s != TextureSign::Signed);
        (unsigned, signed)
    }
}

/// Vertex fetch constant (two dwords; three of them share one 6-dword fetch slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexFetchConstant {
    pub fetch_type: FetchConstantType,
    /// Address in dwords (30 bits).
    pub address_dwords: u32,
    pub endian: Endian,
    /// Size in dwords (24 bits).
    pub size_dwords: u32,
}

impl VertexFetchConstant {
    pub fn decode(d: [u32; 2]) -> Self {
        VertexFetchConstant {
            fetch_type: FetchConstantType::from_bits(d[0]),
            address_dwords: d[0] >> 2,
            endian: Endian::from_bits(d[1]),
            size_dwords: bits(d[1], 2, 24),
        }
    }

    pub fn encode(&self) -> [u32; 2] {
        [
            put(self.fetch_type as u32, 0, 2) | (self.address_dwords << 2),
            put(self.endian as u32, 0, 2) | put(self.size_dwords, 2, 24),
        ]
    }

    pub fn address_bytes(&self) -> u32 {
        self.address_dwords << 2
    }

    pub fn size_bytes(&self) -> u32 {
        self.size_dwords << 2
    }
}

/// Interpretation of one 6-dword fetch constant slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchSlot {
    Texture(TextureFetchConstant),
    Vertex([VertexFetchConstant; 3]),
}

impl FetchSlot {
    /// The type field of dword 0 decides the interpretation of the whole slot.
    pub fn decode(d: [u32; 6]) -> Self {
        match FetchConstantType::from_bits(d[0]) {
            FetchConstantType::Texture | FetchConstantType::InvalidTexture => {
                FetchSlot::Texture(TextureFetchConstant::decode(d))
            }
            _ => FetchSlot::Vertex(vertex_fetch_constants(d)),
        }
    }
}

/// The three vertex fetch constants stored in a 6-dword slot.
pub fn vertex_fetch_constants(d: [u32; 6]) -> [VertexFetchConstant; 3] {
    [
        VertexFetchConstant::decode([d[0], d[1]]),
        VertexFetchConstant::decode([d[2], d[3]]),
        VertexFetchConstant::decode([d[4], d[5]]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_known_dxt5_header() {
        // A 512x512 tiled k_DXT4_5 texture with packed mips and 10 levels.
        let f = TextureFetchConstant::decode([
            0x8400_0002,
            0x0000_0054,
            0x003F_E1FF,
            0x0000_0D10,
            0x0000_0240,
            0x0004_0A00,
        ]);
        assert_eq!(f.fetch_type, FetchConstantType::Texture);
        assert!(f.tiled);
        assert_eq!(f.pitch_div_32, 16);
        assert_eq!(f.format, TextureFormat::k_DXT4_5);
        assert_eq!(f.endianness, Endian::K8in16);
        assert_eq!((f.width, f.height), (512, 512));
        assert_eq!(f.swizzle, [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W]);
        assert_eq!((f.mip_min_level, f.mip_max_level), (0, 9));
        assert_eq!(f.dimension, DataDimension::D2OrStacked);
        assert!(f.packed_mips);
        assert_eq!(f.mip_page, 0x40);
        assert_eq!(f.mip_address(), 0x40000);
        assert_eq!(f.encode(), [0x8400_0002, 0x0000_0054, 0x003F_E1FF, 0x0000_0D10, 0x0000_0240, 0x0004_0A00]);
    }

    #[test]
    fn field_positions() {
        let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, 1280, 720);
        f.sign = [TextureSign::Signed, TextureSign::UnsignedBiased, TextureSign::Gamma, TextureSign::Unsigned];
        f.clamp = [ClampMode::MirrorClampToBorder, ClampMode::ClampToEdge, ClampMode::MirroredRepeat];
        f.exp_adjust = -3;
        f.lod_bias = -100;
        f.grad_exp_adjust_h = -16;
        f.grad_exp_adjust_v = 15;
        f.aniso_bias = -8;
        f.border_color = BorderColor::AcbcryBlack;
        f.aniso_filter = AnisoFilter::Max16to1;
        f.mip_filter = TextureFilter::BaseMap;
        f.mip_min_level = 3;
        f.mip_max_level = 12;
        f.base_page = 0xFFFFF;
        f.mip_page = 0x12345;
        f.endianness = Endian::K8in32;
        f.tiled = true;
        f.stacked = true;
        f.depth = 64;
        f.num_format = NumFormat::Integer;
        f.swizzle = [SwizzleSource::Z, SwizzleSource::Y, SwizzleSource::X, SwizzleSource::One];
        let d = f.encode();
        assert_eq!(d[0] & 3, 2);
        assert_eq!((d[0] >> 2) & 3, 1);
        assert_eq!((d[0] >> 10) & 7, 7);
        assert_eq!(d[0] >> 31, 1);
        assert_eq!((d[0] >> 22) & 0x1FF, 40);
        assert_eq!(d[1] & 63, 6);
        assert_eq!((d[1] >> 6) & 3, 2);
        assert_eq!(d[1] >> 12, 0xFFFFF);
        assert_eq!(d[2] & 0x1FFF, 1279);
        assert_eq!((d[2] >> 13) & 0x1FFF, 719);
        assert_eq!(d[2] >> 26, 63);
        assert_eq!((d[3] >> 13) & 63, 61);
        assert_eq!((d[4] >> 12) & 0x3FF, (-100i32 as u32) & 0x3FF);
        assert_eq!(d[5] >> 12, 0x12345);
        let back = TextureFetchConstant::decode(d);
        assert_eq!(back, f);
        assert_eq!(back.lod_bias_f32(), -3.125);
    }

    #[test]
    fn size_per_dimension() {
        let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8, 1, 1);
        f.dimension = DataDimension::D3;
        f.width = 2048;
        f.height = 1024;
        f.depth = 1024;
        let d = f.encode();
        assert_eq!(d[2], 2047 | (1023 << 11) | (1023 << 22));
        assert_eq!(TextureFetchConstant::decode(d), f);
        f.dimension = DataDimension::D1;
        f.width = 1 << 24;
        f.height = 1;
        f.depth = 1;
        let d = f.encode();
        assert_eq!(d[2], 0xFF_FFFF);
        assert_eq!(TextureFetchConstant::decode(d), f);
    }

    #[test]
    fn swizzled_signs_rules() {
        let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, 4, 4);
        f.sign = [TextureSign::Signed, TextureSign::Signed, TextureSign::Unsigned, TextureSign::Gamma];
        f.swizzle = [SwizzleSource::W, SwizzleSource::X, SwizzleSource::Zero, SwizzleSource::Y];
        assert_eq!(
            f.swizzled_signs(),
            [TextureSign::Gamma, TextureSign::Signed, TextureSign::Unsigned, TextureSign::Signed]
        );
        assert_eq!(f.needed_views(), (true, true));
        f.swizzle = [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::One, SwizzleSource::Zero];
        assert_eq!(f.swizzled_signs(), [TextureSign::Signed; 4]);
        assert_eq!(f.needed_views(), (false, true));
        f.swizzle = [SwizzleSource::Zero; 4];
        assert_eq!(f.swizzled_signs(), [TextureSign::Unsigned; 4]);
        f.sign = [TextureSign::Signed; 4];
        assert_eq!(f.swizzled_signs(), [TextureSign::Signed; 4]);
    }

    #[test]
    fn vertex_fetch() {
        let v = VertexFetchConstant::decode([0x1234_5673, 0x0000_0102]);
        assert_eq!(v.fetch_type, FetchConstantType::Vertex);
        assert_eq!(v.address_bytes(), 0x1234_5670);
        assert_eq!(v.endian, Endian::K8in32);
        assert_eq!(v.size_dwords, 0x40);
        assert_eq!(v.size_bytes(), 0x100);
        assert_eq!(v.encode(), [0x1234_5673, 0x0000_0102]);
        match FetchSlot::decode([0x1000_0003, 2, 0x2000_0003, 6, 0, 0]) {
            FetchSlot::Vertex(v) => {
                assert_eq!(v[1].address_bytes(), 0x2000_0000);
                assert_eq!(v[1].size_dwords, 1);
            }
            FetchSlot::Texture(_) => panic!("expected vertex slot"),
        }
        assert!(matches!(FetchSlot::decode([2, 0, 0, 0, 0, 0]), FetchSlot::Texture(_)));
    }

    #[test]
    fn clamp_for_dimension() {
        let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8, 4, 4);
        f.clamp = [ClampMode::Repeat, ClampMode::MirroredRepeat, ClampMode::ClampToBorder];
        assert_eq!(
            f.clamp_modes_for_dimension(),
            [ClampMode::Repeat, ClampMode::MirroredRepeat, ClampMode::ClampToEdge]
        );
        f.dimension = DataDimension::Cube;
        assert_eq!(f.clamp_modes_for_dimension(), [ClampMode::ClampToEdge; 3]);
        f.dimension = DataDimension::D3;
        assert_eq!(f.clamp_modes_for_dimension(), f.clamp);
    }
}
