//! Xenos hardware enumerations, constants and small numeric helpers.

/// Defines an enum decoded from a register bitfield that has values without a known meaning.
/// Unknown values are preserved in `Unknown(raw)` so decoding is lossless.
macro_rules! partial_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $var:ident = $val:expr),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vm])* $var,)*
            /// A value with no known hardware meaning (raw field value).
            Unknown(u32),
        }
        impl $name {
            /// Decodes a raw field value.
            pub const fn from_raw(raw: u32) -> Self {
                match raw {
                    $($val => Self::$var,)*
                    other => Self::Unknown(other),
                }
            }
            /// Raw field value.
            pub const fn raw(self) -> u32 {
                match self {
                    $(Self::$var => $val,)*
                    Self::Unknown(v) => v,
                }
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::from_raw(0)
            }
        }
    };
}

/// Defines an enum whose variants cover every value of a `bits`-wide field.
macro_rules! total_enum {
    ($(#[$m:meta])* $name:ident : $bits:literal { $(#[$fm:meta])* $first_var:ident = $first_val:expr $(, $(#[$vm:meta])* $var:ident = $val:expr)* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub enum $name {
            $(#[$fm])*
            #[default]
            $first_var,
            $($(#[$vm])* $var,)*
        }
        impl $name {
            /// Decodes a raw field value (bits above the field width are ignored).
            pub const fn from_raw(raw: u32) -> Self {
                match raw & ((1u32 << $bits) - 1) {
                    $($val => Self::$var,)*
                    _ => Self::$first_var,
                }
            }
            /// Raw field value.
            pub const fn raw(self) -> u32 {
                match self {
                    Self::$first_var => $first_val,
                    $(Self::$var => $val,)*
                }
            }
        }
    };
}

total_enum! {
    /// Shader stage selector used by IM_LOAD packets.
    ShaderType: 1 { Vertex = 0, Pixel = 1 }
}

partial_enum! {
    /// VGT_DRAW_INITIATOR::PRIM_TYPE.
    ///
    /// Values 0x10..=0x12 are reused for patch primitives when tessellation is enabled
    /// (see [`PrimitiveType::LINE_PATCH`] and friends).
    PrimitiveType {
        None = 0x00,
        PointList = 0x01,
        LineList = 0x02,
        LineStrip = 0x03,
        TriangleList = 0x04,
        TriangleFan = 0x05,
        TriangleStrip = 0x06,
        TriangleWithWFlags = 0x07,
        /// Three vertices per rectangle; the fourth is derived as v1 + v2 - v0.
        RectangleList = 0x08,
        LineLoop = 0x0C,
        QuadList = 0x0D,
        QuadStrip = 0x0E,
        /// Convex polygon, rasterized like a triangle fan.
        Polygon = 0x0F,
        Copy2DRectListV0 = 0x10,
        Copy2DRectListV1 = 0x11,
        Copy2DRectListV2 = 0x12,
        Copy2DRectListV3 = 0x13,
        FillRectList2D = 0x14,
        LineStrip2D = 0x15,
        TriStrip2D = 0x16,
    }
}

impl PrimitiveType {
    /// Line patch (tessellation), same value as `Copy2DRectListV0`.
    pub const LINE_PATCH: Self = Self::Copy2DRectListV0;
    /// Triangle patch (tessellation), same value as `Copy2DRectListV1`.
    pub const TRIANGLE_PATCH: Self = Self::Copy2DRectListV1;
    /// Quad patch (tessellation), same value as `Copy2DRectListV2`.
    pub const QUAD_PATCH: Self = Self::Copy2DRectListV2;
    /// First primitive type that forces the explicit major mode.
    pub const EXPLICIT_MAJOR_MODE_FORCE_START: u32 = 0x10;

    /// Whether the primitive is a line type (draw_util IsPrimitiveLine).
    pub fn is_line(self, tessellation_enabled: bool) -> bool {
        if tessellation_enabled && self == Self::LINE_PATCH {
            return true;
        }
        matches!(self, Self::LineList | Self::LineStrip | Self::LineLoop | Self::LineStrip2D)
    }

    /// Whether the primitive is polygonal - has faces and is affected by culling, fill mode and
    /// front/back polygon offset (draw_util IsPrimitivePolygonal). Rectangle lists are deliberately
    /// treated as non-polygonal, like in Xenia.
    pub fn is_polygonal(self, tessellation_enabled: bool) -> bool {
        if tessellation_enabled && (self == Self::TRIANGLE_PATCH || self == Self::QUAD_PATCH) {
            return true;
        }
        matches!(
            self,
            Self::TriangleList
                | Self::TriangleFan
                | Self::TriangleStrip
                | Self::TriangleWithWFlags
                | Self::QuadList
                | Self::QuadStrip
                | Self::Polygon
        )
    }

    /// Whether the topology is a "strip"-like one where primitive reset (VGT multi-primitive index
    /// buffer reset) has an effect.
    pub fn supports_primitive_reset(self) -> bool {
        matches!(
            self,
            Self::LineStrip
                | Self::TriangleFan
                | Self::TriangleStrip
                | Self::LineLoop
                | Self::QuadStrip
                | Self::Polygon
                | Self::LineStrip2D
                | Self::TriStrip2D
        )
    }
}

total_enum! {
    /// 32-bit endian swap mode (VGT_DMA_SIZE::SWAP_MODE, vertex fetch constants).
    Endian: 2 {
        None = 0,
        /// Swap bytes within each 16-bit half.
        Swap8In16 = 1,
        /// Full 32-bit byte swap.
        Swap8In32 = 2,
        /// Swap the 16-bit halves.
        Swap16In32 = 3,
    }
}

partial_enum! {
    /// 128-bit endian swap mode (RB_COPY_DEST_INFO::COPY_DEST_ENDIAN, memexport).
    Endian128 {
        None = 0,
        Swap8In16 = 1,
        Swap8In32 = 2,
        Swap16In32 = 3,
        Swap8In64 = 4,
        Swap8In128 = 5,
    }
}

total_enum! {
    /// Index element size.
    IndexFormat: 1 { Int16 = 0, Int32 = 1 }
}

impl IndexFormat {
    /// Size of one index in bytes.
    pub const fn size_bytes(self) -> u32 {
        match self {
            Self::Int16 => 2,
            Self::Int32 => 4,
        }
    }
}

partial_enum! {
    /// VGT_DRAW_INITIATOR::SOURCE_SELECT.
    SourceSelect {
        /// Indices fetched from memory (VGT_DMA_BASE / VGT_DMA_SIZE).
        Dma = 0,
        /// Indices embedded in the command stream (VGT_IMMED_DATA).
        Immediate = 1,
        /// Sequential indices 0..num_indices.
        AutoIndex = 2,
    }
}

partial_enum! {
    /// VGT_DRAW_INITIATOR::MAJOR_MODE.
    MajorMode { Implicit = 0, Explicit = 1 }
}

/// Whether VGT_OUTPUT_PATH_CNTL / VGT_HOS_CNTL are honored for the draw.
pub fn is_major_mode_explicit(major_mode: MajorMode, primitive_type: PrimitiveType) -> bool {
    major_mode != MajorMode::Implicit || primitive_type.raw() >= PrimitiveType::EXPLICIT_MAJOR_MODE_FORCE_START
}

partial_enum! {
    /// Surface number format (RB_COPY_DEST_INFO::COPY_DEST_NUMBER).
    SurfaceNumberFormat {
        UnsignedRepeatingFraction = 0,
        SignedRepeatingFraction = 1,
        UnsignedInteger = 2,
        SignedInteger = 3,
        Float = 7,
    }
}

partial_enum! {
    /// RB_SURFACE_INFO::MSAA_SAMPLES.
    MsaaSamples { X1 = 0, X2 = 1, X4 = 2 }
}

impl MsaaSamples {
    /// Sample count (1, 2 or 4), or `None` for invalid values.
    pub const fn count(self) -> Option<u32> {
        match self {
            Self::X1 => Some(1),
            Self::X2 => Some(2),
            Self::X4 => Some(4),
            Self::Unknown(_) => None,
        }
    }
    /// log2 of the horizontal sample count of a pixel (1 for 4x MSAA).
    pub const fn samples_x_log2(self) -> u32 {
        matches!(self, Self::X4) as u32
    }
    /// log2 of the vertical sample count of a pixel (1 for 2x and 4x MSAA).
    pub const fn samples_y_log2(self) -> u32 {
        matches!(self, Self::X2 | Self::X4) as u32
    }
}

partial_enum! {
    /// Color render target (EDRAM) format, RB_COLOR_INFO::COLOR_FORMAT.
    ColorRenderTargetFormat {
        K8888 = 0,
        K8888Gamma = 1,
        K2101010 = 2,
        /// 7e3 floating-point RGB with 2-bit alpha.
        K2101010Float = 3,
        /// Signed fixed-point -32...32.
        K1616 = 4,
        K16161616 = 5,
        K1616Float = 6,
        K16161616Float = 7,
        K2101010As10101010 = 10,
        K2101010FloatAs16161616 = 12,
        K32Float = 14,
        K3232Float = 15,
    }
}

impl ColorRenderTargetFormat {
    /// Whether each pixel takes 64 bits in EDRAM.
    pub const fn is_64bpp(self) -> bool {
        matches!(self, Self::K16161616 | Self::K16161616Float | Self::K3232Float)
    }
    /// Number of components stored (1, 2 or 4); 0 for unknown formats.
    pub const fn component_count(self) -> u32 {
        match self {
            Self::K8888
            | Self::K8888Gamma
            | Self::K2101010
            | Self::K2101010Float
            | Self::K16161616
            | Self::K16161616Float
            | Self::K2101010As10101010
            | Self::K2101010FloatAs16161616 => 4,
            Self::K1616 | Self::K1616Float | Self::K3232Float => 2,
            Self::K32Float => 1,
            Self::Unknown(_) => 0,
        }
    }
    /// The format the data is actually stored as (aliases collapsed), GetStorageColorFormat.
    pub const fn storage_format(self) -> Self {
        match self {
            Self::K2101010As10101010 => Self::K2101010,
            Self::K2101010FloatAs16161616 => Self::K2101010Float,
            other => other,
        }
    }
}

total_enum! {
    /// Depth render target format, RB_DEPTH_INFO::DEPTH_FORMAT.
    DepthRenderTargetFormat: 1 {
        /// 24-bit unorm depth, 8-bit stencil.
        D24S8 = 0,
        /// 20e4 float depth, 8-bit stencil.
        D24FS8 = 1,
    }
}

total_enum! {
    /// Comparison function for depth, stencil and alpha tests.
    CompareFunction: 3 {
        Never = 0,
        Less = 1,
        Equal = 2,
        LessEqual = 3,
        Greater = 4,
        NotEqual = 5,
        GreaterEqual = 6,
        Always = 7,
    }
}

total_enum! {
    /// Stencil operation.
    StencilOp: 3 {
        Keep = 0,
        Zero = 1,
        Replace = 2,
        IncrementClamp = 3,
        DecrementClamp = 4,
        Invert = 5,
        IncrementWrap = 6,
        DecrementWrap = 7,
    }
}

partial_enum! {
    /// Blend factor, RB_BLENDCONTROL.
    BlendFactor {
        Zero = 0,
        One = 1,
        SrcColor = 4,
        OneMinusSrcColor = 5,
        SrcAlpha = 6,
        OneMinusSrcAlpha = 7,
        DstColor = 8,
        OneMinusDstColor = 9,
        DstAlpha = 10,
        OneMinusDstAlpha = 11,
        ConstantColor = 12,
        OneMinusConstantColor = 13,
        ConstantAlpha = 14,
        OneMinusConstantAlpha = 15,
        SrcAlphaSaturate = 16,
    }
}

partial_enum! {
    /// Blend operation, RB_BLENDCONTROL.
    BlendOp {
        Add = 0,
        Subtract = 1,
        Min = 2,
        Max = 3,
        RevSubtract = 4,
    }
}

partial_enum! {
    /// SQ_PROGRAM_CNTL::VS_EXPORT_MODE.
    VertexShaderExportMode {
        Position1Vector = 0,
        Position2VectorsSprite = 2,
        Position2VectorsEdge = 3,
        Position2VectorsKill = 4,
        Position2VectorsSpriteKill = 5,
        Position2VectorsEdgeKill = 6,
        /// Memory export only, nothing is rasterized.
        Multipass = 7,
    }
}

partial_enum! {
    /// SQ_CONTEXT_MISC::SC_SAMPLE_CNTL.
    SampleControl {
        CentroidsOnly = 0,
        CentersOnly = 1,
        /// Per-interpolator selection via SQ_INTERPOLATOR_CNTL::SAMPLING_PATTERN.
        CentroidsAndCenters = 2,
    }
}

partial_enum! {
    /// VGT_OUTPUT_PATH_CNTL::PATH_SELECT.
    VgtOutputPath { VertexReuse = 0, TessellationEnable = 1, Passthru = 2 }
}

partial_enum! {
    /// VGT_HOS_CNTL::TESS_MODE.
    TessellationMode { Discrete = 0, Continuous = 1, Adaptive = 2 }
}

partial_enum! {
    /// PA_SU_SC_MODE_CNTL::POLY_MODE.
    PolygonModeEnable {
        /// Render filled triangles.
        Disabled = 0,
        /// Use polymode_front_ptype / polymode_back_ptype.
        DualMode = 1,
    }
}

partial_enum! {
    /// PA_SU_SC_MODE_CNTL::POLYMODE_FRONT/BACK_PTYPE.
    PolygonType { Points = 0, Lines = 1, Triangles = 2 }
}

total_enum! {
    /// PA_SU_VTX_CNTL::PIX_CENTER.
    PixelCenter: 1 {
        /// Direct3D 9 convention: pixel centers at integers (needs a half-pixel offset on modern APIs).
        D3DZero = 0,
        /// OpenGL convention: pixel centers at .5.
        OglHalf = 1,
    }
}

total_enum! {
    /// PA_SU_VTX_CNTL::ROUND_MODE.
    VertexRounding: 2 { Truncate = 0, Round = 1, RoundToEven = 2, RoundToOdd = 3 }
}

partial_enum! {
    /// PA_SU_VTX_CNTL::QUANT_MODE.
    VertexQuantization { OneSixteenth = 0, OneEighth = 1, OneQuarter = 2, OneHalf = 3, One = 4 }
}

partial_enum! {
    /// RB_MODECONTROL::EDRAM_MODE.
    EdramMode {
        NoOperation = 0,
        /// Normal color + depth rendering.
        ColorDepth = 4,
        /// Depth/stencil only, pixel shader not executed.
        DepthOnly = 5,
        /// Draws trigger an EDRAM resolve.
        Copy = 6,
    }
}

total_enum! {
    /// RB_COPY_CONTROL::COPY_COMMAND.
    CopyCommand: 2 {
        Raw = 0,
        Convert = 1,
        /// Per-component comparison to k_DXT3A_AS_1_1_1_1.
        ConvertTo1111 = 2,
        Null = 3,
    }
}

partial_enum! {
    /// RB_COPY_CONTROL::COPY_SAMPLE_SELECT.
    CopySampleSelect {
        S0 = 0,
        S1 = 1,
        S2 = 2,
        S3 = 3,
        /// Average of samples 0 and 1.
        S01 = 4,
        /// Average of samples 2 and 3.
        S23 = 5,
        /// Average of all four samples.
        S0123 = 6,
    }
}

impl CopySampleSelect {
    /// Whether a single sample (no averaging) is selected.
    pub fn is_single(self) -> bool {
        matches!(self, Self::S0 | Self::S1 | Self::S2 | Self::S3)
    }
}

total_enum! {
    /// Fetch constant type field.
    FetchConstantType: 2 { InvalidTexture = 0, InvalidVertex = 1, Texture = 2, Vertex = 3 }
}

total_enum! {
    /// Texture / resolve destination format (6 bits). Also used as RB_COPY_DEST_INFO::COPY_DEST_FORMAT
    /// (the ColorFormat subset).
    TextureFormat: 6 {
        K1Reverse = 0, K1 = 1, K8 = 2, K1555 = 3, K565 = 4, K655 = 5, K8888 = 6, K2101010 = 7,
        K8A = 8, K8B = 9, K88 = 10, KCrY1CbY0Rep = 11, KY1CrY0CbRep = 12, K1616Edram = 13,
        K8888A = 14, K4444 = 15, K101111 = 16, K111110 = 17, Dxt1 = 18, Dxt23 = 19, Dxt45 = 20,
        K16161616Edram = 21, K248 = 22, K248Float = 23, K16 = 24, K1616 = 25, K16161616 = 26,
        K16Expand = 27, K1616Expand = 28, K16161616Expand = 29, K16Float = 30, K1616Float = 31,
        K16161616Float = 32, K32 = 33, K3232 = 34, K32323232 = 35, K32Float = 36, K3232Float = 37,
        K32323232Float = 38, K32As8 = 39, K32As88 = 40, K16Mpeg = 41, K1616Mpeg = 42,
        K8Interlaced = 43, K32As8Interlaced = 44, K32As88Interlaced = 45, K16Interlaced = 46,
        K16MpegInterlaced = 47, K1616MpegInterlaced = 48, Dxn = 49, K8888As16161616 = 50,
        Dxt1As16161616 = 51, Dxt23As16161616 = 52, Dxt45As16161616 = 53, K2101010As16161616 = 54,
        K101111As16161616 = 55, K111110As16161616 = 56, K323232Float = 57, Dxt3a = 58, Dxt5a = 59,
        Ctx1 = 60, Dxt3aAs1111 = 61, K8888GammaEdram = 62, K2101010FloatEdram = 63,
    }
}

/// Category of a texture format (texture_info FormatType).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatType {
    Uncompressed,
    Compressed,
    /// Can be the destination of a color resolve.
    Resolvable,
}

/// Static block layout information about a [`TextureFormat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    pub format_type: FormatType,
    pub block_width: u32,
    pub block_height: u32,
    /// Bits per pixel (per texel, not per block).
    pub bits_per_pixel: u32,
}

impl FormatInfo {
    /// Bytes per block.
    pub const fn bytes_per_block(&self) -> u32 {
        self.block_width * self.block_height * self.bits_per_pixel / 8
    }
}

impl TextureFormat {
    /// Block layout information (texture_info_formats.inl).
    pub const fn info(self) -> FormatInfo {
        use FormatType::{Compressed as C, Resolvable as R, Uncompressed as U};
        let (t, bw, bh, bpp) = match self {
            Self::K1Reverse | Self::K1 => (U, 1, 1, 1),
            Self::K8 | Self::K8A | Self::K8B => (R, 1, 1, 8),
            Self::K1555 | Self::K565 | Self::K655 | Self::K88 | Self::K4444 | Self::K16 | Self::K16Float => {
                (R, 1, 1, 16)
            }
            Self::K8888
            | Self::K2101010
            | Self::K8888A
            | Self::K101111
            | Self::K111110
            | Self::K1616
            | Self::K1616Float
            | Self::K32Float
            | Self::K8888As16161616
            | Self::K2101010As16161616
            | Self::K101111As16161616
            | Self::K111110As16161616 => (R, 1, 1, 32),
            Self::K16161616 | Self::K16161616Float | Self::K3232Float => (R, 1, 1, 64),
            Self::K32323232Float => (R, 1, 1, 128),
            Self::KCrY1CbY0Rep | Self::KY1CrY0CbRep => (C, 2, 1, 16),
            Self::K1616Edram | Self::K248 | Self::K248Float | Self::K1616Expand | Self::K32 => (U, 1, 1, 32),
            Self::K1616Mpeg | Self::K1616MpegInterlaced | Self::K8888GammaEdram | Self::K2101010FloatEdram => {
                (U, 1, 1, 32)
            }
            Self::K16161616Edram | Self::K16161616Expand | Self::K3232 => (U, 1, 1, 64),
            Self::K32323232 => (U, 1, 1, 128),
            Self::K323232Float => (U, 1, 1, 96),
            Self::K16Expand | Self::K16Mpeg | Self::K16Interlaced | Self::K16MpegInterlaced => (U, 1, 1, 16),
            Self::K8Interlaced => (U, 1, 1, 8),
            Self::Dxt1 | Self::Dxt1As16161616 | Self::Dxt3a | Self::Dxt5a | Self::Ctx1 | Self::Dxt3aAs1111 => {
                (C, 4, 4, 4)
            }
            Self::Dxt23 | Self::Dxt45 | Self::Dxn | Self::Dxt23As16161616 | Self::Dxt45As16161616 => (C, 4, 4, 8),
            Self::K32As8 | Self::K32As8Interlaced => (C, 4, 1, 8),
            Self::K32As88 => (C, 2, 1, 16),
            Self::K32As88Interlaced => (C, 1, 1, 16),
        };
        FormatInfo { format_type: t, block_width: bw, block_height: bh, bits_per_pixel: bpp }
    }
}

impl DepthRenderTargetFormat {
    /// Texture format a depth resolve writes (DepthRenderTargetToTextureFormat).
    pub const fn texture_format(self) -> TextureFormat {
        match self {
            Self::D24S8 => TextureFormat::K248,
            Self::D24FS8 => TextureFormat::K248Float,
        }
    }
}

/// Whether a color render target format can be copied bit-for-bit into a resolve destination
/// format (IsColorResolveFormatBitwiseEquivalent).
pub fn is_color_resolve_format_bitwise_equivalent(rt: ColorRenderTargetFormat, dest: TextureFormat) -> bool {
    use ColorRenderTargetFormat as C;
    use TextureFormat as T;
    match rt {
        C::K8888 | C::K8888Gamma => matches!(dest, T::K8888 | T::K8888A | T::K8888As16161616),
        C::K2101010 | C::K2101010As10101010 => matches!(dest, T::K2101010 | T::K2101010As16161616),
        C::K1616Float => dest == T::K1616Float,
        C::K16161616Float => dest == T::K16161616Float,
        C::K32Float => dest == T::K32Float,
        C::K3232Float => dest == T::K3232Float,
        _ => false,
    }
}

/// Number of significant bits of a vertex index; the upper 8 bits of 32-bit indices are ignored.
pub const VERTEX_INDEX_BITS: u32 = 24;
/// Mask of the significant vertex index bits.
pub const VERTEX_INDEX_MASK: u32 = (1 << VERTEX_INDEX_BITS) - 1;
/// Maximum number of color render targets.
pub const MAX_COLOR_RENDER_TARGETS: u32 = 4;
/// Maximum number of interpolators.
pub const MAX_INTERPOLATORS: u32 = 16;
/// Maximum number of shader temporary registers.
pub const MAX_SHADER_TEMP_REGISTERS: u32 = 64;
/// Polygon offset scale unit (the scale registers are in 1/16 pixel subpixels).
pub const POLYGON_OFFSET_SCALE_SUBPIXEL_UNIT: f32 = 1.0 / 16.0;
/// Maximum 2D texture / render target width or height.
pub const TEXTURE_2D_MAX_WIDTH_HEIGHT: u32 = 8192;
/// log2 of the resolve rectangle alignment in pixels.
pub const RESOLVE_ALIGNMENT_PIXELS_LOG2: u32 = 3;
/// Resolve rectangles are aligned to 8x8 pixels.
pub const RESOLVE_ALIGNMENT_PIXELS: u32 = 1 << RESOLVE_ALIGNMENT_PIXELS_LOG2;
/// Largest resolve width or height.
pub const MAX_RESOLVE_SIZE: u32 = (1 << 14) - RESOLVE_ALIGNMENT_PIXELS;

/// Xenos sample positions (1/16 pixel units, relative to the pixel center), 2x MSAA, guest order.
pub const XENOS_SAMPLE_POSITIONS_2X: [[i8; 2]; 2] = [[-4, 4], [4, -4]];
/// Xenos sample positions (1/16 pixel units, relative to the pixel center), 4x MSAA, guest order.
pub const XENOS_SAMPLE_POSITIONS_4X: [[i8; 2]; 4] = [[-2, -2], [2, 2], [-6, 6], [6, -6]];

/// Applies a GPU 16-bit endian swap (only `Swap8In16` affects 16-bit values).
pub const fn gpu_swap_u16(value: u16, endian: Endian) -> u16 {
    match endian {
        Endian::Swap8In16 => value.swap_bytes(),
        _ => value,
    }
}

/// Applies a GPU 32-bit endian swap.
pub const fn gpu_swap_u32(value: u32, endian: Endian) -> u32 {
    match endian {
        Endian::None => value,
        Endian::Swap8In16 => ((value << 8) & 0xFF00_FF00) | ((value >> 8) & 0x00FF_00FF),
        Endian::Swap8In32 => value.swap_bytes(),
        Endian::Swap16In32 => value.rotate_left(16),
    }
}

/// Applies a GPU 32-bit endian swap to the bit pattern of a float.
pub fn gpu_swap_f32(value: f32, endian: Endian) -> f32 {
    f32::from_bits(gpu_swap_u32(value.to_bits(), endian))
}

/// Converts a non-negative float to the Xenos 20e4 float24 depth encoding.
pub fn float32_to_20e4(value: f32, round_to_nearest_even: bool) -> u32 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    let mut bits = value.to_bits();
    if bits >= 0x3FFF_FFF8 {
        return 0xFF_FFFF;
    }
    if bits < 0x3880_0000 {
        let shift = (113 - (bits >> 23)).min(24);
        bits = (0x80_0000 | (bits & 0x7F_FFFF)) >> shift;
    } else {
        bits = bits.wrapping_add(0xC800_0000);
    }
    if round_to_nearest_even {
        bits = bits.wrapping_add(3 + ((bits >> 3) & 1));
    }
    (bits >> 3) & 0xFF_FFFF
}

/// Converts a Xenos 20e4 float24 depth value to f32.
pub fn float20e4_to_32(f24: u32) -> f32 {
    let f24 = f24 & 0xFF_FFFF;
    if f24 == 0 {
        return 0.0;
    }
    let mut mantissa = f24 & 0xF_FFFF;
    let mut exponent = f24 >> 20;
    if exponent == 0 {
        let lzcnt = mantissa.leading_zeros() - (32 - 21);
        exponent = (1 - lzcnt as i32) as u32;
        mantissa = (mantissa << lzcnt) & 0xF_FFFF;
    }
    f32::from_bits((exponent.wrapping_add(112) << 23) | (mantissa << 3))
}

/// Converts a float to the Direct3D 11 16.8 fixed-point representation used for rasterization
/// coordinates (round to nearest even, with saturation).
pub fn float_to_d3d11_fixed16p8(value: f32) -> i32 {
    if value.is_nan() || value.abs() < 1.0 / 512.0 {
        return 0;
    }
    if value >= 32768.0 - 1.0 / 256.0 {
        return (1 << 23) - 1;
    }
    if value <= -32768.0 {
        return -32768 * 256;
    }
    (f64::from(value) * 256.0).round_ties_even() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_enum_roundtrip() {
        for raw in 0..64 {
            assert_eq!(PrimitiveType::from_raw(raw).raw(), raw);
            assert_eq!(BlendFactor::from_raw(raw).raw(), raw);
            assert_eq!(TextureFormat::from_raw(raw).raw(), raw);
        }
        assert_eq!(PrimitiveType::from_raw(0x09), PrimitiveType::Unknown(9));
        assert_eq!(PrimitiveType::TRIANGLE_PATCH.raw(), 0x11);
        assert_eq!(MsaaSamples::from_raw(3).count(), None);
        assert_eq!(MsaaSamples::X4.count(), Some(4));
    }

    #[test]
    fn total_enum_masks() {
        assert_eq!(CompareFunction::from_raw(0b1011), CompareFunction::LessEqual);
        assert_eq!(Endian::from_raw(3), Endian::Swap16In32);
        assert_eq!(StencilOp::DecrementWrap.raw(), 7);
    }

    #[test]
    fn primitive_classification() {
        assert!(PrimitiveType::LineLoop.is_line(false));
        assert!(PrimitiveType::LINE_PATCH.is_line(true));
        assert!(!PrimitiveType::LINE_PATCH.is_line(false));
        assert!(PrimitiveType::Polygon.is_polygonal(false));
        assert!(!PrimitiveType::RectangleList.is_polygonal(false));
        assert!(PrimitiveType::QUAD_PATCH.is_polygonal(true));
        assert!(is_major_mode_explicit(MajorMode::Implicit, PrimitiveType::Copy2DRectListV0));
        assert!(!is_major_mode_explicit(MajorMode::Implicit, PrimitiveType::TriangleList));
    }

    #[test]
    fn swaps() {
        assert_eq!(gpu_swap_u32(0x1122_3344, Endian::None), 0x1122_3344);
        assert_eq!(gpu_swap_u32(0x1122_3344, Endian::Swap8In16), 0x2211_4433);
        assert_eq!(gpu_swap_u32(0x1122_3344, Endian::Swap8In32), 0x4433_2211);
        assert_eq!(gpu_swap_u32(0x1122_3344, Endian::Swap16In32), 0x3344_1122);
        assert_eq!(gpu_swap_u16(0x1122, Endian::Swap8In16), 0x2211);
        assert_eq!(gpu_swap_u16(0x1122, Endian::Swap8In32), 0x1122);
        assert_eq!(gpu_swap_f32(f32::from_bits(0x0000_803F), Endian::Swap8In32), 1.0);
    }

    #[test]
    fn float24() {
        assert_eq!(float32_to_20e4(0.0, true), 0);
        assert_eq!(float32_to_20e4(-1.0, true), 0);
        assert_eq!(float32_to_20e4(2.0, true), 0xFF_FFFF);
        // 1.0 = exponent 15 (bias 15), mantissa 0.
        assert_eq!(float32_to_20e4(1.0, false), 15 << 20);
        assert_eq!(float20e4_to_32(15 << 20), 1.0);
        assert_eq!(float20e4_to_32(0x10_0000), 2f32.powi(-14));
        // Denormal: smallest value.
        assert_eq!(float20e4_to_32(1), 2f32.powi(-34));
        for v in [0.5f32, 0.25, 0.75, 0.123, 1e-5, 0.999] {
            let back = float20e4_to_32(float32_to_20e4(v, true));
            assert!((back - v).abs() <= v * 2f32.powi(-19), "{v} {back}");
        }
    }

    #[test]
    fn fixed16p8() {
        assert_eq!(float_to_d3d11_fixed16p8(0.0), 0);
        assert_eq!(float_to_d3d11_fixed16p8(f32::NAN), 0);
        assert_eq!(float_to_d3d11_fixed16p8(1.0), 256);
        assert_eq!(float_to_d3d11_fixed16p8(-1.5), -384);
        assert_eq!(float_to_d3d11_fixed16p8(1.0 / 512.0), 0); // ties to even
        assert_eq!(float_to_d3d11_fixed16p8(3.0 / 512.0), 2);
        assert_eq!(float_to_d3d11_fixed16p8(1e9), (1 << 23) - 1);
        assert_eq!(float_to_d3d11_fixed16p8(-1e9), -32768 * 256);
    }

    #[test]
    fn format_info() {
        assert_eq!(TextureFormat::K8888.info().bytes_per_block(), 4);
        assert_eq!(TextureFormat::Dxt3aAs1111.info().bytes_per_block(), 8);
        assert_eq!(TextureFormat::K32323232Float.info().bytes_per_block(), 16);
        assert_eq!(TextureFormat::K248.info().format_type, FormatType::Uncompressed);
        assert_eq!(TextureFormat::K565.info().format_type, FormatType::Resolvable);
        assert!(is_color_resolve_format_bitwise_equivalent(ColorRenderTargetFormat::K8888Gamma, TextureFormat::K8888));
        assert!(!is_color_resolve_format_bitwise_equivalent(ColorRenderTargetFormat::K1616, TextureFormat::K1616));
        assert_eq!(ColorRenderTargetFormat::K3232Float.component_count(), 2);
        assert!(ColorRenderTargetFormat::K16161616.is_64bpp());
    }
}
