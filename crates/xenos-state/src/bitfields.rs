//! Typed decoders and encoders for Xenos bitfield registers (layouts from Xenia registers.h).
//!
//! Each struct has `decode(raw)` / `encode()` and implements [`RegisterField`] so it can be read
//! straight from a [`RegisterSource`]. Padding bits are not preserved by `encode`.

use crate::RegisterSource;
use crate::registers as r;
use crate::xenos::*;

/// A register with a typed layout.
pub trait RegisterField: Sized {
    /// Register index of the (first) register using this layout.
    const REGISTER: u32;
    /// Decodes a raw register value.
    fn decode(raw: u32) -> Self;
    /// Encodes the fields back into a raw value (padding bits zero).
    fn encode(&self) -> u32;
    /// Reads and decodes the register at [`Self::REGISTER`].
    fn read(regs: &(impl RegisterSource + ?Sized)) -> Self {
        Self::decode(regs.reg(Self::REGISTER))
    }
}

/// Conversion of a single bitfield to and from its typed value.
pub trait FieldValue: Sized {
    fn from_field(raw: u32, width: u32) -> Self;
    fn to_field(self) -> u32;
}

impl FieldValue for bool {
    fn from_field(raw: u32, _: u32) -> Self {
        raw != 0
    }
    fn to_field(self) -> u32 {
        self as u32
    }
}

impl FieldValue for u32 {
    fn from_field(raw: u32, _: u32) -> Self {
        raw
    }
    fn to_field(self) -> u32 {
        self
    }
}

impl FieldValue for i32 {
    fn from_field(raw: u32, width: u32) -> Self {
        ((raw << (32 - width)) as i32) >> (32 - width)
    }
    fn to_field(self) -> u32 {
        self as u32
    }
}

macro_rules! enum_fields {
    ($($t:ty),*) => {$(
        impl FieldValue for $t {
            fn from_field(raw: u32, _: u32) -> Self {
                <$t>::from_raw(raw)
            }
            fn to_field(self) -> u32 {
                self.raw()
            }
        }
    )*};
}

enum_fields!(
    PrimitiveType,
    Endian,
    Endian128,
    IndexFormat,
    SourceSelect,
    MajorMode,
    SurfaceNumberFormat,
    MsaaSamples,
    ColorRenderTargetFormat,
    DepthRenderTargetFormat,
    CompareFunction,
    StencilOp,
    BlendFactor,
    BlendOp,
    VertexShaderExportMode,
    SampleControl,
    VgtOutputPath,
    TessellationMode,
    PolygonModeEnable,
    PolygonType,
    PixelCenter,
    VertexRounding,
    VertexQuantization,
    EdramMode,
    CopyCommand,
    CopySampleSelect,
    FetchConstantType,
    TextureFormat
);

#[inline]
const fn mask(width: u32) -> u32 {
    if width >= 32 { u32::MAX } else { (1u32 << width) - 1 }
}

macro_rules! bitfield_register {
    (
        $(#[$m:meta])*
        $name:ident = $reg:expr;
        $( $(#[$fm:meta])* $field:ident : $ty:ty = [$shift:expr, $width:expr] ),* $(,)?
    ) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            $( $(#[$fm])* pub $field: $ty, )*
        }

        impl $name {
            /// Decodes a raw register value.
            pub fn decode(raw: u32) -> Self {
                Self { $( $field: <$ty as FieldValue>::from_field((raw >> $shift) & mask($width), $width), )* }
            }
            /// Encodes the fields into a raw register value.
            pub fn encode(&self) -> u32 {
                0 $( | ((FieldValue::to_field(self.$field) & mask($width)) << $shift) )*
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::decode(0)
            }
        }

        impl RegisterField for $name {
            const REGISTER: u32 = $reg;
            fn decode(raw: u32) -> Self {
                $name::decode(raw)
            }
            fn encode(&self) -> u32 {
                $name::encode(self)
            }
        }
    };
}

bitfield_register! {
    /// VGT_DRAW_INITIATOR - written by the PM4 DRAW_INDX / DRAW_INDX_2 packets.
    VgtDrawInitiator = r::VGT_DRAW_INITIATOR;
    prim_type: PrimitiveType = [0, 6],
    source_select: SourceSelect = [6, 2],
    /// Explicit major mode enables VGT_OUTPUT_PATH_CNTL (tessellation).
    major_mode: MajorMode = [8, 2],
    index_size: IndexFormat = [11, 1],
    not_eop: bool = [12, 1],
    /// Number of indices (vertices) to draw.
    num_indices: u32 = [16, 16],
}

bitfield_register! {
    /// VGT_DMA_SIZE - index buffer size and endianness, written by DRAW_INDX for DMA sources.
    VgtDmaSize = r::VGT_DMA_SIZE;
    /// Size of the index buffer in indices.
    num_words: u32 = [0, 24],
    swap_mode: Endian = [30, 2],
}

bitfield_register! {
    /// VGT_MULTI_PRIM_IB_RESET_INDX - primitive restart index (compared with the low 24 bits).
    VgtMultiPrimIbResetIndx = r::VGT_MULTI_PRIM_IB_RESET_INDX;
    reset_indx: u32 = [0, 24],
}

bitfield_register! {
    /// VGT_INDX_OFFSET - added to every vertex index after the reset check, 24-bit wrapping.
    VgtIndxOffset = r::VGT_INDX_OFFSET;
    indx_offset: u32 = [0, 24],
}

bitfield_register! {
    /// VGT_MIN_VTX_INDX - lower clamp of offset vertex indices.
    VgtMinVtxIndx = r::VGT_MIN_VTX_INDX;
    min_indx: u32 = [0, 24],
}

bitfield_register! {
    /// VGT_MAX_VTX_INDX - upper clamp of offset vertex indices (usually 0xFFFF or 0xFFFFFF).
    VgtMaxVtxIndx = r::VGT_MAX_VTX_INDX;
    max_indx: u32 = [0, 24],
}

bitfield_register! {
    /// VGT_OUTPUT_PATH_CNTL - only honored in the explicit major mode.
    VgtOutputPathCntl = r::VGT_OUTPUT_PATH_CNTL;
    path_select: VgtOutputPath = [0, 2],
}

bitfield_register! {
    /// VGT_HOS_CNTL - tessellation mode.
    VgtHosCntl = r::VGT_HOS_CNTL;
    tess_mode: TessellationMode = [0, 2],
}

bitfield_register! {
    /// SQ_PROGRAM_CNTL - shader register counts and export configuration.
    SqProgramCntl = r::SQ_PROGRAM_CNTL;
    /// Vertex shader GPR count minus 1.
    vs_num_reg: u32 = [0, 6],
    /// Pixel shader GPR count minus 1.
    ps_num_reg: u32 = [8, 6],
    vs_resource: bool = [16, 1],
    ps_resource: bool = [17, 1],
    /// Whether the pixel shader receives PsParamGen in interpolator `SQ_CONTEXT_MISC::param_gen_pos`.
    param_gen: bool = [18, 1],
    gen_index_pix: bool = [19, 1],
    /// Number of interpolators exported by the vertex shader minus 1.
    vs_export_count: u32 = [20, 4],
    vs_export_mode: VertexShaderExportMode = [24, 3],
    ps_export_mode: u32 = [27, 4],
    /// Whether the vertex index is loaded into r0.x of the vertex shader.
    gen_index_vtx: bool = [31, 1],
}

bitfield_register! {
    /// SQ_CONTEXT_MISC.
    SqContextMisc = r::SQ_CONTEXT_MISC;
    inst_pred_optimize: bool = [0, 1],
    sc_output_screen_xy: bool = [1, 1],
    sc_sample_cntl: SampleControl = [2, 2],
    /// Interpolator index that receives PsParamGen (screen XY, point sprite UV, face/prim flags).
    param_gen_pos: u32 = [8, 8],
    perfcounter_ref: bool = [16, 1],
    yield_optimize: bool = [17, 1],
    tx_cache_sel: bool = [18, 1],
}

bitfield_register! {
    /// SQ_INTERPOLATOR_CNTL.
    SqInterpolatorCntl = r::SQ_INTERPOLATOR_CNTL;
    /// Per-interpolator flat shading bits.
    param_shade: u32 = [0, 16],
    /// Per-interpolator sample location: 0 centroid, 1 center (when sample control selects per-interpolator).
    sampling_pattern: u32 = [16, 16],
}

bitfield_register! {
    /// SQ_VS_CONST / SQ_PS_CONST - float constant window of a shader stage.
    SqConstRange = r::SQ_VS_CONST;
    base: u32 = [0, 9],
    /// Vec4 count minus 1.
    size: u32 = [12, 9],
}

bitfield_register! {
    /// SQ_VS_PROGRAM / SQ_PS_PROGRAM - instruction memory window (Adreno 200 layout). Xbox 360 Direct3D
    /// locates shaders through PM4 IM_LOAD packets instead; see [`crate::shader`].
    SqProgramBase = r::SQ_VS_PROGRAM;
    base: u32 = [0, 12],
    size: u32 = [12, 12],
}

bitfield_register! {
    /// PA_SU_POINT_SIZE - half width/height in 12.4 fixed point.
    PaSuPointSize = r::PA_SU_POINT_SIZE;
    height: u32 = [0, 16],
    width: u32 = [16, 16],
}

bitfield_register! {
    /// PA_SU_POINT_MINMAX - per-vertex point radius clamp in 12.4 fixed point.
    PaSuPointMinmax = r::PA_SU_POINT_MINMAX;
    min_size: u32 = [0, 16],
    max_size: u32 = [16, 16],
}

bitfield_register! {
    /// PA_SU_LINE_CNTL - half line width in 12.4 fixed point.
    PaSuLineCntl = r::PA_SU_LINE_CNTL;
    width: u32 = [0, 16],
}

bitfield_register! {
    /// PA_SU_SC_MODE_CNTL - culling, fill mode, polygon offset and misc rasterizer controls.
    PaSuScModeCntl = r::PA_SU_SC_MODE_CNTL;
    cull_front: bool = [0, 1],
    cull_back: bool = [1, 1],
    /// false - front face is counter-clockwise, true - clockwise.
    face: bool = [2, 1],
    poly_mode: PolygonModeEnable = [3, 2],
    polymode_front_ptype: PolygonType = [5, 3],
    polymode_back_ptype: PolygonType = [8, 3],
    poly_offset_front_enable: bool = [11, 1],
    poly_offset_back_enable: bool = [12, 1],
    /// Polygon offset for non-polygonal primitives (points, lines, rectangles).
    poly_offset_para_enable: bool = [13, 1],
    msaa_enable: bool = [15, 1],
    /// Whether PA_SC_WINDOW_OFFSET is applied to vertex positions.
    vtx_window_offset_enable: bool = [16, 1],
    provoking_vtx_last: bool = [19, 1],
    persp_corr_dis: bool = [20, 1],
    /// Primitive reset enable.
    multi_prim_ib_ena: bool = [21, 1],
    quad_order_enable: bool = [23, 1],
    sc_one_quad_per_clock: bool = [24, 1],
}

bitfield_register! {
    /// PA_SU_VTX_CNTL.
    PaSuVtxCntl = r::PA_SU_VTX_CNTL;
    pix_center: PixelCenter = [0, 1],
    round_mode: VertexRounding = [1, 2],
    quant_mode: VertexQuantization = [3, 3],
}

bitfield_register! {
    /// PA_CL_CLIP_CNTL.
    PaClClipCntl = r::PA_CL_CLIP_CNTL;
    /// User clip plane enable bits 0..5.
    ucp_ena: u32 = [0, 6],
    ps_ucp_mode: u32 = [14, 2],
    clip_disable: bool = [16, 1],
    ucp_cull_only_ena: bool = [17, 1],
    boundary_edge_flag_ena: bool = [18, 1],
    /// Direct3D clip space (0 <= z <= w) instead of OpenGL (-w <= z <= w).
    dx_clip_space_def: bool = [19, 1],
    dis_clip_err_detect: bool = [20, 1],
    vtx_kill_or: bool = [21, 1],
    xy_nan_retain: bool = [22, 1],
    z_nan_retain: bool = [23, 1],
    w_nan_retain: bool = [24, 1],
}

bitfield_register! {
    /// PA_CL_VTE_CNTL - viewport transform enables.
    PaClVteCntl = r::PA_CL_VTE_CNTL;
    vport_x_scale_ena: bool = [0, 1],
    vport_x_offset_ena: bool = [1, 1],
    vport_y_scale_ena: bool = [2, 1],
    vport_y_offset_ena: bool = [3, 1],
    vport_z_scale_ena: bool = [4, 1],
    vport_z_offset_ena: bool = [5, 1],
    /// Vertex XY are pre-divided by W.
    vtx_xy_fmt: bool = [8, 1],
    vtx_z_fmt: bool = [9, 1],
    /// W is 1/W.
    vtx_w0_fmt: bool = [10, 1],
    perfcounter_ref: bool = [11, 1],
}

bitfield_register! {
    /// PA_SC_SCREEN_SCISSOR_TL / PA_SC_SCREEN_SCISSOR_BR - signed 15-bit corner.
    PaScScreenScissor = r::PA_SC_SCREEN_SCISSOR_TL;
    x: i32 = [0, 15],
    y: i32 = [16, 15],
}

bitfield_register! {
    /// PA_SC_WINDOW_OFFSET - signed 15-bit window offset in pixels.
    PaScWindowOffset = r::PA_SC_WINDOW_OFFSET;
    window_x_offset: i32 = [0, 15],
    window_y_offset: i32 = [16, 15],
}

bitfield_register! {
    /// PA_SC_WINDOW_SCISSOR_TL.
    PaScWindowScissorTl = r::PA_SC_WINDOW_SCISSOR_TL;
    tl_x: u32 = [0, 14],
    tl_y: u32 = [16, 14],
    /// When set, the window offset is not applied to the window scissor.
    window_offset_disable: bool = [31, 1],
}

bitfield_register! {
    /// PA_SC_WINDOW_SCISSOR_BR (exclusive).
    PaScWindowScissorBr = r::PA_SC_WINDOW_SCISSOR_BR;
    br_x: u32 = [0, 14],
    br_y: u32 = [16, 14],
}

bitfield_register! {
    /// PA_SC_VIZ_QUERY - occlusion / visibility query control.
    PaScVizQuery = r::PA_SC_VIZ_QUERY;
    viz_query_ena: bool = [0, 1],
    viz_query_id: u32 = [1, 6],
    kill_pix_post_hi_z: bool = [7, 1],
    kill_pix_post_detail_mask: bool = [8, 1],
}

bitfield_register! {
    /// RB_MODECONTROL.
    RbModecontrol = r::RB_MODECONTROL;
    edram_mode: EdramMode = [0, 3],
}

bitfield_register! {
    /// RB_SURFACE_INFO - surface pitch and MSAA mode shared by all render targets.
    RbSurfaceInfo = r::RB_SURFACE_INFO;
    /// Pitch in pixels.
    surface_pitch: u32 = [0, 14],
    msaa_samples: MsaaSamples = [16, 2],
    hiz_pitch: u32 = [18, 14],
}

bitfield_register! {
    /// RB_COLORCONTROL - alpha test and alpha to coverage.
    RbColorcontrol = r::RB_COLORCONTROL;
    alpha_func: CompareFunction = [0, 3],
    alpha_test_enable: bool = [3, 1],
    alpha_to_mask_enable: bool = [4, 1],
    alpha_to_mask_offset0: u32 = [24, 2],
    alpha_to_mask_offset1: u32 = [26, 2],
    alpha_to_mask_offset2: u32 = [28, 2],
    alpha_to_mask_offset3: u32 = [30, 2],
}

bitfield_register! {
    /// RB_COLOR_INFO / RB_COLOR1_INFO / RB_COLOR2_INFO / RB_COLOR3_INFO.
    RbColorInfo = r::RB_COLOR_INFO;
    /// EDRAM base in tiles (11 bits; EDRAM addressing wraps every 2048 tiles).
    color_base: u32 = [0, 11],
    color_base_bit_11: bool = [11, 1],
    color_format: ColorRenderTargetFormat = [16, 4],
    /// Signed exponent bias applied to shader output.
    color_exp_bias: i32 = [20, 6],
}

/// RB_COLOR_INFO register indices of render targets 0..3.
pub const RB_COLOR_INFO_REGISTERS: [u32; 4] =
    [r::RB_COLOR_INFO, r::RB_COLOR1_INFO, r::RB_COLOR2_INFO, r::RB_COLOR3_INFO];

/// RB_BLENDCONTROL register indices of render targets 0..3.
pub const RB_BLENDCONTROL_REGISTERS: [u32; 4] =
    [r::RB_BLENDCONTROL0, r::RB_BLENDCONTROL1, r::RB_BLENDCONTROL2, r::RB_BLENDCONTROL3];

impl RbColorInfo {
    /// Reads the color info of render target `rt` (0..3).
    pub fn read_rt(regs: &(impl RegisterSource + ?Sized), rt: usize) -> Self {
        Self::decode(regs.reg(RB_COLOR_INFO_REGISTERS[rt]))
    }
}

bitfield_register! {
    /// RB_COLOR_MASK - 4 bits (RGBA) per render target.
    RbColorMask = r::RB_COLOR_MASK;
    mask: u32 = [0, 16],
}

impl RbColorMask {
    /// RGBA write mask (bit 0 red .. bit 3 alpha) of render target `rt`.
    pub fn rt(&self, rt: usize) -> u32 {
        (self.mask >> (4 * rt)) & 0xF
    }
}

bitfield_register! {
    /// RB_BLENDCONTROL0..3.
    RbBlendcontrol = r::RB_BLENDCONTROL0;
    color_srcblend: BlendFactor = [0, 5],
    color_comb_fcn: BlendOp = [5, 3],
    color_destblend: BlendFactor = [8, 5],
    alpha_srcblend: BlendFactor = [16, 5],
    alpha_comb_fcn: BlendOp = [21, 3],
    alpha_destblend: BlendFactor = [24, 5],
}

impl RbBlendcontrol {
    /// Reads the blend control of render target `rt` (0..3).
    pub fn read_rt(regs: &(impl RegisterSource + ?Sized), rt: usize) -> Self {
        Self::decode(regs.reg(RB_BLENDCONTROL_REGISTERS[rt]))
    }
}

bitfield_register! {
    /// RB_DEPTHCONTROL.
    RbDepthcontrol = r::RB_DEPTHCONTROL;
    stencil_enable: bool = [0, 1],
    z_enable: bool = [1, 1],
    z_write_enable: bool = [2, 1],
    zfunc: CompareFunction = [4, 3],
    /// Separate back face stencil state (the *_bf fields and RB_STENCILREFMASK_BF).
    backface_enable: bool = [7, 1],
    stencilfunc: CompareFunction = [8, 3],
    stencilfail: StencilOp = [11, 3],
    stencilzpass: StencilOp = [14, 3],
    stencilzfail: StencilOp = [17, 3],
    stencilfunc_bf: CompareFunction = [20, 3],
    stencilfail_bf: StencilOp = [23, 3],
    stencilzpass_bf: StencilOp = [26, 3],
    stencilzfail_bf: StencilOp = [29, 3],
}

bitfield_register! {
    /// RB_HIZCONTROL.
    RbHizcontrol = r::RB_HIZCONTROL;
    hiz_write_enable: bool = [0, 1],
    hiz_enable: bool = [1, 1],
}

bitfield_register! {
    /// RB_STENCILREFMASK / RB_STENCILREFMASK_BF.
    RbStencilrefmask = r::RB_STENCILREFMASK;
    stencilref: u32 = [0, 8],
    stencilmask: u32 = [8, 8],
    stencilwritemask: u32 = [16, 8],
}

bitfield_register! {
    /// RB_DEPTH_INFO.
    RbDepthInfo = r::RB_DEPTH_INFO;
    /// EDRAM base in tiles.
    depth_base: u32 = [0, 11],
    depth_base_bit_11: bool = [11, 1],
    depth_format: DepthRenderTargetFormat = [16, 1],
}

bitfield_register! {
    /// RB_COPY_CONTROL - resolve source and operation.
    RbCopyControl = r::RB_COPY_CONTROL;
    /// 0..3 - color render target, 4+ - depth.
    copy_src_select: u32 = [0, 3],
    copy_sample_select: CopySampleSelect = [4, 3],
    color_clear_enable: bool = [8, 1],
    depth_clear_enable: bool = [9, 1],
    copy_command: CopyCommand = [20, 2],
}

impl RbCopyControl {
    /// Whether the resolve copies depth rather than a color render target.
    pub fn is_copying_depth(&self) -> bool {
        self.copy_src_select >= MAX_COLOR_RENDER_TARGETS
    }
}

bitfield_register! {
    /// RB_COPY_DEST_INFO - resolve destination format.
    RbCopyDestInfo = r::RB_COPY_DEST_INFO;
    copy_dest_endian: Endian128 = [0, 3],
    /// 3D (array / volume) destination.
    copy_dest_array: bool = [3, 1],
    copy_dest_slice: u32 = [4, 3],
    copy_dest_format: TextureFormat = [7, 6],
    copy_dest_number: SurfaceNumberFormat = [13, 3],
    copy_dest_exp_bias: i32 = [16, 6],
    /// Swap red and blue.
    copy_dest_swap: bool = [24, 1],
}

bitfield_register! {
    /// RB_COPY_DEST_PITCH - resolve destination pitch and height in pixels.
    RbCopyDestPitch = r::RB_COPY_DEST_PITCH;
    copy_dest_pitch: u32 = [0, 14],
    copy_dest_height: u32 = [16, 14],
}

/// Vertex fetch constant (two dwords, 3 per fetch constant slot group).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexFetchConstant {
    pub fetch_type: FetchConstantType,
    /// Address in dwords (byte address = address * 4).
    pub address: u32,
    pub endian: Endian,
    /// Size in dwords.
    pub size: u32,
}

impl VertexFetchConstant {
    /// Decodes the two dwords of a vertex fetch constant.
    pub fn decode(dword0: u32, dword1: u32) -> Self {
        Self {
            fetch_type: FetchConstantType::from_raw(dword0 & 3),
            address: dword0 >> 2,
            endian: Endian::from_raw(dword1 & 3),
            size: (dword1 >> 2) & 0xFF_FFFF,
        }
    }
    /// Encodes into the two dwords.
    pub fn encode(&self) -> [u32; 2] {
        [self.fetch_type.raw() | (self.address << 2), self.endian.raw() | ((self.size & 0xFF_FFFF) << 2)]
    }
    /// Reads vertex fetch constant `index` (0..95) from the fetch constant registers.
    pub fn read(regs: &(impl RegisterSource + ?Sized), index: u32) -> Self {
        let base = r::SHADER_CONSTANT_FETCH_00_0 + index * 2;
        Self::decode(regs.reg(base), regs.reg(base + 1))
    }
    /// Byte address in guest physical memory.
    pub fn address_bytes(&self) -> u32 {
        self.address << 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_initiator() {
        // Triangle strip, DMA, implicit, 32-bit indices, 300 indices.
        let raw = 0x06 | (1 << 11) | (300 << 16); // DMA source select is 0.
        let d = VgtDrawInitiator::decode(raw);
        assert_eq!(d.prim_type, PrimitiveType::TriangleStrip);
        assert_eq!(d.source_select, SourceSelect::Dma);
        assert_eq!(d.major_mode, MajorMode::Implicit);
        assert_eq!(d.index_size, IndexFormat::Int32);
        assert!(!d.not_eop);
        assert_eq!(d.num_indices, 300);
        assert_eq!(d.encode(), raw);
        // Auto-indexed rectangle list, explicit major mode, not_eop.
        let raw = 0x08 | (2 << 6) | (1 << 8) | (1 << 12) | (0xFFFF << 16);
        let d = VgtDrawInitiator::decode(raw);
        assert_eq!(d.prim_type, PrimitiveType::RectangleList);
        assert_eq!(d.source_select, SourceSelect::AutoIndex);
        assert_eq!(d.major_mode, MajorMode::Explicit);
        assert!(d.not_eop);
        assert_eq!(d.num_indices, 0xFFFF);
        assert_eq!(d.encode(), raw);
    }

    #[test]
    fn vgt_misc() {
        let s = VgtDmaSize::decode(0x8000_1234);
        assert_eq!(s.num_words, 0x1234);
        assert_eq!(s.swap_mode, Endian::Swap8In32);
        assert_eq!(s.encode(), 0x8000_1234);
        assert_eq!(VgtDmaSize::decode(0x4000_0010).swap_mode, Endian::Swap8In16);
        assert_eq!(VgtMultiPrimIbResetIndx::decode(0xFFFF_FFFF).reset_indx, 0xFF_FFFF);
        assert_eq!(VgtIndxOffset::decode(0xFFFF_FFFE).indx_offset, 0xFF_FFFE);
        assert_eq!(VgtMinVtxIndx::decode(0x10).min_indx, 0x10);
        assert_eq!(VgtMaxVtxIndx::decode(0xFFFF).max_indx, 0xFFFF);
        assert_eq!(VgtOutputPathCntl::decode(1).path_select, VgtOutputPath::TessellationEnable);
        assert_eq!(VgtHosCntl::decode(2).tess_mode, TessellationMode::Adaptive);
    }

    #[test]
    fn sq_registers() {
        let raw = 5 | (31 << 8) | (1 << 16) | (1 << 18) | (7 << 20) | (2 << 24) | (3 << 27) | (1 << 31);
        let p = SqProgramCntl::decode(raw);
        assert_eq!(p.vs_num_reg, 5);
        assert_eq!(p.ps_num_reg, 31);
        assert!(p.vs_resource && !p.ps_resource && p.param_gen && !p.gen_index_pix);
        assert_eq!(p.vs_export_count, 7);
        assert_eq!(p.vs_export_mode, VertexShaderExportMode::Position2VectorsSprite);
        assert_eq!(p.ps_export_mode, 3);
        assert!(p.gen_index_vtx);
        assert_eq!(p.encode(), raw);

        let raw = 1 | (1 << 1) | (2 << 2) | (9 << 8) | (1 << 18);
        let m = SqContextMisc::decode(raw);
        assert!(m.inst_pred_optimize && m.sc_output_screen_xy && m.tx_cache_sel);
        assert_eq!(m.sc_sample_cntl, SampleControl::CentroidsAndCenters);
        assert_eq!(m.param_gen_pos, 9);
        assert_eq!(m.encode(), raw);

        let i = SqInterpolatorCntl::decode(0xABCD_1234);
        assert_eq!(i.param_shade, 0x1234);
        assert_eq!(i.sampling_pattern, 0xABCD);
        let c = SqConstRange::decode(0x0FF << 12 | 0x100);
        assert_eq!((c.base, c.size), (0x100, 0xFF));
    }

    #[test]
    fn pa_registers() {
        let raw = 0b10 | (1 << 2) | (1 << 3) | (1 << 5) | (2 << 8) | (1 << 11) | (1 << 13) | (1 << 15) | (1 << 16);
        let raw = raw | (1 << 19) | (1 << 21);
        let m = PaSuScModeCntl::decode(raw);
        assert!(!m.cull_front && m.cull_back && m.face);
        assert_eq!(m.poly_mode, PolygonModeEnable::DualMode);
        assert_eq!(m.polymode_front_ptype, PolygonType::Lines);
        assert_eq!(m.polymode_back_ptype, PolygonType::Triangles);
        assert!(m.poly_offset_front_enable && !m.poly_offset_back_enable && m.poly_offset_para_enable);
        assert!(m.msaa_enable && m.vtx_window_offset_enable && m.provoking_vtx_last && m.multi_prim_ib_ena);
        assert!(!m.persp_corr_dis);
        assert_eq!(m.encode(), raw);

        let v = PaSuVtxCntl::decode(1 | (2 << 1) | (4 << 3));
        assert_eq!(v.pix_center, PixelCenter::OglHalf);
        assert_eq!(v.round_mode, VertexRounding::RoundToEven);
        assert_eq!(v.quant_mode, VertexQuantization::One);

        let c = PaClClipCntl::decode(0b101 | (1 << 16) | (1 << 19));
        assert_eq!(c.ucp_ena, 0b101);
        assert!(c.clip_disable && c.dx_clip_space_def && !c.vtx_kill_or);

        let vte = PaClVteCntl::decode(0x43F);
        assert!(vte.vport_x_scale_ena && vte.vport_z_offset_ena && vte.vtx_w0_fmt && !vte.vtx_xy_fmt);
        assert_eq!(vte.encode(), 0x43F);

        // Window offset -608, 32 (15-bit signed).
        let raw = ((-608i32 as u32) & 0x7FFF) | (32 << 16);
        let o = PaScWindowOffset::decode(raw);
        assert_eq!((o.window_x_offset, o.window_y_offset), (-608, 32));
        assert_eq!(o.encode(), raw);
        let s = PaScScreenScissor::decode(8192 | (8192 << 16));
        assert_eq!((s.x, s.y), (8192, 8192));
        let tl = PaScWindowScissorTl::decode(16 | (8 << 16) | (1 << 31));
        assert_eq!((tl.tl_x, tl.tl_y, tl.window_offset_disable), (16, 8, true));
        let br = PaScWindowScissorBr::decode(1280 | (720 << 16));
        assert_eq!((br.br_x, br.br_y), (1280, 720));
        let p = PaSuPointSize::decode(0x0010_0008);
        assert_eq!((p.width, p.height), (0x10, 0x8));
        let q = PaScVizQuery::decode(1 | (5 << 1) | (1 << 7));
        assert!(q.viz_query_ena && q.kill_pix_post_hi_z);
        assert_eq!(q.viz_query_id, 5);
        assert_eq!(PaSuLineCntl::decode(0x18).width, 0x18);
    }

    #[test]
    fn rb_registers() {
        assert_eq!(RbModecontrol::decode(6).edram_mode, EdramMode::Copy);
        let s = RbSurfaceInfo::decode(1280 | (2 << 16) | (5 << 18));
        assert_eq!(s.surface_pitch, 1280);
        assert_eq!(s.msaa_samples, MsaaSamples::X4);
        assert_eq!(s.hiz_pitch, 5);

        let raw = 3 | (1 << 3) | (1 << 4) | (3 << 24) | (1 << 26) | (2 << 30);
        let c = RbColorcontrol::decode(raw);
        assert_eq!(c.alpha_func, CompareFunction::LessEqual);
        assert!(c.alpha_test_enable && c.alpha_to_mask_enable);
        assert_eq!(
            (c.alpha_to_mask_offset0, c.alpha_to_mask_offset1, c.alpha_to_mask_offset2, c.alpha_to_mask_offset3),
            (3, 1, 0, 2)
        );
        assert_eq!(c.encode(), raw);

        // Base 0x7FF + bit 11, format 16_16_16_16_FLOAT, exp bias -3.
        let raw = 0x7FF | (1 << 11) | (7 << 16) | (((-3i32 as u32) & 0x3F) << 20);
        let ci = RbColorInfo::decode(raw);
        assert_eq!(ci.color_base, 0x7FF);
        assert!(ci.color_base_bit_11);
        assert_eq!(ci.color_format, ColorRenderTargetFormat::K16161616Float);
        assert_eq!(ci.color_exp_bias, -3);
        assert_eq!(ci.encode(), raw);

        let m = RbColorMask::decode(0x0000_F00F);
        assert_eq!((m.rt(0), m.rt(1), m.rt(2), m.rt(3)), (0xF, 0, 0, 0xF));

        // src=SrcAlpha, op=Add, dst=OneMinusSrcAlpha; alpha src=One, op=Max, dst=Zero.
        let raw = 6 | (7 << 8) | (1 << 16) | (3 << 21); // Add and Zero are 0.
        let b = RbBlendcontrol::decode(raw);
        assert_eq!(b.color_srcblend, BlendFactor::SrcAlpha);
        assert_eq!(b.color_comb_fcn, BlendOp::Add);
        assert_eq!(b.color_destblend, BlendFactor::OneMinusSrcAlpha);
        assert_eq!(b.alpha_srcblend, BlendFactor::One);
        assert_eq!(b.alpha_comb_fcn, BlendOp::Max);
        assert_eq!(b.alpha_destblend, BlendFactor::Zero);
        assert_eq!(b.encode(), raw);

        let raw = 1 | (1 << 1) | (1 << 2) | (3 << 4) | (1 << 7) | (7 << 8) | (2 << 11) | (3 << 14) | (6 << 17);
        let raw = raw | (5 << 20) | (1 << 23) | (4 << 26) | (7 << 29);
        let d = RbDepthcontrol::decode(raw);
        assert!(d.stencil_enable && d.z_enable && d.z_write_enable && d.backface_enable);
        assert_eq!(d.zfunc, CompareFunction::LessEqual);
        assert_eq!(d.stencilfunc, CompareFunction::Always);
        assert_eq!(d.stencilfail, StencilOp::Replace);
        assert_eq!(d.stencilzpass, StencilOp::IncrementClamp);
        assert_eq!(d.stencilzfail, StencilOp::IncrementWrap);
        assert_eq!(d.stencilfunc_bf, CompareFunction::NotEqual);
        assert_eq!(d.stencilfail_bf, StencilOp::Zero);
        assert_eq!(d.stencilzpass_bf, StencilOp::DecrementClamp);
        assert_eq!(d.stencilzfail_bf, StencilOp::DecrementWrap);
        assert_eq!(d.encode(), raw);

        let srm = RbStencilrefmask::decode(0x00FF_7F01);
        assert_eq!((srm.stencilref, srm.stencilmask, srm.stencilwritemask), (1, 0x7F, 0xFF));
        let di = RbDepthInfo::decode(0x123 | (1 << 16));
        assert_eq!(di.depth_base, 0x123);
        assert_eq!(di.depth_format, DepthRenderTargetFormat::D24FS8);
        let h = RbHizcontrol::decode(2);
        assert!(h.hiz_enable && !h.hiz_write_enable);
    }

    #[test]
    fn copy_registers() {
        let raw = 4 | (6 << 4) | (1 << 8) | (1 << 9) | (1 << 20);
        let c = RbCopyControl::decode(raw);
        assert_eq!(c.copy_src_select, 4);
        assert!(c.is_copying_depth());
        assert_eq!(c.copy_sample_select, CopySampleSelect::S0123);
        assert!(c.color_clear_enable && c.depth_clear_enable);
        assert_eq!(c.copy_command, CopyCommand::Convert);
        assert_eq!(c.encode(), raw);

        let raw = 2 | (1 << 3) | (5 << 4) | (6 << 7) | (7 << 13) | (((-5i32 as u32) & 0x3F) << 16) | (1 << 24);
        let d = RbCopyDestInfo::decode(raw);
        assert_eq!(d.copy_dest_endian, Endian128::Swap8In32);
        assert!(d.copy_dest_array && d.copy_dest_swap);
        assert_eq!(d.copy_dest_slice, 5);
        assert_eq!(d.copy_dest_format, TextureFormat::K8888);
        assert_eq!(d.copy_dest_number, SurfaceNumberFormat::Float);
        assert_eq!(d.copy_dest_exp_bias, -5);
        assert_eq!(d.encode(), raw);

        let p = RbCopyDestPitch::decode(1280 | (720 << 16));
        assert_eq!((p.copy_dest_pitch, p.copy_dest_height), (1280, 720));
    }

    #[test]
    fn vertex_fetch() {
        let f = VertexFetchConstant::decode(3 | (0x0100_0000 >> 2 << 2), 2 | (6 << 2));
        assert_eq!(f.fetch_type, FetchConstantType::Vertex);
        assert_eq!(f.address_bytes(), 0x0100_0000);
        assert_eq!(f.endian, Endian::Swap8In32);
        assert_eq!(f.size, 6);
        assert_eq!(VertexFetchConstant::decode(f.encode()[0], f.encode()[1]), f);
        let mut regs = vec![0u32; 0x5003];
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize + 2] = f.encode()[0];
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize + 3] = f.encode()[1];
        assert_eq!(VertexFetchConstant::read(&regs, 1), f);
    }

    #[test]
    fn read_from_register_file() {
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_COLOR2_INFO as usize] = 5 << 16;
        regs[r::RB_BLENDCONTROL3 as usize] = 1;
        regs[r::RB_DEPTHCONTROL as usize] = 2;
        assert_eq!(RbColorInfo::read_rt(&regs, 2).color_format, ColorRenderTargetFormat::K16161616);
        assert_eq!(RbBlendcontrol::read_rt(&regs, 3).color_srcblend, BlendFactor::One);
        assert!(RbDepthcontrol::read(&regs).z_enable);
    }
}
