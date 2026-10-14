//! EDRAM resolves (copies from EDRAM to guest memory, with optional clears), following Xenia
//! draw_util.cc GetResolveInfo / GetResolveEdramTileSpan / ResolveInfo::GetCopyShader.
//!
//! A resolve is triggered by a draw in `EdramMode::Copy`. Direct3D 9 draws a rectangle list whose
//! three vertices (float2 positions in pixels, in vertex fetch constant 0) are written by the CPU; the
//! covered rectangle, clamped by the scissor and aligned to 8x8, is the source region.

use crate::bitfields::*;
use crate::edram::*;
use crate::registers as r;
use crate::tiling::*;
use crate::viewport::scissor;
use crate::xenos::*;
use crate::{GuestMemory, RegisterSource};

/// Host options affecting the resolve description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolveOptions {
    pub draw_resolution_scale_x: u32,
    pub draw_resolution_scale_y: u32,
    /// Whether the Direct3D 9 half-pixel offset is emulated (see `ViewportArgs::half_pixel_offset`).
    pub half_pixel_offset: bool,
    /// Xenia's resolve_resolution_scale_fill_half_pixel_offset hack.
    pub fill_half_pixel_offset_with_scaling: bool,
    /// k_16_16 render targets are emulated as snorm -1...1 instead of -32...32.
    pub fixed_rg16_truncated_to_minus_1_to_1: bool,
    /// k_16_16_16_16 render targets are emulated as snorm -1...1 instead of -32...32.
    pub fixed_rgba16_truncated_to_minus_1_to_1: bool,
}

impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            draw_resolution_scale_x: 1,
            draw_resolution_scale_y: 1,
            half_pixel_offset: true,
            fill_half_pixel_offset_with_scaling: true,
            fixed_rg16_truncated_to_minus_1_to_1: false,
            fixed_rgba16_truncated_to_minus_1_to_1: false,
        }
    }
}

/// EDRAM source (or clear target) of a resolve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResolveEdramInfo {
    /// Row pitch in tiles (with 64bpp doubling).
    pub pitch_tiles: u32,
    pub msaa_samples: MsaaSamples,
    pub is_depth: bool,
    /// Base tile of the tile containing the resolve origin (may exceed 2047; wraps).
    pub base_tiles: u32,
    /// Color or depth format raw value.
    pub format: u32,
    pub format_is_64bpp: bool,
    /// Stretch the first covered row/column over the half-pixel gap with resolution scaling.
    pub fill_half_pixel_offset: bool,
    /// Decode PWL gamma before averaging / conversion (8_8_8_8_GAMMA sources).
    pub decode_pwl_gamma: bool,
}

/// Resolve rectangle in surface pixels (8-aligned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResolveRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ResolveRect {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Host compute shader family needed for the copy (ResolveCopyShaderIndex).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResolveCopyKind {
    /// Bit-exact copy, no conversion.
    Fast32bpp1x2xMsaa,
    Fast32bpp4xMsaa,
    Fast64bpp1x2xMsaa,
    Fast64bpp4xMsaa,
    /// Full conversion to a destination of the given bits per pixel.
    Full8bpp,
    Full16bpp,
    Full32bpp,
    Full64bpp,
    Full128bpp,
    /// CopyCommand::ConvertTo1111 to k_DXT3A_AS_1_1_1_1.
    Dxt3aAs1111,
}

impl ResolveCopyKind {
    /// log2 of the thread group size along X and Y used by Xenia for this shader.
    pub fn group_size_log2(self) -> (u32, u32) {
        match self {
            Self::Fast32bpp1x2xMsaa | Self::Fast32bpp4xMsaa | Self::Full8bpp => (6, 3),
            Self::Fast64bpp1x2xMsaa | Self::Fast64bpp4xMsaa | Self::Full16bpp | Self::Full32bpp | Self::Full64bpp => {
                (5, 3)
            }
            Self::Full128bpp => (4, 3),
            Self::Dxt3aAs1111 => (8, 3),
        }
    }
}

/// Complete description of a resolve.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolveInfo {
    pub copy_control: RbCopyControl,
    /// Source rectangle in pixels; empty means the resolve (copy and clears) must be dropped.
    pub rect: ResolveRect,
    /// Offset of the rectangle origin from the EDRAM base tile origin, in 8-pixel units (0..9, 0..1).
    pub edram_offset_x_div_8: u32,
    pub edram_offset_y_div_8: u32,
    /// Set when copying or clearing depth.
    pub depth_edram: Option<ResolveEdramInfo>,
    /// Set when copying or clearing color.
    pub color_edram: Option<ResolveEdramInfo>,
    /// Register bases without the rectangle origin offset (for locating host render targets).
    pub depth_original_base: u32,
    pub color_original_base: u32,
    /// Sample selection sanitized for the MSAA mode and source type.
    pub copy_sample_select: CopySampleSelect,
    /// RB_COPY_DEST_INFO with the format normalized (depth texture format for depth), exp bias
    /// adjusted, and swap cleared for depth.
    pub copy_dest_info: RbCopyDestInfo,
    /// Destination texture format.
    pub dest_format: TextureFormat,
    /// Destination pitch / height in blocks aligned to 32.
    pub dest_pitch_aligned: u32,
    pub dest_height_aligned: u32,
    /// Offset of the copy origin from `copy_dest_base` in 8-pixel units.
    pub dest_offset_x_div_8: u32,
    pub dest_offset_y_div_8: u32,
    /// RB_COPY_DEST_BASE as written.
    pub copy_dest_base_register: u32,
    /// Address the destination offsets are relative to.
    pub copy_dest_base: u32,
    /// Guest memory range potentially written; length 0 if the destination is unusable (the copy must
    /// be dropped, clears may still happen).
    pub copy_dest_extent_start: u32,
    pub copy_dest_extent_length: u32,
    /// RB_DEPTH_CLEAR.
    pub depth_clear: u32,
    /// RB_COLOR_CLEAR (high 32 bits for 64bpp).
    pub color_clear: u32,
    /// RB_COLOR_CLEAR_LO (low 32 bits for 64bpp).
    pub color_clear_lo: u32,
    /// RB_COPY_FUNC / REF / MASK for ConvertTo1111.
    pub copy_func: u32,
    pub copy_ref: u32,
    pub copy_mask: u32,
    /// Whether ConvertTo1111 can be performed (supported source/destination).
    pub convert_to_1111_supported: bool,
    /// PA_SC_WINDOW_OFFSET (whether or not it applies to the rectangle): with predicated tiling,
    /// the tile's screen origin negated.
    pub window_offset: [i32; 2],
}

/// Resolve failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveError {
    /// CopyCommand::Null.
    UnsupportedCopyCommand(CopyCommand),
    /// Vertex fetch constant 0 is not a 6-dword vertex buffer.
    UnsupportedVertexFetch { fetch_type: FetchConstantType, size: u32 },
    /// The rectangle vertices could not be read.
    VertexDataUnreadable(u32),
    /// More than 4x MSAA.
    InvalidMsaa(u32),
}

impl core::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedCopyCommand(c) => write!(f, "unsupported resolve copy command {c:?}"),
            Self::UnsupportedVertexFetch { fetch_type, size } => {
                write!(f, "unsupported resolve vertex fetch ({fetch_type:?}, {size} dwords)")
            }
            Self::VertexDataUnreadable(a) => write!(f, "resolve vertices at 0x{a:08X} are not readable"),
            Self::InvalidMsaa(v) => write!(f, "invalid MSAA mode {v}"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Clamps the sample selection to what the MSAA mode provides; depth can't be averaged
/// (SanitizeCopySampleSelect).
pub fn sanitize_copy_sample_select(select: CopySampleSelect, msaa: MsaaSamples, is_depth: bool) -> CopySampleSelect {
    use CopySampleSelect as S;
    match msaa {
        MsaaSamples::X4 => {
            let s = if select.raw() > S::S0123.raw() { S::S0123 } else { select };
            if is_depth {
                match s {
                    S::S01 | S::S0123 => S::S0,
                    S::S23 => S::S2,
                    other => other,
                }
            } else {
                s
            }
        }
        MsaaSamples::X2 => {
            let s = match select {
                S::S2 => S::S0,
                S::S3 => S::S1,
                other if other.raw() > S::S01.raw() => S::S01,
                other => other,
            };
            if is_depth && s == S::S01 { S::S0 } else { s }
        }
        _ => S::S0,
    }
}

/// Tiles covered by a resolve area: `(base, row_length_used, rows)`; rows are `pitch_tiles` apart
/// (GetResolveEdramTileSpan).
pub fn resolve_edram_tile_span(
    edram: &ResolveEdramInfo,
    offset_x_div_8: u32,
    offset_y_div_8: u32,
    width_div_8: u32,
    height_div_8: u32,
) -> (u32, u32, u32) {
    let x_scale_log2 = 3 + edram.msaa_samples.samples_x_log2() + edram.format_is_64bpp as u32;
    let x0 = (offset_x_div_8 << x_scale_log2) / EDRAM_TILE_WIDTH_SAMPLES;
    let x1 = ((offset_x_div_8 + width_div_8) << x_scale_log2).div_ceil(EDRAM_TILE_WIDTH_SAMPLES);
    let y_scale_log2 = 3 + edram.msaa_samples.samples_y_log2();
    let y0 = (offset_y_div_8 << y_scale_log2) / EDRAM_TILE_HEIGHT_SAMPLES;
    let y1 = ((offset_y_div_8 + height_div_8) << y_scale_log2).div_ceil(EDRAM_TILE_HEIGHT_SAMPLES);
    (edram.base_tiles + y0 * edram.pitch_tiles + x0, x1 - x0, y1 - y0)
}

fn color_resolve_number_format_matches(format: TextureFormat, number: SurfaceNumberFormat) -> bool {
    use TextureFormat as T;
    match format {
        T::K16Float | T::K1616Float | T::K16161616Float | T::K32Float | T::K3232Float | T::K32323232Float => {
            number == SurfaceNumberFormat::Float
        }
        _ => number == SurfaceNumberFormat::UnsignedRepeatingFraction,
    }
}

impl ResolveInfo {
    pub fn is_copying_depth(&self) -> bool {
        self.copy_control.is_copying_depth()
    }
    pub fn is_clearing_depth(&self) -> bool {
        self.copy_control.depth_clear_enable
    }
    pub fn is_clearing_color(&self) -> bool {
        !self.is_copying_depth() && self.copy_control.color_clear_enable
    }
    /// EDRAM info of the copy source.
    pub fn source_edram(&self) -> Option<&ResolveEdramInfo> {
        if self.is_copying_depth() { self.depth_edram.as_ref() } else { self.color_edram.as_ref() }
    }
    /// Tiles touched by the copy: `(base, row_length_used, rows, pitch)`.
    pub fn copy_edram_tile_span(&self) -> Option<(u32, u32, u32, u32)> {
        let e = self.source_edram()?;
        let (base, len, rows) = resolve_edram_tile_span(
            e,
            self.edram_offset_x_div_8,
            self.edram_offset_y_div_8,
            self.rect.width >> 3,
            self.rect.height >> 3,
        );
        Some((base, len, rows, e.pitch_tiles))
    }
    /// Color clear value as two dwords (low, high): for 64bpp `RB_COLOR_CLEAR_LO, RB_COLOR_CLEAR`,
    /// for 32bpp `RB_COLOR_CLEAR` twice.
    pub fn color_clear_value(&self) -> [u32; 2] {
        match self.color_edram {
            Some(e) if e.format_is_64bpp => [self.color_clear_lo, self.color_clear],
            _ => [self.color_clear, self.color_clear],
        }
    }
    /// Depth clear as float depth and stencil, interpreting RB_DEPTH_CLEAR (depth in bits 8..31,
    /// stencil in 0..7) for the depth format.
    pub fn depth_clear_value(&self) -> (f32, u8) {
        let depth24 = self.depth_clear >> 8;
        let format = self.depth_edram.map(|e| DepthRenderTargetFormat::from_raw(e.format));
        let depth = match format {
            Some(DepthRenderTargetFormat::D24FS8) => float20e4_to_32(depth24),
            _ => (depth24 + (depth24 >> 23)) as f32 * (1.0 / (1u32 << 24) as f32),
        };
        (depth, self.depth_clear as u8)
    }
    /// Copy shader family and Xenia's thread group counts, or `None` if nothing can copy this
    /// configuration (ResolveInfo::GetCopyShader).
    pub fn copy_kind(
        &self,
        draw_resolution_scale_x: u32,
        draw_resolution_scale_y: u32,
    ) -> Option<(ResolveCopyKind, [u32; 2])> {
        let is_depth = self.is_copying_depth();
        let edram = self.source_edram()?;
        let kind = if self.copy_control.copy_command == CopyCommand::ConvertTo1111 {
            if !self.convert_to_1111_supported {
                return None;
            }
            ResolveCopyKind::Dxt3aAs1111
        } else {
            let source_format = ColorRenderTargetFormat::from_raw(edram.format);
            let fast = is_depth
                || (source_format != ColorRenderTargetFormat::K8888Gamma
                    && self.copy_dest_info.copy_dest_exp_bias == 0
                    && self.copy_sample_select.is_single()
                    && is_color_resolve_format_bitwise_equivalent(source_format, self.dest_format)
                    && color_resolve_number_format_matches(self.dest_format, self.copy_dest_info.copy_dest_number));
            if fast {
                match (edram.msaa_samples == MsaaSamples::X4, edram.format_is_64bpp && !is_depth) {
                    (true, true) => ResolveCopyKind::Fast64bpp4xMsaa,
                    (true, false) => ResolveCopyKind::Fast32bpp4xMsaa,
                    (false, true) => ResolveCopyKind::Fast64bpp1x2xMsaa,
                    (false, false) => ResolveCopyKind::Fast32bpp1x2xMsaa,
                }
            } else {
                match self.dest_format.info().bits_per_pixel {
                    8 => ResolveCopyKind::Full8bpp,
                    16 => ResolveCopyKind::Full16bpp,
                    32 => ResolveCopyKind::Full32bpp,
                    64 => ResolveCopyKind::Full64bpp,
                    128 => ResolveCopyKind::Full128bpp,
                    _ => return None,
                }
            }
        };
        let width = self.rect.width * draw_resolution_scale_x.max(1);
        let height = self.rect.height * draw_resolution_scale_y.max(1);
        let (gx, gy) = kind.group_size_log2();
        Some((kind, [width.div_ceil(1 << gx), height.div_ceil(1 << gy)]))
    }
}

/// Builds the resolve description from the registers and the rectangle vertices in guest memory
/// (GetResolveInfo).
pub fn resolve_info(
    regs: &(impl RegisterSource + ?Sized),
    memory: &(impl GuestMemory + ?Sized),
    options: &ResolveOptions,
) -> Result<ResolveInfo, ResolveError> {
    let copy_control = RbCopyControl::read(regs);
    let is_convert_to_1111 = copy_control.copy_command == CopyCommand::ConvertTo1111;
    if copy_control.copy_command == CopyCommand::Null {
        return Err(ResolveError::UnsupportedCopyCommand(copy_control.copy_command));
    }

    // Rectangle vertices: vertex fetch constant 0, 3 x float2.
    let fetch = VertexFetchConstant::read(regs, 0);
    if fetch.fetch_type != FetchConstantType::Vertex || fetch.size != 6 {
        return Err(ResolveError::UnsupportedVertexFetch { fetch_type: fetch.fetch_type, size: fetch.size });
    }
    let mut raw = [0u8; 24];
    if !memory.read_physical(fetch.address_bytes(), &mut raw) {
        return Err(ResolveError::VertexDataUnreadable(fetch.address_bytes()));
    }
    let vtx_cntl = PaSuVtxCntl::read(regs);
    let half_pixel = if vtx_cntl.pix_center == PixelCenter::D3DZero { 0.5 } else { 0.0 };
    let fixed: [i32; 6] = core::array::from_fn(|i| {
        let bits = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
        float_to_d3d11_fixed16p8(f32::from_bits(gpu_swap_u32(bits, fetch.endian)) + half_pixel)
    });
    // Top-left rule: include .5 on the top/left, exclude it on the bottom/right.
    let mut x0 = (fixed[0].min(fixed[2]).min(fixed[4]) + 127) >> 8;
    let mut y0 = (fixed[1].min(fixed[3]).min(fixed[5]) + 127) >> 8;
    let mut x1 = (fixed[0].max(fixed[2]).max(fixed[4]) + 127) >> 8;
    let mut y1 = (fixed[1].max(fixed[3]).max(fixed[5]) + 127) >> 8;
    let window_offset = PaScWindowOffset::read(regs);
    if PaSuScModeCntl::read(regs).vtx_window_offset_enable {
        x0 += window_offset.window_x_offset;
        y0 += window_offset.window_y_offset;
        x1 += window_offset.window_x_offset;
        y1 += window_offset.window_y_offset;
    }
    let sc = scissor(regs, false, false);
    let (sl, st) = (sc.offset[0] as i32, sc.offset[1] as i32);
    let (sr, sb) = (sl + sc.extent[0] as i32, st + sc.extent[1] as i32);
    x0 = x0.clamp(sl, sr);
    y0 = y0.clamp(st, sb);
    x1 = x1.clamp(sl, sr);
    y1 = y1.clamp(st, sb);
    let align = RESOLVE_ALIGNMENT_PIXELS as i32;
    x0 &= !(align - 1);
    y0 &= !(align - 1);
    x1 = (x1 + align - 1) & !(align - 1);
    y1 = (y1 + align - 1) & !(align - 1);

    let surface = RbSurfaceInfo::read(regs);
    let msaa = surface.msaa_samples;
    if let MsaaSamples::Unknown(v) = msaa {
        return Err(ResolveError::InvalidMsaa(v));
    }
    let pitch_aligned = (surface.surface_pitch & !(RESOLVE_ALIGNMENT_PIXELS - 1)) as i32;
    if x1 > pitch_aligned {
        x0 = x0.min(pitch_aligned);
        x1 = x1.min(pitch_aligned);
    }
    if y1 - y0 > MAX_RESOLVE_SIZE as i32 {
        y1 = y0 + MAX_RESOLVE_SIZE as i32;
    }

    let mut info = ResolveInfo { copy_control, ..Default::default() };
    info.window_offset = [window_offset.window_x_offset, window_offset.window_y_offset];
    if x0 >= x1 || y0 >= y1 {
        return Ok(info);
    }
    info.rect = ResolveRect { x: x0 as u32, y: y0 as u32, width: (x1 - x0) as u32, height: (y1 - y0) as u32 };

    let is_depth = copy_control.is_copying_depth();
    info.copy_sample_select = sanitize_copy_sample_select(copy_control.copy_sample_select, msaa, is_depth);

    let dest_info = RbCopyDestInfo::read(regs);
    let depth_info = RbDepthInfo::read(regs);
    let dest_format = if is_depth { depth_info.depth_format.texture_format() } else { dest_info.copy_dest_format };
    info.dest_format = dest_format;

    if is_convert_to_1111
        && !is_depth
        && dest_format == TextureFormat::Dxt3aAs1111
        && !dest_info.copy_dest_array
        && info.copy_sample_select.is_single()
    {
        let source = RbColorInfo::read_rt(regs, copy_control.copy_src_select as usize).color_format;
        info.convert_to_1111_supported =
            matches!(source, ColorRenderTargetFormat::K8888 | ColorRenderTargetFormat::K8888Gamma);
    }
    info.copy_func = regs.reg(r::RB_COPY_FUNC);
    info.copy_ref = regs.reg(r::RB_COPY_REF);
    info.copy_mask = regs.reg(r::RB_COPY_MASK);

    // Destination memory extent.
    let dest_base = regs.reg(r::RB_COPY_DEST_BASE);
    info.copy_dest_base_register = dest_base;
    let dest_pitch = RbCopyDestPitch::read(regs);
    let pitch_aligned = dest_pitch.copy_dest_pitch.next_multiple_of(STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS);
    let mut dest_height = dest_pitch.copy_dest_height;
    if dest_info.copy_dest_array {
        // Direct3D writes pitch * slice height to RB_COPY_SURFACE_SLICE for volume resolves.
        let slice = regs.reg(r::RB_COPY_SURFACE_SLICE);
        if slice != 0 && dest_pitch.copy_dest_pitch != 0 {
            dest_height = slice / dest_pitch.copy_dest_pitch;
        }
    }
    let height_aligned = dest_height.next_multiple_of(STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS);
    info.dest_pitch_aligned = pitch_aligned;
    info.dest_height_aligned = height_aligned;
    let format_info = dest_format.info();
    let usable = if is_convert_to_1111 {
        info.convert_to_1111_supported
    } else {
        is_depth || format_info.format_type == FormatType::Resolvable
    };
    let mut base_adjusted = dest_base;
    let (extent_start, extent_end);
    if usable {
        let bpp_log2 = format_info.bytes_per_block().ilog2();
        let block_log2 = if is_convert_to_1111 { 2 } else { 0 };
        let is_3d = dest_info.copy_dest_array;
        let rel_x_mask = (1u32 << tiled_x_base_granularity_log2(is_3d, bpp_log2)) - 1;
        let rel_y_mask = (1u32 << tiled_y_base_granularity_log2(is_3d, bpp_log2)) - 1;
        let mut addr_base = dest_base;
        let mut ax0 = (x0 as u32) >> block_log2;
        let ay0 = (y0 as u32) >> block_log2;
        if !is_3d {
            // Direct3D advances the base in 32x32 macro tiles; part of the X offset may be in its low bits.
            let macro_bytes_log2 = 2 * TEXTURE_TILE_WIDTH_HEIGHT_LOG2 + bpp_log2;
            let phase = (dest_base & (TEXTURE_SUBRESOURCE_ALIGNMENT_BYTES - 1)) >> macro_bytes_log2;
            addr_base -= phase << macro_bytes_log2;
            ax0 += phase << TEXTURE_TILE_WIDTH_HEIGHT_LOG2;
        }
        let ax1 = ax0 + (((x1 - x0) as u32) >> block_log2);
        let ay1 = ay0 + (((y1 - y0) as u32) >> block_log2);
        base_adjusted = addr_base;
        info.dest_offset_x_div_8 = ((ax0 & rel_x_mask) << block_log2) >> RESOLVE_ALIGNMENT_PIXELS_LOG2;
        info.dest_offset_y_div_8 = ((ay0 & rel_y_mask) << block_log2) >> RESOLVE_ALIGNMENT_PIXELS_LOG2;
        let base_x = (ax0 & !rel_x_mask) as i32;
        let base_y = (ay0 & !rel_y_mask) as i32;
        if is_3d {
            base_adjusted =
                base_adjusted.wrapping_add(tiled_3d(base_x, base_y, 0, pitch_aligned, height_aligned, bpp_log2) as u32);
            let slice = dest_info.copy_dest_slice;
            extent_start = addr_base.wrapping_add(tiled_address_lower_bound_3d(
                ax0,
                ay0,
                slice,
                pitch_aligned,
                height_aligned,
                bpp_log2,
            ) as u32);
            extent_end = addr_base.wrapping_add(tiled_address_upper_bound_3d(
                ax1,
                ay1,
                slice + 1,
                pitch_aligned,
                height_aligned,
                bpp_log2,
            ) as u32);
        } else {
            base_adjusted = base_adjusted.wrapping_add(tiled_2d(base_x, base_y, pitch_aligned, bpp_log2) as u32);
            extent_start = addr_base.wrapping_add(tiled_address_lower_bound_2d(ax0, ay0, pitch_aligned, bpp_log2));
            extent_end = addr_base.wrapping_add(tiled_address_upper_bound_2d(ax1, ay1, pitch_aligned, bpp_log2));
        }
    } else {
        extent_start = base_adjusted;
        extent_end = base_adjusted;
    }
    info.copy_dest_base = base_adjusted;
    info.copy_dest_extent_start = extent_start;
    info.copy_dest_extent_length = extent_end.wrapping_sub(extent_start);

    // EDRAM source / clear targets, with the base moved to the tile containing the origin.
    let sx = msaa.samples_x_log2();
    let sy = msaa.samples_y_log2();
    let x0_samples = (x0 as u32) << sx;
    let y0_samples = (y0 as u32) << sy;
    let base_offset_x_tiles = x0_samples / EDRAM_TILE_WIDTH_SAMPLES;
    let base_offset_y_tiles = y0_samples / EDRAM_TILE_HEIGHT_SAMPLES;
    info.edram_offset_x_div_8 = (x0_samples % EDRAM_TILE_WIDTH_SAMPLES) >> (sx + 3);
    info.edram_offset_y_div_8 = (y0_samples % EDRAM_TILE_HEIGHT_SAMPLES) >> (sy + 3);
    let pitch_tiles = surface_pitch_tiles(surface.surface_pitch, msaa, false);
    let base_offset_tiles = base_offset_y_tiles * pitch_tiles + base_offset_x_tiles;
    let fill_half_pixel_offset = (options.draw_resolution_scale_x > 1 || options.draw_resolution_scale_y > 1)
        && options.fill_half_pixel_offset_with_scaling
        && options.half_pixel_offset
        && vtx_cntl.pix_center == PixelCenter::D3DZero;
    let mut exp_bias = if is_depth { 0 } else { dest_info.copy_dest_exp_bias };
    if is_depth || copy_control.depth_clear_enable {
        info.depth_edram = Some(ResolveEdramInfo {
            pitch_tiles,
            msaa_samples: msaa,
            is_depth: true,
            base_tiles: depth_info.depth_base + base_offset_tiles,
            format: depth_info.depth_format.raw(),
            format_is_64bpp: false,
            fill_half_pixel_offset,
            decode_pwl_gamma: false,
        });
        info.depth_original_base = depth_info.depth_base;
    }
    if !is_depth {
        let color = RbColorInfo::read_rt(regs, copy_control.copy_src_select as usize);
        let is_64bpp = color.color_format.is_64bpp();
        info.color_edram = Some(ResolveEdramInfo {
            pitch_tiles: pitch_tiles << is_64bpp as u32,
            msaa_samples: msaa,
            is_depth: false,
            base_tiles: color.color_base + (base_offset_tiles << is_64bpp as u32),
            format: color.color_format.raw(),
            format_is_64bpp: is_64bpp,
            fill_half_pixel_offset,
            decode_pwl_gamma: true,
        });
        if (options.fixed_rg16_truncated_to_minus_1_to_1 && color.color_format == ColorRenderTargetFormat::K1616)
            || (options.fixed_rgba16_truncated_to_minus_1_to_1
                && color.color_format == ColorRenderTargetFormat::K16161616)
        {
            exp_bias = (exp_bias + 5).min(31);
        }
        info.color_original_base = color.color_base;
    }

    info.copy_dest_info = RbCopyDestInfo {
        copy_dest_format: dest_format,
        copy_dest_exp_bias: exp_bias,
        copy_dest_swap: dest_info.copy_dest_swap && !is_depth,
        ..dest_info
    };
    info.depth_clear = regs.reg(r::RB_DEPTH_CLEAR);
    info.color_clear = regs.reg(r::RB_COLOR_CLEAR);
    info.color_clear_lo = regs.reg(r::RB_COLOR_CLEAR_LO);
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERTEX_ADDR: u32 = 0x1000;

    /// Register file for a 1280x720 resolve of RT0 to a 2D texture, vertices big-endian at VERTEX_ADDR.
    fn setup(vertices: [f32; 6]) -> (Vec<u32>, Vec<u8>) {
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_MODECONTROL as usize] = 6;
        regs[r::RB_SURFACE_INFO as usize] = 1280;
        regs[r::RB_COLOR_INFO as usize] = 100 | (ColorRenderTargetFormat::K8888.raw() << 16);
        regs[r::RB_DEPTH_INFO as usize] = 900;
        regs[r::RB_COPY_CONTROL as usize] =
            RbCopyControl { copy_command: CopyCommand::Raw, ..Default::default() }.encode();
        regs[r::RB_COPY_DEST_INFO as usize] = RbCopyDestInfo {
            copy_dest_endian: Endian128::Swap8In32,
            copy_dest_format: TextureFormat::K8888,
            ..Default::default()
        }
        .encode();
        regs[r::RB_COPY_DEST_PITCH as usize] =
            RbCopyDestPitch { copy_dest_pitch: 1280, copy_dest_height: 720 }.encode();
        regs[r::RB_COPY_DEST_BASE as usize] = 0x0200_0000;
        regs[r::PA_SC_WINDOW_SCISSOR_BR as usize] = PaScWindowScissorBr { br_x: 1280, br_y: 720 }.encode();
        regs[r::PA_SC_SCREEN_SCISSOR_BR as usize] = PaScScreenScissor { x: 8192, y: 8192 }.encode();
        let fetch = VertexFetchConstant {
            fetch_type: FetchConstantType::Vertex,
            address: VERTEX_ADDR >> 2,
            endian: Endian::Swap8In32,
            size: 6,
        }
        .encode();
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize] = fetch[0];
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize + 1] = fetch[1];
        let mut mem = vec![0u8; 0x2000];
        for (i, v) in vertices.iter().enumerate() {
            let at = VERTEX_ADDR as usize + i * 4;
            mem[at..at + 4].copy_from_slice(&v.to_bits().to_be_bytes());
        }
        (regs, mem)
    }

    /// Direct3D 9 resolve rectangle with the -0.5 half-pixel bias.
    fn rect_vertices(x0: f32, y0: f32, x1: f32, y1: f32) -> [f32; 6] {
        [x0 - 0.5, y0 - 0.5, x1 - 0.5, y0 - 0.5, x1 - 0.5, y1 - 0.5]
    }

    #[test]
    fn full_screen_color_resolve() {
        let (regs, mem) = setup(rect_vertices(0.0, 0.0, 1280.0, 720.0));
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!(info.rect, ResolveRect { x: 0, y: 0, width: 1280, height: 720 });
        assert!(!info.is_copying_depth() && info.depth_edram.is_none());
        let c = info.color_edram.unwrap();
        assert_eq!((c.base_tiles, c.pitch_tiles, c.is_depth, c.format_is_64bpp), (100, 16, false, false));
        assert_eq!(info.copy_sample_select, CopySampleSelect::S0);
        assert_eq!(info.dest_format, TextureFormat::K8888);
        assert_eq!(info.copy_dest_info.copy_dest_endian, Endian128::Swap8In32);
        assert_eq!((info.dest_pitch_aligned, info.dest_height_aligned), (1280, 736));
        assert_eq!(info.copy_dest_base, 0x0200_0000);
        assert_eq!(info.copy_dest_extent_start, 0x0200_0000);
        // 1280x736 32bpp tiled = 3768320 bytes; the extent ends at the last 32x32 tile's end.
        assert_eq!(info.copy_dest_extent_length, 1280 * 736 * 4);
        assert_eq!(info.copy_edram_tile_span(), Some((100, 16, 45, 16)));
        assert_eq!(info.copy_kind(1, 1), Some((ResolveCopyKind::Fast32bpp1x2xMsaa, [20, 90])));
    }

    #[test]
    fn subrect_msaa_and_offsets() {
        // 4x MSAA, rectangle (200, 24)-(328, 88), 64bpp source.
        let (mut regs, mem) = setup(rect_vertices(200.0, 24.0, 328.0, 88.0));
        regs[r::RB_SURFACE_INFO as usize] = 1280 | (2 << 16);
        regs[r::RB_COLOR_INFO as usize] = 64 | (ColorRenderTargetFormat::K16161616Float.raw() << 16);
        regs[r::RB_COPY_CONTROL as usize] =
            RbCopyControl { copy_sample_select: CopySampleSelect::S0123, ..Default::default() }.encode();
        regs[r::RB_COPY_DEST_INFO as usize] = RbCopyDestInfo {
            copy_dest_format: TextureFormat::K16161616Float,
            copy_dest_number: SurfaceNumberFormat::Float,
            ..Default::default()
        }
        .encode();
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!(info.rect, ResolveRect { x: 200, y: 24, width: 128, height: 64 });
        // x0 samples = 400 -> tile 5, offset 0; y0 samples = 48 -> tile 3, offset 0.
        let c = info.color_edram.unwrap();
        assert_eq!(c.pitch_tiles, 64);
        assert_eq!(c.base_tiles, 64 + ((3 * 32 + 5) << 1));
        assert_eq!((info.edram_offset_x_div_8, info.edram_offset_y_div_8), (0, 0));
        assert_eq!(info.copy_sample_select, CopySampleSelect::S0123);
        // Averaging needs the full shader.
        assert_eq!(info.copy_kind(1, 1).unwrap().0, ResolveCopyKind::Full64bpp);
        let (base, len, rows, pitch) = info.copy_edram_tile_span().unwrap();
        // 128 px * 2 samples * 2 (64bpp) = 512 samples -> 7 tiles (640 / 80 = 8 boundaries).
        assert_eq!((base, len, rows, pitch), (c.base_tiles, 7, 8, 64));

        // Origin inside a tile: x = 40 px at 4x = 80 samples = tile 1; y = 8 -> 16 samples = tile 1.
        let (mut regs, mem) = setup(rect_vertices(48.0, 8.0, 64.0, 16.0));
        regs[r::RB_SURFACE_INFO as usize] = 1280 | (2 << 16);
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!((info.edram_offset_x_div_8, info.edram_offset_y_div_8), (1, 0));
        assert_eq!(info.color_edram.unwrap().base_tiles, 100 + 32 + 1);
    }

    #[test]
    fn rect_rounding_scissor_and_window_offset() {
        // Unaligned rectangle is expanded to 8-pixel alignment; scissor clamps.
        let (mut regs, mem) = setup(rect_vertices(3.0, 5.0, 1000.0, 1000.0));
        regs[r::PA_SC_WINDOW_SCISSOR_BR as usize] = PaScWindowScissorBr { br_x: 1280, br_y: 700 }.encode();
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!(info.rect, ResolveRect { x: 0, y: 0, width: 1000, height: 704 });
        // Top-left rule: a vertex at 7.5 (+0.5 bias = 8.0) includes column 8.
        let (regs, mem) = setup([7.5, 7.5, 64.0, 7.5, 64.0, 64.0]);
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!((info.rect.x, info.rect.y), (8, 8));
        // Window offset applied to the vertices.
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::PA_SU_SC_MODE_CNTL as usize] = 1 << 16;
        regs[r::PA_SC_WINDOW_OFFSET as usize] = PaScWindowOffset { window_x_offset: 16, window_y_offset: 0 }.encode();
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!(info.rect, ResolveRect { x: 16, y: 0, width: 64, height: 64 });
        // Entirely outside: empty, not an error.
        let (regs, mem) = setup(rect_vertices(2000.0, 0.0, 2100.0, 64.0));
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert!(info.rect.is_empty());
        // Surface pitch clamp.
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 1280.0, 64.0));
        regs[r::RB_SURFACE_INFO as usize] = 1020;
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!(info.rect.width, 1016);
    }

    #[test]
    fn depth_resolve_with_clears() {
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 640.0, 360.0));
        regs[r::RB_SURFACE_INFO as usize] = 640 | (1 << 16);
        regs[r::RB_DEPTH_INFO as usize] = 500 | (1 << 16);
        regs[r::RB_COPY_CONTROL as usize] = RbCopyControl {
            copy_src_select: 4,
            copy_sample_select: CopySampleSelect::S01,
            color_clear_enable: true,
            depth_clear_enable: true,
            copy_command: CopyCommand::Raw,
        }
        .encode();
        regs[r::RB_COPY_DEST_INFO as usize] = RbCopyDestInfo { copy_dest_swap: true, ..Default::default() }.encode();
        regs[r::RB_DEPTH_CLEAR as usize] = (float32_to_20e4(0.5, true) << 8) | 0x7F;
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert!(info.is_copying_depth() && info.is_clearing_depth() && !info.is_clearing_color());
        assert!(info.color_edram.is_none());
        let d = info.depth_edram.unwrap();
        assert_eq!((d.base_tiles, d.pitch_tiles, d.is_depth), (500, 8, true));
        assert_eq!(info.copy_sample_select, CopySampleSelect::S0);
        assert_eq!(info.dest_format, TextureFormat::K248Float);
        assert!(!info.copy_dest_info.copy_dest_swap);
        assert_eq!(info.depth_clear_value(), (0.5, 0x7F));
        assert_eq!(info.copy_kind(1, 1).unwrap().0, ResolveCopyKind::Fast32bpp1x2xMsaa);
    }

    #[test]
    fn color_clear_values_and_exp_bias_hack() {
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::RB_COLOR_INFO as usize] = ColorRenderTargetFormat::K16161616.raw() << 16;
        regs[r::RB_COPY_CONTROL as usize] = RbCopyControl { color_clear_enable: true, ..Default::default() }.encode();
        regs[r::RB_COPY_DEST_INFO as usize] =
            RbCopyDestInfo { copy_dest_format: TextureFormat::K16161616, copy_dest_exp_bias: 2, ..Default::default() }
                .encode();
        regs[r::RB_COLOR_CLEAR as usize] = 0xAAAA_BBBB;
        regs[r::RB_COLOR_CLEAR_LO as usize] = 0x1111_2222;
        let opts = ResolveOptions { fixed_rgba16_truncated_to_minus_1_to_1: true, ..Default::default() };
        let info = resolve_info(&regs, &mem, &opts).unwrap();
        assert!(info.is_clearing_color());
        assert_eq!(info.color_clear_value(), [0x1111_2222, 0xAAAA_BBBB]);
        assert_eq!(info.copy_dest_info.copy_dest_exp_bias, 7);
        assert_eq!(info.copy_kind(1, 1).unwrap().0, ResolveCopyKind::Full64bpp);
    }

    #[test]
    fn convert_to_1111_and_3d_destination() {
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::RB_COPY_CONTROL as usize] =
            RbCopyControl { copy_command: CopyCommand::ConvertTo1111, ..Default::default() }.encode();
        regs[r::RB_COPY_DEST_INFO as usize] =
            RbCopyDestInfo { copy_dest_format: TextureFormat::Dxt3aAs1111, ..Default::default() }.encode();
        regs[r::RB_COPY_FUNC as usize] = 0x4444;
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert!(info.convert_to_1111_supported);
        assert_eq!(info.copy_func, 0x4444);
        assert_eq!(info.copy_kind(1, 1), Some((ResolveCopyKind::Dxt3aAs1111, [1, 8])));
        // 64x64 pixels -> 16x16 blocks of 8 bytes.
        assert!(info.copy_dest_extent_length >= 16 * 16 * 8);

        // Volume destination: slice 2, height from RB_COPY_SURFACE_SLICE.
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::RB_COPY_DEST_INFO as usize] = RbCopyDestInfo {
            copy_dest_format: TextureFormat::K8888,
            copy_dest_array: true,
            copy_dest_slice: 2,
            ..Default::default()
        }
        .encode();
        regs[r::RB_COPY_DEST_PITCH as usize] = RbCopyDestPitch { copy_dest_pitch: 64, copy_dest_height: 0 }.encode();
        regs[r::RB_COPY_SURFACE_SLICE as usize] = 64 * 64;
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!((info.dest_pitch_aligned, info.dest_height_aligned), (64, 64));
        let start = info.copy_dest_extent_start - 0x0200_0000;
        let end = start + info.copy_dest_extent_length;
        for y in 0..64 {
            for x in 0..64 {
                let a = tiled_3d(x, y, 2, 64, 64, 2) as u32;
                assert!(a >= start && a + 4 <= end);
            }
        }
    }

    #[test]
    fn errors_and_unusable_destination() {
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::RB_COPY_CONTROL as usize] =
            RbCopyControl { copy_command: CopyCommand::Null, ..Default::default() }.encode();
        assert_eq!(
            resolve_info(&regs, &mem, &ResolveOptions::default()),
            Err(ResolveError::UnsupportedCopyCommand(CopyCommand::Null))
        );
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize + 1] = 4 << 2;
        assert!(matches!(
            resolve_info(&regs, &mem, &ResolveOptions::default()),
            Err(ResolveError::UnsupportedVertexFetch { size: 4, .. })
        ));
        let (regs, _) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        assert_eq!(
            resolve_info(&regs, &vec![0u8; 16], &ResolveOptions::default()),
            Err(ResolveError::VertexDataUnreadable(VERTEX_ADDR))
        );
        // Non-resolvable destination (DXT1): copy dropped via zero extent.
        let (mut regs, mem) = setup(rect_vertices(0.0, 0.0, 64.0, 64.0));
        regs[r::RB_COPY_DEST_INFO as usize] =
            RbCopyDestInfo { copy_dest_format: TextureFormat::Dxt1, ..Default::default() }.encode();
        let info = resolve_info(&regs, &mem, &ResolveOptions::default()).unwrap();
        assert_eq!(info.copy_dest_extent_length, 0);
    }

    #[test]
    fn sample_select_sanitizing() {
        use CopySampleSelect as S;
        assert_eq!(sanitize_copy_sample_select(S::S3, MsaaSamples::X1, false), S::S0);
        assert_eq!(sanitize_copy_sample_select(S::S3, MsaaSamples::X2, false), S::S1);
        assert_eq!(sanitize_copy_sample_select(S::S0123, MsaaSamples::X2, false), S::S01);
        assert_eq!(sanitize_copy_sample_select(S::S01, MsaaSamples::X2, true), S::S0);
        assert_eq!(sanitize_copy_sample_select(S::Unknown(7), MsaaSamples::X4, false), S::S0123);
        assert_eq!(sanitize_copy_sample_select(S::S23, MsaaSamples::X4, true), S::S2);
        assert_eq!(sanitize_copy_sample_select(S::S23, MsaaSamples::X4, false), S::S23);
    }

    #[test]
    fn tile_span() {
        let e = ResolveEdramInfo { pitch_tiles: 16, base_tiles: 10, ..Default::default() };
        // 1x 32bpp: x 72..152 px -> tiles 0..2 columns, 8 rows -> 1 tile row.
        assert_eq!(resolve_edram_tile_span(&e, 9, 0, 10, 1), (10, 2, 1));
        assert_eq!(resolve_edram_tile_span(&e, 0, 1, 10, 2), (10, 1, 2));
    }
}
