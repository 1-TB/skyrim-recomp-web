//! EDRAM geometry and render target bindings (Xenia xenos.h, render_target_cache, edram.xesli).
//!
//! The 10 MiB EDRAM is 2048 tiles of 80x16 32-bit samples (5120 bytes each). With 2x MSAA a pixel is
//! 1x2 samples, with 4x 2x2 samples, so a 32bpp tile holds 80x16, 80x8 or 40x8 pixels. 64bpp formats
//! take two tiles horizontally per 80 samples (40x16 64-bit samples per tile). Bases are in tiles and
//! the pitch is rounded up to whole tiles. Addressing wraps modulo 2048 tiles.

use crate::RegisterSource;
use crate::bitfields::*;
use crate::xenos::*;

pub const EDRAM_TILE_WIDTH_SAMPLES: u32 = 80;
pub const EDRAM_TILE_HEIGHT_SAMPLES: u32 = 16;
pub const EDRAM_TILE_COUNT: u32 = 2048;
pub const EDRAM_TILE_SIZE_BYTES: u32 = EDRAM_TILE_WIDTH_SAMPLES * EDRAM_TILE_HEIGHT_SAMPLES * 4;
pub const EDRAM_SIZE_BYTES: u32 = EDRAM_TILE_COUNT * EDRAM_TILE_SIZE_BYTES;
/// Width of the base fields in tiles (addressing period).
pub const EDRAM_BASE_TILES_BITS: u32 = 11;
pub const EDRAM_PITCH_TILES_BITS: u32 = 10;

/// Surface pitch in tiles (GetSurfacePitchTiles), doubled for 64bpp.
pub const fn surface_pitch_tiles(pitch_pixels: u32, msaa: MsaaSamples, is_64bpp: bool) -> u32 {
    let pitch_samples = pitch_pixels << msaa.samples_x_log2();
    let tiles = pitch_samples.div_ceil(EDRAM_TILE_WIDTH_SAMPLES);
    if is_64bpp { tiles << 1 } else { tiles }
}

/// Pixel dimensions of one EDRAM tile for the MSAA mode and pixel size.
pub const fn tile_size_pixels(msaa: MsaaSamples, is_64bpp: bool) -> [u32; 2] {
    [
        (EDRAM_TILE_WIDTH_SAMPLES >> is_64bpp as u32) >> msaa.samples_x_log2(),
        EDRAM_TILE_HEIGHT_SAMPLES >> msaa.samples_y_log2(),
    ]
}

/// Adds a window offset in 32bpp tiles to a base, wrapping (AddWindowOffsetToEdramBase).
pub fn add_window_offset_to_edram_base(base_tiles: u32, window_offset_tiles: i32, is_64bpp: bool) -> u32 {
    let offset = window_offset_tiles * if is_64bpp { 2 } else { 1 };
    (base_tiles as i32).wrapping_add(offset) as u32 & (EDRAM_TILE_COUNT - 1)
}

/// Byte offset in EDRAM of a sample of a pixel, in Xenia's canonical EDRAM layout (XeEdramOffsetBytes).
///
/// Samples of a 1x/2x/4x view of the same allocation map to the same storage: for 4x, sample bit 0 is
/// horizontal and bit 1 vertical; for 2x sample 0 is the top one. Depth tiles have their 40-sample
/// halves swapped. `pitch_tiles` already includes the 64bpp doubling. `resolution_scale` is (1, 1) at
/// native resolution (sizes then are in scaled samples).
#[allow(clippy::too_many_arguments)]
pub fn edram_offset_bytes(
    pixel: [u32; 2],
    base_tiles: u32,
    wrap: bool,
    pitch_tiles: u32,
    msaa: MsaaSamples,
    is_depth: bool,
    is_64bpp: bool,
    sample_index: u32,
    resolution_scale: [u32; 2],
) -> u32 {
    let scale = [resolution_scale[0].max(1), resolution_scale[1].max(1)];
    let px = [pixel[0] / scale[0], pixel[1] / scale[1]];
    let sub = [pixel[0] - px[0] * scale[0], pixel[1] - px[1] * scale[1]];
    let guest_sample = match msaa {
        MsaaSamples::X4 => {
            let so = [sample_index & 1, (sample_index >> 1) & 1];
            [((px[0] & !1) << 1) | (px[0] & 1) | (so[0] << 1), ((px[1] & !1) << 1) | (px[1] & 1) | (so[1] << 1)]
        }
        MsaaSamples::X2 => [(px[0] & !2) | ((sample_index & 1) << 1), ((px[1] & !1) << 1) | (px[1] & 1) | (px[0] & 2)],
        _ => px,
    };
    let ints_log2 = is_64bpp as u32;
    let mut s = [guest_sample[0] * scale[0] + sub[0], guest_sample[1] * scale[1] + sub[1]];
    let tile_32 = [EDRAM_TILE_WIDTH_SAMPLES * scale[0], EDRAM_TILE_HEIGHT_SAMPLES * scale[1]];
    let tile = [tile_32[0] >> ints_log2, tile_32[1]];
    let tile_xy = [s[0] / tile[0], s[1] / tile[1]];
    let base = base_tiles.wrapping_add(tile_xy[1] * pitch_tiles + tile_xy[0]);
    s[0] -= tile_xy[0] * tile[0];
    s[1] -= tile_xy[1] * tile[1];
    if is_depth {
        let half = tile[0] >> 1;
        s[0] = if s[0] >= half { s[0] - half } else { s[0] + half };
    }
    let tile_dwords = tile_32[0] * tile_32[1];
    let mut address = base.wrapping_mul(tile_dwords).wrapping_add((s[1] * tile[0] + s[0]) << ints_log2);
    if wrap {
        address %= tile_dwords * EDRAM_TILE_COUNT;
    }
    address << 2
}

/// Packed identity of an EDRAM-backed render target, like Xenia's RenderTargetKey (base, 32bpp pitch,
/// MSAA, depth flag, format). Suitable as a host render target cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderTargetKey {
    pub base_tiles: u32,
    pub pitch_tiles_at_32bpp: u32,
    pub msaa_samples: MsaaSamples,
    pub is_depth: bool,
    /// Color format (storage format, aliases collapsed) or depth format raw value.
    pub format: u32,
}

impl RenderTargetKey {
    /// Color format (for color keys).
    pub fn color_format(&self) -> ColorRenderTargetFormat {
        ColorRenderTargetFormat::from_raw(self.format)
    }
    /// Depth format (for depth keys).
    pub fn depth_format(&self) -> DepthRenderTargetFormat {
        DepthRenderTargetFormat::from_raw(self.format)
    }
    pub fn is_64bpp(&self) -> bool {
        !self.is_depth && self.color_format().is_64bpp()
    }
    /// Row pitch in tiles (doubled for 64bpp).
    pub fn pitch_tiles(&self) -> u32 {
        self.pitch_tiles_at_32bpp << self.is_64bpp() as u32
    }
    /// Width in pixels of the tile-aligned surface.
    pub fn width_pixels(&self) -> u32 {
        self.pitch_tiles_at_32bpp * (EDRAM_TILE_WIDTH_SAMPLES >> self.msaa_samples.samples_x_log2())
    }
    /// Tile size in pixels.
    pub fn tile_size_pixels(&self) -> [u32; 2] {
        tile_size_pixels(self.msaa_samples, self.is_64bpp())
    }
    /// Largest height (in pixels) that fits in one EDRAM addressing period from the base.
    pub fn max_height_pixels(&self) -> u32 {
        let pitch = self.pitch_tiles();
        if pitch == 0 {
            return 0;
        }
        (EDRAM_TILE_COUNT / pitch) * self.tile_size_pixels()[1]
    }
    /// Tiles covered by the first `height_pixels` rows: `(first tile, tile count)`; tiles beyond 2048
    /// wrap to the beginning of EDRAM.
    pub fn tile_span(&self, height_pixels: u32) -> (u32, u32) {
        let rows = height_pixels.div_ceil(self.tile_size_pixels()[1]);
        (self.base_tiles, (rows * self.pitch_tiles()).min(EDRAM_TILE_COUNT))
    }
    /// Packs into a u32 like Xenia (base:11, pitch:8, msaa:2, depth:1, format:4).
    pub fn packed(&self) -> u32 {
        (self.base_tiles & 0x7FF)
            | ((self.pitch_tiles_at_32bpp & 0xFF) << 11)
            | ((self.msaa_samples.raw() & 3) << 19)
            | ((self.is_depth as u32) << 21)
            | ((self.format & 0xF) << 22)
    }
}

/// Whether two wrapped EDRAM tile spans `(start, count)` overlap.
pub fn tile_spans_overlap(a: (u32, u32), b: (u32, u32)) -> bool {
    if a.1 == 0 || b.1 == 0 {
        return false;
    }
    let contains = |span: (u32, u32), tile: u32| (tile.wrapping_sub(span.0) & (EDRAM_TILE_COUNT - 1)) < span.1;
    contains(a, b.0 & (EDRAM_TILE_COUNT - 1)) || contains(b, a.0 & (EDRAM_TILE_COUNT - 1))
}

/// Which attachment a binding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderTargetSlot {
    Depth,
    Color(u32),
}

/// An EDRAM render target used by the draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderTargetBinding {
    pub slot: RenderTargetSlot,
    /// Key with the window offset applied to the base.
    pub key: RenderTargetKey,
    /// Base from the register (without window offset).
    pub guest_base_tiles: u32,
    /// Format as written in the register (before alias collapsing), for color.
    pub guest_color_format: Option<ColorRenderTargetFormat>,
    /// Color exponent bias (shader output is multiplied by 2^bias).
    pub exp_bias: i32,
}

/// Render target setup failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderTargetError {
    /// MSAA mode 3 (invalid).
    InvalidMsaa(u32),
    /// Pitch above the 8192 texture limit.
    PitchTooLarge(u32),
}

/// The render targets a draw writes or tests against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderTargetSetup {
    pub edram_mode: EdramMode,
    pub surface_pitch: u32,
    pub msaa_samples: MsaaSamples,
    pub pitch_tiles_at_32bpp: u32,
    pub depth: Option<RenderTargetBinding>,
    pub color: [Option<RenderTargetBinding>; 4],
}

impl RenderTargetSetup {
    /// Builds the bindings. Depth is bound when the normalized depth control tests depth or stencil;
    /// color targets when their 4 bits in `normalized_color_mask` are non-zero. Nothing is bound when
    /// the surface pitch is zero (no rasterization possible).
    pub fn from_registers(
        regs: &(impl RegisterSource + ?Sized),
        normalized_depth_control: RbDepthcontrol,
        normalized_color_mask: u32,
        window_offset_tiles: i32,
    ) -> Result<Self, RenderTargetError> {
        let surface = RbSurfaceInfo::read(regs);
        let msaa = surface.msaa_samples;
        if let MsaaSamples::Unknown(v) = msaa {
            return Err(RenderTargetError::InvalidMsaa(v));
        }
        if surface.surface_pitch > TEXTURE_2D_MAX_WIDTH_HEIGHT {
            return Err(RenderTargetError::PitchTooLarge(surface.surface_pitch));
        }
        let pitch_tiles_at_32bpp = surface_pitch_tiles(surface.surface_pitch, msaa, false);
        let mut setup = Self {
            edram_mode: RbModecontrol::read(regs).edram_mode,
            surface_pitch: surface.surface_pitch,
            msaa_samples: msaa,
            pitch_tiles_at_32bpp,
            depth: None,
            color: [None; 4],
        };
        if surface.surface_pitch == 0 {
            return Ok(setup);
        }
        if normalized_depth_control.z_enable || normalized_depth_control.stencil_enable {
            let info = RbDepthInfo::read(regs);
            setup.depth = Some(RenderTargetBinding {
                slot: RenderTargetSlot::Depth,
                key: RenderTargetKey {
                    base_tiles: add_window_offset_to_edram_base(info.depth_base, window_offset_tiles, false),
                    pitch_tiles_at_32bpp,
                    msaa_samples: msaa,
                    is_depth: true,
                    format: info.depth_format.raw(),
                },
                guest_base_tiles: info.depth_base,
                guest_color_format: None,
                exp_bias: 0,
            });
        }
        for (rt, slot) in setup.color.iter_mut().enumerate() {
            if (normalized_color_mask >> (4 * rt)) & 0xF == 0 {
                continue;
            }
            let info = RbColorInfo::read_rt(regs, rt);
            let is_64bpp = info.color_format.is_64bpp();
            *slot = Some(RenderTargetBinding {
                slot: RenderTargetSlot::Color(rt as u32),
                key: RenderTargetKey {
                    base_tiles: add_window_offset_to_edram_base(info.color_base, window_offset_tiles, is_64bpp),
                    pitch_tiles_at_32bpp,
                    msaa_samples: msaa,
                    is_depth: false,
                    format: info.color_format.storage_format().raw(),
                },
                guest_base_tiles: info.color_base,
                guest_color_format: Some(info.color_format),
                exp_bias: info.color_exp_bias,
            });
        }
        Ok(setup)
    }

    /// Bound targets in attachment order (depth first).
    pub fn bindings(&self) -> impl Iterator<Item = &RenderTargetBinding> {
        self.depth.iter().chain(self.color.iter().flatten())
    }

    /// The same EDRAM ranges viewed with `msaa` samples: the surface pitch in pixels changes by
    /// the sample counts' horizontal ratio, the bases and pitch in tiles stay.
    pub fn with_msaa(&self, msaa: MsaaSamples) -> Self {
        let mut out = self.clone();
        out.surface_pitch = (self.surface_pitch << self.msaa_samples.samples_x_log2()) >> msaa.samples_x_log2();
        out.msaa_samples = msaa;
        for binding in out.depth.iter_mut().chain(out.color.iter_mut().flatten()) {
            binding.key.msaa_samples = msaa;
        }
        out
    }
}

/// Window offset (in 32bpp tiles) that can be moved from the vertices into the EDRAM bases
/// (GetWindowOffsetTiles). Returns `(offset_tiles, kept_in_viewport)`; a zero offset with
/// `kept_in_viewport` means the offset could be moved in principle but must stay in the viewport.
pub fn window_offset_tiles(
    regs: &(impl RegisterSource + ?Sized),
    normalized_depth_control: RbDepthcontrol,
    normalized_color_mask: u32,
    host_render_targets_used: bool,
) -> (i32, bool) {
    let wo = PaScWindowOffset::read(regs);
    let (wx, wy) = (wo.window_x_offset, wo.window_y_offset);
    if !PaSuScModeCntl::read(regs).vtx_window_offset_enable || wx > 0 || wy > 0 || (wx | wy) == 0 {
        return (0, false);
    }
    let depth_used = normalized_depth_control.z_enable || normalized_depth_control.stencil_enable;
    if !depth_used && normalized_color_mask == 0 {
        return (0, false);
    }
    if (wx != 0 && host_render_targets_used) || PaScWindowScissorTl::read(regs).window_offset_disable {
        return (0, true);
    }
    let surface = RbSurfaceInfo::read(regs);
    if surface.surface_pitch == 0 {
        return (0, true);
    }
    let sx = surface.msaa_samples.samples_x_log2();
    let sy = surface.msaa_samples.samples_y_log2();
    let tile_w = (EDRAM_TILE_WIDTH_SAMPLES >> sx) as i32;
    let tile_h = (EDRAM_TILE_HEIGHT_SAMPLES >> sy) as i32;
    if wx % tile_w != 0 || wy % tile_h != 0 {
        return (0, true);
    }
    let pitch_tiles = surface_pitch_tiles(surface.surface_pitch, surface.msaa_samples, false);
    let offset_tiles = (wy / tile_h) * pitch_tiles as i32 + wx / tile_w;
    if host_render_targets_used {
        let screen_br = PaScScreenScissor::decode(regs.reg(crate::registers::PA_SC_SCREEN_SCISSOR_BR));
        let screen_tl = PaScScreenScissor::decode(regs.reg(crate::registers::PA_SC_SCREEN_SCISSOR_TL));
        let bottom = (PaScWindowScissorBr::read(regs).br_y as i32).min(screen_br.y - wy);
        let rows_end_32 = ((bottom + tile_h - 1) / tile_h) as u32 * pitch_tiles;
        if rows_end_32 > EDRAM_TILE_COUNT {
            return (0, true);
        }
        let rows_end_64 = rows_end_32 * 2;
        if rows_end_64 > EDRAM_TILE_COUNT {
            let top = (PaScWindowScissorTl::read(regs).tl_y as i32).max(screen_tl.y - wy);
            let rows_start_64 = (top / tile_h) as u32 * pitch_tiles * 2;
            if rows_end_64 > EDRAM_TILE_COUNT * 2 || rows_start_64 < EDRAM_TILE_COUNT {
                for rt in 0..4 {
                    if normalized_color_mask & (0xF << (4 * rt)) != 0
                        && RbColorInfo::read_rt(regs, rt).color_format.is_64bpp()
                    {
                        return (0, true);
                    }
                }
            }
        }
    }
    (offset_tiles, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registers as r;

    #[test]
    fn geometry_constants() {
        assert_eq!(EDRAM_TILE_SIZE_BYTES, 5120);
        assert_eq!(EDRAM_SIZE_BYTES, 10 * 1024 * 1024);
        assert_eq!(surface_pitch_tiles(1280, MsaaSamples::X1, false), 16);
        assert_eq!(surface_pitch_tiles(1280, MsaaSamples::X4, false), 32);
        assert_eq!(surface_pitch_tiles(1280, MsaaSamples::X2, true), 32);
        assert_eq!(surface_pitch_tiles(256, MsaaSamples::X1, false), 4);
        assert_eq!(tile_size_pixels(MsaaSamples::X1, false), [80, 16]);
        assert_eq!(tile_size_pixels(MsaaSamples::X2, false), [80, 8]);
        assert_eq!(tile_size_pixels(MsaaSamples::X4, false), [40, 8]);
        assert_eq!(tile_size_pixels(MsaaSamples::X4, true), [20, 8]);
        assert_eq!(add_window_offset_to_edram_base(10, -16, false), 2042);
        assert_eq!(add_window_offset_to_edram_base(10, -16, true), 2026);
        assert_eq!(add_window_offset_to_edram_base(2040, 16, false), 8);
    }

    #[test]
    fn edram_addressing() {
        let one = [1, 1];
        // 1x 32bpp: row-major within a tile, tiles row-major with the pitch.
        assert_eq!(edram_offset_bytes([0, 0], 5, false, 16, MsaaSamples::X1, false, false, 0, one), 5 * 5120);
        assert_eq!(edram_offset_bytes([3, 2], 0, false, 16, MsaaSamples::X1, false, false, 0, one), (2 * 80 + 3) * 4);
        assert_eq!(
            edram_offset_bytes([81, 17], 0, false, 16, MsaaSamples::X1, false, false, 0, one),
            (17 * 5120) + (80 + 1) * 4
        );
        // Depth swaps tile halves.
        assert_eq!(edram_offset_bytes([0, 0], 0, false, 16, MsaaSamples::X1, true, false, 0, one), 40 * 4);
        assert_eq!(edram_offset_bytes([45, 0], 0, false, 16, MsaaSamples::X1, true, false, 0, one), 5 * 4);
        // 64bpp: 40 samples per tile row, 8 bytes each.
        assert_eq!(edram_offset_bytes([41, 0], 0, false, 32, MsaaSamples::X1, false, true, 0, one), 5120 + 8);
        // 2x: sample 0 is the top one, sample 1 is two rows down... in canonical sample coordinates.
        // Pixel (0, 0): sample 0 -> (0, 0); sample 1 -> u = 2.
        assert_eq!(edram_offset_bytes([0, 0], 0, false, 16, MsaaSamples::X2, false, false, 1, one), 2 * 4);
        // Pixel (2, 1): u = 0, v = (0 << 1) | 1 | 2 = 3.
        assert_eq!(edram_offset_bytes([2, 1], 0, false, 16, MsaaSamples::X2, false, false, 0, one), (3 * 80) * 4);
        // 4x: pixel (1, 1) sample 3 -> u = 1 | 2 = 3, v = 1 | 2 = 3.
        assert_eq!(edram_offset_bytes([1, 1], 0, false, 32, MsaaSamples::X4, false, false, 3, one), (3 * 80 + 3) * 4);
        // Wrapping.
        assert_eq!(edram_offset_bytes([0, 16], 2047, true, 1, MsaaSamples::X1, false, false, 0, one), 0);
        assert_eq!(
            edram_offset_bytes([0, 16], 2047, false, 1, MsaaSamples::X1, false, false, 0, one),
            EDRAM_SIZE_BYTES
        );
        // Resolution scale 2: tiles are 160x32 scaled samples.
        assert_eq!(
            edram_offset_bytes([1, 1], 1, false, 16, MsaaSamples::X1, false, false, 0, [2, 2]),
            (5120 + 160 + 1) * 4
        );
    }

    #[test]
    fn keys_and_spans() {
        let key = RenderTargetKey {
            base_tiles: 2000,
            pitch_tiles_at_32bpp: 16,
            msaa_samples: MsaaSamples::X2,
            is_depth: false,
            format: ColorRenderTargetFormat::K16161616Float.raw(),
        };
        assert!(key.is_64bpp());
        assert_eq!(key.pitch_tiles(), 32);
        assert_eq!(key.width_pixels(), 1280);
        assert_eq!(key.tile_size_pixels(), [40, 8]);
        assert_eq!(key.tile_span(720), (2000, 2048));
        assert_eq!(key.tile_span(80), (2000, 10 * 32));
        assert_eq!(key.tile_span(8), (2000, 32));
        assert_eq!(key.max_height_pixels(), 64 * 8);
        assert_eq!(key.packed(), 2000 | (16 << 11) | (1 << 19) | (7 << 22));
        assert!(tile_spans_overlap((2000, 100), (0, 10)));
        assert!(!tile_spans_overlap((2000, 48), (0, 10)));
        assert!(tile_spans_overlap((5, 10), (14, 1)));
        assert!(!tile_spans_overlap((5, 10), (15, 1)));
        assert!(!tile_spans_overlap((5, 0), (5, 1)));
    }

    #[test]
    fn setup_from_registers() {
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_SURFACE_INFO as usize] = 1280 | (2 << 16);
        regs[r::RB_MODECONTROL as usize] = 4;
        regs[r::RB_DEPTH_INFO as usize] = 300 | (1 << 16);
        regs[r::RB_COLOR_INFO as usize] = (ColorRenderTargetFormat::K2101010As10101010.raw() << 16) | (2 << 20);
        regs[r::RB_COLOR1_INFO as usize] = 600 | (ColorRenderTargetFormat::K16161616.raw() << 16);
        let dc = RbDepthcontrol { z_enable: true, ..Default::default() };
        let s = RenderTargetSetup::from_registers(&regs, dc, 0x00FF, -32).unwrap();
        assert_eq!(s.pitch_tiles_at_32bpp, 32);
        let d = s.depth.unwrap();
        assert_eq!((d.key.base_tiles, d.guest_base_tiles), (300 - 32, 300));
        assert_eq!(d.key.depth_format(), DepthRenderTargetFormat::D24FS8);
        let c0 = s.color[0].unwrap();
        assert_eq!(c0.key.base_tiles, 2048 - 32);
        assert_eq!(c0.key.color_format(), ColorRenderTargetFormat::K2101010);
        assert_eq!(c0.guest_color_format, Some(ColorRenderTargetFormat::K2101010As10101010));
        assert_eq!(c0.exp_bias, 2);
        assert_eq!(s.color[1].unwrap().key.base_tiles, 600 - 64);
        assert!(s.color[2].is_none());
        assert_eq!(s.bindings().count(), 3);

        regs[r::RB_SURFACE_INFO as usize] = 1280 | (3 << 16);
        assert_eq!(RenderTargetSetup::from_registers(&regs, dc, 0, 0), Err(RenderTargetError::InvalidMsaa(3)));
        regs[r::RB_SURFACE_INFO as usize] = 0;
        assert_eq!(RenderTargetSetup::from_registers(&regs, dc, 0xF, 0).unwrap().bindings().count(), 0);
    }

    #[test]
    fn window_offset_to_tiles() {
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_SURFACE_INFO as usize] = 1280 | (1 << 16);
        regs[r::PA_SU_SC_MODE_CNTL as usize] = 1 << 16;
        regs[r::PA_SC_WINDOW_OFFSET as usize] = PaScWindowOffset { window_x_offset: 0, window_y_offset: -360 }.encode();
        regs[r::PA_SC_WINDOW_SCISSOR_BR as usize] = PaScWindowScissorBr { br_x: 1280, br_y: 360 }.encode();
        regs[r::PA_SC_SCREEN_SCISSOR_BR as usize] = PaScScreenScissor { x: 8192, y: 8192 }.encode();
        let dc = RbDepthcontrol { z_enable: true, ..Default::default() };
        // 2x MSAA: tile height 8 pixels, pitch 16 tiles: -360 / 8 * 16 = -720.
        assert_eq!(window_offset_tiles(&regs, dc, 0, false), (-720, false));
        assert_eq!(window_offset_tiles(&regs, dc, 0, true), (-720, false));
        assert_eq!(window_offset_tiles(&regs, RbDepthcontrol::default(), 0, false), (0, false));
        // Not tile-aligned.
        regs[r::PA_SC_WINDOW_OFFSET as usize] = PaScWindowOffset { window_x_offset: 0, window_y_offset: -4 }.encode();
        assert_eq!(window_offset_tiles(&regs, dc, 0, false), (0, true));
        // Horizontal offsets stay in the viewport with host render targets.
        regs[r::PA_SC_WINDOW_OFFSET as usize] = PaScWindowOffset { window_x_offset: -80, window_y_offset: 0 }.encode();
        assert_eq!(window_offset_tiles(&regs, dc, 0, true), (0, true));
        assert_eq!(window_offset_tiles(&regs, dc, 0, false), (-1, false));
        // Positive offsets are never moved.
        regs[r::PA_SC_WINDOW_OFFSET as usize] = PaScWindowOffset { window_x_offset: 80, window_y_offset: 0 }.encode();
        assert_eq!(window_offset_tiles(&regs, dc, 0, false), (0, false));
    }
}
