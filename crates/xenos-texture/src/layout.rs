//! Guest memory layout of textures: subresource selection from fetch constants, packed mip
//! tails, per-level strides and exact memory extents (Xenia `texture_util.cc`).

use crate::address::{
    LINEAR_ROW_ALIGNMENT_BYTES, SUBRESOURCE_ALIGNMENT_BYTES, TILE_DEPTH, TILE_WIDTH_HEIGHT,
    tiled_address_upper_bound_2d, tiled_address_upper_bound_3d,
};
use crate::fetch::{DataDimension, TextureFetchConstant};
use crate::format::TextureFormat;

/// Maximum number of mip levels (2D/cube 8192 -> 14 levels).
pub const MAX_MIPS: usize = 14;
pub const MAX_2D_SIZE: u32 = 8192;
/// Wide 1D textures are mapped to rows of this many texels, with at most this many rows.
pub const WIDE_1D_MAX_ROWS: u32 = 128;
/// Guest physical memory is 512 MB.
pub const PHYSICAL_ADDRESS_MASK: u32 = 0x1FFF_FFFF;

fn align(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}

fn log2_ceil(v: u32) -> u32 {
    if v <= 1 { 0 } else { 32 - (v - 1).leading_zeros() }
}

fn log2_floor(v: u32) -> u32 {
    if v == 0 { 0 } else { 31 - v.leading_zeros() }
}

/// Level at which the packed mip tail starts for a texture of the given base size.
pub fn packed_mip_level(width: u32, height: u32) -> u32 {
    let log2_size = log2_ceil(width.min(height));
    log2_size.saturating_sub(4)
}

/// Offset in blocks (x, y) and slices (z) of `mip` inside the packed tail, or `None` if the
/// level is not packed. Sizes are in texels; use depth 1 for non-3D textures.
pub fn packed_mip_offset(
    width: u32,
    height: u32,
    depth: u32,
    format: TextureFormat,
    mip: u32,
) -> Option<(u32, u32, u32)> {
    let log2_width = log2_ceil(width);
    let log2_height = log2_ceil(height);
    let log2_size = log2_width.min(log2_height);
    if log2_size > 4 + mip {
        return None;
    }
    let packed_mip_base = log2_size.saturating_sub(4);
    let packed_mip = mip - packed_mip_base;
    let (mut x, mut y, z);
    if packed_mip < 3 {
        if log2_width > log2_height {
            x = 0;
            y = 16 >> packed_mip;
        } else {
            x = 16 >> packed_mip;
            y = 0;
        }
        z = 0;
    } else {
        let offset;
        if log2_width > log2_height {
            offset = (1u32 << (log2_width - packed_mip_base)) >> (packed_mip - 2);
            x = offset;
            y = 0;
        } else {
            offset = (1u32 << (log2_height - packed_mip_base)) >> (packed_mip - 2);
            x = 0;
            y = offset;
        }
        z = if offset < 4 {
            let log2_depth = log2_ceil(depth);
            if log2_depth > 1 + mip { (log2_depth - mip) * 4 } else { 4 }
        } else {
            0
        };
    }
    let info = format.info();
    x >>= info.width_shift();
    y >>= info.height_shift();
    Some((x, y, z))
}

/// Subresources referenced by a fetch constant after Xenia's cleanup rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSubresources {
    pub dimension: DataDimension,
    pub width: u32,
    pub height: u32,
    /// Depth for 3D, array size for stacked 2D, 6 for cube, 1 otherwise.
    pub depth_or_array_size: u32,
    /// Base row pitch in texels / 32 used for the layout.
    pub pitch_div_32: u32,
    /// Page (address >> 12) of level 0, or 0 if the base level is absent.
    pub base_page: u32,
    /// Page of the mip levels, or 0 if there are none.
    pub mip_page: u32,
    pub mip_min_level: u32,
    pub mip_max_level: u32,
    /// A 1D texture wider than 8192 mapped to rows of 8192 texels (`height` rows).
    pub wide_1d: bool,
}

impl TextureSubresources {
    /// Port of `GetSubresourcesFromFetchConstant` plus the wide-1D and pitch handling of
    /// `BindingInfoFromFetchConstant`.
    pub fn from_fetch(fetch: &TextureFetchConstant) -> Self {
        let (mut width, mut height, depth_or_array_size) = match fetch.dimension {
            DataDimension::D1 => (fetch.width, 1, 1),
            DataDimension::D2OrStacked => (fetch.width, fetch.height, if fetch.stacked { fetch.depth } else { 1 }),
            DataDimension::D3 => (fetch.width, fetch.height, fetch.depth),
            DataDimension::Cube => (fetch.width, fetch.height, 6),
        };
        let mut longest = width.max(height);
        if fetch.dimension == DataDimension::D3 {
            longest = longest.max(depth_or_array_size);
        }
        let size_mip_max_level = log2_floor(longest);

        let mut base_page = fetch.base_page & 0x1FFFF;
        let mut mip_page = fetch.mip_page & 0x1FFFF;
        if mip_page == 0
            && base_page != 0
            && fetch.packed_mips
            && fetch.dimension != DataDimension::D1
            && packed_mip_level(width, height) == 0
        {
            mip_page = base_page;
        }
        let (mut mip_min_level, mut mip_max_level) = if mip_page == 0 {
            (0, 0)
        } else {
            let min = fetch.mip_min_level.min(size_mip_max_level);
            (min, fetch.mip_max_level.min(size_mip_max_level).max(min))
        };
        if mip_max_level != 0 {
            if mip_min_level != 0 && base_page == mip_page {
                base_page = 0;
            }
            if base_page == 0 {
                mip_min_level = mip_min_level.max(1);
            }
        } else {
            mip_page = 0;
        }

        let mut pitch_div_32 = fetch.pitch_div_32;
        let mut wide_1d = false;
        if fetch.dimension == DataDimension::D1 && width > MAX_2D_SIZE {
            let rows = width.div_ceil(MAX_2D_SIZE).min(WIDE_1D_MAX_ROWS);
            width = MAX_2D_SIZE;
            height = rows;
            mip_max_level = 0;
            mip_min_level = 0;
            mip_page = 0;
            pitch_div_32 = MAX_2D_SIZE >> 5;
            wide_1d = true;
        }
        TextureSubresources {
            dimension: fetch.dimension,
            width,
            height,
            depth_or_array_size,
            pitch_div_32,
            base_page,
            mip_page,
            mip_min_level,
            mip_max_level,
            wide_1d,
        }
    }

    pub fn has_base(&self) -> bool {
        self.base_page != 0
    }

    pub fn depth(&self) -> u32 {
        if self.dimension == DataDimension::D3 { self.depth_or_array_size } else { 1 }
    }

    pub fn array_size(&self) -> u32 {
        match self.dimension {
            DataDimension::D2OrStacked => self.depth_or_array_size,
            DataDimension::Cube => 6,
            _ => 1,
        }
    }
}

/// Layout of one stored level (the base, a mip, or a packed mip tail).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LevelLayout {
    /// Distance between block rows in bytes (power-of-two and 256-byte alignment included).
    pub row_pitch_bytes: u32,
    /// Distance between Z slices in block rows.
    pub z_slice_stride_block_rows: u32,
    /// Distance between array slices (4 KB aligned).
    pub array_slice_stride_bytes: u32,
    /// Exclusive upper bound of blocks used (all packed levels for a tail).
    pub x_extent_blocks: u32,
    pub y_extent_blocks: u32,
    pub z_extent: u32,
    /// Bytes actually referenced by one array slice (not 4 KB aligned).
    pub array_slice_data_extent_bytes: u32,
    /// Bytes referenced by all array slices.
    pub level_data_extent_bytes: u32,
}

/// Port of `TextureGuestLayout`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestLayout {
    pub base: LevelLayout,
    /// Indexed by stored level: `1..=min(max_level, packed_level)`, or only `0` when the
    /// packed tail is level 0 (the tail for the mips is then stored like a level 0).
    pub mips: [LevelLayout; MAX_MIPS],
    pub mip_offsets_bytes: [u32; MAX_MIPS],
    pub mips_total_extent_bytes: u32,
    pub max_level: u32,
    /// `u32::MAX` when there is no packed mip tail.
    pub packed_level: u32,
    pub array_size: u32,
}

/// Parameters of [`guest_layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutParams {
    pub dimension: DataDimension,
    pub base_pitch_texels_div_32: u32,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_size: u32,
    pub tiled: bool,
    pub format: TextureFormat,
    pub packed_mips: bool,
    pub has_base: bool,
    pub max_level: u32,
}

impl GuestLayout {
    pub fn packed_level(&self) -> Option<u32> {
        (self.packed_level != u32::MAX).then_some(self.packed_level)
    }

    /// Index into `mips` holding host level `level` (>= 1).
    pub fn stored_mip_index(&self, level: u32) -> usize {
        level.min(self.packed_level) as usize
    }
}

/// Port of `GetGuestTextureLayout`.
pub fn guest_layout(p: &LayoutParams) -> GuestLayout {
    let mut layout = GuestLayout::default();
    let mut height_texels = p.height;
    if p.dimension == DataDimension::D1 && height_texels <= 1 {
        height_texels = 1;
    }
    let width_texels = p.width;
    let depth = if p.dimension == DataDimension::D3 { p.depth_or_array_size } else { 1 };
    layout.array_size = match p.dimension {
        DataDimension::D2OrStacked => p.depth_or_array_size,
        DataDimension::Cube => 6,
        _ => 1,
    };
    layout.packed_level = u32::MAX;
    let info = p.format.info();
    let bytes_per_block = info.bytes_per_block();
    if width_texels == 0 || height_texels == 0 || depth == 0 || layout.array_size == 0 || bytes_per_block == 0 {
        return GuestLayout { packed_level: u32::MAX, ..GuestLayout::default() };
    }
    let mip_array_size = if p.dimension == DataDimension::D2OrStacked && layout.array_size > 1 {
        align(layout.array_size, TILE_DEPTH)
    } else {
        layout.array_size
    };
    let max_level_for_dimensions = log2_floor(width_texels.max(height_texels).max(depth));
    let max_level = p.max_level.min(max_level_for_dimensions);
    layout.max_level = max_level;
    layout.packed_level = if p.packed_mips { packed_mip_level(width_texels, height_texels) } else { u32::MAX };
    let max_stored_level = max_level.min(layout.packed_level);

    let (bw, bh) = (info.block_width, info.block_height);
    let loop_level_last = if layout.packed_level == 0 { u32::from(max_level != 0) } else { max_stored_level };
    let mut mip_offset_bytes = 0u32;
    let first = if p.has_base { 0 } else { 1 };
    for loop_level in first..=loop_level_last {
        let is_base = loop_level == 0;
        let level = if layout.packed_level == 0 { 0 } else { loop_level };

        let (row_pitch_texels_unaligned, z_slice_stride_texel_rows_unaligned) = if is_base {
            (
                p.base_pitch_texels_div_32 << 5,
                if layout.packed_level == 0 { height_texels.next_power_of_two() } else { height_texels },
            )
        } else {
            ((width_texels.next_power_of_two() >> level).max(1), (height_texels.next_power_of_two() >> level).max(1))
        };
        let row_pitch_blocks_alignment = if !p.tiled && !is_base {
            (LINEAR_ROW_ALIGNMENT_BYTES / bytes_per_block).max(TILE_WIDTH_HEIGHT)
        } else {
            TILE_WIDTH_HEIGHT
        };
        let row_pitch_blocks_tile_aligned =
            align(align(row_pitch_texels_unaligned, bw) / bw, row_pitch_blocks_alignment);
        let mut l = LevelLayout {
            row_pitch_bytes: row_pitch_blocks_tile_aligned * bytes_per_block,
            z_slice_stride_block_rows: if p.dimension != DataDimension::D1 {
                align(align(z_slice_stride_texel_rows_unaligned, bh) / bh, TILE_WIDTH_HEIGHT)
            } else {
                1
            },
            ..LevelLayout::default()
        };
        l.array_slice_stride_bytes = l.row_pitch_bytes * l.z_slice_stride_block_rows;
        let z_stride_bytes = l.array_slice_stride_bytes;
        if p.dimension == DataDimension::D3 {
            let level_depth =
                if is_base && layout.packed_level != 0 { depth } else { (depth.next_power_of_two() >> level).max(1) };
            l.array_slice_stride_bytes *= align(level_depth, TILE_DEPTH);
        }
        l.array_slice_stride_bytes = align(l.array_slice_stride_bytes, SUBRESOURCE_ALIGNMENT_BYTES);

        if level == layout.packed_level {
            let last = if is_base { 0 } else { max_level };
            for sub in layout.packed_level..=last {
                let (ox, oy, oz) =
                    packed_mip_offset(width_texels, height_texels, depth, p.format, sub).unwrap_or((0, 0, 0));
                l.x_extent_blocks = l.x_extent_blocks.max(ox + (width_texels >> sub).max(1).div_ceil(bw));
                l.y_extent_blocks = l.y_extent_blocks.max(oy + (height_texels >> sub).max(1).div_ceil(bh));
                l.z_extent = l.z_extent.max(oz + (depth >> sub).max(1));
            }
        } else {
            l.x_extent_blocks = (width_texels >> level).max(1).div_ceil(bw);
            l.y_extent_blocks = (height_texels >> level).max(1).div_ceil(bh);
            l.z_extent = (depth >> level).max(1);
        }
        if p.tiled {
            let log2 = info.bytes_per_block_log2();
            l.array_slice_data_extent_bytes = if p.dimension == DataDimension::D3 {
                tiled_address_upper_bound_3d(
                    l.x_extent_blocks,
                    l.y_extent_blocks,
                    l.z_extent,
                    row_pitch_blocks_tile_aligned,
                    l.z_slice_stride_block_rows,
                    log2,
                ) as u32
            } else {
                tiled_address_upper_bound_2d(l.x_extent_blocks, l.y_extent_blocks, row_pitch_blocks_tile_aligned, log2)
            };
        } else {
            l.array_slice_data_extent_bytes = z_stride_bytes * (l.z_extent - 1)
                + l.row_pitch_bytes * (l.y_extent_blocks - 1)
                + bytes_per_block * l.x_extent_blocks;
        }
        l.level_data_extent_bytes =
            l.array_slice_stride_bytes * (layout.array_size - 1) + l.array_slice_data_extent_bytes;

        if is_base {
            layout.base = l;
        } else {
            layout.mips[level as usize] = l;
            layout.mip_offsets_bytes[level as usize] = mip_offset_bytes;
            layout.mips_total_extent_bytes =
                layout.mips_total_extent_bytes.max(mip_offset_bytes + l.level_data_extent_bytes);
            mip_offset_bytes += l.array_slice_stride_bytes * mip_array_size;
        }
    }
    layout
}

/// Which memory region a [`GuestRange`] belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeKind {
    Base,
    /// A stored mip level (or the packed mip tail) by stored index.
    Mip(u32),
}

/// A physical guest memory range read by a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestRange {
    pub kind: RangeKind,
    pub address: u32,
    pub length: u32,
}

/// Fully resolved addressing information for a fetch constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureGuestInfo {
    pub subresources: TextureSubresources,
    pub format: TextureFormat,
    pub tiled: bool,
    pub packed_mips: bool,
    pub layout: GuestLayout,
}

impl TextureGuestInfo {
    pub fn from_fetch(fetch: &TextureFetchConstant) -> Self {
        let sub = TextureSubresources::from_fetch(fetch);
        let format = fetch.format.base_format();
        let packed_mips = fetch.packed_mips && sub.dimension != DataDimension::D1;
        let layout = guest_layout(&LayoutParams {
            dimension: sub.dimension,
            base_pitch_texels_div_32: sub.pitch_div_32,
            width: sub.width,
            height: sub.height,
            depth_or_array_size: sub.depth_or_array_size,
            tiled: fetch.tiled && sub.dimension != DataDimension::D1,
            format,
            packed_mips,
            has_base: sub.has_base(),
            max_level: sub.mip_max_level,
        });
        TextureGuestInfo { subresources: sub, format, tiled: fetch.tiled, packed_mips, layout }
    }

    /// Guest byte range of the stored level holding host level `level`, and its layout.
    pub fn level_storage(&self, level: u32) -> Option<(u32, &LevelLayout, RangeKind)> {
        let sub = &self.subresources;
        if level == 0 {
            sub.has_base().then_some((
                (sub.base_page << 12) & PHYSICAL_ADDRESS_MASK,
                &self.layout.base,
                RangeKind::Base,
            ))
        } else {
            if sub.mip_page == 0 || level > self.layout.max_level {
                return None;
            }
            let idx = self.layout.stored_mip_index(level);
            let addr = (sub.mip_page << 12).wrapping_add(self.layout.mip_offsets_bytes[idx]) & PHYSICAL_ADDRESS_MASK;
            Some((addr, &self.layout.mips[idx], RangeKind::Mip(idx as u32)))
        }
    }

    /// Exact guest physical memory ranges the texture's loadable levels reference.
    pub fn memory_ranges(&self) -> Vec<GuestRange> {
        let mut out = Vec::new();
        let sub = &self.subresources;
        if sub.has_base() && self.layout.base.level_data_extent_bytes != 0 {
            out.push(GuestRange {
                kind: RangeKind::Base,
                address: (sub.base_page << 12) & PHYSICAL_ADDRESS_MASK,
                length: self.layout.base.level_data_extent_bytes,
            });
        }
        if sub.mip_page != 0 {
            let mut last_idx = usize::MAX;
            for level in 1..=self.layout.max_level {
                let idx = self.layout.stored_mip_index(level);
                if idx == last_idx {
                    continue;
                }
                last_idx = idx;
                let l = &self.layout.mips[idx];
                if l.level_data_extent_bytes != 0 {
                    out.push(GuestRange {
                        kind: RangeKind::Mip(idx as u32),
                        address: (sub.mip_page << 12).wrapping_add(self.layout.mip_offsets_bytes[idx])
                            & PHYSICAL_ADDRESS_MASK,
                        length: l.level_data_extent_bytes,
                    });
                }
            }
        }
        out
    }

    /// Total bytes from the base address and from the mip address (Xenia's
    /// `GetTextureTotalSize` with the extents used for loading).
    pub fn total_sizes(&self) -> (u32, u32) {
        let base = if self.subresources.has_base() { self.layout.base.level_data_extent_bytes } else { 0 };
        let mips = if self.subresources.mip_page != 0 { self.layout.mips_total_extent_bytes } else { 0 };
        (base, mips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::TextureFetchConstant;
    use TextureFormat::*;

    fn params(format: TextureFormat, w: u32, h: u32, tiled: bool, packed: bool, max_level: u32) -> LayoutParams {
        LayoutParams {
            dimension: DataDimension::D2OrStacked,
            base_pitch_texels_div_32: w.div_ceil(32),
            width: w,
            height: h,
            depth_or_array_size: 1,
            tiled,
            format,
            packed_mips: packed,
            has_base: true,
            max_level,
        }
    }

    #[test]
    fn packed_mip_levels() {
        assert_eq!(packed_mip_level(512, 512), 5);
        assert_eq!(packed_mip_level(16, 16), 0);
        assert_eq!(packed_mip_level(17, 1024), 1);
        assert_eq!(packed_mip_level(1280, 720), 6);
        assert_eq!(packed_mip_level(1, 1), 0);
    }

    #[test]
    fn packed_mip_offsets_square() {
        // 64x64: packed from level 2 (16x16).
        assert_eq!(packed_mip_offset(64, 64, 1, k_8_8_8_8, 1), None);
        assert_eq!(packed_mip_offset(64, 64, 1, k_8_8_8_8, 2), Some((16, 0, 0)));
        assert_eq!(packed_mip_offset(64, 64, 1, k_8_8_8_8, 3), Some((8, 0, 0)));
        assert_eq!(packed_mip_offset(64, 64, 1, k_8_8_8_8, 4), Some((4, 0, 0)));
        // 2x2: offset = (1 << (6 - 2)) >> 1 = 8 along Y; z = 4 (1x1 Z packing rule).
        assert_eq!(packed_mip_offset(64, 64, 1, k_8_8_8_8, 5), Some((0, 8, 0)));
        assert_eq!(packed_mip_offset(64, 64, 1, k_8_8_8_8, 6), Some((0, 4, 0)));
        assert_eq!(packed_mip_offset(64, 64, 1, k_DXT1, 2), Some((4, 0, 0)));
        assert_eq!(packed_mip_offset(64, 64, 1, k_DXT1, 5), Some((0, 2, 0)));
        assert_eq!(packed_mip_offset(64, 64, 1, k_DXT1, 6), Some((0, 1, 0)));
        // Base itself packed (16x16 texture).
        assert_eq!(packed_mip_offset(16, 16, 1, k_8, 0), Some((16, 0, 0)));
    }

    #[test]
    fn packed_mip_offsets_wide_and_3d() {
        // 128x32: log2 w=7 > h=5 -> laid out vertically for the first three packed mips.
        assert_eq!(packed_mip_offset(128, 32, 1, k_8, 1), Some((0, 16, 0)));
        assert_eq!(packed_mip_offset(128, 32, 1, k_8, 2), Some((0, 8, 0)));
        // packed_mip 3: (1 << (7 - 1)) >> 1 = 32 along X.
        assert_eq!(packed_mip_offset(128, 32, 1, k_8, 4), Some((32, 0, 0)));
        // 3D 16x16x16: level 3 (2x2x2): packed_mip 3 -> offset (1<<4)>>1 = 8 >= 4, z = 0.
        assert_eq!(packed_mip_offset(16, 16, 16, k_8, 3), Some((0, 8, 0)));
        // Level 4 (1x1x1): offset 4 -> not < 4, z = 0. Level 5 doesn't exist, but offset 2 -> z rule.
        assert_eq!(packed_mip_offset(16, 16, 16, k_8, 4), Some((0, 4, 0)));
        assert_eq!(packed_mip_offset(16, 16, 16, k_8, 5), Some((0, 2, 4)));
        assert_eq!(packed_mip_offset(16, 16, 64, k_8, 5), Some((0, 2, 4)));
        assert_eq!(packed_mip_offset(16, 16, 256, k_8, 5), Some((0, 2, 12)));
    }

    #[test]
    fn layout_256_rgba8_tiled_packed() {
        let l = guest_layout(&params(k_8_8_8_8, 256, 256, true, true, 8));
        assert_eq!(l.packed_level, 4);
        assert_eq!(l.base.row_pitch_bytes, 1024);
        assert_eq!(l.base.z_slice_stride_block_rows, 256);
        assert_eq!(l.base.array_slice_stride_bytes, 262144);
        assert_eq!(l.mips[1].row_pitch_bytes, 512);
        assert_eq!(l.mips[1].array_slice_stride_bytes, 65536);
        assert_eq!(l.mips[2].array_slice_stride_bytes, 16384);
        assert_eq!(l.mips[3].array_slice_stride_bytes, 4096);
        assert_eq!(l.mips[4].array_slice_stride_bytes, 4096);
        assert_eq!(&l.mip_offsets_bytes[1..5], &[0, 65536, 81920, 86016]);
        // Tail extent: 16x16 at x=16 -> 32 blocks wide, 1x1 at y=4 etc.
        assert_eq!((l.mips[4].x_extent_blocks, l.mips[4].y_extent_blocks), (32, 16));
        assert_eq!(l.mips[4].array_slice_data_extent_bytes, 0x1000);
        assert_eq!(l.mips_total_extent_bytes, 86016 + 4096);
        // Tiled base extent: 256x256 32bpp = exactly 256 KB.
        assert_eq!(l.base.level_data_extent_bytes, 262144);
        assert_eq!(l.mips[5], LevelLayout::default());
    }

    #[test]
    fn layout_linear_exact_extent() {
        // 1280x720 linear k_8_8_8_8 must not include the 32-row padding (4E4D083E case).
        let l = guest_layout(&params(k_8_8_8_8, 1280, 720, false, false, 0));
        assert_eq!(l.base.row_pitch_bytes, 5120);
        assert_eq!(l.base.z_slice_stride_block_rows, 736);
        assert_eq!(l.base.array_slice_data_extent_bytes, 1280 * 720 * 4);
        // Linear mips are aligned to 256 bytes per row (max(256 / bpb, 32) blocks).
        let l = guest_layout(&params(k_DXT1, 64, 64, false, false, 6));
        assert_eq!(l.mips[1].row_pitch_bytes, 256);
        assert_eq!(l.mips[1].x_extent_blocks, 8);
        assert_eq!(l.mips[1].array_slice_data_extent_bytes, 256 * 7 + 64);
    }

    #[test]
    fn layout_small_packed_base() {
        // 8x8 linear DXT1 with packed mips: the base itself is in a tail (level 0 packed).
        let mut p = params(k_DXT1, 8, 8, false, true, 3);
        p.base_pitch_texels_div_32 = 4; // 128 texels, 256-byte rows for DXT1.
        let l = guest_layout(&p);
        assert_eq!(l.packed_level, 0);
        // Base tail: 8x8 at x=4 blocks/... 16 >> 0 = 16 texels -> 4 blocks; 2 blocks wide.
        assert_eq!((l.base.x_extent_blocks, l.base.y_extent_blocks), (6, 2));
        assert_eq!(l.base.array_slice_data_extent_bytes, 256 + 6 * 8);
        // Mips tail is stored in mips[0] and covers levels 0..=3.
        assert_eq!(l.mip_offsets_bytes[0], 0);
        assert!(l.mips[0].row_pitch_bytes > 0);
        let x_needed = (0..=3).map(|lv| {
            let (x, _, _) = packed_mip_offset(8, 8, 1, k_DXT1, lv).unwrap();
            x + (8u32 >> lv).max(1).div_ceil(4)
        });
        assert_eq!(l.mips[0].x_extent_blocks, x_needed.max().unwrap());
    }

    #[test]
    fn layout_stacked_and_cube() {
        let mut p = params(k_8_8_8_8, 64, 64, true, false, 6);
        p.depth_or_array_size = 3;
        let l = guest_layout(&p);
        assert_eq!(l.array_size, 3);
        assert_eq!(l.base.array_slice_stride_bytes, 16384);
        assert_eq!(l.base.level_data_extent_bytes, 16384 * 3);
        // Mip offsets advance by the slice stride times the array size aligned to 4.
        assert_eq!(l.mip_offsets_bytes[2], l.mips[1].array_slice_stride_bytes * 4);
        p.dimension = DataDimension::Cube;
        p.depth_or_array_size = 6;
        let l = guest_layout(&p);
        assert_eq!(l.array_size, 6);
        assert_eq!(l.mip_offsets_bytes[2], l.mips[1].array_slice_stride_bytes * 6);
    }

    #[test]
    fn layout_3d() {
        let p = LayoutParams {
            dimension: DataDimension::D3,
            depth_or_array_size: 8,
            ..params(k_8_8_8_8, 32, 32, true, false, 0)
        };
        let l = guest_layout(&p);
        assert_eq!(l.array_size, 1);
        assert_eq!(l.base.z_slice_stride_block_rows, 32);
        assert_eq!(l.base.array_slice_stride_bytes, 32 * 32 * 4 * 8);
        assert_eq!(l.base.array_slice_data_extent_bytes, 32 * 32 * 4 * 8);
        // Mips of a 3D texture use power-of-two depth: 32x32x5 level 1 is 16x16x4 -> padded 32x32x4.
        let p = LayoutParams { depth_or_array_size: 5, max_level: 5, ..p };
        let l = guest_layout(&p);
        assert_eq!(l.base.array_slice_stride_bytes, 32 * 32 * 4 * 8);
        assert_eq!(l.mips[1].array_slice_stride_bytes, 32 * 32 * 4 * 4);
    }

    #[test]
    fn subresources_rules() {
        let mut f = TextureFetchConstant::new_2d(k_DXT1, 512, 256);
        f.base_page = 0x100;
        f.mip_page = 0x200;
        f.mip_max_level = 15;
        f.packed_mips = true;
        let s = TextureSubresources::from_fetch(&f);
        assert_eq!((s.mip_min_level, s.mip_max_level), (0, 9));
        // Missing mip address disables mips.
        f.mip_page = 0;
        let s = TextureSubresources::from_fetch(&f);
        assert_eq!((s.mip_page, s.mip_max_level), (0, 0));
        // Packed level 0 with mip address 0 uses the base address for mips.
        let mut g = TextureFetchConstant::new_2d(k_DXT1, 16, 16);
        g.base_page = 0x100;
        g.packed_mips = true;
        g.mip_max_level = 4;
        let s = TextureSubresources::from_fetch(&g);
        assert_eq!((s.base_page, s.mip_page, s.mip_max_level), (0x100, 0x100, 4));
        // mip_min_level > 0 and base == mip drops the base.
        g.mip_min_level = 1;
        g.mip_page = 0x100;
        let s = TextureSubresources::from_fetch(&g);
        assert_eq!((s.base_page, s.mip_min_level), (0, 1));
        // Wide 1D.
        let mut w = TextureFetchConstant::new_2d(k_8_8_8_8, 1, 1);
        w.dimension = DataDimension::D1;
        w.width = 786432;
        w.base_page = 1;
        let s = TextureSubresources::from_fetch(&w);
        assert!(s.wide_1d);
        assert_eq!((s.width, s.height, s.pitch_div_32), (8192, 96, 256));
    }

    #[test]
    fn memory_ranges_cover_levels() {
        let mut f = TextureFetchConstant::new_2d(k_8_8_8_8, 256, 256);
        f.tiled = true;
        f.packed_mips = true;
        f.base_page = 0x1000;
        f.mip_page = 0x1040;
        f.mip_max_level = 8;
        let info = TextureGuestInfo::from_fetch(&f);
        let r = info.memory_ranges();
        assert_eq!(r[0], GuestRange { kind: RangeKind::Base, address: 0x100_0000, length: 262144 });
        assert_eq!(r.len(), 5);
        assert_eq!(r[4].kind, RangeKind::Mip(4));
        assert_eq!(r[4].address, 0x104_0000 + 86016);
        assert_eq!(info.total_sizes(), (262144, 86016 + 4096));
        let (addr, _, kind) = info.level_storage(7).unwrap();
        assert_eq!((addr, kind), (0x104_0000 + 86016, RangeKind::Mip(4)));
        assert!(info.level_storage(9).is_none());
    }
}
