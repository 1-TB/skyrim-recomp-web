//! Xenos tiled texture addressing (Xenia texture_address.h and the tiled extent bounds from
//! texture_util), used to compute resolve destination addresses and the memory they touch.

/// Pitch and height of tiled storage are aligned to 32 blocks.
pub const STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS_LOG2: u32 = 5;
pub const STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS: u32 = 1 << STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS_LOG2;
/// 32x32-block tiles.
pub const TEXTURE_TILE_WIDTH_HEIGHT_LOG2: u32 = 5;
pub const TEXTURE_TILE_WIDTH_HEIGHT: u32 = 1 << TEXTURE_TILE_WIDTH_HEIGHT_LOG2;
/// 3D tiles are 4 slices deep.
pub const TEXTURE_TILE_DEPTH: u32 = 4;
/// Subresources are aligned to 4 KiB.
pub const TEXTURE_SUBRESOURCE_ALIGNMENT_BYTES_LOG2: u32 = 12;
pub const TEXTURE_SUBRESOURCE_ALIGNMENT_BYTES: u32 = 1 << TEXTURE_SUBRESOURCE_ALIGNMENT_BYTES_LOG2;

const MACRO_TILE_WIDTH_LOG2: u32 = 5;
const MACRO_TILE_HEIGHT_2D_LOG2: u32 = 5;
const MACRO_TILE_HEIGHT_3D_LOG2: u32 = 4;
const MACRO_TILE_DEPTH_LOG2: u32 = 2;

/// log2 of the X granularity at which tiled bases can be offset (GetTextureTiledXBaseGranularityLog2).
pub const fn tiled_x_base_granularity_log2(is_3d: bool, bytes_per_block_log2: u32) -> u32 {
    let v = bytes_per_block_log2 + is_3d as u32;
    7 - if v < 2 { v } else { 2 }
}

/// log2 of the Y granularity at which tiled bases can be offset (GetTextureTiledYBaseGranularityLog2).
pub const fn tiled_y_base_granularity_log2(is_3d: bool, bytes_per_block_log2: u32) -> u32 {
    if is_3d { 5 } else { 7 - if bytes_per_block_log2 < 2 { bytes_per_block_log2 } else { 2 } }
}

fn tiled_combine(outer_inner_bytes: i64, bank: u32, pipe: u32, y_lsb: u32) -> i64 {
    i64::from((y_lsb << 4) | (pipe << 6) | (bank << 11))
        | (outer_inner_bytes & 0b1111)
        | (((outer_inner_bytes >> 4) & 0b1) << 5)
        | (((outer_inner_bytes >> 5) & 0b111) << 8)
        | ((outer_inner_bytes >> 8) << 12)
}

/// Byte offset of block (x, y) in a 2D tiled surface with a 32-aligned pitch in blocks.
pub fn tiled_2d(x: i32, y: i32, pitch_aligned: u32, bytes_per_block_log2: u32) -> i32 {
    let outer_blocks = ((y >> MACRO_TILE_HEIGHT_2D_LOG2) * (pitch_aligned >> MACRO_TILE_WIDTH_LOG2) as i32
        + (x >> MACRO_TILE_WIDTH_LOG2))
        << 6;
    let inner_blocks = (((y >> 1) & 0b111) << 3) | (x & 0b111);
    let outer_inner_bytes = (outer_blocks | inner_blocks) << bytes_per_block_log2;
    let bank = ((y >> 4) & 1) as u32;
    let pipe = (((x >> 3) & 0b11) ^ (((y >> 3) & 1) << 1)) as u32;
    tiled_combine(i64::from(outer_inner_bytes), bank, pipe, (y & 1) as u32) as i32
}

/// Byte offset of block (x, y, z) in a 3D tiled surface with 32-aligned pitch and height in blocks.
pub fn tiled_3d(x: i32, y: i32, z: i32, pitch_aligned: u32, height_aligned: u32, bytes_per_block_log2: u32) -> i64 {
    let outer_blocks = ((((z >> MACRO_TILE_DEPTH_LOG2) * (height_aligned >> MACRO_TILE_HEIGHT_3D_LOG2) as i32
        + (y >> MACRO_TILE_HEIGHT_3D_LOG2))
        * (pitch_aligned >> MACRO_TILE_WIDTH_LOG2) as i32)
        + (x >> MACRO_TILE_WIDTH_LOG2))
        << 7;
    let inner_blocks = ((z & 0b11) << 5) | (((y >> 1) & 0b11) << 3) | (x & 0b111);
    let outer_inner_bytes = i64::from(outer_blocks | inner_blocks) << bytes_per_block_log2;
    let bank = (((y >> 3) ^ (z >> 2)) & 1) as u32;
    let pipe = (((x >> 3) & 0b11) as u32) ^ (bank << 1);
    tiled_combine(outer_inner_bytes, bank, pipe, (y & 1) as u32)
}

/// Lowest byte offset a 2D tiled region starting at (left, top) can touch.
pub fn tiled_address_lower_bound_2d(left: u32, top: u32, pitch_aligned: u32, bytes_per_block_log2: u32) -> u32 {
    let m = !(TEXTURE_TILE_WIDTH_HEIGHT - 1);
    tiled_2d((left & m) as i32, (top & m) as i32, pitch_aligned, bytes_per_block_log2) as u32
}

/// Lowest byte offset a 3D tiled region starting at (left, top, front) can touch.
pub fn tiled_address_lower_bound_3d(
    left: u32,
    top: u32,
    front: u32,
    pitch_aligned: u32,
    height_aligned: u32,
    bytes_per_block_log2: u32,
) -> u64 {
    let front_aligned = front & !(TEXTURE_TILE_DEPTH - 1);
    let mut top_aligned = top & !(TEXTURE_TILE_WIDTH_HEIGHT - 1);
    if front_aligned & TEXTURE_TILE_DEPTH != 0 {
        top_aligned += 8;
    }
    tiled_3d(
        (left & !(TEXTURE_TILE_WIDTH_HEIGHT - 1)) as i32,
        top_aligned as i32,
        front_aligned as i32,
        pitch_aligned,
        height_aligned,
        bytes_per_block_log2,
    ) as u64
}

/// Byte offset past the last byte a 2D tiled region ending (exclusive) at (right, bottom) can touch.
pub fn tiled_address_upper_bound_2d(right: u32, bottom: u32, pitch_aligned: u32, bytes_per_block_log2: u32) -> u32 {
    if right == 0 || bottom == 0 {
        return 0;
    }
    let m = !(TEXTURE_TILE_WIDTH_HEIGHT - 1);
    let mut upper =
        tiled_2d(((right - 1) & m) as i32, ((bottom - 1) & m) as i32, pitch_aligned, bytes_per_block_log2) as u32;
    upper += match bytes_per_block_log2 {
        0 => 0xA00,
        1 => 0xC00,
        n => 0x400 << n,
    };
    upper
}

/// Byte offset past the last byte a 3D tiled region ending (exclusive) at (right, bottom, back) can
/// touch.
pub fn tiled_address_upper_bound_3d(
    right: u32,
    bottom: u32,
    back: u32,
    pitch_aligned: u32,
    height_aligned: u32,
    bytes_per_block_log2: u32,
) -> u64 {
    if right == 0 || bottom == 0 || back == 0 {
        return 0;
    }
    let y_first = if right > pitch_aligned { 0 } else { (bottom - 1) & !(TEXTURE_TILE_WIDTH_HEIGHT - 1) };
    let z_first = (back - 1) & !(TEXTURE_TILE_DEPTH - 1);
    let mut upper = 0i64;
    for z in z_first..back {
        for y in y_first..bottom {
            let mut x = 7u32;
            loop {
                let x_last = x.min(right - 1);
                upper = upper.max(tiled_3d(
                    x_last as i32,
                    y as i32,
                    z as i32,
                    pitch_aligned,
                    height_aligned,
                    bytes_per_block_log2,
                ));
                if x_last == right - 1 {
                    break;
                }
                x += 8;
            }
        }
    }
    upper as u64 + (1u64 << bytes_per_block_log2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_2d_basics() {
        assert_eq!(tiled_2d(0, 0, 32, 2), 0);
        // Within the first 8 blocks of a row, 4 bytes per block.
        assert_eq!(tiled_2d(1, 0, 32, 2), 4);
        assert_eq!(tiled_2d(3, 0, 32, 2), 12);
        // Odd rows set the y LSB bit (16 bytes).
        assert_eq!(tiled_2d(0, 1, 32, 2), 16);
        // x = 8 moves to the next pipe (bit 6).
        assert_eq!(tiled_2d(8, 0, 32, 2), 64);
        // Next 32x32 macro tile along X: outer block 64 * 4 bytes = 256 -> bit 12.
        assert_eq!(tiled_2d(32, 0, 64, 2), 1 << 12);
        // All addresses of a 32x32 4bpp-block tile are distinct and stay inside 4 KiB.
        let mut seen = std::collections::HashSet::new();
        for y in 0..32 {
            for x in 0..32 {
                let a = tiled_2d(x, y, 32, 2);
                assert!((0..4096).contains(&a));
                assert_eq!(a % 4, 0);
                assert!(seen.insert(a));
            }
        }
    }

    #[test]
    fn tiled_3d_distinct() {
        let mut seen = std::collections::HashSet::new();
        for z in 0..4 {
            for y in 0..16 {
                for x in 0..32 {
                    assert!(seen.insert(tiled_3d(x, y, z, 32, 32, 2)));
                }
            }
        }
        assert_eq!(seen.len(), 32 * 16 * 4);
    }

    #[test]
    fn bounds_cover_region() {
        for bpb in 0..5 {
            let (pitch, l, t, r, b) = (128u32, 40u32, 8u32, 100u32, 70u32);
            let lo = tiled_address_lower_bound_2d(l, t, pitch, bpb);
            let hi = tiled_address_upper_bound_2d(r, b, pitch, bpb);
            for y in t..b {
                for x in l..r {
                    let a = tiled_2d(x as i32, y as i32, pitch, bpb) as u32;
                    assert!(a >= lo && a + (1 << bpb) <= hi, "bpb {bpb} ({x},{y}) {a:#x} not in {lo:#x}..{hi:#x}");
                }
            }
            let lo = tiled_address_lower_bound_3d(l, t, 1, pitch, 96, bpb);
            let hi = tiled_address_upper_bound_3d(r, b, 6, pitch, 96, bpb);
            for z in 1..6 {
                for y in t..b {
                    for x in l..r {
                        let a = tiled_3d(x as i32, y as i32, z, pitch, 96, bpb) as u64;
                        assert!(a >= lo && a + (1 << bpb) <= hi);
                    }
                }
            }
        }
        assert_eq!(tiled_address_upper_bound_2d(0, 5, 32, 2), 0);
        assert_eq!(tiled_x_base_granularity_log2(false, 2), 5);
        assert_eq!(tiled_y_base_granularity_log2(false, 0), 7);
        assert_eq!(tiled_y_base_granularity_log2(true, 0), 5);
    }
}
