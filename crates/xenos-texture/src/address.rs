//! Tiled texture addressing (Xenia `texture_address.h` / `texture_util` bounds).
//!
//! All coordinates are in blocks. A tiled byte address is composed of:
//! `[:12]` outer_inner[:8], `[11]` bank, `[10:8]` outer_inner[7:5], `[7:6]` pipe,
//! `[5]` outer_inner[4], `[4]` y & 1, `[3:0]` outer_inner[3:0].

/// Strides of subresources are computed with dimensions padded to 32x32(x4) blocks.
pub const STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS: u32 = 32;
pub const STORAGE_DEPTH_ALIGNMENT_BLOCKS: u32 = 4;
pub const TILE_WIDTH_HEIGHT: u32 = 32;
pub const TILE_DEPTH: u32 = 4;
/// Subresources (levels, array slices, the texture itself) are aligned to 4 KB.
pub const SUBRESOURCE_ALIGNMENT_BYTES: u32 = 4096;
/// Rows of linear textures are aligned to 256 bytes.
pub const LINEAR_ROW_ALIGNMENT_BYTES: u32 = 256;

const MACRO_TILE_WIDTH_LOG2: u32 = 5;
const MACRO_TILE_HEIGHT_2D_LOG2: u32 = 5;
const MACRO_TILE_HEIGHT_3D_LOG2: u32 = 4;
const MACRO_TILE_DEPTH_LOG2: u32 = 2;

fn tiled_combine(outer_inner_bytes: i64, bank: u32, pipe: u32, y_lsb: u32) -> i64 {
    i64::from((y_lsb << 4) | (pipe << 6) | (bank << 11))
        | (outer_inner_bytes & 0b1111)
        | (((outer_inner_bytes >> 4) & 0b1) << 5)
        | (((outer_inner_bytes >> 5) & 0b111) << 8)
        | ((outer_inner_bytes >> 8) << 12)
}

/// Byte offset of block (x, y) in a 2D tiled surface. `pitch_aligned` is in blocks and must be
/// a multiple of 32. Works for negative coordinates too (used for offset resolves).
pub fn tiled_2d(x: i32, y: i32, pitch_aligned: u32, bytes_per_block_log2: u32) -> i32 {
    debug_assert_eq!(pitch_aligned % STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS, 0);
    let outer_blocks = ((y >> MACRO_TILE_HEIGHT_2D_LOG2) * (pitch_aligned >> MACRO_TILE_WIDTH_LOG2) as i32
        + (x >> MACRO_TILE_WIDTH_LOG2))
        << 6;
    let inner_blocks = (((y >> 1) & 0b111) << 3) | (x & 0b111);
    let outer_inner_bytes = (outer_blocks | inner_blocks) << bytes_per_block_log2;
    let bank = ((y >> 4) & 1) as u32;
    let pipe = (((x >> 3) & 0b11) ^ (((y >> 3) & 1) << 1)) as u32;
    tiled_combine(i64::from(outer_inner_bytes), bank, pipe, (y & 1) as u32) as i32
}

/// Byte offset of block (x, y, z) in a 3D tiled surface. `pitch_aligned` and `height_aligned`
/// are in blocks and must be multiples of 32.
pub fn tiled_3d(x: i32, y: i32, z: i32, pitch_aligned: u32, height_aligned: u32, bytes_per_block_log2: u32) -> i64 {
    debug_assert_eq!(pitch_aligned % STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS, 0);
    debug_assert_eq!(height_aligned % STORAGE_PITCH_HEIGHT_ALIGNMENT_BLOCKS, 0);
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

/// Lowest address touched by blocks at or after (left, top) within the containing tiles.
pub fn tiled_address_lower_bound_2d(left: u32, top: u32, pitch_aligned: u32, bytes_per_block_log2: u32) -> u32 {
    tiled_2d(
        (left & !(TILE_WIDTH_HEIGHT - 1)) as i32,
        (top & !(TILE_WIDTH_HEIGHT - 1)) as i32,
        pitch_aligned,
        bytes_per_block_log2,
    ) as u32
}

pub fn tiled_address_lower_bound_3d(
    left: u32,
    top: u32,
    front: u32,
    pitch_aligned: u32,
    height_aligned: u32,
    bytes_per_block_log2: u32,
) -> u64 {
    let front_aligned = front & !(TILE_DEPTH - 1);
    let mut top_aligned = top & !(TILE_WIDTH_HEIGHT - 1);
    if front_aligned & TILE_DEPTH != 0 {
        top_aligned += 8;
    }
    tiled_3d(
        (left & !(TILE_WIDTH_HEIGHT - 1)) as i32,
        top_aligned as i32,
        front_aligned as i32,
        pitch_aligned,
        height_aligned,
        bytes_per_block_log2,
    ) as u64
}

/// Exclusive upper bound of addresses used by blocks [0, right) x [0, bottom).
pub fn tiled_address_upper_bound_2d(right: u32, bottom: u32, pitch_aligned: u32, bytes_per_block_log2: u32) -> u32 {
    if right == 0 || bottom == 0 {
        return 0;
    }
    let origin = tiled_2d(
        ((right - 1) & !(TILE_WIDTH_HEIGHT - 1)) as i32,
        ((bottom - 1) & !(TILE_WIDTH_HEIGHT - 1)) as i32,
        pitch_aligned,
        bytes_per_block_log2,
    ) as u32;
    origin
        + match bytes_per_block_log2 {
            0 => 0xA00,
            1 => 0xC00,
            n => 0x400 << n,
        }
}

/// Exclusive upper bound of addresses used by blocks [0, right) x [0, bottom) x [0, back).
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
    // Bank/pipe selection can place earlier runs of 8 blocks higher, so check each run's last
    // block in the last 32-row x 4-slice portion (all rows if the width exceeds the pitch).
    let y_first = if right > pitch_aligned { 0 } else { (bottom - 1) & !(TILE_WIDTH_HEIGHT - 1) };
    let z_first = (back - 1) & !(TILE_DEPTH - 1);
    let mut upper: i64 = 0;
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

    /// Classic Xbox 360 2D tiling formula (crunch `crn_decomp.h` / old Xenia), written
    /// independently of the bit-decomposition form above.
    fn reference_tiled_2d(x: u32, y: u32, pitch: u32, log2_bpp: u32) -> u32 {
        let pitch = pitch.div_ceil(32) * 32;
        let macro_ = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (log2_bpp + 7);
        let micro = ((x & 7) + ((y & 0xE) << 2)) << log2_bpp;
        let offset = macro_ + ((micro & !0xF) << 1) + (micro & 0xF) + ((y & 1) << 4);
        ((offset & !0x1FF) << 3)
            + ((y & 16) << 7)
            + ((offset & 0x1C0) << 2)
            + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6)
            + (offset & 0x3F)
    }

    /// 3D tiling assembled bit by bit from the documented address layout.
    fn reference_tiled_3d(x: u32, y: u32, z: u32, pitch: u32, height: u32, log2_bpp: u32) -> u64 {
        let tiles_x = (pitch / 32) as u64;
        let tiles_y16 = (height / 16) as u64;
        let outer = ((z as u64 / 4) * tiles_y16 + (y as u64 / 16)) * tiles_x + (x as u64 / 32);
        let inner = ((z as u64 % 4) << 5) | ((((y >> 1) & 3) as u64) << 3) | (x as u64 & 7);
        let oi = ((outer << 7) | inner) << log2_bpp;
        let bank = ((y >> 3) ^ (z >> 2)) as u64 & 1;
        let pipe = (((x >> 3) & 3) as u64) ^ (bank << 1);
        let mut a = 0u64;
        a |= oi & 0xF;
        a |= ((y & 1) as u64) << 4;
        a |= ((oi >> 4) & 1) << 5;
        a |= pipe << 6;
        a |= ((oi >> 5) & 7) << 8;
        a |= bank << 11;
        a |= (oi >> 8) << 12;
        a
    }

    #[test]
    fn tiled_2d_matches_reference() {
        for log2 in 0..=4 {
            for pitch in [32u32, 64, 96, 160, 256] {
                for y in 0..80 {
                    for x in 0..pitch {
                        assert_eq!(
                            tiled_2d(x as i32, y as i32, pitch, log2) as u32,
                            reference_tiled_2d(x, y, pitch, log2),
                            "x={x} y={y} pitch={pitch} log2={log2}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_3d_matches_reference() {
        for log2 in 0..=4 {
            for (pitch, height) in [(32u32, 32u32), (64, 32), (96, 64)] {
                for z in 0..12 {
                    for y in 0..height {
                        for x in 0..pitch {
                            assert_eq!(
                                tiled_3d(x as i32, y as i32, z as i32, pitch, height, log2) as u64,
                                reference_tiled_3d(x, y, z, pitch, height, log2),
                                "x={x} y={y} z={z} log2={log2}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_2d_known_values() {
        // 32bpp: block (8, 0) -> pipe 1 -> 0x40; (0, 1) -> y lsb -> 0x10; (1, 0) -> 4.
        assert_eq!(tiled_2d(1, 0, 32, 2), 4);
        assert_eq!(tiled_2d(0, 1, 32, 2), 0x10);
        assert_eq!(tiled_2d(8, 0, 32, 2), 0x40);
        // (0, 2): inner = 8 blocks = 32 bytes -> outer_inner bit 5 -> address bit 8.
        assert_eq!(tiled_2d(0, 2, 32, 2), 0x100);
        // (0, 8): inner y[3:1] = 4 -> 32 blocks -> bit 10, plus pipe[1] = y[3] -> 0x80.
        assert_eq!(tiled_2d(0, 8, 32, 2), 0x480);
        // (0, 16): bank -> 0x800.
        assert_eq!(tiled_2d(0, 16, 32, 2), 0x800);
        // Second macro tile in the row at 32bpp starts at 64 blocks * 4 B << 4 ... = 0x1000.
        assert_eq!(tiled_2d(32, 0, 64, 2), 0x1000);
        // Next tile row with a 64-block pitch: outer = 2 * 64 blocks = 512 bytes * 4 = 0x2000.
        assert_eq!(tiled_2d(0, 32, 64, 2), 0x2000);
        // Negative offsets are symmetric for tile-aligned coordinates.
        assert_eq!(tiled_2d(-32, -32, 64, 2), -0x2000 - 0x1000);
    }

    #[test]
    fn tiled_is_bijective_within_tile_periods() {
        for log2 in 0..=4u32 {
            let (pitch, height) = (128u32, 128u32);
            let mut seen = std::collections::HashSet::new();
            for y in 0..height {
                for x in 0..pitch {
                    let a = tiled_2d(x as i32, y as i32, pitch, log2);
                    assert_eq!(a as u32 & ((1 << log2) - 1), 0);
                    assert!(seen.insert(a), "collision log2={log2} x={x} y={y}");
                }
            }
            let mut seen = std::collections::HashSet::new();
            for z in 0..8 {
                for y in 0..32 {
                    for x in 0..64 {
                        assert!(seen.insert(tiled_3d(x, y, z, 64, 32, log2)), "3d collision");
                    }
                }
            }
        }
    }

    #[test]
    fn upper_bounds_cover_all_blocks() {
        for log2 in 0..=4u32 {
            for (w, h, pitch) in [(1u32, 1u32, 32u32), (20, 7, 32), (64, 64, 64), (100, 40, 128), (32, 32, 32)] {
                let ub = tiled_address_upper_bound_2d(w, h, pitch, log2);
                let lb = tiled_address_lower_bound_2d(0, 0, pitch, log2);
                for y in 0..h {
                    for x in 0..w {
                        let a = tiled_2d(x as i32, y as i32, pitch, log2) as u32;
                        assert!(a >= lb && a + (1 << log2) <= ub, "2d log2={log2} {w}x{h} ({x},{y})");
                    }
                }
            }
            for (w, h, d, pitch, height) in [(5u32, 5u32, 3u32, 32u32, 32u32), (40, 33, 9, 64, 64)] {
                let ub = tiled_address_upper_bound_3d(w, h, d, pitch, height, log2);
                for z in 0..d {
                    for y in 0..h {
                        for x in 0..w {
                            let a = tiled_3d(x as i32, y as i32, z as i32, pitch, height, log2) as u64;
                            assert!(a + (1 << log2) <= ub, "3d");
                        }
                    }
                }
            }
        }
        assert_eq!(tiled_address_upper_bound_2d(0, 5, 32, 2), 0);
        assert_eq!(tiled_address_upper_bound_2d(32, 32, 32, 2), 0x1000);
        assert_eq!(tiled_address_upper_bound_2d(1, 1, 32, 0), 0xA00);
    }

    #[test]
    fn tiled_3d_known_values() {
        // Slice 1: inner z = 1 -> 32 blocks * 4 B = 128 -> address bit 10.
        assert_eq!(tiled_3d(0, 0, 1, 32, 32, 2), 0x400);
        // Slice 4: next 4-slice group (2 macro tile rows of 16 down), bank and pipe[1] flip.
        assert_eq!(tiled_3d(0, 0, 4, 32, 32, 2), 0x4000 | 0x800 | 0x80);
        // Row 16 is the second 32x16 macro tile row.
        assert_eq!(tiled_3d(0, 16, 0, 32, 32, 2), 0x2000);
        assert_eq!(tiled_3d(0, 8, 0, 32, 32, 2), 0x800 | 0x80);
    }
}
