//! End-to-end loading tests: synthetic guest textures are tiled with independently written
//! reference formulas, then loaded back through the public API.

use xenos_texture::layout::{LayoutParams, guest_layout, packed_mip_offset};
use xenos_texture::*;

/// Classic Xbox 360 2D tiler (crunch / old Xenia formulation).
fn ref_tiled_2d(x: u32, y: u32, pitch: u32, log2_bpp: u32) -> u32 {
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

/// Old Xenia 3D tiler formulation.
fn ref_tiled_3d(x: u32, y: u32, z: u32, pitch: u32, height: u32, log2_bpp: u32) -> u32 {
    let pitch = pitch.div_ceil(32) * 32;
    let height = height.div_ceil(32) * 32;
    let macro_outer = ((y >> 4) + (z >> 2) * (height >> 4)) * (pitch >> 5);
    let macro_ = (((x >> 5) + macro_outer) << (log2_bpp + 6)) << 1;
    let micro = (((x & 7) + ((y & 6) << 2)) << (log2_bpp + 6)) >> 6;
    let offset_outer = ((y >> 3) + (z >> 2)) & 1;
    let offset1 = offset_outer + ((((x >> 3) + (offset_outer << 1)) & 3) << 1);
    let offset2 = ((macro_ + (micro & !15)) << 1) + (micro & 15) + ((z & 3) << (log2_bpp + 6)) + ((y & 1) << 4);
    let mut address = (offset1 & 1) << 3;
    address += (offset2 >> 6) & 7;
    address <<= 3;
    address += offset1 & !1;
    address <<= 2;
    address += offset2 & !511;
    address <<= 3;
    address += offset2 & 63;
    address
}

struct Mem(Vec<u8>);

impl Mem {
    fn new() -> Self {
        Mem(vec![0; 32 << 20])
    }
    fn put(&mut self, addr: u32, bytes: &[u8]) {
        self.0[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
    }
    /// Converts the "after swap" view into guest byte order (all swaps are involutions).
    fn finish(mut self, endian: Endian) -> Vec<u8> {
        endian.swap_in_place(&mut self.0);
        self.0
    }
}

fn texel_rgba(x: u32, y: u32, z: u32) -> [u8; 4] {
    [x as u8, y as u8, (x ^ y ^ (z * 37)) as u8, 0xA0 | z as u8]
}

#[test]
fn ref_3d_tiler_agrees_with_crate() {
    for log2 in 0..=4 {
        for z in 0..9 {
            for y in 0..40 {
                for x in 0..70 {
                    assert_eq!(
                        ref_tiled_3d(x, y, z, 96, 64, log2) as i64,
                        address::tiled_3d(x as i32, y as i32, z as i32, 96, 64, log2),
                        "({x},{y},{z}) log2={log2}"
                    );
                }
            }
        }
    }
}

#[test]
fn rgba8_tiled_2d_round_trip() {
    let (w, h) = (100u32, 60u32);
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, w, h);
    f.tiled = true;
    f.pitch_div_32 = 4;
    f.base_page = 0x123;
    f.endianness = Endian::K8in32;
    let mut mem = Mem::new();
    for y in 0..h {
        for x in 0..w {
            mem.put((0x123 << 12) + ref_tiled_2d(x, y, 128, 2), &texel_rgba(x, y, 0));
        }
    }
    let mem = mem.finish(Endian::K8in32);
    let t = load_texture(&f, &mem, &LoadOptions::default()).unwrap();
    assert_eq!(t.mapping.host_format, HostFormat::Rgba8Unorm);
    assert_eq!(t.dimension, HostDimension::D2);
    assert_eq!(t.levels.len(), 1);
    let l = &t.levels[0];
    assert_eq!((l.width, l.height, l.bytes_per_row, l.rows_per_image), (100, 60, 400, 60));
    for y in 0..h {
        for x in 0..w {
            let i = (y * 400 + x * 4) as usize;
            assert_eq!(&l.data[i..i + 4], &texel_rgba(x, y, 0), "({x},{y})");
        }
    }
    assert_eq!(t.guest_ranges[0].address, 0x123 << 12);
    // Row alignment for buffer copies.
    let t = load_texture(&f, &mem, &LoadOptions { row_alignment: 256, ..Default::default() }).unwrap();
    assert_eq!(t.levels[0].bytes_per_row, 512);
    assert_eq!(&t.levels[0].data[512 * 7 + 40..512 * 7 + 44], &texel_rgba(10, 7, 0));
}

fn rgb565(r: u32, g: u32, b: u32) -> u16 {
    (((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)) as u16
}

fn dxt1_block(level: u32, bx: u32, by: u32) -> [u8; 8] {
    let c0 = rgb565(level * 30 + 8, bx * 16, by * 16) | 0x0800;
    let mut b = [0u8; 8];
    b[..2].copy_from_slice(&c0.to_le_bytes());
    b
}

fn expand565(c: u16) -> [u8; 4] {
    let r = u32::from(c >> 11) & 31;
    let g = u32::from(c >> 5) & 63;
    let b = u32::from(c) & 31;
    [((r << 3) | (r >> 2)) as u8, ((g << 2) | (g >> 4)) as u8, ((b << 3) | (b >> 2)) as u8, 255]
}

/// Writes a DXT1 2D texture with a full (packed) mip chain using the reference tiler.
fn build_dxt1_chain(w: u32, h: u32, tiled: bool, base_page: u32, mip_page: u32) -> (TextureFetchConstant, Vec<u8>) {
    let max_level = 31 - w.max(h).leading_zeros();
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_DXT1, w, h);
    f.tiled = tiled;
    // Linear D3D pitch keeps block rows 256-byte aligned: 32 blocks of 8 bytes = 128 texels.
    f.pitch_div_32 = if tiled { w.div_ceil(32) } else { w.div_ceil(128) * 4 };
    f.base_page = base_page;
    f.mip_page = mip_page;
    f.mip_max_level = max_level;
    f.packed_mips = true;
    f.endianness = Endian::K8in16;
    let layout = guest_layout(&LayoutParams {
        dimension: DataDimension::D2OrStacked,
        base_pitch_texels_div_32: f.pitch_div_32,
        width: w,
        height: h,
        depth_or_array_size: 1,
        tiled,
        format: TextureFormat::k_DXT1,
        packed_mips: true,
        has_base: true,
        max_level,
    });
    let mut mem = Mem::new();
    for level in 0..=max_level {
        let (addr, l) = if level == 0 {
            (base_page << 12, &layout.base)
        } else {
            let idx = level.min(layout.packed_level) as usize;
            ((mip_page << 12) + layout.mip_offsets_bytes[idx], &layout.mips[idx])
        };
        let (ox, oy, _) = packed_mip_offset(w, h, 1, TextureFormat::k_DXT1, level).unwrap_or((0, 0, 0));
        let pitch = l.row_pitch_bytes / 8;
        let (bw, bh) = ((w >> level).max(1).div_ceil(4), (h >> level).max(1).div_ceil(4));
        for by in 0..bh {
            for bx in 0..bw {
                let (x, y) = (bx + ox, by + oy);
                let off = if tiled { ref_tiled_2d(x, y, pitch, 3) } else { y * l.row_pitch_bytes + x * 8 };
                mem.put(addr + off, &dxt1_block(level, bx, by));
            }
        }
    }
    (f, mem.finish(Endian::K8in16))
}

#[test]
fn dxt1_packed_chain_bc_passthrough() {
    for tiled in [true, false] {
        let (f, mem) = build_dxt1_chain(64, 64, tiled, 0x200, 0x300);
        let opts = LoadOptions { caps: HostCaps { texture_compression_bc: true, norm16: false }, ..Default::default() };
        let t = load_texture(&f, &mem, &opts).unwrap();
        assert_eq!(t.mapping.host_format, HostFormat::Bc1RgbaUnorm);
        assert_eq!(t.mip_level_count, 7);
        assert_eq!(t.levels.len(), 7);
        for l in &t.levels {
            let bw = (64u32 >> l.level).max(1).div_ceil(4);
            assert_eq!((l.copy_width, l.copy_height), (bw * 4, bw * 4), "level {}", l.level);
            assert_eq!(l.bytes_per_row, bw * 8);
            for by in 0..bw {
                for bx in 0..bw {
                    let i = (by * l.bytes_per_row + bx * 8) as usize;
                    assert_eq!(
                        &l.data[i..i + 8],
                        &dxt1_block(l.level, bx, by),
                        "tiled={tiled} level {} ({bx},{by})",
                        l.level
                    );
                }
            }
        }
    }
}

#[test]
fn dxt1_packed_chain_decompressed() {
    let (f, mem) = build_dxt1_chain(64, 32, true, 0x200, 0x300);
    let t = load_texture(&f, &mem, &LoadOptions::default()).unwrap();
    assert_eq!(t.mapping.host_format, HostFormat::Rgba8Unorm);
    for l in &t.levels {
        assert_eq!((l.width, l.height), ((64 >> l.level).max(1), (32 >> l.level).max(1)));
        for y in 0..l.height {
            for x in 0..l.width {
                let c0 = u16::from_le_bytes(dxt1_block(l.level, x / 4, y / 4)[..2].try_into().unwrap());
                let i = (y * l.bytes_per_row + x * 4) as usize;
                assert_eq!(&l.data[i..i + 4], &expand565(c0), "level {} ({x},{y})", l.level);
            }
        }
    }
}

#[test]
fn small_packed_base_texture() {
    // 8x8 DXT1: the base level is itself in a packed tail (stored at x = 4 blocks).
    let (f, mem) = build_dxt1_chain(8, 8, false, 0x200, 0x300);
    let opts = LoadOptions { caps: HostCaps { texture_compression_bc: true, norm16: false }, ..Default::default() };
    let t = load_texture(&f, &mem, &opts).unwrap();
    assert_eq!(t.levels.len(), 4);
    assert_eq!(&t.levels[0].data[8..16], &dxt1_block(0, 1, 0));
    assert_eq!(&t.levels[3].data[..8], &dxt1_block(3, 0, 0));
}

#[test]
fn tiled_3d_round_trip() {
    let (w, h, d) = (40u32, 20u32, 6u32);
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8, w, h);
    f.dimension = DataDimension::D3;
    f.depth = d;
    f.tiled = true;
    f.pitch_div_32 = 2;
    f.base_page = 0x40;
    let mut mem = Mem::new();
    let val = |x: u32, y: u32, z: u32| (x * 7 + y * 13 + z * 61) as u8;
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                mem.put((0x40 << 12) + ref_tiled_3d(x, y, z, 64, 32, 0), &[val(x, y, z)]);
            }
        }
    }
    let mem = mem.finish(Endian::None);
    let t = load_texture(&f, &mem, &LoadOptions::default()).unwrap();
    assert_eq!(t.dimension, HostDimension::D3);
    let l = &t.levels[0];
    assert_eq!((l.depth_or_array_layers, l.rows_per_image, l.bytes_per_row), (6, 20, 40));
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                assert_eq!(l.data[(z * 800 + y * 40 + x) as usize], val(x, y, z), "({x},{y},{z})");
            }
        }
    }
}

#[test]
fn cube_with_mips() {
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, 64, 64);
    f.dimension = DataDimension::Cube;
    f.depth = 6;
    f.tiled = true;
    f.pitch_div_32 = 2;
    f.base_page = 0x100;
    f.mip_page = 0x200;
    f.mip_max_level = 1;
    let mut mem = Mem::new();
    for face in 0..6u32 {
        for y in 0..64 {
            for x in 0..64 {
                // Base face stride: 64x64x4 = 16 KB.
                mem.put((0x100 << 12) + face * 16384 + ref_tiled_2d(x, y, 64, 2), &texel_rgba(x, y, face));
            }
        }
        for y in 0..32 {
            for x in 0..32 {
                mem.put((0x200 << 12) + face * 4096 + ref_tiled_2d(x, y, 32, 2), &texel_rgba(x + 100, y, face));
            }
        }
    }
    let mem = mem.finish(Endian::None);
    let t = load_texture(&f, &mem, &LoadOptions::default()).unwrap();
    assert_eq!(t.dimension, HostDimension::Cube);
    assert_eq!(t.depth_or_array_layers, 6);
    assert_eq!(t.levels.len(), 2);
    for face in 0..6u32 {
        let l0 = &t.levels[0];
        let i = (face * 64 * 256 + 5 * 256 + 9 * 4) as usize;
        assert_eq!(&l0.data[i..i + 4], &texel_rgba(9, 5, face));
        let l1 = &t.levels[1];
        let i = (face * 32 * 128 + 31 * 128 + 31 * 4) as usize;
        assert_eq!(&l1.data[i..i + 4], &texel_rgba(131, 31, face));
    }
}

#[test]
fn stacked_linear_dxt5_decompressed() {
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_DXT4_5, 8, 8);
    f.stacked = true;
    f.depth = 3;
    // Base pitches are aligned to 32 blocks: 128 texels = 32 blocks * 16 B = 512-byte rows.
    f.pitch_div_32 = 4;
    f.base_page = 0x10;
    let mut mem = Mem::new();
    // Slice stride: 512 B * 32 block rows = 16 KB.
    for layer in 0..3u32 {
        for by in 0..2u32 {
            for bx in 0..2u32 {
                let mut block = [0u8; 16];
                block[0] = (layer * 50 + bx * 10 + by) as u8;
                block[1] = block[0];
                block[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes());
                mem.put((0x10 << 12) + layer * 16384 + by * 512 + bx * 16, &block);
            }
        }
    }
    let t = load_texture(&f, &mem.finish(Endian::None), &LoadOptions::default()).unwrap();
    assert_eq!(t.dimension, HostDimension::D2Array);
    let l = &t.levels[0];
    assert_eq!(l.depth_or_array_layers, 3);
    for layer in 0..3u32 {
        for (x, y) in [(0u32, 0u32), (5, 2), (7, 7)] {
            let i = (layer * 8 * 32 + y * 32 + x * 4) as usize;
            assert_eq!(l.data[i + 3], (layer * 50 + (x / 4) * 10 + y / 4) as u8);
            assert_eq!(&l.data[i..i + 3], &[255, 255, 255]);
        }
    }
}

#[test]
fn wide_1d_maps_to_rows() {
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, 1, 1);
    f.dimension = DataDimension::D1;
    f.width = 10000;
    f.base_page = 0x1;
    let mut mem = Mem::new();
    for x in 0..10000u32 {
        mem.put(0x1000 + x * 4, &x.to_le_bytes());
    }
    let t = load_texture(&f, &mem.finish(Endian::None), &LoadOptions::default()).unwrap();
    assert_eq!(t.dimension, HostDimension::D2);
    let l = &t.levels[0];
    assert_eq!((l.width, l.height), (8192, 2));
    let i = (8192 * 4 + 100 * 4) as usize;
    assert_eq!(u32::from_le_bytes(l.data[i..i + 4].try_into().unwrap()), 8292);
}

#[test]
fn yuv_odd_width_and_signed_view() {
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_Y1_Cr_Y0_Cb_REP, 5, 1);
    f.base_page = 0x2;
    let mut mem = Mem::new();
    mem.put(0x2000, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let mem = mem.finish(Endian::None);
    let t = load_texture(&f, &mem, &LoadOptions::default()).unwrap();
    let l = &t.levels[0];
    assert_eq!(l.bytes_per_row, 20);
    assert_eq!(&l.data[..20], &[3, 2, 1, 1, 3, 4, 1, 1, 7, 6, 5, 5, 7, 8, 5, 5, 11, 10, 9, 9]);

    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, 2, 1);
    f.base_page = 0x2;
    f.sign = [TextureSign::Signed; 4];
    let t = load_texture(&f, &mem, &LoadOptions { signed: true, ..Default::default() }).unwrap();
    assert_eq!(t.mapping.host_format, HostFormat::Rgba8Snorm);
    assert_eq!(&t.levels[0].data, &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn frontbuffer_tiled_bgra() {
    let (w, h) = (1280u32, 720u32);
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, w, h);
    f.tiled = true;
    f.pitch_div_32 = 40;
    f.base_page = 0x1000;
    f.endianness = Endian::K8in32;
    f.swizzle = [SwizzleSource::Z, SwizzleSource::Y, SwizzleSource::X, SwizzleSource::One];
    let mut mem = Mem::new();
    for y in 0..h {
        for x in 0..w {
            // X = blue, Y = green, Z = red.
            let [r, g, b, _] = texel_rgba(x, y, 1);
            mem.put((0x1000 << 12) + ref_tiled_2d(x, y, 1280, 2), &[b, g, r, 0]);
        }
    }
    let img = frontbuffer_to_rgba8(&f, &mem.finish(Endian::K8in32)).unwrap();
    assert_eq!((img.width, img.height, img.data.len()), (1280, 720, 1280 * 720 * 4));
    for (x, y) in [(0u32, 0u32), (1279, 719), (640, 360), (33, 17)] {
        let i = ((y * w + x) * 4) as usize;
        let [r, g, b, _] = texel_rgba(x, y, 1);
        assert_eq!(&img.data[i..i + 4], &[r, g, b, 255]);
    }
}

#[test]
fn frontbuffer_2_10_10_10() {
    let mut f = TextureFetchConstant::new_2d(TextureFormat::k_2_10_10_10_AS_16_16_16_16, 32, 2);
    f.base_page = 0x3;
    f.endianness = Endian::K8in32;
    let v: u32 = 1023 | (512 << 10) | (3 << 30);
    let mut mem = Mem::new();
    mem.put(0x3000 + 4 * 33, &v.to_le_bytes()); // (1, 1) with a 32-texel pitch.
    let img = frontbuffer_to_rgba8(&f, &mem.finish(Endian::K8in32)).unwrap();
    assert_eq!(&img.data[4 * 33..4 * 34], &[255, 128, 0, 255]);
    assert_eq!(&img.data[..4], &[0, 0, 0, 0]);
}

#[test]
fn errors() {
    let f = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, 4, 4);
    let mem = vec![0u8; 16];
    assert_eq!(load_texture(&f, &mem, &LoadOptions::default()), Err(LoadError::NoData));
    let mut g = f.clone();
    g.base_page = 1;
    g.format = TextureFormat::k_16_MPEG;
    assert!(matches!(load_texture(&g, &mem, &LoadOptions::default()), Err(LoadError::Mapping(_))));
    g.fetch_type = FetchConstantType::Vertex;
    assert!(matches!(load_texture(&g, &mem, &LoadOptions::default()), Err(LoadError::NotATexture(_))));
    // Reads past the end of guest memory are zero-filled rather than panicking.
    let mut h = f.clone();
    h.base_page = 0xFFFFF;
    let t = load_texture(&h, &mem, &LoadOptions::default()).unwrap();
    assert!(t.levels[0].data.iter().all(|&b| b == 0));
}
