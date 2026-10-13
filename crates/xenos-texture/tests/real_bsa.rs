//! Local-only check against real Xbox 360 Skyrim textures. Nothing from the game is stored in
//! the repository; the archive is read from the path in `XENOS_TEXTURE_BSA`:
//!
//! ```sh
//! XENOS_TEXTURE_BSA="game/Data/Skyrim - Textures.bsa" XENOS_TEXTURE_LIMIT=300 \
//!     cargo test -p xenos-texture --test real_bsa -- --ignored --nocapture
//! ```
//!
//! `XENOS_TEXTURE_DUMP_DIR` additionally writes decoded RGBA8 images as PAM files, and
//! `XENOS_TEXTURE_VERBOSE` lists mip levels whose mean color departs from the base.

use lzxd::{Lzxd, WindowSize};
use xenos_texture::*;

struct BsaFile {
    name: String,
    offset: u64,
    size: u32,
    compressed: bool,
}

fn le32(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn be32(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}

/// Minimal BSA v104 directory reader.
fn read_bsa_index(data: &[u8]) -> (u32, Vec<BsaFile>) {
    assert_eq!(&data[0..4], b"BSA\0");
    let (offset, flags, folders, files, file_names_len) =
        (le32(data, 8) as usize, le32(data, 12), le32(data, 16), le32(data, 20), le32(data, 28) as usize);
    let mut p = offset;
    let mut counts = Vec::new();
    for _ in 0..folders {
        counts.push(le32(data, p + 8));
        p += 16;
    }
    let mut out = Vec::new();
    for count in counts {
        let len = data[p] as usize;
        let folder = String::from_utf8_lossy(&data[p + 1..p + len]).trim_end_matches('\0').to_string();
        p += 1 + len;
        for _ in 0..count {
            let size = le32(data, p + 8);
            out.push(BsaFile {
                name: folder.clone(),
                offset: u64::from(le32(data, p + 12)),
                size: size & 0x3FFF_FFFF,
                compressed: (size & 0x4000_0000 != 0) != (flags & 4 != 0),
            });
            p += 16;
        }
    }
    assert_eq!(out.len(), files as usize);
    for (file, name) in out.iter_mut().zip(data[p..p + file_names_len].split(|&b| b == 0)) {
        file.name = format!("{}\\{}", file.name, String::from_utf8_lossy(name));
    }
    (flags, out)
}

/// XMemCompress (LZX, 128 KB window) stream with Xbox frame headers.
fn xmem_decompress(input: &[u8]) -> Option<Vec<u8>> {
    let mut lz = Lzxd::new(WindowSize::KB128);
    let mut out = Vec::new();
    let mut p = 0;
    while p + 2 <= input.len() {
        let (frame, block, header) = if input[p] == 0xFF {
            let be16 = |i: usize| usize::from(u16::from_be_bytes([input[i], input[i + 1]]));
            (be16(p + 1), be16(p + 3), 5)
        } else {
            (0x8000, usize::from(u16::from_be_bytes([input[p], input[p + 1]])), 2)
        };
        if block == 0 || p + header + block > input.len() {
            break;
        }
        out.extend_from_slice(lz.decompress_next(&input[p + header..p + header + block], frame).ok()?);
        p += header + block;
    }
    Some(out)
}

struct Ddx {
    fetch: TextureFetchConstant,
    mips: Vec<u8>,
    base: Vec<u8>,
    high: Option<(TextureFetchConstant, Vec<u8>)>,
}

fn fetch_from_header(h: &[u8]) -> TextureFetchConstant {
    let d: [u32; 6] = std::array::from_fn(|i| be32(h, 28 + i * 4));
    TextureFetchConstant::decode(d)
}

/// Bethesda "3XDO" texture: magic + 4 bytes, D3D texture header (52 bytes, big-endian fetch
/// constant at +28), total uncompressed size, then three compressed chunk sizes. With a mip
/// address the chunks are mips, base, and an optional high-resolution base preceded by its
/// own D3D header; without one, the first chunk is the base (including any packed tail).
/// "3XDR" files use a different, unidentified data layout and are skipped.
fn parse_ddx(data: &[u8]) -> Option<Ddx> {
    if &data[0..4] != b"3XDO" {
        return None;
    }
    let fetch = fetch_from_header(&data[8..60]);
    let sizes = [be32(data, 64) as usize, be32(data, 68) as usize, be32(data, 72) as usize];
    let mut p = 76;
    let mut chunks = Vec::new();
    for &len in &sizes[..2] {
        chunks.push(if len != 0 { xmem_decompress(&data[p..p + len])? } else { Vec::new() });
        p += len;
    }
    let (base, mips) = if fetch.mip_page != 0 && !chunks[1].is_empty() {
        (chunks.pop()?, chunks.pop()?)
    } else {
        (chunks.swap_remove(0), Vec::new())
    };
    let high_len = sizes[2];
    let high = if high_len != 0 && p + 52 + high_len <= data.len() {
        Some((fetch_from_header(&data[p..p + 52]), xmem_decompress(&data[p + 52..p + 52 + high_len])?))
    } else {
        None
    };
    Some(Ddx { fetch, mips, base, high })
}

fn mean_rgb(l: &LoadedLevel) -> [f64; 3] {
    let mut sum = [0u64; 3];
    for y in 0..l.height as usize {
        for x in 0..l.width as usize {
            for (c, s) in sum.iter_mut().enumerate() {
                *s += u64::from(l.data[y * l.bytes_per_row as usize + x * 4 + c]);
            }
        }
    }
    let n = f64::from(l.width * l.height);
    sum.map(|s| s as f64 / n)
}

/// Mean absolute difference of horizontally and vertically adjacent texels (RGBA8 rows).
fn roughness(data: &[u8], w: usize, h: usize, stride: usize) -> f64 {
    let (mut sum, mut n) = (0u64, 0u64);
    for y in 0..h {
        for x in 0..w {
            let i = y * stride + x * 4;
            for (dx, dy) in [(1, 0), (0, 1)] {
                if x + dx < w && y + dy < h {
                    let j = (y + dy) * stride + (x + dx) * 4;
                    for c in 0..3 {
                        sum += u64::from(data[i + c].abs_diff(data[j + c]));
                    }
                    n += 3;
                }
            }
        }
    }
    if n == 0 { 0.0 } else { sum as f64 / n as f64 }
}

fn dump(dir: &str, name: &str, level: &LoadedLevel) {
    let file = name.replace(['\\', '/'], "_");
    let mut pam = format!(
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
        level.width, level.height
    )
    .into_bytes();
    for y in 0..level.height as usize {
        let row = y * level.bytes_per_row as usize;
        pam.extend_from_slice(&level.data[row..row + level.width as usize * 4]);
    }
    std::fs::write(format!("{dir}/{file}.l{}.pam", level.level), pam).unwrap();
}

#[test]
#[ignore = "needs a local Skyrim - Textures.bsa (XENOS_TEXTURE_BSA)"]
fn real_textures_decode() {
    let Ok(path) = std::env::var("XENOS_TEXTURE_BSA") else {
        eprintln!("XENOS_TEXTURE_BSA not set; skipping");
        return;
    };
    let limit: usize = std::env::var("XENOS_TEXTURE_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let dump_dir = std::env::var("XENOS_TEXTURE_DUMP_DIR").ok();
    let data = std::fs::read(&path).unwrap();
    let (flags, files) = read_bsa_index(&data);
    let embedded_names = flags & 0x100 != 0;

    let mut formats = std::collections::BTreeMap::<String, usize>::new();
    let (mut checked, mut skipped, mut smoother, mut rough_total, mut scrambled_total) = (0, 0, 0, 0.0, 0.0);
    let (mut packed_levels, mut packed_consistent) = (0, 0);
    for file in files.iter().filter(|f| f.name.ends_with(".ddx")).take(limit) {
        let mut raw = &data[file.offset as usize..file.offset as usize + file.size as usize];
        if embedded_names {
            raw = &raw[1 + raw[0] as usize..];
        }
        if file.compressed {
            eprintln!("{}: zlib-compressed BSA entry", file.name);
            skipped += 1;
            continue;
        }
        let Some(ddx) = parse_ddx(raw) else {
            eprintln!("{}: not a 3XDO texture ({:?})", file.name, String::from_utf8_lossy(&raw[..4]));
            skipped += 1;
            continue;
        };
        *formats.entry(format!("{:?}", ddx.fetch.format)).or_default() += 1;

        let mut textures = vec![(ddx.fetch.clone(), ddx.base, ddx.mips)];
        if let Some((hf, hdata)) = ddx.high {
            textures.push((hf, hdata, Vec::new()));
        }
        for (mut fetch, base, mips) in textures {
            const BASE_PAGE: u32 = 0x100;
            let mip_page_rel = fetch.mip_page;
            fetch.base_page = BASE_PAGE;
            fetch.mip_page = if mip_page_rel != 0 && !mips.is_empty() { BASE_PAGE + mip_page_rel } else { 0 };
            let info = TextureGuestInfo::from_fetch(&fetch);
            let (base_size, mips_size) = info.total_sizes();
            assert!(base.len() as u32 >= base_size, "{}: base {} < {}", file.name, base.len(), base_size);
            if fetch.mip_page != 0 {
                assert!(mips.len() as u32 >= mips_size, "{}: mips {} < {}", file.name, mips.len(), mips_size);
            }
            let mut mem =
                vec![0u8; ((BASE_PAGE << 12) as usize) + (mip_page_rel << 12) as usize + mips.len() + base.len()];
            let b = (BASE_PAGE << 12) as usize;
            mem[b..b + base.len()].copy_from_slice(&base);
            if fetch.mip_page != 0 {
                let m = (fetch.mip_page << 12) as usize;
                mem[m..m + mips.len()].copy_from_slice(&mips);
            }

            let bc_caps = HostCaps { texture_compression_bc: true, norm16: false };
            let passthrough = match load_texture(&fetch, &mem, &LoadOptions { caps: bc_caps, ..Default::default() }) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}: {e}", file.name);
                    skipped += 1;
                    continue;
                }
            };
            let decoded = load_texture(&fetch, &mem, &LoadOptions::default()).unwrap();
            assert_eq!(passthrough.levels.len(), decoded.levels.len());
            assert_eq!(decoded.levels.len() as u32, info.layout.max_level + 1, "{}", file.name);
            if decoded.mapping.host_format != HostFormat::Rgba8Unorm {
                checked += 1;
                continue;
            }
            let l0 = &decoded.levels[0];
            let (w, h) = (l0.width as usize, l0.height as usize);
            let rough = roughness(&l0.data, w, h, l0.bytes_per_row as usize);
            // Same blocks decoded in guest memory order (i.e. without untiling) for comparison.
            let mut linear = fetch.clone();
            linear.tiled = false;
            let scrambled = load_texture(&linear, &mem, &LoadOptions::default()).unwrap();
            let s0 = &scrambled.levels[0];
            let scrambled_rough = roughness(&s0.data, w, h, s0.bytes_per_row as usize);
            if fetch.tiled && w >= 64 && h >= 64 {
                rough_total += rough;
                scrambled_total += scrambled_rough;
                if rough < scrambled_rough {
                    smoother += 1;
                }
            }
            // Mips (including packed-tail levels) of the same image keep roughly its mean color.
            let base_mean = mean_rgb(l0);
            for l in &decoded.levels[1..] {
                if info.layout.packed_level().is_some_and(|p| l.level >= p) {
                    packed_levels += 1;
                    let mean = mean_rgb(l);
                    if (0..3).all(|c| (mean[c] - base_mean[c]).abs() < 48.0) {
                        packed_consistent += 1;
                    } else if std::env::var("XENOS_TEXTURE_VERBOSE").is_ok() {
                        eprintln!("{} level {}: mean {mean:?} vs base {base_mean:?}", file.name, l.level);
                    }
                }
            }
            if let Some(dir) = &dump_dir {
                for l in decoded.levels.iter().take(2) {
                    dump(dir, &format!("{}.{}x{}", file.name, l0.width, l0.height), l);
                }
            }
            checked += 1;
        }
    }
    eprintln!("checked {checked} textures, skipped {skipped}; formats {formats:?}");
    eprintln!(
        "untiled smoother than raw order in {smoother} cases; mean roughness {:.2} vs {:.2}",
        rough_total / checked.max(1) as f64,
        scrambled_total / checked.max(1) as f64
    );
    eprintln!("packed-tail levels with base-consistent mean color: {packed_consistent}/{packed_levels}");
    assert!(checked > 0);
    assert!(packed_consistent * 10 >= packed_levels * 9);
    assert!(rough_total < scrambled_total * 0.8);
}
