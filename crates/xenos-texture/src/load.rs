//! Loading textures from guest memory into upload-ready host data.

use crate::address::{tiled_2d, tiled_3d};
use crate::convert::{convert_block, f16_to_f32};
use crate::fetch::{DataDimension, FetchConstantType, SwizzleSource, TextureFetchConstant};
use crate::host::{Conversion, HostCaps, HostChannel, HostFormat, HostMapping, MappingError, host_mapping};
use crate::layout::{GuestRange, TextureGuestInfo, packed_mip_offset};

/// Read access to guest physical memory. Out-of-range bytes should be filled with zeros.
pub trait GuestMemory {
    fn read(&self, phys_addr: u32, out: &mut [u8]);

    /// If this memory tracks writes: a number that has changed whenever a byte of
    /// `[phys_addr, phys_addr + len)` was written since the last call for it. Without tracking
    /// (`None`), callers compare contents instead.
    fn version(&self, _phys_addr: u32, _len: u32) -> Option<u64> {
        None
    }
}

/// A flat byte slice where index == physical address.
impl GuestMemory for [u8] {
    fn read(&self, phys_addr: u32, out: &mut [u8]) {
        let start = phys_addr as usize;
        let available = self.len().saturating_sub(start).min(out.len());
        if available > 0 {
            out[..available].copy_from_slice(&self[start..start + available]);
        }
        out[available..].fill(0);
    }
}

impl GuestMemory for Vec<u8> {
    fn read(&self, phys_addr: u32, out: &mut [u8]) {
        self.as_slice().read(phys_addr, out);
    }
}

impl<T: GuestMemory + ?Sized> GuestMemory for &T {
    fn read(&self, phys_addr: u32, out: &mut [u8]) {
        (**self).read(phys_addr, out);
    }
}

/// Host texture dimension to create.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostDimension {
    D1,
    D2,
    /// Stacked 2D textures (array layers).
    D2Array,
    /// Six layers in +X, -X, +Y, -Y, +Z, -Z order (same as WebGPU cube layers).
    Cube,
    D3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadOptions {
    pub caps: HostCaps,
    /// Load the signed view (two's complement interpretation) instead of the unsigned one.
    pub signed: bool,
    /// `bytes_per_row` alignment of the output (1 for `writeTexture`, 256 for buffer copies).
    pub row_alignment: u32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { caps: HostCaps::default(), signed: false, row_alignment: 1 }
    }
}

/// One host mip level with all its layers / depth slices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedLevel {
    pub level: u32,
    /// Logical size of this level in texels.
    pub width: u32,
    pub height: u32,
    /// Array layers (2D array / cube) or depth slices (3D).
    pub depth_or_array_layers: u32,
    /// Copy extent in texels: the logical size rounded up to the host block size.
    pub copy_width: u32,
    pub copy_height: u32,
    pub bytes_per_row: u32,
    /// Block rows per layer / slice.
    pub rows_per_image: u32,
    /// `depth_or_array_layers * rows_per_image * bytes_per_row` bytes.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTexture {
    pub mapping: HostMapping,
    pub dimension: HostDimension,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
    /// Level count of the host texture to create (levels `0..=mip_max_level`).
    pub mip_level_count: u32,
    /// Lowest level the sampler should use (`lod_min_clamp`).
    pub mip_min_level: u32,
    /// Levels that have data, in increasing order. Level 0 is absent when the fetch constant
    /// has no base address.
    pub levels: Vec<LoadedLevel>,
    /// Guest memory read to build the texture (for invalidation tracking).
    pub guest_ranges: Vec<GuestRange>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    NotATexture(FetchConstantType),
    /// Neither a base nor a mip address is set.
    NoData,
    Mapping(MappingError),
    /// Front buffer format without an RGBA8 conversion.
    UnsupportedFrontBufferFormat(HostFormat),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::NotATexture(t) => write!(f, "fetch constant type {t:?} is not a texture"),
            LoadError::NoData => write!(f, "texture has no base or mip address"),
            LoadError::Mapping(e) => e.fmt(f),
            LoadError::UnsupportedFrontBufferFormat(h) => write!(f, "front buffer host format {h:?} unsupported"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<MappingError> for LoadError {
    fn from(e: MappingError) -> Self {
        LoadError::Mapping(e)
    }
}

/// Returns the host dimension for a fetch constant.
pub fn host_dimension(fetch: &TextureFetchConstant, info: &TextureGuestInfo) -> HostDimension {
    match fetch.dimension {
        DataDimension::D1 if info.subresources.wide_1d => HostDimension::D2,
        DataDimension::D1 => HostDimension::D1,
        DataDimension::D2OrStacked if fetch.stacked => HostDimension::D2Array,
        DataDimension::D2OrStacked => HostDimension::D2,
        DataDimension::D3 => HostDimension::D3,
        DataDimension::Cube => HostDimension::Cube,
    }
}

/// Loads every level present in guest memory into host data in the mapped host format.
pub fn load_texture(
    fetch: &TextureFetchConstant,
    mem: &dyn GuestMemory,
    options: &LoadOptions,
) -> Result<LoadedTexture, LoadError> {
    if !matches!(fetch.fetch_type, FetchConstantType::Texture | FetchConstantType::InvalidTexture) {
        return Err(LoadError::NotATexture(fetch.fetch_type));
    }
    let info = TextureGuestInfo::from_fetch(fetch);
    let sub = info.subresources;
    if sub.base_page == 0 && sub.mip_page == 0 {
        return Err(LoadError::NoData);
    }
    let mapping = host_mapping(info.format, options.signed, options.caps, sub.width, sub.height)?;
    let dimension = host_dimension(fetch, &info);
    let max_level = info.layout.max_level;
    let first_level = if sub.has_base() { 0 } else { 1 };
    let mut levels = Vec::new();
    for level in first_level..=max_level {
        if let Some(l) = load_level(fetch, &info, &mapping, mem, options, level) {
            levels.push(l);
        }
    }
    Ok(LoadedTexture {
        mapping,
        dimension,
        width: sub.width,
        height: sub.height,
        depth_or_array_layers: sub.depth_or_array_size,
        mip_level_count: max_level + 1,
        mip_min_level: sub.mip_min_level,
        levels,
        guest_ranges: info.memory_ranges(),
    })
}

/// Reads a whole stored level (all slices) and applies the endian swap.
fn read_region(mem: &dyn GuestMemory, address: u32, length: u32, fetch: &TextureFetchConstant) -> Vec<u8> {
    let mut data = vec![0u8; (length as usize).next_multiple_of(4)];
    mem.read(address, &mut data);
    fetch.endianness.swap_in_place(&mut data);
    data
}

fn load_level(
    fetch: &TextureFetchConstant,
    info: &TextureGuestInfo,
    mapping: &HostMapping,
    mem: &dyn GuestMemory,
    options: &LoadOptions,
    level: u32,
) -> Option<LoadedLevel> {
    let (address, layout, _) = info.level_storage(level)?;
    let sub = &info.subresources;
    let format_info = info.format.info();
    let (bw, bh, bpb) = (format_info.block_width, format_info.block_height, format_info.bytes_per_block());
    let bpb_log2 = format_info.bytes_per_block_log2();
    let is_3d = sub.dimension == DataDimension::D3;
    let tiled = fetch.tiled && sub.dimension != DataDimension::D1;

    let width = (sub.width >> level).max(1);
    let height = (sub.height >> level).max(1);
    let depth = (sub.depth() >> level).max(1);
    let layers = sub.array_size();
    let (ox, oy, oz) = if info.packed_mips && level >= info.layout.packed_level {
        packed_mip_offset(sub.width, sub.height, sub.depth(), info.format, level).unwrap_or((0, 0, 0))
    } else {
        (0, 0, 0)
    };

    let region = read_region(mem, address, layout.level_data_extent_bytes, fetch);
    let pitch_blocks = layout.row_pitch_bytes / bpb;
    let z_rows = layout.z_slice_stride_block_rows;
    let guest_offset = |x: u32, y: u32, z: u32| -> i64 {
        if tiled {
            if is_3d {
                tiled_3d(x as i32, y as i32, z as i32, pitch_blocks, z_rows, bpb_log2)
            } else {
                i64::from(tiled_2d(x as i32, y as i32, pitch_blocks, bpb_log2))
            }
        } else {
            (u64::from(z) * u64::from(z_rows) + u64::from(y)) as i64 * i64::from(layout.row_pitch_bytes)
                + i64::from(x) * i64::from(bpb)
        }
    };

    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let host_block = mapping.host_format.block_size();
    let host_unit_bytes = mapping.host_format.bytes_per_block();
    // Host data either keeps guest blocks (compressed passthrough or 1x1 blocks) or is per texel.
    let per_block = host_block > 1 || (bw == 1 && bh == 1);
    let (units_x, units_y) = if per_block { (blocks_x, blocks_y) } else { (width, height) };
    let bytes_per_row = (units_x * host_unit_bytes).next_multiple_of(options.row_alignment.max(1));
    let slices = if is_3d { depth } else { layers };
    let mut data = vec![0u8; bytes_per_row as usize * units_y as usize * slices as usize];
    let image_bytes = bytes_per_row as usize * units_y as usize;

    let texels_per_block = (bw * bh) as usize;
    let mut converted = vec![0u8; (texels_per_block * host_unit_bytes as usize).max(bpb as usize)];
    let mut src = vec![0u8; bpb as usize];
    for slice in 0..slices {
        let (layer, z) = if is_3d { (0, slice) } else { (slice, 0) };
        let base = i64::from(layer) * i64::from(layout.array_slice_stride_bytes);
        let image = &mut data[slice as usize * image_bytes..(slice as usize + 1) * image_bytes];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let offset = base + guest_offset(bx + ox, by + oy, z + oz);
                match usize::try_from(offset).ok().and_then(|o| region.get(o..o + bpb as usize)) {
                    Some(bytes) => src.copy_from_slice(bytes),
                    None => src.fill(0),
                }
                if per_block {
                    let dst = by as usize * bytes_per_row as usize + (bx * host_unit_bytes) as usize;
                    if mapping.conversion == Conversion::Copy {
                        image[dst..dst + bpb as usize].copy_from_slice(&src);
                    } else {
                        convert_block(mapping.conversion, &src, &mut converted);
                        image[dst..dst + host_unit_bytes as usize]
                            .copy_from_slice(&converted[..host_unit_bytes as usize]);
                    }
                } else {
                    convert_block(mapping.conversion, &src, &mut converted);
                    for ty in 0..bh {
                        let y = by * bh + ty;
                        if y >= height {
                            break;
                        }
                        let x0 = bx * bw;
                        let n = bw.min(width - x0) as usize;
                        let hb = host_unit_bytes as usize;
                        let dst = y as usize * bytes_per_row as usize + x0 as usize * hb;
                        let s = (ty * bw) as usize * hb;
                        image[dst..dst + n * hb].copy_from_slice(&converted[s..s + n * hb]);
                    }
                }
            }
        }
    }
    let (copy_width, copy_height) = if host_block > 1 { (blocks_x * bw, blocks_y * bh) } else { (width, height) };
    Some(LoadedLevel {
        level,
        width,
        height,
        depth_or_array_layers: slices,
        copy_width,
        copy_height,
        bytes_per_row,
        rows_per_image: units_y,
        data,
    })
}

/// Top-down RGBA8 image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

fn host_texel_to_rgba8(format: HostFormat, t: &[u8]) -> Option<[u8; 4]> {
    let unorm = |f: f32| (f.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Some(match format {
        HostFormat::Rgba8Unorm | HostFormat::Rgba8UnormSrgb => [t[0], t[1], t[2], t[3]],
        HostFormat::Bgra8Unorm | HostFormat::Bgra8UnormSrgb => [t[2], t[1], t[0], t[3]],
        HostFormat::Rgb10a2Unorm => {
            let v = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
            let c10 = |s: u32| (((v >> s) & 1023) * 255 + 511) / 1023;
            [c10(0) as u8, c10(10) as u8, c10(20) as u8, ((v >> 30) * 85) as u8]
        }
        HostFormat::Rgba16Float => {
            let h = |i: usize| unorm(f16_to_f32(u16::from_le_bytes([t[i * 2], t[i * 2 + 1]])));
            [h(0), h(1), h(2), h(3)]
        }
        HostFormat::Rgba16Unorm => {
            let h = |i: usize| ((u32::from(u16::from_le_bytes([t[i * 2], t[i * 2 + 1]])) * 255 + 32767) / 65535) as u8;
            [h(0), h(1), h(2), h(3)]
        }
        HostFormat::Rgba32Float => {
            let f = |i: usize| unorm(f32::from_le_bytes([t[i * 4], t[i * 4 + 1], t[i * 4 + 2], t[i * 4 + 3]]));
            [f(0), f(1), f(2), f(3)]
        }
        _ => return None,
    })
}

/// Converts the front buffer described by a VdSwap fetch constant (typically tiled
/// k_8_8_8_8 or k_2_10_10_10_AS_16_16_16_16) to top-down RGBA8 rows, applying the fetch
/// constant swizzle.
pub fn frontbuffer_to_rgba8(fetch: &TextureFetchConstant, mem: &dyn GuestMemory) -> Result<Rgba8Image, LoadError> {
    let mut fb = fetch.clone();
    fb.mip_page = 0;
    fb.mip_max_level = 0;
    fb.mip_min_level = 0;
    fb.dimension = DataDimension::D2OrStacked;
    fb.stacked = false;
    if fb.base_page == 0 {
        return Err(LoadError::NoData);
    }
    let caps = HostCaps::default();
    let texture = load_texture(&fb, mem, &LoadOptions { caps, signed: false, row_alignment: 1 })?;
    let level = &texture.levels[0];
    let host = texture.mapping.host_format;
    if host.is_compressed() || host_texel_to_rgba8(host, &[0u8; 16]).is_none() {
        return Err(LoadError::UnsupportedFrontBufferFormat(host));
    }
    let unit = host.bytes_per_block() as usize;
    let channel = |c: HostChannel| match c {
        HostChannel::R => 0,
        HostChannel::G => 1,
        HostChannel::B => 2,
        HostChannel::A => 3,
    };
    let mut data = Vec::with_capacity(level.width as usize * level.height as usize * 4);
    for y in 0..level.height as usize {
        let row = &level.data[y * level.bytes_per_row as usize..];
        for x in 0..level.width as usize {
            let rgba = host_texel_to_rgba8(host, &row[x * unit..x * unit + unit]).unwrap_or_default();
            for s in fetch.swizzle {
                data.push(match s {
                    SwizzleSource::Zero => 0,
                    SwizzleSource::One => 255,
                    s => rgba[channel(texture.mapping.component_map[s.component().unwrap_or(0)])],
                });
            }
        }
    }
    Ok(Rgba8Image { width: level.width, height: level.height, data })
}
