//! Host texture and sampler cache for Xenos texture fetches.
//!
//! [`TextureCache::bind`] turns one shader texture binding ([`xenos_shader::TextureBinding`]) plus
//! the 192 fetch constant dwords (registers `0x4800..0x48BF`, host order) into WebGPU views and a
//! sampler ([`BoundTexture`]). Following the xenos-shader contract, views always expose the
//! *unsigned* guest data (UNORM or float, never sRGB); the shader applies signedness, bias,
//! gamma, integer scaling, swizzle, exponent and LOD bias itself.
//!
//! Caching:
//! - Host textures are keyed by everything that determines their contents ([`TextureKey`]): the
//!   resolved subresources (addresses, sizes, pitch, mip range), base format, tiling, endianness,
//!   packing and dimension. Fetch constant aliases with identical data share one texture.
//! - Guest CPU writes are not tracked, so a texture's guest ranges are hashed on its first use
//!   in each frame (after [`TextureCache::end_frame`]); changed contents load a new texture.
//! - [`TextureCache::invalidate`] drops overlapping textures immediately (e.g. after a resolve).
//! - [`TextureCache::end_frame`] evicts textures unused for [`CacheConfig::max_unused_frames`]
//!   frames, then least recently used ones while over [`CacheConfig::max_host_bytes`].
//!
//! Textures are never overwritten in place: replaced or evicted textures stay alive for as long
//! as bind groups created from earlier [`BoundTexture`]s reference them.
//!
//! Anything that cannot be loaded (invalid or empty fetch constants, unsupported formats, sizes
//! over the device limits, dimension mismatches) binds a zeroed 1x1 fallback of the right view
//! dimension and logs a warning once per fetch constant.

mod format;
mod gpu_load;
mod hash;

pub use gpu_load::GpuSource;

use std::collections::HashSet;

use rustc_hash::FxHashMap;

use gpu_webgpu::{
    AddressMode, Device, FeatureName, FilterMode, MipmapFilterMode, Sampler, SamplerBindingType, SamplerDescriptor,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureSampleType, TextureUsage,
    TextureView, TextureViewDescriptor, TextureViewDimension, extent, origin,
};
use xenos_shader::TextureBinding;
use xenos_shader::TextureBindingDimension;
use xenos_shader::parsed;
use xenos_texture::load::host_dimension;
use xenos_texture::{
    AnisoFilter, DataDimension, Endian, FetchConstantType, FetchInstructionFilters, GuestMemory, GuestRange, HostCaps,
    HostDimension, LoadOptions, TextureFetchConstant, TextureFilter, TextureGuestInfo, load_texture,
    sampler_descriptor,
};

/// Fetch constant dwords (`SHADER_CONSTANT_FETCH_00_0..31_5`).
pub const FETCH_CONSTANT_DWORDS: usize = 192;

/// First fetch constant register.
pub const FETCH_CONSTANT_REGISTER: u32 = xenos_state::registers::SHADER_CONSTANT_FETCH_00_0;

/// The fetch constants within a register file indexed by register number.
pub fn fetch_constants(registers: &[u32]) -> Option<&[u32; FETCH_CONSTANT_DWORDS]> {
    let start = FETCH_CONSTANT_REGISTER as usize;
    registers.get(start..start + FETCH_CONSTANT_DWORDS)?.try_into().ok()
}

/// Physical address bits of guest memory (texture ranges are physical).
const PHYSICAL_ADDRESS_MASK: u32 = 0x1FFF_FFFF;

/// Bound on remembered failure keys / warnings before they are forgotten (and may log again).
const MAX_REMEMBERED_FAILURES: usize = 4096;

/// Upper limit of WebGPU `maxAnisotropy`.
const MAX_ANISOTROPY: u16 = 16;

/// Optional device features as the xenos-texture host capabilities (BC passthrough with
/// `texture-compression-bc`, 16-bit normalized formats with `texture-formats-tier1`).
/// wgpu-native's `WGPUNativeFeature_TextureFormat16bitNorm` also enables them.
pub fn host_caps(device: &Device) -> HostCaps {
    HostCaps {
        texture_compression_bc: device.has_feature(FeatureName::TextureCompressionBC),
        norm16: device.has_feature(FeatureName::TextureFormatsTier1)
            || (cfg!(not(target_os = "emscripten")) && device.has_feature(NATIVE_TEXTURE_FORMAT_16BIT_NORM)),
    }
}

/// `WGPUNativeFeature_TextureFormat16bitNorm` (wgpu-native `wgpu.h`).
pub const NATIVE_TEXTURE_FORMAT_16BIT_NORM: FeatureName = FeatureName(0x0003_000B);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    /// Textures not bound for more than this many frames are dropped at [`TextureCache::end_frame`].
    pub max_unused_frames: u64,
    /// Soft budget for uploaded texture data; least recently used textures not bound in the
    /// last frame are dropped while it is exceeded.
    pub max_host_bytes: u64,
    /// Load fetch constants of type `InvalidTexture` too (Xenia's
    /// `gpu_allow_invalid_fetch_constants`); otherwise they bind the fallback.
    pub allow_invalid_fetch_constants: bool,
    /// Convert textures on the GPU (raw guest bytes in, a compute pass untiles and converts) when
    /// the format allows, instead of texel by texel on the CPU.
    pub gpu_conversion: bool,
    /// Reuse dropped textures for new loads of the same shape and format (`TexturePool`).
    pub reuse_textures: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_unused_frames: 300,
            max_host_bytes: 512 << 20,
            allow_invalid_fetch_constants: false,
            gpu_conversion: true,
            reuse_textures: true,
        }
    }
}

/// Counters for diagnostics and tests. `textures`, `host_bytes`, `pooled`, `pooled_bytes`,
/// `retired` and `samplers` are current values, the rest are totals since creation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub textures: usize,
    pub host_bytes: u64,
    /// Dropped textures kept for reuse.
    pub pooled: usize,
    pub pooled_bytes: u64,
    /// Dropped textures waiting for their frame to finish (then pooled or destroyed).
    pub retired: usize,
    pub samplers: usize,
    /// Textures created from guest memory (including reloads).
    pub loads: u64,
    /// Loads caused by changed guest memory found during revalidation.
    pub reloads: u64,
    /// Binds served by an existing texture.
    pub hits: u64,
    /// Binds that used a fallback view instead of guest data.
    pub fallbacks: u64,
    /// Textures dropped by [`TextureCache::invalidate`].
    pub invalidated: u64,
    /// Textures dropped by age or budget.
    pub evicted: u64,
    /// Guest bytes hashed for revalidation.
    pub hashed_bytes: u64,
    /// Textures loaded from GPU memory (resolve output).
    pub gpu_loads: u64,
    /// Textures copied raw from guest memory and converted on the GPU.
    pub gpu_conversions: u64,
    /// Textures converted on the CPU.
    pub cpu_loads: u64,
    /// Loads into a reused (pooled) texture instead of a new one.
    pub reused: u64,
}

/// Views and sampler for one shader texture binding.
///
/// - `view` matches `binding.dimension`: `texture_2d` (D2), `texture_cube` (Cube) or
///   `texture_3d` (D3OrStacked) and goes to `binding.texture_binding`.
/// - `stacked_view` is the `texture_2d_array` view for `binding.stacked_texture_binding` (only
///   for D3OrStacked). Whichever of the 3D / stacked views the fetch constant does not select is
///   a zeroed dummy.
/// - `sampler` goes to `binding.sampler_binding`.
/// - `sample_type` (`Float` or `UnfilterableFloat`) and `filterable` describe the views for the
///   bind group layout: unfilterable views come with non-filtering samplers, so the layout needs
///   `TextureSampleType::UnfilterableFloat` + `SamplerBindingType::NonFiltering` (see
///   [`BoundTexture::sampler_binding_type`]). Filterable views are also valid in unfilterable
///   layout entries, the reverse is not.
#[derive(Clone, Debug)]
pub struct BoundTexture {
    pub view: TextureView,
    pub stacked_view: Option<TextureView>,
    pub sampler: Sampler,
    pub sample_type: TextureSampleType,
    pub filterable: bool,
    /// No guest data is bound (fallback views only).
    pub fallback: bool,
}

impl BoundTexture {
    pub fn sampler_binding_type(&self) -> SamplerBindingType {
        if self.filterable { SamplerBindingType::Filtering } else { SamplerBindingType::NonFiltering }
    }
}

/// Everything that determines a host texture's contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub dimension: DataDimension,
    pub stacked: bool,
    /// Base format (aliases with identical data share textures).
    pub format: xenos_texture::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_size: u32,
    pub pitch_div_32: u32,
    pub tiled: bool,
    pub endianness: Endian,
    pub packed_mips: bool,
    pub base_page: u32,
    pub mip_page: u32,
    pub mip_max_level: u32,
}

impl TextureKey {
    /// `None` when the fetch constant references no guest data.
    pub fn from_fetch(fetch: &TextureFetchConstant) -> Option<Self> {
        let info = TextureGuestInfo::from_fetch(fetch);
        let sub = info.subresources;
        if sub.base_page == 0 && sub.mip_page == 0 {
            return None;
        }
        Some(TextureKey {
            dimension: fetch.dimension,
            stacked: fetch.stacked && fetch.dimension == DataDimension::D2OrStacked,
            format: info.format,
            width: sub.width,
            height: sub.height,
            depth_or_array_size: sub.depth_or_array_size,
            pitch_div_32: sub.pitch_div_32,
            tiled: fetch.tiled && fetch.dimension != DataDimension::D1,
            endianness: fetch.endianness,
            packed_mips: info.packed_mips,
            base_page: sub.base_page,
            mip_page: sub.mip_page,
            mip_max_level: sub.mip_max_level,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewKind {
    D2,
    Cube,
    D3,
    D2Array,
}

impl ViewKind {
    fn index(self) -> usize {
        self as usize
    }

    fn dimension(self) -> TextureViewDimension {
        match self {
            ViewKind::D2 => TextureViewDimension::D2,
            ViewKind::Cube => TextureViewDimension::Cube,
            ViewKind::D3 => TextureViewDimension::D3,
            ViewKind::D2Array => TextureViewDimension::D2Array,
        }
    }
}

struct Entry {
    texture: Texture,
    /// What the texture can be reused for once dropped (see `TexturePool`).
    pool_key: PoolKey,
    is_3d: bool,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    filterable: bool,
    views: [Option<TextureView>; 4],
    ranges: Vec<GuestRange>,
    hash: u64,
    bytes: u64,
    last_used: u64,
    validated: u64,
}

impl Entry {
    /// The texture and its views, for the pool.
    fn pooled(&self) -> Pooled {
        Pooled { key: self.pool_key, texture: self.texture.clone(), views: self.views.clone(), bytes: self.bytes }
    }

    fn supports(&self, kind: ViewKind) -> bool {
        match kind {
            ViewKind::D2 | ViewKind::D2Array => !self.is_3d,
            ViewKind::Cube => !self.is_3d && self.depth_or_array_layers >= 6 && self.width == self.height,
            ViewKind::D3 => self.is_3d,
        }
    }

    fn view(&mut self, kind: ViewKind) -> Option<TextureView> {
        if !self.supports(kind) {
            return None;
        }
        let texture = &self.texture;
        let layers = self.depth_or_array_layers;
        let view = self.views[kind.index()].get_or_insert_with(|| {
            texture.create_view(Some(&TextureViewDescriptor {
                label: "xenos texture view",
                dimension: kind.dimension(),
                array_layer_count: Some(match kind {
                    ViewKind::D2 | ViewKind::D3 => 1,
                    ViewKind::Cube => 6,
                    ViewKind::D2Array => layers,
                }),
                aspect: TextureAspect::All,
                ..Default::default()
            }))
        });
        Some(view.clone())
    }
}

/// Textures with the same shape and format are interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    width: u32,
    height: u32,
    layers: u32,
    mip_level_count: u32,
    is_3d: bool,
    format: gpu_webgpu::TextureFormat,
}

/// A dropped texture with its views, reusable once nothing recorded uses it anymore.
struct Pooled {
    key: PoolKey,
    texture: Texture,
    views: [Option<TextureView>; 4],
    bytes: u64,
}

/// Textures kept for reuse at most per key, and in all.
const POOL_PER_KEY: usize = 4;
const POOL_MAX_BYTES: u64 = 256 << 20;
/// Frames a pooled texture waits for reuse before it's destroyed. Reloads of resolved render
/// targets take theirs back within a frame or two; without a limit the pool keeps the textures
/// of every shape seen once (~130 MiB of GPU memory in Skyrim's Helgen) until it's full.
const POOL_MAX_FRAMES: u64 = 60;

/// Dropped textures kept for new loads of the same shape and format: titles resolve render
/// targets to the same addresses every frame, so the textures reading them are reloaded every
/// frame, and creating textures (and the views and bind groups that follow) is expensive in
/// browsers. A reused texture keeps its views, so bind groups cached with them stay valid.
#[derive(Default)]
struct TexturePool {
    /// Pooled textures with the frame they were given back in, oldest first.
    free: FxHashMap<PoolKey, Vec<(Pooled, u64)>>,
    bytes: u64,
}

impl TexturePool {
    /// Keeps `p` (dropped in `frame`) if there's room; otherwise destroys it.
    fn give(&mut self, p: Pooled, frame: u64) {
        let list = self.free.entry(p.key).or_default();
        if list.len() >= POOL_PER_KEY || self.bytes + p.bytes > POOL_MAX_BYTES {
            p.texture.destroy();
            return;
        }
        self.bytes += p.bytes;
        list.push((p, frame));
    }

    fn take(&mut self, key: &PoolKey) -> Option<Pooled> {
        let (p, _) = self.free.get_mut(key)?.pop()?;
        self.bytes -= p.bytes;
        Some(p)
    }

    /// Destroys the textures pooled before `oldest`.
    fn expire(&mut self, oldest: u64) {
        let bytes = &mut self.bytes;
        self.free.retain(|_, list| {
            list.retain(|(p, frame)| {
                let keep = *frame >= oldest;
                if !keep {
                    p.texture.destroy();
                    *bytes -= p.bytes;
                }
                keep
            });
            !list.is_empty()
        });
    }
}

/// Zeroed 1x1 views for every view dimension.
struct Fallback {
    views: [TextureView; 4],
}

impl Fallback {
    fn new(device: &Device) -> Self {
        let usage = TextureUsage::TextureBinding | TextureUsage::CopySrc;
        let layered = device.create_texture(&TextureDescriptor {
            label: "xenos fallback 2d",
            size: extent(1, 1, 6),
            format: gpu_webgpu::TextureFormat::RGBA8Unorm,
            usage,
            ..Default::default()
        });
        let volume = device.create_texture(&TextureDescriptor {
            label: "xenos fallback 3d",
            size: extent(1, 1, 1),
            dimension: TextureDimension::D3,
            format: gpu_webgpu::TextureFormat::RGBA8Unorm,
            usage,
            ..Default::default()
        });
        let view = |texture: &Texture, kind: ViewKind, layers: u32| {
            texture.create_view(Some(&TextureViewDescriptor {
                label: "xenos fallback view",
                dimension: kind.dimension(),
                array_layer_count: Some(layers),
                aspect: TextureAspect::All,
                ..Default::default()
            }))
        };
        Fallback {
            views: [
                view(&layered, ViewKind::D2, 1),
                view(&layered, ViewKind::Cube, 6),
                view(&volume, ViewKind::D3, 1),
                view(&layered, ViewKind::D2Array, 1),
            ],
        }
    }

    fn view(&self, kind: ViewKind) -> TextureView {
        self.views[kind.index()].clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SamplerKey {
    address: [AddressMode; 3],
    mag: FilterMode,
    min: FilterMode,
    mip: MipmapFilterMode,
    lod_min_bits: u32,
    lod_max_bits: u32,
    anisotropy: u16,
}

#[derive(Clone, Copy)]
struct DeviceLimits {
    max_2d: u32,
    max_3d: u32,
    max_layers: u32,
}

pub struct TextureCache {
    caps: HostCaps,
    bc_sliced_3d: bool,
    float32_filterable: bool,
    limits: DeviceLimits,
    config: CacheConfig,
    frame: u64,
    textures: FxHashMap<TextureKey, Entry>,
    samplers: FxHashMap<SamplerKey, Sampler>,
    fallback: Fallback,
    /// Keys whose load failed for reasons independent of memory contents.
    failed: HashSet<TextureKey>,
    /// Format and conversion of the textures converted on the CPU so far (each logged once).
    cpu_load_kinds: HashSet<String>,
    /// Textures dropped from the cache, to reuse or destroy once submitted (see `destroy_retired`).
    retired: Vec<Pooled>,
    /// Textures retired before the last `destroy_retired`, released by the next one.
    retiring: Vec<Pooled>,
    pool: TexturePool,
    warned: HashSet<([u32; 6], u8)>,
    scratch: Vec<u8>,
    stats: CacheStats,
    gpu_loader: gpu_load::GpuLoader,
}

impl TextureCache {
    /// `caps` is masked with the device's features, so passing more than it supports is safe.
    pub fn new(device: &Device, caps: HostCaps) -> Self {
        Self::with_config(device, caps, CacheConfig::default())
    }

    /// A cache using every optional feature the device has.
    pub fn for_device(device: &Device) -> Self {
        Self::new(device, host_caps(device))
    }

    pub fn with_config(device: &Device, caps: HostCaps, config: CacheConfig) -> Self {
        let available = host_caps(device);
        let caps = HostCaps {
            texture_compression_bc: caps.texture_compression_bc && available.texture_compression_bc,
            norm16: caps.norm16 && available.norm16,
        };
        let l = device.limits();
        TextureCache {
            caps,
            bc_sliced_3d: device.has_feature(FeatureName::TextureCompressionBCSliced3D),
            float32_filterable: device.has_feature(FeatureName::Float32Filterable),
            limits: DeviceLimits {
                max_2d: l.maxTextureDimension2D,
                max_3d: l.maxTextureDimension3D,
                max_layers: l.maxTextureArrayLayers,
            },
            config,
            frame: 1,
            textures: FxHashMap::default(),
            samplers: FxHashMap::default(),
            fallback: Fallback::new(device),
            gpu_loader: gpu_load::GpuLoader::default(),
            failed: HashSet::new(),
            cpu_load_kinds: HashSet::new(),
            retired: Vec::new(),
            retiring: Vec::new(),
            pool: TexturePool::default(),
            warned: HashSet::new(),
            scratch: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn caps(&self) -> HostCaps {
        self.caps
    }

    pub fn config(&self) -> CacheConfig {
        self.config
    }

    pub fn set_config(&mut self, config: CacheConfig) {
        self.config = config;
    }

    /// Frames ended so far plus one.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            textures: self.textures.len(),
            host_bytes: self.textures.values().map(|e| e.bytes).sum(),
            pooled: self.pool.free.values().map(Vec::len).sum(),
            pooled_bytes: self.pool.bytes,
            retired: self.retired.len() + self.retiring.len(),
            samplers: self.samplers.len(),
            ..self.stats
        }
    }

    /// Views + sampler for one shader texture binding, given all fetch constants.
    pub fn bind(
        &mut self,
        device: &Device,
        memory: &dyn GuestMemory,
        fetch_constants: &[u32; FETCH_CONSTANT_DWORDS],
        binding: &TextureBinding,
    ) -> BoundTexture {
        self.bind_with(device, memory, fetch_constants, binding, None)
    }

    /// [`TextureCache::bind`] where part of guest memory may only be current in a GPU copy
    /// (`gpu`): textures entirely in such memory are loaded from it on the GPU.
    pub fn bind_with(
        &mut self,
        device: &Device,
        memory: &dyn GuestMemory,
        fetch_constants: &[u32; FETCH_CONSTANT_DWORDS],
        binding: &TextureBinding,
        gpu: Option<&mut dyn GpuSource>,
    ) -> BoundTexture {
        let slot = (binding.fetch_constant as usize & 31) * 6;
        let dwords: [u32; 6] = fetch_constants[slot..slot + 6].try_into().unwrap();
        let fetch = TextureFetchConstant::decode(dwords);
        let main_kind = match binding.dimension {
            TextureBindingDimension::D2 => ViewKind::D2,
            TextureBindingDimension::Cube => ViewKind::Cube,
            TextureBindingDimension::D3OrStacked => ViewKind::D3,
        };
        let wants_stacked = binding.stacked_texture_binding.is_some();

        let mut real: (Option<TextureView>, Option<TextureView>, bool) = (None, None, true);
        if let Some(key) = self.texture(device, memory, &fetch, dwords, gpu) {
            let entry = self.textures.get_mut(&key).expect("texture just resolved");
            let main = entry.view(main_kind);
            let stacked = if wants_stacked { entry.view(ViewKind::D2Array) } else { None };
            if main.is_none() && stacked.is_none() {
                let what = if entry.is_3d { "3D" } else { "2D" };
                let message = format!(
                    "{what} texture {}x{}x{} cannot be bound as {:?}",
                    entry.width, entry.height, entry.depth_or_array_layers, binding.dimension
                );
                self.warn_once(dwords, 1, &message);
            } else {
                real = (main, stacked, entry.filterable);
            }
        }
        let fallback = real.0.is_none() && real.1.is_none();
        if fallback {
            self.stats.fallbacks += 1;
        }
        let (main, stacked, filterable) = real;
        let view = main.unwrap_or_else(|| self.fallback.view(main_kind));
        let stacked_view = wants_stacked.then(|| stacked.unwrap_or_else(|| self.fallback.view(ViewKind::D2Array)));
        let filters = instruction_filters(binding);
        let sampler = self.sampler(device, &fetch, &filters, filterable);
        BoundTexture {
            view,
            stacked_view,
            sampler,
            sample_type: if filterable { TextureSampleType::Float } else { TextureSampleType::UnfilterableFloat },
            filterable,
            fallback,
        }
    }

    /// Guest physical memory `[addr, addr + len)` changed (e.g. a resolve wrote it): drops every
    /// texture reading from it. Bind groups holding its views keep the old contents. Returns the
    /// raw handles of the dropped textures' views (for callers remembering bindings).
    pub fn invalidate(&mut self, addr: u32, len: u32) -> Vec<usize> {
        let mut dropped_views = Vec::new();
        if len == 0 {
            return dropped_views;
        }
        let start = u64::from(addr & PHYSICAL_ADDRESS_MASK);
        let end = start + u64::from(len);
        let retired = &mut self.retired;
        let mut invalidated = 0;
        self.textures.retain(|_, e| {
            let keep = !e.ranges.iter().any(|r| {
                let (rs, re) = (u64::from(r.address), u64::from(r.address) + u64::from(r.length));
                rs < end && start < re
            });
            if !keep {
                retired.push(e.pooled());
                dropped_views.extend(e.views.iter().flatten().map(|v| v.raw() as usize));
                invalidated += 1;
            }
            keep
        });
        self.stats.invalidated += invalidated;
        dropped_views
    }

    /// Frame boundary: textures get revalidated on their next use, old ones are evicted.
    pub fn end_frame(&mut self) {
        let ended = self.frame;
        self.frame += 1;
        let max_unused = self.config.max_unused_frames;
        let before = self.textures.len();
        let retired = &mut self.retired;
        self.textures.retain(|_, e| {
            let keep = ended - e.last_used <= max_unused;
            if !keep {
                retired.push(e.pooled());
            }
            keep
        });
        let mut total: u64 = self.textures.values().map(|e| e.bytes).sum();
        if total > self.config.max_host_bytes {
            let mut candidates: Vec<(u64, TextureKey, u64)> = self
                .textures
                .iter()
                .filter(|(_, e)| e.last_used < ended)
                .map(|(k, e)| (e.last_used, *k, e.bytes))
                .collect();
            candidates.sort_unstable_by_key(|c| c.0);
            for (_, key, bytes) in candidates {
                if total <= self.config.max_host_bytes {
                    break;
                }
                if let Some(e) = self.textures.remove(&key) {
                    self.retired.push(e.pooled());
                }
                total -= bytes;
            }
        }
        self.stats.evicted += (before - self.textures.len()) as u64;
    }

    /// Frees the GPU memory of the textures dropped before the previous call (call once per
    /// frame, after submitting it): a texture dropped in frame N is destroyed at the end of frame
    /// N + 1, when nothing recorded uses it anymore and the presented front buffer (taken from
    /// this cache) has moved on. Browsers otherwise keep a released texture's memory until the
    /// JavaScript garbage collector gets to it, which in a worker can take long enough to run
    /// out of GPU memory.
    pub fn destroy_retired(&mut self) {
        for p in self.retiring.drain(..) {
            if self.config.reuse_textures {
                self.pool.give(p, self.frame);
            } else {
                p.texture.destroy();
            }
        }
        std::mem::swap(&mut self.retiring, &mut self.retired);
        self.pool.expire(self.frame.saturating_sub(POOL_MAX_FRAMES));
    }

    /// Drops every texture and sampler (e.g. after the device was lost and recreated, together
    /// with a new cache) and forgets failures.
    pub fn clear(&mut self) {
        self.textures.clear();
        self.retired.clear();
        self.retiring.clear();
        self.pool = TexturePool::default();
        self.samplers.clear();
        self.failed.clear();
        self.warned.clear();
    }

    fn warn_once(&mut self, dwords: [u32; 6], reason: u8, message: &str) {
        if self.warned.len() >= MAX_REMEMBERED_FAILURES {
            self.warned.clear();
        }
        if self.warned.insert((dwords, reason)) {
            log::warn!("texture fetch constant {dwords:08X?}: {message}; binding a fallback");
        }
    }

    /// Resolves the cached (revalidated) or newly loaded texture for a fetch constant.
    fn texture(
        &mut self,
        device: &Device,
        memory: &dyn GuestMemory,
        fetch: &TextureFetchConstant,
        dwords: [u32; 6],
        gpu: Option<&mut dyn GpuSource>,
    ) -> Option<TextureKey> {
        let valid_type = match fetch.fetch_type {
            FetchConstantType::Texture => true,
            FetchConstantType::InvalidTexture => self.config.allow_invalid_fetch_constants,
            _ => false,
        };
        if !valid_type {
            self.warn_once(dwords, 2, &format!("fetch constant type is {:?}", fetch.fetch_type));
            return None;
        }
        let Some(key) = TextureKey::from_fetch(fetch) else {
            self.warn_once(dwords, 3, "no base or mip address");
            return None;
        };
        let frame = self.frame;
        if let Some(entry) = self.textures.get_mut(&key) {
            let current = entry.validated == frame || {
                entry.validated = frame;
                let hash = hash_ranges(memory, &entry.ranges, &mut self.scratch, &mut self.stats.hashed_bytes);
                hash == entry.hash
            };
            if current {
                entry.last_used = frame;
                self.stats.hits += 1;
                return Some(key);
            }
            if let Some(e) = self.textures.remove(&key) {
                self.retired.push(e.pooled());
            }
            self.stats.reloads += 1;
        }
        if self.failed.contains(&key) {
            self.warn_once(dwords, 4, "texture failed to load before");
            return None;
        }
        match self.load(device, memory, fetch, gpu) {
            Ok(entry) => {
                self.stats.loads += 1;
                self.textures.insert(key, entry);
                Some(key)
            }
            Err(message) => {
                if self.failed.len() >= MAX_REMEMBERED_FAILURES {
                    self.failed.clear();
                }
                self.failed.insert(key);
                self.warn_once(dwords, 5, &message);
                None
            }
        }
    }

    fn load(
        &mut self,
        device: &Device,
        memory: &dyn GuestMemory,
        fetch: &TextureFetchConstant,
        gpu: Option<&mut dyn GpuSource>,
    ) -> Result<Entry, String> {
        let info = TextureGuestInfo::from_fetch(fetch);
        let sub = info.subresources;
        let host_dim = host_dimension(fetch, &info);
        let is_3d = host_dim == HostDimension::D3;
        let (width, height, layers) = (sub.width, sub.height, sub.depth_or_array_size.max(1));
        let limits = self.limits;
        let fits = if is_3d {
            width <= limits.max_3d && height <= limits.max_3d && layers <= limits.max_3d
        } else {
            width <= limits.max_2d && height <= limits.max_2d && layers <= limits.max_layers
        };
        if !fits {
            return Err(format!("{width}x{height}x{layers} {host_dim:?} exceeds the device limits"));
        }
        if host_dim == HostDimension::Cube && width != height {
            return Err(format!("cube map faces are not square ({width}x{height})"));
        }

        let mut caps = self.caps;
        if is_3d && !self.bc_sliced_3d {
            caps.texture_compression_bc = false;
        }
        // Hash before reading the data: a concurrent guest write in between is caught next frame.
        let ranges = info.memory_ranges();
        let started = std::time::Instant::now();
        let hash = hash_ranges(memory, &ranges, &mut self.scratch, &mut self.stats.hashed_bytes);
        let hashed = started.elapsed();
        // From the GPU copy of guest memory, like Xenia: required where only it is current
        // (resolve output, also when a texture only partly covers it), and otherwise cheaper
        // than uploading the texture's bytes, as textures are reloaded whole when any of their
        // memory changes while the copy is updated by changed page.
        if let Some(gpu) = gpu
            && (gpu.covers(&ranges) || (self.config.gpu_conversion && gpu.sync(&ranges)))
        {
            match self.load_on_gpu(device, &*gpu, fetch, &info, caps, ranges.clone(), hash) {
                Some(entry) => return Ok(entry),
                None => log::debug!("{}: not loadable from GPU memory; using guest memory", info.format.name()),
            }
        }
        if self.config.gpu_conversion
            && let Some(entry) = self.load_converting_on_gpu(device, memory, fetch, &info, caps, ranges.clone(), hash)
        {
            let total = started.elapsed();
            if total.as_millis() >= 50 {
                log::debug!(
                    "texture cache: {} {width}x{height}x{layers} took {total:?} on the GPU path",
                    info.format.name()
                );
            }
            return Ok(entry);
        }
        let loaded = load_texture(fetch, memory, &LoadOptions { caps, signed: false, row_alignment: 1 })
            .map_err(|e| e.to_string())?;
        let converted = started.elapsed();
        let host_format = loaded.mapping.host_format;
        if !format::is_float_sampleable(host_format) {
            return Err(format!("host format {} is not float-sampleable", host_format.webgpu_name()));
        }
        if let Some(why) = loaded.mapping.lossy {
            log::debug!("{} as {}: {why}", info.format.name(), host_format.webgpu_name());
        }
        self.stats.cpu_loads += 1;
        let kind = format!("{} as {} ({:?})", info.format.name(), host_format.webgpu_name(), loaded.mapping.conversion);
        if self.cpu_load_kinds.len() < 256 && self.cpu_load_kinds.insert(kind.clone()) {
            log::info!("texture cache: {kind} textures are converted on the CPU");
        }
        let longest = if is_3d { width.max(height).max(layers) } else { width.max(height) };
        let mip_level_count = loaded.mip_level_count.clamp(1, 32 - longest.leading_zeros());
        let pool_key =
            PoolKey { width, height, layers, mip_level_count, is_3d, format: format::webgpu_format(host_format) };
        let fills_all = (0..mip_level_count).all(|m| loaded.levels.iter().any(|l| l.level == m));
        let (texture, views) = self.new_texture(device, pool_key, fills_all);
        let mut bytes = 0u64;
        for level in loaded.levels.iter().filter(|l| l.level < mip_level_count) {
            device.queue().write_texture(
                &TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level.level,
                    origin: origin(0, 0, 0),
                    aspect: TextureAspect::All,
                },
                &level.data,
                level.bytes_per_row,
                Some(level.rows_per_image),
                extent(level.copy_width, level.copy_height, level.depth_or_array_layers),
            );
            bytes += level.data.len() as u64;
        }
        let total = started.elapsed();
        if total.as_millis() >= 50 {
            log::debug!(
                "texture cache: {} {width}x{height}x{layers} took {total:?} (hash {hashed:?}, convert {:?}, upload {:?})",
                info.format.name(),
                converted - hashed,
                total - converted
            );
        }
        Ok(Entry {
            texture,
            is_3d,
            width,
            height,
            depth_or_array_layers: layers,
            filterable: format::is_filterable(host_format, self.float32_filterable),
            views,
            pool_key,
            ranges,
            hash,
            bytes,
            last_used: self.frame,
            validated: self.frame,
        })
    }

    /// Loads a texture whose guest data is only current in GPU memory. `None` if the format's
    /// conversion or the data's placement isn't supported there.
    #[allow(clippy::too_many_arguments)]
    fn load_on_gpu(
        &mut self,
        device: &Device,
        gpu: &dyn GpuSource,
        fetch: &TextureFetchConstant,
        info: &TextureGuestInfo,
        caps: HostCaps,
        ranges: Vec<GuestRange>,
        hash: u64,
    ) -> Option<Entry> {
        self.gpu_converted(device, GpuData::Shared(gpu), fetch, info, caps, ranges, hash)
    }

    /// Loads a texture from guest memory, converting it on the GPU when its format allows.
    #[allow(clippy::too_many_arguments)]
    fn load_converting_on_gpu(
        &mut self,
        device: &Device,
        memory: &dyn GuestMemory,
        fetch: &TextureFetchConstant,
        info: &TextureGuestInfo,
        caps: HostCaps,
        ranges: Vec<GuestRange>,
        hash: u64,
    ) -> Option<Entry> {
        self.gpu_converted(device, GpuData::Memory(memory), fetch, info, caps, ranges, hash)
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_converted(
        &mut self,
        device: &Device,
        data: GpuData<'_>,
        fetch: &TextureFetchConstant,
        info: &TextureGuestInfo,
        caps: HostCaps,
        ranges: Vec<GuestRange>,
        hash: u64,
    ) -> Option<Entry> {
        let sub = info.subresources;
        let mapping = xenos_texture::host::host_mapping(info.format, false, caps, sub.width, sub.height).ok()?;
        let host_format = mapping.host_format;
        if !format::is_float_sampleable(host_format) {
            return None;
        }
        let is_3d = host_dimension(fetch, info) == HostDimension::D3;
        let (width, height, layers) = (sub.width, sub.height, sub.depth_or_array_size.max(1));
        let longest = if is_3d { width.max(height).max(layers) } else { width.max(height) };
        let mip_level_count = (info.layout.max_level + 1).clamp(1, 32 - longest.leading_zeros());
        let plan = gpu_load::plan(fetch, info, &mapping, mip_level_count)?;
        let pool_key =
            PoolKey { width, height, layers, mip_level_count, is_3d, format: format::webgpu_format(host_format) };
        let (texture, views) = self.new_texture(device, pool_key, plan.covers_levels(mip_level_count));
        match data {
            GpuData::Shared(gpu) => {
                if !self.gpu_loader.encode(device, gpu, &plan, &texture) {
                    return None;
                }
                self.stats.gpu_loads += 1;
            }
            GpuData::Memory(memory) => {
                self.gpu_loader.encode_from_memory(device, memory, &plan, &texture);
                self.stats.gpu_conversions += 1;
            }
        }
        Some(Entry {
            texture,
            is_3d,
            width,
            height,
            depth_or_array_layers: layers,
            filterable: format::is_filterable(host_format, self.float32_filterable),
            views,
            pool_key,
            ranges,
            hash,
            bytes: plan.host_bytes(),
            last_used: self.frame,
            validated: self.frame,
        })
    }

    /// A texture for `key`: a pooled one (with its views) or a new one. Pooled textures hold
    /// their previous contents, so they're only used when the load writes every level
    /// (`fills_all`); a new texture starts out zeroed.
    fn new_texture(&mut self, device: &Device, key: PoolKey, fills_all: bool) -> (Texture, [Option<TextureView>; 4]) {
        if fills_all
            && self.config.reuse_textures
            && let Some(p) = self.pool.take(&key)
        {
            self.stats.reused += 1;
            return (p.texture, p.views);
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: "xenos texture",
            size: extent(key.width, key.height, key.layers),
            mip_level_count: key.mip_level_count,
            dimension: if key.is_3d { TextureDimension::D3 } else { TextureDimension::D2 },
            format: key.format,
            usage: TextureUsage::TextureBinding | TextureUsage::CopyDst | TextureUsage::CopySrc,
            ..Default::default()
        });
        (texture, [None, None, None, None])
    }

    fn sampler(
        &mut self,
        device: &Device,
        fetch: &TextureFetchConstant,
        filters: &FetchInstructionFilters,
        filterable: bool,
    ) -> Sampler {
        let desc = sampler_descriptor(fetch, filters);
        let filter = |f: xenos_texture::FilterMode| match f {
            xenos_texture::FilterMode::Nearest => FilterMode::Nearest,
            xenos_texture::FilterMode::Linear => FilterMode::Linear,
        };
        let address = |a: xenos_texture::AddressMode| match a {
            xenos_texture::AddressMode::ClampToEdge => AddressMode::ClampToEdge,
            xenos_texture::AddressMode::Repeat => AddressMode::Repeat,
            xenos_texture::AddressMode::MirrorRepeat => AddressMode::MirrorRepeat,
        };
        let mut key = SamplerKey {
            address: [address(desc.address_mode_u), address(desc.address_mode_v), address(desc.address_mode_w)],
            mag: filter(desc.mag_filter),
            min: filter(desc.min_filter),
            mip: match desc.mipmap_filter {
                xenos_texture::FilterMode::Nearest => MipmapFilterMode::Nearest,
                xenos_texture::FilterMode::Linear => MipmapFilterMode::Linear,
            },
            lod_min_bits: 0,
            lod_max_bits: 0,
            anisotropy: desc.max_anisotropy.clamp(1, MAX_ANISOTROPY),
        };
        if !filterable {
            key.mag = FilterMode::Nearest;
            key.min = FilterMode::Nearest;
            key.mip = MipmapFilterMode::Nearest;
        }
        let all_linear =
            key.mag == FilterMode::Linear && key.min == FilterMode::Linear && key.mip == MipmapFilterMode::Linear;
        if !all_linear {
            key.anisotropy = 1;
        }
        let lod_min = if desc.lod_min_clamp.is_finite() { desc.lod_min_clamp.clamp(0.0, 32.0) } else { 0.0 };
        let lod_max = if desc.lod_max_clamp.is_finite() { desc.lod_max_clamp.clamp(lod_min, 32.0) } else { 32.0 };
        key.lod_min_bits = lod_min.to_bits();
        key.lod_max_bits = lod_max.to_bits();
        self.samplers
            .entry(key)
            .or_insert_with(|| {
                if !desc.losses.is_empty() {
                    log::debug!("sampler approximations: {:?}", desc.losses);
                }
                device.create_sampler(&SamplerDescriptor {
                    label: "xenos sampler",
                    address_mode_u: key.address[0],
                    address_mode_v: key.address[1],
                    address_mode_w: key.address[2],
                    mag_filter: key.mag,
                    min_filter: key.min,
                    mipmap_filter: key.mip,
                    lod_min_clamp: lod_min,
                    lod_max_clamp: lod_max,
                    compare: None,
                    max_anisotropy: key.anisotropy,
                })
            })
            .clone()
    }
}

/// Where a GPU-converted texture's guest bytes come from.
enum GpuData<'a> {
    /// A GPU copy of guest memory holding the only current data.
    Shared(&'a dyn GpuSource),
    /// Guest memory (copied raw to the GPU).
    Memory(&'a dyn GuestMemory),
}

/// Identifies the current contents of `ranges`: their write versions if `memory` tracks writes,
/// else a hash of the bytes.
fn hash_ranges(memory: &dyn GuestMemory, ranges: &[GuestRange], scratch: &mut Vec<u8>, hashed: &mut u64) -> u64 {
    let versions: Option<Vec<u64>> = ranges.iter().map(|r| memory.version(r.address, r.length)).collect();
    if let Some(versions) = versions {
        return versions
            .iter()
            .zip(ranges)
            .fold(0u64, |h, (v, r)| hash::hash(h ^ u64::from(r.address), &v.to_le_bytes()));
    }
    let mut h = 0u64;
    for r in ranges {
        scratch.clear();
        scratch.resize(r.length as usize, 0);
        memory.read(r.address, scratch);
        h = hash::hash(h ^ u64::from(r.address), scratch);
        *hashed += u64::from(r.length);
    }
    h
}

/// The fetch instruction's filter overrides in xenos-texture terms.
pub fn instruction_filters(binding: &TextureBinding) -> FetchInstructionFilters {
    let filter = |f: parsed::TextureFilter| match f {
        parsed::TextureFilter::Point => TextureFilter::Point,
        parsed::TextureFilter::Linear => TextureFilter::Linear,
        parsed::TextureFilter::BaseMap => TextureFilter::BaseMap,
        parsed::TextureFilter::UseFetchConst => TextureFilter::UseFetchConst,
    };
    FetchInstructionFilters {
        mag_filter: filter(binding.mag_filter),
        min_filter: filter(binding.min_filter),
        mip_filter: filter(binding.mip_filter),
        aniso_filter: match binding.aniso_filter {
            parsed::AnisoFilter::Disabled => AnisoFilter::Disabled,
            parsed::AnisoFilter::Max1To1 => AnisoFilter::Max1to1,
            parsed::AnisoFilter::Max2To1 => AnisoFilter::Max2to1,
            parsed::AnisoFilter::Max4To1 => AnisoFilter::Max4to1,
            parsed::AnisoFilter::Max8To1 => AnisoFilter::Max8to1,
            parsed::AnisoFilter::Max16To1 => AnisoFilter::Max16to1,
            parsed::AnisoFilter::Invalid6 => AnisoFilter::Reserved6,
            parsed::AnisoFilter::UseFetchConst => AnisoFilter::UseFetchConst,
        },
    }
}
