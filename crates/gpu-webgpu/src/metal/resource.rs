//! Buffers, textures, views, samplers and bind groups.

use std::cell::{Cell, RefCell};
use std::ptr;
use std::rc::Rc;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use super::{Device, Id, MtlBuffer, MtlTexture, Shared, conv, set_label};
use crate::pending::{Completer, Pending};
use crate::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferDescriptor, BufferUsage, Error,
    ErrorType, MapMode, Result, SamplerDescriptor, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsage, TextureViewDescriptor, TextureViewDimension,
};

/// Implements identity (`raw`, `PartialEq`, `Debug`) for an `Rc`-backed handle.
macro_rules! rc_handle {
    ($name:ident) => {
        impl $name {
            /// A stable identity for the object (not a Metal pointer).
            pub fn raw(&self) -> *const ::std::ffi::c_void {
                ::std::rc::Rc::as_ptr(&self.inner).cast()
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                ::std::rc::Rc::ptr_eq(&self.inner, &other.inner)
            }
        }

        impl Eq for $name {}

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({:p})", stringify!($name), ::std::rc::Rc::as_ptr(&self.inner))
            }
        }
    };
}
pub(crate) use rc_handle;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Map {
    Unmapped,
    Pending,
    Mapped { write: bool, start: u64, end: u64 },
}

pub(crate) struct BufferInner {
    /// `None` once destroyed, or when allocation failed (reported as out of memory).
    raw: RefCell<Option<MtlBuffer>>,
    label: Rc<str>,
    size: u64,
    usage: BufferUsage,
    map: Cell<Map>,
    /// Bumped per map request so a stale completion (after an unmap) cannot clobber a newer one.
    generation: Cell<u64>,
    shared: Rc<Shared>,
}

/// A buffer in shared (CPU-visible) memory.
#[derive(Clone)]
pub struct Buffer {
    pub(crate) inner: Rc<BufferInner>,
}

rc_handle!(Buffer);

impl Device {
    pub fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Buffer {
        crate::counters::add(&crate::counters::count::buffers, 1);
        // Metal rejects empty buffers; WebGPU allows them.
        let length = desc.size.max(4).next_multiple_of(4) as usize;
        // WebGPU allows MAP_READ only with COPY_DST and MAP_WRITE only with COPY_SRC.
        let map_read_extra = BufferUsage(desc.usage.0 & !(BufferUsage::MapRead.0 | BufferUsage::CopyDst.0));
        let map_write_extra = BufferUsage(desc.usage.0 & !(BufferUsage::MapWrite.0 | BufferUsage::CopySrc.0));
        if (desc.usage.contains(BufferUsage::MapRead) && map_read_extra != BufferUsage::None)
            || (desc.usage.contains(BufferUsage::MapWrite) && map_write_extra != BufferUsage::None)
        {
            self.shared.report(
                ErrorType::Validation,
                format!("buffer {:?}: usage {:?} combines mapping with other uses", desc.label, desc.usage),
            );
        }
        let raw = self.shared.raw.newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared);
        match &raw {
            Some(b) => {
                set_label!(b, desc.label);
                self.shared.zero_buffer(b, desc.mapped_at_creation);
            }
            None => self.shared.report(
                ErrorType::OutOfMemory,
                format!("cannot allocate buffer {:?} ({} bytes)", desc.label, desc.size),
            ),
        }
        let buffer = Buffer {
            inner: Rc::new(BufferInner {
                raw: RefCell::new(raw),
                label: desc.label.into(),
                size: desc.size,
                usage: desc.usage,
                map: Cell::new(Map::Unmapped),
                generation: Cell::new(0),
                shared: self.shared.clone(),
            }),
        };
        crate::memory::track(buffer.raw() as usize, "buffer", desc.label, desc.size);
        if desc.mapped_at_creation && buffer.mtl().is_some() {
            buffer.inner.map.set(Map::Mapped { write: true, start: 0, end: desc.size });
        }
        buffer
    }

    /// A buffer initialised with `data` (zero-padded to a multiple of 4 bytes).
    pub fn create_buffer_init(&self, label: &str, usage: BufferUsage, data: &[u8]) -> Buffer {
        let size = (data.len() as u64).div_ceil(4).max(1) * 4;
        let buffer = self.create_buffer(&BufferDescriptor { label, size, usage, mapped_at_creation: true });
        let mut padded = data.to_vec();
        padded.resize(size as usize, 0);
        // Only fails when allocation failed, which was already reported.
        let _ = buffer.write_mapped(0, &padded);
        buffer.unmap();
        buffer
    }
}

impl Buffer {
    pub(crate) fn mtl(&self) -> Option<MtlBuffer> {
        self.inner.raw.borrow().clone()
    }

    pub fn size(&self) -> u64 {
        self.inner.size
    }

    pub fn label(&self) -> &str {
        &self.inner.label
    }

    pub fn usage(&self) -> BufferUsage {
        self.inner.usage
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.inner.map.get(), Map::Mapped { .. })
    }

    /// Maps `offset..offset + size` (`size: None` = to the end; offset a multiple of 8, size of 4).
    /// Resolves once the GPU has finished all work submitted so far (which covers every use of
    /// the buffer); the memory is shared, so mapping only waits.
    pub fn map_async(&self, mode: MapMode, offset: u64, size: Option<u64>) -> Pending<Result<()>> {
        let inner = &self.inner;
        let write = mode.contains(MapMode::Write);
        let size = size.unwrap_or_else(|| inner.size.saturating_sub(offset));
        let needed = if write { BufferUsage::MapWrite } else { BufferUsage::MapRead };
        let problem = if inner.map.get() != Map::Unmapped {
            Some("buffer is already mapped or being mapped".to_string())
        } else if mode != MapMode::Read && mode != MapMode::Write {
            Some(format!("map mode must be Read or Write, got {mode:?}"))
        } else if !inner.usage.contains(needed) {
            Some(format!("buffer usage {:?} lacks {needed:?}", inner.usage))
        } else if !offset.is_multiple_of(8)
            || !size.is_multiple_of(4)
            || offset.checked_add(size).is_none_or(|e| e > inner.size)
        {
            Some(format!("range {offset}+{size} is misaligned or exceeds {} bytes", inner.size))
        } else if self.mtl().is_none() {
            Some("buffer is destroyed or was never allocated".to_string())
        } else {
            None
        };
        if let Some(problem) = problem {
            return Pending::ready(Err(Error::Map(problem)));
        }
        let generation = inner.generation.get() + 1;
        inner.generation.set(generation);
        inner.map.set(Map::Pending);
        let (pending, done): (_, Completer<Result<()>>) = Pending::pair();
        let state = inner.clone();
        inner.shared.after_submitted(Box::new(move || {
            if state.generation.get() == generation && state.map.get() == Map::Pending {
                state.map.set(Map::Mapped { write, start: offset, end: offset + size });
                done.complete(Ok(()));
            } else {
                done.complete(Err(Error::Map("unmapped before the mapping completed".into())));
            }
        }));
        pending
    }

    fn mapped_ptr(&self, offset: u64, len: usize, need_write: bool) -> Result<*mut u8> {
        match self.inner.map.get() {
            Map::Mapped { write, start, end } if (write || !need_write) && offset >= start && offset + len as u64 <= end => {
                let raw = self.mtl().ok_or_else(|| Error::Map("buffer is destroyed".into()))?;
                Ok(unsafe { raw.contents().as_ptr().cast::<u8>().add(offset as usize) })
            }
            Map::Mapped { write: false, .. } if need_write => Err(Error::Map("buffer is mapped for reading".into())),
            Map::Mapped { start, end, .. } => {
                Err(Error::Map(format!("range {offset}+{len} is outside the mapped range {start}..{end}")))
            }
            _ => Err(Error::Map("buffer is not mapped".into())),
        }
    }

    /// Copies out of the mapped range (any alignment).
    pub fn read_mapped(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let src = self.mapped_ptr(offset, dst.len(), false)?;
        unsafe { ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    /// Copies into a range mapped for writing (or mapped at creation).
    pub fn write_mapped(&self, offset: u64, src: &[u8]) -> Result<()> {
        let dst = self.mapped_ptr(offset, src.len(), true)?;
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        Ok(())
    }

    /// Copies `src` straight into the buffer's memory at `offset`, without mapping and without
    /// ordering against the GPU: Metal buffers here live in memory shared with the CPU. The
    /// caller makes sure no submitted work still uses the range, and writes before submitting
    /// work that reads it. `false` if the range is out of bounds or the buffer is destroyed.
    pub fn write_unsynchronized(&self, offset: u64, src: &[u8]) -> bool {
        let Some(raw) = self.mtl() else { return false };
        if offset.checked_add(src.len() as u64).is_none_or(|end| end > raw.length() as u64) {
            return false;
        }
        // SAFETY: in bounds (checked above) of a CPU-visible buffer; ordering against the GPU is
        // the caller's (see above).
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), raw.contents().as_ptr().cast::<u8>().add(offset as usize), src.len()) };
        true
    }

    /// Unmaps (cancelling a pending map, which then reports an error).
    pub fn unmap(&self) {
        self.inner.generation.set(self.inner.generation.get() + 1);
        self.inner.map.set(Map::Unmapped);
    }

    /// Releases the memory once the GPU no longer uses it (command buffers keep their own
    /// references until they complete).
    pub fn destroy(&self) {
        self.unmap();
        crate::memory::untrack(self.raw() as usize);
        self.inner.raw.borrow_mut().take();
    }
}

/// What zero-initialisation needs to know about a new texture.
pub(crate) struct TextureInfo {
    pub(crate) format: TextureFormat,
    pub(crate) dimension: TextureDimension,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth_or_array_layers: u32,
    pub(crate) mip_level_count: u32,
    /// Cleared with render passes rather than copies.
    pub(crate) render: bool,
}

pub(crate) struct TextureInner {
    raw: RefCell<Option<MtlTexture>>,
    pub(crate) format: TextureFormat,
    pub(crate) dimension: TextureDimension,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth_or_array_layers: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) sample_count: u32,
    pub(crate) usage: TextureUsage,
    /// Surface textures are not tracked as allocations.
    tracked: bool,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        if self.tracked {
            crate::memory::freed(self as *const TextureInner as usize);
        }
    }
}

#[derive(Clone)]
pub struct Texture {
    pub(crate) inner: Rc<TextureInner>,
}

rc_handle!(Texture);

impl Device {
    pub fn create_texture(&self, desc: &TextureDescriptor<'_>) -> Texture {
        crate::counters::add(&crate::counters::count::textures, 1);
        crate::counters::created("texture", desc.label);
        let s = desc.size;
        let raw = match conv::pixel_format(desc.format) {
            None => {
                self.shared.report(ErrorType::Validation, format!("texture format {:?} is not supported", desc.format));
                None
            }
            Some(pixel_format) => {
                let d = MTLTextureDescriptor::new();
                let layers = s.depthOrArrayLayers.max(1);
                let ty = match desc.dimension {
                    TextureDimension::D1 => MTLTextureType::Type1D,
                    TextureDimension::D3 => MTLTextureType::Type3D,
                    _ if desc.sample_count > 1 && layers > 1 => MTLTextureType::Type2DMultisampleArray,
                    _ if desc.sample_count > 1 => MTLTextureType::Type2DMultisample,
                    _ if layers > 1 => MTLTextureType::Type2DArray,
                    _ => MTLTextureType::Type2D,
                };
                unsafe {
                    d.setTextureType(ty);
                    d.setPixelFormat(pixel_format);
                    d.setWidth(s.width.max(1) as usize);
                    d.setHeight(s.height.max(1) as usize);
                    if desc.dimension == TextureDimension::D3 {
                        d.setDepth(layers as usize);
                    } else {
                        d.setArrayLength(layers as usize);
                    }
                    d.setMipmapLevelCount(desc.mip_level_count.max(1) as usize);
                    d.setSampleCount(desc.sample_count.max(1) as usize);
                }
                let mut usage = MTLTextureUsage::Unknown;
                if desc.usage.contains(TextureUsage::TextureBinding) {
                    usage |= MTLTextureUsage::ShaderRead;
                }
                if desc.usage.contains(TextureUsage::StorageBinding) {
                    usage |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
                }
                if desc.usage.contains(TextureUsage::RenderAttachment) {
                    usage |= MTLTextureUsage::RenderTarget;
                }
                // Stencil views of combined formats and sRGB/linear reinterpretation.
                if (conv::has_depth(desc.format) && conv::has_stencil(desc.format)) || !desc.view_formats.is_empty() {
                    usage |= MTLTextureUsage::PixelFormatView;
                }
                d.setUsage(usage);
                d.setStorageMode(MTLStorageMode::Private);
                let raw = self.shared.raw.newTextureWithDescriptor(&d);
                match &raw {
                    Some(t) => {
                        set_label!(t, desc.label);
                        self.shared.zero_texture(
                            t,
                            &TextureInfo {
                                format: desc.format,
                                dimension: desc.dimension,
                                width: s.width.max(1),
                                height: s.height.max(1),
                                depth_or_array_layers: layers,
                                mip_level_count: desc.mip_level_count,
                                render: desc.usage.contains(TextureUsage::RenderAttachment) || desc.sample_count > 1,
                            },
                        );
                    }
                    None => self.shared.report(
                        ErrorType::OutOfMemory,
                        format!("cannot create texture {:?} ({}x{}x{} {:?})", desc.label, s.width, s.height, layers, desc.format),
                    ),
                }
                raw
            }
        };
        let texture = Texture {
            inner: Rc::new(TextureInner {
                raw: RefCell::new(raw),
                format: desc.format,
                dimension: desc.dimension,
                width: s.width,
                height: s.height,
                depth_or_array_layers: s.depthOrArrayLayers,
                mip_level_count: desc.mip_level_count,
                sample_count: desc.sample_count,
                usage: desc.usage,
                tracked: true,
            }),
        };
        let bytes = crate::memory::texture_bytes(
            desc.format,
            s.width,
            s.height,
            s.depthOrArrayLayers,
            desc.mip_level_count,
            desc.sample_count,
        );
        crate::memory::track(texture.raw() as usize, "texture", desc.label, bytes);
        texture
    }
}

impl Texture {
    /// Wraps a drawable's texture.
    pub(crate) fn from_surface(raw: MtlTexture, format: TextureFormat, usage: TextureUsage) -> Self {
        let (width, height) = (raw.width() as u32, raw.height() as u32);
        Texture {
            inner: Rc::new(TextureInner {
                raw: RefCell::new(Some(raw)),
                format,
                dimension: TextureDimension::D2,
                width,
                height,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage,
                tracked: false,
            }),
        }
    }

    pub(crate) fn mtl(&self) -> Option<MtlTexture> {
        self.inner.raw.borrow().clone()
    }

    pub fn create_view(&self, desc: Option<&TextureViewDescriptor<'_>>) -> TextureView {
        crate::counters::add(&crate::counters::count::views, 1);
        let t = &self.inner;
        let d = desc.copied().unwrap_or_default();
        let format = if d.format == TextureFormat::Undefined { t.format } else { d.format };
        let layers = if t.dimension == TextureDimension::D3 { 1 } else { t.depth_or_array_layers.max(1) };
        let dimension = match d.dimension {
            TextureViewDimension::Undefined => match t.dimension {
                TextureDimension::D1 => TextureViewDimension::D1,
                TextureDimension::D3 => TextureViewDimension::D3,
                _ if layers > 1 => TextureViewDimension::D2Array,
                _ => TextureViewDimension::D2,
            },
            other => other,
        };
        let levels = d.mip_level_count.unwrap_or(t.mip_level_count.saturating_sub(d.base_mip_level)).max(1);
        let slices = match dimension {
            TextureViewDimension::D2 | TextureViewDimension::D1 | TextureViewDimension::D3 => 1,
            TextureViewDimension::Cube => 6,
            _ => d.array_layer_count.unwrap_or(layers.saturating_sub(d.base_array_layer)).max(1),
        };
        let slices = d.array_layer_count.unwrap_or(slices);
        let base = self.mtl();
        let raw = base.as_ref().and_then(|base| {
            let pixel_format = match d.aspect {
                TextureAspect::StencilOnly if conv::has_depth(t.format) => MTLPixelFormat::X32_Stencil8,
                _ => conv::pixel_format(format)?,
            };
            let ty = conv::view_type(dimension, t.sample_count > 1);
            let whole = pixel_format == base.pixelFormat()
                && ty == base.textureType()
                && d.base_mip_level == 0
                && levels as usize == base.mipmapLevelCount()
                && d.base_array_layer == 0
                && slices as usize == base.arrayLength();
            if whole {
                return Some(base.clone());
            }
            let view = unsafe {
                base.newTextureViewWithPixelFormat_textureType_levels_slices(
                    pixel_format,
                    ty,
                    objc2_foundation::NSRange::new(d.base_mip_level as usize, levels as usize),
                    objc2_foundation::NSRange::new(d.base_array_layer as usize, slices as usize),
                )
            };
            if let Some(v) = &view {
                set_label!(v, d.label);
            }
            view
        });
        TextureView {
            inner: Rc::new(ViewInner {
                raw,
                base,
                base_level: d.base_mip_level,
                base_layer: d.base_array_layer,
                format,
                texture: self.inner.clone(),
            }),
        }
    }

    pub fn width(&self) -> u32 {
        self.inner.width
    }

    pub fn height(&self) -> u32 {
        self.inner.height
    }

    pub fn depth_or_array_layers(&self) -> u32 {
        self.inner.depth_or_array_layers
    }

    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }

    pub fn sample_count(&self) -> u32 {
        self.inner.sample_count
    }

    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }

    pub fn usage(&self) -> TextureUsage {
        self.inner.usage
    }

    /// Drops this handle's Metal texture; views and running command buffers keep theirs.
    pub fn destroy(&self) {
        if self.inner.tracked {
            // Views share the inner object (and their own Metal textures hold the memory).
            crate::memory::destroyed(self.raw() as usize, Rc::as_ptr(&self.inner) as usize);
        }
        self.inner.raw.borrow_mut().take();
    }
}

pub(crate) struct ViewInner {
    /// The view's own Metal texture (the base texture itself when the view covers all of it).
    pub(crate) raw: Option<MtlTexture>,
    /// For attachments, which take the base texture plus a level and slice.
    pub(crate) base: Option<MtlTexture>,
    pub(crate) base_level: u32,
    pub(crate) base_layer: u32,
    pub(crate) format: TextureFormat,
    pub(crate) texture: Rc<TextureInner>,
}

#[derive(Clone)]
pub struct TextureView {
    pub(crate) inner: Rc<ViewInner>,
}

rc_handle!(TextureView);

pub(crate) struct SamplerInner {
    pub(crate) raw: Option<Id<dyn MTLSamplerState>>,
}

#[derive(Clone)]
pub struct Sampler {
    pub(crate) inner: Rc<SamplerInner>,
}

rc_handle!(Sampler);

impl Device {
    pub fn create_sampler(&self, desc: &SamplerDescriptor<'_>) -> Sampler {
        crate::counters::add(&crate::counters::count::samplers, 1);
        let d = MTLSamplerDescriptor::new();
        d.setSAddressMode(conv::address_mode(desc.address_mode_u));
        d.setTAddressMode(conv::address_mode(desc.address_mode_v));
        d.setRAddressMode(conv::address_mode(desc.address_mode_w));
        d.setMagFilter(conv::min_mag_filter(desc.mag_filter));
        d.setMinFilter(conv::min_mag_filter(desc.min_filter));
        d.setMipFilter(conv::mip_filter(desc.mipmap_filter));
        d.setLodMinClamp(desc.lod_min_clamp);
        d.setLodMaxClamp(desc.lod_max_clamp);
        d.setMaxAnisotropy(desc.max_anisotropy.clamp(1, 16) as usize);
        if let Some(c) = desc.compare {
            d.setCompareFunction(conv::compare(c));
        }
        set_label!(d, desc.label);
        let raw = self.shared.raw.newSamplerStateWithDescriptor(&d);
        if raw.is_none() {
            self.shared.report(ErrorType::Internal, format!("cannot create sampler {:?}", desc.label));
        }
        Sampler { inner: Rc::new(SamplerInner { raw }) }
    }
}

pub(crate) struct LayoutInner {
    /// Sorted by binding.
    pub(crate) entries: Vec<BindGroupLayoutEntry>,
}

#[derive(Clone)]
pub struct BindGroupLayout {
    pub(crate) inner: Rc<LayoutInner>,
}

rc_handle!(BindGroupLayout);

/// What a bind group entry resolves to on the Metal side.
#[derive(Clone)]
pub(crate) enum Resource {
    Buffer {
        raw: Option<MtlBuffer>,
        offset: u64,
        /// Bytes visible through the binding (for runtime-sized arrays' lengths).
        size: u64,
        /// Index into the dynamic offsets passed to `set_bind_group`.
        dynamic: Option<u8>,
    },
    Texture(Option<MtlTexture>),
    Sampler(Option<Id<dyn MTLSamplerState>>),
}

pub(crate) struct GroupInner {
    /// Sorted by binding.
    pub(crate) resources: Vec<(u32, Resource)>,
    /// Keeps the bound handles' identities alive (callers key caches by them).
    _keep: Vec<KeepAlive>,
}

impl GroupInner {
    pub(crate) fn get(&self, binding: u32) -> Option<&Resource> {
        self.resources.binary_search_by_key(&binding, |(b, _)| *b).ok().map(|i| &self.resources[i].1)
    }
}

#[allow(dead_code)]
enum KeepAlive {
    Buffer(Buffer),
    View(TextureView),
    Sampler(Sampler),
}

#[derive(Clone)]
pub struct BindGroup {
    pub(crate) inner: Rc<GroupInner>,
}

rc_handle!(BindGroup);

pub(crate) struct PipelineLayoutInner {
    /// Metal slots come from the shaders (see `shader`); the layouts are only kept alive.
    _groups: Vec<BindGroupLayout>,
}

#[derive(Clone)]
pub struct PipelineLayout {
    pub(crate) inner: Rc<PipelineLayoutInner>,
}

rc_handle!(PipelineLayout);

impl Device {
    pub fn create_bind_group_layout(&self, _label: &str, entries: &[BindGroupLayoutEntry]) -> BindGroupLayout {
        let mut entries = entries.to_vec();
        entries.sort_by_key(|e| e.binding);
        BindGroupLayout { inner: Rc::new(LayoutInner { entries }) }
    }

    pub fn create_bind_group(&self, label: &str, layout: &BindGroupLayout, entries: &[BindGroupEntry<'_>]) -> BindGroup {
        crate::counters::add(&crate::counters::count::bind_groups, 1);
        crate::counters::created("bind group", label);
        // WebGPU orders dynamic offsets by binding number.
        let mut dynamic = 0u8;
        let dynamic_index: Vec<(u32, u8)> = layout
            .inner
            .entries
            .iter()
            .filter(|e| matches!(e.ty, BindingType::Buffer { has_dynamic_offset: true, .. }))
            .map(|e| {
                dynamic += 1;
                (e.binding, dynamic - 1)
            })
            .collect();
        let mut keep = Vec::with_capacity(entries.len());
        let mut resources: Vec<(u32, Resource)> = entries
            .iter()
            .map(|e| {
                let r = match e.resource {
                    BindingResource::Buffer { buffer, offset, size } => {
                        keep.push(KeepAlive::Buffer(buffer.clone()));
                        Resource::Buffer {
                            raw: buffer.mtl(),
                            offset,
                            size: size.unwrap_or(buffer.size().saturating_sub(offset)),
                            dynamic: dynamic_index.iter().find(|(b, _)| *b == e.binding).map(|(_, i)| *i),
                        }
                    }
                    BindingResource::TextureView(v) => {
                        keep.push(KeepAlive::View(v.clone()));
                        Resource::Texture(v.inner.raw.clone())
                    }
                    BindingResource::Sampler(s) => {
                        keep.push(KeepAlive::Sampler(s.clone()));
                        Resource::Sampler(s.inner.raw.clone())
                    }
                };
                (e.binding, r)
            })
            .collect();
        resources.sort_by_key(|(b, _)| *b);
        if let Some(missing) =
            layout.inner.entries.iter().find(|l| resources.binary_search_by_key(&l.binding, |(b, _)| *b).is_err())
        {
            self.shared.report(
                ErrorType::Validation,
                format!("bind group {label:?} lacks binding {} of its layout", missing.binding),
            );
        }
        BindGroup { inner: Rc::new(GroupInner { resources, _keep: keep }) }
    }

    pub fn create_pipeline_layout(&self, _label: &str, layouts: &[&BindGroupLayout]) -> PipelineLayout {
        PipelineLayout { inner: Rc::new(PipelineLayoutInner { _groups: layouts.iter().map(|l| (*l).clone()).collect() }) }
    }
}

/// The Metal texture a view renders into: base texture, level, slice.
pub(crate) fn attachment(view: &TextureView) -> Option<(&ProtocolObject<dyn MTLTexture>, usize, usize)> {
    let v = &view.inner;
    v.base.as_deref().map(|b| (b, v.base_level as usize, v.base_layer as usize))
}

impl TextureView {
    pub(crate) fn format(&self) -> TextureFormat {
        self.inner.format
    }

    pub(crate) fn sample_count(&self) -> u32 {
        self.inner.texture.sample_count
    }

    /// The attachment size at the view's level.
    pub(crate) fn extent(&self) -> (u32, u32) {
        let t = &self.inner.texture;
        ((t.width >> self.inner.base_level).max(1), (t.height >> self.inner.base_level).max(1))
    }
}
