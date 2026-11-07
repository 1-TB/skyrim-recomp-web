//! Command encoders and passes.
//!
//! A `CommandEncoder` owns one Metal command buffer; copies go into a blit encoder opened on
//! demand, passes into their own encoders. Bind groups are applied lazily before each draw or
//! dispatch, through the pipeline's slot map (see `shader`), skipping slots already holding the
//! same object.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use super::resource::{GroupInner, Resource, TextureInner, attachment};
use super::shader::{ComputeState, RenderState, StageBindings};
use super::{Device, Id, MtlBuffer, MtlCommandBuffer, conv, ns, set_label};
use crate::{
    BindGroup, Buffer, Color, ComputePipeline, Extent3d, IndexFormat, Origin3d, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPipeline, TexelCopyBufferInfo, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDimension,
};

/// A finished command buffer, committed by `Queue::submit`.
pub struct CommandBuffer {
    pub(crate) raw: MtlCommandBuffer,
}

impl CommandBuffer {
    pub fn raw(&self) -> *const c_void {
        Retained::as_ptr(&self.raw).cast()
    }
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CommandBuffer({:p})", Retained::as_ptr(&self.raw))
    }
}

pub struct CommandEncoder {
    pub(crate) raw: MtlCommandBuffer,
    blit: Option<Id<dyn MTLBlitCommandEncoder>>,
    /// The compute encoder of the last compute pass, left open so that the next compute pass
    /// continues it: on Apple GPUs every encoder boundary leaves the GPU idle for tens of
    /// microseconds, and the renderer records many small compute passes in a row. Dispatches
    /// in one (serial) compute encoder still run in order with their writes visible.
    compute: Option<Id<dyn MTLComputeCommandEncoder>>,
    device: Device,
}

impl std::fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CommandEncoder({:p})", Retained::as_ptr(&self.raw))
    }
}

impl Device {
    pub fn create_command_encoder(&self, label: &str) -> CommandEncoder {
        let raw = self.shared.new_command_buffer();
        set_label!(raw, label);
        CommandEncoder { raw, blit: None, compute: None, device: self.clone() }
    }
}

/// The part of a mip level a copy may touch: Metal rejects copies past the level's edge, which
/// WebGPU allows for compressed formats (sizes rounded up to whole blocks).
fn clamp_copy(texture: &TextureInner, level: u32, origin: Origin3d, size: Extent3d) -> Extent3d {
    let w = (texture.width >> level).max(1);
    let h = (texture.height >> level).max(1);
    let d = if texture.dimension == TextureDimension::D3 { (texture.depth_or_array_layers >> level).max(1) } else { u32::MAX };
    crate::extent(
        size.width.min(w.saturating_sub(origin.x)),
        size.height.min(h.saturating_sub(origin.y)),
        size.depthOrArrayLayers.min(d.saturating_sub(origin.z)),
    )
}

fn blit_option(texture: &TextureInner, aspect: TextureAspect) -> MTLBlitOption {
    let combined = conv::has_depth(texture.format) && conv::has_stencil(texture.format);
    match aspect {
        TextureAspect::DepthOnly if combined => MTLBlitOption::DepthFromDepthStencil,
        TextureAspect::StencilOnly if combined => MTLBlitOption::StencilFromDepthStencil,
        _ => MTLBlitOption::None,
    }
}

/// Bytes per row of blocks and rows of blocks per image of a buffer layout.
fn buffer_strides(texture: &TextureInner, bytes_per_row: u32, rows_per_image: Option<u32>, size: Extent3d) -> (usize, usize) {
    let (_, bh) = conv::block_dim(texture.format);
    let rows = rows_per_image.unwrap_or(size.height).div_ceil(bh);
    (bytes_per_row as usize, bytes_per_row as usize * rows as usize)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_buffer_to_texture(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    src: &ProtocolObject<dyn MTLBuffer>,
    offset: u64,
    bytes_per_row: u32,
    rows_per_image: Option<u32>,
    dst: &ProtocolObject<dyn MTLTexture>,
    texture: &Texture,
    level: u32,
    origin: Origin3d,
    aspect: TextureAspect,
    size: Extent3d,
) {
    let t = &texture.inner;
    let (row, image) = buffer_strides(t, bytes_per_row, rows_per_image, size);
    let copy = clamp_copy(t, level, origin, size);
    if copy.width == 0 || copy.height == 0 || copy.depthOrArrayLayers == 0 {
        return;
    }
    let option = blit_option(t, aspect);
    let is_3d = t.dimension == TextureDimension::D3;
    let (layers, depth) = if is_3d { (1, copy.depthOrArrayLayers) } else { (copy.depthOrArrayLayers, 1) };
    for layer in 0..layers {
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                src,
                offset as usize + layer as usize * image,
                row,
                if is_3d { image } else { 0 },
                MTLSize { width: copy.width as usize, height: copy.height as usize, depth: depth as usize },
                dst,
                if is_3d { 0 } else { (origin.z + layer) as usize },
                level as usize,
                MTLOrigin { x: origin.x as usize, y: origin.y as usize, z: if is_3d { origin.z as usize } else { 0 } },
                option,
            )
        };
    }
}

impl CommandEncoder {
    fn blit(&mut self) -> &ProtocolObject<dyn MTLBlitCommandEncoder> {
        self.end_compute();
        if self.blit.is_none() {
            self.blit = Some(self.raw.blitCommandEncoder().expect("cannot create a Metal blit encoder"));
        }
        self.blit.as_deref().unwrap()
    }

    /// Ends the open blit or compute encoder, before another kind of encoder starts.
    pub(crate) fn end_blit(&mut self) {
        if let Some(b) = self.blit.take() {
            b.endEncoding();
        }
        self.end_compute();
    }

    fn end_compute(&mut self) {
        if let Some(c) = self.compute.take() {
            c.endEncoding();
        }
    }

    /// `color_attachments` entries that are `None` leave that target slot unbound.
    pub fn begin_render_pass(
        &mut self,
        label: &str,
        color_attachments: &[Option<RenderPassColorAttachment<'_>>],
        depth_stencil: Option<&RenderPassDepthStencilAttachment<'_>>,
    ) -> RenderPass<'_> {
        crate::counters::add(&crate::counters::count::render_passes, 1);
        self.end_blit();
        let raw = autoreleasepool(|_| {
            let d = MTLRenderPassDescriptor::new();
            let mut extent = None;
            for (i, c) in color_attachments.iter().enumerate() {
                let Some(c) = c else { continue };
                let Some((texture, level, slice)) = attachment(c.view) else { continue };
                extent.get_or_insert((c.view.extent(), c.view.sample_count()));
                let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
                a.setTexture(Some(texture));
                a.setLevel(level);
                match c.depth_slice {
                    Some(z) => a.setDepthPlane(z as usize),
                    None => a.setSlice(slice),
                }
                let resolve = c.resolve_target.and_then(attachment);
                if let Some((rt, rl, rs)) = resolve {
                    a.setResolveTexture(Some(rt));
                    a.setResolveLevel(rl);
                    a.setResolveSlice(rs);
                }
                a.setLoadAction(conv::load_action(c.load_op));
                a.setStoreAction(conv::store_action(c.store_op, resolve.is_some()));
                let Color { r, g, b, a: alpha } = c.clear_value;
                a.setClearColor(MTLClearColor { red: r, green: g, blue: b, alpha });
            }
            if let Some(ds) = depth_stencil
                && let Some((texture, level, slice)) = attachment(ds.view)
            {
                extent.get_or_insert((ds.view.extent(), ds.view.sample_count()));
                let format = ds.view.format();
                if conv::has_depth(format) {
                    let a = d.depthAttachment();
                    a.setTexture(Some(texture));
                    a.setLevel(level);
                    a.setSlice(slice);
                    if ds.depth_read_only {
                        a.setLoadAction(MTLLoadAction::Load);
                        a.setStoreAction(MTLStoreAction::Store);
                    } else {
                        a.setLoadAction(conv::load_action(ds.depth_load_op));
                        a.setStoreAction(conv::store_action(ds.depth_store_op, false));
                    }
                    a.setClearDepth(f64::from(ds.depth_clear_value));
                }
                if conv::has_stencil(format) {
                    let a = d.stencilAttachment();
                    a.setTexture(Some(texture));
                    a.setLevel(level);
                    a.setSlice(slice);
                    if ds.stencil_read_only {
                        a.setLoadAction(MTLLoadAction::Load);
                        a.setStoreAction(MTLStoreAction::Store);
                    } else {
                        a.setLoadAction(conv::load_action(ds.stencil_load_op));
                        a.setStoreAction(conv::store_action(ds.stencil_store_op, false));
                    }
                    a.setClearStencil(ds.stencil_clear_value);
                }
            }
            if let Some(((w, h), samples)) = extent {
                d.setRenderTargetWidth(w as usize);
                d.setRenderTargetHeight(h as usize);
                d.setDefaultRasterSampleCount(samples as usize);
            }
            super::timing::render_pass(&self.raw, label, &d);
            self.raw.renderCommandEncoderWithDescriptor(&d)
        })
        .expect("cannot create a Metal render encoder");
        set_label!(raw, label);
        RenderPass {
            raw,
            ended: false,
            pipeline: None,
            groups: Default::default(),
            dirty_groups: 0,
            vertex_buffers: Default::default(),
            dirty_vertex: 0,
            index: None,
            vs_cache: SlotCache::default(),
            fs_cache: SlotCache::default(),
            _encoder: PhantomData,
        }
    }

    pub fn begin_compute_pass(&mut self, label: &str) -> ComputePass<'_> {
        crate::counters::add(&crate::counters::count::compute_passes, 1);
        self.end_blit();
        if let Some(b) = self.blit.take() {
            b.endEncoding();
        }
        // Timed passes need encoders of their own (the samples bracket an encoder).
        let timed = super::timing::compute_pass(&self.raw, label);
        if timed.is_some() {
            self.end_compute();
        }
        let raw = match self.compute.take() {
            Some(open) => {
                open.insertDebugSignpost(&ns(label));
                open
            }
            None => {
                let raw = autoreleasepool(|_| match timed {
                    Some(d) => self.raw.computeCommandEncoderWithDescriptor(&d),
                    None => self.raw.computeCommandEncoder(),
                })
                .expect("cannot create a Metal compute encoder");
                set_label!(raw, label);
                raw
            }
        };
        ComputePass {
            raw,
            home: &mut self.compute,
            ended: false,
            pipeline: None,
            groups: Default::default(),
            dirty_groups: 0,
            cache: SlotCache::default(),
        }
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        crate::counters::add(&crate::counters::count::copies, 1);
        let (Some(s), Some(d)) = (src.mtl(), dst.mtl()) else { return };
        if size == 0 {
            return;
        }
        unsafe {
            self.blit().copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &s,
                src_offset as usize,
                &d,
                dst_offset as usize,
                size as usize,
            )
        };
    }

    pub fn copy_buffer_to_texture(&mut self, src: &TexelCopyBufferInfo<'_>, dst: &TexelCopyTextureInfo<'_>, size: Extent3d) {
        crate::counters::add(&crate::counters::count::copies, 1);
        let (Some(s), Some(t)) = (src.buffer.mtl(), dst.texture.mtl()) else { return };
        let blit = self.blit();
        encode_buffer_to_texture(
            blit,
            &s,
            src.offset,
            src.bytes_per_row,
            src.rows_per_image,
            &t,
            dst.texture,
            dst.mip_level,
            dst.origin,
            dst.aspect,
            size,
        );
    }

    pub fn copy_texture_to_buffer(&mut self, src: &TexelCopyTextureInfo<'_>, dst: &TexelCopyBufferInfo<'_>, size: Extent3d) {
        let (Some(t), Some(b)) = (src.texture.mtl(), dst.buffer.mtl()) else { return };
        let ti = &src.texture.inner;
        let (row, image) = buffer_strides(ti, dst.bytes_per_row, dst.rows_per_image, size);
        let copy = clamp_copy(ti, src.mip_level, src.origin, size);
        if copy.width == 0 || copy.height == 0 || copy.depthOrArrayLayers == 0 {
            return;
        }
        let option = blit_option(ti, src.aspect);
        let is_3d = ti.dimension == TextureDimension::D3;
        let (layers, depth) = if is_3d { (1, copy.depthOrArrayLayers) } else { (copy.depthOrArrayLayers, 1) };
        let blit = self.blit();
        for layer in 0..layers {
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                    &t,
                    if is_3d { 0 } else { (src.origin.z + layer) as usize },
                    src.mip_level as usize,
                    MTLOrigin { x: src.origin.x as usize, y: src.origin.y as usize, z: if is_3d { src.origin.z as usize } else { 0 } },
                    MTLSize { width: copy.width as usize, height: copy.height as usize, depth: depth as usize },
                    &b,
                    dst.offset as usize + layer as usize * image,
                    row,
                    if is_3d { image } else { 0 },
                    option,
                )
            };
        }
    }

    pub fn copy_texture_to_texture(&mut self, src: &TexelCopyTextureInfo<'_>, dst: &TexelCopyTextureInfo<'_>, size: Extent3d) {
        let (Some(s), Some(d)) = (src.texture.mtl(), dst.texture.mtl()) else { return };
        let copy = clamp_copy(&src.texture.inner, src.mip_level, src.origin, size);
        if copy.width == 0 || copy.height == 0 || copy.depthOrArrayLayers == 0 {
            return;
        }
        let is_3d = src.texture.inner.dimension == TextureDimension::D3;
        let (layers, depth) = if is_3d { (1, copy.depthOrArrayLayers) } else { (copy.depthOrArrayLayers, 1) };
        let blit = self.blit();
        for layer in 0..layers {
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &s,
                    if is_3d { 0 } else { (src.origin.z + layer) as usize },
                    src.mip_level as usize,
                    MTLOrigin { x: src.origin.x as usize, y: src.origin.y as usize, z: if is_3d { src.origin.z as usize } else { 0 } },
                    MTLSize { width: copy.width as usize, height: copy.height as usize, depth: depth as usize },
                    &d,
                    if is_3d { 0 } else { (dst.origin.z + layer) as usize },
                    dst.mip_level as usize,
                    MTLOrigin { x: dst.origin.x as usize, y: dst.origin.y as usize, z: if is_3d { dst.origin.z as usize } else { 0 } },
                )
            };
        }
    }

    /// Zeroes `offset..offset + size` (`None` = to the end).
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        let Some(b) = buffer.mtl() else { return };
        let size = size.unwrap_or(buffer.size().saturating_sub(offset));
        if size == 0 {
            return;
        }
        self.blit().fillBuffer_range_value(&b, objc2_foundation::NSRange::new(offset as usize, size as usize), 0);
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.pushDebugGroup(&ns(label));
    }

    pub fn pop_debug_group(&mut self) {
        self.raw.popDebugGroup();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.blit().insertDebugSignpost(&ns(label));
    }

    pub fn finish(mut self, label: &str) -> CommandBuffer {
        self.end_blit();
        if !label.is_empty() && self.raw.label().is_none() {
            self.raw.setLabel(Some(&ns(label)));
        }
        let _ = &self.device;
        super::timing::finished(&self.raw);
        CommandBuffer { raw: self.raw.clone() }
    }
}

const MAX_GROUPS: usize = 8;
const MAX_VERTEX_BUFFERS: usize = 8;

/// A bound group and its dynamic offsets.
#[derive(Clone, Default)]
struct BoundGroup {
    group: Option<Rc<GroupInner>>,
    offsets: [u32; 8],
}

/// What each Metal slot of one stage currently holds, to skip redundant binds.
#[derive(Default)]
struct SlotCache {
    buffers: [(usize, u64); 31],
    textures: Vec<usize>,
    samplers: [usize; 16],
}

/// Binds `bindings`' slots from `groups` (only groups in `dirty`) through `set`.
fn apply_groups(
    bindings: &StageBindings,
    groups: &[BoundGroup; MAX_GROUPS],
    dirty: u32,
    cache: &mut SlotCache,
    mut set_buffer: impl FnMut(Option<&ProtocolObject<dyn MTLBuffer>>, usize, usize, bool),
    mut set_texture: impl FnMut(Option<&ProtocolObject<dyn MTLTexture>>, usize),
    mut set_sampler: impl FnMut(Option<&ProtocolObject<dyn MTLSamplerState>>, usize),
    mut set_sizes: impl FnMut(&[u32], usize),
) {
    if bindings.groups & dirty == 0 {
        return;
    }
    let lookup = |group: u32, binding: u32| groups.get(group as usize).and_then(|g| g.group.as_ref().map(|r| (r, &g.offsets))).and_then(|(r, o)| r.get(binding).map(|res| (res, o)));
    for s in &bindings.buffers {
        if dirty & (1 << s.group) == 0 {
            continue;
        }
        if let Some((Resource::Buffer { raw, offset, dynamic, .. }, offsets)) = lookup(s.group, s.binding) {
            let offset = *offset + dynamic.map_or(0, |d| u64::from(offsets[d as usize]));
            let ptr = raw.as_ref().map_or(0, |b| Retained::as_ptr(b) as usize);
            let cached = cache.buffers[s.index];
            if cached != (ptr, offset) {
                set_buffer(raw.as_deref(), offset as usize, s.index, cached.0 == ptr && ptr != 0);
                cache.buffers[s.index] = (ptr, offset);
            }
        }
    }
    if cache.textures.len() < bindings.textures.len() {
        cache.textures.resize(bindings.textures.len(), 0);
    }
    for s in &bindings.textures {
        if dirty & (1 << s.group) == 0 {
            continue;
        }
        if let Some((Resource::Texture(raw), _)) = lookup(s.group, s.binding) {
            let ptr = raw.as_ref().map_or(0, |t| Retained::as_ptr(t) as usize);
            if cache.textures[s.index] != ptr {
                set_texture(raw.as_deref(), s.index);
                cache.textures[s.index] = ptr;
            }
        }
    }
    for s in &bindings.samplers {
        if dirty & (1 << s.group) == 0 {
            continue;
        }
        if let Some((Resource::Sampler(raw), _)) = lookup(s.group, s.binding) {
            let ptr = raw.as_ref().map_or(0, |t| Retained::as_ptr(t) as usize);
            if cache.samplers[s.index] != ptr {
                set_sampler(raw.as_deref(), s.index);
                cache.samplers[s.index] = ptr;
            }
        }
    }
    if let Some((slot, sized)) = &bindings.sizes {
        let sizes: Vec<u32> = sized
            .iter()
            .map(|&(g, b)| match lookup(g, b) {
                Some((Resource::Buffer { raw, offset, size, .. }, _)) => {
                    // Never more than the allocation holds (a failed allocation is tiny).
                    let real = raw.as_ref().map_or(0, |r| r.length() as u64).saturating_sub(*offset);
                    (*size).min(real).min(u64::from(u32::MAX)) as u32
                }
                _ => 0,
            })
            .collect();
        set_sizes(&sizes, *slot);
        cache.buffers[*slot] = (usize::MAX, 0);
    }
}

/// An open render pass; borrowing the encoder keeps it unusable until [`RenderPass::end`]
/// (dropping the pass ends it as well).
pub struct RenderPass<'e> {
    raw: Id<dyn MTLRenderCommandEncoder>,
    ended: bool,
    pipeline: Option<Rc<super::shader::RenderPipelineInner>>,
    groups: [BoundGroup; MAX_GROUPS],
    dirty_groups: u32,
    vertex_buffers: [Option<(MtlBuffer, u64)>; MAX_VERTEX_BUFFERS],
    dirty_vertex: u32,
    index: Option<(MtlBuffer, MTLIndexType, u64, u64)>,
    vs_cache: SlotCache,
    fs_cache: SlotCache,
    _encoder: PhantomData<&'e mut CommandEncoder>,
}

impl RenderPass<'_> {
    pub fn raw(&self) -> *const c_void {
        Retained::as_ptr(&self.raw).cast()
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        crate::counters::add(&crate::counters::count::set_pipelines, 1);
        if self.pipeline.as_ref().is_some_and(|p| Rc::ptr_eq(p, &pipeline.inner)) {
            return;
        }
        if let Some(s) = &pipeline.inner.state {
            let r = &self.raw;
            r.setRenderPipelineState(&s.raw);
            if let Some(ds) = &s.depth_stencil {
                r.setDepthStencilState(Some(ds));
            }
            r.setCullMode(s.cull);
            r.setFrontFacingWinding(s.winding);
            r.setDepthClipMode(s.depth_clip);
            r.setDepthBias_slopeScale_clamp(s.depth_bias.0, s.depth_bias.1, s.depth_bias.2);
            // Slot assignments differ between pipelines.
            let old_base = self.pipeline.as_ref().and_then(|p| p.state.as_ref()).map(|p| p.vertex_buffer_base);
            if old_base != Some(s.vertex_buffer_base) {
                self.dirty_vertex = !0;
            }
        }
        self.pipeline = Some(pipeline.inner.clone());
        self.dirty_groups = !0;
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, dynamic_offsets: &[u32]) {
        crate::counters::add(&crate::counters::count::set_bind_groups, 1);
        let slot = &mut self.groups[index as usize];
        slot.group = Some(group.inner.clone());
        for (o, d) in slot.offsets.iter_mut().zip(dynamic_offsets) {
            *o = *d;
        }
        self.dirty_groups |= 1 << index;
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64, _size: Option<u64>) {
        self.vertex_buffers[slot as usize] = buffer.mtl().map(|b| (b, offset));
        self.dirty_vertex |= 1 << slot;
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, format: IndexFormat, offset: u64, _size: Option<u64>) {
        let (ty, size) = conv::index_type(format);
        self.index = buffer.mtl().map(|b| (b, ty, size, offset));
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        self.raw.setViewport(MTLViewport {
            originX: f64::from(x),
            originY: f64::from(y),
            width: f64::from(width),
            height: f64::from(height),
            znear: f64::from(min_depth),
            zfar: f64::from(max_depth),
        });
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.raw.setScissorRect(MTLScissorRect { x: x as usize, y: y as usize, width: width as usize, height: height as usize });
    }

    pub fn set_blend_constant(&mut self, color: Color) {
        self.raw.setBlendColorRed_green_blue_alpha(color.r as f32, color.g as f32, color.b as f32, color.a as f32);
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.raw.setStencilReferenceValue(reference);
    }

    /// Applies pending bind groups and vertex buffers; `None` if the pipeline is unusable.
    fn flush(&mut self) -> Option<MTLPrimitiveType> {
        let pipeline = self.pipeline.clone()?;
        let s: &RenderState = pipeline.state.as_ref()?;
        let r = &self.raw;
        if self.dirty_groups != 0 {
            let dirty = self.dirty_groups;
            apply_groups(
                &s.vs,
                &self.groups,
                dirty,
                &mut self.vs_cache,
                |b, o, i, same| unsafe {
                    if same {
                        r.setVertexBufferOffset_atIndex(o, i)
                    } else {
                        r.setVertexBuffer_offset_atIndex(b, o, i)
                    }
                },
                |t, i| unsafe { r.setVertexTexture_atIndex(t, i) },
                |t, i| unsafe { r.setVertexSamplerState_atIndex(t, i) },
                |sizes, i| unsafe {
                    r.setVertexBytes_length_atIndex(NonNull::new(sizes.as_ptr().cast_mut().cast()).unwrap(), sizes.len() * 4, i)
                },
            );
            apply_groups(
                &s.fs,
                &self.groups,
                dirty,
                &mut self.fs_cache,
                |b, o, i, same| unsafe {
                    if same {
                        r.setFragmentBufferOffset_atIndex(o, i)
                    } else {
                        r.setFragmentBuffer_offset_atIndex(b, o, i)
                    }
                },
                |t, i| unsafe { r.setFragmentTexture_atIndex(t, i) },
                |t, i| unsafe { r.setFragmentSamplerState_atIndex(t, i) },
                |sizes, i| unsafe {
                    r.setFragmentBytes_length_atIndex(NonNull::new(sizes.as_ptr().cast_mut().cast()).unwrap(), sizes.len() * 4, i)
                },
            );
            self.dirty_groups = 0;
        }
        if self.dirty_vertex != 0 {
            for (i, vb) in self.vertex_buffers.iter().enumerate() {
                if self.dirty_vertex & (1 << i) != 0
                    && let Some((b, offset)) = vb
                {
                    let slot = s.vertex_buffer_base + i;
                    unsafe { r.setVertexBuffer_offset_atIndex(Some(b), *offset as usize, slot) };
                    self.vs_cache.buffers[slot] = (usize::MAX, 0);
                }
            }
            self.dirty_vertex = 0;
        }
        Some(s.primitive)
    }

    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        crate::counters::add(&crate::counters::count::draws, 1);
        let Some(primitive) = self.flush() else { return };
        if vertex_count == 0 || instance_count == 0 {
            return;
        }
        unsafe {
            self.raw.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                primitive,
                first_vertex as usize,
                vertex_count as usize,
                instance_count as usize,
                first_instance as usize,
            )
        };
    }

    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
        crate::counters::add(&crate::counters::count::draws, 1);
        let Some(primitive) = self.flush() else { return };
        let Some((buffer, ty, size, offset)) = self.index.clone() else { return };
        if index_count == 0 || instance_count == 0 {
            return;
        }
        unsafe {
            self.raw.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                primitive,
                index_count as usize,
                ty,
                &buffer,
                (offset + u64::from(first_index) * size) as usize,
                instance_count as usize,
                base_vertex as isize,
                first_instance as usize,
            )
        };
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let Some(primitive) = self.flush() else { return };
        let Some(b) = buffer.mtl() else { return };
        unsafe { self.raw.drawPrimitives_indirectBuffer_indirectBufferOffset(primitive, &b, offset as usize) };
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let Some(primitive) = self.flush() else { return };
        let Some((index, ty, _, index_offset)) = self.index.clone() else { return };
        let Some(b) = buffer.mtl() else { return };
        unsafe {
            self.raw.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                primitive,
                ty,
                &index,
                index_offset as usize,
                &b,
                offset as usize,
            )
        };
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.pushDebugGroup(&ns(label));
    }

    pub fn pop_debug_group(&mut self) {
        self.raw.popDebugGroup();
    }

    pub fn insert_debug_marker(&mut self, label: &str) {
        self.raw.insertDebugSignpost(&ns(label));
    }

    pub fn end(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if !self.ended {
            self.ended = true;
            self.raw.endEncoding();
        }
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}

/// An open compute pass (see [`RenderPass`]).
pub struct ComputePass<'e> {
    raw: Id<dyn MTLComputeCommandEncoder>,
    /// Where the encoder goes back to (still open) when the pass ends.
    home: &'e mut Option<Id<dyn MTLComputeCommandEncoder>>,
    ended: bool,
    pipeline: Option<Rc<super::shader::ComputePipelineInner>>,
    groups: [BoundGroup; MAX_GROUPS],
    dirty_groups: u32,
    cache: SlotCache,
}

impl ComputePass<'_> {
    pub fn raw(&self) -> *const c_void {
        Retained::as_ptr(&self.raw).cast()
    }

    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        crate::counters::add(&crate::counters::count::set_pipelines, 1);
        if let Some(s) = &pipeline.inner.state {
            self.raw.setComputePipelineState(&s.raw);
            for (i, &len) in s.bindings.workgroup_memory.iter().enumerate() {
                unsafe { self.raw.setThreadgroupMemoryLength_atIndex(len.next_multiple_of(16), i) };
            }
        }
        self.pipeline = Some(pipeline.inner.clone());
        self.dirty_groups = !0;
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, dynamic_offsets: &[u32]) {
        crate::counters::add(&crate::counters::count::set_bind_groups, 1);
        let slot = &mut self.groups[index as usize];
        slot.group = Some(group.inner.clone());
        for (o, d) in slot.offsets.iter_mut().zip(dynamic_offsets) {
            *o = *d;
        }
        self.dirty_groups |= 1 << index;
    }

    fn flush(&mut self) -> Option<MTLSize> {
        let pipeline = self.pipeline.clone()?;
        let s: &ComputeState = pipeline.state.as_ref()?;
        let r = &self.raw;
        if self.dirty_groups != 0 {
            apply_groups(
                &s.bindings,
                &self.groups,
                self.dirty_groups,
                &mut self.cache,
                |b, o, i, same| unsafe {
                    if same {
                        r.setBufferOffset_atIndex(o, i)
                    } else {
                        r.setBuffer_offset_atIndex(b, o, i)
                    }
                },
                |t, i| unsafe { r.setTexture_atIndex(t, i) },
                |t, i| unsafe { r.setSamplerState_atIndex(t, i) },
                |sizes, i| unsafe {
                    r.setBytes_length_atIndex(NonNull::new(sizes.as_ptr().cast_mut().cast()).unwrap(), sizes.len() * 4, i)
                },
            );
            self.dirty_groups = 0;
        }
        Some(s.workgroup_size)
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        crate::counters::add(&crate::counters::count::dispatches, 1);
        let Some(size) = self.flush() else { return };
        if x == 0 || y == 0 || z == 0 {
            return;
        }
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: x as usize, height: y as usize, depth: z as usize },
            size,
        );
    }

    pub fn dispatch_workgroups_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let Some(size) = self.flush() else { return };
        let Some(b) = buffer.mtl() else { return };
        unsafe { self.raw.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&b, offset as usize, size) };
    }

    pub fn push_debug_group(&mut self, label: &str) {
        self.raw.pushDebugGroup(&ns(label));
    }

    pub fn pop_debug_group(&mut self) {
        self.raw.popDebugGroup();
    }

    pub fn end(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if !self.ended {
            self.ended = true;
            if super::timing::enabled() {
                self.raw.endEncoding();
            } else {
                *self.home = Some(self.raw.clone());
            }
        }
    }
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}
