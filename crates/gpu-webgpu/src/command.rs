use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use crate::handle::{handle, label, string, sv, wbool};
use crate::pending::{Pending, complete};
use crate::{
    BindGroup, Buffer, Color, ComputePipeline, Device, Error, Extent3d, IndexFormat, LoadOp, Origin3d, RenderPipeline,
    Result, StoreOp, Texture, TextureAspect, TextureView, ffi,
};

handle!(Queue, WGPUQueue, wgpuQueueAddRef, wgpuQueueRelease);
handle!(CommandEncoder, WGPUCommandEncoder, wgpuCommandEncoderAddRef, wgpuCommandEncoderRelease);
handle!(CommandBuffer, WGPUCommandBuffer, wgpuCommandBufferAddRef, wgpuCommandBufferRelease);

/// A buffer side of a texture copy. `bytes_per_row` must be a multiple of 256 for
/// buffer<->texture copies (not for [`Queue::write_texture`]).
#[derive(Clone, Copy, Debug)]
pub struct TexelCopyBufferInfo<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub bytes_per_row: u32,
    /// Needed only for multi-layer/3D copies.
    pub rows_per_image: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TexelCopyTextureInfo<'a> {
    pub texture: &'a Texture,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

impl<'a> TexelCopyTextureInfo<'a> {
    /// Mip 0, origin 0, all aspects.
    pub fn whole(texture: &'a Texture) -> Self {
        Self { texture, mip_level: 0, origin: crate::origin(0, 0, 0), aspect: TextureAspect::All }
    }

    fn to_ffi(self) -> ffi::WGPUTexelCopyTextureInfo {
        ffi::WGPUTexelCopyTextureInfo {
            texture: self.texture.raw(),
            mipLevel: self.mip_level,
            origin: self.origin,
            aspect: self.aspect,
        }
    }
}

fn buffer_layout(offset: u64, bytes_per_row: u32, rows_per_image: Option<u32>) -> ffi::WGPUTexelCopyBufferLayout {
    ffi::WGPUTexelCopyBufferLayout {
        offset,
        bytesPerRow: bytes_per_row,
        rowsPerImage: rows_per_image.unwrap_or(ffi::WGPU_COPY_STRIDE_UNDEFINED),
    }
}

impl TexelCopyBufferInfo<'_> {
    fn to_ffi(self) -> ffi::WGPUTexelCopyBufferInfo {
        ffi::WGPUTexelCopyBufferInfo {
            layout: buffer_layout(self.offset, self.bytes_per_row, self.rows_per_image),
            buffer: self.buffer.raw(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderPassColorAttachment<'a> {
    pub view: &'a TextureView,
    pub resolve_target: Option<&'a TextureView>,
    /// Slice of a 3D texture to render to.
    pub depth_slice: Option<u32>,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_value: Color,
}

impl<'a> RenderPassColorAttachment<'a> {
    /// Clear to `color`, then store.
    pub fn clear(view: &'a TextureView, color: Color) -> Self {
        Self {
            view,
            resolve_target: None,
            depth_slice: None,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_value: color,
        }
    }

    /// Keep the current contents, then store.
    pub fn load(view: &'a TextureView) -> Self {
        Self { load_op: LoadOp::Load, ..Self::clear(view, Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }) }
    }
}

/// Ops of an aspect the format lacks (or of a read-only aspect) must stay `Undefined`.
#[derive(Clone, Copy, Debug)]
pub struct RenderPassDepthStencilAttachment<'a> {
    pub view: &'a TextureView,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub depth_clear_value: f32,
    pub depth_read_only: bool,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub stencil_clear_value: u32,
    pub stencil_read_only: bool,
}

impl<'a> RenderPassDepthStencilAttachment<'a> {
    /// Depth-only attachment cleared to `depth` and stored.
    pub fn depth_clear(view: &'a TextureView, depth: f32) -> Self {
        Self {
            view,
            depth_load_op: LoadOp::Clear,
            depth_store_op: StoreOp::Store,
            depth_clear_value: depth,
            depth_read_only: false,
            stencil_load_op: LoadOp::Undefined,
            stencil_store_op: StoreOp::Undefined,
            stencil_clear_value: 0,
            stencil_read_only: false,
        }
    }
}

impl Device {
    pub fn create_command_encoder(&self, label_: &str) -> CommandEncoder {
        let desc = ffi::WGPUCommandEncoderDescriptor { label: label(label_), ..Default::default() };
        CommandEncoder::from_created(unsafe { ffi::wgpuDeviceCreateCommandEncoder(self.raw(), &desc) }, "encoder")
            .expect("wgpuDeviceCreateCommandEncoder returned null")
    }
}

impl CommandEncoder {
    /// `color_attachments` entries that are `None` leave that target slot unbound.
    pub fn begin_render_pass(
        &mut self,
        label_: &str,
        color_attachments: &[Option<RenderPassColorAttachment<'_>>],
        depth_stencil: Option<&RenderPassDepthStencilAttachment<'_>>,
    ) -> RenderPass<'_> {
        crate::counters::add(&crate::counters::count::render_passes, 1);
        let colors: Vec<_> = color_attachments
            .iter()
            .map(|c| match c {
                Some(c) => ffi::WGPURenderPassColorAttachment {
                    view: c.view.raw(),
                    depthSlice: c.depth_slice.unwrap_or(ffi::WGPU_DEPTH_SLICE_UNDEFINED),
                    resolveTarget: c.resolve_target.map_or(ptr::null_mut(), |v| v.raw()),
                    loadOp: c.load_op,
                    storeOp: c.store_op,
                    clearValue: c.clear_value,
                    ..Default::default()
                },
                None => ffi::WGPURenderPassColorAttachment::default(),
            })
            .collect();
        let depth = depth_stencil.map(|d| ffi::WGPURenderPassDepthStencilAttachment {
            view: d.view.raw(),
            depthLoadOp: d.depth_load_op,
            depthStoreOp: d.depth_store_op,
            depthClearValue: d.depth_clear_value,
            depthReadOnly: wbool(d.depth_read_only),
            stencilLoadOp: d.stencil_load_op,
            stencilStoreOp: d.stencil_store_op,
            stencilClearValue: d.stencil_clear_value,
            stencilReadOnly: wbool(d.stencil_read_only),
            ..Default::default()
        });
        let desc = ffi::WGPURenderPassDescriptor {
            label: label(label_),
            colorAttachmentCount: colors.len(),
            colorAttachments: colors.as_ptr(),
            depthStencilAttachment: depth.as_ref().map_or(ptr::null(), ptr::from_ref),
            ..Default::default()
        };
        let raw = unsafe { ffi::wgpuCommandEncoderBeginRenderPass(self.raw(), &desc) };
        assert!(!raw.is_null(), "wgpuCommandEncoderBeginRenderPass returned null");
        RenderPass { raw, ended: false, _encoder: PhantomData }
    }

    pub fn begin_compute_pass(&mut self, label_: &str) -> ComputePass<'_> {
        crate::counters::add(&crate::counters::count::compute_passes, 1);
        let desc = ffi::WGPUComputePassDescriptor { label: label(label_), ..Default::default() };
        let raw = unsafe { ffi::wgpuCommandEncoderBeginComputePass(self.raw(), &desc) };
        assert!(!raw.is_null(), "wgpuCommandEncoderBeginComputePass returned null");
        ComputePass { raw, ended: false, _encoder: PhantomData }
    }

    pub fn copy_buffer_to_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) {
        crate::counters::add(&crate::counters::count::copies, 1);
        unsafe {
            ffi::wgpuCommandEncoderCopyBufferToBuffer(self.raw(), src.raw(), src_offset, dst.raw(), dst_offset, size)
        };
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        src: &TexelCopyBufferInfo<'_>,
        dst: &TexelCopyTextureInfo<'_>,
        size: Extent3d,
    ) {
        crate::counters::add(&crate::counters::count::copies, 1);
        unsafe { ffi::wgpuCommandEncoderCopyBufferToTexture(self.raw(), &src.to_ffi(), &dst.to_ffi(), &size) };
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        src: &TexelCopyTextureInfo<'_>,
        dst: &TexelCopyBufferInfo<'_>,
        size: Extent3d,
    ) {
        unsafe { ffi::wgpuCommandEncoderCopyTextureToBuffer(self.raw(), &src.to_ffi(), &dst.to_ffi(), &size) };
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: &TexelCopyTextureInfo<'_>,
        dst: &TexelCopyTextureInfo<'_>,
        size: Extent3d,
    ) {
        unsafe { ffi::wgpuCommandEncoderCopyTextureToTexture(self.raw(), &src.to_ffi(), &dst.to_ffi(), &size) };
    }

    /// Zeroes `offset..offset + size` (`None` = to the end).
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: Option<u64>) {
        unsafe {
            ffi::wgpuCommandEncoderClearBuffer(self.raw(), buffer.raw(), offset, size.unwrap_or(ffi::WGPU_WHOLE_SIZE))
        };
    }

    pub fn push_debug_group(&mut self, label_: &str) {
        unsafe { ffi::wgpuCommandEncoderPushDebugGroup(self.raw(), sv(label_)) };
    }

    pub fn pop_debug_group(&mut self) {
        unsafe { ffi::wgpuCommandEncoderPopDebugGroup(self.raw()) };
    }

    pub fn insert_debug_marker(&mut self, label_: &str) {
        unsafe { ffi::wgpuCommandEncoderInsertDebugMarker(self.raw(), sv(label_)) };
    }

    pub fn finish(self, label_: &str) -> CommandBuffer {
        let desc = ffi::WGPUCommandBufferDescriptor { label: label(label_), ..Default::default() };
        CommandBuffer::from_created(unsafe { ffi::wgpuCommandEncoderFinish(self.raw(), &desc) }, "command buffer")
            .expect("wgpuCommandEncoderFinish returned null")
    }
}

/// An open render pass; borrowing the encoder keeps it unusable until [`RenderPass::end`]
/// (dropping the pass ends it as well).
pub struct RenderPass<'e> {
    raw: ffi::WGPURenderPassEncoder,
    ended: bool,
    _encoder: PhantomData<&'e mut CommandEncoder>,
}

impl RenderPass<'_> {
    pub fn raw(&self) -> ffi::WGPURenderPassEncoder {
        self.raw
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        crate::counters::add(&crate::counters::count::set_pipelines, 1);
        unsafe { ffi::wgpuRenderPassEncoderSetPipeline(self.raw, pipeline.raw()) };
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, dynamic_offsets: &[u32]) {
        crate::counters::add(&crate::counters::count::set_bind_groups, 1);
        unsafe {
            ffi::wgpuRenderPassEncoderSetBindGroup(
                self.raw,
                index,
                group.raw(),
                dynamic_offsets.len(),
                dynamic_offsets.as_ptr(),
            )
        };
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64, size: Option<u64>) {
        unsafe {
            ffi::wgpuRenderPassEncoderSetVertexBuffer(
                self.raw,
                slot,
                buffer.raw(),
                offset,
                size.unwrap_or(ffi::WGPU_WHOLE_SIZE),
            )
        };
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, format: IndexFormat, offset: u64, size: Option<u64>) {
        unsafe {
            ffi::wgpuRenderPassEncoderSetIndexBuffer(
                self.raw,
                buffer.raw(),
                format,
                offset,
                size.unwrap_or(ffi::WGPU_WHOLE_SIZE),
            )
        };
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        unsafe { ffi::wgpuRenderPassEncoderSetViewport(self.raw, x, y, width, height, min_depth, max_depth) };
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        unsafe { ffi::wgpuRenderPassEncoderSetScissorRect(self.raw, x, y, width, height) };
    }

    pub fn set_blend_constant(&mut self, color: Color) {
        unsafe { ffi::wgpuRenderPassEncoderSetBlendConstant(self.raw, &color) };
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { ffi::wgpuRenderPassEncoderSetStencilReference(self.raw, reference) };
    }

    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        crate::counters::add(&crate::counters::count::draws, 1);
        unsafe { ffi::wgpuRenderPassEncoderDraw(self.raw, vertex_count, instance_count, first_vertex, first_instance) };
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        crate::counters::add(&crate::counters::count::draws, 1);
        unsafe {
            ffi::wgpuRenderPassEncoderDrawIndexed(
                self.raw,
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            )
        };
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: u64) {
        unsafe { ffi::wgpuRenderPassEncoderDrawIndirect(self.raw, buffer.raw(), offset) };
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64) {
        unsafe { ffi::wgpuRenderPassEncoderDrawIndexedIndirect(self.raw, buffer.raw(), offset) };
    }

    pub fn push_debug_group(&mut self, label_: &str) {
        unsafe { ffi::wgpuRenderPassEncoderPushDebugGroup(self.raw, sv(label_)) };
    }

    pub fn pop_debug_group(&mut self) {
        unsafe { ffi::wgpuRenderPassEncoderPopDebugGroup(self.raw) };
    }

    pub fn insert_debug_marker(&mut self, label_: &str) {
        unsafe { ffi::wgpuRenderPassEncoderInsertDebugMarker(self.raw, sv(label_)) };
    }

    pub fn end(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if !self.ended {
            self.ended = true;
            unsafe { ffi::wgpuRenderPassEncoderEnd(self.raw) };
        }
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        self.finish();
        unsafe { ffi::wgpuRenderPassEncoderRelease(self.raw) };
    }
}

/// An open compute pass (see [`RenderPass`]).
pub struct ComputePass<'e> {
    raw: ffi::WGPUComputePassEncoder,
    ended: bool,
    _encoder: PhantomData<&'e mut CommandEncoder>,
}

impl ComputePass<'_> {
    pub fn raw(&self) -> ffi::WGPUComputePassEncoder {
        self.raw
    }

    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        crate::counters::add(&crate::counters::count::set_pipelines, 1);
        unsafe { ffi::wgpuComputePassEncoderSetPipeline(self.raw, pipeline.raw()) };
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup, dynamic_offsets: &[u32]) {
        crate::counters::add(&crate::counters::count::set_bind_groups, 1);
        unsafe {
            ffi::wgpuComputePassEncoderSetBindGroup(
                self.raw,
                index,
                group.raw(),
                dynamic_offsets.len(),
                dynamic_offsets.as_ptr(),
            )
        };
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        crate::counters::add(&crate::counters::count::dispatches, 1);
        unsafe { ffi::wgpuComputePassEncoderDispatchWorkgroups(self.raw, x, y, z) };
    }

    pub fn dispatch_workgroups_indirect(&mut self, buffer: &Buffer, offset: u64) {
        unsafe { ffi::wgpuComputePassEncoderDispatchWorkgroupsIndirect(self.raw, buffer.raw(), offset) };
    }

    pub fn push_debug_group(&mut self, label_: &str) {
        unsafe { ffi::wgpuComputePassEncoderPushDebugGroup(self.raw, sv(label_)) };
    }

    pub fn pop_debug_group(&mut self) {
        unsafe { ffi::wgpuComputePassEncoderPopDebugGroup(self.raw) };
    }

    pub fn end(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if !self.ended {
            self.ended = true;
            unsafe { ffi::wgpuComputePassEncoderEnd(self.raw) };
        }
    }
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.finish();
        unsafe { ffi::wgpuComputePassEncoderRelease(self.raw) };
    }
}

impl Queue {
    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) {
        crate::counters::add(&crate::counters::count::submits, 1);
        let buffers: Vec<CommandBuffer> = buffers.into_iter().collect();
        let raws: Vec<_> = buffers.iter().map(|b| b.raw()).collect();
        unsafe { ffi::wgpuQueueSubmit(self.raw(), raws.len(), raws.as_ptr()) };
    }

    /// `offset` and `data.len()` must be multiples of 4.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        crate::counters::add(&crate::counters::count::write_buffer_bytes, data.len() as u64);
        crate::counters::add(&crate::counters::count::write_buffers, 1);
        unsafe { ffi::wgpuQueueWriteBuffer(self.raw(), buffer.raw(), offset, data.as_ptr().cast(), data.len()) };
    }

    /// Uploads `data` laid out with `bytes_per_row` (no 256-byte alignment needed here).
    pub fn write_texture(
        &self,
        dst: &TexelCopyTextureInfo<'_>,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: Option<u32>,
        size: Extent3d,
    ) {
        crate::counters::add(&crate::counters::count::write_textures, 1);
        let layout = buffer_layout(0, bytes_per_row, rows_per_image);
        unsafe {
            ffi::wgpuQueueWriteTexture(self.raw(), &dst.to_ffi(), data.as_ptr().cast(), data.len(), &layout, &size)
        };
    }

    /// Resolves once all work submitted so far has finished on the GPU.
    pub fn on_submitted_work_done(&self) -> Pending<Result<()>> {
        unsafe extern "C" fn done(
            status: ffi::WGPUQueueWorkDoneStatus,
            message: ffi::WGPUStringView,
            userdata: *mut c_void,
            _: *mut c_void,
        ) {
            let result = if status == ffi::WGPUQueueWorkDoneStatus::Success {
                Ok(())
            } else {
                Err(Error::WorkDone(format!("status {}: {}", status.0, string(message))))
            };
            unsafe { complete(userdata, result) };
        }
        let (pending, userdata) = Pending::new();
        let info = ffi::WGPUQueueWorkDoneCallbackInfo {
            mode: ffi::WGPUCallbackMode::AllowProcessEvents,
            callback: Some(done),
            userdata1: userdata,
            ..Default::default()
        };
        unsafe { ffi::wgpuQueueOnSubmittedWorkDone(self.raw(), info) };
        pending
    }
}
