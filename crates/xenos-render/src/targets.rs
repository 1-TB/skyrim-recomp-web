//! The executor's interface to the EDRAM emulation (host render targets and resolves).

use gpu_webgpu::{CommandEncoder, Device, TextureFormat, TextureView};
pub use xenos_edram::SharedMemoryTarget;
use xenos_state::edram::RenderTargetSetup;
use xenos_state::resolve::ResolveInfo;

use crate::PhysicalMemory;

/// Host attachments standing in for the EDRAM ranges a draw uses.
#[derive(Clone)]
pub struct DrawTargets {
    pub colors: [Option<(TextureView, TextureFormat)>; 4],
    pub depth: Option<(TextureView, TextureFormat)>,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    /// Pipeline sample mask (e.g. guest 2x MSAA in 2 of 4 host samples).
    pub sample_mask: u32,
    /// Whether each color target's format supports blending.
    pub blendable: [bool; 4],
    /// Clears the pass does first (as load ops), in place of passes of their own.
    pub clears: AttachmentClears,
}

/// Clear values of a pass's attachments (`None`: keep the contents).
#[derive(Clone, Copy, Default)]
pub struct AttachmentClears {
    pub colors: [Option<gpu_webgpu::Color>; 4],
    pub depth: Option<f32>,
    pub stencil: Option<u32>,
}

impl AttachmentClears {
    pub fn any(&self) -> bool {
        self.colors.iter().any(Option::is_some) || self.depth.is_some() || self.stencil.is_some()
    }
}

impl DrawTargets {
    /// Whether both describe the same attachments (so draws can share a render pass).
    pub fn same_attachments(&self, other: &DrawTargets) -> bool {
        self.colors == other.colors && self.depth == other.depth && self.sample_count == other.sample_count
    }
}

/// Progress of a resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveStatus {
    /// GPU work or a readback is still in flight; poll again later.
    Pending,
    /// Finished; `written` is the guest physical range stored to, if any — in guest memory, or
    /// only in the GPU copy of it (`in_gpu_memory`, see [`RenderTargetCache::resolve`]).
    Done { written: Option<(u32, u32)>, in_gpu_memory: bool },
}

pub trait RenderTargetCache {
    /// Attachments for `setup`, at least `min_height` rows tall. May encode copies (ownership
    /// transfers between EDRAM layouts) into `encoder`; the caller encodes the draw's render pass
    /// after them in the same encoder. `None` if the setup can't be represented (logged).
    fn targets(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        setup: &RenderTargetSetup,
        min_height: u32,
    ) -> Option<DrawTargets>;

    /// Starts a resolve. With `shared` (the GPU copy of guest memory) the copy is stored there
    /// instead of guest memory, and with `encoder` too the resolve is recorded into it after
    /// the rendering recorded there (done once it's submitted); otherwise all earlier rendering
    /// has been submitted and the resolve submits its own work.
    fn resolve(
        &mut self,
        device: &Device,
        info: &ResolveInfo,
        memory: &dyn PhysicalMemory,
        shared: Option<&SharedMemoryTarget<'_>>,
        encoder: Option<&mut CommandEncoder>,
    ) -> ResolveStatus;

    /// Continues the resolve [`RenderTargetCache::resolve`] left pending.
    fn poll_resolve(&mut self, device: &Device, memory: &dyn PhysicalMemory) -> ResolveStatus;

    /// Frame boundary; everything recorded has been submitted, so unused targets can be freed.
    fn end_frame(&mut self);

    /// Sends data the work recorded so far needs (call before submitting it).
    fn flush_uploads(&mut self, device: &Device);

    /// The render target whose host texture holds EDRAM tile `tile`, if any.
    fn tile_owner(&self, _tile: u32) -> Option<xenos_state::edram::RenderTargetKey> {
        None
    }

    /// Host pixels per guest pixel along each axis: attachments are this much larger than the
    /// guest's render targets, and draws scale their viewport and scissor to match.
    fn resolution_scale(&self) -> u32 {
        1
    }
}
