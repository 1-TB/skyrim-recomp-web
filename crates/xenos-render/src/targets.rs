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

    /// Starts a resolve. All earlier rendering has been submitted. With `shared` (the GPU copy of
    /// guest memory) the copy may be stored there instead of guest memory.
    fn resolve(
        &mut self,
        device: &Device,
        info: &ResolveInfo,
        memory: &dyn PhysicalMemory,
        shared: Option<&SharedMemoryTarget<'_>>,
    ) -> ResolveStatus;

    /// Continues the resolve [`RenderTargetCache::resolve`] left pending.
    fn poll_resolve(&mut self, device: &Device, memory: &dyn PhysicalMemory) -> ResolveStatus;

    /// Frame boundary; everything recorded has been submitted, so unused targets can be freed.
    fn end_frame(&mut self);
}
