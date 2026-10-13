//! [`RenderTargetCache`] over xenos-edram's host render targets and resolves.

use gpu_webgpu::{CommandEncoder, Device};
use xenos_edram::{EdramCache, GuestMemoryAccess, PendingResolve};
use xenos_state::edram::RenderTargetSetup;
use xenos_state::resolve::ResolveInfo;

use crate::PhysicalMemory;
use crate::targets::{DrawTargets, RenderTargetCache, ResolveStatus, SharedMemoryTarget};

struct Memory<'a>(&'a dyn PhysicalMemory);

impl GuestMemoryAccess for Memory<'_> {
    fn read(&self, addr: u32, out: &mut [u8]) {
        self.0.read(addr, out);
    }

    fn write(&self, addr: u32, data: &[u8]) {
        self.0.write(addr, data);
    }
}

pub struct EdramTargets {
    cache: EdramCache,
    pending: Option<PendingResolve>,
}

impl EdramTargets {
    pub fn new(device: &Device) -> Self {
        EdramTargets { cache: EdramCache::new(device), pending: None }
    }
}

impl RenderTargetCache for EdramTargets {
    fn targets(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        setup: &RenderTargetSetup,
        min_height: u32,
    ) -> Option<DrawTargets> {
        match self.cache.targets(device, encoder, setup, min_height) {
            Ok(t) if t.is_empty() => None,
            Ok(t) => Some(DrawTargets {
                blendable: std::array::from_fn(|i| t.color_info[i].is_none_or(|c| c.blendable)),
                colors: t.colors,
                depth: t.depth,
                width: t.width,
                height: t.height,
                sample_count: t.sample_count,
                sample_mask: t.sample_mask,
            }),
            Err(e) => {
                log::warn!("edram: render targets: {e}");
                None
            }
        }
    }

    fn resolve(
        &mut self,
        device: &Device,
        info: &ResolveInfo,
        memory: &dyn PhysicalMemory,
        shared: Option<&SharedMemoryTarget<'_>>,
    ) -> ResolveStatus {
        match self.cache.begin_resolve_to(device, info, shared) {
            Ok(pending) => {
                self.pending = Some(pending);
                self.poll_resolve(device, memory)
            }
            Err(e) => {
                log::warn!("edram: resolve: {e}");
                ResolveStatus::Done { written: None, in_gpu_memory: false }
            }
        }
    }

    fn poll_resolve(&mut self, device: &Device, memory: &dyn PhysicalMemory) -> ResolveStatus {
        let Some(pending) = self.pending.as_mut() else {
            return ResolveStatus::Done { written: None, in_gpu_memory: false };
        };
        // Stored into GPU memory: complete without a readback.
        let in_gpu_memory = pending.is_complete();
        match self.cache.poll_resolve(device, pending, &Memory(memory)) {
            None => ResolveStatus::Pending,
            Some(result) => {
                self.pending = None;
                match result {
                    Ok(r) => ResolveStatus::Done { written: r.written, in_gpu_memory },
                    Err(e) => {
                        log::warn!("edram: resolve: {e}");
                        ResolveStatus::Done { written: None, in_gpu_memory: false }
                    }
                }
            }
        }
    }
}
