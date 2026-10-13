//! The renderer: replays the Xenos command stream with WebGPU.
//!
//! WebGPU objects belong to the thread that created them, so all GPU work happens on one
//! renderer thread. The command processor (in the kernel) runs elsewhere; it submits
//! [`Command`]s through a [`RendererHandle`] in stream order and, wherever the guest CPU may
//! observe the GPU's results (fences, resolves into memory, interrupts), waits for the renderer
//! to catch up with [`RendererHandle::wait_idle`].
//!
//! The renderer reads guest memory (front buffers, vertex data, textures) itself through
//! [`PhysicalMemory`], at physical addresses as the GPU sees them.

pub mod debug;
pub mod edram;
pub mod executor;
mod present;
mod queue;
pub mod registers;
mod renderer;
mod ring;
mod shaders;
mod shared_memory;
pub mod targets;

#[cfg(not(target_os = "emscripten"))]
pub mod native;
#[cfg(target_os = "emscripten")]
pub mod web;

use std::sync::Arc;

pub use xenos_shader::ShaderStage;

pub use queue::RendererHandle;
pub use renderer::{TargetsFactory, wanted_features};

/// Guest physical memory, shared with the emulated CPU.
pub trait PhysicalMemory: Send + Sync {
    /// Copies `out.len()` bytes starting at physical address `addr`. Unmapped bytes read as 0.
    fn read(&self, addr: u32, out: &mut [u8]);

    /// Stores `data` at physical address `addr` (resolves write their results back to guest
    /// memory). Writes to unmapped addresses are dropped.
    fn write(&self, addr: u32, data: &[u8]);

    /// If writes to this memory are tracked: a number that has changed whenever a byte of
    /// `[addr, addr + len)` was written since the last call for it, so pages are compared by
    /// version instead of by content. `None` (untracked) makes callers hash the contents.
    fn version(&self, _addr: u32, _len: u32) -> Option<u64> {
        None
    }
}

/// Work for the renderer, in command-stream order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Register writes (index, value) in stream order, including the constant registers.
    Registers(Vec<(u32, u32)>),
    /// Shader microcode (host-order dwords) loaded for a stage (IM_LOAD / IM_LOAD_IMMEDIATE).
    Shader { stage: ShaderStage, code: Arc<[u32]> },
    /// DRAW_INDX(_2): the draw initiator and index buffer registers were written before it.
    Draw,
    /// A frame is complete: show the front buffer described by the texture fetch constant
    /// VdSwap built (`fetch`, host-order dwords).
    Swap { fetch: [u32; 6], width: u32, height: u32 },
}

/// Adapts [`PhysicalMemory`] to the texture loader's memory trait.
pub(crate) struct TextureMemory(pub Arc<dyn PhysicalMemory>);

impl xenos_texture::GuestMemory for TextureMemory {
    fn read(&self, phys_addr: u32, out: &mut [u8]) {
        self.0.read(phys_addr, out);
    }

    fn version(&self, phys_addr: u32, len: u32) -> Option<u64> {
        self.0.version(phys_addr, len)
    }
}
