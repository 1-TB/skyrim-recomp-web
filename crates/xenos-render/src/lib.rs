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

mod d3d;
pub mod debug;
pub mod edram;
pub mod executor;
mod extent;
mod float_patch;
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
pub use shared_memory::{MirrorStats, default_mirror_limit, set_mirror_limit};

/// Whether renderers started afterwards use phone-sized GPU memory budgets.
pub(crate) static HANDHELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Phone and tablet browsers close tabs long before desktop ones: renderers started afterwards
/// keep less in GPU memory (a 64 MiB guest memory mirror unless set explicitly; textures within
/// 56 MiB plus 16 MiB kept for reuse, dropped after 4 seconds unused instead of 10).
pub fn handheld_budgets() {
    HANDHELD.store(true, std::sync::atomic::Ordering::Relaxed);
    default_mirror_limit(64);
}

/// Renderers started afterwards render from the title's D3D calls (`--gpu d3d`): see `d3d.rs`.
pub(crate) static D3D_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes renderers started afterwards render from D3D calls: render targets are the D3D
/// surfaces (no EDRAM), and clears, resolves and presents come as [`Command::D3d`].
pub fn set_d3d_mode(on: bool) {
    D3D_MODE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Resolution scale of renderers started afterwards (see [`set_resolution_scale`]).
pub(crate) static RESOLUTION_SCALE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Renderers started afterwards draw at `scale` (1 to 3) times the guest resolution along each
/// axis, like Xenia's draw resolution scaling: EDRAM render targets are that much larger.
pub fn set_resolution_scale(scale: u32) -> Result<(), String> {
    if !(1..=3).contains(&scale) {
        return Err(format!("resolution scale {scale} is not 1, 2 or 3"));
    }
    RESOLUTION_SCALE.store(scale, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

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
#[derive(Debug, Clone, PartialEq)]
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
    /// `--gpu d3d`: a D3D call of the title, rendered without the command stream.
    D3d(d3d_hle::command::Command),
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
