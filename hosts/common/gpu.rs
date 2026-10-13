//! Connects the kernel's command processor to the WebGPU renderer.

use std::sync::Arc;

use kernel_hle::Kernel;
use kernel_hle::gpu::GpuBackend;
use xenos_render::{Command, PhysicalMemory, RendererHandle, ShaderStage};

const PAGE: u32 = 4096;

/// Guest physical memory as the GPU sees it (addresses 0..512 MiB).
pub struct GuestPhysical(pub &'static Kernel);

impl GuestPhysical {
    /// Calls `f(guest_address, offset, len)` for each page-bounded piece of `[addr, addr + len)`
    /// that is backed by guest memory.
    fn pieces(&self, addr: u32, len: usize, mut f: impl FnMut(u32, usize, usize)) {
        let mut done = 0usize;
        while done < len {
            let phys = addr.wrapping_add(done as u32);
            let n = ((PAGE - phys % PAGE) as usize).min(len - done);
            if let Some(guest) = self.0.rt.space.translate_physical(phys) {
                f(guest, done, n);
            }
            done += n;
        }
    }
}

impl PhysicalMemory for GuestPhysical {
    fn read(&self, addr: u32, out: &mut [u8]) {
        out.fill(0);
        let mem = self.0.mem();
        let len = out.len();
        self.pieces(addr, len, |guest, offset, n| mem.read_bytes(guest, &mut out[offset..offset + n]));
    }

    fn write(&self, addr: u32, data: &[u8]) {
        let mem = self.0.mem();
        self.pieces(addr, data.len(), |guest, offset, n| mem.write_bytes(guest, &data[offset..offset + n]));
    }

    fn version(&self, addr: u32, len: u32) -> Option<u64> {
        // Guest stores, recompiled or the kernel's, mark the pages they write.
        Some(runtime_core::dirty::range_version(addr, len))
    }
}

/// [`GpuBackend`] over the renderer thread's queue.
pub struct Renderer(pub RendererHandle);

impl GpuBackend for Renderer {
    fn registers(&self, writes: Vec<(u32, u32)>) {
        self.0.submit(Command::Registers(writes));
    }

    fn shader(&self, pixel: bool, code: Arc<[u32]>) {
        let stage = if pixel { ShaderStage::Pixel } else { ShaderStage::Vertex };
        self.0.submit(Command::Shader { stage, code });
    }

    fn draw(&self) {
        self.0.submit(Command::Draw);
    }

    fn swap(&self, fetch: [u32; 6], width: u32, height: u32) {
        self.0.submit(Command::Swap { fetch, width, height });
    }

    fn wait_idle(&self) {
        self.0.wait_idle();
    }
}

/// The EDRAM emulation for the renderer thread.
pub fn render_targets() -> xenos_render::TargetsFactory {
    Box::new(|device| Box::new(xenos_render::edram::EdramTargets::new(device)))
}

/// Installs `handle` as the kernel's GPU backend.
pub fn install(kernel: &'static Kernel, handle: RendererHandle) {
    if !kernel.gpu.set_backend(Arc::new(Renderer(handle))) {
        log::warn!("GPU backend already installed");
    }
}
