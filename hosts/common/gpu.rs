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

    fn d3d(&self, cmd: d3d_hle::command::Command) -> u64 {
        self.0.submit(Command::D3d(cmd))
    }

    fn wait_for(&self, seq: u64) {
        self.0.wait_for(seq);
    }

    fn submitted(&self) -> Option<u64> {
        Some(self.0.submitted())
    }

    fn completed(&self) -> u64 {
        self.0.completed()
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

/// Logs the memory the runtime holds: right away whenever guest commits or GPU memory grew by
/// [`REPORT_GROWTH`] since the last line, else every [`REPORT_PERIOD`]. Phones close tabs that
/// use too much without an error, so the last line before a silent stop shows what grew.
/// Checked by the main thread's watchdog ([`memory_report_tick`]) rather than a thread of its
/// own: on the web each thread is a worker, ~8 MiB each in WebKit.
pub fn start_memory_report() {
    *MEMORY_REPORT.lock().unwrap() = Some(MemoryReport::new());
}

/// Checks the memory report started with [`start_memory_report`]; called every
/// [`MEMORY_REPORT_TICK`].
pub fn memory_report_tick() {
    if let Some(report) = MEMORY_REPORT.lock().unwrap().as_mut() {
        report.tick();
    }
}

pub const MEMORY_REPORT_TICK: std::time::Duration = std::time::Duration::from_millis(250);
const REPORT_GROWTH: u64 = 16 << 20;
const REPORT_PERIOD: std::time::Duration = std::time::Duration::from_secs(10);

static MEMORY_REPORT: std::sync::Mutex<Option<MemoryReport>> = std::sync::Mutex::new(None);

struct MemoryReport {
    /// Guest committed and GPU bytes at the last line.
    last: (u64, u64),
    last_at: std::time::Instant,
    uploads_at_report: u64,
    uploads_at_tick: u64,
    /// Highest upload rate between two ticks since the last line.
    peak_tick: u64,
    tick_at: std::time::Instant,
}

/// Upload bytes (write_buffer + write_texture): browsers copy each upload out of the shared
/// wasm memory, which counts against the page's process (WebKit's in particular).
fn uploaded() -> u64 {
    let c = gpu_webgpu::counters::snapshot();
    c.write_buffer_bytes + c.write_texture_bytes
}

impl MemoryReport {
    fn new() -> Self {
        let now = std::time::Instant::now();
        MemoryReport {
            last: (0, 0),
            last_at: now - REPORT_PERIOD,
            uploads_at_report: uploaded(),
            uploads_at_tick: uploaded(),
            peak_tick: 0,
            tick_at: now,
        }
    }

    fn tick(&mut self) {
        let mib = |b: u64| b >> 20;
        let now_uploaded = uploaded();
        let tick = self.tick_at.elapsed().as_secs_f64().max(0.001);
        self.peak_tick = self.peak_tick.max(((now_uploaded - self.uploads_at_tick) as f64 / tick) as u64);
        (self.uploads_at_tick, self.tick_at) = (now_uploaded, std::time::Instant::now());
        let committed = runtime_core::memory::committed_bytes();
        let (gpu, gpu_peak) = gpu_webgpu::memory::totals();
        let last = self.last;
        let grew = committed >= last.0 + REPORT_GROWTH || gpu >= last.1 + REPORT_GROWTH;
        if !grew && self.last_at.elapsed() < REPORT_PERIOD {
            return;
        }
        #[cfg(target_os = "emscripten")]
        let wasm = format!(
            ", host heap {} MiB, wasm memory {} MiB",
            mib(runtime_core::memory::host_heap_top()),
            mib(core::arch::wasm32::memory_size(0) as u64 * 65536)
        );
        #[cfg(target_vendor = "apple")]
        let wasm = apple_footprint();
        #[cfg(not(any(target_os = "emscripten", target_vendor = "apple")))]
        let wasm = String::new();
        let (pool_used, pool_size) = runtime_core::memory::linear_pool_usage();
        // On GPU growth, what grew: a phone can be killed before the next report.
        let labels = if gpu >= last.1 + REPORT_GROWTH {
            let top: Vec<String> = gpu_webgpu::memory::live_by_label()
                .into_iter()
                .take(6)
                .map(|g| format!("{} \"{}\" {} MiB ({})", g.kind, g.label, mib(g.bytes), g.count))
                .collect();
            format!("; largest: {}", top.join(", "))
        } else {
            String::new()
        };
        let rate = (now_uploaded - self.uploads_at_report) as f64 / self.last_at.elapsed().as_secs_f64().max(0.001);
        log::info!(
            "memory: guest committed {} MiB (linear pool {pool_used}/{pool_size} MiB), GPU {} MiB (peak {} MiB){wasm}; uploads {:.1} MiB/s (peak {:.1} MiB/s){labels}",
            mib(committed),
            mib(gpu),
            mib(gpu_peak),
            rate / f64::from(1 << 20),
            self.peak_tick as f64 / f64::from(1 << 20)
        );
        self.last = (committed, gpu);
        self.last_at = std::time::Instant::now();
        (self.uploads_at_report, self.peak_tick) = (now_uploaded, 0);
    }
}

/// The process's physical footprint, what iOS compares against its per-app limit (Metal
/// allocations included: the GPU shares the memory), and on iOS how much is left before the
/// system ends the app.
#[cfg(target_vendor = "apple")]
fn apple_footprint() -> String {
    /// The start of `task_vm_info` (mach/task_info.h, packed to 4 bytes) up to and including
    /// `phys_footprint`, which revision 1 added.
    #[repr(C, packed(4))]
    #[derive(Default)]
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        sizes: [u64; 14],
        phys_footprint: u64,
    }
    const TASK_VM_INFO: i32 = 22;
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        #[cfg(target_os = "ios")]
        fn os_proc_available_memory() -> usize;
    }
    let mut info = TaskVmInfo::default();
    let mut count = (size_of::<TaskVmInfo>() / 4) as u32;
    // SAFETY: task_info writes at most `count` 32-bit words into `info`.
    let rc = unsafe { task_info(mach_task_self_, TASK_VM_INFO, (&raw mut info).cast(), &mut count) };
    if rc != 0 {
        return String::new();
    }
    let footprint = info.phys_footprint >> 20;
    #[cfg(target_os = "ios")]
    // SAFETY: no arguments; returns 0 outside an app.
    return format!(", footprint {footprint} MiB, {} MiB left", unsafe { os_proc_available_memory() } >> 20);
    #[cfg(not(target_os = "ios"))]
    format!(", footprint {footprint} MiB")
}
