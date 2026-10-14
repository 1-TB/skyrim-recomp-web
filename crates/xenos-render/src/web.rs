//! Web renderer thread: a pthread that owns the page canvas (as an OffscreenCanvas) and runs
//! as ticks of its event loop, because WebGPU promises and canvas presentation only advance
//! when the worker returns to it. Each tick drains the command queue (up to a resolve waiting
//! for its readback) and presents the newest swap; ticks follow each other immediately while
//! there is work, so a resolve costs about a millisecond rather than a display frame.

use std::sync::Arc;

use gpu_webgpu::*;

use crate::renderer::{Renderer, TargetsFactory, wanted_features};
use crate::{PhysicalMemory, RendererHandle};

/// Canvas backing-store size.
pub const WIDTH: u32 = 1280;
pub const HEIGHT: u32 = 720;
const STACK_SIZE: usize = 1 << 20;

/// Starts the renderer on its own pthread, presenting to the canvas `selector` (e.g.
/// `"#canvas"`). The host must link with `--use-port=emdawnwebgpu:cpp_bindings=false
/// -sOFFSCREENCANVAS_SUPPORT=1` and the page must set `Module.canvas`.
pub fn start(memory: Arc<dyn PhysicalMemory>, selector: &str, targets: TargetsFactory) -> Result<RendererHandle> {
    let handle = RendererHandle::new();
    let renderer = handle.clone();
    let canvas = selector.to_string();
    web::spawn_canvas_thread(selector, STACK_SIZE, move || run(renderer, memory, &canvas, targets))?;
    Ok(handle)
}

enum Phase {
    Adapter(Pending<Result<Adapter>>),
    Device(Adapter, Pending<Result<Device>>),
    Run(Box<Running>),
}

struct Running {
    renderer: Renderer,
    ticks: u64,
    /// Presenter frame shown on the canvas.
    presented: u64,
    last_report: f64,
    perf: Perf,
    /// When frames were shown since the last perf report.
    presents: Vec<std::time::Instant>,
}

fn run(handle: RendererHandle, memory: Arc<dyn PhysicalMemory>, canvas: &str, targets: TargetsFactory) {
    let mut targets = Some(targets);
    let fail = |handle: &RendererHandle, what: &str, e: Error| {
        log::error!("renderer: {what}: {e}");
        handle.close();
    };
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(e) => return fail(&handle, "instance", e),
    };
    // SAFETY: the canvas was transferred to this thread by spawn_canvas_thread.
    let surface = match unsafe { instance.create_surface(SurfaceTarget::Canvas(canvas)) } {
        Ok(s) => s,
        Err(e) => return fail(&handle, "surface", e),
    };
    let request =
        instance.request_adapter(&RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() });
    let mut phase = Some(Phase::Adapter(request));
    web::run_event_loop(move || {
        instance.process_events();
        let next = match phase.take() {
            Some(Phase::Adapter(p)) => match p.try_take() {
                None => Phase::Adapter(p),
                Some(Ok(adapter)) => {
                    let info = adapter.info();
                    log::info!("renderer: {:?} {} {}", info.backend, info.vendor, info.description);
                    // Xenos programs need more than WebGPU's default limits; take what the adapter offers.
                    let features = wanted_features(&adapter.features());
                    let request = adapter.request_device(&DeviceDescriptor {
                        label: "xenos",
                        required_features: &features,
                        required_limits: Some(adapter.limits()),
                    });
                    Phase::Device(adapter, request)
                }
                Some(Err(e)) => {
                    fail(&handle, "adapter", e);
                    return web::Next::Stop;
                }
            },
            Some(Phase::Device(adapter, p)) => match p.try_take() {
                None => Phase::Device(adapter, p),
                Some(Ok(device)) => {
                    match configure(&surface, &adapter, device, memory.clone(), targets.take().unwrap()) {
                        Ok(r) => Phase::Run(Box::new(r)),
                        Err(e) => {
                            fail(&handle, "surface configuration", e);
                            return web::Next::Stop;
                        }
                    }
                }
                Some(Err(e)) => {
                    fail(&handle, "device", e);
                    return web::Next::Stop;
                }
            },
            Some(Phase::Run(mut r)) => {
                let next = match r.tick(&handle, &surface) {
                    Ok(next) => next,
                    Err(e) => {
                        fail(&handle, "frame", e);
                        return web::Next::Stop;
                    }
                };
                phase = Some(Phase::Run(r));
                return next;
            }
            None => return web::Next::Stop,
        };
        phase = Some(next);
        web::Next::After(1)
    });
}

fn configure(
    surface: &Surface,
    adapter: &Adapter,
    device: Device,
    memory: Arc<dyn PhysicalMemory>,
    targets: TargetsFactory,
) -> Result<Running> {
    let caps = surface.capabilities(adapter)?;
    let format = *caps.formats.first().ok_or(Error::Surface("no surface formats".into()))?;
    surface.configure(
        &device,
        &SurfaceConfiguration {
            format,
            usage: TextureUsage::RenderAttachment,
            width: WIDTH,
            height: HEIGHT,
            view_formats: &[],
            alpha_mode: CompositeAlphaMode::Opaque,
            present_mode: PresentMode::Fifo,
        },
    );
    Ok(Running {
        renderer: Renderer::new(&device, memory, format, targets),
        ticks: 0,
        presented: u64::MAX,
        last_report: 0.0,
        perf: Perf::default(),
        presents: Vec::new(),
    })
}

/// Where the renderer thread's time went over a reporting window (logged every few seconds, to
/// tell whether the GPU, the renderer or the emulated CPU limits the frame rate).
#[derive(Default)]
struct Perf {
    start_ms: f64,
    last_ms: f64,
    /// Executing commands.
    work_ms: f64,
    /// Blocked on the GPU: a resolve, or the frames-in-flight limit.
    gpu_ms: f64,
    frames: u64,
    draws: u64,
    calls: gpu_webgpu::counters::Counters,
    times: crate::executor::OpTimes,
}

const PERF_WINDOW_MS: f64 = 5000.0;
/// Longest sleep without work (GPU callbacks and the perf log still need ticks).
const IDLE_POLL_MS: u32 = 16;

impl Running {
    fn report_perf(&mut self, now: f64) {
        let p = &mut self.perf;
        let frames = self.renderer.presenter.frames;
        let draws = self.renderer.executor.stats.draws;
        if p.start_ms > 0.0 && now - p.start_ms >= PERF_WINDOW_MS {
            let span = now - p.start_ms;
            let n = frames - p.frames;
            let pct = |ms: f64| 100.0 * ms / span;
            log::info!(
                "perf: {:.1} fps, {} draws/frame; renderer thread: {:.0}% executing, {:.0}% waiting for the GPU, \
                 {:.0}% waiting for the game{}",
                n as f64 * 1000.0 / span,
                (draws - p.draws) / n.max(1),
                pct(p.work_ms),
                pct(p.gpu_ms),
                pct((span - p.work_ms - p.gpu_ms).max(0.0)),
                crate::debug::describe(),
            );
            log::info!(
                "perf: per frame: {}",
                crate::renderer::calls_per_frame(&gpu_webgpu::counters::snapshot().since(&p.calls), n)
            );
            let times = self.renderer.executor.times();
            let ms = |now: std::time::Duration, then: std::time::Duration| (now - then).as_secs_f64() * 1000.0;
            log::info!(
                "perf: renderer time: {:.0} ms binding new texture sets, {:.0} ms creating pipelines, {:.0} ms on \
                 shaders, {:.0} ms recording resolves, {:.0} ms submitting",
                ms(times.texture_sets, p.times.texture_sets),
                ms(times.pipelines, p.times.pipelines),
                ms(times.shaders, p.times.shaders),
                ms(times.resolves, p.times.resolves),
                ms(times.submits, p.times.submits),
            );
            log::info!("perf: {}", self.renderer.gpu_memory());
            log::info!("perf: {}", crate::renderer::pacing(&mut self.presents, &mut self.renderer.swap_timings));
        }
        if p.start_ms == 0.0 || now - p.start_ms >= PERF_WINDOW_MS {
            *p = Perf {
                start_ms: now,
                last_ms: now,
                frames,
                draws,
                calls: gpu_webgpu::counters::snapshot(),
                times: self.renderer.executor.times(),
                ..Perf::default()
            };
        }
    }

    fn tick(&mut self, handle: &RendererHandle, surface: &Surface) -> Result<web::Next> {
        handle.unpark();
        let before = handle.completed();
        let started = web::now_ms();
        // Yield every few milliseconds even with work queued (see `Renderer::pump`).
        self.renderer.pump(handle, None, Some(8.0), |_, _| {});
        let progressed = handle.completed() != before;
        let waiting = self.renderer.executor.busy();
        let pumped = web::now_ms();
        // Time between ticks counts as waiting on whatever this tick found.
        let since_last = (started - self.perf.last_ms).max(0.0);
        if progressed {
            self.perf.work_ms += pumped - started;
        }
        if waiting {
            self.perf.gpu_ms += since_last + if progressed { 0.0 } else { pumped - started };
        }
        self.perf.last_ms = pumped;
        self.ticks += 1;
        let now = web::now_ms();
        self.report_perf(now);
        if now - self.last_report >= 2000.0 {
            self.last_report = now;
            log::debug!(
                "renderer: tick {}, completed {} of {}, backlog {}, resolving {waiting}, {:?}",
                self.ticks,
                handle.completed(),
                handle.submitted(),
                self.renderer.backlog_len(),
                self.renderer.executor.stats
            );
        }
        if self.renderer.presenter.frames != self.presented {
            self.presented = self.renderer.presenter.frames;
            let device = self.renderer.device.clone();
            let frame = surface.current_frame()?;
            let view = frame.texture.create_view(None);
            let mut encoder = device.create_command_encoder("present");
            self.renderer.presenter.encode(&mut encoder, &view, WIDTH, HEIGHT);
            device.queue().submit([encoder.finish("present")]);
            frame.present();
            self.presents.push(std::time::Instant::now());
        }
        for e in self.renderer.device.take_errors() {
            log::error!("renderer: GPU error: {e:?}");
        }
        if waiting {
            // Blocked on the GPU (a resolve readback or the frames-in-flight limit): poll on a
            // timer instead of spinning, which would take a core from the title's threads.
            return Ok(web::Next::After(1));
        }
        if progressed || self.renderer.backlog_len() > 0 {
            return Ok(web::Next::Now);
        }
        // Out of work: sleep until the command processor submits more. It often waits for the
        // renderer to catch up (before fences and interrupts), so waking late stalls both.
        Ok(match handle.park() {
            Some((word, value)) => web::Next::OnChange { addr: word, value, timeout_ms: IDLE_POLL_MS },
            None => web::Next::Now,
        })
    }
}
