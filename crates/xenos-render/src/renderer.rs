//! The renderer thread's main object: executes queued commands in order and presents swaps.

use std::sync::Arc;
use std::time::Duration;

use gpu_webgpu::{Device, FeatureName, TextureFormat};

use crate::executor::Executor;
use crate::present::Presenter;
use crate::targets::RenderTargetCache;
use crate::{Command, PhysicalMemory, RendererHandle};

/// Milliseconds on a monotonic clock.
pub(crate) fn now_ms() -> f64 {
    #[cfg(target_os = "emscripten")]
    {
        gpu_webgpu::web::now_ms()
    }
    #[cfg(not(target_os = "emscripten"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

/// Creates the EDRAM emulation on the renderer thread once the device exists.
pub type TargetsFactory = Box<dyn FnOnce(&Device) -> Box<dyn RenderTargetCache> + Send>;

/// Optional features the renderer and its caches use when the adapter has them.
pub fn wanted_features(available: &[FeatureName]) -> Vec<FeatureName> {
    let mut wanted = vec![
        FeatureName::TextureCompressionBC,
        FeatureName::TextureCompressionBCSliced3D,
        FeatureName::Float32Filterable,
        FeatureName::TextureFormatsTier1,
        FeatureName::Depth32FloatStencil8,
    ];
    #[cfg(not(target_os = "emscripten"))]
    wanted.push(xenos_texture_cache::NATIVE_TEXTURE_FORMAT_16BIT_NORM);
    if crate::debug::GPU_TIME.load(std::sync::atomic::Ordering::Relaxed) {
        wanted.push(FeatureName::TimestampQuery);
    }
    if crate::debug::NO_D32S8.load(std::sync::atomic::Ordering::Relaxed) {
        wanted.retain(|f| *f != FeatureName::Depth32FloatStencil8);
    }
    if crate::debug::NO_BC.load(std::sync::atomic::Ordering::Relaxed) {
        wanted.retain(|f| *f != FeatureName::TextureCompressionBC && *f != FeatureName::TextureCompressionBCSliced3D);
    }
    let has = |f| if available.contains(&f) { "yes" } else { "no" };
    log::info!(
        "renderer: texture compression BC {}, ETC2 {}, ASTC {}; depth32float-stencil8 {}, float32-filterable {}, \
         texture-formats-tier1 {}{}",
        has(FeatureName::TextureCompressionBC),
        has(FeatureName::TextureCompressionETC2),
        has(FeatureName::TextureCompressionASTC),
        has(FeatureName::Depth32FloatStencil8),
        has(FeatureName::Float32Filterable),
        has(FeatureName::TextureFormatsTier1),
        crate::debug::describe()
    );
    wanted.retain(|f| available.contains(f));
    wanted
}

pub(crate) struct Renderer {
    pub device: Device,
    pub executor: Executor,
    pub presenter: Presenter,
    /// Commands taken from the queue but not executed yet (waiting behind a resolve).
    backlog: std::collections::VecDeque<(u64, Command)>,
    /// A command whose resolve is in flight; it completes when the resolve does.
    blocked: Option<u64>,
    /// Frame count and WebGPU call counters at the last frame report.
    calls_at: (u64, gpu_webgpu::counters::Counters),
    /// Per swap executed since the host last took them: when the title submitted it, and how long
    /// it waited in the queue before the renderer got to it.
    pub swap_timings: Vec<(std::time::Instant, Duration)>,
}

impl Renderer {
    pub fn new(
        device: &Device,
        memory: Arc<dyn PhysicalMemory>,
        present_format: TextureFormat,
        targets: TargetsFactory,
    ) -> Self {
        let targets = targets(device);
        if crate::debug::GPU_TIME.load(std::sync::atomic::Ordering::Relaxed) {
            if gpu_webgpu::timing::enable(device) {
                log::info!("renderer: measuring GPU time per pass");
            } else {
                log::warn!("renderer: gpu-time: the device has no timestamp queries");
            }
        }
        Renderer {
            device: device.clone(),
            executor: Executor::new(device, memory, targets),
            presenter: Presenter::new(device, present_format),
            backlog: Default::default(),
            blocked: None,
            calls_at: (0, gpu_webgpu::counters::snapshot()),
            swap_timings: Vec::new(),
        }
    }

    #[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
    pub fn backlog_len(&self) -> usize {
        self.backlog.len()
    }

    /// Executes queued commands (waiting up to `timeout` for some if there are none) until the
    /// queue is drained or a resolve is in flight. `on_swap` runs for each swap after its frame
    /// has been submitted, with the front buffer size if it could be decoded.
    ///
    /// With `budget_ms` it returns once that much time has been spent even if more work is
    /// queued: on the web the thread must yield to its event loop regularly, or WebGPU calls
    /// (whose commands only reach the GPU process between tasks) eventually block.
    pub fn pump(
        &mut self,
        handle: &RendererHandle,
        timeout: Option<Duration>,
        budget_ms: Option<f64>,
        mut on_swap: impl FnMut(&mut Self, Option<(u32, u32)>),
    ) {
        if self.backlog.is_empty() {
            self.backlog = handle.take(timeout);
        }
        let start = now_ms();
        // Completion is reported once per batch (waiters are woken at most then), the time
        // budget is checked every few commands, and commands are timed on a sample: on the web
        // every clock read is a call out to JavaScript.
        let mut done = None;
        let mut executed = 0u64;
        while !self.executor.busy() {
            if executed.is_multiple_of(BUDGET_CHECK_INTERVAL) && budget_ms.is_some_and(|b| now_ms() - start >= b) {
                break;
            }
            if let Some(seq) = self.blocked.take() {
                done = Some(seq);
            }
            let Some((seq, cmd)) = self.backlog.pop_front() else {
                // More may have arrived meanwhile.
                self.backlog = handle.take(None);
                if self.backlog.is_empty() {
                    break;
                }
                continue;
            };
            executed += 1;
            let what: (&str, usize) = match &cmd {
                Command::Registers(w) => ("registers", w.len()),
                Command::Shader { code, .. } => ("shader", code.len()),
                Command::Draw => ("draw", 0),
                Command::Swap { .. } => ("swap", 0),
            };
            log::trace!("renderer: command {seq}: {what:?}");
            let timed = matches!(cmd, Command::Swap { .. }) || seq.is_multiple_of(TIMED_COMMAND_INTERVAL);
            let started = if timed { now_ms() } else { 0.0 };
            match cmd {
                Command::Registers(writes) => self.executor.write_registers(&writes),
                Command::Shader { stage, code } => self.executor.load_shader(stage, code),
                Command::Draw => self.executor.draw(),
                Command::Swap { fetch, .. } => {
                    if let Some(submitted) = handle.take_swap_time() {
                        self.swap_timings.push((submitted, submitted.elapsed()));
                    }
                    self.executor.end_frame();
                    let front = self.executor.front_buffer(fetch);
                    let size = self.presenter.swap(&self.device, front);
                    gpu_webgpu::timing::collect();
                    if self.presenter.frames == 1 || self.presenter.frames.is_multiple_of(300) {
                        log::info!(
                            "renderer: frame {}: {:?}; textures {:?}; shared memory {:?}; shaders: {:.0} ms translating, {:.0} ms creating modules, {:.0} ms creating pipelines",
                            self.presenter.frames,
                            self.executor.stats,
                            self.executor.texture_stats(),
                            self.executor.mirror_stats(),
                            self.executor.shader_times()[0].as_secs_f64() * 1000.0,
                            self.executor.shader_times()[1].as_secs_f64() * 1000.0,
                            self.executor.shader_times()[2].as_secs_f64() * 1000.0
                        );
                        let calls = gpu_webgpu::counters::snapshot();
                        let frames = self.presenter.frames - self.calls_at.0;
                        log::info!("renderer: per frame: {}", calls_per_frame(&calls.since(&self.calls_at.1), frames));
                        log::info!("renderer: {}", self.gpu_memory());
                        if let Some(times) = gpu_times(frames) {
                            log::info!("renderer: {times}");
                        }
                        self.calls_at = (self.presenter.frames, calls);
                    }
                    on_swap(self, size);
                }
            }
            if timed {
                let took = now_ms() - started;
                if took > 250.0 {
                    let stages = if what.0 == "draw" { self.executor.last_draw_timing() } else { String::new() };
                    log::warn!(
                        "renderer: command {seq} {what:?} took {took:.0} ms ({stages}); {:?}",
                        self.executor.stats
                    );
                }
            }
            // A draw that started a resolve completes once the resolve has.
            if self.executor.busy() {
                self.blocked = Some(seq);
            } else {
                done = Some(seq);
            }
        }
        if let Some(seq) = done {
            handle.complete(seq);
        }
    }
}

/// GPU time per frame by pass label since the last call (`gpu-time`), over `frames` frames.
pub(crate) fn gpu_times(frames: u64) -> Option<String> {
    if !gpu_webgpu::timing::enabled() {
        return None;
    }
    let (times, untimed) = gpu_webgpu::timing::take();
    let frames = frames.max(1) as f64;
    let total: u64 = times.iter().map(|(_, t)| t.nanos).sum();
    let parts: Vec<String> = times
        .iter()
        .map(|(label, t)| {
            format!("{label} {:.2} ms ({:.0} passes)", t.nanos as f64 / 1e6 / frames, t.passes as f64 / frames)
        })
        .collect();
    Some(format!(
        "GPU time per frame {:.2} ms: {}{}",
        total as f64 / 1e6 / frames,
        parts.join(", "),
        if untimed > 0 { format!("; {untimed} passes untimed") } else { String::new() }
    ))
}

impl Renderer {
    /// GPU memory held: buffers and textures not yet destroyed (largest groups by label), and
    /// the texture cache's share.
    pub(crate) fn gpu_memory(&self) -> String {
        let groups = gpu_webgpu::memory::live_by_label();
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        let total: u64 = groups.iter().map(|g| g.bytes).sum();
        let top: Vec<String> = groups
            .iter()
            .take(8)
            .map(|g| format!("{} \"{}\" {:.1} MiB ({})", g.kind, g.label, mib(g.bytes), g.count))
            .collect();
        let cache = self.executor.texture_stats();
        let calls = gpu_webgpu::counters::snapshot();
        format!(
            "GPU memory {:.1} MiB in {} objects; texture cache {:.1} MiB in {} textures; {} shader modules \
             ({:.1} MiB of WGSL), {} pipelines; largest: {}",
            mib(total),
            groups.iter().map(|g| g.count).sum::<u64>(),
            mib(cache.host_bytes),
            cache.textures,
            calls.shader_modules,
            mib(calls.shader_bytes),
            calls.pipelines,
            top.join(", ")
        )
    }
}

/// Frame pacing since the last call, for the perf log: intervals between frames shown
/// (`presents`, which is cleared), between the title's swaps, and how long swaps waited in the
/// queue before the renderer executed them. Stutter shows as a high p95/max next to a normal
/// median; latency as a long queue wait.
pub(crate) fn pacing(
    presents: &mut Vec<std::time::Instant>,
    swaps: &mut Vec<(std::time::Instant, Duration)>,
) -> String {
    fn intervals(times: impl Iterator<Item = std::time::Instant>) -> Vec<f64> {
        let times: Vec<_> = times.collect();
        let mut v: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]).as_secs_f64() * 1000.0).collect();
        v.sort_by(f64::total_cmp);
        v
    }
    fn describe(v: &[f64]) -> String {
        if v.is_empty() {
            return "-".into();
        }
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        let hitches = v.iter().filter(|&&ms| ms > 50.0).count();
        format!("p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms, {hitches} over 50 ms", at(0.5), at(0.95), at(1.0))
    }
    let shown = intervals(presents.drain(..));
    let issued = intervals(swaps.iter().map(|s| s.0));
    let waits: Vec<f64> = swaps.drain(..).map(|s| s.1.as_secs_f64() * 1000.0).collect();
    let wait_avg = waits.iter().sum::<f64>() / waits.len().max(1) as f64;
    let wait_max = waits.iter().copied().fold(0.0, f64::max);
    format!(
        "frames shown: {}; title's swaps: {}; swap queue wait avg {wait_avg:.1} ms, max {wait_max:.1} ms",
        describe(&shown),
        describe(&issued)
    )
}

/// Commands between checks of the pump's time budget.
const BUDGET_CHECK_INTERVAL: u64 = 16;
/// One command in this many is timed for the slow-command warning (and every swap).
const TIMED_COMMAND_INTERVAL: u64 = 64;

/// WebGPU operations per frame over `frames` frames, for the perf log.
pub(crate) fn calls_per_frame(c: &gpu_webgpu::counters::Counters, frames: u64) -> String {
    let f = frames.max(1);
    let by_label: Vec<String> = gpu_webgpu::counters::take_created_by_label()
        .into_iter()
        .take(6)
        .map(|(kind, label, n)| format!("{kind} \"{label}\" {}", n / f))
        .collect();
    let uploads: Vec<String> = gpu_webgpu::counters::take_written_by_label()
        .into_iter()
        .take(6)
        .map(|(label, calls, bytes)| format!("\"{label}\" {}x {} KiB", calls / f, bytes / f / 1024))
        .collect();
    format!(
        "{} draws, {} set_pipeline, {} set_bind_group, {} render passes, {} compute passes ({} dispatches), \
         {} submits, {} write_buffer ({} KiB), {} write_texture, {} copies; created: {} buffers, {} textures, \
         {} views, {} bind groups, {} samplers, {} pipelines (most: {}); uploads: {}",
        c.draws / f,
        c.set_pipelines / f,
        c.set_bind_groups / f,
        c.render_passes / f,
        c.compute_passes / f,
        c.dispatches / f,
        c.submits / f,
        c.write_buffers / f,
        c.write_buffer_bytes / f / 1024,
        c.write_textures / f,
        c.copies / f,
        c.buffers / f,
        c.textures / f,
        c.views / f,
        c.bind_groups / f,
        c.samplers / f,
        c.pipelines / f,
        by_label.join(", "),
        uploads.join(", "),
    )
}
