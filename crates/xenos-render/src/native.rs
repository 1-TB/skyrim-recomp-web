//! Native renderer thread. Presents each swap in a window when the host gives it one, and
//! otherwise renders offscreen; either way frames can be written out as PNG files.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use gpu_webgpu::*;

use crate::renderer::{Renderer, TargetsFactory, wanted_features};
use crate::{PhysicalMemory, RendererHandle};

#[derive(Debug, Clone, Default)]
pub struct NativeOptions {
    /// Write every `dump_every`-th presented frame to `<dir>/frame_NNNNNN.png`.
    pub dump_dir: Option<PathBuf>,
    pub dump_every: u64,
    /// Present in this window.
    pub window: Option<Window>,
}

/// A window to present in: its window-system handles (addresses, so the renderer thread can
/// take them) and its size in pixels, which the host updates when the window is resized.
#[derive(Debug, Clone)]
pub struct Window {
    pub handles: WindowHandles,
    /// Width in the high 32 bits, height in the low ones.
    pub size: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Copy)]
pub enum WindowHandles {
    Xlib {
        display: usize,
        window: u64,
    },
    Xcb {
        connection: usize,
        window: u32,
    },
    Wayland {
        display: usize,
        surface: usize,
    },
    /// A `CAMetalLayer` (macOS, iOS) that the host keeps alive and sized to its view.
    Metal {
        layer: usize,
    },
}

impl Window {
    pub fn pack_size(width: u32, height: u32) -> u64 {
        (u64::from(width) << 32) | u64::from(height)
    }

    fn size(&self) -> (u32, u32) {
        let v = self.size.load(Ordering::Relaxed);
        ((v >> 32) as u32, v as u32)
    }
}

/// The window's surface and how it is configured.
struct Presentation {
    window: Window,
    surface: Surface,
    format: TextureFormat,
    present_mode: PresentMode,
    configured: (u32, u32),
}

impl Presentation {
    fn configure(&mut self, device: &Device) {
        let (width, height) = self.window.size();
        self.configured = (width.max(1), height.max(1));
        self.surface.configure(
            device,
            &SurfaceConfiguration {
                format: self.format,
                usage: TextureUsage::RenderAttachment,
                width: self.configured.0,
                height: self.configured.1,
                view_formats: &[],
                alpha_mode: CompositeAlphaMode::Opaque,
                present_mode: self.present_mode,
            },
        );
    }

    /// Shows the presenter's current frame, reconfiguring first if the window was resized.
    fn present(&mut self, device: &Device, presenter: &crate::present::Presenter) {
        let size = self.window.size();
        if size.0 == 0 || size.1 == 0 {
            return; // Minimized.
        }
        if size != self.configured {
            self.configure(device);
        }
        let frame = match self.surface.current_frame() {
            Ok(frame) => frame,
            Err(Error::SurfaceStatus(_)) => {
                // Outdated or lost (e.g. mid-resize): reconfigure and show the next frame.
                self.configure(device);
                return;
            }
            Err(e) => {
                log::warn!("renderer: no window frame: {e}");
                return;
            }
        };
        let view = frame.texture.create_view(None);
        let mut encoder = device.create_command_encoder("present");
        presenter.encode(device, &mut encoder, &view, self.configured.0, self.configured.1);
        device.queue().submit([encoder.finish("present")]);
        frame.present();
    }
}

static PAUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Stops (or restarts) the renderer's GPU work: iOS aborts command buffers submitted while an
/// app is in the background, and a lost device can't be recovered. The renderer notices within
/// one pump (a few milliseconds).
pub fn set_paused(paused: bool) {
    PAUSED.store(paused, Ordering::Relaxed);
}

/// Starts the renderer thread. If no WebGPU device can be created the error is logged and
/// the handle never blocks (submitted work is dropped).
pub fn start(
    memory: Arc<dyn PhysicalMemory>,
    options: NativeOptions,
    targets: TargetsFactory,
) -> std::io::Result<RendererHandle> {
    let handle = RendererHandle::new();
    let renderer = handle.clone();
    std::thread::Builder::new().name("renderer".into()).spawn(move || {
        // As the title's threads (runtime_core::thread::prefer_performance_cores).
        #[cfg(target_vendor = "apple")]
        // SAFETY: changes only this thread's scheduling class.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
        }
        if let Err(e) = run(&renderer, memory, &options, targets) {
            log::error!("renderer stopped: {e}");
        }
        renderer.close();
    })?;
    Ok(handle)
}

struct Offscreen {
    width: u32,
    height: u32,
    texture: Texture,
    view: TextureView,
    readback: Buffer,
    row: u32,
    /// RGBA8 or BGRA8: the presenter's pipeline is built for the window's format.
    format: TextureFormat,
}

impl Offscreen {
    fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: "present target",
            size: extent(width, height, 1),
            format,
            usage: TextureUsage::RenderAttachment | TextureUsage::CopySrc,
            ..Default::default()
        });
        let view = texture.create_view(None);
        let row = (width * 4).next_multiple_of(256);
        let readback = device.create_buffer(&BufferDescriptor {
            label: "present readback",
            size: u64::from(row * height),
            usage: BufferUsage::MapRead | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        Offscreen { width, height, texture, view, readback, row, format }
    }
}

fn run(
    handle: &RendererHandle,
    memory: Arc<dyn PhysicalMemory>,
    options: &NativeOptions,
    targets: TargetsFactory,
) -> Result<()> {
    let instance = Instance::new()?;
    let surface = match &options.window {
        // SAFETY: the host keeps the window open for the life of the process.
        Some(window) => Some(unsafe {
            instance.create_surface(match window.handles {
                WindowHandles::Xlib { display, window } => SurfaceTarget::Xlib { display: display as *mut _, window },
                WindowHandles::Xcb { connection, window } => {
                    SurfaceTarget::Xcb { connection: connection as *mut _, window }
                }
                WindowHandles::Wayland { display, surface } => {
                    SurfaceTarget::Wayland { display: display as *mut _, surface: surface as *mut _ }
                }
                WindowHandles::Metal { layer } => SurfaceTarget::Metal { layer: layer as *mut _ },
            })?
        }),
        None => None,
    };
    let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: surface.as_ref(),
        ..Default::default()
    }))?;
    let info = adapter.info();
    log::info!("renderer: {:?} {} ({})", info.backend, info.description, implementation());
    // Xenos programs need more than WebGPU's default limits (e.g. 16 interpolators plus
    // front_facing), so take everything the adapter offers.
    let features = wanted_features(&adapter.features());
    let device = instance.block_on(adapter.request_device(&DeviceDescriptor {
        label: "xenos",
        required_features: &features,
        required_limits: Some(adapter.limits()),
    }))?;
    // RGBA8 (what the PNG dumps read back) when the window takes it.
    let mut presentation = match (surface, &options.window) {
        (Some(surface), Some(window)) => {
            let caps = surface.capabilities(&adapter)?;
            let format = [TextureFormat::RGBA8Unorm, TextureFormat::BGRA8Unorm]
                .into_iter()
                .find(|f| caps.formats.contains(f))
                .or(caps.formats.first().copied())
                .ok_or(Error::Surface("the window offers no formats".into()))?;
            // Mailbox never blocks the renderer on the display (FIFO would, and stalls for good
            // on Wayland while the window is hidden).
            let present_mode = if caps.present_modes.contains(&PresentMode::Mailbox) {
                PresentMode::Mailbox
            } else {
                PresentMode::Fifo
            };
            log::info!("renderer: presenting in a window as {format:?}, {present_mode:?}");
            let mut p = Presentation { window: window.clone(), surface, format, present_mode, configured: (0, 0) };
            p.configure(&device);
            Some(p)
        }
        _ => None,
    };
    let format = presentation.as_ref().map_or(TextureFormat::RGBA8Unorm, |p| p.format);
    let mut renderer = Renderer::new(&device, memory, format, targets);
    let mut offscreen: Option<Offscreen> = None;
    // Frames are written from an offscreen copy in the window's format, so the presenter's
    // pipeline fits it; PNGs need 8-bit RGBA or BGRA (Metal layers are BGRA).
    let dumpable = matches!(format, TextureFormat::RGBA8Unorm | TextureFormat::BGRA8Unorm);
    if options.dump_dir.is_some() && !dumpable {
        log::warn!("renderer: the window is {format:?}; frames are not dumped");
    }
    if let Some(dir) = &options.dump_dir {
        std::fs::create_dir_all(dir).map_err(|e| Error::Invalid(format!("{}: {e}", dir.display())))?;
    }
    let every = options.dump_every.max(1);
    let mut presents = Vec::new();
    let mut busy = Duration::ZERO;
    let mut report_at = std::time::Instant::now();
    let mut report_calls = gpu_webgpu::counters::snapshot();
    let mut report_times = renderer.executor.times();
    let mut report_draws = renderer.executor.stats.draws;

    loop {
        if PAUSED.load(Ordering::Relaxed) {
            // Nothing is submitted while paused; a title waiting on the renderer waits too.
            std::thread::park_timeout(Duration::from_millis(50));
            continue;
        }
        instance.process_events();
        let started = std::time::Instant::now();
        let before = handle.completed();
        renderer.pump(handle, Some(Duration::from_millis(5)), None, |r, size| {
            let Some((w, h)) = size else { return };
            if let Some(p) = presentation.as_mut() {
                p.present(&r.device, &r.presenter);
            }
            presents.push(std::time::Instant::now());
            let Some(dir) = &options.dump_dir else { return };
            if !dumpable {
                return;
            }
            if !r.presenter.frames.is_multiple_of(every) {
                return;
            }
            let target = match offscreen.take() {
                Some(o) if o.width == w && o.height == h => o,
                _ => Offscreen::new(&r.device, w, h, format),
            };
            let path = dir.join(format!("frame_{:06}.png", r.presenter.frames));
            match dump(&r.device, &r.presenter, &target, &path) {
                Ok(()) => log::info!("renderer: wrote {}", path.display()),
                Err(e) => log::error!("renderer: {}: {e}", path.display()),
            }
            offscreen = Some(target);
        });
        for e in device.take_errors() {
            log::error!("renderer: GPU error: {e:?}");
        }
        // Time spent executing commands (not waiting for the title to submit some, nor for the
        // GPU to free constant buffers).
        if handle.completed() != before || renderer.backlog_len() > 0 {
            busy += started.elapsed().saturating_sub(std::mem::take(&mut renderer.executor.floats.waited));
        }
        renderer.executor.floats.waited = Duration::ZERO;
        // Waiting for the GPU (frames in flight, a resolve, uploads): block until it gets
        // somewhere rather than spinning through pump (which kept a core busy reading clocks).
        if renderer.executor.busy() {
            device.wait_for_gpu_progress();
        }
        let window = report_at.elapsed();
        if window >= Duration::from_secs(5) {
            let frames = presents.len() as u64;
            let fps = frames as f64 / window.as_secs_f64();
            let draws = renderer.executor.stats.draws;
            log::info!(
                "perf: {fps:.1} fps, {} draws/frame; renderer busy {:.0}%; {}",
                (draws - report_draws) / frames.max(1),
                100.0 * busy.as_secs_f64() / window.as_secs_f64(),
                crate::renderer::pacing(&mut presents, &mut renderer.swap_timings)
            );
            let calls = gpu_webgpu::counters::snapshot();
            log::info!("perf: per frame: {}", crate::renderer::calls_per_frame(&calls.since(&report_calls), frames));
            let times = renderer.executor.times();
            let ms = |now: Duration, then: Duration| (now - then).as_secs_f64() * 1000.0;
            log::info!(
                "perf: renderer time: {:.0} ms binding new texture sets, {:.0} ms creating pipelines, {:.0} ms on \
                 shaders, {:.0} ms recording resolves, {:.0} ms submitting",
                ms(times.texture_sets, report_times.texture_sets),
                ms(times.pipelines, report_times.pipelines),
                ms(times.shaders, report_times.shaders),
                ms(times.resolves, report_times.resolves),
                ms(times.submits, report_times.submits),
            );
            report_calls = calls;
            report_times = times;
            report_draws = draws;
            busy = Duration::ZERO;
            report_at = std::time::Instant::now();
        }
    }
}

/// Renders the current front buffer into `target`, reads it back and writes a PNG.
fn dump(device: &Device, presenter: &crate::present::Presenter, target: &Offscreen, path: &Path) -> Result<()> {
    let mut encoder = device.create_command_encoder("dump");
    presenter.encode(device, &mut encoder, &target.view, target.width, target.height);
    encoder.copy_texture_to_buffer(
        &TexelCopyTextureInfo::whole(&target.texture),
        &TexelCopyBufferInfo { buffer: &target.readback, offset: 0, bytes_per_row: target.row, rows_per_image: None },
        extent(target.width, target.height, 1),
    );
    device.queue().submit([encoder.finish("dump")]);
    device.block_on(target.readback.map_async(MapMode::Read, 0, None))?;
    let mut padded = vec![0u8; (target.row * target.height) as usize];
    target.readback.read_mapped(0, &mut padded)?;
    target.readback.unmap();
    let mut rgba: Vec<u8> =
        padded.chunks(target.row as usize).flat_map(|r| &r[..target.width as usize * 4]).copied().collect();
    if target.format == TextureFormat::BGRA8Unorm {
        rgba.chunks_exact_mut(4).for_each(|p| p.swap(0, 2));
    }
    write_png(path, target.width, target.height, &rgba).map_err(|e| Error::Invalid(e.to_string()))
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(std::io::Error::other)?;
    w.write_image_data(rgba).map_err(std::io::Error::other)
}

const VIEW_DUMP: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

/// Debug aid (`--render-debug trace-frame`): writes the first `width` x `height` texels of a
/// single-sampled color view as an opaque PNG (channels clamped to 0..1). Blocks until done.
pub(crate) fn dump_view(device: &Device, view: &TextureView, width: u32, height: u32, path: &Path) -> Result<()> {
    let module = device.create_shader_module_wgsl("view dump", VIEW_DUMP);
    let layout = device.create_bind_group_layout(
        "view dump",
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStage::Fragment,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::UnfilterableFloat,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
        }],
    );
    let pipeline_layout = device.create_pipeline_layout("view dump", &[&layout]);
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: "view dump",
        layout: Some(&pipeline_layout),
        vertex: ProgrammableStage::new(&module, "vs"),
        vertex_buffers: &[],
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            stage: ProgrammableStage::new(&module, "fs"),
            targets: &[ColorTargetState::new(TextureFormat::RGBA8Unorm)],
        }),
    });
    let group = device.create_bind_group(
        "view dump",
        &layout,
        &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(view) }],
    );
    let target = Offscreen::new(device, width, height, TextureFormat::RGBA8Unorm);
    let mut encoder = device.create_command_encoder("view dump");
    let mut pass = encoder.begin_render_pass(
        "view dump",
        &[Some(RenderPassColorAttachment::clear(&target.view, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }))],
        None,
    );
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.draw(3, 1, 0, 0);
    pass.end();
    encoder.copy_texture_to_buffer(
        &TexelCopyTextureInfo::whole(&target.texture),
        &TexelCopyBufferInfo { buffer: &target.readback, offset: 0, bytes_per_row: target.row, rows_per_image: None },
        extent(width, height, 1),
    );
    device.queue().submit([encoder.finish("view dump")]);
    device.block_on(target.readback.map_async(MapMode::Read, 0, None))?;
    let mut padded = vec![0u8; (target.row * height) as usize];
    target.readback.read_mapped(0, &mut padded)?;
    target.readback.unmap();
    let rgba: Vec<u8> = padded.chunks(target.row as usize).flat_map(|r| &r[..width as usize * 4]).copied().collect();
    write_png(path, width, height, &rgba).map_err(|e| Error::Invalid(e.to_string()))
}
