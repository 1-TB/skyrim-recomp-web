//! Native renderer thread: headless (the debug host has no window). Frames are rendered
//! offscreen and, when asked, written out as PNG files.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpu_webgpu::*;

use crate::renderer::{Renderer, TargetsFactory, wanted_features};
use crate::{PhysicalMemory, RendererHandle};

#[derive(Debug, Clone, Default)]
pub struct NativeOptions {
    /// Write every `dump_every`-th presented frame to `<dir>/frame_NNNNNN.png`.
    pub dump_dir: Option<PathBuf>,
    pub dump_every: u64,
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
}

impl Offscreen {
    fn new(device: &Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: "present target",
            size: extent(width, height, 1),
            format: TextureFormat::RGBA8Unorm,
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
        Offscreen { width, height, texture, view, readback, row }
    }
}

fn run(
    handle: &RendererHandle,
    memory: Arc<dyn PhysicalMemory>,
    options: &NativeOptions,
    targets: TargetsFactory,
) -> Result<()> {
    let instance = Instance::new()?;
    let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions::default()))?;
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
    let mut renderer = Renderer::new(&device, memory, TextureFormat::RGBA8Unorm, targets);
    let mut offscreen: Option<Offscreen> = None;
    if let Some(dir) = &options.dump_dir {
        std::fs::create_dir_all(dir).map_err(|e| Error::Invalid(format!("{}: {e}", dir.display())))?;
    }
    let every = options.dump_every.max(1);

    loop {
        instance.process_events();
        renderer.pump(handle, Some(Duration::from_millis(5)), None, |r, size| {
            let Some((w, h)) = size else { return };
            let Some(dir) = &options.dump_dir else { return };
            if !r.presenter.frames.is_multiple_of(every) {
                return;
            }
            let target = match offscreen.take() {
                Some(o) if o.width == w && o.height == h => o,
                _ => Offscreen::new(&r.device, w, h),
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
    }
}

/// Renders the current front buffer into `target`, reads it back and writes a PNG.
fn dump(device: &Device, presenter: &crate::present::Presenter, target: &Offscreen, path: &Path) -> Result<()> {
    let mut encoder = device.create_command_encoder("dump");
    presenter.encode(&mut encoder, &target.view, target.width, target.height);
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
    let rgba: Vec<u8> =
        padded.chunks(target.row as usize).flat_map(|r| &r[..target.width as usize * 4]).copied().collect();
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
    let target = Offscreen::new(device, width, height);
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
