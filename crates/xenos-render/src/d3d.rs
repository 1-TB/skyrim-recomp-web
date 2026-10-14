//! `--gpu d3d`: rendering from the title's D3D calls instead of the command stream (see the
//! `d3d-hle` crate). Each D3D surface is a host texture of its own (no EDRAM), clears clear it,
//! a resolve copies it into a host texture standing for the destination texture, and a present
//! shows the texture the frame was resolved to.
//!
//! Draws are not executed yet, so the picture shows every color surface the frame used side by
//! side (each pass's clear color); `--render-debug d3d-front` shows the front buffer instead.

use std::collections::HashMap;

use d3d_hle::args::{ClearArgs, ResolveArgs, TextureHeader, clear, resolve};
use d3d_hle::command::{Command, Surface};
use gpu_webgpu::*;
use xenos_state::xenos::ColorRenderTargetFormat;

use crate::present::FrontBuffer;

/// A surface's host texture.
struct HostSurface {
    texture: Texture,
    view: TextureView,
    format: TextureFormat,
    width: u32,
    height: u32,
    depth: bool,
    /// D24FS8: the host depth holds half the guest value (as the executor's viewport maps it).
    float24: bool,
    /// The surface header it was made for (a different one replaces it).
    header: [u32; d3d_hle::args::SURFACE_WORDS],
}

/// A resolve destination's host texture.
struct HostTexture {
    texture: Texture,
    view: TextureView,
    format: TextureFormat,
    width: u32,
    height: u32,
}

/// A surface's key: its object address, and whether it is bound as depth/stencil (the title
/// binds some surface objects both ways).
type SurfaceKey = (u32, bool);

pub(crate) struct D3dRenderer {
    surfaces: HashMap<SurfaceKey, HostSurface>,
    /// Bound render targets 0-3 and depth/stencil surface.
    targets: [Option<SurfaceKey>; 4],
    depth: Option<SurfaceKey>,
    /// Resolve destinations by physical base address.
    textures: HashMap<u32, HostTexture>,
    depth32_stencil8: bool,
    warned: std::collections::HashSet<String>,
    /// Color surfaces used this frame, in order of first use (for the mosaic).
    used: Vec<SurfaceKey>,
    /// What the frame did (logged for one frame in [`LOG_PERIOD`]).
    frame_log: Vec<String>,
    presents: u64,
    mosaic: Option<Mosaic>,
}

/// Presents between per-frame operation logs.
const LOG_PERIOD: u64 = 600;
/// Size of the surface mosaic.
const MOSAIC_SIZE: (u32, u32) = (1280, 720);

const MOSAIC_SHADER: &str = r#"
@group(0) @binding(0) var surface: texture_2d<f32>;
@group(0) @binding(1) var<uniform> rect: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(surface));
    let p = clamp((pos.xy - rect.xy) / rect.zw, vec2<f32>(0.0), vec2<f32>(0.999));
    let c = textureLoad(surface, vec2<i32>(p * size), 0);
    return vec4<f32>(c.rgb, 1.0);
}
"#;

/// The debug view of all color surfaces of a frame.
struct Mosaic {
    /// Kept for the view's lifetime.
    _texture: Texture,
    view: TextureView,
    pipeline: RenderPipeline,
    layout: BindGroupLayout,
}

impl Mosaic {
    fn new(device: &Device) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: "d3d surface mosaic",
            size: extent(MOSAIC_SIZE.0, MOSAIC_SIZE.1, 1),
            format: TextureFormat::RGBA8Unorm,
            usage: TextureUsage::RenderAttachment | TextureUsage::TextureBinding,
            ..Default::default()
        });
        let view = texture.create_view(None);
        let layout = device.create_bind_group_layout(
            "d3d surface mosaic",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::Fragment,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::UnfilterableFloat,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::Fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 16,
                    },
                },
            ],
        );
        let pipeline_layout = device.create_pipeline_layout("d3d surface mosaic", &[&layout]);
        let module = device.create_shader_module_wgsl("d3d surface mosaic", MOSAIC_SHADER);
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "d3d surface mosaic",
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
        Mosaic { _texture: texture, view, pipeline, layout }
    }
}

impl D3dRenderer {
    pub fn new(device: &Device) -> Self {
        D3dRenderer {
            surfaces: HashMap::new(),
            targets: [None; 4],
            depth: None,
            textures: HashMap::new(),
            depth32_stencil8: device.has_feature(FeatureName::Depth32FloatStencil8),
            warned: Default::default(),
            used: Vec::new(),
            frame_log: Vec::new(),
            presents: 0,
            mosaic: None,
        }
    }

    fn logging(&self) -> bool {
        self.presents.is_multiple_of(LOG_PERIOD)
    }

    fn note(&mut self, op: impl FnOnce() -> String) {
        if self.logging() {
            self.frame_log.push(op());
        }
    }

    /// Records that a color surface was used this frame (for the mosaic).
    fn use_surface(&mut self, key: Option<SurfaceKey>) {
        if let Some(key) = key.filter(|k| !k.1)
            && !self.used.contains(&key)
        {
            self.used.push(key);
        }
    }

    /// Draws the frame's color surfaces into the mosaic and returns it as the front buffer.
    fn mosaic(&mut self, device: &Device) -> Option<FrontBuffer> {
        let mosaic = self.mosaic.get_or_insert_with(|| Mosaic::new(device));
        let surfaces: Vec<&HostSurface> = self.used.iter().filter_map(|k| self.surfaces.get(k)).collect();
        let n = surfaces.len().max(1) as u32;
        let cols = (n as f32).sqrt().ceil() as u32;
        let rows = n.div_ceil(cols);
        let (tw, th) = (MOSAIC_SIZE.0 / cols, MOSAIC_SIZE.1 / rows);
        let mut encoder = device.create_command_encoder("d3d surface mosaic");
        let mut buffers = Vec::new();
        {
            let black = Color { r: 0.1, g: 0.0, b: 0.1, a: 1.0 };
            let mut pass = encoder.begin_render_pass(
                "d3d surface mosaic",
                &[Some(RenderPassColorAttachment::clear(&mosaic.view, black))],
                None,
            );
            pass.set_pipeline(&mosaic.pipeline);
            for (i, s) in surfaces.iter().enumerate() {
                let (x, y) = ((i as u32 % cols) * tw, (i as u32 / cols) * th);
                // Each tile keeps its surface's aspect, with a 2-pixel gap.
                let scale = ((tw - 4) as f32 / s.width as f32).min((th - 4) as f32 / s.height as f32);
                let (w, h) = ((s.width as f32 * scale).max(1.0), (s.height as f32 * scale).max(1.0));
                let rect = [x as f32 + 2.0, y as f32 + 2.0, w, h];
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: "d3d surface mosaic tile",
                    size: 16,
                    usage: BufferUsage::Uniform | BufferUsage::CopyDst,
                    mapped_at_creation: false,
                });
                let bytes: Vec<u8> = rect.iter().flat_map(|v| v.to_le_bytes()).collect();
                device.queue().write_buffer(&buffer, 0, &bytes);
                let group = device.create_bind_group(
                    "d3d surface mosaic tile",
                    &mosaic.layout,
                    &[
                        BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&s.view) },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer { buffer: &buffer, offset: 0, size: None },
                        },
                    ],
                );
                pass.set_bind_group(0, &group, &[]);
                pass.set_viewport(rect[0], rect[1], w, h, 0.0, 1.0);
                pass.draw(3, 1, 0, 0);
                buffers.push(buffer);
            }
            pass.end();
        }
        device.queue().submit([encoder.finish("d3d surface mosaic")]);
        for b in buffers {
            b.destroy();
        }
        Some(FrontBuffer {
            view: mosaic.view.clone(),
            filterable: true,
            width: MOSAIC_SIZE.0,
            height: MOSAIC_SIZE.1,
            swizzle: [0, 1, 2, 3],
        })
    }

    fn warn_once(&mut self, message: String) {
        if self.warned.len() < 256 && self.warned.insert(message.clone()) {
            log::warn!("d3d renderer: {message}");
        }
    }

    /// Executes `cmd`; a present returns the front buffer to show.
    pub fn execute(&mut self, device: &Device, cmd: Command) -> Option<FrontBuffer> {
        match cmd {
            Command::SetRenderTarget { index, surface } => {
                let key = surface.map(|s| self.surface(device, &s));
                if let Some(slot) = self.targets.get_mut(index as usize) {
                    *slot = key;
                }
                self.note(|| format!("SetRenderTarget {index} {key:X?}"));
            }
            Command::SetDepthStencil { surface } => {
                self.depth = surface.map(|s| self.surface(device, &s));
                let key = self.depth;
                self.note(|| format!("SetDepthStencilSurface {key:X?}"));
            }
            Command::Clear(args) => {
                for i in 0..4 {
                    if args.flags & (1 << i) != 0 {
                        self.use_surface(self.targets[i]);
                    }
                }
                self.note(|| {
                    format!(
                        "Clear flags {:#X} color {:#010X} z {} stencil {}",
                        args.flags, args.color, args.z, args.stencil
                    )
                });
                self.clear(device, &args);
            }
            Command::Resolve { args, dest } => {
                self.use_surface(args.source_target().and_then(|i| self.targets[i as usize]));
                self.note(|| {
                    format!(
                        "Resolve flags {:#X} rect {:?} to {:#010X}",
                        args.flags,
                        args.source_rect,
                        dest.map_or(0, |d| d.base_address())
                    )
                });
                self.resolve(device, &args, dest.as_ref());
            }
            Command::Present { front } => {
                let shown = if crate::debug::D3D_FRONT.load(std::sync::atomic::Ordering::Relaxed) {
                    self.front_buffer(front.as_ref())
                } else {
                    self.mosaic(device)
                };
                if self.logging() {
                    log::info!(
                        "d3d renderer: frame {}: {} (front buffer {})",
                        self.presents,
                        self.frame_log.join("; "),
                        if shown.is_some() { "shown" } else { "not resolved yet" }
                    );
                }
                self.frame_log.clear();
                self.used.clear();
                self.presents += 1;
                return shown;
            }
        }
        None
    }

    /// The host texture for `surface` (created, or replaced when its header changed); returns
    /// its key.
    fn surface(&mut self, device: &Device, s: &Surface) -> SurfaceKey {
        let key = (s.address, s.depth);
        if self.surfaces.get(&key).is_some_and(|h| h.header == s.header.words) {
            return key;
        }
        let format = if s.depth {
            if self.depth32_stencil8 { TextureFormat::Depth32FloatStencil8 } else { TextureFormat::Depth24PlusStencil8 }
        } else {
            let guest = ColorRenderTargetFormat::from_raw(s.format());
            xenos_edram::host_color_format(guest).unwrap_or_else(|| {
                self.warn_once(format!("render target format {guest:?} not supported; using rgba8"));
                TextureFormat::RGBA8Unorm
            })
        };
        let (width, height) = (s.width(), s.height());
        let texture = device.create_texture(&TextureDescriptor {
            label: if s.depth { "d3d depth surface" } else { "d3d color surface" },
            size: extent(width, height, 1),
            format,
            usage: TextureUsage::RenderAttachment | TextureUsage::TextureBinding | TextureUsage::CopySrc,
            ..Default::default()
        });
        let view = texture.create_view(None);
        log::debug!("d3d renderer: surface {:#010X} {width}x{height} {format:?}", s.address);
        if let Some(old) = self.surfaces.insert(
            key,
            HostSurface {
                texture,
                view,
                format,
                width,
                height,
                depth: s.depth,
                float24: s.depth && s.format() & 1 != 0,
                header: s.header.words,
            },
        ) {
            old.texture.destroy();
        }
        key
    }

    /// Clears the bound surfaces the flags name (the whole surface: the title's clear rectangles
    /// cover the viewport).
    fn clear(&mut self, device: &Device, args: &ClearArgs) {
        let rgba = args.rgba().map(f64::from);
        let color = Color { r: rgba[0], g: rgba[1], b: rgba[2], a: rgba[3] };
        let mut encoder = device.create_command_encoder("d3d clear");
        for (i, target) in self.targets.iter().enumerate() {
            if args.flags & (1 << i) == 0 {
                continue;
            }
            if let Some(s) = target.and_then(|t| self.surfaces.get(&t)) {
                let pass = encoder.begin_render_pass(
                    "d3d clear",
                    &[Some(RenderPassColorAttachment::clear(&s.view, color))],
                    None,
                );
                pass.end();
            }
        }
        if args.flags & (clear::ZBUFFER | clear::STENCIL) != 0
            && let Some(s) = self.depth.and_then(|d| self.surfaces.get(&d))
        {
            let keep_depth = args.flags & clear::ZBUFFER == 0;
            let keep_stencil = args.flags & clear::STENCIL == 0;
            let depth = RenderPassDepthStencilAttachment {
                view: &s.view,
                depth_load_op: if keep_depth { LoadOp::Load } else { LoadOp::Clear },
                depth_store_op: StoreOp::Store,
                depth_clear_value: if s.float24 { args.z.clamp(0.0, 2.0) * 0.5 } else { args.z.clamp(0.0, 1.0) },
                depth_read_only: false,
                stencil_load_op: if keep_stencil { LoadOp::Load } else { LoadOp::Clear },
                stencil_store_op: StoreOp::Store,
                stencil_clear_value: args.stencil & 0xFF,
                stencil_read_only: false,
            };
            let pass = encoder.begin_render_pass("d3d clear", &[], Some(&depth));
            pass.end();
        }
        device.queue().submit([encoder.finish("d3d clear")]);
    }

    /// Copies the source surface into the destination's host texture, then performs the
    /// requested clears.
    fn resolve(&mut self, device: &Device, args: &ResolveArgs, dest: Option<&TextureHeader>) {
        let source = match args.source_target() {
            Some(i) => self.targets[i as usize],
            None => self.depth,
        };
        if let (Some(dest), Some(source)) = (dest, source) {
            let fetch = xenos_texture::TextureFetchConstant::decode(dest.fetch);
            let base = dest.base_address() & 0x1FFF_FFFF;
            let Some(s) = self.surfaces.get(&source) else { return };
            if s.depth {
                // Depth resolves feed shadows and the like: nothing to show yet.
                return;
            }
            let (format, sw, sh) = (s.format, s.width, s.height);
            let (width, height) = (fetch.width.max(1), fetch.height.max(1));
            let fits =
                self.textures.get(&base).is_some_and(|t| t.format == format && t.width == width && t.height == height);
            if !fits {
                let texture = device.create_texture(&TextureDescriptor {
                    label: "d3d resolve destination",
                    size: extent(width, height, 1),
                    format,
                    usage: TextureUsage::TextureBinding | TextureUsage::CopyDst,
                    ..Default::default()
                });
                let view = texture.create_view(None);
                if let Some(old) = self.textures.insert(base, HostTexture { texture, view, format, width, height }) {
                    old.texture.destroy();
                }
            }
            let [x0, y0, x1, y1] = args.source_rect.unwrap_or([0, 0, sw as i32, sh as i32]);
            let [dx, dy] = args.dest_point.unwrap_or([0, 0]);
            let (x0, y0) = (x0.max(0) as u32, y0.max(0) as u32);
            let (dx, dy) = (dx.max(0) as u32, dy.max(0) as u32);
            let w = (x1.max(0) as u32).min(sw).saturating_sub(x0).min(width.saturating_sub(dx));
            let h = (y1.max(0) as u32).min(sh).saturating_sub(y0).min(height.saturating_sub(dy));
            if w > 0 && h > 0 {
                let s = &self.surfaces[&source];
                let t = &self.textures[&base];
                let mut encoder = device.create_command_encoder("d3d resolve");
                encoder.copy_texture_to_texture(
                    &TexelCopyTextureInfo {
                        texture: &s.texture,
                        mip_level: 0,
                        origin: Origin3d { x: x0, y: y0, z: 0 },
                        aspect: TextureAspect::All,
                    },
                    &TexelCopyTextureInfo {
                        texture: &t.texture,
                        mip_level: 0,
                        origin: Origin3d { x: dx, y: dy, z: 0 },
                        aspect: TextureAspect::All,
                    },
                    extent(w, h, 1),
                );
                device.queue().submit([encoder.finish("d3d resolve")]);
            }
        }
        let mut flags = 0;
        if args.flags & resolve::CLEAR_RENDERTARGET != 0 {
            flags |= 1 << args.source_target().unwrap_or(0);
        }
        if args.flags & resolve::CLEAR_DEPTHSTENCIL != 0 {
            flags |= clear::ZBUFFER | clear::STENCIL;
        }
        if flags != 0 {
            let c = args.clear_color.unwrap_or([0.0; 4]);
            let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            let color = byte(c[3]) << 24 | byte(c[0]) << 16 | byte(c[1]) << 8 | byte(c[2]);
            self.clear(
                device,
                &ClearArgs { rects: Vec::new(), flags, color, z: args.clear_z, stencil: args.clear_stencil },
            );
        }
    }

    fn front_buffer(&mut self, front: Option<&TextureHeader>) -> Option<FrontBuffer> {
        let base = front?.base_address() & 0x1FFF_FFFF;
        let t = self.textures.get(&base)?;
        Some(FrontBuffer {
            view: t.view.clone(),
            filterable: true,
            width: t.width,
            height: t.height,
            swizzle: [0, 1, 2, 3],
        })
    }
}

/// The executor's render targets under `--gpu d3d`: the D3D surfaces bound to each slot.
pub(crate) struct D3dTargets(pub std::rc::Rc<std::cell::RefCell<D3dRenderer>>);

impl crate::targets::RenderTargetCache for D3dTargets {
    fn targets(
        &mut self,
        _device: &Device,
        _encoder: &mut CommandEncoder,
        setup: &xenos_state::edram::RenderTargetSetup,
        _min_height: u32,
    ) -> Option<crate::targets::DrawTargets> {
        let mut r = self.0.borrow_mut();
        // Attachments of a pass must share their size: the first bound one decides.
        let mut size = None;
        let mut warning = None;
        let mut pick = |key: Option<SurfaceKey>, r: &D3dRenderer| {
            let s = r.surfaces.get(&key?)?;
            match size {
                None => size = Some((s.width, s.height)),
                Some(wh) if wh != (s.width, s.height) => {
                    warning = Some(format!(
                        "surfaces of different sizes bound together: {wh:?} and {}x{}",
                        s.width, s.height
                    ));
                    return None;
                }
                Some(_) => {}
            }
            Some((s.view.clone(), s.format))
        };
        let colors: [Option<(TextureView, TextureFormat)>; 4] =
            std::array::from_fn(|i| setup.color[i].as_ref().and_then(|_| pick(r.targets[i], &r)));
        let depth = setup.depth.as_ref().and_then(|_| pick(r.depth, &r));
        if let Some(w) = warning {
            r.warn_once(w);
        }
        let (width, height) = size?;
        Some(crate::targets::DrawTargets {
            blendable: colors.each_ref().map(|c| {
                c.as_ref().is_none_or(|(_, f)| !matches!(*f, TextureFormat::R32Float | TextureFormat::RG32Float))
            }),
            colors,
            depth,
            width,
            height,
            sample_count: 1,
            sample_mask: !0,
        })
    }

    fn resolve(
        &mut self,
        _device: &Device,
        _info: &xenos_state::resolve::ResolveInfo,
        _memory: &dyn crate::PhysicalMemory,
        _shared: Option<&crate::targets::SharedMemoryTarget<'_>>,
        _encoder: Option<&mut CommandEncoder>,
    ) -> crate::targets::ResolveStatus {
        // Resolves are D3D calls here; the command stream's copies never reach the executor.
        crate::targets::ResolveStatus::Done { written: None, in_gpu_memory: false }
    }

    fn poll_resolve(&mut self, _device: &Device, _memory: &dyn crate::PhysicalMemory) -> crate::targets::ResolveStatus {
        crate::targets::ResolveStatus::Done { written: None, in_gpu_memory: false }
    }

    fn end_frame(&mut self) {}

    fn flush_uploads(&mut self, _device: &Device) {}
}
