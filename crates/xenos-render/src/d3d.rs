//! `--gpu d3d`: rendering with the title's D3D calls in place of EDRAM (see the `d3d-hle` crate).
//! Each D3D surface is a host texture of its own (multisampled like the guest's), which the
//! executor draws into ([`D3dTargets`]); clears clear it; a resolve's copy draw converts and
//! stores the surface into the GPU copy of guest memory in the destination's format, where
//! textures (and the front buffer the swap shows) are loaded from, and does the clears the
//! resolve asks for.
//!
//! Predicated tiling isn't emulated: a surface the title renders in tiles is as tall as the
//! screen here, each draw runs once over all of it (see the command processor), and each tile's
//! resolve reads the tile's rows. Where the title relies on surfaces sharing EDRAM (one written,
//! another at the same base read), the contents are carried over when the other is bound.
//! `--render-debug d3d-surfaces` shows every color surface the frame used side by side instead
//! of the front buffer.

use std::collections::HashMap;

use d3d_hle::args::{ClearArgs, clear};
use d3d_hle::command::{Command, Surface};
use gpu_webgpu::*;
use xenos_state::edram::{EDRAM_TILE_COUNT, EDRAM_TILE_WIDTH_SAMPLES, RenderTargetKey};
use xenos_state::xenos::{ColorRenderTargetFormat, MsaaSamples};

use crate::present::FrontBuffer;

/// A surface's host texture.
struct HostSurface {
    texture: Texture,
    view: TextureView,
    /// The views the EDRAM shaders bind (resolves, transfers).
    views: xenos_edram::ExternalViews,
    format: TextureFormat,
    width: u32,
    height: u32,
    /// Host samples: 1, or 4 for both 2x and 4x MSAA, like the EDRAM render targets (2x uses
    /// host samples 0 and 3).
    samples: u32,
    /// Guest MSAA is 2x (draws then write only host samples 0 and 3).
    msaa_2x: bool,
    depth: bool,
    /// D24FS8: the host depth holds half the guest value (as the executor's viewport maps it).
    float24: bool,
    /// The surface header it was made for (a different one replaces it).
    header: [u32; d3d_hle::args::SURFACE_WORDS],
    /// Where it is in EDRAM on the console, and its layout there.
    edram: RenderTargetKey,
    /// The last frame (present) it was bound or drawn to in.
    last_used: u64,
    /// Its texture in [`D3dRenderer::textures`].
    texture_key: TextureKey,
}

/// Surfaces with the same EDRAM layout, size and format share one texture: they are the same
/// memory on the console (the title makes several surface objects for one buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TextureKey {
    edram: RenderTargetKey,
    width: u32,
    height: u32,
    samples: u32,
    format: TextureFormat,
}

/// A texture shared by the surfaces with its [`TextureKey`].
struct SharedTexture {
    texture: Texture,
    view: TextureView,
    views: xenos_edram::ExternalViews,
    users: u32,
}

impl HostSurface {
    fn external(&self, offset: [u32; 2]) -> xenos_edram::ExternalSource<'_> {
        xenos_edram::ExternalSource {
            texture: &self.texture,
            views: &self.views,
            format: self.format,
            width: self.width,
            height: self.height,
            samples: self.samples,
            depth: self.depth,
            base_tiles: self.edram.base_tiles,
            offset,
        }
    }
}

/// A surface's key: its object address, and whether it is bound as depth/stencil (the title
/// binds some surface objects both ways).
type SurfaceKey = (u32, bool);

pub(crate) struct D3dRenderer {
    surfaces: HashMap<SurfaceKey, HostSurface>,
    /// Rows surfaces were grown to: a surface rendered with predicated tiling is one tile tall
    /// on the console, and stands for the whole screen here (see [`D3dRenderer::grow`]).
    rows: HashMap<SurfaceKey, u32>,
    /// The surfaces' textures.
    textures: HashMap<TextureKey, SharedTexture>,
    /// The surface last drawn to or cleared at each EDRAM base tile. Surfaces at the same base
    /// share their EDRAM on the console, and titles rely on it (Skyrim restores its depth
    /// buffer by drawing the depth texture into a color target at the depth buffer's base):
    /// binding one after another was written gets the other's contents (see
    /// [`D3dTargets::targets`]).
    edram_owners: HashMap<u32, SurfaceKey>,
    /// Surfaces sharing EDRAM, as found when binding them (logged with the surface counts).
    aliasing: AliasingStats,
    aliasing_seen: std::collections::HashSet<(bool, Layout, Layout)>,
    /// Textures replaced while the executor's batch may still use them, destroyed at the end of
    /// the frame (after the batch is submitted).
    retired: Vec<Texture>,
    /// Bound render targets 0-3 and depth/stencil surface.
    targets: [Option<SurfaceKey>; 4],
    depth: Option<SurfaceKey>,
    depth32_stencil8: bool,
    warned: std::collections::HashSet<String>,
    /// Color surfaces used this frame, in order of first use (for the mosaic).
    used: Vec<SurfaceKey>,
    /// Converts and stores resolves into the GPU copy of guest memory (its EDRAM-layout
    /// conversion and tiling shaders; it holds no render targets here).
    edram: Option<xenos_edram::EdramCache>,
    /// What the frame did (logged for one frame in [`LOG_PERIOD`]).
    frame_log: Vec<String>,
    presents: u64,
    mosaic: Option<Mosaic>,
    /// Clears not done yet, by texture: the next pass drawing to the texture does them as load
    /// ops ([`D3dTargets::targets`]) rather than each taking a pass of its own (which on a
    /// tile-based GPU, as phones have, stores the whole surface for the next pass to load).
    /// Anything else reading or writing the surfaces does them first ([`D3dRenderer::do_clears`]).
    pending_clears: Vec<(TextureKey, TextureView, PendingClear)>,
    /// Host pixels per guest pixel along each axis (`--resolution-scale`): surface textures are
    /// this much larger, and resolves average each guest pixel's host pixels.
    scale: u32,
}

#[derive(Clone, Copy)]
enum PendingClear {
    Color(Color),
    DepthStencil { depth: Option<f32>, stencil: Option<u32> },
}

/// Presents between per-frame operation logs.
const LOG_PERIOD: u64 = 600;
/// Frames a surface stays without being bound (see [`D3dRenderer::evict`]).
const EVICT_FRAMES: u64 = 60;
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
            rows: HashMap::new(),
            textures: HashMap::new(),
            edram_owners: HashMap::new(),
            aliasing: AliasingStats::default(),
            aliasing_seen: Default::default(),
            retired: Vec::new(),
            targets: [None; 4],
            depth: None,
            depth32_stencil8: device.has_feature(FeatureName::Depth32FloatStencil8),
            warned: Default::default(),
            used: Vec::new(),
            edram: None,
            frame_log: Vec::new(),
            presents: 0,
            mosaic: None,
            pending_clears: Vec::new(),
            scale: crate::RESOLUTION_SCALE.load(std::sync::atomic::Ordering::Relaxed).clamp(1, 3),
        }
    }

    /// The resolve and transfer shaders, built for the resolution scale.
    fn edram_cache(&self, device: &Device) -> xenos_edram::EdramCache {
        let options = xenos_edram::EdramOptions { resolution_scale: self.scale, ..Default::default() };
        xenos_edram::EdramCache::with_options(device, options)
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
        let mut encoder = device.create_command_encoder("d3d surface mosaic");
        self.do_clears(&mut encoder);
        let mosaic = self.mosaic.get_or_insert_with(|| Mosaic::new(device));
        // Multisampled surfaces are left out (the tile shader reads single-sampled textures).
        let surfaces: Vec<&HostSurface> =
            self.used.iter().filter_map(|k| self.surfaces.get(k)).filter(|s| s.samples == 1).collect();
        let n = surfaces.len().max(1) as u32;
        let cols = (n as f32).sqrt().ceil() as u32;
        let rows = n.div_ceil(cols);
        let (tw, th) = (MOSAIC_SIZE.0 / cols, MOSAIC_SIZE.1 / rows);
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

    /// Executes `cmd` (its GPU work is recorded later, see `pending_clears`); a present returns
    /// the surface mosaic to show (`--render-debug d3d-surfaces`).
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
                self.clear(&args);
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
            }
            Command::Present { .. } => {
                let shown = if crate::debug::D3D_SURFACES.load(std::sync::atomic::Ordering::Relaxed) {
                    self.mosaic(device)
                } else {
                    None
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
                self.evict();
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
        let frame = self.presents;
        if let Some(h) = self.surfaces.get_mut(&key).filter(|h| h.header == s.header.words) {
            h.last_used = frame;
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
        let width = s.width() * self.scale;
        let height = s.height().max(self.rows.get(&key).copied().unwrap_or(0)) * self.scale;
        // Phones and tablets draw multisampled surfaces single-sampled: MSAA would take most of
        // their memory (a whole-screen 4x depth surface is 28 MiB) for little visible gain.
        let samples =
            if s.samples() > 1 && !crate::HANDHELD.load(std::sync::atomic::Ordering::Relaxed) { 4 } else { 1 };
        let edram = edram_key(s);
        let texture_key = TextureKey { edram, width, height, samples, format };
        let (texture, view, views) = self.acquire(device, texture_key, s.depth);
        log::debug!("d3d renderer: surface {:#010X} {width}x{height}x{samples} {format:?}", s.address);
        if let Some(old) = self.surfaces.insert(
            key,
            HostSurface {
                texture,
                view,
                views,
                format,
                width,
                height,
                samples,
                msaa_2x: s.samples() == 2 && samples > 1,
                depth: s.depth,
                float24: s.depth && s.format() & 1 != 0,
                header: s.header.words,
                edram,
                last_used: frame,
                texture_key,
            },
        ) {
            self.release(old.texture_key);
        }
        key
    }

    /// The texture for surfaces with `key` (created for the first).
    fn acquire(
        &mut self,
        device: &Device,
        key: TextureKey,
        depth: bool,
    ) -> (Texture, TextureView, xenos_edram::ExternalViews) {
        let shared = self.textures.entry(key).or_insert_with(|| {
            let (texture, view) = surface_texture(device, depth, key.width, key.height, key.samples, key.format);
            let views = xenos_edram::ExternalViews::new(&texture, depth);
            SharedTexture { texture, view, views, users: 0 }
        });
        shared.users += 1;
        (shared.texture.clone(), shared.view.clone(), shared.views.clone())
    }

    /// A surface no longer uses the texture with `key`; the last one retires it.
    fn release(&mut self, key: TextureKey) {
        if let Some(shared) = self.textures.get_mut(&key) {
            shared.users -= 1;
            if shared.users == 0
                && let Some(shared) = self.textures.remove(&key)
            {
                self.retired.push(shared.texture);
                self.pending_clears.retain(|(k, _, _)| *k != key);
            }
        }
    }

    /// Drops the surfaces not bound for [`EVICT_FRAMES`] frames: the title creates surface
    /// objects for passes it stops running (and new ones at other addresses), whose textures
    /// would otherwise stay for good.
    fn evict(&mut self) {
        let now = self.presents;
        let bound: Vec<SurfaceKey> = self.targets.iter().flatten().chain(self.depth.iter()).copied().collect();
        let old: Vec<SurfaceKey> = self
            .surfaces
            .iter()
            .filter(|(k, s)| now.saturating_sub(s.last_used) > EVICT_FRAMES && !bound.contains(k))
            .map(|(k, _)| *k)
            .collect();
        for key in old {
            if let Some(s) = self.surfaces.remove(&key) {
                log::debug!("d3d renderer: surface {:#010X} evicted", key.0);
                self.release(s.texture_key);
            }
            self.rows.remove(&key);
            self.edram_owners.retain(|_, owner| *owner != key);
        }
        if self.logging() {
            let bytes: u64 = self.textures.keys().map(texture_bytes).sum();
            log::info!(
                "d3d renderer: {} surfaces in {} textures, {} MiB; EDRAM aliasing {:?}",
                self.surfaces.len(),
                self.textures.len(),
                bytes >> 20,
                self.aliasing
            );
        }
    }

    /// Makes surface `key` at least `rows` (guest rows) tall. With predicated tiling the title's surfaces are
    /// one tile tall and each tile is drawn with a window offset; here the draws cover the
    /// whole surface (see the command processor), so it must be the screen's height, which
    /// the draws' viewport and scissor tell. The texture is replaced (its contents are lost:
    /// this happens once, on the surface's first tiled use).
    fn grow(&mut self, device: &Device, key: SurfaceKey, rows: u32) {
        let Some(s) = self.surfaces.get(&key) else { return };
        let rows = rows.min(device.limits().maxTextureDimension2D / self.scale);
        let height = rows * self.scale;
        if s.height >= height {
            return;
        }
        log::debug!("d3d renderer: surface {:#010X} grows from {} to {height} rows", key.0, s.height);
        let (old, depth) = (s.texture_key, s.depth);
        let new = TextureKey { height, ..old };
        let (texture, view, views) = self.acquire(device, new, depth);
        self.release(old);
        let s = self.surfaces.get_mut(&key).expect("checked above");
        (s.texture, s.view, s.views, s.height, s.texture_key) = (texture, view, views, height, new);
        self.rows.insert(key, rows);
    }

    /// Makes `key` the owner of its EDRAM base, first copying the previous owner's contents
    /// into it when the two lay out the same EDRAM the same way (same size and samples, any
    /// format: what the console would see through the shared memory). Surfaces that only
    /// partly overlap aren't converted.
    fn take_edram(&mut self, device: &Device, encoder: &mut CommandEncoder, key: SurfaceKey) {
        let Some(s) = self.surfaces.get(&key) else { return };
        let base = s.edram.base_tiles;
        self.note_overlaps(key);
        let previous = self.edram_owners.insert(base, key);
        let Some(p) = previous.filter(|&p| p != key).and_then(|p| self.surfaces.get(&p)) else { return };
        let s = &self.surfaces[&key];
        let same_layout = (p.width, p.height, p.samples, p.edram.msaa_samples, p.edram.pitch_tiles_at_32bpp)
            == (s.width, s.height, s.samples, s.edram.msaa_samples, s.edram.pitch_tiles_at_32bpp);
        // Surfaces with the same format share their texture already.
        if p.texture_key == s.texture_key {
            return;
        }
        if !same_layout {
            self.aliasing.same_base_other_layout += 1;
            if self.aliasing_seen.len() < 64 && self.aliasing_seen.insert((false, layout(p), layout(s))) {
                log::info!(
                    "d3d renderer: EDRAM aliasing not carried over: same base {base}: {} then {}",
                    describe(p),
                    describe(s)
                );
            }
            return;
        }
        self.aliasing.carried += 1;
        self.do_clears(encoder);
        let s = &self.surfaces[&key];
        let p = &self.surfaces[&previous.expect("found above")];
        if self.edram.is_none() {
            self.edram = Some(self.edram_cache(device));
        }
        let edram = self.edram.as_mut().expect("made above");
        let done =
            edram.transfer_external(device, encoder, (&p.external([0, 0]), p.edram), (&s.external([0, 0]), s.edram));
        if done {
            log::debug!(
                "d3d renderer: EDRAM base {base} passes from {:#010X} to {:#010X}",
                previous.unwrap_or_default().0,
                key.0
            );
        }
    }

    /// Counts (and logs the first of each kind) the surfaces drawn this frame at another EDRAM
    /// base whose EDRAM range overlaps `key`'s: their contents aren't carried over.
    fn note_overlaps(&mut self, key: SurfaceKey) {
        let s = &self.surfaces[&key];
        let range = edram_range(s);
        let now = self.presents;
        for (&base, owner) in &self.edram_owners {
            if base == range.0 || *owner == key {
                continue;
            }
            let Some(o) = self.surfaces.get(owner).filter(|o| o.last_used == now) else { continue };
            if ranges_overlap(range, edram_range(o)) {
                self.aliasing.overlap_other_base += 1;
                if self.aliasing_seen.len() < 64 && self.aliasing_seen.insert((true, layout(s), layout(o))) {
                    log::info!(
                        "d3d renderer: EDRAM ranges overlap, not carried over: {} over {}",
                        describe(s),
                        describe(o)
                    );
                }
            }
        }
    }

    /// Clears the bound surfaces the flags name (the whole surface: the title's clear rectangles
    /// cover the viewport). They're done by the next pass drawing to them, or before anything
    /// else uses them (see `pending_clears`).
    fn clear(&mut self, args: &ClearArgs) {
        let rgba = args.rgba().map(f64::from);
        let color = Color { r: rgba[0], g: rgba[1], b: rgba[2], a: rgba[3] };
        for (i, target) in self.targets.into_iter().enumerate() {
            if args.flags & (1 << i) == 0 {
                continue;
            }
            if let Some((key, s)) = target.and_then(|t| Some((t, self.surfaces.get(&t)?))) {
                self.edram_owners.insert(s.edram.base_tiles, key);
                let (texture, view) = (s.texture_key, s.view.clone());
                self.pend_clear(texture, view, PendingClear::Color(color));
            }
        }
        if args.flags & (clear::ZBUFFER | clear::STENCIL) != 0
            && let Some((key, s)) = self.depth.and_then(|d| Some((d, self.surfaces.get(&d)?)))
        {
            self.edram_owners.insert(s.edram.base_tiles, key);
            let z = if s.float24 { args.z.clamp(0.0, 2.0) * 0.5 } else { args.z.clamp(0.0, 1.0) };
            let clear = PendingClear::DepthStencil {
                depth: (args.flags & clear::ZBUFFER != 0).then_some(z),
                stencil: (args.flags & clear::STENCIL != 0).then_some(args.stencil & 0xFF),
            };
            let (texture, view) = (s.texture_key, s.view.clone());
            self.pend_clear(texture, view, clear);
        }
    }

    /// Adds a clear of `texture`, replacing (for each aspect it clears) an earlier one.
    fn pend_clear(&mut self, texture: TextureKey, view: TextureView, clear: PendingClear) {
        let Some((_, _, old)) = self.pending_clears.iter_mut().find(|(k, _, _)| *k == texture) else {
            self.pending_clears.push((texture, view, clear));
            return;
        };
        *old = match (*old, clear) {
            (
                PendingClear::DepthStencil { depth: d0, stencil: s0 },
                PendingClear::DepthStencil { depth: d1, stencil: s1 },
            ) => PendingClear::DepthStencil { depth: d1.or(d0), stencil: s1.or(s0) },
            _ => clear,
        };
    }

    /// Records the pending clears, each as a pass of its own.
    fn do_clears(&mut self, encoder: &mut CommandEncoder) {
        for (_, view, clear) in self.pending_clears.drain(..) {
            match clear {
                PendingClear::Color(color) => {
                    let attachment = RenderPassColorAttachment::clear(&view, color);
                    encoder.begin_render_pass("d3d clear", &[Some(attachment)], None).end();
                }
                PendingClear::DepthStencil { depth, stencil } => {
                    let attachment = RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_load_op: if depth.is_some() { LoadOp::Clear } else { LoadOp::Load },
                        depth_store_op: StoreOp::Store,
                        depth_clear_value: depth.unwrap_or(0.0),
                        depth_read_only: false,
                        stencil_load_op: if stencil.is_some() { LoadOp::Clear } else { LoadOp::Load },
                        stencil_store_op: StoreOp::Store,
                        stencil_clear_value: stencil.unwrap_or(0),
                        stencil_read_only: false,
                    };
                    encoder.begin_render_pass("d3d clear", &[], Some(&attachment)).end();
                }
            }
        }
    }
}

/// A surface's place and layout in EDRAM.
fn edram_key(s: &Surface) -> RenderTargetKey {
    let msaa = MsaaSamples::from_raw((s.header.surface_info() >> 16) & 3);
    let format =
        if s.depth { s.format() } else { ColorRenderTargetFormat::from_raw(s.format()).storage_format().raw() };
    RenderTargetKey {
        base_tiles: s.header.info() & 0xFFF,
        pitch_tiles_at_32bpp: (s.width() << msaa.samples_x_log2()).div_ceil(EDRAM_TILE_WIDTH_SAMPLES),
        msaa_samples: msaa,
        is_depth: s.depth,
        format,
    }
}

/// Surfaces found sharing EDRAM (see [`D3dRenderer::take_edram`]).
#[derive(Clone, Copy, Debug, Default)]
struct AliasingStats {
    /// Same base and layout, another format: contents converted.
    carried: u64,
    /// Same base, another size or sample count: not carried over.
    same_base_other_layout: u64,
    /// Overlapping ranges from different bases: not carried over.
    overlap_other_base: u64,
}

/// A surface's EDRAM tiles on the console: (first tile, count), from its EDRAM size.
fn edram_range(s: &HostSurface) -> (u32, u32) {
    const TILE_BYTES: u32 = 80 * 16 * 4;
    (s.edram.base_tiles, s.header[11].div_ceil(TILE_BYTES).clamp(1, EDRAM_TILE_COUNT))
}

fn ranges_overlap(a: (u32, u32), b: (u32, u32)) -> bool {
    // Ranges wrap around the end of EDRAM.
    let starts_in = |x: (u32, u32), y: (u32, u32)| (y.0.wrapping_sub(x.0) & (EDRAM_TILE_COUNT - 1)) < x.1;
    starts_in(a, b) || starts_in(b, a)
}

/// What [`describe`] tells about a surface: depth, width, height, samples, EDRAM format, base
/// and tiles.
type Layout = (bool, u32, u32, u32, u32, u32, u32);

fn layout(s: &HostSurface) -> Layout {
    (s.depth, s.width, s.height, s.samples, s.edram.format, s.edram.base_tiles, edram_range(s).1)
}

fn describe(s: &HostSurface) -> String {
    let kind = if s.depth { "depth" } else { "color" };
    format!(
        "{kind} {}x{}x{} format {} at {} ({} tiles)",
        s.width,
        s.height,
        s.samples,
        s.edram.format,
        s.edram.base_tiles,
        edram_range(s).1
    )
}

fn texture_bytes(key: &TextureKey) -> u64 {
    // Depth/stencil: 32-bit float depth plus stencil, or 24+8 bits.
    let per_sample = format_block_info(key.format)
        .map_or(if key.format == TextureFormat::Depth32FloatStencil8 { 8 } else { 4 }, |(bytes, _)| bytes);
    u64::from(key.width) * u64::from(key.height) * u64::from(key.samples) * u64::from(per_sample)
}

fn surface_texture(
    device: &Device,
    depth: bool,
    width: u32,
    height: u32,
    samples: u32,
    format: TextureFormat,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: if depth { "d3d depth surface" } else { "d3d color surface" },
        size: extent(width, height, 1),
        sample_count: samples,
        format,
        usage: TextureUsage::RenderAttachment | TextureUsage::TextureBinding,
        ..Default::default()
    });
    let view = texture.create_view(None);
    (texture, view)
}

/// The executor's render targets under `--gpu d3d`: the D3D surfaces bound to each slot.
pub(crate) struct D3dTargets(pub std::rc::Rc<std::cell::RefCell<D3dRenderer>>);

impl crate::targets::RenderTargetCache for D3dTargets {
    fn targets(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        setup: &xenos_state::edram::RenderTargetSetup,
        min_height: u32,
    ) -> Option<crate::targets::DrawTargets> {
        let mut r = self.0.borrow_mut();
        let bound: Vec<SurfaceKey> = (0..4)
            .filter(|&i| setup.color[i].is_some())
            .filter_map(|i| r.targets[i])
            .chain(r.depth.filter(|_| setup.depth.is_some()))
            .collect();
        let frame = r.presents;
        for &key in &bound {
            r.grow(device, key, min_height);
            if let Some(s) = r.surfaces.get_mut(&key) {
                s.last_used = frame;
            }
        }
        for &key in &bound {
            r.take_edram(device, encoder, key);
        }
        // Attachments of a pass must share their size and sample count: the first bound one
        // decides.
        let mut size = None;
        let mut warning = None;
        let mut pick = |key: Option<SurfaceKey>, r: &D3dRenderer| {
            let s = r.surfaces.get(&key?)?;
            let this = (s.width, s.height, s.samples, s.msaa_2x);
            match size {
                None => size = Some(this),
                Some(first) if first != this => {
                    warning = Some(format!("surfaces of different sizes bound together: {first:?} and {this:?}"));
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
        let (width, height, samples, msaa_2x) = size?;
        // The pass does the clears of its attachments; the others are done before it.
        let mut clears = crate::targets::AttachmentClears::default();
        let pending = std::mem::take(&mut r.pending_clears);
        for (texture, view, clear) in pending {
            let attached = |a: &Option<(TextureView, TextureFormat)>| a.as_ref().is_some_and(|(v, _)| *v == view);
            match (clear, colors.iter().position(attached)) {
                (PendingClear::Color(color), Some(i)) => clears.colors[i] = Some(color),
                (PendingClear::DepthStencil { depth: d, stencil: s }, _) if attached(&depth) => {
                    (clears.depth, clears.stencil) = (d, s);
                }
                _ => r.pending_clears.push((texture, view, clear)),
            }
        }
        r.do_clears(encoder);
        Some(crate::targets::DrawTargets {
            blendable: colors.each_ref().map(|c| {
                c.as_ref().is_none_or(|(_, f)| !matches!(*f, TextureFormat::R32Float | TextureFormat::RG32Float))
            }),
            colors,
            depth,
            width,
            height,
            sample_count: samples,
            sample_mask: if msaa_2x { 0b1001 } else { !0 },
            clears,
        })
    }

    /// A resolve's copy draw (the only D3D-internal draws that run): the bound surface the copy
    /// reads is converted and stored into the GPU copy of guest memory, so textures made from
    /// it see the result (the front buffer included), and the clears the resolve asks for are
    /// done to the bound surfaces.
    fn resolve(
        &mut self,
        device: &Device,
        info: &xenos_state::resolve::ResolveInfo,
        _memory: &dyn crate::PhysicalMemory,
        shared: Option<&crate::targets::SharedMemoryTarget<'_>>,
        encoder: Option<&mut CommandEncoder>,
    ) -> crate::targets::ResolveStatus {
        let none = crate::targets::ResolveStatus::Done { written: None, in_gpu_memory: false };
        let (Some(shared), Some(encoder)) = (shared, encoder) else {
            return none;
        };
        let mut r = self.0.borrow_mut();
        r.do_clears(encoder);
        // The cache is taken out while it runs (the surfaces stay borrowed from `r`).
        let mut edram = r.edram.take().unwrap_or_else(|| r.edram_cache(device));
        // With predicated tiling the resolve covers one tile, whose screen origin the window
        // offset moved to the EDRAM origin; the surfaces hold the whole screen.
        let offset = info.window_offset.map(|v| (-v).max(0) as u32);
        let external = |key: Option<SurfaceKey>| key.and_then(|k| r.surfaces.get(&k)).map(|s| s.external(offset));
        let color = external(r.targets[(info.copy_control.copy_src_select & 3) as usize]);
        let depth = external(r.depth);
        let result = edram.resolve_external(device, encoder, info, shared, color.as_ref(), depth.as_ref());
        r.edram = Some(edram);
        match result {
            Ok(result) => crate::targets::ResolveStatus::Done { written: result.written, in_gpu_memory: true },
            Err(e) => {
                r.warn_once(format!("resolve: {e}"));
                none
            }
        }
    }

    fn poll_resolve(&mut self, _device: &Device, _memory: &dyn crate::PhysicalMemory) -> crate::targets::ResolveStatus {
        crate::targets::ResolveStatus::Done { written: None, in_gpu_memory: false }
    }

    /// The batch is submitted: the textures replaced during it can go.
    fn end_frame(&mut self) {
        for texture in self.0.borrow_mut().retired.drain(..) {
            texture.destroy();
        }
    }

    fn flush_uploads(&mut self, device: &Device) {
        if let Some(edram) = &self.0.borrow().edram {
            edram.flush_uploads(device);
        }
    }

    fn resolution_scale(&self) -> u32 {
        self.0.borrow().scale
    }
}
