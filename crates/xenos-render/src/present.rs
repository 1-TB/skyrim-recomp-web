//! Shows the guest's front buffer (VdSwap's fetch constant) in the presentation target:
//! letterboxed to keep its aspect, zoomed to fill the target (cropping the overhang) or
//! stretched, per [`set_aspect`]. The front buffer comes from the texture cache, which loads it
//! from the GPU copy of guest memory when a resolve stored it there.

use gpu_webgpu::*;
use xenos_texture::{SwizzleSource, TextureFetchConstant};

const BLIT: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// The part of the frame shown: offset (xy) and size (zw) in texture coordinates.
@group(0) @binding(3) var<uniform> crop: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = crop.xy + vec2<f32>(p.x, 1.0 - p.y) * crop.zw;
    return o;
}

@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var frame_sampler: sampler;
// Per output channel: host channel 0-3, 4 = zero, 5 = one (the fetch constant's swizzle).
@group(0) @binding(2) var<uniform> swizzle: vec4<u32>;

fn pick(t: vec4<f32>, s: u32) -> f32 {
    switch s {
        case 0u: { return t.x; }
        case 1u: { return t.y; }
        case 2u: { return t.z; }
        case 3u: { return t.w; }
        case 4u: { return 0.0; }
        default: { return 1.0; }
    }
}

// Catmull-Rom (bicubic) filtering in 9 bilinear taps (the 16 texels' weights folded pairwise):
// sharper than bilinear when the frame is scaled up to the screen.
fn catmull_rom(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(frame));
    let pos = uv * size;
    let center = floor(pos - 0.5) + 0.5;
    let f = pos - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (center - 1.0) / size;
    let t3 = (center + 2.0) / size;
    let t12 = (center + w2 / w12) / size;
    var c = vec4<f32>(0.0);
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t0.x, t0.y), 0.0) * w0.x * w0.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t12.x, t0.y), 0.0) * w12.x * w0.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t3.x, t0.y), 0.0) * w3.x * w0.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t0.x, t12.y), 0.0) * w0.x * w12.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t12.x, t12.y), 0.0) * w12.x * w12.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t3.x, t12.y), 0.0) * w3.x * w12.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t0.x, t3.y), 0.0) * w0.x * w3.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t12.x, t3.y), 0.0) * w12.x * w3.y;
    c += textureSampleLevel(frame, frame_sampler, vec2<f32>(t3.x, t3.y), 0.0) * w3.x * w3.y;
    return max(c, vec4<f32>(0.0));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Scaled up (the viewport is larger than the frame): bicubic; otherwise bilinear.
    let scale = fwidth(in.uv) * vec2<f32>(textureDimensions(frame));
    var t: vec4<f32>;
    if (max(scale.x, scale.y) < 0.99) {
        t = catmull_rom(in.uv);
    } else {
        t = textureSampleLevel(frame, frame_sampler, in.uv, 0.0);
    }
    return vec4<f32>(pick(t, swizzle.x), pick(t, swizzle.y), pick(t, swizzle.z), 1.0);
}
"#;

const BACKGROUND: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

/// A front buffer ready to show: the texture cache's view of it and how its channels map.
pub(crate) struct FrontBuffer {
    pub view: TextureView,
    pub filterable: bool,
    pub width: u32,
    pub height: u32,
    pub swizzle: [u32; 4],
}

impl FrontBuffer {
    /// The fetch constant a front buffer is loaded with (base level only, plain 2D), and the
    /// output channel mapping for `component_map` (guest component → host channel).
    pub fn fetch_and_swizzle(
        fetch: &TextureFetchConstant,
        component_map: [xenos_texture::HostChannel; 4],
    ) -> (TextureFetchConstant, [u32; 4]) {
        let mut fb = fetch.clone();
        fb.mip_page = 0;
        fb.mip_max_level = 0;
        fb.mip_min_level = 0;
        fb.dimension = xenos_texture::DataDimension::D2OrStacked;
        fb.stacked = false;
        let channel = |c: xenos_texture::HostChannel| match c {
            xenos_texture::HostChannel::R => 0,
            xenos_texture::HostChannel::G => 1,
            xenos_texture::HostChannel::B => 2,
            xenos_texture::HostChannel::A => 3,
        };
        let swizzle = fetch.swizzle.map(|s| match s {
            SwizzleSource::Zero => 4,
            SwizzleSource::One => 5,
            s => channel(component_map[s.component().unwrap_or(0)]),
        });
        (fb, swizzle)
    }
}

struct Shown {
    width: u32,
    height: u32,
    group: BindGroup,
}

/// Slots of the crop ring: each encode takes the next one, so a crop written for one target
/// never replaces another's before that one's work is submitted.
const CROP_SLOTS: u64 = 8;
const CROP_SLOT: u64 = 256;
/// Crop ring slot 0 always holds the whole frame.
const WHOLE: u32 = 0;

/// How frames fill a target of another aspect ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aspect {
    /// The whole frame, as large as it fits, with black bars (the default).
    Fit,
    /// Zoomed to cover the target, cropping the edges that overhang.
    Fill,
    /// Stretched to the target.
    Stretch,
}

impl std::str::FromStr for Aspect {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Aspect, String> {
        match s {
            "fit" => Ok(Aspect::Fit),
            "fill" => Ok(Aspect::Fill),
            "stretch" => Ok(Aspect::Stretch),
            _ => Err(format!("aspect {s}: not fit, fill or stretch")),
        }
    }
}

static ASPECT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// How presenters fit frames to a target of another aspect ratio from now on.
pub fn set_aspect(aspect: Aspect) {
    ASPECT.store(aspect as u8, std::sync::atomic::Ordering::Relaxed);
}

fn aspect() -> Aspect {
    match ASPECT.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Aspect::Fill,
        2 => Aspect::Stretch,
        _ => Aspect::Fit,
    }
}

/// Where a frame goes in a target.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    /// The rectangle of the target drawn to (x, y, width, height).
    dest: (f32, f32, f32, f32),
    /// The part of the frame shown there (offset and size in texture coordinates).
    crop: [f32; 4],
    /// The size the whole frame is scaled to (larger than `dest` where cropped).
    scaled: (u32, u32),
}

fn place(aspect: Aspect, frame: (u32, u32), target: (u32, u32)) -> Placement {
    let whole = [0.0, 0.0, 1.0, 1.0];
    let (tw, th) = (target.0 as f32, target.1 as f32);
    match aspect {
        Aspect::Fit => {
            let dest = letterbox(frame, target);
            Placement { dest, crop: whole, scaled: (dest.2.round() as u32, dest.3.round() as u32) }
        }
        Aspect::Stretch => Placement { dest: (0.0, 0.0, tw, th), crop: whole, scaled: target },
        Aspect::Fill => {
            let (fw, fh) = (frame.0.max(1) as f32, frame.1.max(1) as f32);
            let scale = (tw / fw).max(th / fh);
            let (w, h) = (fw * scale, fh * scale);
            let (cw, ch) = (tw / w, th / h);
            Placement {
                dest: (0.0, 0.0, tw, th),
                crop: [(1.0 - cw) / 2.0, (1.0 - ch) / 2.0, cw, ch],
                scaled: (w.round() as u32, h.round() as u32),
            }
        }
    }
}

/// MetalFX upscaling for one frame size and output size.
struct Upscale {
    scaler: SpatialScaler,
    /// (frame, output) sizes it was made for.
    sizes: ((u32, u32), (u32, u32)),
    /// The scaler's output with the identity swizzle (the frame's was applied going in).
    group: BindGroup,
}

/// Scale frames up to the window with MetalFX instead of bicubic filtering (where available).
static METALFX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Presenters scale frames up to the window with MetalFX's spatial scaler when the GPU backend
/// has it (Metal), and with bicubic filtering otherwise.
pub fn set_metalfx(on: bool) {
    METALFX.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) struct Presenter {
    pipeline: RenderPipeline,
    layout: BindGroupLayout,
    sampler: Sampler,
    /// The front buffer's channel mapping (rewritten per swap: buffers are costly to create).
    swizzle: Buffer,
    shown: Option<Shown>,
    /// Swaps shown so far.
    pub frames: u64,
    format: TextureFormat,
    /// The identity channel mapping, for showing the upscaled frame.
    identity: Buffer,
    /// `CROP_SLOTS` crops; slot `WHOLE` is the whole frame.
    crops: Buffer,
    /// The crop ring slot the next encode writes (1 to `CROP_SLOTS - 1`) and the crop in each.
    crop_slots: std::cell::RefCell<(u32, [[f32; 4]; CROP_SLOTS as usize])>,
    upscale: std::cell::RefCell<Option<Upscale>>,
    /// MetalFX couldn't scale between these sizes (not retried for them).
    upscale_failed: std::cell::Cell<Option<((u32, u32), (u32, u32))>>,
}

/// The largest rectangle with the frame's aspect that fits the target, centred.
fn letterbox(frame: (u32, u32), target: (u32, u32)) -> (f32, f32, f32, f32) {
    let (fw, fh) = (frame.0.max(1) as f32, frame.1.max(1) as f32);
    let (tw, th) = (target.0 as f32, target.1 as f32);
    let scale = (tw / fw).min(th / fh);
    let (w, h) = (fw * scale, fh * scale);
    ((tw - w) / 2.0, (th - h) / 2.0, w, h)
}

impl Presenter {
    pub(crate) fn new(device: &Device, target_format: TextureFormat) -> Self {
        let module = device.create_shader_module_wgsl("present", BLIT);
        let layout = device.create_bind_group_layout(
            "present",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::Fragment,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::Fragment,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::Fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 16,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::Vertex,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: 16,
                    },
                },
            ],
        );
        let pipeline_layout = device.create_pipeline_layout("present", &[&layout]);
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "present",
            layout: Some(&pipeline_layout),
            vertex: ProgrammableStage::new(&module, "vs"),
            vertex_buffers: &[],
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                stage: ProgrammableStage::new(&module, "fs"),
                targets: &[ColorTargetState::new(target_format)],
            }),
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: "present",
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let swizzle = device.create_buffer(&BufferDescriptor {
            label: "present swizzle",
            size: 16,
            usage: BufferUsage::Uniform | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        let identity = device.create_buffer(&BufferDescriptor {
            label: "present identity swizzle",
            size: 16,
            usage: BufferUsage::Uniform | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        let bytes: Vec<u8> = [0u32, 1, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        device.queue().write_buffer(&identity, 0, &bytes);
        let crops = device.create_buffer(&BufferDescriptor {
            label: "present crops",
            size: CROP_SLOTS * CROP_SLOT,
            usage: BufferUsage::Uniform | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        let whole = [0.0f32, 0.0, 1.0, 1.0];
        let bytes: Vec<u8> = whole.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.queue().write_buffer(&crops, u64::from(WHOLE) * CROP_SLOT, &bytes);
        Presenter {
            pipeline,
            layout,
            sampler,
            swizzle,
            shown: None,
            frames: 0,
            format: target_format,
            identity,
            crop_slots: std::cell::RefCell::new((1, [[f32::NAN; 4]; CROP_SLOTS as usize])),
            crops,
            upscale: std::cell::RefCell::new(None),
            upscale_failed: std::cell::Cell::new(None),
        }
    }

    /// Shows `front` from now on (or keeps the previous frame if there is none).
    pub(crate) fn swap(&mut self, device: &Device, front: Option<FrontBuffer>) -> Option<(u32, u32)> {
        let Some(front) = front else {
            return self.shown.as_ref().map(|s| (s.width, s.height));
        };
        if !front.filterable {
            log::warn!("present: front buffer format is not filterable");
            return None;
        }
        let bytes: Vec<u8> = front.swizzle.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.queue().write_buffer(&self.swizzle, 0, &bytes);
        let swizzle = &self.swizzle;
        let group = device.create_bind_group(
            "present",
            &self.layout,
            &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&front.view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&self.sampler) },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer { buffer: swizzle, offset: 0, size: None },
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer { buffer: &self.crops, offset: 0, size: Some(16) },
                },
            ],
        );
        self.shown = Some(Shown { width: front.width, height: front.height, group });
        self.frames += 1;
        Some((front.width, front.height))
    }

    /// Draws the latest front buffer into `target` (black if there is none yet).
    pub(crate) fn encode(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        width: u32,
        height: u32,
    ) {
        let placement = self.shown.as_ref().map(|s| place(aspect(), (s.width, s.height), (width, height)));
        if let (Some(shown), Some(p)) = (&self.shown, placement) {
            if METALFX.load(std::sync::atomic::Ordering::Relaxed)
                && (p.scaled.0 > shown.width || p.scaled.1 > shown.height)
                && self.encode_upscaled(device, encoder, shown, target, p)
            {
                return;
            }
        }
        let mut pass =
            encoder.begin_render_pass("present", &[Some(RenderPassColorAttachment::clear(target, BACKGROUND))], None);
        if let (Some(shown), Some(p)) = (&self.shown, placement) {
            let crop = self.crop_offset(device, p.crop);
            let (x, y, w, h) = p.dest;
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &shown.group, &[crop]);
            pass.set_viewport(x, y, w, h, 0.0, 1.0);
            pass.draw(3, 1, 0, 0);
        }
        pass.end();
    }

    /// The dynamic offset of a crop ring slot holding `crop` (written for the next submit).
    fn crop_offset(&self, device: &Device, crop: [f32; 4]) -> u32 {
        if crop == [0.0, 0.0, 1.0, 1.0] {
            return WHOLE * CROP_SLOT as u32;
        }
        let mut slots = self.crop_slots.borrow_mut();
        let (next, held) = &mut *slots;
        // The previous slot already holds it (the same placement as last time).
        let previous = if *next == 1 { CROP_SLOTS as u32 - 1 } else { *next - 1 };
        if held[previous as usize] == crop {
            return previous * CROP_SLOT as u32;
        }
        let slot = *next;
        *next = if slot + 1 == CROP_SLOTS as u32 { 1 } else { slot + 1 };
        held[slot as usize] = crop;
        let bytes: Vec<u8> = crop.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.queue().write_buffer(&self.crops, u64::from(slot) * CROP_SLOT, &bytes);
        slot * CROP_SLOT as u32
    }

    /// Draws the frame into the MetalFX scaler's input (1:1, applying its swizzle), scales it to
    /// `p.scaled` and draws the placement's crop of that to its rectangle of `target`. `false`
    /// if MetalFX can't.
    fn encode_upscaled(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        shown: &Shown,
        target: &TextureView,
        p: Placement,
    ) -> bool {
        let out = p.scaled;
        let sizes = ((shown.width, shown.height), out);
        if self.upscale_failed.get() == Some(sizes) {
            return false;
        }
        let mut upscale = self.upscale.borrow_mut();
        if upscale.as_ref().is_none_or(|u| u.sizes != sizes) {
            *upscale = None;
            let Some(scaler) = device.create_spatial_scaler(sizes.0, out, self.format) else {
                log::warn!("present: MetalFX can't scale {}x{} to {}x{}; using bicubic filtering", sizes.0.0, sizes.0.1, out.0, out.1);
                self.upscale_failed.set(Some(sizes));
                return false;
            };
            log::info!("present: MetalFX scales {}x{} to {}x{}", sizes.0.0, sizes.0.1, out.0, out.1);
            let group = device.create_bind_group(
                "present upscaled",
                &self.layout,
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(scaler.output()) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&self.sampler) },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer { buffer: &self.identity, offset: 0, size: None },
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Buffer { buffer: &self.crops, offset: 0, size: Some(16) },
                    },
                ],
            );
            *upscale = Some(Upscale { scaler, sizes, group });
        }
        let u = upscale.as_ref().expect("made above");
        let mut pass = encoder.begin_render_pass(
            "present metalfx input",
            &[Some(RenderPassColorAttachment::clear(u.scaler.input(), BACKGROUND))],
            None,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &shown.group, &[WHOLE * CROP_SLOT as u32]);
        pass.set_viewport(0.0, 0.0, shown.width as f32, shown.height as f32, 0.0, 1.0);
        pass.draw(3, 1, 0, 0);
        pass.end();
        encoder.spatial_scale(&u.scaler);
        let mut pass =
            encoder.begin_render_pass("present", &[Some(RenderPassColorAttachment::clear(target, BACKGROUND))], None);
        let crop = self.crop_offset(device, p.crop);
        let (x, y, w, h) = p.dest;
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &u.group, &[crop]);
        pass.set_viewport(x.round(), y.round(), w.round(), h.round(), 0.0, 1.0);
        pass.draw(3, 1, 0, 0);
        pass.end();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{Aspect, letterbox, place};

    #[test]
    fn placements() {
        let fit = place(Aspect::Fit, (1280, 720), (2868, 1320));
        assert_eq!(fit.crop, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(fit.scaled, (2347, 1320));
        let fill = place(Aspect::Fill, (1280, 720), (2868, 1320));
        assert_eq!(fill.dest, (0.0, 0.0, 2868.0, 1320.0));
        assert_eq!(fill.scaled, (2868, 1613));
        assert!((fill.crop[1] - (1.0 - 1320.0 / 1613.25) / 2.0).abs() < 1e-5 && fill.crop[0] == 0.0);
        let stretch = place(Aspect::Stretch, (1280, 720), (2868, 1320));
        assert_eq!((stretch.dest, stretch.scaled), ((0.0, 0.0, 2868.0, 1320.0), (2868, 1320)));
    }

    #[test]
    fn letterbox_keeps_aspect() {
        assert_eq!(letterbox((1280, 720), (1280, 720)), (0.0, 0.0, 1280.0, 720.0));
        let (x, y, w, h) = letterbox((640, 480), (1280, 720));
        assert_eq!((w, h), (960.0, 720.0));
        assert_eq!((x, y), (160.0, 0.0));
    }
}
