//! Shows the guest's front buffer (VdSwap's fetch constant) in the presentation target,
//! letterboxed to keep its aspect. The front buffer comes from the texture cache, which loads it
//! from the GPU copy of guest memory when a resolve stored it there.

use gpu_webgpu::*;
use xenos_texture::{SwizzleSource, TextureFetchConstant};

const BLIT: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
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

pub(crate) struct Presenter {
    pipeline: RenderPipeline,
    layout: BindGroupLayout,
    sampler: Sampler,
    /// The front buffer's channel mapping (rewritten per swap: buffers are costly to create).
    swizzle: Buffer,
    shown: Option<Shown>,
    /// Swaps shown so far.
    pub frames: u64,
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
        Presenter { pipeline, layout, sampler, swizzle, shown: None, frames: 0 }
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
            ],
        );
        self.shown = Some(Shown { width: front.width, height: front.height, group });
        self.frames += 1;
        Some((front.width, front.height))
    }

    /// Draws the latest front buffer into `target` (black if there is none yet).
    pub(crate) fn encode(&self, encoder: &mut CommandEncoder, target: &TextureView, width: u32, height: u32) {
        let mut pass =
            encoder.begin_render_pass("present", &[Some(RenderPassColorAttachment::clear(target, BACKGROUND))], None);
        if let Some(shown) = &self.shown {
            let (x, y, w, h) = letterbox((shown.width, shown.height), (width, height));
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &shown.group, &[]);
            pass.set_viewport(x, y, w, h, 0.0, 1.0);
            pass.draw(3, 1, 0, 0);
        }
        pass.end();
    }
}

#[cfg(test)]
mod tests {
    use super::letterbox;

    #[test]
    fn letterbox_keeps_aspect() {
        assert_eq!(letterbox((1280, 720), (1280, 720)), (0.0, 0.0, 1280.0, 720.0));
        let (x, y, w, h) = letterbox((640, 480), (1280, 720));
        assert_eq!((w, h), (960.0, 720.0));
        assert_eq!((x, y), (160.0, 0.0));
    }
}
