//! Test scene shared by the native example/test and the web example: a compute pass fills a
//! storage buffer (a tint plus a number sequence), then a render pass clears the target and
//! draws a full-screen triangle, scissored to the middle 3/4, sampling an uploaded 2x2 texture
//! (red, green / blue, yellow) multiplied by the computed tint.

use gpu_webgpu::*;

pub const CLEAR: Color = Color { r: 0.2, g: 0.4, b: 0.6, a: 1.0 };
pub const CLEAR_RGBA: [u8; 4] = [51, 102, 153, 255];
pub const SEQ_LEN: usize = 60;
/// Storage buffer: `tint: vec4<f32>` followed by `seq: array<u32, 60>`.
pub const STORAGE_SIZE: u64 = 16 + 4 * SEQ_LEN as u64;

const TEXELS: [[u8; 4]; 4] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]];

const COMPUTE: &str = r#"
struct Out {
    tint: vec4<f32>,
    seq: array<u32, 60>,
}
@group(0) @binding(0) var<storage, read_write> out_buf: Out;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 60u) {
        out_buf.seq[id.x] = id.x * id.x + 1u;
    }
    if (id.x == 0u) {
        out_buf.tint = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
}
"#;

const DRAW: &str = r#"
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

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> tint: vec4<f32>;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * tint;
}
"#;

pub struct Scene {
    pub storage: Buffer,
    render_pipeline: RenderPipeline,
    render_group: BindGroup,
    compute_pipeline: ComputePipeline,
    compute_group: BindGroup,
}

impl Scene {
    pub fn new(device: &Device, target_format: TextureFormat) -> Self {
        let compute = device.create_shader_module_wgsl("fill", COMPUTE);
        let draw = device.create_shader_module_wgsl("draw", DRAW);

        let texture = device.create_texture(&TextureDescriptor {
            label: "checker",
            size: extent(2, 2, 1),
            format: TextureFormat::RGBA8Unorm,
            usage: TextureUsage::TextureBinding | TextureUsage::CopyDst,
            ..Default::default()
        });
        let texels: Vec<u8> = TEXELS.concat();
        device.queue().write_texture(&TexelCopyTextureInfo::whole(&texture), &texels, 8, None, extent(2, 2, 1));
        let view = texture.create_view(None);
        let sampler = device.create_sampler(&SamplerDescriptor { label: "nearest", ..Default::default() });

        let storage = device.create_buffer(&BufferDescriptor {
            label: "compute out",
            size: STORAGE_SIZE,
            usage: BufferUsage::Storage | BufferUsage::CopySrc,
            mapped_at_creation: false,
        });

        // Compute: layout derived from the shader.
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "fill",
            layout: None,
            stage: ProgrammableStage::new(&compute, "cs"),
        });
        let compute_group = device.create_bind_group(
            "compute",
            &compute_pipeline.get_bind_group_layout(0),
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer { buffer: &storage, offset: 0, size: None },
            }],
        );

        // Render: explicit layout.
        let layout = device.create_bind_group_layout(
            "draw",
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
                        ty: BufferBindingType::ReadOnlyStorage,
                        has_dynamic_offset: false,
                        min_binding_size: 16,
                    },
                },
            ],
        );
        let pipeline_layout = device.create_pipeline_layout("draw", &[&layout]);
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "triangle",
            layout: Some(&pipeline_layout),
            vertex: ProgrammableStage::new(&draw, "vs"),
            vertex_buffers: &[],
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                stage: ProgrammableStage::new(&draw, "fs"),
                targets: &[ColorTargetState::new(target_format)],
            }),
        });
        let render_group = device.create_bind_group(
            "draw",
            &layout,
            &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer { buffer: &storage, offset: 0, size: Some(16) },
                },
            ],
        );
        Self { storage, render_pipeline, render_group, compute_pipeline, compute_group }
    }

    /// Records the compute pass and the render pass into `encoder`.
    pub fn encode(&self, encoder: &mut CommandEncoder, target: &TextureView, width: u32, height: u32) {
        let mut cpass = encoder.begin_compute_pass("fill");
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.compute_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
        cpass.end();

        let mut rpass =
            encoder.begin_render_pass("draw", &[Some(RenderPassColorAttachment::clear(target, CLEAR))], None);
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &self.render_group, &[]);
        rpass.set_scissor_rect(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        rpass.draw(3, 1, 0, 0);
        rpass.end();
    }
}

/// Probe points (fractions of the target size) and the RGBA expected there.
pub const PROBES: [(f32, f32, [u8; 4]); 6] = [
    (0.0625, 0.0625, CLEAR_RGBA),
    (0.25, 0.25, TEXELS[0]),
    (0.75, 0.25, TEXELS[1]),
    (0.25, 0.75, TEXELS[2]),
    (0.75, 0.75, TEXELS[3]),
    (0.9375, 0.9375, CLEAR_RGBA),
];

/// Checks `pixel(x, y) -> RGBA` against [`PROBES`] (±2 per channel); returns the failures.
pub fn check(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<String> {
    let mut failures = Vec::new();
    for (fx, fy, want) in PROBES {
        let (x, y) = ((fx * width as f32) as u32, (fy * height as f32) as u32);
        let got = pixel(x, y);
        if got.iter().zip(want).any(|(&g, w)| g.abs_diff(w) > 2) {
            failures.push(format!("pixel ({x},{y}): got {got:?}, want {want:?}"));
        }
    }
    failures
}

/// Expected `seq` contents written by the compute shader.
pub fn expected_seq() -> Vec<u32> {
    (0..SEQ_LEN as u32).map(|i| i * i + 1).collect()
}
