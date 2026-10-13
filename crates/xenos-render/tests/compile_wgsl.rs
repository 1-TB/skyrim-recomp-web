//! Compiles a directory of translated shaders (`*_Vertex.wgsl` / `*_Pixel.wgsl`, as written by
//! xenos-shader's `XENOS_SHADER_DUMP`) into real render pipelines, so the host WebGPU
//! implementation's whole shader path runs, not just WGSL validation. Ignored: needs a dump
//! directory in `XENOS_WGSL_DIR` and a WebGPU adapter.
#![cfg(not(target_os = "emscripten"))]

use std::path::PathBuf;

use gpu_webgpu::*;

/// Vertex stage for pixel shaders: position plus all 16 interpolators, which every translated
/// pixel shader can link against.
const PASSTHROUGH_VS: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) i0: vec4<f32>, @location(1) i1: vec4<f32>, @location(2) i2: vec4<f32>,
    @location(3) i3: vec4<f32>, @location(4) i4: vec4<f32>, @location(5) i5: vec4<f32>,
    @location(6) i6: vec4<f32>, @location(7) i7: vec4<f32>, @location(8) i8: vec4<f32>,
    @location(9) i9: vec4<f32>, @location(10) i10: vec4<f32>, @location(11) i11: vec4<f32>,
    @location(12) i12: vec4<f32>, @location(13) i13: vec4<f32>, @location(14) i14: vec4<f32>,
    @location(15) i15: vec4<f32>,
}
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var o: VsOut;
    o.pos = vec4<f32>(f32(i), 0.0, 0.0, 1.0);
    return o;
}
"#;

#[test]
#[ignore]
fn compile_translated_shaders() {
    let Some(dir) = std::env::var_os("XENOS_WGSL_DIR").map(PathBuf::from) else {
        eprintln!("XENOS_WGSL_DIR not set");
        return;
    };
    let instance = Instance::new().expect("instance");
    let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions::default())).expect("adapter");
    eprintln!("adapter max inter-stage variables: {}", adapter.limits().maxInterStageShaderVariables);
    let device = instance
        .block_on(
            adapter.request_device(&DeviceDescriptor { required_limits: Some(adapter.limits()), ..Default::default() }),
        )
        .expect("device");
    let passthrough = device.create_shader_module_wgsl("passthrough", PASSTHROUGH_VS);
    let targets: Vec<ColorTargetState> = (0..4)
        .map(|_| ColorTargetState {
            write_mask: ColorWriteMask::None,
            ..ColorTargetState::new(TextureFormat::RGBA16Float)
        })
        .collect();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("dump dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wgsl"))
        .collect();
    files.sort();
    let mut failures = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let vertex = name.contains("_Vertex");
        device.push_error_scope(ErrorFilter::Validation);
        let module = device.create_shader_module_wgsl(&name, &source);
        let _pipeline = if vertex {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: &name,
                layout: None,
                vertex: ProgrammableStage::new(&module, "xe_vs_main"),
                vertex_buffers: &[],
                primitive: PrimitiveState::default(),
                // WebGPU needs at least one attachment; a depth-only pass is the vertex-only case.
                depth_stencil: Some(DepthStencilState::depth(TextureFormat::Depth32Float, true, CompareFunction::Less)),
                multisample: MultisampleState::default(),
                fragment: None,
            })
        } else {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: &name,
                layout: None,
                vertex: ProgrammableStage::new(&passthrough, "vs"),
                vertex_buffers: &[],
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    stage: ProgrammableStage::new(&module, "xe_ps_main"),
                    targets: &targets,
                }),
            })
        };
        if let Some(e) = device.block_on(device.pop_error_scope()).expect("error scope") {
            failures.push(format!("{name}: {e}"));
        }
    }
    eprintln!("{} shaders, {} failed", files.len(), failures.len());
    for f in failures.iter().take(10) {
        eprintln!("{f}");
    }
    assert!(failures.is_empty());
}
