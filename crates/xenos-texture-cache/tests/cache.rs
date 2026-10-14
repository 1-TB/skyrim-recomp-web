//! Needs a WebGPU adapter: a GPU, or lavapipe via VK_ICD_FILENAMES.
#![cfg(not(target_os = "emscripten"))]

use gpu_webgpu::*;
use xenos_shader::{TextureBinding, TextureBindingDimension, parsed};
use xenos_texture::address::{tiled_2d, tiled_3d};
use xenos_texture::{
    AnisoFilter, DataDimension, Endian, FetchConstantType, TextureFetchConstant, TextureFilter, TextureGuestInfo,
};
use xenos_texture_cache::*;

type GuestFormat = xenos_texture::TextureFormat;

const MEMORY_SIZE: usize = 8 << 20;

struct Gpu {
    _instance: Instance,
    device: Device,
}

/// `optional_features`: request BC, float32-filterable and 16-bit normalized formats when the
/// adapter has them.
fn gpu(optional_features: bool) -> Gpu {
    let instance = Instance::new().expect("instance");
    let adapter = instance
        .block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .expect("no WebGPU adapter (set VK_ICD_FILENAMES to lavapipe on GPU-less machines)");
    let wanted = [
        FeatureName::TextureCompressionBC,
        FeatureName::TextureCompressionBCSliced3D,
        FeatureName::Float32Filterable,
        FeatureName::TextureFormatsTier1,
        NATIVE_TEXTURE_FORMAT_16BIT_NORM,
    ];
    let features: Vec<FeatureName> =
        if optional_features { wanted.into_iter().filter(|&f| adapter.has_feature(f)).collect() } else { vec![] };
    let device = instance
        .block_on(adapter.request_device(&DeviceDescriptor {
            label: "texture cache test",
            required_features: &features,
            required_limits: Some(adapter.limits()),
        }))
        .expect("device");
    Gpu { _instance: instance, device }
}

fn binding(slot: u32, dimension: TextureBindingDimension) -> TextureBinding {
    TextureBinding {
        fetch_constant: slot,
        dimension,
        texture_binding: 0,
        stacked_texture_binding: (dimension == TextureBindingDimension::D3OrStacked).then_some(1),
        sampler_binding: 2,
        mag_filter: parsed::TextureFilter::UseFetchConst,
        min_filter: parsed::TextureFilter::UseFetchConst,
        mip_filter: parsed::TextureFilter::UseFetchConst,
        aniso_filter: parsed::AnisoFilter::UseFetchConst,
    }
}

fn constants(slots: &[(u32, &TextureFetchConstant)]) -> [u32; FETCH_CONSTANT_DWORDS] {
    let mut fc = [0u32; FETCH_CONSTANT_DWORDS];
    for &(slot, fetch) in slots {
        let s = slot as usize * 6;
        fc[s..s + 6].copy_from_slice(&fetch.encode());
    }
    fc
}

fn texture_2d(format: GuestFormat, width: u32, height: u32, base: u32, tiled: bool) -> TextureFetchConstant {
    let mut f = TextureFetchConstant::new_2d(format, width, height);
    f.base_page = base >> 12;
    f.tiled = tiled;
    f
}

/// Writes one stored level of `fetch` into guest memory. `block(x, y, slice)` returns a block's
/// bytes in host (little-endian) order; the fetch constant's endian swap is applied here.
fn write_level(mem: &mut [u8], fetch: &TextureFetchConstant, level: u32, block: impl Fn(u32, u32, u32) -> Vec<u8>) {
    let info = TextureGuestInfo::from_fetch(fetch);
    let (address, layout, _) = info.level_storage(level).expect("level stored");
    let fi = info.format.info();
    let (bw, bh, bpb, log2) = (fi.block_width, fi.block_height, fi.bytes_per_block(), fi.bytes_per_block_log2());
    let sub = info.subresources;
    let is_3d = sub.dimension == DataDimension::D3;
    let tiled = fetch.tiled && sub.dimension != DataDimension::D1;
    let width = (sub.width >> level).max(1);
    let height = (sub.height >> level).max(1);
    let slices = if is_3d { (sub.depth() >> level).max(1) } else { sub.array_size() };
    let pitch_blocks = layout.row_pitch_bytes / bpb;
    let z_rows = layout.z_slice_stride_block_rows;
    for s in 0..slices {
        for by in 0..height.div_ceil(bh) {
            for bx in 0..width.div_ceil(bw) {
                let offset = if tiled && is_3d {
                    tiled_3d(bx as i32, by as i32, s as i32, pitch_blocks, z_rows, log2) as u32
                } else if tiled {
                    s * layout.array_slice_stride_bytes + tiled_2d(bx as i32, by as i32, pitch_blocks, log2) as u32
                } else if is_3d {
                    (s * z_rows + by) * layout.row_pitch_bytes + bx * bpb
                } else {
                    s * layout.array_slice_stride_bytes + by * layout.row_pitch_bytes + bx * bpb
                };
                let mut bytes = block(bx, by, s);
                assert_eq!(bytes.len(), bpb as usize);
                fetch.endianness.swap_in_place(&mut bytes);
                let at = (address + offset) as usize;
                mem[at..at + bytes.len()].copy_from_slice(&bytes);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Read {
    D2,
    Array,
    D3,
    Cube,
}

/// Runs `expr` (a vec4<f32> using `t`, `s` and `id`) for every invocation of a
/// `count` dispatch and returns the results, indexed `x + y * count.0 + z * count.0 * count.1`.
fn run(gpu: &Gpu, bound: &BoundTexture, read: Read, expr: &str, count: (u32, u32, u32)) -> Vec<[f32; 4]> {
    let device = &gpu.device;
    device.push_error_scope(ErrorFilter::Validation);
    let (decl, view_dimension, view) = match read {
        Read::D2 => ("texture_2d<f32>", TextureViewDimension::D2, &bound.view),
        Read::Array => ("texture_2d_array<f32>", TextureViewDimension::D2Array, bound.stacked_view.as_ref().unwrap()),
        Read::D3 => ("texture_3d<f32>", TextureViewDimension::D3, &bound.view),
        Read::Cube => ("texture_cube<f32>", TextureViewDimension::Cube, &bound.view),
    };
    let wgsl = format!(
        "@group(0) @binding(0) var t: {decl};
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var s: sampler;
fn cube_dir(face: u32, x: u32, y: u32, size: vec2<f32>) -> vec3<f32> {{
    let sc = (f32(x) + 0.5) / size.x * 2.0 - 1.0;
    let tc = (f32(y) + 0.5) / size.y * 2.0 - 1.0;
    switch (face) {{
        case 0u: {{ return vec3<f32>(1.0, -tc, -sc); }}
        case 1u: {{ return vec3<f32>(-1.0, -tc, sc); }}
        case 2u: {{ return vec3<f32>(sc, 1.0, tc); }}
        case 3u: {{ return vec3<f32>(sc, -1.0, -tc); }}
        case 4u: {{ return vec3<f32>(sc, -tc, 1.0); }}
        default: {{ return vec3<f32>(-sc, -tc, -1.0); }}
    }}
}}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    out[id.x + id.y * {x}u + id.z * {xy}u] = {expr};
}}
",
        x = count.0,
        xy = count.0 * count.1
    );
    let module = device.create_shader_module_wgsl("readback", &wgsl);
    let layout = device.create_bind_group_layout(
        "readback",
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStage::Compute,
                ty: BindingType::Texture { sample_type: bound.sample_type, view_dimension, multisampled: false },
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStage::Compute,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage,
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                },
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStage::Compute,
                ty: BindingType::Sampler(bound.sampler_binding_type()),
            },
        ],
    );
    let pipeline_layout = device.create_pipeline_layout("readback", &[&layout]);
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: "readback",
        layout: Some(&pipeline_layout),
        stage: ProgrammableStage::new(&module, "main"),
    });
    let n = (count.0 * count.1 * count.2) as u64;
    let storage = device.create_buffer(&BufferDescriptor {
        label: "texels",
        size: n * 16,
        usage: BufferUsage::Storage | BufferUsage::CopySrc,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: "readback",
        size: n * 16,
        usage: BufferUsage::MapRead | BufferUsage::CopyDst,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "readback",
        &layout,
        &[
            BindGroupEntry { binding: 0, resource: BindingResource::TextureView(view) },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer { buffer: &storage, offset: 0, size: None },
            },
            BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&bound.sampler) },
        ],
    );
    let mut encoder = device.create_command_encoder("readback");
    {
        let mut pass = encoder.begin_compute_pass("readback");
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.0, count.1, count.2);
        pass.end();
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, n * 16);
    device.queue().submit([encoder.finish("readback")]);
    let error = device.block_on(device.pop_error_scope()).expect("pop error scope");
    assert!(error.is_none(), "validation error: {error:?}");
    device.block_on(readback.map_async(MapMode::Read, 0, None)).expect("map");
    let mut bytes = vec![0u8; (n * 16) as usize];
    readback.read_mapped(0, &mut bytes).expect("read");
    readback.unmap();
    bytes
        .chunks(16)
        .map(|c| std::array::from_fn(|i| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap())))
        .collect()
}

/// Every texel of a level through `textureLoad`.
fn load_texels(gpu: &Gpu, bound: &BoundTexture, read: Read, size: (u32, u32, u32), level: u32) -> Vec<[f32; 4]> {
    let expr = match read {
        Read::D2 => format!("textureLoad(t, vec2<i32>(id.xy), {level})"),
        Read::Array => format!("textureLoad(t, vec2<i32>(id.xy), i32(id.z), {level})"),
        Read::D3 => format!("textureLoad(t, vec3<i32>(id), {level})"),
        Read::Cube => {
            format!("textureSampleLevel(t, s, cube_dir(id.z, id.x, id.y, vec2<f32>({}.0, {}.0)), 0.0)", size.0, size.1)
        }
    };
    run(gpu, bound, read, &expr, size)
}

fn assert_close(actual: [f32; 4], expected: [f32; 4], tolerance: f32, what: &str) {
    let ok = actual.iter().zip(expected).all(|(a, e)| (a - e).abs() <= tolerance);
    assert!(ok, "{what}: got {actual:?}, expected {expected:?}");
}

fn unorm(bytes: [u8; 4]) -> [f32; 4] {
    bytes.map(|b| f32::from(b) / 255.0)
}

fn rgba_pattern(x: u32, y: u32, s: u32) -> [u8; 4] {
    [(x * 7 + s * 50) as u8, (y * 11) as u8, (x ^ y) as u8 * 3, 255 - s as u8]
}

fn assert_no_errors(gpu: &Gpu) {
    let errors = gpu.device.take_errors();
    assert!(errors.is_empty(), "uncaptured errors: {errors:?}");
}

#[test]
fn tiled_2d_rgba8_loads_once_and_is_shared() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut fetch = texture_2d(GuestFormat::k_8_8_8_8, 64, 32, 0x10000, true);
    fetch.endianness = Endian::K8in32;
    write_level(&mut mem, &fetch, 0, |x, y, _| rgba_pattern(x, y, 0).to_vec());
    // A second fetch constant with the same data but different sampling state.
    let mut other = fetch.clone();
    other.swizzle = [xenos_texture::SwizzleSource::W; 4];
    other.mag_filter = TextureFilter::Point;
    other.format = GuestFormat::k_8_8_8_8_AS_16_16_16_16;
    let fc = constants(&[(3, &fetch), (9, &other)]);

    let mut cache = TextureCache::for_device(&gpu.device);
    let a = cache.bind(&gpu.device, &mem, &fc, &binding(3, TextureBindingDimension::D2));
    assert!(!a.fallback && a.filterable && a.stacked_view.is_none());
    assert_eq!(a.sample_type, TextureSampleType::Float);
    let texels = load_texels(&gpu, &a, Read::D2, (64, 32, 1), 0);
    for y in 0..32 {
        for x in 0..64 {
            assert_close(texels[(y * 64 + x) as usize], unorm(rgba_pattern(x, y, 0)), 1e-6, &format!("({x}, {y})"));
        }
    }
    let again = cache.bind(&gpu.device, &mem, &fc, &binding(3, TextureBindingDimension::D2));
    assert_eq!(again.view, a.view);
    assert_eq!(again.sampler, a.sampler);
    let shared = cache.bind(&gpu.device, &mem, &fc, &binding(9, TextureBindingDimension::D2));
    assert_eq!(shared.view, a.view, "aliases with identical data share the texture");
    assert_ne!(shared.sampler, a.sampler, "different filters need another sampler");
    let stats = cache.stats();
    assert_eq!((stats.loads, stats.hits, stats.textures, stats.samplers), (1, 2, 1, 2));
    assert_eq!(stats.host_bytes, 64 * 32 * 4);
    assert_no_errors(&gpu);
}

#[test]
fn linear_mips_and_1d() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut fetch = texture_2d(GuestFormat::k_8_8_8_8, 64, 64, 0x20000, false);
    fetch.mip_page = 0x40000 >> 12;
    fetch.mip_max_level = 3;
    for level in 0..=3 {
        write_level(&mut mem, &fetch, level, |x, y, _| rgba_pattern(x, y, level).to_vec());
    }
    let mut line = TextureFetchConstant::new_2d(GuestFormat::k_8_8_8_8, 40, 1);
    line.dimension = DataDimension::D1;
    line.base_page = 0x80000 >> 12;
    write_level(&mut mem, &line, 0, |x, _, _| rgba_pattern(x, 0, 7).to_vec());
    let fc = constants(&[(0, &fetch), (1, &line)]);

    let mut cache = TextureCache::for_device(&gpu.device);
    let bound = cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D2));
    for level in 0..=3u32 {
        let size = 64 >> level;
        let texels = load_texels(&gpu, &bound, Read::D2, (size, size, 1), level);
        for y in 0..size {
            for x in 0..size {
                let expected = unorm(rgba_pattern(x, y, level));
                assert_close(texels[(y * size + x) as usize], expected, 1e-6, &format!("level {level} ({x}, {y})"));
            }
        }
    }
    let bound = cache.bind(&gpu.device, &mem, &fc, &binding(1, TextureBindingDimension::D2));
    let texels = load_texels(&gpu, &bound, Read::D2, (40, 1, 1), 0);
    for x in 0..40 {
        assert_close(texels[x as usize], unorm(rgba_pattern(x, 0, 7)), 1e-6, &format!("1D texel {x}"));
    }
    assert_no_errors(&gpu);
}

fn rgb565(r: u32, g: u32, b: u32) -> u16 {
    ((r << 11) | (g << 5) | b) as u16
}

#[test]
fn dxt1_passthrough_matches_cpu_decompression() {
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut fetch = texture_2d(GuestFormat::k_DXT1, 32, 32, 0x30000, true);
    fetch.endianness = Endian::K8in16;
    // Solid blocks: color0 > color1 (4-color mode), every index 0.
    let color = |bx: u32, by: u32| rgb565(bx * 4 + 3, by * 8 + 1, (bx + by) * 2 + 1);
    write_level(&mut mem, &fetch, 0, |bx, by, _| {
        let c0 = color(bx, by).to_le_bytes();
        vec![c0[0], c0[1], 0, 0, 0, 0, 0, 0]
    });
    let fc = constants(&[(5, &fetch)]);
    let expected = |x: u32, y: u32| {
        let c = u32::from(color(x / 4, y / 4));
        [((c >> 11) & 31) as f32 / 31.0, ((c >> 5) & 63) as f32 / 63.0, (c & 31) as f32 / 31.0, 1.0]
    };
    for features in [true, false] {
        let gpu = gpu(features);
        let mut cache = TextureCache::for_device(&gpu.device);
        assert_eq!(
            cache.caps().texture_compression_bc,
            features && gpu.device.has_feature(FeatureName::TextureCompressionBC)
        );
        let bound = cache.bind(&gpu.device, &mem, &fc, &binding(5, TextureBindingDimension::D2));
        assert!(!bound.fallback);
        let texels = load_texels(&gpu, &bound, Read::D2, (32, 32, 1), 0);
        for y in 0..32 {
            for x in 0..32 {
                let what = format!("bc={features} ({x}, {y})");
                assert_close(texels[(y * 32 + x) as usize], expected(x, y), 0.02, &what);
            }
        }
        // BC passthrough keeps the compressed size.
        let bytes = cache.stats().host_bytes;
        assert_eq!(bytes, if cache.caps().texture_compression_bc { 64 * 8 } else { 32 * 32 * 4 });
        assert_no_errors(&gpu);
    }
}

#[test]
fn format_classes() {
    for features in [true, false] {
        let gpu = gpu(features);
        let mut mem = vec![0u8; MEMORY_SIZE];
        let float32_filterable = gpu.device.has_feature(FeatureName::Float32Filterable);
        struct Case {
            format: GuestFormat,
            block: fn(u32, u32) -> Vec<u8>,
            expected: fn(u32, u32) -> [f32; 4],
            tolerance: f32,
            filterable: Option<bool>,
        }
        let cases = [
            Case {
                format: GuestFormat::k_8,
                block: |x, y| vec![(x * 16 + y) as u8],
                expected: |x, y| [(x * 16 + y) as f32 / 255.0, 0.0, 0.0, 1.0],
                tolerance: 1e-6,
                filterable: Some(true),
            },
            Case {
                format: GuestFormat::k_5_6_5,
                block: |x, y| ((x & 31) | ((y * 4) << 5) | ((31 - x) << 11)).to_le_bytes()[..2].to_vec(),
                expected: |x, y| [(x & 31) as f32 / 31.0, (y * 4) as f32 / 63.0, (31 - x) as f32 / 31.0, 1.0],
                tolerance: 0.003,
                filterable: Some(true),
            },
            Case {
                format: GuestFormat::k_16_16_FLOAT,
                block: |x, _| [0x3800u16 + x as u16, 0xC000].iter().flat_map(|h| h.to_le_bytes()).collect(),
                expected: |x, _| [0.5 + x as f32 / 2048.0, -2.0, 0.0, 1.0],
                tolerance: 1e-6,
                filterable: Some(true),
            },
            Case {
                format: GuestFormat::k_32_FLOAT,
                block: |x, y| (x as f32 * 1.5 - y as f32 * 1000.0).to_le_bytes().to_vec(),
                expected: |x, y| [x as f32 * 1.5 - y as f32 * 1000.0, 0.0, 0.0, 1.0],
                tolerance: 0.0,
                filterable: None,
            },
            Case {
                format: GuestFormat::k_16,
                block: |x, y| ((x * 4096 + y) as u16).to_le_bytes().to_vec(),
                expected: |x, y| [(x * 4096 + y) as f32 / 65535.0, 0.0, 0.0, 1.0],
                tolerance: 2e-3,
                filterable: Some(true),
            },
            Case {
                format: GuestFormat::k_2_10_10_10,
                block: |x, y| ((x * 60) | ((y * 100) << 10) | (1023 << 20) | (2 << 30)).to_le_bytes().to_vec(),
                expected: |x, y| [(x * 60) as f32 / 1023.0, (y * 100) as f32 / 1023.0, 1.0, 2.0 / 3.0],
                tolerance: 1e-4,
                filterable: Some(true),
            },
            Case {
                format: GuestFormat::k_DXN,
                block: |x, y| vec![(x * 30) as u8, 0, 0, 0, 0, 0, 0, 0, (y * 30) as u8, 0, 0, 0, 0, 0, 0, 0],
                expected: |x, y| [(x / 4 * 30) as f32 / 255.0, (y / 4 * 30) as f32 / 255.0, 0.0, 1.0],
                tolerance: 0.004,
                filterable: Some(true),
            },
        ];
        let mut cache = TextureCache::for_device(&gpu.device);
        let norm16 = gpu.device.has_feature(NATIVE_TEXTURE_FORMAT_16BIT_NORM)
            || gpu.device.has_feature(FeatureName::TextureFormatsTier1);
        assert_eq!(cache.caps().norm16, norm16);
        for (i, case) in cases.iter().enumerate() {
            let fetch = texture_2d(case.format, 8, 8, 0x10000 * (i as u32 + 1), true);
            write_level(&mut mem, &fetch, 0, |x, y, _| (case.block)(x, y));
            let fc = constants(&[(i as u32, &fetch)]);
            let bound = cache.bind(&gpu.device, &mem, &fc, &binding(i as u32, TextureBindingDimension::D2));
            let name = case.format.name();
            assert!(!bound.fallback, "{name}");
            let filterable = case.filterable.unwrap_or(float32_filterable);
            assert_eq!(bound.filterable, filterable, "{name}");
            let sample_type = if filterable { TextureSampleType::Float } else { TextureSampleType::UnfilterableFloat };
            assert_eq!(bound.sample_type, sample_type, "{name}");
            let texels = load_texels(&gpu, &bound, Read::D2, (8, 8, 1), 0);
            for y in 0..8 {
                for x in 0..8 {
                    let mut expected = (case.expected)(x, y);
                    // Host formats with fewer channels read back 0 for G/B and 1 for A.
                    if case.format == GuestFormat::k_DXN {
                        expected[2] = 0.0;
                    }
                    let what = format!("{name} features={features} ({x}, {y})");
                    assert_close(texels[(y * 8 + x) as usize], expected, case.tolerance, &what);
                }
            }
            // Sampling through the bound sampler works with the reported binding types.
            let sampled =
                run(&gpu, &bound, Read::D2, "textureSampleLevel(t, s, vec2<f32>(0.0625, 0.0625), 0.0)", (1, 1, 1));
            let mut expected = (case.expected)(0, 0);
            if case.format == GuestFormat::k_DXN {
                expected[2] = 0.0;
            }
            assert_close(sampled[0], expected, case.tolerance, &format!("{name} sampled"));
        }
        assert_no_errors(&gpu);
    }
}

#[test]
fn cube_maps_and_their_stacked_view() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut fetch = texture_2d(GuestFormat::k_8_8_8_8, 8, 8, 0x100000, true);
    fetch.dimension = DataDimension::Cube;
    fetch.depth = 6;
    write_level(&mut mem, &fetch, 0, |x, y, face| rgba_pattern(x, y, face).to_vec());
    let fc = constants(&[(2, &fetch)]);
    let mut cache = TextureCache::for_device(&gpu.device);

    let cube = cache.bind(&gpu.device, &mem, &fc, &binding(2, TextureBindingDimension::Cube));
    assert!(!cube.fallback);
    let texels = load_texels(&gpu, &cube, Read::Cube, (8, 8, 6), 0);
    for face in 0..6 {
        for y in 0..8 {
            for x in 0..8 {
                let i = (face * 64 + y * 8 + x) as usize;
                assert_close(texels[i], unorm(rgba_pattern(x, y, face)), 1e-6, &format!("face {face} ({x}, {y})"));
            }
        }
    }
    // Bound to a tfetch3D: the faces are the stacked layers, the 3D view is a zero dummy.
    let stacked = cache.bind(&gpu.device, &mem, &fc, &binding(2, TextureBindingDimension::D3OrStacked));
    assert!(!stacked.fallback);
    let layers = load_texels(&gpu, &stacked, Read::Array, (8, 8, 6), 0);
    assert_eq!(layers, texels);
    assert_eq!(load_texels(&gpu, &stacked, Read::D3, (1, 1, 1), 0), vec![[0.0; 4]]);
    // As a tfetch2D: face 0.
    let flat = cache.bind(&gpu.device, &mem, &fc, &binding(2, TextureBindingDimension::D2));
    assert_eq!(load_texels(&gpu, &flat, Read::D2, (8, 8, 1), 0), texels[..64]);
    assert_eq!(cache.stats().textures, 1);
    assert_no_errors(&gpu);
}

#[test]
fn stacked_and_3d_textures() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut stacked = texture_2d(GuestFormat::k_8_8_8_8, 8, 8, 0x100000, true);
    stacked.stacked = true;
    stacked.depth = 4;
    write_level(&mut mem, &stacked, 0, |x, y, s| rgba_pattern(x, y, s).to_vec());
    let mut volume = texture_2d(GuestFormat::k_8_8_8_8, 16, 8, 0x200000, true);
    volume.dimension = DataDimension::D3;
    volume.depth = 4;
    write_level(&mut mem, &volume, 0, |x, y, z| rgba_pattern(x, y, z + 1).to_vec());
    let mut linear_volume = volume.clone();
    linear_volume.tiled = false;
    linear_volume.base_page = 0x300000 >> 12;
    write_level(&mut mem, &linear_volume, 0, |x, y, z| rgba_pattern(x, y, z + 1).to_vec());
    let fc = constants(&[(0, &stacked), (1, &volume), (2, &linear_volume)]);
    let mut cache = TextureCache::for_device(&gpu.device);

    let bound = cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D3OrStacked));
    let layers = load_texels(&gpu, &bound, Read::Array, (8, 8, 4), 0);
    for s in 0..4 {
        for y in 0..8 {
            for x in 0..8 {
                let i = (s * 64 + y * 8 + x) as usize;
                assert_close(layers[i], unorm(rgba_pattern(x, y, s)), 1e-6, &format!("layer {s} ({x}, {y})"));
            }
        }
    }
    assert_eq!(load_texels(&gpu, &bound, Read::D3, (1, 1, 1), 0), vec![[0.0; 4]]);

    for slot in [1, 2] {
        let bound = cache.bind(&gpu.device, &mem, &fc, &binding(slot, TextureBindingDimension::D3OrStacked));
        assert!(!bound.fallback);
        let texels = load_texels(&gpu, &bound, Read::D3, (16, 8, 4), 0);
        for z in 0..4 {
            for y in 0..8 {
                for x in 0..16 {
                    let i = (z * 128 + y * 16 + x) as usize;
                    let what = format!("slot {slot} ({x}, {y}, {z})");
                    assert_close(texels[i], unorm(rgba_pattern(x, y, z + 1)), 1e-6, &what);
                }
            }
        }
        assert_eq!(load_texels(&gpu, &bound, Read::Array, (1, 1, 1), 0), vec![[0.0; 4]]);
    }
    // A 3D texture cannot be a tfetch2D: fallback, but valid.
    let mismatch = cache.bind(&gpu.device, &mem, &fc, &binding(1, TextureBindingDimension::D2));
    assert!(mismatch.fallback);
    assert_eq!(load_texels(&gpu, &mismatch, Read::D2, (1, 1, 1), 0), vec![[0.0; 4]]);
    assert_no_errors(&gpu);
}

#[test]
fn revalidation_and_invalidation() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let a = texture_2d(GuestFormat::k_8_8_8_8, 32, 32, 0x10000, true);
    let b = texture_2d(GuestFormat::k_8_8_8_8, 32, 32, 0x20000, true);
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 0).to_vec());
    write_level(&mut mem, &b, 0, |x, y, _| rgba_pattern(x, y, 1).to_vec());
    let fc = constants(&[(0, &a), (1, &b)]);
    let bind_a = binding(0, TextureBindingDimension::D2);
    let bind_b = binding(1, TextureBindingDimension::D2);
    let mut cache = TextureCache::for_device(&gpu.device);

    let first = cache.bind(&gpu.device, &mem, &fc, &bind_a);
    // CPU writes are only noticed on the first use after a frame boundary.
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 2).to_vec());
    assert_eq!(cache.bind(&gpu.device, &mem, &fc, &bind_a).view, first.view);
    cache.end_frame();
    let hashed = cache.stats().hashed_bytes;
    let second = cache.bind(&gpu.device, &mem, &fc, &bind_a);
    assert_ne!(second.view, first.view);
    assert_eq!(cache.stats().reloads, 1);
    assert_eq!(cache.stats().hashed_bytes - hashed, 2 * 32 * 32 * 4, "revalidation hash + reload hash");
    let texels = load_texels(&gpu, &second, Read::D2, (32, 32, 1), 0);
    assert_close(texels[33], unorm(rgba_pattern(1, 1, 2)), 1e-6, "reloaded texel");
    // The replaced texture stays usable for bind groups made from it.
    let old = load_texels(&gpu, &first, Read::D2, (32, 32, 1), 0);
    assert_close(old[33], unorm(rgba_pattern(1, 1, 0)), 1e-6, "old texel");

    // Unchanged memory: hashed once per frame, same texture.
    cache.end_frame();
    let hashed = cache.stats().hashed_bytes;
    assert_eq!(cache.bind(&gpu.device, &mem, &fc, &bind_a).view, second.view);
    assert_eq!(cache.bind(&gpu.device, &mem, &fc, &bind_a).view, second.view);
    assert_eq!(cache.stats().hashed_bytes - hashed, 32 * 32 * 4);

    // Explicit invalidation drops overlapping textures immediately, others stay.
    let tex_b = cache.bind(&gpu.device, &mem, &fc, &bind_b);
    cache.invalidate(0x10000 + 100, 4);
    assert_eq!(cache.stats().invalidated, 1);
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 3).to_vec());
    let third = cache.bind(&gpu.device, &mem, &fc, &bind_a);
    assert_ne!(third.view, second.view);
    assert_close(load_texels(&gpu, &third, Read::D2, (32, 32, 1), 0)[33], unorm(rgba_pattern(1, 1, 3)), 1e-6, "c");
    assert_eq!(cache.bind(&gpu.device, &mem, &fc, &bind_b).view, tex_b.view);
    // Ranges ending right before / starting right after do not overlap; virtual addresses
    // are masked to physical ones.
    cache.invalidate(0x10000 - 16, 16);
    cache.invalidate(0x20000 + 32 * 32 * 4, 64);
    assert_eq!(cache.stats().invalidated, 1);
    cache.invalidate(0xA002_0000, 1);
    assert_eq!(cache.stats().invalidated, 2);
    assert_no_errors(&gpu);
}

#[test]
fn dropped_textures_are_reused_with_their_views() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let a = texture_2d(GuestFormat::k_8_8_8_8, 32, 32, 0x10000, true);
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 0).to_vec());
    let fc = constants(&[(0, &a)]);
    let bind_a = binding(0, TextureBindingDimension::D2);
    let mut cache = TextureCache::for_device(&gpu.device);

    let first = cache.bind(&gpu.device, &mem, &fc, &bind_a);
    cache.invalidate(0x10000, 4);
    // Dropped textures are kept for a frame (work recorded with them may not be submitted yet)
    // before they can be reused.
    cache.destroy_retired();
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 1).to_vec());
    let second = cache.bind(&gpu.device, &mem, &fc, &bind_a);
    assert_ne!(second.view, first.view, "still retiring: a new texture");
    assert_eq!(cache.stats().reused, 0);
    cache.destroy_retired();
    cache.invalidate(0x10000, 4);
    cache.destroy_retired();
    cache.destroy_retired();
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 2).to_vec());
    let third = cache.bind(&gpu.device, &mem, &fc, &bind_a);
    assert_eq!(cache.stats().reused, 1);
    assert!(third.view == first.view || third.view == second.view, "a pooled texture and its view");
    assert_close(
        load_texels(&gpu, &third, Read::D2, (32, 32, 1), 0)[33],
        unorm(rgba_pattern(1, 1, 2)),
        1e-6,
        "new data",
    );
    assert_no_errors(&gpu);
}

#[test]
fn invalidated_textures_reload_in_place_from_gpu_memory() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let a = texture_2d(GuestFormat::k_8_8_8_8, 32, 32, 0x10000, true);
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 0).to_vec());
    let fc = constants(&[(0, &a)]);
    let bind_a = binding(0, TextureBindingDimension::D2);
    let device = &gpu.device;
    let shared = device.create_buffer_init(
        "guest memory",
        BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc,
        &mem,
    );
    let mut source = WholeMemory { buffers: vec![shared] };
    let mut cache = TextureCache::for_device(device);

    // Invalidated (by a resolve) and loaded again from the GPU copy: the same texture, refilled
    // after the work recorded before.
    let first = cache.bind_with(device, &mem, &fc, &bind_a, Some(&mut source));
    cache.invalidate(0x10000, 4);
    write_level(&mut mem, &a, 0, |x, y, _| rgba_pattern(x, y, 1).to_vec());
    device.queue().write_buffer(&source.buffers[0], 0, &mem);
    let second = cache.bind_with(device, &mem, &fc, &bind_a, Some(&mut source));
    assert_eq!(second.view, first.view);
    assert_eq!(cache.stats().reloaded_in_place, 1);
    let texel = load_texels(&gpu, &second, Read::D2, (32, 32, 1), 0)[33];
    assert_close(texel, unorm(rgba_pattern(1, 1, 1)), 1e-6, "reloaded in place");

    // Loaded from guest memory instead (submitted on its own, ahead of recorded work): a new
    // texture.
    cache.invalidate(0x10000, 4);
    let third = cache.bind(device, &mem, &fc, &bind_a);
    assert_ne!(third.view, second.view);
    assert_eq!(cache.stats().reloaded_in_place, 1);
    assert_no_errors(&gpu);
}

#[test]
fn eviction_by_age_and_budget() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let fetches: Vec<TextureFetchConstant> =
        (0..3).map(|i| texture_2d(GuestFormat::k_8_8_8_8, 32, 32, 0x10000 * (i + 1), true)).collect();
    for (i, f) in fetches.iter().enumerate() {
        write_level(&mut mem, f, 0, |x, y, _| rgba_pattern(x, y, i as u32).to_vec());
    }
    let fc = constants(&[(0, &fetches[0]), (1, &fetches[1]), (2, &fetches[2])]);
    let config = CacheConfig { max_unused_frames: 2, ..Default::default() };
    let mut cache = TextureCache::with_config(&gpu.device, host_caps(&gpu.device), config);
    cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D2));
    cache.end_frame();
    cache.bind(&gpu.device, &mem, &fc, &binding(1, TextureBindingDimension::D2));
    cache.end_frame();
    cache.end_frame();
    // Texture 0 has been unused for two frames (2 and 3): still within the limit.
    assert_eq!((cache.stats().textures, cache.stats().evicted), (2, 0));
    cache.end_frame();
    assert_eq!((cache.stats().textures, cache.stats().evicted), (1, 1));
    cache.end_frame();
    assert_eq!((cache.stats().textures, cache.stats().evicted), (0, 2));

    // Budget of two textures: the least recently used one not used in the last frame goes.
    cache.set_config(CacheConfig { max_unused_frames: 100, max_host_bytes: 2 * 32 * 32 * 4, ..config });
    for slot in 0..3 {
        cache.bind(&gpu.device, &mem, &fc, &binding(slot, TextureBindingDimension::D2));
        cache.end_frame();
    }
    let stats = cache.stats();
    assert_eq!((stats.textures, stats.host_bytes, stats.evicted), (2, 2 * 32 * 32 * 4, 3));
    let loads = stats.loads;
    cache.bind(&gpu.device, &mem, &fc, &binding(2, TextureBindingDimension::D2));
    cache.bind(&gpu.device, &mem, &fc, &binding(1, TextureBindingDimension::D2));
    assert_eq!(cache.stats().loads, loads, "the two most recent textures stayed");
    cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D2));
    assert_eq!(cache.stats().loads, loads + 1);
    // Over budget, but everything was used in the last frame: nothing is evicted.
    cache.end_frame();
    assert_eq!(cache.stats().textures, 3);
    assert_no_errors(&gpu);
}

#[test]
fn fallbacks_for_unusable_fetch_constants() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut cache = TextureCache::for_device(&gpu.device);
    let zero = [0u32; FETCH_CONSTANT_DWORDS];
    for dimension in [TextureBindingDimension::D2, TextureBindingDimension::Cube, TextureBindingDimension::D3OrStacked]
    {
        let bound = cache.bind(&gpu.device, &mem, &zero, &binding(7, dimension));
        assert!(bound.fallback && bound.filterable);
        assert_eq!(bound.stacked_view.is_some(), dimension == TextureBindingDimension::D3OrStacked);
        let read = match dimension {
            TextureBindingDimension::D2 => Read::D2,
            TextureBindingDimension::Cube => Read::Cube,
            TextureBindingDimension::D3OrStacked => Read::D3,
        };
        assert_eq!(load_texels(&gpu, &bound, read, (1, 1, 1), 0), vec![[0.0; 4]]);
        if dimension == TextureBindingDimension::D3OrStacked {
            assert_eq!(load_texels(&gpu, &bound, Read::Array, (1, 1, 1), 0), vec![[0.0; 4]]);
        }
    }

    let unsupported = texture_2d(GuestFormat::k_1, 32, 32, 0x10000, false);
    let mut not_square = texture_2d(GuestFormat::k_8_8_8_8, 16, 8, 0x20000, true);
    not_square.dimension = DataDimension::Cube;
    let mut vertex = texture_2d(GuestFormat::k_8_8_8_8, 16, 16, 0x30000, true);
    vertex.fetch_type = FetchConstantType::Vertex;
    let mut invalid = texture_2d(GuestFormat::k_8_8_8_8, 16, 16, 0x30000, true);
    invalid.fetch_type = FetchConstantType::InvalidTexture;
    write_level(&mut mem, &invalid, 0, |x, y, _| rgba_pattern(x, y, 0).to_vec());
    let no_data = texture_2d(GuestFormat::k_8_8_8_8, 16, 16, 0, true);
    let fc = constants(&[(0, &unsupported), (1, &not_square), (2, &vertex), (3, &invalid), (4, &no_data)]);
    for (slot, dimension) in [
        (0, TextureBindingDimension::D2),
        (1, TextureBindingDimension::Cube),
        (2, TextureBindingDimension::D2),
        (3, TextureBindingDimension::D2),
        (4, TextureBindingDimension::D2),
    ] {
        for _ in 0..2 {
            let bound = cache.bind(&gpu.device, &mem, &fc, &binding(slot, dimension));
            assert!(bound.fallback, "slot {slot}");
        }
    }
    assert_eq!(cache.stats().loads, 0);
    assert_eq!(cache.stats().fallbacks, 3 + 10);
    // Opting into InvalidTexture fetch constants loads them.
    cache.set_config(CacheConfig { allow_invalid_fetch_constants: true, ..cache.config() });
    assert!(!cache.bind(&gpu.device, &mem, &fc, &binding(3, TextureBindingDimension::D2)).fallback);
    assert_no_errors(&gpu);
}

#[test]
fn sampler_mapping_and_filter_overrides() {
    let gpu = gpu(true);
    let mut mem = vec![0u8; MEMORY_SIZE];
    // 2x1 black/white texture, linear filtering in the fetch constant.
    let mut fetch = texture_2d(GuestFormat::k_8_8_8_8, 2, 1, 0x10000, false);
    fetch.clamp = [xenos_texture::ClampMode::ClampToEdge; 3];
    write_level(&mut mem, &fetch, 0, |x, _, _| vec![(x * 255) as u8; 4]);
    let mut aniso = fetch.clone();
    aniso.aniso_filter = AnisoFilter::Max16to1;
    aniso.mip_filter = TextureFilter::Linear;
    let fc = constants(&[(0, &fetch), (1, &aniso)]);
    let mut cache = TextureCache::for_device(&gpu.device);

    let sample = "textureSampleLevel(t, s, vec2<f32>(0.5, 0.5), 0.0)";
    let linear = cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D2));
    assert_close(run(&gpu, &linear, Read::D2, sample, (1, 1, 1))[0], [0.5; 4], 0.01, "linear");
    let mut point_binding = binding(0, TextureBindingDimension::D2);
    point_binding.mag_filter = parsed::TextureFilter::Point;
    point_binding.min_filter = parsed::TextureFilter::Point;
    let point = cache.bind(&gpu.device, &mem, &fc, &point_binding);
    assert_eq!(point.view, linear.view);
    assert_ne!(point.sampler, linear.sampler);
    let v = run(&gpu, &point, Read::D2, sample, (1, 1, 1))[0];
    assert!(v == [0.0; 4] || v == [1.0; 4], "point sampling gave {v:?}");
    // Left edge: repeat would blend in the right texel, clamp does not.
    let edge = run(&gpu, &linear, Read::D2, "textureSampleLevel(t, s, vec2<f32>(0.0, 0.5), 0.0)", (1, 1, 1))[0];
    assert_close(edge, [0.0; 4], 0.01, "clamp to edge");

    // Identical effective samplers are shared; anisotropy creates a valid sampler.
    assert_eq!(cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D2)).sampler, linear.sampler);
    let aniso_bound = cache.bind(&gpu.device, &mem, &fc, &binding(1, TextureBindingDimension::D2));
    assert_close(run(&gpu, &aniso_bound, Read::D2, sample, (1, 1, 1))[0], [0.5; 4], 0.01, "anisotropic");
    let mut no_aniso = binding(1, TextureBindingDimension::D2);
    no_aniso.aniso_filter = parsed::AnisoFilter::Disabled;
    no_aniso.mip_filter = parsed::TextureFilter::Point;
    // Without anisotropy and mip filtering it is the plain bilinear sampler again.
    assert_eq!(cache.bind(&gpu.device, &mem, &fc, &no_aniso).sampler, linear.sampler);
    assert_eq!(cache.stats().samplers, 3);
    assert_no_errors(&gpu);
}

#[test]
fn unfilterable_formats_get_non_filtering_samplers() {
    // Without float32-filterable, r32float must be sampled with a non-filtering sampler even
    // when the fetch constant asks for linear / anisotropic filtering.
    let gpu = gpu(false);
    let mut mem = vec![0u8; MEMORY_SIZE];
    let mut fetch = texture_2d(GuestFormat::k_32_FLOAT, 2, 1, 0x10000, false);
    fetch.aniso_filter = AnisoFilter::Max8to1;
    fetch.mip_filter = TextureFilter::Linear;
    fetch.clamp = [xenos_texture::ClampMode::ClampToEdge; 3];
    write_level(&mut mem, &fetch, 0, |x, _, _| (x as f32 * 10.0).to_le_bytes().to_vec());
    let fc = constants(&[(0, &fetch)]);
    let mut cache = TextureCache::for_device(&gpu.device);
    let bound = cache.bind(&gpu.device, &mem, &fc, &binding(0, TextureBindingDimension::D2));
    assert!(!bound.filterable && !bound.fallback);
    assert_eq!(bound.sample_type, TextureSampleType::UnfilterableFloat);
    assert_eq!(bound.sampler_binding_type(), SamplerBindingType::NonFiltering);
    let v = run(&gpu, &bound, Read::D2, "textureSampleLevel(t, s, vec2<f32>(0.75, 0.5), 0.0)", (1, 1, 1))[0];
    assert_eq!(v, [10.0, 0.0, 0.0, 1.0]);
    assert_no_errors(&gpu);
}

/// A GPU copy of the whole test memory in one buffer that is authoritative everywhere.
struct WholeMemory {
    buffers: Vec<Buffer>,
}

impl GpuSource for WholeMemory {
    fn covers(&mut self, _ranges: &[xenos_texture::GuestRange]) -> bool {
        true
    }
    fn sync(&mut self, _ranges: &[xenos_texture::GuestRange]) -> bool {
        true
    }
    fn gather(&self, encoder: &mut CommandEncoder, address: u32, len: u32, dst: &Buffer, dst_offset: u64) -> bool {
        encoder.copy_buffer_to_buffer(&self.buffers[0], u64::from(address), dst, dst_offset, u64::from(len));
        true
    }
    fn submit(&mut self, device: &Device, commands: CommandBuffer) {
        device.queue().submit([commands]);
    }
}

/// Pseudo-random bytes for `block`, keeping float formats finite (no all-ones exponents).
fn block_bytes(format: GuestFormat, bpb: u32, x: u32, y: u32, s: u32) -> Vec<u8> {
    let mut h = (x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B) ^ s.wrapping_mul(0xC2B2_AE35)) | 1;
    let mut bytes: Vec<u8> = (0..bpb)
        .map(|_| {
            h ^= h << 13;
            h ^= h >> 17;
            h ^= h << 5;
            h as u8
        })
        .collect();
    match format {
        GuestFormat::k_16_FLOAT | GuestFormat::k_16_16_FLOAT | GuestFormat::k_16_16_16_16_FLOAT => {
            for half in bytes.chunks_mut(2) {
                half[1] &= 0xBB;
            }
        }
        GuestFormat::k_32_FLOAT | GuestFormat::k_32_32_FLOAT | GuestFormat::k_32_32_32_32_FLOAT => {
            for word in bytes.chunks_mut(4) {
                word[3] &= 0xBF;
            }
        }
        _ => {}
    }
    bytes
}

#[test]
fn gpu_loads_match_cpu_loads() {
    for features in [true, false] {
        let gpu = gpu(features);
        let device = &gpu.device;
        let mut mem = vec![0u8; MEMORY_SIZE];
        let formats = [
            GuestFormat::k_8,
            GuestFormat::k_8_8,
            GuestFormat::k_5_6_5,
            GuestFormat::k_1_5_5_5,
            GuestFormat::k_4_4_4_4,
            GuestFormat::k_8_8_8_8,
            GuestFormat::k_2_10_10_10,
            GuestFormat::k_16,
            GuestFormat::k_16_16_16_16,
            GuestFormat::k_16_FLOAT,
            GuestFormat::k_16_16_FLOAT,
            GuestFormat::k_16_16_16_16_FLOAT,
            GuestFormat::k_32_FLOAT,
            GuestFormat::k_32_32_32_32_FLOAT,
            GuestFormat::k_16_16_EDRAM,
            GuestFormat::k_2_10_10_10_FLOAT_EDRAM,
            GuestFormat::k_24_8,
            GuestFormat::k_24_8_FLOAT,
        ];
        let mut fetches = Vec::new();
        for (i, &format) in formats.iter().enumerate() {
            let base = 0x40000 * (i as u32 + 1);
            for (variant, tiled, endian) in
                [(0, true, Endian::None), (1, false, Endian::K8in32), (2, true, Endian::K8in16)]
            {
                let mut fetch = texture_2d(format, 40, 24, base + variant * 0x10000, tiled);
                fetch.endianness = endian;
                let bpb = format.info().bytes_per_block();
                write_level(&mut mem, &fetch, 0, |x, y, s| block_bytes(format, bpb, x, y, s));
                fetches.push((format, fetch));
            }
        }
        // Mips (with a packed tail), a stacked array and a volume.
        let mut mipped = texture_2d(GuestFormat::k_8_8_8_8, 64, 64, 0x500000, true);
        mipped.mip_page = 0x520000 >> 12;
        mipped.mip_max_level = 6;
        mipped.packed_mips = true;
        for level in 0..=6 {
            write_level(&mut mem, &mipped, level, |x, y, s| {
                block_bytes(GuestFormat::k_8_8_8_8, 4, x + level * 100, y, s)
            });
        }
        fetches.push((GuestFormat::k_8_8_8_8, mipped));
        let mut stacked = texture_2d(GuestFormat::k_16_16_FLOAT, 16, 8, 0x600000, true);
        stacked.stacked = true;
        stacked.depth = 3;
        write_level(&mut mem, &stacked, 0, |x, y, s| block_bytes(GuestFormat::k_16_16_FLOAT, 4, x, y, s));
        let mut volume = texture_2d(GuestFormat::k_8_8_8_8, 16, 8, 0x680000, true);
        volume.dimension = DataDimension::D3;
        volume.depth = 4;
        write_level(&mut mem, &volume, 0, |x, y, s| block_bytes(GuestFormat::k_8_8_8_8, 4, x, y, s));

        let shared = device.create_buffer_init(
            "guest memory",
            BufferUsage::Storage | BufferUsage::CopyDst | BufferUsage::CopySrc,
            &mem,
        );
        let mut source = WholeMemory { buffers: vec![shared] };
        let cpu_only = CacheConfig { gpu_conversion: false, ..CacheConfig::default() };
        let mut cpu_cache = TextureCache::with_config(device, host_caps(device), cpu_only);
        let mut gpu_cache = TextureCache::for_device(device);
        let mut converting = TextureCache::for_device(device);
        let compare = |cpu: &BoundTexture,
                       gpu_bound: &BoundTexture,
                       read: Read,
                       size: (u32, u32, u32),
                       level: u32,
                       what: &str| {
            let a = load_texels(&gpu, cpu, read, size, level);
            let b = load_texels(&gpu, gpu_bound, read, size, level);
            for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                assert!(a.map(f32::to_bits) == b.map(f32::to_bits), "{what} texel {i}: cpu {a:?} gpu {b:?}");
            }
        };
        let mut expected_gpu_loads = 0;
        for (i, (format, fetch)) in fetches.iter().enumerate() {
            let fc = constants(&[(0, fetch)]);
            let b = binding(0, TextureBindingDimension::D2);
            let cpu = cpu_cache.bind(device, &mem, &fc, &b);
            let gpu_bound = gpu_cache.bind_with(device, &mem, &fc, &b, Some(&mut source));
            let converted = converting.bind(device, &mem, &fc, &b);
            assert!(!cpu.fallback && !gpu_bound.fallback && !converted.fallback, "{}", format.name());
            expected_gpu_loads += 1;
            assert_eq!(gpu_cache.stats().gpu_loads, expected_gpu_loads, "{} loaded on the GPU", format.name());
            assert_eq!(
                converting.stats().gpu_conversions,
                expected_gpu_loads,
                "{} converted on the GPU",
                format.name()
            );
            assert_eq!(cpu_cache.stats().gpu_conversions, 0);
            let levels = if fetch.mip_page != 0 { 7 } else { 1 };
            for level in 0..levels {
                let size = ((fetch.width >> level).max(1), (fetch.height >> level).max(1), 1);
                let what = format!("{} #{i} features={features} level {level}", format.name());
                compare(&cpu, &gpu_bound, Read::D2, size, level, &what);
                compare(&cpu, &converted, Read::D2, size, level, &format!("{what} (from memory)"));
            }
        }
        let fc = constants(&[(0, &stacked), (1, &volume)]);
        for (slot, read, size) in [(0, Read::Array, (16, 8, 3)), (1, Read::D3, (16, 8, 4))] {
            let b = binding(slot, TextureBindingDimension::D3OrStacked);
            let cpu = cpu_cache.bind(device, &mem, &fc, &b);
            let gpu_bound = gpu_cache.bind_with(device, &mem, &fc, &b, Some(&mut source));
            compare(&cpu, &gpu_bound, read, size, 0, &format!("slot {slot} features={features}"));
        }
        assert_eq!(gpu_cache.stats().gpu_loads, expected_gpu_loads + 2);
        assert_no_errors(&gpu);
    }
}
