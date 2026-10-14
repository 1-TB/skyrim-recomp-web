//! End-to-end checks on a real WebGPU device (a GPU, or lavapipe via VK_ICD_FILENAMES): render
//! known patterns into EDRAM targets, transfer and resolve them, and compare guest memory with
//! independently computed bytes (xenos-state's tiling and EDRAM addressing).
#![cfg(not(target_os = "emscripten"))]

use std::cell::RefCell;
use std::collections::HashMap;

use gpu_webgpu::*;
use xenos_edram::*;
use xenos_state::bitfields::*;
use xenos_state::edram::*;
use xenos_state::registers as r;
use xenos_state::resolve::{ResolveInfo, ResolveOptions, resolve_info};
use xenos_state::tiling::{tiled_2d, tiled_3d};
use xenos_state::xenos::{
    ColorRenderTargetFormat as C, CopyCommand, CopySampleSelect, DepthRenderTargetFormat as D, EdramMode, Endian,
    Endian128, FetchConstantType, MsaaSamples, SurfaceNumberFormat, TextureFormat as T, float32_to_20e4,
};

struct Gpu {
    _instance: Instance,
    device: Device,
}

fn gpu() -> Gpu {
    let instance = Instance::new().expect("instance");
    let adapter = instance
        .block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .expect("no WebGPU adapter (set VK_ICD_FILENAMES to lavapipe on GPU-less machines)");
    let features: Vec<FeatureName> =
        [FeatureName::Depth32FloatStencil8].into_iter().filter(|f| adapter.has_feature(*f)).collect();
    let device = instance
        .block_on(adapter.request_device(&DeviceDescriptor {
            required_features: &features,
            required_limits: Some(adapter.limits()),
            ..Default::default()
        }))
        .expect("device");
    Gpu { _instance: instance, device }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let errors = self.device.take_errors();
            assert!(errors.is_empty(), "WebGPU errors: {errors:#?}");
        }
    }
}

const MEM_SIZE: usize = 0x0200_0000;
const DEST: u32 = 0x0100_0000;

struct Mem(RefCell<Vec<u8>>);

impl Mem {
    fn new() -> Self {
        Mem(RefCell::new((0..MEM_SIZE).map(|i| (i * 7 + 3) as u8).collect()))
    }
    fn snapshot(&self) -> Vec<u8> {
        self.0.borrow().clone()
    }
}

impl GuestMemoryAccess for Mem {
    fn read(&self, addr: u32, out: &mut [u8]) {
        let m = self.0.borrow();
        out.copy_from_slice(&m[addr as usize..addr as usize + out.len()]);
    }
    fn write(&self, addr: u32, data: &[u8]) {
        self.0.borrow_mut()[addr as usize..addr as usize + data.len()].copy_from_slice(data);
    }
}

fn put(mem: &mut [u8], addr: u32, bytes: &[u8]) {
    mem[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
}

/// Compares whole memories, reporting the first differing bytes.
fn assert_mem(mem: &Mem, expected: &[u8]) {
    let got = mem.0.borrow();
    let diffs: Vec<_> = (0..MEM_SIZE).filter(|&i| got[i] != expected[i]).take(8).collect();
    let report: Vec<String> =
        diffs.iter().map(|&i| format!("0x{i:08X}: got {:02X} want {:02X}", got[i], expected[i])).collect();
    assert!(diffs.is_empty(), "memory differs:\n{}", report.join("\n"));
}

fn key(base: u32, pitch: u32, msaa: MsaaSamples, is_depth: bool, format: u32) -> RenderTargetKey {
    RenderTargetKey {
        base_tiles: base,
        pitch_tiles_at_32bpp: surface_pitch_tiles(pitch, msaa, false),
        msaa_samples: msaa,
        is_depth,
        format,
    }
}

fn empty_setup(pitch: u32, msaa: MsaaSamples) -> RenderTargetSetup {
    RenderTargetSetup {
        edram_mode: EdramMode::ColorDepth,
        surface_pitch: pitch,
        msaa_samples: msaa,
        pitch_tiles_at_32bpp: surface_pitch_tiles(pitch, msaa, false),
        depth: None,
        color: [None; 4],
    }
}

fn color_binding(slot: u32, base: u32, pitch: u32, msaa: MsaaSamples, format: C) -> RenderTargetBinding {
    RenderTargetBinding {
        slot: RenderTargetSlot::Color(slot),
        key: key(base, pitch, msaa, false, format.storage_format().raw()),
        guest_base_tiles: base,
        guest_color_format: Some(format),
        exp_bias: 0,
    }
}

fn color_setup(pitch: u32, msaa: MsaaSamples, base: u32, format: C) -> RenderTargetSetup {
    let mut s = empty_setup(pitch, msaa);
    s.color[0] = Some(color_binding(0, base, pitch, msaa, format));
    s
}

fn depth_setup(pitch: u32, msaa: MsaaSamples, base: u32, format: D) -> RenderTargetSetup {
    let mut s = empty_setup(pitch, msaa);
    s.depth = Some(RenderTargetBinding {
        slot: RenderTargetSlot::Depth,
        key: key(base, pitch, msaa, true, format.raw()),
        guest_base_tiles: base,
        guest_color_format: None,
        exp_bias: 0,
    });
    s
}

const VS: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}
"#;

/// Gets the targets for `setup` and draws the `fs` entry point of `wgsl` over `scissor` in each
/// `(scissor, stencil reference)` pass.
fn draw(
    gpu: &Gpu,
    cache: &mut EdramCache,
    setup: &RenderTargetSetup,
    rows: u32,
    wgsl: &str,
    passes: &[([u32; 4], u32)],
) -> DrawTargets {
    let device = &gpu.device;
    let mut enc = device.create_command_encoder("test draw");
    let t = cache.targets(device, &mut enc, setup, rows).expect("targets");
    let module = device.create_shader_module_wgsl("pattern", &format!("{VS}{wgsl}"));
    let color_targets: Vec<ColorTargetState> =
        t.colors[0].as_ref().map(|(_, f)| ColorTargetState::new(*f)).into_iter().collect();
    let replace = StencilFaceState { pass_op: StencilOperation::Replace, ..Default::default() };
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: "pattern",
        layout: None,
        vertex: ProgrammableStage::new(&module, "vs"),
        vertex_buffers: &[],
        primitive: PrimitiveState::default(),
        depth_stencil: t.depth.as_ref().map(|(_, f)| DepthStencilState {
            stencil_front: replace,
            stencil_back: replace,
            stencil_read_mask: 0xFF,
            stencil_write_mask: 0xFF,
            ..DepthStencilState::depth(*f, true, CompareFunction::Always)
        }),
        multisample: MultisampleState { count: t.sample_count, mask: t.sample_mask, alpha_to_coverage_enabled: false },
        fragment: Some(FragmentState { stage: ProgrammableStage::new(&module, "fs"), targets: &color_targets }),
    });
    {
        let colors: Vec<_> =
            t.colors[0].as_ref().map(|(v, _)| Some(RenderPassColorAttachment::load(v))).into_iter().collect();
        let depth = t.depth.as_ref().map(|(v, _)| RenderPassDepthStencilAttachment {
            view: v,
            depth_load_op: LoadOp::Load,
            depth_store_op: StoreOp::Store,
            depth_clear_value: 0.0,
            depth_read_only: false,
            stencil_load_op: LoadOp::Load,
            stencil_store_op: StoreOp::Store,
            stencil_clear_value: 0,
            stencil_read_only: false,
        });
        let mut pass = enc.begin_render_pass("pattern", &colors, depth.as_ref());
        pass.set_pipeline(&pipeline);
        for &([x0, y0, x1, y1], stencil) in passes {
            pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
            pass.set_stencil_reference(stencil);
            pass.draw(3, 1, 0, 0);
        }
        pass.end();
    }
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("test draw")]);
    t
}

/// Register-level description of a resolve, turned into a `ResolveInfo` by xenos-state.
#[derive(Clone)]
struct Resolve {
    pitch: u32,
    msaa: MsaaSamples,
    color_base: u32,
    color_format: C,
    depth_base: u32,
    depth_format: D,
    src_select: u32,
    sample_select: CopySampleSelect,
    rect: [f32; 4],
    command: CopyCommand,
    dest_format: T,
    dest_num: SurfaceNumberFormat,
    endian: Endian128,
    swap: bool,
    exp_bias: i32,
    dest_pitch: u32,
    dest_height: u32,
    array: bool,
    slice: u32,
    surface_slice: u32,
    clear_color: bool,
    clear_depth: bool,
    color_clear: [u32; 2],
    depth_clear: u32,
    copy_func: u32,
    copy_ref: u32,
    copy_mask: u32,
}

impl Default for Resolve {
    fn default() -> Self {
        Self {
            pitch: 256,
            msaa: MsaaSamples::X1,
            color_base: 0,
            color_format: C::K8888,
            depth_base: 0,
            depth_format: D::D24S8,
            src_select: 0,
            sample_select: CopySampleSelect::S0,
            rect: [0.0, 0.0, 256.0, 64.0],
            command: CopyCommand::Raw,
            dest_format: T::K8888,
            dest_num: SurfaceNumberFormat::UnsignedRepeatingFraction,
            endian: Endian128::None,
            swap: false,
            exp_bias: 0,
            dest_pitch: 256,
            dest_height: 64,
            array: false,
            slice: 0,
            surface_slice: 0,
            clear_color: false,
            clear_depth: false,
            color_clear: [0; 2],
            depth_clear: 0,
            copy_func: 0,
            copy_ref: 0,
            copy_mask: 0,
        }
    }
}

impl Resolve {
    fn info(&self) -> ResolveInfo {
        const VERTICES: u32 = 0x1000;
        let mut regs = vec![0u32; 0x5003];
        regs[r::RB_MODECONTROL as usize] = 6;
        regs[r::RB_SURFACE_INFO as usize] =
            RbSurfaceInfo { surface_pitch: self.pitch, msaa_samples: self.msaa, hiz_pitch: 0 }.encode();
        regs[r::RB_COLOR_INFO as usize] =
            RbColorInfo { color_base: self.color_base, color_format: self.color_format, ..Default::default() }.encode();
        regs[r::RB_DEPTH_INFO as usize] =
            RbDepthInfo { depth_base: self.depth_base, depth_format: self.depth_format, ..Default::default() }.encode();
        regs[r::RB_COPY_CONTROL as usize] = RbCopyControl {
            copy_src_select: self.src_select,
            copy_sample_select: self.sample_select,
            color_clear_enable: self.clear_color,
            depth_clear_enable: self.clear_depth,
            copy_command: self.command,
        }
        .encode();
        regs[r::RB_COPY_DEST_INFO as usize] = RbCopyDestInfo {
            copy_dest_endian: self.endian,
            copy_dest_array: self.array,
            copy_dest_slice: self.slice,
            copy_dest_format: self.dest_format,
            copy_dest_number: self.dest_num,
            copy_dest_exp_bias: self.exp_bias,
            copy_dest_swap: self.swap,
        }
        .encode();
        regs[r::RB_COPY_DEST_PITCH as usize] =
            RbCopyDestPitch { copy_dest_pitch: self.dest_pitch, copy_dest_height: self.dest_height }.encode();
        regs[r::RB_COPY_DEST_BASE as usize] = DEST;
        regs[r::RB_COPY_SURFACE_SLICE as usize] = self.surface_slice;
        regs[r::RB_COLOR_CLEAR_LO as usize] = self.color_clear[0];
        regs[r::RB_COLOR_CLEAR as usize] = self.color_clear[1];
        regs[r::RB_DEPTH_CLEAR as usize] = self.depth_clear;
        regs[r::RB_COPY_FUNC as usize] = self.copy_func;
        regs[r::RB_COPY_REF as usize] = self.copy_ref;
        regs[r::RB_COPY_MASK as usize] = self.copy_mask;
        regs[r::PA_SC_WINDOW_SCISSOR_BR as usize] = PaScWindowScissorBr { br_x: 8192, br_y: 8192 }.encode();
        regs[r::PA_SC_SCREEN_SCISSOR_BR as usize] = PaScScreenScissor { x: 8192, y: 8192 }.encode();
        let fetch = VertexFetchConstant {
            fetch_type: FetchConstantType::Vertex,
            address: VERTICES >> 2,
            endian: Endian::Swap8In32,
            size: 6,
        }
        .encode();
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize] = fetch[0];
        regs[r::SHADER_CONSTANT_FETCH_00_0 as usize + 1] = fetch[1];
        let [x0, y0, x1, y1] = self.rect;
        let vertices = [x0 - 0.5, y0 - 0.5, x1 - 0.5, y0 - 0.5, x1 - 0.5, y1 - 0.5];
        let mut mem = vec![0u8; 0x2000];
        for (i, v) in vertices.iter().enumerate() {
            let at = VERTICES as usize + i * 4;
            mem[at..at + 4].copy_from_slice(&v.to_bits().to_be_bytes());
        }
        resolve_info(&regs, &mem, &ResolveOptions::default()).expect("resolve info")
    }
}

fn run(gpu: &Gpu, cache: &mut EdramCache, info: &ResolveInfo, mem: &Mem) -> ResolveResult {
    let mut pending = cache.begin_resolve(&gpu.device, info).expect("begin resolve");
    cache.wait_resolve(&gpu.device, &mut pending, mem).expect("resolve")
}

/// 8888 test pattern: EDRAM dword of pixel (x, y).
fn pattern_8888(x: u32, y: u32) -> u32 {
    (x & 255) | ((y & 255) << 8) | (((x * 7 + y * 3) & 255) << 16) | (((x ^ y) & 255) << 24)
}

const FS_8888: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return vec4<f32>(f32(x & 255u), f32(y & 255u), f32((x * 7u + y * 3u) & 255u), f32((x ^ y) & 255u)) / 255.0;
}
"#;

fn addr_2d(x: u32, y: u32, pitch_aligned: u32, bpp_log2: u32) -> u32 {
    DEST + tiled_2d(x as i32, y as i32, pitch_aligned, bpp_log2) as u32
}

#[test]
fn fast_8888_subrect_endian_swap_and_poll() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    let t =
        draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    assert_eq!((t.width, t.sample_count), (320, 1));
    assert_eq!(t.colors[0].as_ref().unwrap().1, TextureFormat::RGBA8Unorm);

    let desc = Resolve { rect: [16.0, 8.0, 240.0, 56.0], endian: Endian128::Swap8In32, ..Default::default() };
    let info = desc.info();
    assert_eq!(info.copy_kind(1, 1).unwrap().0, xenos_state::resolve::ResolveCopyKind::Fast32bpp1x2xMsaa);
    let mut expected = mem.snapshot();
    let mut pending = cache.begin_resolve(&gpu.device, &info).expect("begin");
    let result = loop {
        if let Some(result) = cache.poll_resolve(&gpu.device, &mut pending, &mem) {
            break result.expect("resolve");
        }
        std::thread::yield_now();
    };
    assert!(pending.is_complete());
    assert_eq!(result.written, Some((info.copy_dest_extent_start, info.copy_dest_extent_length)));
    for y in 8..56 {
        for x in 16..240 {
            put(&mut expected, addr_2d(x, y, 256, 2), &pattern_8888(x, y).to_be_bytes());
        }
    }
    assert_mem(&mem, &expected);

    // Red/blue swap, no endian swap.
    let info = Resolve { swap: true, ..Default::default() }.info();
    run(&gpu, &mut cache, &info, &mem);
    for y in 0..64 {
        for x in 0..256 {
            let v = pattern_8888(x, y);
            let swapped = (v & 0xFF00_FF00) | ((v & 0xFF) << 16) | ((v >> 16) & 0xFF);
            put(&mut expected, addr_2d(x, y, 256, 2), &swapped.to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);
}

fn unorm8(v: u32) -> f32 {
    v as f32 / 255.0
}

fn pack_unorm(v: f32, bits: u32) -> u32 {
    (v.clamp(0.0, 1.0) * ((1u32 << bits) - 1) as f32 + 0.5) as u32
}

#[test]
fn full_conversion_565_with_exp_bias_and_8in16() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    let info = Resolve {
        rect: [0.0, 0.0, 64.0, 32.0],
        dest_format: T::K565,
        exp_bias: 1,
        endian: Endian128::Swap8In16,
        dest_pitch: 64,
        dest_height: 32,
        ..Default::default()
    }
    .info();
    let mut expected = mem.snapshot();
    run(&gpu, &mut cache, &info, &mem);
    for y in 0..32 {
        for x in 0..64 {
            let v = pattern_8888(x, y);
            let c = |shift: u32| unorm8((v >> shift) & 255) * 2.0;
            let packed = pack_unorm(c(0), 5) | (pack_unorm(c(8), 6) << 5) | (pack_unorm(c(16), 5) << 11);
            put(&mut expected, addr_2d(x, y, 64, 1), &(packed as u16).to_be_bytes());
        }
    }
    assert_mem(&mem, &expected);
}

fn f16(v: f32) -> u32 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    assert!(exp > 0 && exp < 31 && b & 0x1FFF == 0, "{v} is not an exact normal f16");
    ((b >> 31) << 15) | ((exp as u32) << 10) | ((b >> 13) & 0x3FF)
}

fn rgba16f_pattern(x: u32, y: u32) -> [f32; 4] {
    [(x + 1) as f32 / 4.0, -((y + 1) as f32), (x + y + 2) as f32 * 0.5, 1.0 + (x & 3) as f32]
}

const FS_RGBA16F: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return vec4<f32>(f32(x + 1u) / 4.0, -f32(y + 1u), f32(x + y + 2u) * 0.5, 1.0 + f32(x & 3u));
}
"#;

#[test]
fn rgba16f_64bpp_fast_and_full() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    let t = draw(
        &gpu,
        &mut cache,
        &color_setup(128, MsaaSamples::X1, 64, C::K16161616Float),
        48,
        FS_RGBA16F,
        &[([0, 0, 160, 48], 0)],
    );
    assert_eq!(t.colors[0].as_ref().unwrap().1, TextureFormat::RGBA16Float);
    let base = Resolve {
        pitch: 128,
        color_base: 64,
        color_format: C::K16161616Float,
        rect: [8.0, 8.0, 120.0, 40.0],
        dest_pitch: 128,
        dest_height: 48,
        ..Default::default()
    };
    let info = Resolve {
        dest_format: T::K16161616Float,
        dest_num: SurfaceNumberFormat::Float,
        endian: Endian128::Swap8In64,
        ..base.clone()
    }
    .info();
    assert_eq!(info.copy_kind(1, 1).unwrap().0, xenos_state::resolve::ResolveCopyKind::Fast64bpp1x2xMsaa);
    let mut expected = mem.snapshot();
    run(&gpu, &mut cache, &info, &mem);
    for y in 8..40 {
        for x in 8..120 {
            let c = rgba16f_pattern(x, y);
            let lo = f16(c[0]) | (f16(c[1]) << 16);
            let hi = f16(c[2]) | (f16(c[3]) << 16);
            let mut bytes = [lo.to_le_bytes(), hi.to_le_bytes()].concat();
            bytes.reverse();
            put(&mut expected, addr_2d(x, y, 128, 3), &bytes);
        }
    }
    assert_mem(&mem, &expected);

    // Full conversion to 8888 with exp bias -2 and a red/blue swap.
    let info = Resolve { dest_format: T::K8888, exp_bias: -2, swap: true, ..base }.info();
    run(&gpu, &mut cache, &info, &mem);
    for y in 8..40 {
        for x in 8..120 {
            let c = rgba16f_pattern(x, y).map(|v| v * 0.25);
            let packed = pack_unorm(c[2], 8)
                | (pack_unorm(c[1], 8) << 8)
                | (pack_unorm(c[0], 8) << 16)
                | (pack_unorm(c[3], 8) << 24);
            put(&mut expected, addr_2d(x, y, 128, 2), &packed.to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);
}

const FS_MSAA: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>, @builtin(sample_index) s: u32) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return vec4<f32>(f32((x + 40u * s) & 255u), f32((y * 4u + s) & 255u), f32(s * 60u), 255.0) / 255.0;
}
"#;

fn msaa_pattern(x: u32, y: u32, host_sample: u32) -> u32 {
    ((x + 40 * host_sample) & 255) | (((y * 4 + host_sample) & 255) << 8) | ((host_sample * 60) << 16) | (255 << 24)
}

fn check_close(mem: &Mem, addr: u32, want: u32, what: &str) {
    let mut got = [0u8; 4];
    mem.read(addr, &mut got);
    let got = u32::from_le_bytes(got);
    for c in 0..4 {
        let (g, w) = ((got >> (8 * c)) & 255, (want >> (8 * c)) & 255);
        assert!(g.abs_diff(w) <= 1, "{what}: got {got:08X} want {want:08X}");
    }
}

#[test]
fn msaa_4x_and_2x_sample_selection_and_averaging() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    for (msaa, base) in [(MsaaSamples::X4, 0), (MsaaSamples::X2, 300)] {
        let t = draw(&gpu, &mut cache, &color_setup(128, msaa, base, C::K8888), 32, FS_MSAA, &[([0, 0, 128, 32], 0)]);
        assert_eq!(t.sample_count, 4);
        assert_eq!(t.sample_mask, if msaa == MsaaSamples::X2 { 0b1001 } else { !0 });
        let host = |guest: u32| if msaa == MsaaSamples::X2 { guest * 3 } else { guest };
        let base_resolve = Resolve {
            pitch: 128,
            msaa,
            color_base: base,
            rect: [0.0, 0.0, 128.0, 32.0],
            dest_pitch: 128,
            dest_height: 32,
            ..Default::default()
        };
        // Single sample: the fast path, bit-exact.
        let single = if msaa == MsaaSamples::X4 { CopySampleSelect::S2 } else { CopySampleSelect::S1 };
        let info = Resolve { sample_select: single, ..base_resolve.clone() }.info();
        let mut expected = mem.snapshot();
        run(&gpu, &mut cache, &info, &mem);
        for y in 0..32 {
            for x in 0..128 {
                put(&mut expected, addr_2d(x, y, 128, 2), &msaa_pattern(x, y, host(single.raw())).to_le_bytes());
            }
        }
        assert_mem(&mem, &expected);
        // Averaging all samples.
        let info = Resolve { sample_select: CopySampleSelect::S0123, ..base_resolve }.info();
        run(&gpu, &mut cache, &info, &mem);
        let guest_samples = msaa.count().unwrap();
        for y in 0..32 {
            for x in 0..128 {
                let mut want = 0;
                for c in 0..4 {
                    let sum: f32 =
                        (0..guest_samples).map(|s| unorm8((msaa_pattern(x, y, host(s)) >> (8 * c)) & 255)).sum();
                    want |= pack_unorm(sum / guest_samples as f32, 8) << (8 * c);
                }
                check_close(&mem, addr_2d(x, y, 128, 2), want, &format!("{msaa:?} ({x}, {y})"));
            }
        }
    }
}

const FS_DEPTH: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return f32((x + y * 3u) & 1023u) / 1024.0;
}
"#;

fn depth_value(x: u32, y: u32) -> f32 {
    ((x + y * 3) & 1023) as f32 / 1024.0
}

fn unorm24(v: f32) -> u32 {
    (v as f64 * 16777215.0).round() as u32
}

#[test]
fn depth_resolves_d24s8_and_d24fs8() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    let float_exact = cache.depth_host_format(D::D24FS8) == TextureFormat::Depth32FloatStencil8;
    for (format, base) in [(D::D24S8, 0), (D::D24FS8, 512)] {
        let t = draw(
            &gpu,
            &mut cache,
            &depth_setup(128, MsaaSamples::X1, base, format),
            32,
            FS_DEPTH,
            &[([0, 0, 64, 32], 0x5A), ([64, 0, 160, 32], 0xC3)],
        );
        assert!(t.colors.iter().all(Option::is_none) && t.depth.is_some());
        assert_eq!(t.depth_info.unwrap().float24, format == D::D24FS8);
        let info = Resolve {
            pitch: 128,
            depth_base: base,
            depth_format: format,
            src_select: 4,
            rect: [0.0, 0.0, 128.0, 32.0],
            endian: Endian128::Swap8In32,
            dest_pitch: 128,
            dest_height: 32,
            ..Default::default()
        }
        .info();
        let mut expected = mem.snapshot();
        run(&gpu, &mut cache, &info, &mem);
        if format == D::D24FS8 && !float_exact {
            continue;
        }
        for y in 0..32 {
            for x in 0..128 {
                let d = depth_value(x, y);
                // Host float24 depth is half the guest value (see xenos-edram's pack_depth).
                let d24 = if format == D::D24FS8 { float32_to_20e4(d * 2.0, true) } else { unorm24(d) };
                let stencil = if x < 64 { 0x5A } else { 0xC3 };
                put(&mut expected, addr_2d(x, y, 128, 2), &((d24 << 8) | stencil).to_be_bytes());
            }
        }
        assert_mem(&mem, &expected);
    }
}

#[test]
fn clears_with_and_without_copies() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    // Clear only: DXT1 is not a resolvable destination, so the copy is dropped.
    let info = Resolve {
        rect: [32.0, 16.0, 96.0, 48.0],
        dest_format: T::Dxt1,
        clear_color: true,
        color_clear: [0, 0x1122_3344],
        ..Default::default()
    }
    .info();
    assert_eq!(info.copy_dest_extent_length, 0);
    let mut expected = mem.snapshot();
    let result = run(&gpu, &mut cache, &info, &mem);
    assert_eq!(result, ResolveResult { written: None, cleared_depth: false, cleared_color: true });
    assert_mem(&mem, &expected);
    run(&gpu, &mut cache, &Resolve::default().info(), &mem);
    for y in 0..64 {
        for x in 0..256 {
            let inside = (32..96).contains(&x) && (16..48).contains(&y);
            let v = if inside { 0x1122_3344 } else { pattern_8888(x, y) };
            put(&mut expected, addr_2d(x, y, 256, 2), &v.to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);

    // Depth copy and clear in one resolve: the copy sees the old contents.
    draw(&gpu, &mut cache, &depth_setup(128, MsaaSamples::X1, 700, D::D24S8), 32, FS_DEPTH, &[([0, 0, 160, 32], 0x77)]);
    let depth = Resolve {
        pitch: 128,
        depth_base: 700,
        src_select: 4,
        rect: [0.0, 0.0, 128.0, 32.0],
        dest_pitch: 128,
        dest_height: 32,
        ..Default::default()
    };
    let clear = 0x0012_3456u32;
    let info =
        Resolve { clear_depth: true, depth_clear: (clear << 8) | 0x9B, rect: [0.0, 8.0, 64.0, 24.0], ..depth.clone() }
            .info();
    let result = run(&gpu, &mut cache, &info, &mem);
    assert!(result.cleared_depth && !result.cleared_color);
    for y in 8..24 {
        for x in 0..64 {
            put(&mut expected, addr_2d(x, y, 128, 2), &((unorm24(depth_value(x, y)) << 8) | 0x77).to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);
    run(&gpu, &mut cache, &depth.info(), &mem);
    for y in 0..32 {
        for x in 0..128 {
            let inside = x < 64 && (8..24).contains(&y);
            let v = if inside { (clear << 8) | 0x9B } else { (unorm24(depth_value(x, y)) << 8) | 0x77 };
            put(&mut expected, addr_2d(x, y, 128, 2), &v.to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);

    // 64bpp clear values: RB_COLOR_CLEAR_LO is the low dword.
    let rgba = Resolve {
        pitch: 128,
        color_base: 1000,
        color_format: C::K16161616Float,
        rect: [0.0, 0.0, 64.0, 16.0],
        dest_format: T::K16161616Float,
        dest_num: SurfaceNumberFormat::Float,
        dest_pitch: 64,
        dest_height: 16,
        ..Default::default()
    };
    let info = Resolve { clear_color: true, color_clear: [0x3C00_4000, 0xBC00_3800], ..rgba.clone() }.info();
    run(&gpu, &mut cache, &info, &mem);
    run(&gpu, &mut cache, &rgba.info(), &mem);
    for y in 0..16 {
        for x in 0..64 {
            let bytes = [0x3C00_4000u32.to_le_bytes(), 0xBC00_3800u32.to_le_bytes()].concat();
            put(&mut expected, addr_2d(x, y, 64, 3), &bytes);
        }
    }
    assert_mem(&mem, &expected);
}

/// Like `assert_mem`, but 32-bit words may differ by one in their upper 24 bits (depth that went
/// through a depth24plus target whose loads aren't exact).
fn assert_mem_depth_tolerant(mem: &Mem, expected: &[u8]) {
    let got = mem.0.borrow();
    let word = |m: &[u8], i: usize| u32::from_le_bytes(m[i..i + 4].try_into().unwrap());
    for i in (0..MEM_SIZE).step_by(4) {
        let (g, w) = (word(&got, i), word(expected, i));
        assert!(g == w || ((g >> 8).abs_diff(w >> 8) <= 1 && g & 0xFF == w & 0xFF), "0x{i:08X}: {g:08X} vs {w:08X}");
    }
}

/// EDRAM byte offset -> dword of a 32bpp color surface filled with `f(x, y)`.
fn edram_map(base: u32, pitch: u32, width: u32, height: u32, f: impl Fn(u32, u32) -> u32) -> HashMap<u32, u32> {
    let pitch_tiles = surface_pitch_tiles(pitch, MsaaSamples::X1, false);
    let mut map = HashMap::new();
    for y in 0..height {
        for x in 0..width {
            let o = edram_offset_bytes([x, y], base, true, pitch_tiles, MsaaSamples::X1, false, false, 0, [1, 1]);
            map.insert(o, f(x, y));
        }
    }
    map
}

#[test]
fn ownership_transfers_between_pitches_formats_and_depth() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    ownership_transfers(&gpu, &mut cache);
    let mut cache = EdramCache::with_options(&gpu.device, EdramOptions { d24s8_float_host: true });
    ownership_transfers(&gpu, &mut cache);
}

fn ownership_transfers(gpu: &Gpu, cache: &mut EdramCache) {
    let exact_depth = cache.depth_host_format(D::D24S8) == TextureFormat::Depth32FloatStencil8;
    let check_depth = |mem: &Mem, expected: &[u8]| {
        if exact_depth { assert_mem(mem, expected) } else { assert_mem_depth_tolerant(mem, expected) }
    };
    let mem = Mem::new();
    let device = &gpu.device;
    // 8888 at pitch 256 (4 tiles per row), 64 rows = 16 tiles.
    draw(gpu, cache, &color_setup(256, MsaaSamples::X1, 10, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    let a = key(10, 256, MsaaSamples::X1, false, C::K8888.raw());
    assert_eq!(cache.ownership(), &[OwnedRange { start: 10, end: 26, owner: a }]);
    let original = edram_map(10, 256, 320, 64, pattern_8888);

    // Same tiles through pitch 160 (2 tiles per row): 128 rows.
    let mut enc = device.create_command_encoder("transfer");
    cache.targets(device, &mut enc, &color_setup(160, MsaaSamples::X1, 10, C::K8888), 128).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("transfer")]);
    let b = key(10, 160, MsaaSamples::X1, false, C::K8888.raw());
    assert_eq!(cache.ownership(), &[OwnedRange { start: 10, end: 26, owner: b }]);
    let info = Resolve {
        pitch: 160,
        color_base: 10,
        rect: [0.0, 0.0, 160.0, 128.0],
        dest_pitch: 160,
        dest_height: 128,
        ..Default::default()
    }
    .info();
    let mut expected = mem.snapshot();
    run(gpu, cache, &info, &mem);
    let pitch_b = surface_pitch_tiles(160, MsaaSamples::X1, false);
    for y in 0..128 {
        for x in 0..160 {
            let o = edram_offset_bytes([x, y], 10, true, pitch_b, MsaaSamples::X1, false, false, 0, [1, 1]);
            put(&mut expected, addr_2d(x, y, 160, 2), &original[&o].to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);

    // Color -> depth (tile halves swapped, stencil through the 8 bit passes) -> color again.
    let mut enc = device.create_command_encoder("to depth");
    cache.targets(device, &mut enc, &depth_setup(256, MsaaSamples::X1, 10, D::D24S8), 64).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("to depth")]);
    let info = Resolve { depth_base: 10, src_select: 4, rect: [0.0, 0.0, 256.0, 64.0], ..Default::default() }.info();
    run(gpu, cache, &info, &mem);
    let pitch_a = surface_pitch_tiles(256, MsaaSamples::X1, false);
    for y in 0..64 {
        for x in 0..256 {
            let o = edram_offset_bytes([x, y], 10, true, pitch_a, MsaaSamples::X1, true, false, 0, [1, 1]);
            put(&mut expected, addr_2d(x, y, 256, 2), &original[&o].to_le_bytes());
        }
    }
    check_depth(&mem, &expected);
    let mut enc = device.create_command_encoder("back to color");
    cache.targets(device, &mut enc, &color_setup(256, MsaaSamples::X1, 10, C::K8888), 64).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("back to color")]);
    run(gpu, cache, &Resolve { color_base: 10, ..Default::default() }.info(), &mem);
    for y in 0..64 {
        for x in 0..256 {
            put(&mut expected, addr_2d(x, y, 256, 2), &pattern_8888(x, y).to_le_bytes());
        }
    }
    check_depth(&mem, &expected);
    assert_eq!(cache.render_target_count(), 3);
    cache.trim();
    assert_eq!(cache.render_target_count(), 1);
}

/// Color tiles taken over by a float24 depth buffer and given back unchanged must keep every bit,
/// including patterns whose top 24 bits are 20e4 values of 1 or more (alpha 0xF0 and up), which
/// a host depth buffer limited to [0, 1] would clamp. Skyrim draws depth-only HUD passes over its
/// finished frame: clamping there turned the brightest areas pure red (only the stencil byte, red,
/// survived).
#[test]
fn color_through_float24_depth_is_lossless() {
    let gpu = gpu();
    let device = &gpu.device;
    let mut cache = EdramCache::new(device);
    let mem = Mem::new();
    draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 10, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    let mut enc = device.create_command_encoder("to depth");
    cache.targets(device, &mut enc, &depth_setup(256, MsaaSamples::X1, 10, D::D24FS8), 64).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("to depth")]);
    let mut enc = device.create_command_encoder("back to color");
    cache.targets(device, &mut enc, &color_setup(256, MsaaSamples::X1, 10, C::K8888), 64).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("back to color")]);
    let mut expected = mem.snapshot();
    run(&gpu, &mut cache, &Resolve { color_base: 10, ..Default::default() }.info(), &mem);
    for y in 0..64 {
        for x in 0..256 {
            put(&mut expected, addr_2d(x, y, 256, 2), &pattern_8888(x, y).to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);
}

const FS_HALF_SAFE: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return vec4<f32>(f32(x & 255u), f32(0x40u | (y & 0x33u)), f32((x * 7u + y * 3u) & 255u),
                     f32(0x40u | ((x ^ y) & 0x33u))) / 255.0;
}
"#;

fn pattern_half_safe(x: u32, y: u32) -> u32 {
    (x & 255) | ((0x40 | (y & 0x33)) << 8) | (((x * 7 + y * 3) & 255) << 16) | ((0x40 | ((x ^ y) & 0x33)) << 24)
}

#[test]
fn transfer_32bpp_to_64bpp_preserves_bits() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    let device = &gpu.device;
    draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 32, FS_HALF_SAFE, &[([0, 0, 320, 32], 0)]);
    let original = edram_map(0, 256, 320, 32, pattern_half_safe);
    let mut enc = device.create_command_encoder("to 64bpp");
    cache.targets(device, &mut enc, &color_setup(256, MsaaSamples::X1, 0, C::K16161616Float), 16).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("to 64bpp")]);
    let info = Resolve {
        color_format: C::K16161616Float,
        rect: [0.0, 0.0, 160.0, 16.0],
        dest_format: T::K16161616Float,
        dest_num: SurfaceNumberFormat::Float,
        dest_pitch: 160,
        dest_height: 16,
        ..Default::default()
    }
    .info();
    let mut expected = mem.snapshot();
    run(&gpu, &mut cache, &info, &mem);
    let pitch_tiles = surface_pitch_tiles(256, MsaaSamples::X1, true);
    for y in 0..16 {
        for x in 0..160 {
            let o = edram_offset_bytes([x, y], 0, true, pitch_tiles, MsaaSamples::X1, false, true, 0, [1, 1]);
            let bytes = [original[&o].to_le_bytes(), original[&(o + 4)].to_le_bytes()].concat();
            put(&mut expected, addr_2d(x, y, 160, 3), &bytes);
        }
    }
    assert_mem(&mem, &expected);
}

const FS_GAMMA: &str = r#"
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return vec4<f32>(f32(x & 255u) / 255.0, f32((y * 9u) & 255u) / 255.0, f32((x + y) & 255u) / 255.0, 0.5);
}
"#;

#[test]
fn gamma_targets_store_linear_and_resolve_decoded() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    let device = &gpu.device;
    let t = draw(
        &gpu,
        &mut cache,
        &color_setup(256, MsaaSamples::X1, 0, C::K8888Gamma),
        16,
        FS_GAMMA,
        &[([0, 0, 320, 16], 0)],
    );
    assert_eq!(t.colors[0].as_ref().unwrap().1, TextureFormat::RGBA16Float);
    let linear =
        |x: u32, y: u32| [(x & 255) as f32 / 255.0, ((y * 9) & 255) as f32 / 255.0, ((x + y) & 255) as f32 / 255.0];
    let base = Resolve { color_format: C::K8888Gamma, rect: [0.0, 0.0, 256.0, 16.0], ..Default::default() };
    let mut expected = mem.snapshot();
    run(&gpu, &mut cache, &base.info(), &mem);
    for y in 0..16 {
        for x in 0..256 {
            let l = linear(x, y);
            let mut v = 128 << 24;
            for (c, l) in l.iter().enumerate() {
                v |= pack_unorm(pwl_gamma_to_linear(linear_to_pwl_gamma(*l)), 8) << (8 * c);
            }
            put(&mut expected, addr_2d(x, y, 256, 2), &v.to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);
    // Viewed as k_8_8_8_8 the same tiles hold the PWL codes.
    let mut enc = device.create_command_encoder("gamma as 8888");
    cache.targets(device, &mut enc, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 16).expect("targets");
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("gamma as 8888")]);
    run(&gpu, &mut cache, &Resolve { rect: [0.0, 0.0, 256.0, 16.0], ..Default::default() }.info(), &mem);
    for y in 0..16 {
        for x in 0..256 {
            let l = linear(x, y);
            let mut v = 128 << 24;
            for (c, l) in l.iter().enumerate() {
                v |= (linear_to_pwl_gamma(*l) as u32) << (8 * c);
            }
            put(&mut expected, addr_2d(x, y, 256, 2), &v.to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);
}

#[test]
fn volume_slice_and_dxt3a_1111_destinations() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    let info = Resolve {
        rect: [0.0, 0.0, 64.0, 64.0],
        array: true,
        slice: 2,
        dest_pitch: 64,
        dest_height: 0,
        surface_slice: 64 * 64,
        ..Default::default()
    }
    .info();
    let mut expected = mem.snapshot();
    run(&gpu, &mut cache, &info, &mem);
    for y in 0..64 {
        for x in 0..64 {
            let a = DEST + tiled_3d(x as i32, y as i32, 2, 64, 64, 2) as u32;
            put(&mut expected, a, &pattern_8888(x, y).to_le_bytes());
        }
    }
    assert_mem(&mem, &expected);

    // Per component: R ref < value, G ==, B >, A <=; reference 0x80 with a full mask.
    let (func, reference) = (0x6421u32, 0x8080_8080u32);
    let info = Resolve {
        rect: [0.0, 0.0, 64.0, 32.0],
        command: CopyCommand::ConvertTo1111,
        dest_format: T::Dxt3aAs1111,
        dest_pitch: 16,
        dest_height: 8,
        copy_func: func,
        copy_ref: reference,
        copy_mask: 0xFFFF_FFFF,
        endian: Endian128::Swap8In32,
        ..Default::default()
    }
    .info();
    assert!(info.convert_to_1111_supported);
    run(&gpu, &mut cache, &info, &mem);
    let bits = |x: u32, y: u32| {
        let p = pattern_8888(x, y);
        (0..4).fold(0u32, |acc, c| {
            let (v, r) = ((p >> (8 * c)) & 255, (reference >> (8 * c)) & 255);
            let relation = if v < r {
                4
            } else if v == r {
                2
            } else {
                1
            };
            acc | ((((func >> (4 * c)) & relation) != 0) as u32) << c
        })
    };
    for by in 0..8 {
        for bx in 0..16 {
            let row = |r: u32| (0..4).fold(0u32, |acc, i| acc | (bits(bx * 4 + i, by * 4 + r) << (4 * i)));
            let block = [(row(0) | (row(1) << 16)).to_be_bytes(), (row(2) | (row(3) << 16)).to_be_bytes()].concat();
            put(&mut expected, addr_2d(bx, by, 32, 3), &block);
        }
    }
    assert_mem(&mem, &expected);
}

#[test]
fn aliased_bindings_and_errors() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let device = &gpu.device;
    let mut enc = device.create_command_encoder("errors");

    let mut setup = color_setup(256, MsaaSamples::X1, 0, C::K8888);
    setup.color[1] = Some(color_binding(1, 0, 256, MsaaSamples::X1, C::K2101010));
    setup.color[2] = Some(color_binding(2, 64, 256, MsaaSamples::X1, C::K3232Float));
    setup.depth = depth_setup(256, MsaaSamples::X1, 0, D::D24S8).depth;
    let t = cache.targets(device, &mut enc, &setup, 64).expect("targets");
    assert!(t.colors[0].is_some() && t.colors[1].is_none() && t.depth.is_none());
    assert_eq!(t.colors[2].as_ref().unwrap().1, TextureFormat::RG32Float);
    assert!(!t.color_info[2].unwrap().blendable);
    // The first target stops at the second one's base.
    assert_eq!(cache.ownership()[0], OwnedRange { start: 0, end: 16, owner: setup.color[0].unwrap().key });

    let mut bad = color_setup(256, MsaaSamples::X1, 0, C::K8888);
    bad.msaa_samples = MsaaSamples::Unknown(3);
    assert_eq!(cache.targets(device, &mut enc, &bad, 64).unwrap_err(), EdramError::InvalidMsaa(3));
    let bad = color_setup(256, MsaaSamples::X1, 0, C::Unknown(9));
    assert_eq!(cache.targets(device, &mut enc, &bad, 64).unwrap_err(), EdramError::UnsupportedFormat(9));
    let wide = color_setup(8192, MsaaSamples::X4, 0, C::K8888);
    assert!(matches!(cache.targets(device, &mut enc, &wide, 64), Err(EdramError::PitchTooLarge { .. })));
    let none = cache.targets(device, &mut enc, &empty_setup(0, MsaaSamples::X1), 64).expect("empty");
    assert!(none.is_empty());
    cache.flush_uploads(device);
    device.queue().submit([enc.finish("errors")]);

    let mut info = Resolve::default().info();
    info.dest_format = T::Dxt1;
    assert_eq!(cache.begin_resolve(device, &info).err(), Some(EdramError::UnsupportedCopy(T::Dxt1)));
    let empty = Resolve { rect: [300.0, 0.0, 400.0, 8.0], ..Default::default() }.info();
    assert!(empty.rect.is_empty());
    let mut pending = cache.begin_resolve(device, &empty).expect("empty resolve");
    let mem = Mem::new();
    assert_eq!(
        cache.poll_resolve(device, &mut pending, &mem),
        Some(Ok(ResolveResult { written: None, cleared_depth: false, cleared_color: false }))
    );
}

#[test]
fn msaa_transfers_to_1x_and_back() {
    let gpu = gpu();
    let mut cache = EdramCache::new(&gpu.device);
    let mem = Mem::new();
    let device = &gpu.device;
    for (msaa, base) in [(MsaaSamples::X4, 0), (MsaaSamples::X2, 400)] {
        let host = |guest: u32| if msaa == MsaaSamples::X2 { guest * 3 } else { guest };
        draw(&gpu, &mut cache, &color_setup(128, msaa, base, C::K8888), 32, FS_MSAA, &[([0, 0, 160, 32], 0)]);
        let pitch_tiles = surface_pitch_tiles(128, msaa, false);
        let mut by_offset = HashMap::new();
        for y in 0..32 {
            for x in 0..160 {
                for s in 0..msaa.count().unwrap() {
                    let o = edram_offset_bytes([x, y], base, true, pitch_tiles, msaa, false, false, s, [1, 1]);
                    by_offset.insert(o, msaa_pattern(x, y, host(s)));
                }
            }
        }
        // The same tiles as a 1x surface with the same pitch in tiles.
        let pitch_1x = pitch_tiles * 80;
        let rows_1x = 32 << msaa.samples_y_log2();
        let mut enc = device.create_command_encoder("to 1x");
        cache.targets(device, &mut enc, &color_setup(pitch_1x, MsaaSamples::X1, base, C::K8888), rows_1x).unwrap();
        cache.flush_uploads(device);
        device.queue().submit([enc.finish("to 1x")]);
        let info = Resolve {
            pitch: pitch_1x,
            color_base: base,
            rect: [0.0, 0.0, pitch_1x as f32, rows_1x as f32],
            dest_pitch: pitch_1x,
            dest_height: rows_1x,
            ..Default::default()
        }
        .info();
        let mut expected = mem.snapshot();
        run(&gpu, &mut cache, &info, &mem);
        for y in 0..rows_1x {
            for x in 0..pitch_1x {
                let o = edram_offset_bytes([x, y], base, true, pitch_tiles, MsaaSamples::X1, false, false, 0, [1, 1]);
                put(&mut expected, addr_2d(x, y, pitch_1x, 2), &by_offset[&o].to_le_bytes());
            }
        }
        assert_mem(&mem, &expected);
        // Back to the multisampled surface: every guest sample is restored.
        let mut enc = device.create_command_encoder("back to msaa");
        cache.targets(device, &mut enc, &color_setup(128, msaa, base, C::K8888), 32).unwrap();
        cache.flush_uploads(device);
        device.queue().submit([enc.finish("back to msaa")]);
        let last = msaa.count().unwrap() - 1;
        let info = Resolve {
            pitch: 128,
            msaa,
            color_base: base,
            sample_select: CopySampleSelect::from_raw(last),
            rect: [0.0, 0.0, 128.0, 32.0],
            dest_pitch: 128,
            dest_height: 32,
            ..Default::default()
        }
        .info();
        run(&gpu, &mut cache, &info, &mem);
        for y in 0..32 {
            for x in 0..128 {
                put(&mut expected, addr_2d(x, y, 128, 2), &msaa_pattern(x, y, host(last)).to_le_bytes());
            }
        }
        assert_mem(&mem, &expected);
    }
}

/// Reads `len` bytes at `offset` of a storage buffer.
fn read_buffer(gpu: &Gpu, buffer: &Buffer, offset: u64, len: u64) -> Vec<u8> {
    let device = &gpu.device;
    let staging = device.create_buffer(&BufferDescriptor {
        label: "readback",
        size: len,
        usage: BufferUsage::MapRead | BufferUsage::CopyDst,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder("readback");
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, len);
    device.queue().submit([enc.finish("readback")]);
    device.block_on(staging.map_async(MapMode::Read, 0, None)).expect("map");
    let mut out = vec![0u8; len as usize];
    staging.read_mapped(0, &mut out).expect("read");
    out
}

#[test]
fn shared_memory_store_matches_readback() {
    let gpu = gpu();
    let device = &gpu.device;
    let mut cache = EdramCache::new(device);
    // Guest memory mirrored in 64 KiB blocks, block n in slot n except that the two halves are
    // swapped, so the table is really consulted.
    const BLOCK: u32 = 64 << 10;
    let blocks = MEM_SIZE as u32 / BLOCK;
    let slot_of = |b: u32| (b + blocks / 2) % blocks;
    let initial = Mem::new().snapshot();
    let mut mirrored = vec![0u8; MEM_SIZE];
    for b in 0..blocks {
        let (from, to) = ((b * BLOCK) as usize, (slot_of(b) * BLOCK) as usize);
        mirrored[to..to + BLOCK as usize].copy_from_slice(&initial[from..from + BLOCK as usize]);
    }
    // The slots in four buffers.
    const CHUNK: usize = MEM_SIZE / 4;
    let chunks: Vec<Buffer> = mirrored
        .chunks(CHUNK)
        .map(|c| {
            device.create_buffer_init("shared", BufferUsage::Storage | BufferUsage::CopySrc | BufferUsage::CopyDst, c)
        })
        .collect();
    let table: Vec<u8> = (0..blocks).flat_map(|b| (slot_of(b) + 1).to_le_bytes()).collect();
    let table = device.create_buffer_init("shared table", BufferUsage::Storage, &table);
    let target = SharedMemoryTarget { chunks: &chunks, table: &table, block_bytes: BLOCK, chunk_bytes: CHUNK as u32 };
    // Reads guest bytes [start, start + len) back from the mirror.
    let read_guest = |start: u32, len: u32| -> Vec<u8> {
        let all: Vec<u8> = chunks.iter().flat_map(|c| read_buffer(&gpu, c, 0, CHUNK as u64)).collect();
        (start..start + len).map(|a| all[(slot_of(a / BLOCK) * BLOCK + a % BLOCK) as usize]).collect()
    };

    let check = |cache: &mut EdramCache, info: ResolveInfo, what: &str| {
        for (c, data) in chunks.iter().zip(mirrored.chunks(CHUNK)) {
            device.queue().write_buffer(c, 0, data);
        }
        let mem = Mem::new();
        run(&gpu, cache, &info, &mem);
        let pending = cache.begin_resolve_to(device, &info, Some(&target)).expect("begin");
        assert!(pending.is_complete(), "{what}: stored resolves complete at once");
        let mut pending = pending;
        let result = cache.poll_resolve(device, &mut pending, &mem).expect("done").expect("ok");
        let (start, len) = result.written.expect("written range");
        assert_eq!((start, len), (info.copy_dest_extent_start, info.copy_dest_extent_length), "{what}");
        let got = read_guest(start, len);
        let want = &mem.snapshot()[start as usize..(start + len) as usize];
        let diffs: Vec<String> = (0..len as usize)
            .filter(|&i| got[i] != want[i])
            .take(8)
            .map(|i| format!("0x{:08X}: got {:02X} want {:02X}", start as usize + i, got[i], want[i]))
            .collect();
        assert!(diffs.is_empty(), "{what}: stored bytes differ from the readback:\n{}", diffs.join("\n"));
    };

    draw(&gpu, &mut cache, &color_setup(256, MsaaSamples::X1, 0, C::K8888), 64, FS_8888, &[([0, 0, 320, 64], 0)]);
    let base8888 = Resolve::default();
    check(
        &mut cache,
        Resolve { rect: [16.0, 8.0, 240.0, 56.0], endian: Endian128::Swap8In32, ..base8888.clone() }.info(),
        "8888 8in32 subrect",
    );
    check(&mut cache, Resolve { swap: true, endian: Endian128::Swap16In32, ..base8888.clone() }.info(), "8888 16in32");
    check(
        &mut cache,
        Resolve {
            rect: [0.0, 0.0, 64.0, 32.0],
            command: CopyCommand::Convert,
            dest_format: T::K565,
            endian: Endian128::Swap8In16,
            dest_pitch: 64,
            dest_height: 32,
            ..base8888.clone()
        }
        .info(),
        "565 8in16",
    );
    check(
        &mut cache,
        Resolve {
            rect: [8.0, 0.0, 72.0, 32.0],
            command: CopyCommand::Convert,
            dest_format: T::K8,
            dest_pitch: 96,
            dest_height: 32,
            ..base8888.clone()
        }
        .info(),
        "8bpp",
    );
    check(
        &mut cache,
        Resolve {
            rect: [0.0, 0.0, 64.0, 64.0],
            array: true,
            slice: 2,
            dest_pitch: 64,
            dest_height: 0,
            surface_slice: 64 * 64,
            ..base8888.clone()
        }
        .info(),
        "3D slice",
    );
    check(
        &mut cache,
        Resolve {
            rect: [0.0, 0.0, 64.0, 32.0],
            command: CopyCommand::ConvertTo1111,
            dest_format: T::Dxt3aAs1111,
            dest_pitch: 16,
            dest_height: 8,
            copy_func: 0x6421,
            copy_ref: 0x8080_8080,
            copy_mask: 0xFFFF_FFFF,
            endian: Endian128::Swap8In32,
            ..base8888
        }
        .info(),
        "1111 blocks",
    );

    draw(
        &gpu,
        &mut cache,
        &color_setup(128, MsaaSamples::X1, 64, C::K16161616Float),
        48,
        FS_RGBA16F,
        &[([0, 0, 160, 48], 0)],
    );
    let base16f = Resolve {
        pitch: 128,
        color_base: 64,
        color_format: C::K16161616Float,
        rect: [8.0, 8.0, 120.0, 40.0],
        dest_pitch: 128,
        dest_height: 48,
        dest_format: T::K16161616Float,
        dest_num: SurfaceNumberFormat::Float,
        ..Default::default()
    };
    for endian in [Endian128::Swap8In64, Endian128::Swap8In32, Endian128::Swap8In128] {
        check(&mut cache, Resolve { endian, ..base16f.clone() }.info(), &format!("64bpp {endian:?}"));
    }
    for endian in [Endian128::Swap8In128, Endian128::Swap8In64, Endian128::Swap8In16] {
        check(
            &mut cache,
            Resolve { command: CopyCommand::Convert, dest_format: T::K32323232Float, endian, ..base16f.clone() }.info(),
            &format!("128bpp {endian:?}"),
        );
    }
}
