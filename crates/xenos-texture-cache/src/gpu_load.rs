//! Texture loading from a GPU copy of guest memory, for textures whose data only exists there
//! (EDRAM resolves stored straight into GPU memory leave guest memory itself unchanged).
//!
//! A compute shader does what `xenos_texture::load_texture` does on the CPU — address the guest
//! blocks (tiled 2D/3D or linear, packed mips), apply the fetch constant's endian swap, convert
//! to the host format — writing rows ready for `copy_buffer_to_texture`. Only the conversions
//! resolves can produce are supported; [`plan`] returns `None` for the rest.

use gpu_webgpu::*;
use xenos_texture::host::{Conversion, HostMapping};
use xenos_texture::layout::packed_mip_offset;
use xenos_texture::{DataDimension, Endian, GuestMemory, GuestRange, TextureFetchConstant, TextureGuestInfo};

/// Guest physical memory mirrored on the GPU, as split buffers.
pub trait GpuSource {
    /// Whether the GPU copy is the up-to-date version of all of `ranges` (and guest memory may
    /// be stale there), so the texture must be loaded from it.
    fn covers(&mut self, ranges: &[GuestRange]) -> bool;
    /// The buffers (`buffers[i]` holds physical bytes `i * buffer_bytes ..`) and their size.
    fn buffers(&self) -> (&[Buffer], u64);
}

/// Row alignment of `copy_buffer_to_texture`.
const ROW_ALIGNMENT: u32 = 256;

/// One level's dispatch.
pub(crate) struct LevelPlan {
    level: u32,
    params: [u32; 32],
    blocks: (u32, u32, u32),
    out_offset: u64,
    bytes_per_row: u32,
    rows_per_image: u32,
    copy: (u32, u32, u32),
    /// Physical address range of the level's guest data.
    source: (u32, u32),
}

pub(crate) struct Plan {
    levels: Vec<LevelPlan>,
    /// Size of the converted rows (padded for the copy).
    out_bytes: u64,
    /// Host bytes of all levels without padding.
    host_bytes: u64,
}

impl Plan {
    /// Whether the dispatches write every level below `count` (all of each level's texels).
    pub(crate) fn covers_levels(&self, count: u32) -> bool {
        (0..count).all(|m| self.levels.iter().any(|l| l.level == m))
    }

    /// Host bytes of all levels, for the cache's memory budget.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.host_bytes
    }
}

fn conversion_code(mapping: &HostMapping) -> Option<(u32, [u32; 2], u32)> {
    Some(match mapping.conversion {
        Conversion::Copy => (0, [0, 0], 0),
        Conversion::PackedToRgba8 { fields, count, signed: false } => {
            let pack = |f: &[(u8, u8)]| {
                f.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, (s, b))| acc | (u32::from(*s) | u32::from(*b) << 8) << (16 * i))
            };
            (1, [pack(&fields[0..2]), pack(&fields[2..4])], u32::from(count))
        }
        Conversion::Norm16ToFloat16 { components, signed: false } => (2, [0, 0], u32::from(components)),
        Conversion::Edram16ToFloat16 { components } => (3, [0, 0], u32::from(components)),
        Conversion::Float7e3ToRgba16Float => (4, [0, 0], 4),
        Conversion::Depth24UnormToFloat32 => (5, [0, 0], 1),
        Conversion::Depth20e4ToFloat32 => (6, [0, 0], 1),
        _ => return None,
    })
}

/// The dispatches loading `fetch` (whose host mapping is `mapping`) for levels up to
/// `level_count`, or `None` if the GPU loader can't produce its host data.
pub(crate) fn plan(
    fetch: &TextureFetchConstant,
    info: &TextureGuestInfo,
    mapping: &HostMapping,
    level_count: u32,
) -> Option<Plan> {
    let (code, fields, count) = conversion_code(mapping)?;
    let format_info = info.format.info();
    let (bw, bh, bpb) = (format_info.block_width, format_info.block_height, format_info.bytes_per_block());
    let host_block = mapping.host_format.block_size();
    let host_unit = mapping.host_format.bytes_per_block();
    // Per-texel conversions of multi-texel blocks (BC decompression etc.) stay on the CPU.
    if !(host_block > 1 || (bw == 1 && bh == 1)) || bpb > 16 || host_unit > 16 {
        return None;
    }
    if code != 0 && bpb != 2 && bpb != 4 && bpb != 8 {
        return None;
    }
    let sub = &info.subresources;
    let is_3d = sub.dimension == DataDimension::D3;
    let tiled = fetch.tiled && sub.dimension != DataDimension::D1;
    let first_level = if sub.has_base() { 0 } else { 1 };
    let mut levels = Vec::new();
    let mut out_bytes = 0u64;
    let mut host_bytes = 0u64;
    for level in first_level..=info.layout.max_level.min(level_count.saturating_sub(1)) {
        let Some((address, layout, _)) = info.level_storage(level) else { continue };
        let width = (sub.width >> level).max(1);
        let height = (sub.height >> level).max(1);
        let depth = (sub.depth() >> level).max(1);
        let slices = if is_3d { depth } else { sub.array_size() };
        let (ox, oy, oz) = if info.packed_mips && level >= info.layout.packed_level {
            packed_mip_offset(sub.width, sub.height, sub.depth(), info.format, level).unwrap_or((0, 0, 0))
        } else {
            (0, 0, 0)
        };
        let (blocks_x, blocks_y) = (width.div_ceil(bw), height.div_ceil(bh));
        let bytes_per_row = (blocks_x * host_unit).next_multiple_of(ROW_ALIGNMENT);
        let image_bytes = bytes_per_row * blocks_y;
        let copy = if host_block > 1 { (blocks_x * bw, blocks_y * bh, slices) } else { (width, height, slices) };
        let mut params = [0u32; 32];
        params[..24].copy_from_slice(&[
            address,
            tiled as u32,
            is_3d as u32,
            layout.row_pitch_bytes / bpb,
            layout.z_slice_stride_block_rows,
            layout.row_pitch_bytes,
            bpb,
            bpb.ilog2(),
            blocks_x,
            blocks_y,
            slices,
            layout.array_slice_stride_bytes,
            ox,
            oy,
            oz,
            match fetch.endianness {
                Endian::None => 0,
                Endian::K8in16 => 1,
                Endian::K8in32 => 2,
                Endian::K16in32 => 3,
            },
            code,
            bytes_per_row,
            image_bytes,
            host_unit,
            layout.level_data_extent_bytes,
            0, // buffer base, filled in by `encode`
            fields[0],
            fields[1],
        ]);
        params[24] = count;
        levels.push(LevelPlan {
            level,
            params,
            blocks: (blocks_x, blocks_y, slices),
            out_offset: out_bytes,
            bytes_per_row,
            rows_per_image: blocks_y,
            copy,
            source: (address, layout.level_data_extent_bytes),
        });
        out_bytes += u64::from(image_bytes) * u64::from(slices);
        host_bytes += u64::from(blocks_x * host_unit) * u64::from(blocks_y) * u64::from(slices);
    }
    (!levels.is_empty()).then_some(Plan { levels, out_bytes, host_bytes })
}

/// Parameter slots (one per level, `PARAMS_STRIDE` bytes apart) in the loader's uniform buffer.
const PARAM_SLOTS: usize = 32;
/// Uniform binding offsets must be multiples of 256.
const PARAMS_STRIDE: u64 = 256;

/// The loader's buffers live as long as it does and are rewritten for every load (with
/// `Queue::write_buffer`, ordered after earlier submissions): creating buffers and bind groups is
/// expensive in browsers (a fraction of a millisecond each), and a frame loads dozens of
/// textures.
#[derive(Default)]
pub(crate) struct GpuLoader {
    /// Plain-store and atomic variants.
    pipelines: [Option<(BindGroupLayout, ComputePipeline)>; 2],
    /// Per-level parameters, `PARAM_SLOTS` slots.
    params: Option<Buffer>,
    /// Converted texels, before the copy into the texture (grows as needed).
    out: Option<Buffer>,
    /// Raw guest bytes for loads from guest memory (grows as needed).
    upload: Option<Buffer>,
    /// Bind groups by (atomic, parameter slot, source buffer, output buffer).
    groups: rustc_hash::FxHashMap<(bool, usize, usize, usize), BindGroup>,
}

/// A buffer of at least `size` bytes in `slot`: the existing one if big enough, else a new one
/// (rounded up to a power of two; the old one is destroyed, which waits for work submitted with
/// it). Returns whether it was replaced.
fn ensure(slot: &mut Option<Buffer>, device: &Device, label: &str, usage: BufferUsage, size: u64) -> bool {
    if slot.as_ref().is_some_and(|b| b.size() >= size) {
        return false;
    }
    if let Some(old) = slot.take() {
        old.destroy();
    }
    *slot = Some(device.create_buffer(&BufferDescriptor {
        label,
        size: size.max(1 << 16).next_power_of_two(),
        usage,
        mapped_at_creation: false,
    }));
    true
}

impl GpuLoader {
    fn pipeline(&mut self, device: &Device, atomic: bool) -> &(BindGroupLayout, ComputePipeline) {
        self.pipelines[atomic as usize].get_or_insert_with(|| {
            let entry = |binding, ty| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStage::Compute,
                ty: BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: 0 },
            };
            let bgl = device.create_bind_group_layout(
                "texture gpu load",
                &[
                    entry(0, BufferBindingType::Uniform),
                    entry(1, BufferBindingType::ReadOnlyStorage),
                    entry(2, BufferBindingType::Storage),
                ],
            );
            let layout = device.create_pipeline_layout("texture gpu load", &[&bgl]);
            let module = device.create_shader_module_wgsl("texture gpu load", &shader_variant(SHADER, atomic));
            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: "texture gpu load",
                layout: Some(&layout),
                stage: ProgrammableStage::new(&module, "main"),
            });
            (bgl, pipeline)
        })
    }

    /// Fills `texture` from the GPU copy of guest memory and submits the work. Returns `false`
    /// (nothing done) if a level's data spans two of the source's buffers.
    pub(crate) fn encode(&mut self, device: &Device, source: &dyn GpuSource, plan: &Plan, texture: &Texture) -> bool {
        let (buffers, buffer_bytes) = source.buffers();
        let mut sources = Vec::new();
        for level in &plan.levels {
            let (start, len) = (u64::from(level.source.0), u64::from(level.source.1.max(1)));
            let index = start / buffer_bytes;
            if (start + len - 1) / buffer_bytes != index || index as usize >= buffers.len() {
                return false;
            }
            sources.push((buffers[index as usize].clone(), (index * buffer_bytes) as u32));
        }
        self.dispatch(device, &sources, plan, texture);
        true
    }

    /// Fills `texture` from guest memory: each level's raw bytes are copied into the upload
    /// buffer and converted there (instead of texel by texel on the CPU).
    pub(crate) fn encode_from_memory(
        &mut self,
        device: &Device,
        memory: &dyn GuestMemory,
        plan: &Plan,
        texture: &Texture,
    ) -> u64 {
        // Levels at 256-byte aligned offsets of the upload buffer.
        let mut offsets = Vec::with_capacity(plan.levels.len());
        let mut total = 0u64;
        for level in &plan.levels {
            offsets.push(total);
            total = (total + u64::from(level.source.1).next_multiple_of(4).max(4)).next_multiple_of(256);
        }
        if ensure(&mut self.upload, device, "texture upload", BufferUsage::Storage | BufferUsage::CopyDst, total) {
            self.groups.clear();
        }
        let upload = self.upload.clone().expect("ensured above");
        let mut uploaded = 0u64;
        let mut bytes = Vec::new();
        let sources: Vec<(Buffer, u32)> = plan
            .levels
            .iter()
            .zip(&offsets)
            .map(|(level, &offset)| {
                let (address, len) = level.source;
                bytes.clear();
                bytes.resize((len as usize).next_multiple_of(4).max(4), 0);
                memory.read(address, &mut bytes);
                device.queue().write_buffer(&upload, offset, &bytes);
                uploaded += bytes.len() as u64;
                // The shader indexes the source from the guest address `base` on: the level's
                // address sits `offset` bytes into the buffer.
                (upload.clone(), address.wrapping_sub(offset as u32))
            })
            .collect();
        self.dispatch(device, &sources, plan, texture);
        uploaded
    }

    /// Converts each level from its source buffer (holding guest bytes from address `base` on)
    /// into `texture`, and submits.
    fn dispatch(&mut self, device: &Device, sources: &[(Buffer, u32)], plan: &Plan, texture: &Texture) {
        assert!(plan.levels.len() <= PARAM_SLOTS, "more levels than parameter slots");
        let mut replaced = ensure(
            &mut self.out,
            device,
            "texture gpu load",
            BufferUsage::Storage | BufferUsage::CopySrc,
            plan.out_bytes.max(4),
        );
        replaced |= ensure(
            &mut self.params,
            device,
            "texture gpu load params",
            BufferUsage::Uniform | BufferUsage::CopyDst,
            PARAM_SLOTS as u64 * PARAMS_STRIDE,
        );
        if replaced {
            self.groups.clear();
        }
        let (out, params) = (self.out.clone().expect("ensured"), self.params.clone().expect("ensured"));
        // Host texels narrower than a dword share dwords between invocations.
        let atomic = plan.levels.first().is_some_and(|l| l.params[19] < 4);
        let (bgl, pipeline) = self.pipeline(device, atomic).clone();
        let mut encoder = device.create_command_encoder("texture gpu load");
        let mut groups = Vec::with_capacity(plan.levels.len());
        for (slot, (level, (buffer, base))) in plan.levels.iter().zip(sources).enumerate() {
            let mut words = level.params;
            words[21] = *base;
            words[25] = (level.out_offset / 4) as u32;
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            device.queue().write_buffer(&params, slot as u64 * PARAMS_STRIDE, &bytes);
            let key = (atomic, slot, buffer.raw() as usize, out.raw() as usize);
            let group = self.groups.entry(key).or_insert_with(|| {
                device.create_bind_group(
                    "texture gpu load",
                    &bgl,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer {
                                buffer: &params,
                                offset: slot as u64 * PARAMS_STRIDE,
                                size: Some(bytes.len() as u64),
                            },
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer { buffer, offset: 0, size: None },
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer { buffer: &out, offset: 0, size: None },
                        },
                    ],
                )
            });
            groups.push(group.clone());
        }
        {
            let mut pass = encoder.begin_compute_pass("texture gpu load");
            pass.set_pipeline(&pipeline);
            for (level, group) in plan.levels.iter().zip(&groups) {
                let (x, y, z) = level.blocks;
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(x.div_ceil(8), y.div_ceil(8), z);
            }
            pass.end();
        }
        for level in &plan.levels {
            encoder.copy_buffer_to_texture(
                &TexelCopyBufferInfo {
                    buffer: &out,
                    offset: level.out_offset,
                    bytes_per_row: level.bytes_per_row,
                    rows_per_image: Some(level.rows_per_image),
                },
                &TexelCopyTextureInfo {
                    texture,
                    mip_level: level.level,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                extent(level.copy.0, level.copy.1, level.copy.2),
            );
        }
        device.queue().submit([encoder.finish("texture gpu load")]);
    }
}

/// Parameters (32 u32): 0 level address, 1 tiled, 2 is_3d, 3 pitch in blocks, 4 z stride in block
/// rows, 5 linear row pitch bytes, 6 bytes per block, 7 log2, 8/9 blocks x/y, 10 slices, 11 array
/// slice stride, 12-14 packed mip offset, 15 endian, 16 conversion, 17 output bytes per row,
/// 18 output bytes per slice, 19 host bytes per block, 20 level data extent, 21 guest address of
/// `src[0]`, 22/23 packed fields (shift | bits << 8, two per word), 24 component count,
/// 25 output offset in words.
const SHADER: &str = r#"
struct Params { v: array<vec4<u32>, 8> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: DST_TYPE;

fn store_word(i: u32, v: u32) {
    STORE_WORD
}

fn prm(i: u32) -> u32 { return params.v[i >> 2u][i & 3u]; }

fn tiled_combine(oib: i32, bank: u32, pipe: u32, y_lsb: u32) -> i32 {
    return i32((y_lsb << 4u) | (pipe << 6u) | (bank << 11u)) | (oib & 15) | (((oib >> 4u) & 1) << 5u)
        | (((oib >> 5u) & 7) << 8u) | ((oib >> 8u) << 12u);
}

fn tiled_2d(x: i32, y: i32, pitch: u32, bpb_log2: u32) -> i32 {
    let outer = ((y >> 5u) * i32(pitch >> 5u) + (x >> 5u)) << 6u;
    let inner = (((y >> 1u) & 7) << 3u) | (x & 7);
    let oib = (outer | inner) << bpb_log2;
    let bank = u32((y >> 4u) & 1);
    let pipe = u32(((x >> 3u) & 3) ^ (((y >> 3u) & 1) << 1u));
    return tiled_combine(oib, bank, pipe, u32(y & 1));
}

fn tiled_3d(x: i32, y: i32, z: i32, pitch: u32, height: u32, bpb_log2: u32) -> i32 {
    let outer = ((((z >> 2u) * i32(height >> 4u) + (y >> 4u)) * i32(pitch >> 5u)) + (x >> 5u)) << 7u;
    let inner = ((z & 3) << 5u) | (((y >> 1u) & 3) << 3u) | (x & 7);
    let oib = (outer | inner) << bpb_log2;
    let bank = u32(((y >> 3u) ^ (z >> 2u)) & 1);
    let pipe = u32((x >> 3u) & 3) ^ (bank << 1u);
    return tiled_combine(oib, bank, pipe, u32(y & 1));
}

fn swap32(w: u32, endian: u32) -> u32 {
    switch endian {
        case 1u: { return ((w & 0x00FF00FFu) << 8u) | ((w >> 8u) & 0x00FF00FFu); }
        case 2u: { return (w << 24u) | ((w & 0xFF00u) << 8u) | ((w >> 8u) & 0xFF00u) | (w >> 24u); }
        case 3u: { return (w << 16u) | (w >> 16u); }
        default: { return w; }
    }
}

// Exact port of xenos_texture::convert::f32_to_f16 (round to nearest even).
fn f16(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    let sign = (x >> 16u) & 0x8000u;
    let exp = i32((x >> 23u) & 0xFFu);
    let man = x & 0x7FFFFFu;
    if (exp == 0xFF) {
        return sign | 0x7C00u | select(0u, 0x200u, man != 0u);
    }
    let e = exp - 127 + 15;
    if (e >= 31) { return sign | 0x7C00u; }
    if (e <= 0) {
        if (e < -10) { return sign; }
        let m = man | 0x800000u;
        let shift = u32(14 - e);
        var h = m >> shift;
        let rem = m & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) == 1u)) { h += 1u; }
        return sign | h;
    }
    var h = (u32(e) << 10u) | (man >> 13u);
    let rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) { h += 1u; }
    return sign | h;
}

fn f7e3(bits: u32) -> f32 {
    let v = bits & 0x3FFu;
    if (v == 0u) { return 0.0; }
    var mantissa = v & 0x7Fu;
    var exponent = v >> 7u;
    if (exponent == 0u) {
        let lz = 7u - (31u - countLeadingZeros(mantissa));
        exponent = 1u - lz;
        mantissa = (mantissa << lz) & 0x7Fu;
    }
    return bitcast<f32>(((exponent + 124u) << 23u) | (mantissa << 16u));
}

// Xenos 20e4 float depth (24 bits) to f32.
fn f20e4(bits: u32) -> f32 {
    let v = bits & 0xFFFFFFu;
    if (v == 0u) { return 0.0; }
    var mantissa = v & 0xFFFFFu;
    var exponent = v >> 20u;
    if (exponent == 0u) {
        let lz = 20u - (31u - countLeadingZeros(mantissa));
        exponent = 1u - lz;
        mantissa = (mantissa << lz) & 0xFFFFFu;
    }
    return bitcast<f32>(((exponent + 112u) << 23u) | (mantissa << 3u));
}

fn put_bits(byte_addr: u32, value: u32, mask: u32) {
    let i = byte_addr >> 2u;
    let shift = (byte_addr & 3u) * 8u;
    PUT_BITS
}

fn field(i: u32) -> vec2<u32> {
    let packed = prm(22u + (i >> 1u)) >> (16u * (i & 1u));
    return vec2<u32>(packed & 0xFFu, (packed >> 8u) & 0xFFu);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    let slice = gid.z;
    if (bx >= prm(8u) || by >= prm(9u) || slice >= prm(10u)) { return; }
    let bpb = prm(6u);
    let is_3d = prm(2u) != 0u;
    let layer = select(slice, 0u, is_3d);
    let z = select(0u, slice, is_3d);
    let x = bx + prm(12u);
    let y = by + prm(13u);
    let zz = z + prm(14u);
    var offset: i32;
    if (prm(1u) != 0u) {
        if (is_3d) {
            offset = tiled_3d(i32(x), i32(y), i32(zz), prm(3u), prm(4u), prm(7u));
        } else {
            offset = tiled_2d(i32(x), i32(y), prm(3u), prm(7u));
        }
    } else {
        offset = i32((zz * prm(4u) + y) * prm(5u) + x * bpb);
    }
    offset += i32(layer * prm(11u));
    let endian = prm(15u);
    var w = array<u32, 4>(0u, 0u, 0u, 0u);
    if (offset >= 0 && u32(offset) + bpb <= prm(20u)) {
        let a = prm(0u) + u32(offset) - prm(21u);
        if (bpb >= 4u) {
            for (var k = 0u; k < bpb / 4u; k++) {
                w[k] = swap32(src[(a >> 2u) + k], endian);
            }
        } else {
            let word = swap32(src[a >> 2u], endian);
            w[0] = (word >> ((a & 3u) * 8u)) & select(0xFFu, 0xFFFFu, bpb == 2u);
        }
    }
    let unit = prm(19u);
    let out = prm(25u) * 4u + slice * prm(18u) + by * prm(17u) + bx * unit;
    let conversion = prm(16u);
    var o = array<u32, 4>(w[0], w[1], w[2], w[3]);
    switch conversion {
        case 1u: {
            // Packed fixed point to RGBA8 (missing components 255).
            var rgba = 0u;
            for (var c = 0u; c < 4u; c++) {
                var byte = 255u;
                if (c < prm(24u)) {
                    let f = field(c);
                    let max = (1u << f.y) - 1u;
                    let raw = (w[0] >> f.x) & max;
                    byte = (raw * 255u + max / 2u) / max;
                }
                rgba |= byte << (8u * c);
            }
            o[0] = rgba;
        }
        case 2u, 3u: {
            // 16-bit unorm (2) or EDRAM -32..32 fixed point (3) components to halves.
            var halves = array<u32, 4>(0u, 0u, 0u, 0u);
            for (var c = 0u; c < prm(24u); c++) {
                let raw = (w[c >> 1u] >> (16u * (c & 1u))) & 0xFFFFu;
                var f: f32;
                if (conversion == 2u) {
                    f = f32(raw) / 65535.0;
                } else {
                    let s = i32(raw << 16u) >> 16u;
                    f = max(f32(s) * (32.0 / 32767.0), -32.0);
                }
                halves[c] = f16(f);
            }
            o[0] = halves[0] | (halves[1] << 16u);
            o[1] = halves[2] | (halves[3] << 16u);
        }
        case 4u: {
            let v = w[0];
            o[0] = f16(f7e3(v)) | (f16(f7e3(v >> 10u)) << 16u);
            o[1] = f16(f7e3(v >> 20u)) | (f16(f32(v >> 30u) / 3.0) << 16u);
        }
        case 5u: {
            // 24-bit unorm depth in the high bits.
            let n = w[0] >> 8u;
            o[0] = bitcast<u32>(f32(n + (n >> 23u)) * (1.0 / 16777216.0));
        }
        case 6u: {
            o[0] = bitcast<u32>(f20e4(w[0] >> 8u));
        }
        default: {}
    }
    if (unit >= 4u) {
        for (var k = 0u; k < unit / 4u; k++) {
            store_word((out >> 2u) + k, o[k]);
        }
    } else {
        put_bits(out, o[0], select(0xFFu, 0xFFFFu, unit == 2u));
    }
}
"#;

/// The shader with plain stores (`atomic == false`: every write is a whole aligned dword) or
/// with atomics (8/16-bit elements share dwords with neighbours written by other invocations).
fn shader_variant(source: &str, atomic: bool) -> String {
    let (ty, store, put) = if atomic {
        (
            "array<atomic<u32>>",
            "atomicStore(&dst[i], v);",
            "atomicAnd(&dst[i], ~(mask << shift));\n    atomicOr(&dst[i], (value & mask) << shift);",
        )
    } else {
        ("array<u32>", "dst[i] = v;", "dst[i] = (dst[i] & ~(mask << shift)) | ((value & mask) << shift);")
    };
    source.replace("DST_TYPE", ty).replace("STORE_WORD", store).replace("PUT_BITS", put)
}
