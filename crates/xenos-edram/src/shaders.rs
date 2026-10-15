//! WGSL for ownership transfers, EDRAM dumps (host target -> EDRAM-layout buffer), resolve copies
//! and clears. Formats are runtime codes (see `format::DEPTH_FORMAT_CODE_BASE`); only the source
//! texture binding types and the destination sample count are baked into module variants.

/// How a host render target is read in a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SrcKind {
    Color { multisampled: bool },
    Depth { multisampled: bool },
}

impl SrcKind {
    pub fn multisampled(self) -> bool {
        match self {
            SrcKind::Color { multisampled } | SrcKind::Depth { multisampled } => multisampled,
        }
    }
}

/// EDRAM addressing, format packing (host value -> EDRAM bits) and unpacking, for host render
/// targets at resolution scale `scale` (see [`crate::EdramOptions::resolution_scale`]).
fn common(scale: u32) -> String {
    COMMON.replace("RES_SCALE_VALUE", &format!("{scale}u"))
}

/// With resolution scaling (like Xenia's), a tile of a host render target has `scale` x `scale`
/// host samples per guest sample; everything else (tiles, bases, pitches, ownership) stays in
/// guest tiles. The EDRAM-layout buffer holds the scaled samples, tile by tile.
const COMMON: &str = r#"
const RES_SCALE: u32 = RES_SCALE_VALUE;
const TILE_DWORDS: u32 = 1280u * RES_SCALE * RES_SCALE;
const EDRAM_TILES: u32 = 2048u;
const EDRAM_DWORDS: u32 = 2048u * TILE_DWORDS;

// Sample coordinates (canonical EDRAM sample grid, scaled) of a host pixel's sample.
fn guest_sample_coords(px: vec2<u32>, sample: u32, msaa: u32) -> vec2<u32> {
    if (msaa == 2u) {
        return vec2<u32>(((px.x & ~1u) << 1u) | (px.x & 1u) | ((sample & 1u) << 1u),
                         ((px.y & ~1u) << 1u) | (px.y & 1u) | (((sample >> 1u) & 1u) << 1u));
    }
    if (msaa == 1u) {
        return vec2<u32>((px.x & ~2u) | ((sample & 1u) << 1u), ((px.y & ~1u) << 1u) | (px.y & 1u) | (px.x & 2u));
    }
    return px;
}

// Wrapped EDRAM dword index of the first dword of a pixel's sample.
fn edram_dword(px: vec2<u32>, sample: u32, base_tiles: u32, pitch_tiles: u32, msaa: u32, is_depth: bool,
               is_64: bool) -> u32 {
    let s = guest_sample_coords(px, sample, msaa);
    let tw = select(80u, 40u, is_64) * RES_SCALE;
    let th = 16u * RES_SCALE;
    let tx = s.x / tw;
    let ty = s.y / th;
    var sx = s.x - tx * tw;
    let sy = s.y - ty * th;
    if (is_depth) {
        let half = tw >> 1u;
        sx = select(sx + half, sx - half, sx >= half);
    }
    let tile = (base_tiles + ty * pitch_tiles + tx) & (EDRAM_TILES - 1u);
    return tile * TILE_DWORDS + ((sy * tw + sx) << u32(is_64));
}

struct EdramLoc {
    px: vec2<u32>,
    sample: u32,
    half: u32,
}

// Inverse of edram_dword for a surface at base_tiles: the pixel, sample and dword half holding the
// dword (the first addressing period from the base).
fn edram_locate(dword: u32, base_tiles: u32, pitch_tiles: u32, msaa: u32, is_depth: bool, is_64: bool) -> EdramLoc {
    let tile = dword / TILE_DWORDS;
    let within = dword - tile * TILE_DWORDS;
    let rel = (tile - base_tiles) & (EDRAM_TILES - 1u);
    let pitch = max(pitch_tiles, 1u);
    let ty = rel / pitch;
    let tx = rel - ty * pitch;
    let tw = select(80u, 40u, is_64) * RES_SCALE;
    let idx = within >> u32(is_64);
    var loc: EdramLoc;
    loc.half = within & u32(is_64);
    var sx = idx % tw;
    let sy = idx / tw;
    if (is_depth) {
        let half = tw >> 1u;
        sx = select(sx + half, sx - half, sx >= half);
    }
    let g = vec2<u32>(tx * tw + sx, ty * 16u * RES_SCALE + sy);
    if (msaa == 2u) {
        loc.px = vec2<u32>(((g.x >> 2u) << 1u) | (g.x & 1u), ((g.y >> 2u) << 1u) | (g.y & 1u));
        loc.sample = ((g.x >> 1u) & 1u) | (((g.y >> 1u) & 1u) << 1u);
    } else if (msaa == 1u) {
        loc.px = vec2<u32>((g.x & ~2u) | (((g.y >> 1u) & 1u) << 1u), ((g.y >> 2u) << 1u) | (g.y & 1u));
        loc.sample = (g.x >> 1u) & 1u;
    } else {
        loc.px = g;
        loc.sample = 0u;
    }
    return loc;
}

// 2x MSAA is hosted on 4x targets: guest samples 0 and 1 are host samples 0 and 3.
fn host_sample_of_guest(sample: u32, msaa: u32) -> u32 {
    return select(sample, sample * 3u, msaa == 1u);
}

fn guest_sample_of_host(sample: u32, msaa: u32) -> u32 {
    return select(sample, sample >> 1u, msaa == 1u);
}

fn f32_to_7e3(v: f32) -> u32 {
    let c = select(clamp(v, 0.0, 31.875), 0.0, v != v);
    let f = bitcast<u32>(c);
    var biased: u32;
    if (f < 0x3E800000u) {
        biased = ((f & 0x7FFFFFu) | 0x800000u) >> min(125u - (f >> 23u), 24u);
    } else {
        biased = f - (124u << 23u);
    }
    return ((biased + 0x7FFFu + ((biased >> 16u) & 1u)) >> 16u) & 0x3FFu;
}

fn f7e3_to_f32(bits: u32) -> f32 {
    let f10 = bits & 0x3FFu;
    if (f10 == 0u) {
        return 0.0;
    }
    var m = f10 & 0x7Fu;
    var e = f10 >> 7u;
    if (e == 0u) {
        let lz = 7u - firstLeadingBit(m);
        e = 1u - lz;
        m = (m << lz) & 0x7Fu;
    }
    return bitcast<f32>(((e + 124u) << 23u) | (m << 16u));
}

fn f32_to_20e4(v: f32) -> u32 {
    if (!(v > 0.0)) {
        return 0u;
    }
    var bits = bitcast<u32>(v);
    if (bits >= 0x3FFFFFF8u) {
        return 0xFFFFFFu;
    }
    if (bits < 0x38800000u) {
        bits = (0x800000u | (bits & 0x7FFFFFu)) >> min(113u - (bits >> 23u), 24u);
    } else {
        bits = bits + 0xC8000000u;
    }
    bits = bits + 3u + ((bits >> 3u) & 1u);
    return (bits >> 3u) & 0xFFFFFFu;
}

fn f20e4_to_f32(bits: u32) -> f32 {
    let f24 = bits & 0xFFFFFFu;
    if (f24 == 0u) {
        return 0.0;
    }
    var m = f24 & 0xFFFFFu;
    var e = f24 >> 20u;
    if (e == 0u) {
        let lz = 20u - firstLeadingBit(m);
        e = 1u - lz;
        m = (m << lz) & 0xFFFFFu;
    }
    return bitcast<f32>(((e + 112u) << 23u) | (m << 3u));
}

// unorm24 <-> f32 as x / (2^24 - 1). Writes round up, so both rounding and truncating unorm24
// depth conversions store x; v * 2^24 - v is within 0.5 of the integer for exact reads.
fn unorm24_to_f32(bits: u32) -> f32 {
    let x = bits & 0xFFFFFFu;
    let xf = f32(x);
    var t = xf * (1.0 / 16777216.0) + xf * (1.0 / 281474976710656.0);
    if (x != 0u && t * 16777216.0 - xf < t) {
        t = bitcast<f32>(bitcast<u32>(t) + 1u);
    }
    return t;
}

fn f32_to_unorm24(v: f32) -> u32 {
    let c = select(clamp(v, 0.0, 1.0), 0.0, v != v);
    return u32(round(c * 16777216.0 - c));
}

fn linear_to_pwl8(v: f32) -> u32 {
    let k = u32(round(clamp(v, 0.0, 1.0) * 1023.0));
    if (k < 64u) {
        return k;
    }
    if (k < 128u) {
        return (k >> 1u) + 32u;
    }
    if (k < 512u) {
        return (k >> 2u) + 64u;
    }
    return (k >> 3u) + 128u;
}

fn pwl8_to_linear(b: u32) -> f32 {
    var k: u32;
    if (b < 64u) {
        k = b;
    } else if (b < 96u) {
        k = 2u * b - 64u;
    } else if (b < 192u) {
        let l = 4u * b - 256u;
        k = l + (l >> 8u);
    } else {
        let l = 8u * b - 1024u;
        k = l + (l >> 7u);
    }
    return f32(k) / 1023.0;
}

fn unorm_bits(v: f32, max_value: f32) -> u32 {
    return u32(round(clamp(v, 0.0, 1.0) * max_value));
}

fn edram16_from_f32(v: f32) -> u32 {
    return u32(i32(round(clamp(v, -32.0, 32.0) * (32767.0 / 32.0)))) & 0xFFFFu;
}

fn edram16_to_f32(x: u32) -> f32 {
    return max(-32.0, f32(i32(x << 16u) >> 16u) * (32.0 / 32767.0));
}

// Host render target value -> EDRAM bits (dword 0, dword 1 for 64bpp).
fn pack_color(c: vec4<f32>, fmt: u32) -> vec2<u32> {
    switch (fmt) {
        case 0u: {
            return vec2<u32>(pack4x8unorm(c), 0u);
        }
        case 1u: {
            return vec2<u32>(linear_to_pwl8(c.r) | (linear_to_pwl8(c.g) << 8u) | (linear_to_pwl8(c.b) << 16u)
                             | (unorm_bits(c.a, 255.0) << 24u), 0u);
        }
        case 2u, 10u: {
            return vec2<u32>(unorm_bits(c.r, 1023.0) | (unorm_bits(c.g, 1023.0) << 10u)
                             | (unorm_bits(c.b, 1023.0) << 20u) | (unorm_bits(c.a, 3.0) << 30u), 0u);
        }
        case 3u, 12u: {
            return vec2<u32>(f32_to_7e3(c.r) | (f32_to_7e3(c.g) << 10u) | (f32_to_7e3(c.b) << 20u)
                             | (unorm_bits(c.a, 3.0) << 30u), 0u);
        }
        case 4u: {
            return vec2<u32>(edram16_from_f32(c.r) | (edram16_from_f32(c.g) << 16u), 0u);
        }
        case 5u: {
            return vec2<u32>(edram16_from_f32(c.r) | (edram16_from_f32(c.g) << 16u),
                             edram16_from_f32(c.b) | (edram16_from_f32(c.a) << 16u));
        }
        case 6u: {
            return vec2<u32>(pack2x16float(c.rg), 0u);
        }
        case 7u: {
            return vec2<u32>(pack2x16float(c.rg), pack2x16float(c.ba));
        }
        case 14u: {
            return vec2<u32>(bitcast<u32>(c.r), 0u);
        }
        case 15u: {
            return vec2<u32>(bitcast<u32>(c.r), bitcast<u32>(c.g));
        }
        default: {
            return vec2<u32>(0u, 0u);
        }
    }
}

// EDRAM bits -> host render target value (also the resolve's unpacked color, with gamma decoded).
fn unpack_color(lo: u32, hi: u32, fmt: u32) -> vec4<f32> {
    switch (fmt) {
        case 0u: {
            return unpack4x8unorm(lo);
        }
        case 1u: {
            return vec4<f32>(pwl8_to_linear(lo & 0xFFu), pwl8_to_linear((lo >> 8u) & 0xFFu),
                             pwl8_to_linear((lo >> 16u) & 0xFFu), f32(lo >> 24u) / 255.0);
        }
        case 2u, 10u: {
            return vec4<f32>(f32(lo & 1023u) / 1023.0, f32((lo >> 10u) & 1023u) / 1023.0,
                             f32((lo >> 20u) & 1023u) / 1023.0, f32(lo >> 30u) / 3.0);
        }
        case 3u, 12u: {
            return vec4<f32>(f7e3_to_f32(lo), f7e3_to_f32(lo >> 10u), f7e3_to_f32(lo >> 20u), f32(lo >> 30u) / 3.0);
        }
        case 4u: {
            return vec4<f32>(edram16_to_f32(lo), edram16_to_f32(lo >> 16u), 0.0, 0.0);
        }
        case 5u: {
            return vec4<f32>(edram16_to_f32(lo), edram16_to_f32(lo >> 16u), edram16_to_f32(hi),
                             edram16_to_f32(hi >> 16u));
        }
        case 6u: {
            return vec4<f32>(unpack2x16float(lo), 0.0, 0.0);
        }
        case 7u: {
            return vec4<f32>(unpack2x16float(lo), unpack2x16float(hi));
        }
        case 14u: {
            return vec4<f32>(bitcast<f32>(lo), 0.0, 0.0, 0.0);
        }
        case 15u: {
            return vec4<f32>(bitcast<f32>(lo), bitcast<f32>(hi), 0.0, 0.0);
        }
        default: {
            return vec4<f32>(0.0);
        }
    }
}

// Depth codes: 16 = D24S8, 17 = D24FS8. EDRAM depth dwords are depth << 8 | stencil.
// Host D24FS8 depth holds half the guest value (like Xenia's full_float24_in_0_to_1, matched by
// the viewport's depth range): 20e4 spans [0, 2) and host depth buffers only [0, 1], and EDRAM
// tiles of other formats are reinterpreted as depth when a title aliases them (Skyrim's HUD
// passes do, over the finished frame), so every 24-bit pattern must survive the round trip.
fn pack_depth(d: f32, stencil: u32, fmt: u32) -> u32 {
    var d24: u32;
    if (fmt == 17u) {
        d24 = f32_to_20e4(d * 2.0);
    } else {
        d24 = f32_to_unorm24(d);
    }
    return (d24 << 8u) | (stencil & 0xFFu);
}

fn unpack_depth(bits: u32, fmt: u32) -> f32 {
    if (fmt == 17u) {
        return f20e4_to_f32(bits >> 8u) * 0.5;
    }
    return unorm24_to_f32(bits >> 8u);
}

fn fullscreen(i: u32) -> vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}
"#;

fn src_decl(kind: SrcKind) -> &'static str {
    match kind {
        SrcKind::Color { multisampled: false } => "@group(0) @binding(1) var src_tex: texture_2d<f32>;\n",
        SrcKind::Color { multisampled: true } => "@group(0) @binding(1) var src_tex: texture_multisampled_2d<f32>;\n",
        SrcKind::Depth { multisampled: false } => {
            "@group(0) @binding(1) var src_depth: texture_depth_2d;\n\
             @group(0) @binding(2) var src_stencil: texture_2d<u32>;\n"
        }
        SrcKind::Depth { multisampled: true } => {
            "@group(0) @binding(1) var src_depth: texture_depth_multisampled_2d;\n\
             @group(0) @binding(2) var src_stencil: texture_multisampled_2d<u32>;\n"
        }
    }
}

fn src_load(kind: SrcKind) -> String {
    let index = if kind.multisampled() { "i32(sample)" } else { "0" };
    let body = match kind {
        SrcKind::Color { .. } => format!("return pack_color(textureLoad(src_tex, c, {index}), fmt);"),
        SrcKind::Depth { .. } => format!(
            "return vec2<u32>(pack_depth(textureLoad(src_depth, c, {index}), textureLoad(src_stencil, c, {index}).r, \
             fmt), 0u);"
        ),
    };
    format!(
        "fn load_src_packed(px: vec2<u32>, sample: u32, fmt: u32) -> vec2<u32> {{\n    let c = vec2<i32>(px);\n    \
         {body}\n}}\n"
    )
}

/// Ownership transfer: entry points `vs`, `fs_color`, and `fs_depth_scratch` (depth, plus the
/// stencil byte | 0x100 in a `u32` color output for [`stencil_bits_module`]).
pub(crate) fn transfer_module(src: SrcKind, dest_multisampled: bool, scale: u32) -> String {
    let (sample_param, sample) =
        if dest_multisampled { (", @builtin(sample_index) si: u32", "si") } else { ("", "0u") };
    format!(
        "{common}{decl}{load}{body}",
        common = common(scale),
        decl = src_decl(src),
        load = src_load(src),
        body = TRANSFER_BODY.replace("SAMPLE_PARAM", sample_param).replace("SAMPLE_INDEX", sample),
    )
}

const TRANSFER_BODY: &str = r#"
struct TransferParams {
    dst_base: u32,
    dst_pitch: u32,
    dst_msaa: u32,
    dst_flags: u32,
    dst_format: u32,
    src_base: u32,
    src_pitch: u32,
    src_msaa: u32,
    src_flags: u32,
    src_format: u32,
    src_width: u32,
    src_height: u32,
    range_start: u32,
    range_len: u32,
    pad0: u32,
    pad1: u32,
    cutout: vec4<u32>,
}

@group(0) @binding(0) var<uniform> p: TransferParams;

fn src_dword(a: u32) -> u32 {
    let loc = edram_locate(a, p.src_base, p.src_pitch, p.src_msaa, (p.src_flags & 1u) != 0u, (p.src_flags & 2u) != 0u);
    if (loc.px.x >= p.src_width || loc.px.y >= p.src_height) {
        return 0u;
    }
    let v = load_src_packed(loc.px, host_sample_of_guest(loc.sample, p.src_msaa), p.src_format);
    return select(v.x, v.y, loc.half != 0u);
}

fn fetch(pos: vec4<f32>, host_sample: u32) -> vec2<u32> {
    let px = vec2<u32>(pos.xy);
    if (all(px >= p.cutout.xy) && all(px < p.cutout.zw)) {
        discard;
    }
    let is_64 = (p.dst_flags & 2u) != 0u;
    let a = edram_dword(px, guest_sample_of_host(host_sample, p.dst_msaa), p.dst_base, p.dst_pitch, p.dst_msaa,
                        (p.dst_flags & 1u) != 0u, is_64);
    if ((((a / TILE_DWORDS) - p.range_start) & (EDRAM_TILES - 1u)) >= p.range_len) {
        discard;
    }
    var hi = 0u;
    if (is_64) {
        hi = src_dword(a + 1u);
    }
    return vec2<u32>(src_dword(a), hi);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return fullscreen(i);
}

@fragment
fn fs_color(@builtin(position) pos: vec4<f32>SAMPLE_PARAM) -> @location(0) vec4<f32> {
    let v = fetch(pos, SAMPLE_INDEX);
    return unpack_color(v.x, v.y, p.dst_format);
}

struct DepthScratch {
    @builtin(frag_depth) depth: f32,
    @location(0) stencil: u32,
}

@fragment
fn fs_depth_scratch(@builtin(position) pos: vec4<f32>SAMPLE_PARAM) -> DepthScratch {
    let v = fetch(pos, SAMPLE_INDEX);
    return DepthScratch(unpack_depth(v.x, p.dst_format), (v.x & 0xFFu) | 0x100u);
}
"#;

/// Writes one stencil bit (override `STENCIL_BIT`) from the scratch texture `fs_depth_scratch`
/// filled: fragments it didn't write (0) or whose bit is clear are discarded. Entries `vs`, `fs`.
/// Stencil can't be exported from shaders, and these passes only load one texel each instead of
/// fetching the source through the EDRAM layout eight more times.
pub(crate) fn stencil_bits_module(multisampled: bool) -> String {
    let (ty, param, sample) = if multisampled {
        ("texture_multisampled_2d<u32>", ", @builtin(sample_index) si: u32", "i32(si)")
    } else {
        ("texture_2d<u32>", "", "0")
    };
    format!(
        "@group(0) @binding(0) var scratch: {ty};\n\
         override STENCIL_BIT: u32 = 0u;\n\
         @vertex\n\
         fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{\n    \
             let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n    \
             return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);\n\
         }}\n\
         @fragment\n\
         fn fs(@builtin(position) pos: vec4<f32>{param}) {{\n    \
             let v = textureLoad(scratch, vec2<i32>(pos.xy), {sample}).r;\n    \
             if ((v & 0x100u) == 0u || ((v >> STENCIL_BIT) & 1u) == 0u) {{\n        \
                 discard;\n    \
             }}\n\
         }}\n"
    )
}

/// Host render target -> EDRAM-layout storage buffer, one invocation per dword. Entry `main`.
pub(crate) fn dump_module(src: SrcKind, scale: u32) -> String {
    format!("{}{}{}{DUMP_BODY}", common(scale), src_decl(src), src_load(src))
}

const DUMP_BODY: &str = r#"
struct DumpParams {
    start: u32,
    count: u32,
    base: u32,
    pitch: u32,
    msaa: u32,
    flags: u32,
    format: u32,
    width: u32,
    height: u32,
    // Where the base's pixel (0, 0) is in the source (host pixels).
    offset_x: u32,
    offset_y: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> p: DumpParams;
@group(0) @binding(3) var<storage, read_write> edram: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.y * groups.x * 256u + gid.x;
    if (i >= p.count) {
        return;
    }
    let a = (p.start + i) % EDRAM_DWORDS;
    let loc = edram_locate(a, p.base, p.pitch, p.msaa, (p.flags & 1u) != 0u, (p.flags & 2u) != 0u);
    let px = loc.px + vec2<u32>(p.offset_x, p.offset_y);
    if (px.x >= p.width || px.y >= p.height) {
        return;
    }
    let v = load_src_packed(px, host_sample_of_guest(loc.sample, p.msaa), p.format);
    edram[a] = select(v.x, v.y, loc.half != 0u);
}
"#;

/// Clears a scissored area of a host target to an EDRAM-format value: `vs`, `fs_color`, `fs_depth`.
pub(crate) fn clear_module(scale: u32) -> String {
    format!("{}{CLEAR_BODY}", common(scale))
}

const CLEAR_BODY: &str = r#"
struct ClearParams {
    lo: u32,
    hi: u32,
    format: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> p: ClearParams;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return fullscreen(i);
}

@fragment
fn fs_color() -> @location(0) vec4<f32> {
    return unpack_color(p.lo, p.hi, p.format);
}

@fragment
fn fs_depth() -> @builtin(frag_depth) f32 {
    return unpack_depth(p.lo, p.format);
}
"#;

/// Resolve copy from the EDRAM-layout buffer into linear destination words, at the guest's
/// resolution: with resolution scaling, converted colors average each guest pixel's scale x scale
/// host pixels, and raw copies take the host pixel at its center. Entry `main`.
pub(crate) fn resolve_module(scale: u32) -> String {
    format!("{}{RESOLVE_FROM_EDRAM}{RESOLVE_BODY}", common(scale))
}

/// The resolve shader reading a host texture standing for the whole surface instead of the
/// EDRAM-layout buffer (`--gpu d3d`): no dump into EDRAM layout first. Entry `main`.
pub(crate) fn resolve_direct_module(src: SrcKind, scale: u32) -> String {
    format!("{}{}{}{RESOLVE_FROM_TEXTURE}{RESOLVE_BODY}", common(scale), src_decl(src), src_load(src))
}

const RESOLVE_FROM_EDRAM: &str = r#"
@group(0) @binding(1) var<storage, read> edram: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_words: array<u32>;

// Host pixel `host`'s sample as stored in EDRAM (high word for 64bpp).
fn fetch_packed(host: vec2<u32>, sample: u32, depth: bool) -> vec2<u32> {
    let a = edram_dword(host, sample, p.base, p.pitch, p.msaa, depth, (p.flags & 2u) != 0u);
    var hi = 0u;
    if ((p.flags & 2u) != 0u) {
        hi = edram[a + 1u];
    }
    return vec2<u32>(edram[a], hi);
}
"#;

const RESOLVE_FROM_TEXTURE: &str = r#"
@group(0) @binding(3) var<storage, read_write> out_words: array<u32>;

// Host pixel `host`'s sample packed as EDRAM stores it (high word for 64bpp).
fn fetch_packed(host: vec2<u32>, sample: u32, depth: bool) -> vec2<u32> {
    let fmt = select(p.src_format, 16u + p.src_format, (p.flags & 1u) != 0u);
    let px = host + vec2<u32>(p.src_offset_x, p.src_offset_y);
    return load_src_packed(px, host_sample_of_guest(sample, p.msaa), fmt);
}
"#;

const RESOLVE_BODY: &str = r#"
struct ResolveParams {
    width: u32,
    height: u32,
    offset_x: u32,
    offset_y: u32,
    base: u32,
    pitch: u32,
    msaa: u32,
    // 1: depth source, 2: 64bpp source, 4: red/blue swap, 8: decode PWL gamma.
    flags: u32,
    src_format: u32,
    // 0: raw 32bpp, 1: raw 64bpp, 2: full conversion, 3: k_DXT3A_AS_1_1_1_1.
    mode: u32,
    sample_select: u32,
    dest_format: u32,
    dest_num: u32,
    exp_factor: f32,
    words: u32,
    dest_bpp: u32,
    copy_func: u32,
    copy_ref: u32,
    copy_mask: u32,
    // Direct resolves: where the base's pixel (0, 0) is in the source texture (host pixels).
    src_offset_x: u32,
    src_offset_y: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> p: ResolveParams;

// The host pixel at the center of guest pixel `pos`.
fn center(pos: vec2<u32>) -> vec2<u32> {
    return pos * RES_SCALE + vec2<u32>(RES_SCALE >> 1u);
}

// Guest pixel `pos`'s sample, averaged over its host pixels.
fn load_rgba(pos: vec2<u32>, sample: u32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var y = 0u; y < RES_SCALE; y++) {
        for (var x = 0u; x < RES_SCALE; x++) {
            sum += load_rgba_host(pos * RES_SCALE + vec2<u32>(x, y), sample);
        }
    }
    return sum / f32(RES_SCALE * RES_SCALE);
}

fn load_rgba_host(host: vec2<u32>, sample: u32) -> vec4<f32> {
    let v = fetch_packed(host, sample, false);
    var fmt = p.src_format;
    if (fmt == 1u && (p.flags & 8u) == 0u) {
        fmt = 0u;
    }
    return unpack_color(v.x, v.y, fmt);
}

fn first_sample(sel: u32) -> u32 {
    if (sel <= 3u) {
        return sel;
    }
    return select(0u, 2u, sel == 5u);
}

fn pack_fixed(v: f32, bits: u32, num: u32) -> u32 {
    let umax = (1u << bits) - 1u;
    var packed: u32;
    let half = select(-0.5, 0.5, v >= 0.0);
    if (num == 1u) {
        let pmax = f32((1u << (bits - 1u)) - 1u);
        packed = u32(i32(clamp(v, -1.0, 1.0) * pmax + half));
    } else if (num == 2u) {
        packed = u32(clamp(v, 0.0, f32(umax)) + 0.5);
    } else if (num == 3u) {
        let smax = f32((1u << (bits - 1u)) - 1u);
        packed = u32(i32(clamp(v, -smax - 1.0, smax) + half));
    } else {
        packed = u32(clamp(v, 0.0, 1.0) * f32(umax) + 0.5);
    }
    return packed & umax;
}

fn pack_dest_16(c: vec4<f32>) -> u32 {
    let n = p.dest_num;
    switch (p.dest_format) {
        case 3u: {
            return pack_fixed(c.r, 5u, n) | (pack_fixed(c.g, 5u, n) << 5u) | (pack_fixed(c.b, 5u, n) << 10u)
                | (pack_fixed(c.a, 1u, n) << 15u);
        }
        case 4u: {
            return pack_fixed(c.r, 5u, n) | (pack_fixed(c.g, 6u, n) << 5u) | (pack_fixed(c.b, 5u, n) << 11u);
        }
        case 5u: {
            return pack_fixed(c.r, 5u, n) | (pack_fixed(c.g, 5u, n) << 5u) | (pack_fixed(c.b, 6u, n) << 10u);
        }
        case 10u: {
            return pack_fixed(c.r, 8u, n) | (pack_fixed(c.g, 8u, n) << 8u);
        }
        case 15u: {
            return pack_fixed(c.r, 4u, n) | (pack_fixed(c.g, 4u, n) << 4u) | (pack_fixed(c.b, 4u, n) << 8u)
                | (pack_fixed(c.a, 4u, n) << 12u);
        }
        case 24u: {
            return pack_fixed(c.r, 16u, n);
        }
        default: {
            return pack2x16float(vec2<f32>(c.r, 0.0)) & 0xFFFFu;
        }
    }
}

fn pack_dest_32(c: vec4<f32>) -> u32 {
    let n = p.dest_num;
    switch (p.dest_format) {
        case 6u, 14u, 50u: {
            return pack_fixed(c.r, 8u, n) | (pack_fixed(c.g, 8u, n) << 8u) | (pack_fixed(c.b, 8u, n) << 16u)
                | (pack_fixed(c.a, 8u, n) << 24u);
        }
        case 7u, 54u: {
            return pack_fixed(c.r, 10u, n) | (pack_fixed(c.g, 10u, n) << 10u) | (pack_fixed(c.b, 10u, n) << 20u)
                | (pack_fixed(c.a, 2u, n) << 30u);
        }
        case 16u, 55u: {
            return pack_fixed(c.r, 11u, n) | (pack_fixed(c.g, 11u, n) << 11u) | (pack_fixed(c.b, 10u, n) << 22u);
        }
        case 17u, 56u: {
            return pack_fixed(c.r, 10u, n) | (pack_fixed(c.g, 11u, n) << 10u) | (pack_fixed(c.b, 11u, n) << 21u);
        }
        case 13u, 25u: {
            return pack_fixed(c.r, 16u, n) | (pack_fixed(c.g, 16u, n) << 16u);
        }
        case 31u: {
            return pack2x16float(c.rg);
        }
        default: {
            return bitcast<u32>(c.r);
        }
    }
}

fn pack_dest_64(c: vec4<f32>) -> vec2<u32> {
    let n = p.dest_num;
    switch (p.dest_format) {
        case 21u, 26u: {
            return vec2<u32>(pack_fixed(c.r, 16u, n) | (pack_fixed(c.g, 16u, n) << 16u),
                             pack_fixed(c.b, 16u, n) | (pack_fixed(c.a, 16u, n) << 16u));
        }
        case 32u: {
            return vec2<u32>(pack2x16float(c.rg), pack2x16float(c.ba));
        }
        default: {
            return bitcast<vec2<u32>>(c.rg);
        }
    }
}

fn swap_rb_32(v: u32, fmt: u32) -> u32 {
    if (fmt == 0u || fmt == 1u) {
        return (v & 0xFF00FF00u) | ((v & 0xFFu) << 16u) | ((v >> 16u) & 0xFFu);
    }
    if (fmt == 2u || fmt == 3u || fmt == 10u || fmt == 12u) {
        return (v & 0xC00FFC00u) | ((v & 0x3FFu) << 20u) | ((v >> 20u) & 0x3FFu);
    }
    return v;
}

fn bit_1111(pixel: u32, c: u32) -> u32 {
    let shift = c << 3u;
    let m = (p.copy_mask >> shift) & 0xFFu;
    let value = (pixel >> shift) & m;
    let reference = (p.copy_ref >> shift) & m;
    var relation = 1u;
    if (value < reference) {
        relation = 4u;
    } else if (value == reference) {
        relation = 2u;
    }
    return select(0u, 1u, ((p.copy_func >> (c << 2u)) & relation) != 0u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    let pos = gid.xy + vec2<u32>(p.offset_x, p.offset_y);
    let o = (gid.y * p.width + gid.x) * p.words;
    let first = first_sample(p.sample_select);
    let swap = (p.flags & 4u) != 0u;
    switch (p.mode) {
        case 0u: {
            var v = fetch_packed(center(pos), first, (p.flags & 1u) != 0u).x;
            if (swap) {
                v = swap_rb_32(v, p.src_format);
            }
            out_words[o] = v;
        }
        case 1u: {
            let packed = fetch_packed(center(pos), first, false);
            var lo = packed.x;
            var hi = packed.y;
            if (swap && (p.src_format == 5u || p.src_format == 7u)) {
                let t = lo;
                lo = (lo & 0xFFFF0000u) | (hi & 0xFFFFu);
                hi = (hi & 0xFFFF0000u) | (t & 0xFFFFu);
            }
            out_words[o] = lo;
            out_words[o + 1u] = hi;
        }
        case 3u: {
            let v = fetch_packed(center(pos), first, false).x;
            out_words[o] = bit_1111(v, 0u) | (bit_1111(v, 1u) << 1u) | (bit_1111(v, 2u) << 2u)
                | (bit_1111(v, 3u) << 3u);
        }
        default: {
            var c = load_rgba(pos, first);
            var factor = p.exp_factor;
            if (p.sample_select >= 4u) {
                c += load_rgba(pos, first | 1u);
                factor *= 0.5;
                if (p.sample_select >= 6u) {
                    c += load_rgba(pos, 2u) + load_rgba(pos, 3u);
                    factor *= 0.5;
                }
            }
            c *= factor;
            if (p.dest_bpp == 8u) {
                // D3D routes alpha (k_8 with swap) or blue (other 8bpp with swap) to the stored component.
                var v = c.r;
                if (swap) {
                    v = select(c.b, c.a, p.dest_format == 2u);
                    // Formats without alpha read 1 per sample here.
                    let f = p.src_format;
                    if (p.dest_format == 2u && (f == 4u || f == 6u || f == 14u || f == 15u)) {
                        v = p.exp_factor;
                    }
                }
                out_words[o] = pack_fixed(v, 8u, p.dest_num);
                return;
            }
            if (swap) {
                c = c.bgra;
            }
            switch (p.dest_bpp) {
                case 16u: {
                    out_words[o] = pack_dest_16(c);
                }
                case 64u: {
                    let v = pack_dest_64(c);
                    out_words[o] = v.x;
                    out_words[o + 1u] = v.y;
                }
                case 128u: {
                    let v = bitcast<vec4<u32>>(c);
                    out_words[o] = v.x;
                    out_words[o + 1u] = v.y;
                    out_words[o + 2u] = v.z;
                    out_words[o + 3u] = v.w;
                }
                default: {
                    out_words[o] = pack_dest_32(c);
                }
            }
        }
    }
}
"#;

/// Writes resolve output (linear words from the resolve shader) to the tiled destination in a
/// GPU copy of guest memory, with the destination endian swap — the GPU counterpart of
/// `DestLayout::write`. The copy is a set of blocks in slots of the `dst_N` buffers, located
/// through `table` (per block, its slot + 1; 0 for blocks that aren't mirrored, whose writes are
/// dropped). Parameters (16 u32): base, pitch_aligned, height_aligned, is_3d, slice, offset_x,
/// offset_y, bytes_per_element, endian (Endian128), width, height, words_per_pixel, block_1111,
/// log2 of the dwords per block | log2 of the dwords per `dst_N` buffer << 8, extent_start,
/// extent_len.
pub(crate) fn store_module(atomic: bool) -> String {
    shader_variant(STORE, atomic)
}

const STORE: &str = r#"
struct Params { v: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
DST_DECLS
fn prm(i: u32) -> u32 { return params.v[i >> 2u][i & 3u]; }

// Where physical dword `i` is in the `dst_N` buffers as a whole, or 0xFFFFFFFF if its block
// isn't mirrored.
fn slot_word(i: u32) -> u32 {
    let shift = prm(13u) & 0xFFu;
    let block = i >> shift;
    if (block >= arrayLength(&table)) { return 0xFFFFFFFFu; }
    let slot = table[block];
    if (slot == 0u) { return 0xFFFFFFFFu; }
    return ((slot - 1u) << shift) | (i & ((1u << shift) - 1u));
}

fn store_word(i: u32, v: u32) {
    let j = slot_word(i);
    if (j == 0xFFFFFFFFu) { return; }
    let chunk_shift = prm(13u) >> 8u;
    let k = j & ((1u << chunk_shift) - 1u);
    switch (j >> chunk_shift) {
STORE_CASES        default: {}
    }
}

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

fn bswap(w: u32) -> u32 {
    return (w << 24u) | ((w & 0xFF00u) << 8u) | ((w >> 8u) & 0xFF00u) | (w >> 24u);
}

// 32-bit swap modes (8in16, 8in32, 16in32) of one dword; others leave it.
fn swap32(w: u32, endian: u32) -> u32 {
    switch endian {
        case 1u: { return ((w & 0x00FF00FFu) << 8u) | ((w >> 8u) & 0x00FF00FFu); }
        case 2u: { return bswap(w); }
        case 3u: { return (w << 16u) | (w >> 16u); }
        default: { return w; }
    }
}

fn put_bits(byte_addr: u32, value: u32, mask: u32) {
    let j = slot_word(byte_addr >> 2u);
    if (j == 0xFFFFFFFFu) { return; }
    let shift = (byte_addr & 3u) * 8u;
    let chunk_shift = prm(13u) >> 8u;
    let k = j & ((1u << chunk_shift) - 1u);
    switch (j >> chunk_shift) {
PUT_CASES        default: {}
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = prm(9u);
    let height = prm(10u);
    let bpe = prm(7u);
    let endian = prm(8u);
    let block_1111 = prm(12u) != 0u;
    var w = array<u32, 4>(0u, 0u, 0u, 0u);
    var ex: u32;
    var ey: u32;
    if (block_1111) {
        if (gid.x >= width / 4u || gid.y >= height / 4u) { return; }
        var rows = array<u32, 4>(0u, 0u, 0u, 0u);
        for (var r = 0u; r < 4u; r++) {
            for (var i = 0u; i < 4u; i++) {
                let px = gid.x * 4u + i;
                let py = gid.y * 4u + r;
                rows[r] |= (src[py * width + px] & 0xFu) << (4u * i);
            }
        }
        w[0] = rows[0] | (rows[1] << 16u);
        w[1] = rows[2] | (rows[3] << 16u);
        ex = gid.x + prm(5u) / 4u;
        ey = gid.y + prm(6u) / 4u;
    } else {
        if (gid.x >= width || gid.y >= height) { return; }
        let wpp = prm(11u);
        let i = (gid.y * width + gid.x) * wpp;
        for (var k = 0u; k < wpp; k++) {
            w[k] = src[i + k];
        }
        ex = gid.x + prm(5u);
        ey = gid.y + prm(6u);
    }
    let bpb_log2 = countTrailingZeros(bpe);
    var rel: i32;
    if (prm(3u) != 0u) {
        rel = tiled_3d(i32(ex), i32(ey), i32(prm(4u)), prm(1u), prm(2u), bpb_log2);
    } else {
        rel = tiled_2d(i32(ex), i32(ey), prm(1u), bpb_log2);
    }
    let address = i32(prm(0u)) + rel;
    let offset = address - i32(prm(14u));
    if (offset < 0 || u32(offset) + bpe > prm(15u)) { return; }
    let a = u32(address);
    switch bpe {
        case 1u: { put_bits(a, w[0], 0xFFu); }
        case 2u: {
            var v = w[0] & 0xFFFFu;
            if (endian == 1u) { v = ((v & 0xFFu) << 8u) | (v >> 8u); }
            put_bits(a, v, 0xFFFFu);
        }
        case 4u: { store_word(a >> 2u, swap32(w[0], endian)); }
        case 8u: {
            var o0 = swap32(w[0], endian);
            var o1 = swap32(w[1], endian);
            if (endian == 4u) { o0 = bswap(w[1]); o1 = bswap(w[0]); }
            if (endian == 5u) { o0 = w[0]; o1 = w[1]; }
            store_word(a >> 2u, o0);
            store_word((a >> 2u) + 1u, o1);
        }
        default: {
            var o = array<u32, 4>(swap32(w[0], endian), swap32(w[1], endian), swap32(w[2], endian), swap32(w[3], endian));
            if (endian == 4u) { o = array<u32, 4>(bswap(w[1]), bswap(w[0]), bswap(w[3]), bswap(w[2])); }
            if (endian == 5u) { o = array<u32, 4>(bswap(w[3]), bswap(w[2]), bswap(w[1]), bswap(w[0])); }
            for (var k = 0u; k < 4u; k++) {
                store_word((a >> 2u) + k, o[k]);
            }
        }
    }
}
"#;

/// Destination buffers of the store shader (bindings 3.., after the table).
pub(crate) const STORE_DST_BUFFERS: u32 = 4;

/// The shader with plain stores (`atomic == false`: every write is a whole aligned dword) or
/// with atomics (8/16-bit elements share dwords with neighbours written by other invocations).
fn shader_variant(source: &str, atomic: bool) -> String {
    let ty = if atomic { "array<atomic<u32>>" } else { "array<u32>" };
    let mut decls = String::new();
    let mut stores = String::new();
    let mut puts = String::new();
    for n in 0..STORE_DST_BUFFERS {
        decls += &format!("@group(0) @binding({}) var<storage, read_write> dst_{n}: {ty};\n", 3 + n);
        let (store, put) = if atomic {
            (
                format!("atomicStore(&dst_{n}[k], v);"),
                format!("atomicAnd(&dst_{n}[k], ~(mask << shift)); atomicOr(&dst_{n}[k], (value & mask) << shift);"),
            )
        } else {
            (
                format!("dst_{n}[k] = v;"),
                format!("dst_{n}[k] = (dst_{n}[k] & ~(mask << shift)) | ((value & mask) << shift);"),
            )
        };
        stores += &format!("        case {n}u: {{ {store} }}\n");
        puts += &format!("        case {n}u: {{ {put} }}\n");
    }
    source.replace("DST_DECLS", &decls).replace("STORE_CASES", &stores).replace("PUT_CASES", &puts)
}
