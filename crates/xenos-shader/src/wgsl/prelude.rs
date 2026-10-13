//! WGSL helper functions included in every translated shader (unused ones cost nothing).

/// Packed per-format texture information, indexed by `xenos::TextureFormat`: bits 0:2 component
/// count, then four 6-bit fixed-point component widths (0 = float, compressed or unknown).
pub(crate) fn texture_format_table() -> String {
    // (component count, widths) per format 0..63.
    const FORMATS: [(u32, [u32; 4]); 64] = [
        (1, [1, 0, 0, 0]),     // k_1_REVERSE
        (1, [1, 0, 0, 0]),     // k_1
        (1, [8, 0, 0, 0]),     // k_8
        (4, [5, 5, 5, 1]),     // k_1_5_5_5
        (3, [5, 6, 5, 0]),     // k_5_6_5
        (3, [5, 5, 6, 0]),     // k_6_5_5
        (4, [8, 8, 8, 8]),     // k_8_8_8_8
        (4, [10, 10, 10, 2]),  // k_2_10_10_10
        (1, [8, 0, 0, 0]),     // k_8_A
        (1, [8, 0, 0, 0]),     // k_8_B
        (2, [8, 8, 0, 0]),     // k_8_8
        (4, [8, 8, 8, 8]),     // k_Cr_Y1_Cb_Y0_REP
        (4, [8, 8, 8, 8]),     // k_Y1_Cr_Y0_Cb_REP
        (2, [16, 16, 0, 0]),   // k_16_16_EDRAM
        (4, [8, 8, 8, 8]),     // k_8_8_8_8_A
        (4, [4, 4, 4, 4]),     // k_4_4_4_4
        (3, [11, 11, 10, 0]),  // k_10_11_11
        (3, [10, 11, 11, 0]),  // k_11_11_10
        (4, [0, 0, 0, 0]),     // k_DXT1
        (4, [0, 0, 0, 0]),     // k_DXT2_3
        (4, [0, 0, 0, 0]),     // k_DXT4_5
        (4, [16, 16, 16, 16]), // k_16_16_16_16_EDRAM
        (1, [0, 0, 0, 0]),     // k_24_8
        (1, [0, 0, 0, 0]),     // k_24_8_FLOAT
        (1, [16, 0, 0, 0]),    // k_16
        (2, [16, 16, 0, 0]),   // k_16_16
        (4, [16, 16, 16, 16]), // k_16_16_16_16
        (1, [0, 0, 0, 0]),     // k_16_EXPAND
        (2, [0, 0, 0, 0]),     // k_16_16_EXPAND
        (4, [0, 0, 0, 0]),     // k_16_16_16_16_EXPAND
        (1, [0, 0, 0, 0]),     // k_16_FLOAT
        (2, [0, 0, 0, 0]),     // k_16_16_FLOAT
        (4, [0, 0, 0, 0]),     // k_16_16_16_16_FLOAT
        (1, [0, 0, 0, 0]),     // k_32
        (2, [0, 0, 0, 0]),     // k_32_32
        (4, [0, 0, 0, 0]),     // k_32_32_32_32
        (1, [0, 0, 0, 0]),     // k_32_FLOAT
        (2, [0, 0, 0, 0]),     // k_32_32_FLOAT
        (4, [0, 0, 0, 0]),     // k_32_32_32_32_FLOAT
        (1, [8, 0, 0, 0]),     // k_32_AS_8
        (2, [8, 8, 0, 0]),     // k_32_AS_8_8
        (1, [16, 0, 0, 0]),    // k_16_MPEG
        (2, [16, 16, 0, 0]),   // k_16_16_MPEG
        (1, [8, 0, 0, 0]),     // k_8_INTERLACED
        (1, [8, 0, 0, 0]),     // k_32_AS_8_INTERLACED
        (2, [8, 8, 0, 0]),     // k_32_AS_8_8_INTERLACED
        (1, [16, 0, 0, 0]),    // k_16_INTERLACED
        (1, [16, 0, 0, 0]),    // k_16_MPEG_INTERLACED
        (2, [16, 16, 0, 0]),   // k_16_16_MPEG_INTERLACED
        (2, [8, 8, 0, 0]),     // k_DXN
        (4, [8, 8, 8, 8]),     // k_8_8_8_8_AS_16_16_16_16
        (4, [0, 0, 0, 0]),     // k_DXT1_AS_16_16_16_16
        (4, [0, 0, 0, 0]),     // k_DXT2_3_AS_16_16_16_16
        (4, [0, 0, 0, 0]),     // k_DXT4_5_AS_16_16_16_16
        (4, [10, 10, 10, 2]),  // k_2_10_10_10_AS_16_16_16_16
        (3, [11, 11, 10, 0]),  // k_10_11_11_AS_16_16_16_16
        (3, [10, 11, 11, 0]),  // k_11_11_10_AS_16_16_16_16
        (3, [0, 0, 0, 0]),     // k_32_32_32_FLOAT
        (1, [8, 0, 0, 0]),     // k_DXT3A
        (1, [8, 0, 0, 0]),     // k_DXT5A
        (2, [8, 8, 0, 0]),     // k_CTX1
        (4, [0, 0, 0, 0]),     // k_DXT3A_AS_1_1_1_1
        (4, [8, 8, 8, 8]),     // k_8_8_8_8_GAMMA_EDRAM
        (4, [0, 0, 0, 0]),     // k_2_10_10_10_FLOAT_EDRAM
    ];
    let entries: Vec<String> = FORMATS
        .iter()
        .map(|(count, w)| format!("{}u", count | (w[0] << 3) | (w[1] << 9) | (w[2] << 15) | (w[3] << 21)))
        .collect();
    format!("var<private> xe_texture_formats: array<u32, 64> = array<u32, 64>({});\n", entries.join(", "))
}

pub(crate) const PRELUDE: &str = r#"
const XE_FLT_MAX: f32 = 0x1.fffffep+127f;

// Shader Model 3 multiplication: +-0 (or denormal) times anything, including Inf and NaN, is +0.
fn xe_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(a * b, vec4<f32>(0.0), (a == vec4<f32>(0.0)) | (b == vec4<f32>(0.0)));
}

fn xe_mul1(a: f32, b: f32) -> f32 {
    return select(a * b, 0.0, a == 0.0 || b == 0.0);
}

// Direct3D 9 max/min: `a >= b ? a : b` / `a < b ? a : b` (NaN selects b).
fn xe_max(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(b, a, a >= b);
}

fn xe_min(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(b, a, a < b);
}

fn xe_max1(a: f32, b: f32) -> f32 {
    return select(b, a, a >= b);
}

fn xe_min1(a: f32, b: f32) -> f32 {
    return select(b, a, a < b);
}

fn xe_isfinite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7F800000u) != 0x7F800000u;
}

// Rounds a finite value to 21 mantissa bits (precision of the transcendental unit).
fn xe_rfp(v: f32) -> f32 {
    let bits = bitcast<u32>(v);
    let truncated = bits & 0xFFFFFFFCu;
    let rounded = truncated + 4u;
    let overflow = (rounded & 0x7F800000u) == 0x7F800000u;
    var r = select(truncated, select(rounded, truncated, overflow), (bits & 3u) >= 2u);
    r = select(r, bits, (bits & 0x7F800000u) == 0x7F800000u);
    return bitcast<f32>(r);
}

fn xe_inf_to_max(v: f32) -> f32 {
    let bits = bitcast<u32>(v);
    if (bits == 0x7F800000u) {
        return XE_FLT_MAX;
    }
    if (bits == 0xFF800000u) {
        return -XE_FLT_MAX;
    }
    return v;
}

fn xe_inf_to_zero(v: f32) -> f32 {
    let bits = bitcast<u32>(v);
    if (bits == 0x7F800000u) {
        return 0.0;
    }
    if (bits == 0xFF800000u) {
        return -0.0;
    }
    return v;
}

fn xe_dp4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let p = xe_mul(a, b);
    return 0.0 + p.x + p.y + p.z + p.w;
}

fn xe_dp3(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let p = xe_mul(a, b);
    return 0.0 + p.x + p.y + p.z;
}

fn xe_dp2add(a: vec4<f32>, b: vec4<f32>, c: f32) -> f32 {
    let p = xe_mul(a, b);
    return 0.0 + p.x + p.y + c;
}

// cube with the operand already swizzled (src0.zzxy): x = s.z, y = s.w, z = s.x.
fn xe_cube(s: vec4<f32>) -> vec4<f32> {
    let x = s.z;
    let y = s.w;
    let z = s.x;
    var r: vec4<f32>;
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) {
        let n = z < 0.0;
        r = vec4<f32>(-y, select(x, -x, n), z, select(4.0, 5.0, n));
    } else if (abs(y) >= abs(x)) {
        let n = y < 0.0;
        r = vec4<f32>(select(z, -z, n), x, y, select(2.0, 3.0, n));
    } else {
        let n = x < 0.0;
        r = vec4<f32>(-y, select(-z, z, n), x, select(0.0, 1.0, n));
    }
    r.z = r.z * 2.0;
    return r;
}

fn xe_max4(v: vec4<f32>) -> f32 {
    if (v.x >= v.y && v.x >= v.z && v.x >= v.w) {
        return v.x;
    }
    if (v.y >= v.z && v.y >= v.w) {
        return v.y;
    }
    if (v.z >= v.w) {
        return v.z;
    }
    return v.w;
}

fn xe_bool(b: u32) -> bool {
    return ((xe_bool_loop.bools[b >> 7u][(b >> 5u) & 3u] >> (b & 31u)) & 1u) != 0u;
}

fn xe_loop_constant(i: u32) -> u32 {
    return xe_bool_loop.loops[i >> 2u][i & 3u];
}

fn xe_fetch_word(i: u32) -> u32 {
    return xe_fetch.words[i >> 2u][i & 3u];
}

// Relatively addressed float constant; out-of-range indices read zero.
fn xe_float_constant(index: i32) -> vec4<f32> {
    if (index < 0 || index > 255) {
        return vec4<f32>(0.0);
    }
    return xe_float_constants[u32(index)];
}

fn xe_swap32(v: u32, endian: u32) -> u32 {
    var r = v;
    if (endian == 1u || endian == 2u) {
        r = ((r >> 8u) & 0x00FF00FFu) | ((r & 0x00FF00FFu) << 8u);
    }
    if (endian == 2u || endian == 3u) {
        r = (r >> 16u) | (r << 16u);
    }
    return r;
}

// Loads a dword of a vertex buffer; words at or past the end of the buffer read as zero.
fn xe_vfetch_word(address: i32, end: i32, endian: u32) -> u32 {
    if (u32(address) >= u32(end)) {
        return 0u;
    }
    return xe_swap32(xe_shared_load(u32(address)), endian);
}

// Xenos float16: no infinities or NaN, exponent 31 extends the range.
fn xe_half_to_f32(h: u32) -> f32 {
    let s = select(1.0, -1.0, (h & 0x8000u) != 0u);
    let e = (h >> 10u) & 0x1Fu;
    let m = f32(h & 0x3FFu);
    if (e == 0u) {
        return s * ldexp(m, -24);
    }
    return s * ldexp(1.0 + m * (1.0 / 1024.0), i32(e) - 15);
}

fn xe_clamp_point_size(v: f32) -> f32 {
    let lo = bitcast<i32>(xe_system.point_vertex_diameter_min);
    let hi = bitcast<i32>(xe_system.point_vertex_diameter_max);
    return bitcast<f32>(min(max(bitcast<i32>(v), lo), hi));
}

fn xe_pwl_gamma_to_linear(gamma: f32) -> f32 {
    let g = clamp(gamma, 0.0, 1.0);
    let p1 = g >= 64.0 / 255.0;
    let p2 = g >= 96.0 / 255.0;
    let p3 = g >= 192.0 / 255.0;
    let scale = select(select(1.0 / 1024.0, 2.0 / 1024.0, p1), select(4.0 / 1024.0, 8.0 / 1024.0, p3), p2);
    let offset = select(select(0.0, -64.0, p1), select(-256.0, -1024.0, p3), p2);
    var l = g * (255.0 * 1024.0) * scale + offset;
    l = l + trunc(l * scale);
    return l * (1.0 / 1023.0);
}

// Texture fetch constant fields (fc = texture fetch constant index 0..31).
fn xe_tex_data_dimension(fc: u32) -> u32 {
    return (xe_fetch_word(fc * 6u + 5u) >> 9u) & 3u;
}

fn xe_tex_is_3d(fc: u32) -> bool {
    return xe_tex_data_dimension(fc) == 2u;
}

// Width and height for 1D and 2D fetches (a 1D fetch constant has height 1).
fn xe_tex_size_2d(fc: u32) -> vec2<f32> {
    let w2 = xe_fetch_word(fc * 6u + 2u);
    if (xe_tex_data_dimension(fc) == 0u) {
        return vec2<f32>(f32((w2 & 0xFFFFFFu) + 1u), 1.0);
    }
    return vec2<f32>(f32((w2 & 0x1FFFu) + 1u), f32(((w2 >> 13u) & 0x1FFFu) + 1u));
}

// Width, height and depth (3D) or stack depth (stacked 2D).
fn xe_tex_size_3d(fc: u32) -> vec3<f32> {
    let w2 = xe_fetch_word(fc * 6u + 2u);
    if (xe_tex_is_3d(fc)) {
        return vec3<f32>(f32((w2 & 0x7FFu) + 1u), f32(((w2 >> 11u) & 0x7FFu) + 1u), f32((w2 >> 22u) + 1u));
    }
    return vec3<f32>(f32((w2 & 0x1FFFu) + 1u), f32(((w2 >> 13u) & 0x1FFFu) + 1u), f32((w2 >> 26u) + 1u));
}

fn xe_tex_lod_bias(fc: u32) -> f32 {
    return f32(extractBits(bitcast<i32>(xe_fetch_word(fc * 6u + 4u)), 12u, 10u)) * (1.0 / 32.0);
}

fn xe_tex_grad_exp_adjust(fc: u32) -> vec2<f32> {
    let w4 = bitcast<i32>(xe_fetch_word(fc * 6u + 4u));
    return vec2<f32>(f32(extractBits(w4, 22u, 5u)), f32(extractBits(w4, 27u, 5u)));
}

// Whether the fetch constant filters between the layers of a stacked texture.
fn xe_tex_vol_linear(fc: u32) -> bool {
    return (xe_fetch_word(fc * 6u + 4u) & 3u) != 0u;
}

// Converts SC/TC (1...2) and the face index of tfetchCube into a direction.
fn xe_cube_direction(c: vec3<f32>) -> vec3<f32> {
    let sc = c.x * 2.0 - 3.0;
    let tc = c.y * 2.0 - 3.0;
    let face = u32(clamp(c.z, 0.0, 5.0));
    let n = (face & 1u) != 0u;
    let sign = select(1.0, -1.0, n);
    switch (face >> 1u) {
        case 0u: {
            return vec3<f32>(sign, -tc, select(-sc, sc, n));
        }
        case 1u: {
            return vec3<f32>(sc, sign, select(tc, -tc, n));
        }
        default: {
            return vec3<f32>(select(sc, -sc, n), -tc, sign);
        }
    }
}

// Applies the per-component signedness (unsigned / signed / unsigned biased / gamma) and the
// integer number format of a fetch constant to one fixed-point component sampled as UNORM.
fn xe_tex_component(v: f32, sign: u32, width: u32, integer: bool) -> f32 {
    var r = v;
    switch (sign) {
        case 1u: {
            if (width > 1u) {
                let max_u = f32((1u << width) - 1u);
                var n = round(v * max_u);
                if (n >= f32(1u << (width - 1u))) {
                    n = n - (max_u + 1.0);
                }
                r = max(n / f32((1u << (width - 1u)) - 1u), -1.0);
            }
        }
        case 2u: {
            if (width > 1u) {
                let half_range = f32(1u << (width - 1u));
                r = (v * (2.0 * half_range - 1.0) - half_range) / (half_range - 1.0);
            } else {
                r = v * 2.0 - 1.0;
            }
            r = max(r, -1.0);
        }
        case 3u: {
            r = xe_pwl_gamma_to_linear(v);
        }
        default: {}
    }
    if (integer && width > 0u) {
        if (sign == 1u || sign == 2u) {
            r = r * f32((1u << (width - 1u)) - 1u);
        } else {
            r = r * f32((1u << width) - 1u);
        }
    }
    return r;
}

// What the guest render target format would store: Xenos converts color exports to the target
// format, which saturates (and turns NaN into 0); host float targets keep anything. Range 1: 7e3
// [0, 31.875]; 2: 16-bit EDRAM fixed point [-32, 32]; 0: as is.
fn xe_color_range(c: vec4<f32>, range: u32) -> vec4<f32> {
    if (range == 0u) {
        return c;
    }
    let finite = select(vec4<f32>(0.0), c, c == c);
    if (range == 1u) {
        return clamp(finite, vec4<f32>(0.0), vec4<f32>(31.875, 31.875, 31.875, 1.0));
    }
    return clamp(finite, vec4<f32>(-32.0), vec4<f32>(32.0));
}

fn xe_tex_swizzle_component(v: vec4<f32>, s: u32) -> f32 {
    if (s < 4u) {
        return v[s];
    }
    return select(0.0, 1.0, s == 5u);
}

// Turns host sampling results (guest data components in RGBA) into the guest tfetch result:
// replication of the last component of the format, signedness, number format, the fetch constant
// swizzle and the exponent bias.
fn xe_tex_post(sampled: vec4<f32>, fc: u32) -> vec4<f32> {
    let w0 = xe_fetch_word(fc * 6u);
    let w1 = xe_fetch_word(fc * 6u + 1u);
    let w3 = xe_fetch_word(fc * 6u + 3u);
    let info = xe_texture_formats[w1 & 63u];
    let count = max(info & 7u, 1u);
    let integer = (w3 & 1u) != 0u;
    var data: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let c = min(i, count - 1u);
        let width = (info >> (3u + 6u * c)) & 63u;
        let sign = (w0 >> (2u + 2u * c)) & 3u;
        data[i] = xe_tex_component(sampled[c], sign, width, integer);
    }
    let swizzle = (w3 >> 1u) & 0xFFFu;
    let result = vec4<f32>(
        xe_tex_swizzle_component(data, swizzle & 7u),
        xe_tex_swizzle_component(data, (swizzle >> 3u) & 7u),
        xe_tex_swizzle_component(data, (swizzle >> 6u) & 7u),
        xe_tex_swizzle_component(data, (swizzle >> 9u) & 7u));
    return result * ldexp(1.0, extractBits(bitcast<i32>(w3), 13u, 6u));
}

fn xe_compare(function: u32, a: f32, b: f32) -> bool {
    switch (function) {
        case 0u: {
            return false;
        }
        case 1u: {
            return a < b;
        }
        case 2u: {
            return a == b;
        }
        case 3u: {
            return a <= b;
        }
        case 4u: {
            return a > b;
        }
        case 5u: {
            return a != b;
        }
        case 6u: {
            return a >= b;
        }
        default: {
            return true;
        }
    }
}
"#;
