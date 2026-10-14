//! Fixed resource binding interface shared by every translated shader.
//!
//! | group | binding | resource |
//! |-------|---------|----------|
//! | 0 | 0 | vertex shaders: `var<storage, read> xe_float_constants: array<vec4<f32>, 256>` — c0..c255 |
//! | 0 | 1 | `var<uniform> xe_bool_loop: XeBoolLoopConstants` — [`BoolLoopConstants`] (registers 0x4900..0x4927) |
//! | 0 | 2 | `var<uniform> xe_fetch: XeFetchConstants` — 192 raw fetch constant dwords (registers 0x4800..0x48BF) |
//! | 0 | 3 | `var<uniform> xe_system: XeSystemConstants` — [`SystemConstants`] |
//! | 0 | 4 | pixel shaders: `var<storage, read> xe_float_constants: array<vec4<f32>, 256>` — c0..c255 |
//! | 1 | 0..3 | `var<storage, read> xe_shared_memory_N: array<u32>` — the mirrored blocks of guest physical memory: slots of [`SHARED_MEMORY_BLOCK_BYTES`], [`SHARED_MEMORY_CHUNK_BYTES`] of them per buffer ([`SHARED_MEMORY_CHUNKS`] buffers) |
//! | 1 | 4 | `var<storage, read> xe_shared_table: array<u32>` — per block of physical memory, its slot + 1 (0: not mirrored, reads as zero); dword `i` (bytes `4i..4i+3` as stored) is read with `xe_shared_load(i)` |
//! | 2 | n | textures and samplers of the texture fetches, see [`crate::TextureBinding`] |

pub const FLOAT_CONSTANTS_GROUP: u32 = 0;
/// The vertex shader's float constants.
pub const FLOAT_CONSTANTS_BINDING: u32 = 0;
/// The pixel shader's float constants.
pub const PIXEL_FLOAT_CONSTANTS_BINDING: u32 = 4;
pub const BOOL_LOOP_CONSTANTS_GROUP: u32 = 0;
pub const BOOL_LOOP_CONSTANTS_BINDING: u32 = 1;
pub const FETCH_CONSTANTS_GROUP: u32 = 0;
pub const FETCH_CONSTANTS_BINDING: u32 = 2;
pub const SYSTEM_CONSTANTS_GROUP: u32 = 0;
pub const SYSTEM_CONSTANTS_BINDING: u32 = 3;
pub const SHARED_MEMORY_GROUP: u32 = 1;
/// Binding of the first buffer of mirrored blocks of guest physical memory; the others follow.
pub const SHARED_MEMORY_BINDING: u32 = 0;
/// Buffers of mirrored blocks.
pub const SHARED_MEMORY_CHUNKS: u32 = 4;
/// Bytes (slots) per buffer of mirrored blocks.
pub const SHARED_MEMORY_CHUNK_BYTES: u32 = 32 << 20;
/// Binding of the block -> slot table.
pub const SHARED_MEMORY_TABLE_BINDING: u32 = SHARED_MEMORY_BINDING + SHARED_MEMORY_CHUNKS;
/// Guest physical memory is mirrored in blocks of this size, each in any slot.
pub const SHARED_MEMORY_BLOCK_BYTES: u32 = 64 << 10;
/// Blocks of guest physical memory (512 MiB): entries of the slot table.
pub const SHARED_MEMORY_BLOCKS: u32 = (512 << 20) / SHARED_MEMORY_BLOCK_BYTES;
pub const TEXTURE_GROUP: u32 = 2;

/// Number of vec4 float constants of a shader stage.
pub const FLOAT_CONSTANT_COUNT: u32 = 256;
/// Size of a stage's float constant binding in bytes. Only the constants a shader reads
/// ([`crate::ShaderInfo::float_constants_read`]) need to be valid.
pub const FLOAT_CONSTANTS_SIZE: u64 = FLOAT_CONSTANT_COUNT as u64 * 16;
/// Raw fetch constant dwords (32 texture fetch constants of 6 dwords, or 96 vertex ones of 2).
pub const FETCH_CONSTANT_DWORDS: usize = 192;
/// Size of the fetch constant buffer in bytes.
pub const FETCH_CONSTANTS_SIZE: u64 = FETCH_CONSTANT_DWORDS as u64 * 4;

/// Bool and loop constants, a verbatim copy of registers 0x4900..0x4907 (`SHADER_CONSTANT_BOOL_*`,
/// bool `b` is bit `b & 31` of `bools[b >> 5]`) and 0x4908..0x4927 (`SHADER_CONSTANT_LOOP_*`: count
/// in bits 0:7, initial aL in 8:15, signed aL step in 16:23). Microcode indices are used directly
/// for both shader stages.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoolLoopConstants {
    pub bools: [u32; 8],
    pub loops: [u32; 32],
}

impl BoolLoopConstants {
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C) plain u32 arrays without padding.
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u8>(), std::mem::size_of::<Self>()) }
    }
}

/// [`SystemConstants::flags`] bits.
pub mod flags {
    /// PA_CL_VTE_CNTL::VTX_XY_FMT set: the shader exported X/W, Y/W - multiply back by W.
    pub const XY_DIVIDED_BY_W: u32 = 1 << 0;
    /// PA_CL_VTE_CNTL::VTX_Z_FMT set: the shader exported Z/W.
    pub const Z_DIVIDED_BY_W: u32 = 1 << 1;
    /// PA_CL_VTE_CNTL::VTX_W0_FMT set: the exported W is W, not 1/W.
    pub const W_NOT_RECIPROCAL: u32 = 1 << 2;
    /// Load the guest vertex index from shared memory at `vertex_index_load_address` indexed by the
    /// host vertex index (instead of using the host index directly).
    pub const VERTEX_INDEX_LOAD: u32 = 1 << 3;
    /// With `VERTEX_INDEX_LOAD`, indices are 32-bit (16-bit otherwise).
    pub const VERTEX_INDEX_LOAD_32BIT: u32 = 1 << 4;
    /// The primitive is a polygon (for the ps_param_gen front/back face bit).
    pub const PRIMITIVE_POLYGONAL: u32 = 1 << 5;
    /// The primitive is a line (for ps_param_gen).
    pub const PRIMITIVE_LINE: u32 = 1 << 6;
    /// The primitive is a point sprite (for ps_param_gen; point coordinates are not available in
    /// WebGPU, so ZW are zero).
    pub const PRIMITIVE_POINT: u32 = 1 << 7;
    /// Write the pixel parameters (SQ_PROGRAM_CNTL::PARAM_GEN) to `ps_param_gen_register`.
    pub const PS_PARAM_GEN: u32 = 1 << 8;
    /// Apply the alpha test to color target 0 (otherwise `alpha_test_function` is ignored).
    pub const ALPHA_TEST: u32 = 1 << 9;
    /// Two bits per color target from here on: the range its guest format stores, which host
    /// float targets don't enforce by themselves (`COLOR_RANGE_*`).
    pub const COLOR_RANGE_SHIFT: u32 = 10;
    /// No clamp (unorm/snorm host formats clamp; 32-bit and 16-bit float targets store as is).
    pub const COLOR_RANGE_NONE: u32 = 0;
    /// 7e3 (k_2_10_10_10_FLOAT): [0, 31.875].
    pub const COLOR_RANGE_7E3: u32 = 1;
    /// 16-bit EDRAM fixed point (k_16_16, k_16_16_16_16): [-32, 32].
    pub const COLOR_RANGE_FIXED16: u32 = 2;
}

/// Alpha test comparison functions (`xenos::CompareFunction`).
pub mod compare_function {
    pub const NEVER: u32 = 0;
    pub const LESS: u32 = 1;
    pub const EQUAL: u32 = 2;
    pub const LESS_EQUAL: u32 = 3;
    pub const GREATER: u32 = 4;
    pub const NOT_EQUAL: u32 = 5;
    pub const GREATER_EQUAL: u32 = 6;
    pub const ALWAYS: u32 = 7;
}

/// Per-draw constants the translated shaders need beyond the guest registers. Layout is identical
/// to `XeSystemConstants` in [`SYSTEM_CONSTANTS_WGSL`] (checked by tests).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SystemConstants {
    /// [`flags`] bits.
    pub flags: u32,
    /// Endian swap (`xenos::Endian`) of the guest vertex index (VGT_DMA_SWAP for host-indexed
    /// draws from raw guest index data, or 0).
    pub vertex_index_endian: u32,
    /// Added to the vertex index (VGT_INDX_OFFSET).
    pub vertex_base_index: i32,
    /// Byte address of the guest index buffer for `flags::VERTEX_INDEX_LOAD`.
    pub vertex_index_load_address: u32,
    /// The vertex index is clamped to `[vertex_index_min, vertex_index_max]` (VGT_MIN/MAX_VTX_INDX).
    pub vertex_index_min: u32,
    pub vertex_index_max: u32,
    /// [`compare_function`] for the alpha test of color target 0 (with `flags::ALPHA_TEST`).
    pub alpha_test_function: u32,
    pub alpha_test_reference: f32,
    /// Guest clip space to host clip space: `xyz = xyz * ndc_scale + ndc_offset * w`.
    pub ndc_scale: [f32; 3],
    /// Minimum per-vertex point diameter (oPts.x is clamped as an integer, like PA_SU_POINT_MINMAX).
    pub point_vertex_diameter_min: f32,
    pub ndc_offset: [f32; 3],
    pub point_vertex_diameter_max: f32,
    /// Diameter used when the vertex shader doesn't export one (PA_SU_POINT_SIZE), in guest pixels.
    pub point_constant_diameter: [f32; 2],
    /// Guest pixel diameter to host NDC radius scale, for point sprite expansion by the host.
    pub point_screen_diameter_to_ndc_radius: [f32; 2],
    /// Added to the integer pixel position written by ps_param_gen (window offset).
    pub ps_param_gen_offset: [f32; 2],
    /// Temporary register receiving the pixel parameters with `flags::PS_PARAM_GEN`
    /// (SQ_CONTEXT_MISC::PARAM_GEN_POS).
    pub ps_param_gen_register: u32,
    /// Host pixels per guest pixel along each axis (render targets drawn at a higher resolution
    /// than the guest's; ps_param_gen reports guest pixels).
    pub resolution_scale: f32,
    /// Multiplier for each color target (2^RB_COLOR_INFO::COLOR_EXP_BIAS).
    pub color_exp_bias: [f32; 4],
}

impl Default for SystemConstants {
    fn default() -> Self {
        SystemConstants {
            flags: flags::W_NOT_RECIPROCAL,
            vertex_index_endian: 0,
            vertex_base_index: 0,
            vertex_index_load_address: 0,
            vertex_index_min: 0,
            vertex_index_max: u32::MAX,
            alpha_test_function: compare_function::ALWAYS,
            alpha_test_reference: 0.0,
            ndc_scale: [1.0; 3],
            point_vertex_diameter_min: 0.0,
            ndc_offset: [0.0; 3],
            point_vertex_diameter_max: f32::MAX,
            point_constant_diameter: [1.0; 2],
            point_screen_diameter_to_ndc_radius: [0.0; 2],
            ps_param_gen_offset: [0.0; 2],
            ps_param_gen_register: 0,
            resolution_scale: 1.0,
            color_exp_bias: [1.0; 4],
        }
    }
}

impl SystemConstants {
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C), every field is 4-byte aligned plain data and there is no padding.
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u8>(), std::mem::size_of::<Self>()) }
    }
}

/// WGSL declaration of the system constants, matching [`SystemConstants`].
pub const SYSTEM_CONSTANTS_WGSL: &str = "struct XeSystemConstants {
    flags: u32,
    vertex_index_endian: u32,
    vertex_base_index: i32,
    vertex_index_load_address: u32,
    vertex_index_min: u32,
    vertex_index_max: u32,
    alpha_test_function: u32,
    alpha_test_reference: f32,
    ndc_scale: vec3<f32>,
    point_vertex_diameter_min: f32,
    ndc_offset: vec3<f32>,
    point_vertex_diameter_max: f32,
    point_constant_diameter: vec2<f32>,
    point_screen_diameter_to_ndc_radius: vec2<f32>,
    ps_param_gen_offset: vec2<f32>,
    ps_param_gen_register: u32,
    resolution_scale: f32,
    color_exp_bias: vec4<f32>,
}
";

/// WGSL declarations of groups 0 and 1, identical in every translated shader of `stage`.
pub fn common_declarations(stage: crate::ShaderStage) -> String {
    let float_binding = match stage {
        crate::ShaderStage::Vertex => FLOAT_CONSTANTS_BINDING,
        crate::ShaderStage::Pixel => PIXEL_FLOAT_CONSTANTS_BINDING,
    };
    format!(
        "{SYSTEM_CONSTANTS_WGSL}
struct XeBoolLoopConstants {{
    bools: array<vec4<u32>, 2>,
    loops: array<vec4<u32>, 8>,
}}

struct XeFetchConstants {{
    words: array<vec4<u32>, 48>,
}}

@group({FLOAT_CONSTANTS_GROUP}) @binding({float_binding}) var<storage, read> xe_float_constants: array<vec4<f32>, {FLOAT_CONSTANT_COUNT}>;
@group({BOOL_LOOP_CONSTANTS_GROUP}) @binding({BOOL_LOOP_CONSTANTS_BINDING}) var<uniform> xe_bool_loop: XeBoolLoopConstants;
@group({FETCH_CONSTANTS_GROUP}) @binding({FETCH_CONSTANTS_BINDING}) var<uniform> xe_fetch: XeFetchConstants;
@group({SYSTEM_CONSTANTS_GROUP}) @binding({SYSTEM_CONSTANTS_BINDING}) var<uniform> xe_system: XeSystemConstants;
{shared}
// Dword `i` of guest physical memory (zero where it isn't mirrored).
fn xe_shared_load(i: u32) -> u32 {{
    let block = i >> {shift}u;
    if (block >= {blocks}u) {{ return 0u; }}
    let slot = xe_shared_table[block];
    if (slot == 0u) {{ return 0u; }}
    let o = (((slot - 1u) << {shift}u) & {chunk_mask:#X}u) | (i & {mask:#X}u);
    switch ((slot - 1u) >> {chunk_shift}u) {{
{cases}        default: {{}}
    }}
    return 0u;
}}
",
        shared = (0..SHARED_MEMORY_CHUNKS)
            .map(|n| format!(
                "@group({SHARED_MEMORY_GROUP}) @binding({}) var<storage, read> xe_shared_memory_{n}: array<u32>;\n",
                SHARED_MEMORY_BINDING + n
            ))
            .chain(std::iter::once(format!(
                "@group({SHARED_MEMORY_GROUP}) @binding({SHARED_MEMORY_TABLE_BINDING}) var<storage, read> xe_shared_table: array<u32>;\n"
            )))
            .collect::<String>(),
        shift = (SHARED_MEMORY_BLOCK_BYTES / 4).ilog2(),
        mask = SHARED_MEMORY_BLOCK_BYTES / 4 - 1,
        chunk_mask = SHARED_MEMORY_CHUNK_BYTES / 4 - 1,
        chunk_shift = (SHARED_MEMORY_CHUNK_BYTES / SHARED_MEMORY_BLOCK_BYTES).ilog2(),
        blocks = SHARED_MEMORY_BLOCKS,
        cases = (0..SHARED_MEMORY_CHUNKS)
            .map(|n| format!(
                "        case {n}u: {{ if (o < arrayLength(&xe_shared_memory_{n})) {{ return xe_shared_memory_{n}[o]; }} }}\n"
            ))
            .collect::<String>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_sizes() {
        assert_eq!(std::mem::size_of::<BoolLoopConstants>(), 160);
        assert_eq!(std::mem::size_of::<SystemConstants>(), 112);
        assert_eq!(std::mem::offset_of!(SystemConstants, color_exp_bias), 96);
    }
}
