//! Turns the Xenos command stream (register writes, shader loads, draws) into WebGPU work.
//!
//! Draws are recorded per render pass and encoded when the render targets change; the batch is
//! submitted at resolves, swaps, when a per-draw ring fills up, and when guest memory a recorded
//! draw read has changed since. Per draw:
//! - EDRAM copy mode: a resolve (after submitting everything before it).
//! - Otherwise: primitive conversion, shader translation (cached), guest memory sync, textures,
//!   render targets, constants and pipeline, then the draw is recorded.

use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use gpu_webgpu::*;
use rustc_hash::FxHashMap;
use xenos_shader::bindings::{self, SystemConstants, compare_function, flags};
use xenos_shader::bindings::{SHARED_MEMORY_BLOCK_BYTES, SHARED_MEMORY_CHUNK_BYTES};
use xenos_shader::{
    ShaderStage as Stage, TextureBinding, TextureBindingDimension, TranslationOptions, VertexExpansion,
};
use xenos_state::RegisterSource;
use xenos_state::bitfields::*;
use xenos_state::blend::{HostBlendFactor, HostBlendOperation, OutputMergerState, normalized_color_mask};
use xenos_state::depth_stencil::{DepthStencilState as GuestDepthStencil, normalized_depth_control};
use xenos_state::edram::RenderTargetSetup;
use xenos_state::primitive::{
    ConversionOptions, DrawDescription, HostDraw, HostIndices, HostTopology, IndexBufferInfo, IndexSource,
    PrimitiveExpansion, convert_primitive, process_draw,
};
use xenos_state::resolve::{ResolveOptions, resolve_info};
use xenos_state::shader::ShaderProgramSetup;
use xenos_state::viewport::{
    CullMode as GuestCull, FrontFace as GuestFrontFace, RasterizerState, ViewportArgs, ViewportRegisters,
    host_viewport_info, scissor,
};
use xenos_state::xenos::{self, EdramMode, SourceSelect};
use xenos_texture_cache::{GpuSource, TextureCache};

use crate::float_patch::FloatPatcher;
use crate::present::FrontBuffer;
use crate::registers::{ActiveShaders, RegisterFile, dirty};
use crate::ring::Ring;
use crate::shaders::{CachedShader, ShaderCache};
use crate::shared_memory::SharedMemory;
use crate::targets::{DrawTargets, RenderTargetCache, ResolveStatus, SharedMemoryTarget};
use crate::{PhysicalMemory, TextureMemory};

/// Pixel shader textures start at this `@group(2)` binding (vertex shader ones at 0); a shader
/// uses at most 32 fetch constants with 3 bindings each.
const PIXEL_TEXTURE_BINDING_BASE: u32 = 96;
/// Per-submission constants (a submission is made first when full; Skyrim submits every ~20
/// draws on average, at each resolve).
const CONSTANT_RING_SIZE: u64 = 4 << 20;
/// Per-submission indices (converted on the CPU, or copied out of shared memory); a draw with
/// more is skipped.
const INDEX_RING_SIZE: u64 = 4 << 20;
const FLOAT_CONSTANTS_SIZE: u64 = bindings::FLOAT_CONSTANTS_SIZE;
const BOOL_LOOP_SIZE: u64 = 160;
const FETCH_SIZE: u64 = bindings::FETCH_CONSTANTS_SIZE;
const SYSTEM_SIZE: u64 = std::mem::size_of::<SystemConstants>() as u64;

/// The shared memory as the texture cache's source of GPU-only data.
struct SharedSource<'a> {
    device: &'a Device,
    shared: &'a mut SharedMemory,
    memory: &'a dyn PhysicalMemory,
    /// The batch being recorded: texture loads are queued after what it has so far.
    encoder: &'a mut Option<CommandEncoder>,
    queued: &'a mut Vec<CommandBuffer>,
}

impl GpuSource for SharedSource<'_> {
    fn covers(&mut self, ranges: &[xenos_texture::GuestRange]) -> bool {
        let ranges: Vec<(u32, u32)> = ranges.iter().map(|r| (r.address, r.length)).collect();
        self.shared.gpu_covers(self.memory, &ranges)
    }

    fn sync(&mut self, ranges: &[xenos_texture::GuestRange]) -> bool {
        // A page that changed after a recorded draw read it can't be updated before that draw
        // is submitted.
        ranges.iter().all(|r| self.shared.sync(self.device, self.memory, r.address, r.length).is_ok())
    }

    fn gather(&self, encoder: &mut CommandEncoder, address: u32, len: u32, dst: &Buffer, dst_offset: u64) -> bool {
        self.shared.gather(encoder, address, len, dst, dst_offset)
    }

    fn submit(&mut self, _device: &Device, commands: CommandBuffer) {
        // After everything encoded so far (the pass being recorded is encoded later, after it).
        if let Some(encoder) = self.encoder.take() {
            self.queued.push(encoder.finish("xenos"));
        }
        self.queued.push(commands);
    }
}

/// Guest physical memory for xenos-state.
struct StateMemory<'a>(&'a dyn PhysicalMemory);

impl xenos_state::GuestMemory for StateMemory<'_> {
    fn read_physical(&self, address: u32, out: &mut [u8]) -> bool {
        if u64::from(address) + out.len() as u64 > crate::shared_memory::PHYSICAL_SIZE {
            return false;
        }
        self.0.read(address, out);
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SlotKind {
    Texture { dimension: TextureViewDimension, filterable: bool },
    Sampler { filtering: bool },
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ColorKey {
    format: TextureFormat,
    blend: Option<[(BlendOperation, BlendFactor, BlendFactor); 2]>,
    write_mask: u32,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DepthKey {
    format: TextureFormat,
    write: bool,
    compare: CompareFunction,
    stencil: [(CompareFunction, StencilOperation, StencilOperation, StencilOperation); 2],
    read_mask: u32,
    write_mask: u32,
    bias: i32,
    slope_scale_bits: u32,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    vertex: u64,
    pixel: Option<u64>,
    /// Which texture bind group layout (see `texture_layouts`).
    texture_layout: u32,
    topology: PrimitiveTopology,
    strip_index_format: IndexFormat,
    cull: CullMode,
    front_face: FrontFace,
    colors: [Option<ColorKey>; 4],
    depth: Option<DepthKey>,
    /// Color attachment slots of the pass (bound slots up to the last one).
    color_slots: usize,
    sample_count: u32,
    sample_mask: u32,
    alpha_to_coverage: bool,
}

/// Pipelines are checked for validation errors before use: wgpu-native aborts the process when a
/// submission contains an invalid one.
enum PipelineEntry {
    Ready(RenderPipeline),
    /// Web only: compiling in the background; draws using it are skipped until it's ready.
    #[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
    Compiling(Pending<gpu_webgpu::Result<RenderPipeline>>),
    Failed,
}

/// Number of color attachment slots a pass over `targets` has.
fn color_slots(targets: &DrawTargets) -> usize {
    targets.colors.iter().rposition(Option::is_some).map_or(0, |i| i + 1)
}

struct RecordedDraw {
    pipeline: RenderPipeline,
    textures: BindGroup,
    /// Dynamic offsets of group 0, in binding order.
    constant_offsets: [u32; 5],
    viewport: [f32; 6],
    scissor: [u32; 4],
    blend_constant: Color,
    stencil_reference: u32,
    indices: Option<DrawIndices>,
    count: u32,
}

/// Records when draw stage `$i` started, for the sampled draws (see `Executor::timing_sample`).
/// A macro so it borrows only those fields.
macro_rules! mark {
    ($executor:ident, $i:expr) => {
        if $executor.timing_sample {
            $executor.stage_marks[$i] = crate::renderer::now_ms();
        }
    };
}

/// A draw's texture bindings: what `texture_set` remembers.
struct TextureSet {
    /// Shader ids and the fetch constant words the bindings read.
    key: Vec<u32>,
    /// Raw handles of the texture views bound.
    views: Vec<usize>,
    /// Its bind group layout's id (see `texture_layouts`).
    layout_id: u32,
    pipeline_layout: PipelineLayout,
    group: BindGroup,
}

/// Frames an unused texture bind group is kept.
const TEXTURE_GROUP_FRAMES: u64 = 120;

/// Texture sets remembered at most (the memo is cleared when full).
const MAX_TEXTURE_SETS: usize = 4096;

/// Where a recorded draw's indices are read from.
#[derive(Clone, Copy)]
enum DrawIndices {
    /// Converted on the CPU and stored in the index ring.
    Ring { format: IndexFormat, offset: u64 },
    /// The guest index buffer, copied on the GPU from shared memory to the gathered index ring.
    Gathered { format: IndexFormat, offset: u64 },
    /// The guest index buffer itself, in shared memory buffer `buffer`.
    Shared { buffer: usize, format: IndexFormat, offset: u64 },
}

/// The dword-aligned physical range holding `ib`'s indices (start, length): what is copied to
/// the index ring when they can't be bound where they're mirrored.
fn gathered_index_range(ib: &IndexBufferInfo) -> (u32, u64) {
    let start = ib.guest_base & !3;
    (start, (u64::from(ib.guest_base) + u64::from(ib.size_bytes())).next_multiple_of(4) - u64::from(start))
}

/// Whether the draw can read its guest index buffer straight from shared memory (like Xenia's
/// renderers do), with the vertex shader applying the endian swap to each index: lists and strips
/// (whose index buffer must then also be mirrored in consecutive slots). Strips restart at the
/// host's 0xFFFF / 0xFFFFFFFF, which are the same before and after the swap, so their guest reset
/// index must be that (or reset be off: a guest index of all ones is then taken as a restart,
/// which titles don't rely on).
fn direct_index_buffer(desc: &DrawDescription) -> Option<IndexBufferInfo> {
    use xenos::PrimitiveType as P;
    let ib = desc.index_buffer?;
    if desc.tessellation.is_some() || ib.count == 0 {
        return None;
    }
    let strip = match desc.primitive_type {
        P::LineList | P::TriangleList => false,
        P::LineStrip | P::LineStrip2D | P::TriangleStrip | P::TriStrip2D => true,
        _ => return None,
    };
    if strip {
        let all_ones = match ib.format {
            xenos::IndexFormat::Int16 => 0xFFFF,
            xenos::IndexFormat::Int32 => 0xFFFF_FFFF,
        };
        if desc.reset_index.is_some_and(|r| r != all_ones) {
            return None;
        }
    }
    Some(ib)
}

struct Pass {
    setup: RenderTargetSetup,
    targets: DrawTargets,
    draws: Vec<RecordedDraw>,
}

/// Ring offsets of constants that haven't changed since they were pushed (valid until the ring
/// rewinds).
#[derive(Default)]
struct ConstantCache {
    /// Per stage (vertex, pixel): the float constants last pushed (from c0).
    float: [Option<ConstantBlock>; 2],
    bool_loop: Option<ConstantBlock>,
    fetch: Option<ConstantBlock>,
    system: Option<ConstantBlock>,
    scratch: Vec<u8>,
    /// Changed float constant vec4s of a block being pushed.
    values: Vec<u8>,
}

/// A block of constants in the ring and its contents.
struct ConstantBlock {
    offset: u32,
    data: Vec<u8>,
}

impl ConstantBlock {
    /// The offset of `data` in the ring: `slot`'s if it holds the same bytes (or more, starting
    /// with them), else of a new copy bound with a `window` byte binding, which becomes `slot`.
    fn push(slot: &mut Option<Self>, ring: &mut Ring, data: &[u8], window: u64, stats: &mut ExecutorStats) -> u32 {
        if let Some(b) = slot
            && b.data.starts_with(data)
        {
            stats.constant_blocks_reused += 1;
            return b.offset;
        }
        let offset = ring.push_window(data, window).expect("constant ring space was checked") as u32;
        match slot {
            Some(b) => {
                b.offset = offset;
                b.data.clear();
                b.data.extend_from_slice(data);
            }
            None => *slot = Some(ConstantBlock { offset, data: data.to_vec() }),
        }
        offset
    }
}

/// Stages of a draw; `stage_marks[i]` is when stage `i` of the last draw started and the last
/// mark when it was recorded.
const DRAW_STAGES: [&str; 9] = [
    "shaders",
    "output state",
    "ring space",
    "vertex memory",
    "textures",
    "targets",
    "constants",
    "pipeline",
    "indices",
];

pub struct Executor {
    stage_marks: [f64; DRAW_STAGES.len() + 1],
    /// The draw being recorded updates `stage_marks`.
    timing_sample: bool,
    /// Texture bindings of recent draws by shader pair and the fetch constant words they read
    /// (see `TextureSet`); cleared whenever the texture cache may answer differently.
    texture_sets: FxHashMap<u64, Rc<TextureSet>>,
    texture_set_key: Vec<u32>,
    device: Device,
    memory: Arc<dyn PhysicalMemory>,
    regs: RegisterFile,
    active: ActiveShaders,
    shaders: ShaderCache,
    shared: SharedMemory,
    constants: Ring,
    indices: Ring,
    /// Guest index buffers copied out of shared memory where they aren't in consecutive slots
    /// (written by commands only).
    gathered_indices: Ring,
    constants_group: BindGroup,
    /// Float constant blocks, built on the GPU from per-draw patches.
    floats: FloatPatcher,
    shared_group: BindGroup,
    /// [`SharedMemory::generation`] `shared_group` was made for.
    shared_generation: u64,
    constants_layout: BindGroupLayout,
    shared_layout: BindGroupLayout,
    /// Texture bind group layouts by their slots, with the pipeline layouts using them and ids
    /// (for pipeline keys).
    texture_layouts: FxHashMap<Vec<(u32, SlotKind)>, (BindGroupLayout, PipelineLayout, u32)>,
    pipelines: FxHashMap<PipelineKey, PipelineEntry>,
    /// Texture bind groups by layout and bound handles, with the frame they were last used in.
    /// Texture bind groups by layout and resource handles, with the frame each was last used and
    /// the handles themselves: holding them keeps their addresses from being reused by new
    /// views while the key exists (a reused address would match a stale group).
    texture_groups: FxHashMap<Vec<u64>, (BindGroup, u64, Vec<BindingKind>)>,
    /// Completion of each submitted frame not yet known to be finished on the GPU.
    frames_in_flight: std::collections::VecDeque<Pending<gpu_webgpu::Result<()>>>,
    frame: u64,
    /// Color targets of traced passes to write out after the next submit (name, size).
    trace_dumps: Vec<(TextureView, u32, u32, String)>,
    textures: TextureCache,
    targets: Box<dyn RenderTargetCache>,
    encoder: Option<CommandEncoder>,
    /// Recorded work closed into command buffers (to put texture loads after it), submitted
    /// before `encoder` with the batch.
    queued: Vec<CommandBuffer>,
    /// Guest memory ranges a draw reads (kept to reuse the allocation).
    vertex_ranges: Vec<(u32, u32)>,
    /// Time spent creating pipelines synchronously (native).
    pipeline_time: std::time::Duration,
    pass: Option<Pass>,
    constant_cache: ConstantCache,
    resolving: bool,
    max_texture_dimension: u32,
    warned: HashSet<String>,
    pub stats: ExecutorStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutorStats {
    pub draws: u64,
    pub skipped_draws: u64,
    /// Draws skipped without a warning (nothing to rasterize), by reason.
    pub culled: u64,
    pub no_targets: u64,
    pub empty_viewport: u64,
    pub pipeline_unavailable: u64,
    pub resolves: u64,
    /// Resolves stored straight into GPU memory.
    pub gpu_resolves: u64,
    pub submissions: u64,
    pub pipelines: u64,
    /// Bytes sent to the GPU: per-draw constants, converted indices, and guest memory pages.
    pub constant_upload_bytes: u64,
    /// Of `constant_upload_bytes`: float constants of vertex and pixel shaders (patches: changed
    /// vec4s and records), and how many new blocks were for shaders that index them dynamically
    /// (all 256 bound).
    pub float_upload_bytes: [u64; 2],
    pub float_uploads_dynamic: [u64; 2],
    /// Of the float constant uploads: vec4s that differed from the stage's previous block (the
    /// rest were sent again unchanged).
    pub float_vec4s_changed: [u64; 2],
    /// Constant blocks whose contents matched the last ones pushed (not uploaded again).
    pub constant_blocks_reused: u64,
    pub index_upload_bytes: u64,
    pub memory_upload_bytes: u64,
}

/// The bind group of the shared memory slot buffers and table (recreated when a slot buffer is
/// created).
fn shared_memory_group(device: &Device, layout: &BindGroupLayout, shared: &SharedMemory) -> BindGroup {
    device.create_bind_group(
        "xenos shared memory",
        layout,
        &shared
            .chunks
            .iter()
            .zip(bindings::SHARED_MEMORY_BINDING..)
            .chain(std::iter::once((&shared.table, bindings::SHARED_MEMORY_TABLE_BINDING)))
            .map(|(buffer, binding)| BindGroupEntry {
                binding,
                resource: BindingResource::Buffer { buffer, offset: 0, size: None },
            })
            .collect::<Vec<_>>(),
    )
}

fn layout_entry(binding: u32, visibility: ShaderStage, ty: BufferBindingType, size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer { ty, has_dynamic_offset: true, min_binding_size: size },
    }
}

fn topology(t: HostTopology) -> PrimitiveTopology {
    match t {
        HostTopology::PointList => PrimitiveTopology::PointList,
        HostTopology::LineList => PrimitiveTopology::LineList,
        HostTopology::LineStrip => PrimitiveTopology::LineStrip,
        HostTopology::TriangleList => PrimitiveTopology::TriangleList,
        HostTopology::TriangleStrip => PrimitiveTopology::TriangleStrip,
    }
}

fn compare(f: xenos::CompareFunction) -> CompareFunction {
    use xenos::CompareFunction as C;
    match f {
        C::Never => CompareFunction::Never,
        C::Less => CompareFunction::Less,
        C::Equal => CompareFunction::Equal,
        C::LessEqual => CompareFunction::LessEqual,
        C::Greater => CompareFunction::Greater,
        C::NotEqual => CompareFunction::NotEqual,
        C::GreaterEqual => CompareFunction::GreaterEqual,
        C::Always => CompareFunction::Always,
    }
}

fn stencil_op(op: xenos::StencilOp) -> StencilOperation {
    use xenos::StencilOp as S;
    match op {
        S::Keep => StencilOperation::Keep,
        S::Zero => StencilOperation::Zero,
        S::Replace => StencilOperation::Replace,
        S::IncrementClamp => StencilOperation::IncrementClamp,
        S::DecrementClamp => StencilOperation::DecrementClamp,
        S::Invert => StencilOperation::Invert,
        S::IncrementWrap => StencilOperation::IncrementWrap,
        S::DecrementWrap => StencilOperation::DecrementWrap,
    }
}

fn blend_factor(f: HostBlendFactor) -> BlendFactor {
    match f {
        HostBlendFactor::Zero => BlendFactor::Zero,
        HostBlendFactor::One => BlendFactor::One,
        HostBlendFactor::Src => BlendFactor::Src,
        HostBlendFactor::OneMinusSrc => BlendFactor::OneMinusSrc,
        HostBlendFactor::SrcAlpha => BlendFactor::SrcAlpha,
        HostBlendFactor::OneMinusSrcAlpha => BlendFactor::OneMinusSrcAlpha,
        HostBlendFactor::Dst => BlendFactor::Dst,
        HostBlendFactor::OneMinusDst => BlendFactor::OneMinusDst,
        HostBlendFactor::DstAlpha => BlendFactor::DstAlpha,
        HostBlendFactor::OneMinusDstAlpha => BlendFactor::OneMinusDstAlpha,
        HostBlendFactor::SrcAlphaSaturated => BlendFactor::SrcAlphaSaturated,
        HostBlendFactor::Constant => BlendFactor::Constant,
        HostBlendFactor::OneMinusConstant => BlendFactor::OneMinusConstant,
    }
}

fn blend_operation(op: HostBlendOperation) -> BlendOperation {
    match op {
        HostBlendOperation::Add => BlendOperation::Add,
        HostBlendOperation::Subtract => BlendOperation::Subtract,
        HostBlendOperation::ReverseSubtract => BlendOperation::ReverseSubtract,
        HostBlendOperation::Min => BlendOperation::Min,
        HostBlendOperation::Max => BlendOperation::Max,
    }
}

fn has_stencil(format: TextureFormat) -> bool {
    matches!(format, TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 | TextureFormat::Stencil8)
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only used with plain integer data.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// What a skipped draw is reported as (once per distinct reason).
enum Skip {
    Silent,
    Culled,
    NoTargets,
    EmptyViewport,
    NoPipeline,
    Warn(String),
}

impl Executor {
    pub fn new(device: &Device, memory: Arc<dyn PhysicalMemory>, targets: Box<dyn RenderTargetCache>) -> Self {
        let limits = device.limits();
        let alignment = u64::from(limits.minUniformBufferOffsetAlignment.max(limits.minStorageBufferOffsetAlignment));
        let constants = Ring::new(
            device,
            "xenos constants",
            CONSTANT_RING_SIZE,
            BufferUsage::Uniform | BufferUsage::Storage,
            alignment,
        );
        let indices = Ring::new(device, "xenos indices", INDEX_RING_SIZE, BufferUsage::Index, 4);
        let gathered_indices = Ring::new(device, "xenos gathered indices", INDEX_RING_SIZE, BufferUsage::Index, 4);
        let shared = SharedMemory::new(device);
        let both = ShaderStage::Vertex | ShaderStage::Fragment;
        let constants_layout = device.create_bind_group_layout(
            "xenos constants",
            &[
                layout_entry(
                    bindings::FLOAT_CONSTANTS_BINDING,
                    ShaderStage::Vertex,
                    BufferBindingType::ReadOnlyStorage,
                    FLOAT_CONSTANTS_SIZE,
                ),
                layout_entry(bindings::BOOL_LOOP_CONSTANTS_BINDING, both, BufferBindingType::Uniform, BOOL_LOOP_SIZE),
                layout_entry(bindings::FETCH_CONSTANTS_BINDING, both, BufferBindingType::Uniform, FETCH_SIZE),
                layout_entry(bindings::SYSTEM_CONSTANTS_BINDING, both, BufferBindingType::Uniform, SYSTEM_SIZE),
                layout_entry(
                    bindings::PIXEL_FLOAT_CONSTANTS_BINDING,
                    ShaderStage::Fragment,
                    BufferBindingType::ReadOnlyStorage,
                    FLOAT_CONSTANTS_SIZE,
                ),
            ],
        );
        let floats = FloatPatcher::new(device, alignment);
        let constants_group = device.create_bind_group(
            "xenos constants",
            &constants_layout,
            &[
                (bindings::FLOAT_CONSTANTS_BINDING, FLOAT_CONSTANTS_SIZE, &floats.buffer),
                (bindings::BOOL_LOOP_CONSTANTS_BINDING, BOOL_LOOP_SIZE, &constants.buffer),
                (bindings::FETCH_CONSTANTS_BINDING, FETCH_SIZE, &constants.buffer),
                (bindings::SYSTEM_CONSTANTS_BINDING, SYSTEM_SIZE, &constants.buffer),
                (bindings::PIXEL_FLOAT_CONSTANTS_BINDING, FLOAT_CONSTANTS_SIZE, &floats.buffer),
            ]
            .map(|(binding, size, buffer)| BindGroupEntry {
                binding,
                resource: BindingResource::Buffer { buffer, offset: 0, size: Some(size) },
            }),
        );
        let shared_layout = device.create_bind_group_layout(
            "xenos shared memory",
            &(0..=bindings::SHARED_MEMORY_CHUNKS)
                .map(|n| BindGroupLayoutEntry {
                    binding: bindings::SHARED_MEMORY_BINDING + n,
                    visibility: both,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::ReadOnlyStorage,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    },
                })
                .collect::<Vec<_>>(),
        );
        let shared_group = shared_memory_group(device, &shared_layout, &shared);
        Executor {
            stage_marks: [0.0; DRAW_STAGES.len() + 1],
            timing_sample: false,
            texture_sets: FxHashMap::default(),
            texture_set_key: Vec::new(),
            device: device.clone(),
            memory,
            regs: RegisterFile::default(),
            active: ActiveShaders::default(),
            shaders: ShaderCache::default(),
            shared,
            constants,
            indices,
            gathered_indices,
            constants_group,
            floats,
            shared_generation: 0,
            shared_group,
            constants_layout,
            shared_layout,
            texture_layouts: FxHashMap::default(),
            pipelines: FxHashMap::default(),
            texture_groups: FxHashMap::default(),
            frames_in_flight: Default::default(),
            frame: 0,
            trace_dumps: Vec::new(),
            textures: {
                let config = xenos_texture_cache::CacheConfig {
                    reuse_textures: !crate::debug::NO_TEXTURE_POOL.load(std::sync::atomic::Ordering::Relaxed),
                    gpu_conversion: !crate::debug::CPU_TEXTURES.load(std::sync::atomic::Ordering::Relaxed),
                    ..Default::default()
                };
                let config = if crate::HANDHELD.load(std::sync::atomic::Ordering::Relaxed) {
                    xenos_texture_cache::CacheConfig { max_unused_frames: 120, ..config }
                } else {
                    config
                };
                TextureCache::with_config(device, xenos_texture_cache::host_caps(device), config)
            },
            targets,
            encoder: None,
            queued: Vec::new(),
            pipeline_time: std::time::Duration::ZERO,
            vertex_ranges: Vec::new(),
            pass: None,
            constant_cache: ConstantCache::default(),
            resolving: false,
            max_texture_dimension: limits.maxTextureDimension2D,
            warned: HashSet::new(),
            stats: ExecutorStats::default(),
        }
    }

    /// The front buffer VdSwap described with `fetch` (texture fetch constant dwords), loaded
    /// through the texture cache (from GPU memory if a resolve stored it there).
    pub(crate) fn front_buffer(&mut self, fetch: [u32; 6]) -> Option<FrontBuffer> {
        let fetch = xenos_texture::TextureFetchConstant::decode(fetch);
        log::trace!("renderer: front buffer {fetch:?}");
        if fetch.base_page == 0 {
            return None;
        }
        let caps = self.textures.caps();
        let mapping = xenos_texture::host::host_mapping(fetch.format, false, caps, fetch.width, fetch.height)
            .map_err(|e| self.warn_once(format!("front buffer: {e:?}")))
            .ok()?;
        let (fb, swizzle) = FrontBuffer::fetch_and_swizzle(&fetch, mapping.component_map);
        let mut constants = [0u32; 192];
        constants[..6].copy_from_slice(&fb.encode());
        let binding = TextureBinding {
            fetch_constant: 0,
            dimension: TextureBindingDimension::D2,
            texture_binding: 0,
            stacked_texture_binding: None,
            sampler_binding: 1,
            mag_filter: xenos_shader::parsed::TextureFilter::UseFetchConst,
            min_filter: xenos_shader::parsed::TextureFilter::UseFetchConst,
            mip_filter: xenos_shader::parsed::TextureFilter::UseFetchConst,
            aniso_filter: xenos_shader::parsed::AnisoFilter::UseFetchConst,
        };
        let texture_memory = TextureMemory(self.memory.clone());
        let uploaded = self.shared.uploaded;
        let mut source = SharedSource {
            device: &self.device,
            shared: &mut self.shared,
            memory: &*self.memory,
            encoder: &mut self.encoder,
            queued: &mut self.queued,
        };
        let bound = self.textures.bind_with(&self.device, &texture_memory, &constants, &binding, Some(&mut source));
        self.stats.memory_upload_bytes += self.shared.uploaded - uploaded;
        // The presenter shows it right away: submit its load.
        self.flush();
        if bound.fallback {
            return None;
        }
        Some(FrontBuffer {
            view: bound.view,
            filterable: bound.filterable,
            width: fetch.width,
            height: fetch.height,
            swizzle,
        })
    }

    /// Time spent translating shaders, creating their modules and (natively, where it's
    /// synchronous) creating pipelines so far.
    pub fn shader_times(&self) -> [std::time::Duration; 3] {
        [self.shaders.translate_time, self.shaders.module_time, self.pipeline_time]
    }

    pub fn mirror_stats(&self) -> crate::shared_memory::MirrorStats {
        self.shared.stats()
    }

    pub fn texture_stats(&self) -> xenos_texture_cache::CacheStats {
        self.textures.stats()
    }

    pub fn registers(&self) -> &RegisterFile {
        &self.regs
    }

    pub fn write_registers(&mut self, writes: &[(u32, u32)]) {
        for &(index, value) in writes {
            self.regs.write(index, value);
        }
    }

    pub fn load_shader(&mut self, stage: Stage, code: Arc<[u32]>) {
        self.active.set(stage, code);
    }

    /// Whether a resolve is still in flight (polls it). Nothing else may run until it finishes.
    pub fn busy(&mut self) -> bool {
        // Frames the GPU hasn't finished: past MAX_FRAMES_IN_FLIGHT, wait. Nothing else stops
        // the renderer running ahead of a slower GPU, and every queued frame keeps its uploads
        // and released resources alive until the GPU is done with it (out of memory follows).
        while self.frames_in_flight.front().is_some_and(|f| f.is_ready()) {
            if let Some(Err(e)) = self.frames_in_flight.pop_front().and_then(|f| f.try_take()) {
                log::warn!("renderer: waiting for a frame failed: {e}");
            }
        }
        if self.frames_in_flight.len() > crate::debug::MAX_FRAMES_IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed) {
            self.device.poll();
            return true;
        }
        if !self.resolving {
            return false;
        }
        self.device.poll();
        match self.targets.poll_resolve(&self.device, &*self.memory) {
            ResolveStatus::Pending => true,
            ResolveStatus::Done { written, in_gpu_memory } => {
                self.finish_resolve(written, in_gpu_memory);
                false
            }
        }
    }

    /// Frame boundary: submits everything and ages caches.
    pub fn end_frame(&mut self) {
        self.flush();
        // Textures are revalidated against guest memory once per frame.
        self.forget_texture_sets();
        self.textures.destroy_retired();
        self.targets.end_frame();
        self.shared.end_frame();
        self.frames_in_flight.push_back(self.device.queue().on_submitted_work_done());
        self.textures.end_frame();
        let frame = self.frame;
        // Kept a while unused: pooled textures (with the same views) come back a few frames later.
        self.texture_groups.retain(|_, (_, used, _)| *used + TEXTURE_GROUP_FRAMES >= frame);
        self.frame += 1;
    }

    /// Encodes and submits everything recorded.
    pub fn flush(&mut self) {
        self.encode_pass();
        self.stats.constant_upload_bytes += self.constants.upload(&self.device);
        self.stats.index_upload_bytes += self.indices.upload(&self.device);
        // The float constant blocks are built first, in a command buffer of their own.
        let patches = self.floats.encode(&self.device).map(|(commands, bytes)| {
            self.stats.constant_upload_bytes += bytes;
            commands
        });
        self.targets.flush_uploads(&self.device);
        let main = self.encoder.take().map(|encoder| encoder.finish("xenos"));
        if patches.is_some() || main.is_some() || !self.queued.is_empty() {
            self.device.queue().submit(patches.into_iter().chain(self.queued.drain(..)).chain(main));
            self.stats.submissions += 1;
        }
        self.textures.end_batch();
        self.floats.rewind();
        #[cfg(not(target_os = "emscripten"))]
        for (view, width, height, name) in std::mem::take(&mut self.trace_dumps) {
            let path = std::path::Path::new("target/shader-dump").join(format!("{name}.png"));
            if let Err(e) = crate::native::dump_view(&self.device, &view, width, height, &path) {
                log::error!("trace: {}: {e}", path.display());
            }
        }
        self.constants.rewind();
        self.indices.rewind();
        self.gathered_indices.rewind();
        self.shared.end_batch();
        // The ring rewound: nothing pushed before can be reused.
        let scratch = std::mem::take(&mut self.constant_cache.scratch);
        let values = std::mem::take(&mut self.constant_cache.values);
        self.constant_cache = ConstantCache { scratch, values, ..Default::default() };
    }

    fn encoder(&mut self) -> &mut CommandEncoder {
        self.encoder.get_or_insert_with(|| self.device.create_command_encoder("xenos"))
    }

    /// Encodes the recorded render pass (if any) into the open encoder.
    fn encode_pass(&mut self) {
        let Some(pass) = self.pass.take() else { return };
        if pass.draws.is_empty() {
            return;
        }
        let constants_group = self.constants_group.clone();
        if self.shared_generation != self.shared.generation {
            self.shared_group = shared_memory_group(&self.device, &self.shared_layout, &self.shared);
            self.shared_generation = self.shared.generation;
        }
        let shared_group = self.shared_group.clone();
        let index_buffer = self.indices.buffer.clone();
        let gathered_buffer = self.gathered_indices.buffer.clone();
        let shared_buffers = self.shared.chunks.clone();
        let encoder = self.encoder();
        let colors: Vec<_> = pass.targets.colors[..color_slots(&pass.targets)]
            .iter()
            .map(|c| c.as_ref().map(|(view, _)| RenderPassColorAttachment::load(view)))
            .collect();
        let depth = pass.targets.depth.as_ref().map(|(view, format)| {
            let stencil = has_stencil(*format);
            RenderPassDepthStencilAttachment {
                view,
                depth_load_op: LoadOp::Load,
                depth_store_op: StoreOp::Store,
                depth_clear_value: 0.0,
                depth_read_only: false,
                stencil_load_op: if stencil { LoadOp::Load } else { LoadOp::Undefined },
                stencil_store_op: if stencil { StoreOp::Store } else { StoreOp::Undefined },
                stencil_clear_value: 0,
                stencil_read_only: false,
            }
        });
        let mut rp = encoder.begin_render_pass("xenos draws", &colors, depth.as_ref());
        rp.set_bind_group(bindings::SHARED_MEMORY_GROUP, &shared_group, &[]);
        // Each WebGPU call has a fixed cost (in browsers especially): state equal to the previous
        // draw's isn't set again, and index buffers are bound whole once, each draw selecting
        // its indices with `first_index`.
        let mut last_pipeline = None;
        let mut last_constants = None;
        let mut last_textures = None;
        let mut last_viewport = None;
        let mut last_scissor = None;
        let mut last_blend = None;
        let mut last_stencil = None;
        // (shared memory buffer, usize::MAX for the ring or usize::MAX - 1 for the gathered ring;
        // format)
        let mut last_index_buffer: Option<(usize, IndexFormat)> = None;
        let one_pixel = crate::debug::SCISSOR_ONE_PIXEL.load(std::sync::atomic::Ordering::Relaxed);
        let skip_draws = crate::debug::SKIP_DRAW_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        for d in &pass.draws {
            let pipeline = d.pipeline.raw() as usize;
            if last_pipeline != Some(pipeline) {
                rp.set_pipeline(&d.pipeline);
                last_pipeline = Some(pipeline);
            }
            if last_constants != Some(d.constant_offsets) {
                rp.set_bind_group(bindings::FLOAT_CONSTANTS_GROUP, &constants_group, &d.constant_offsets);
                last_constants = Some(d.constant_offsets);
            }
            let textures = d.textures.raw() as usize;
            if last_textures != Some(textures) {
                rp.set_bind_group(bindings::TEXTURE_GROUP, &d.textures, &[]);
                last_textures = Some(textures);
            }
            let viewport = d.viewport.map(f32::to_bits);
            if last_viewport != Some(viewport) {
                let [x, y, w, h, z0, z1] = d.viewport;
                rp.set_viewport(x, y, w, h, z0, z1);
                last_viewport = Some(viewport);
            }
            let scissor = if one_pixel { [d.scissor[0], d.scissor[1], 1, 1] } else { d.scissor };
            if last_scissor != Some(scissor) {
                let [sx, sy, sw, sh] = scissor;
                rp.set_scissor_rect(sx, sy, sw, sh);
                last_scissor = Some(scissor);
            }
            let c = d.blend_constant;
            let blend = [c.r, c.g, c.b, c.a].map(f64::to_bits);
            if last_blend != Some(blend) {
                rp.set_blend_constant(d.blend_constant);
                last_blend = Some(blend);
            }
            if last_stencil != Some(d.stencil_reference) {
                rp.set_stencil_reference(d.stencil_reference);
                last_stencil = Some(d.stencil_reference);
            }
            if skip_draws {
                continue;
            }
            let (source, format, offset) = match d.indices {
                None => {
                    rp.draw(d.count, 1, 0, 0);
                    continue;
                }
                Some(DrawIndices::Ring { format, offset, .. }) => (usize::MAX, format, offset),
                Some(DrawIndices::Gathered { format, offset, .. }) => (usize::MAX - 1, format, offset),
                Some(DrawIndices::Shared { buffer, format, offset, .. }) => (buffer, format, offset),
            };
            if last_index_buffer != Some((source, format)) {
                let buffer = match source {
                    usize::MAX => &index_buffer,
                    s if s == usize::MAX - 1 => &gathered_buffer,
                    s => &shared_buffers[s],
                };
                rp.set_index_buffer(buffer, format, 0, None);
                last_index_buffer = Some((source, format));
            }
            let index_size = if format == IndexFormat::Uint32 { 4 } else { 2 };
            rp.draw_indexed(d.count, 1, (offset / index_size) as u32, 0, 0);
        }
        rp.end();
        if crate::debug::tracing(self.frame) && pass.targets.sample_count == 1 {
            for (slot, color) in pass.targets.colors.iter().enumerate() {
                let Some((view, format)) = color else { continue };
                let name = format!("f{}-d{}-rt{slot}-{format:?}", self.frame, self.stats.draws);
                self.trace_dumps.push((view.clone(), pass.targets.width, pass.targets.height, name));
            }
        }
    }

    fn warn_once(&mut self, message: String) {
        if self.warned.len() < 4096 && self.warned.insert(message.clone()) {
            log::warn!("renderer: {message}");
        }
    }

    /// Executes the draw described by the registers (VGT_DRAW_INITIATOR etc. already written).
    pub fn draw(&mut self) {
        debug_assert!(!self.resolving, "draw while a resolve is pending");
        let mode = RbModecontrol::read(&self.regs).edram_mode;
        match mode {
            EdramMode::Copy => return self.resolve(),
            EdramMode::ColorDepth | EdramMode::DepthOnly => {}
            _ => return,
        }
        log::trace!("renderer: draw {}", self.stats.draws + self.stats.skipped_draws);
        match self.record_draw() {
            Ok(()) => self.stats.draws += 1,
            Err(skip) => {
                self.stats.skipped_draws += 1;
                match skip {
                    Skip::Warn(m) => self.warn_once(m),
                    Skip::Culled => self.stats.culled += 1,
                    Skip::NoTargets => self.stats.no_targets += 1,
                    Skip::EmptyViewport => self.stats.empty_viewport += 1,
                    Skip::NoPipeline => self.stats.pipeline_unavailable += 1,
                    Skip::Silent => {}
                }
            }
        }
    }

    fn resolve(&mut self) {
        // Stored into the GPU copy of guest memory, a resolve is recorded into the batch after
        // the rendering before it; read back, it needs that rendering submitted.
        let gpu_store = !crate::debug::CPU_RESOLVES.load(std::sync::atomic::Ordering::Relaxed);
        if gpu_store {
            self.encode_pass();
        } else {
            self.flush();
        }
        let memory = self.memory.clone();
        let info = match resolve_info(&self.regs, &StateMemory(&*memory), &ResolveOptions::default()) {
            Ok(i) => i,
            Err(e) => return self.warn_once(format!("resolve skipped: {e:?}")),
        };
        self.stats.resolves += 1;
        // Resolves are stored into the GPU copy of guest memory (no readback; Xenia does the same
        // by default). The destination pages are brought up to date first, so the bytes the copy
        // doesn't cover stay correct there.
        log::trace!("renderer: resolve {info:?}");
        if crate::debug::tracing(self.frame) {
            log::info!("trace frame {} resolve after draw {}: {info:?}", self.frame, self.stats.draws);
        }
        // The destination's pages are synced (and so in the batch's pages: a later CPU write to
        // them can't be uploaded under the resolve output before the batch is submitted).
        let shared = gpu_store.then(|| {
            self.sync_shared(info.copy_dest_extent_start, info.copy_dest_extent_length);
        });
        let target = SharedMemoryTarget {
            chunks: &self.shared.chunks,
            table: &self.shared.table,
            block_bytes: SHARED_MEMORY_BLOCK_BYTES,
            chunk_bytes: SHARED_MEMORY_CHUNK_BYTES,
        };
        let encoder =
            gpu_store.then(|| self.encoder.get_or_insert_with(|| self.device.create_command_encoder("xenos")));
        match self.targets.resolve(&self.device, &info, &*memory, shared.map(|()| &target), encoder) {
            ResolveStatus::Pending => self.resolving = true,
            ResolveStatus::Done { written, in_gpu_memory } => self.finish_resolve(written, in_gpu_memory),
        }
    }

    fn finish_resolve(&mut self, written: Option<(u32, u32)>, in_gpu_memory: bool) {
        self.resolving = false;
        if let Some((addr, len)) = written {
            let dropped = self.textures.invalidate(addr, len);
            if !dropped.is_empty() {
                // Only the remembered sets binding a dropped texture are out of date.
                self.texture_sets.retain(|_, set| !set.views.iter().any(|v| dropped.contains(v)));
            }
            if in_gpu_memory {
                self.stats.gpu_resolves += 1;
                self.shared.mark_gpu_written(addr, len);
            } else {
                self.shared.invalidate(addr, len);
            }
        }
    }

    fn shader(&mut self, stage: Stage, options: &TranslationOptions) -> std::result::Result<Rc<CachedShader>, Skip> {
        let code = match stage {
            Stage::Vertex => self.active.vertex.clone(),
            Stage::Pixel => self.active.pixel.clone(),
        }
        .ok_or(Skip::Warn(format!("draw without a {stage:?} shader")))?;
        self.shaders.get(&self.device, stage, &code.0, code.1, options).ok_or(Skip::NoPipeline)
    }

    fn sync_shared(&mut self, addr: u32, len: u32) {
        let memory = self.memory.clone();
        let uploaded = self.shared.uploaded;
        if self.shared.sync(&self.device, &*memory, addr, len).is_err() {
            self.flush();
            self.shared.sync(&self.device, &*memory, addr, len).expect("no conflicts in an empty batch");
        }
        self.stats.memory_upload_bytes += self.shared.uploaded - uploaded;
    }

    /// Milliseconds each stage of the last sampled draw (one in 32) took, for diagnosing slow draws.
    pub fn last_draw_timing(&self) -> String {
        DRAW_STAGES
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{name} {:.1}", (self.stage_marks[i + 1] - self.stage_marks[i]).max(0.0)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn record_draw(&mut self) -> std::result::Result<(), Skip> {
        // Stage timing (a clock read per stage, a JavaScript call on the web) for one draw in 32.
        self.timing_sample = (self.stats.draws + self.stats.skipped_draws).is_multiple_of(32);
        let regs = &self.regs;
        let desc = DrawDescription::from_registers(regs);
        let program = ShaderProgramSetup::from_registers(regs);
        if !program.rasterization_enabled {
            return Err(Skip::Warn("memory export only draw (not emulated)".into()));
        }
        let options =
            ConversionOptions { bake_index_params: false, allow_16bit_indices: true, expand_point_sprites: true };
        // Mirrored first, to see whether it can be bound where it is: in consecutive slots of
        // one buffer, or else copied (on the GPU) to the index ring.
        let direct = match direct_index_buffer(&desc) {
            Some(ib) => {
                self.sync_shared(ib.guest_base, ib.size_bytes());
                Some((ib, self.shared.contiguous(ib.guest_base, ib.size_bytes())))
            }
            None => None,
        };
        let host = match direct {
            // Drawn with the guest indices as they are: only the topology is needed.
            Some((ib, ..)) => convert_primitive(
                desc.primitive_type,
                IndexSource::Auto { count: ib.count },
                None,
                desc.index_params,
                &options,
            ),
            None => process_draw(&desc, &StateMemory(&*self.memory), &options),
        }
        .map_err(|e| Skip::Warn(format!("draw skipped: {e}")))?;
        if host.vertex_count == 0 {
            return Err(Skip::Silent);
        }
        let rasterizer = RasterizerState::from_registers(&self.regs, desc.primitive_type, false);
        if rasterizer.cull_all {
            return Err(Skip::Culled);
        }

        mark!(self, 0);
        // Shaders.
        let interpolators = program.interpolator_count.clamp(1, 16);
        let expansion = match host.expansion {
            PrimitiveExpansion::None => VertexExpansion::None,
            PrimitiveExpansion::PointSprite => VertexExpansion::PointSprite,
            PrimitiveExpansion::RectangleList => VertexExpansion::RectangleList,
        };
        let vs_options = TranslationOptions {
            vs_output_interpolators: (1 << interpolators) - 1,
            vs_point_size_location: None,
            ps_interpolator_count: None,
            ps_param_gen: false,
            texture_binding_base: 0,
            vs_expansion: expansion,
        };
        let vs = self.shader(Stage::Vertex, &vs_options)?;
        let ps = if program.pixel_shader_enabled {
            let ps_options = TranslationOptions {
                vs_output_interpolators: 0,
                vs_point_size_location: None,
                ps_interpolator_count: Some(interpolators),
                ps_param_gen: program.param_gen_interpolator.is_some(),
                texture_binding_base: PIXEL_TEXTURE_BINDING_BASE,
                vs_expansion: VertexExpansion::None,
            };
            Some(self.shader(Stage::Pixel, &ps_options)?)
        } else {
            None
        };
        let regs = &self.regs;

        mark!(self, 1);
        // Output state.
        let depth_control = normalized_depth_control(regs, false);
        let ps_colors = ps.as_ref().map_or(0, |p| p.translation.color_targets_written);
        let color_mask = normalized_color_mask(regs, ps_colors, false);
        let setup = RenderTargetSetup::from_registers(regs, depth_control, color_mask, 0)
            .map_err(|e| Skip::Warn(format!("render targets: {e:?}")))?;
        if setup.bindings().next().is_none() {
            return Err(Skip::Warn(format!(
                "draw without targets: pixel shader writes {ps_colors:#X}, RB_COLOR_MASK {:#X}, RB_DEPTHCONTROL {:#X}, \
                 RB_MODECONTROL {:#X}, RB_SURFACE_INFO {:#X}",
                regs.reg(xenos_state::registers::RB_COLOR_MASK),
                regs.reg(xenos_state::registers::RB_DEPTHCONTROL),
                regs.reg(xenos_state::registers::RB_MODECONTROL),
                regs.reg(xenos_state::registers::RB_SURFACE_INFO),
            )));
        }
        let output = OutputMergerState::from_registers(regs, color_mask);
        let back_faces_visible = rasterizer.polygonal && rasterizer.cull_mode != GuestCull::Back;
        let depth_stencil = GuestDepthStencil::from_registers(regs, depth_control, back_faces_visible);
        let viewport_regs = ViewportRegisters::from_registers(regs, depth_control.z_enable);
        let viewport = host_viewport_info(&viewport_regs, &ViewportArgs::webgpu(self.max_texture_dimension));
        let guest_scissor = scissor(regs, true, false);
        // Rows the draw can touch, which is how tall its render targets are made and how much
        // EDRAM it claims. With clipping, primitives stay inside the host viewport, so below it
        // nothing is drawn even when the scissor is the whole surface (as it often is). Without
        // clipping and with an 8192x8192 scissor (Direct3D 9's screen-space draws), the vertices
        // themselves are run on the CPU to find out, like Xenia does.
        let mut min_height = (guest_scissor.offset[1] + guest_scissor.extent[1]).min(self.max_texture_dimension);
        if !viewport_regs.clip_cntl.clip_disable {
            let viewport_bottom = i64::from(viewport.xy_offset[1]) + i64::from(viewport.xy_extent[1]);
            min_height = min_height.min(viewport_bottom.clamp(0, i64::from(u32::MAX)) as u32);
        } else {
            let unclamped = scissor(regs, false, false);
            if unclamped.offset[0] + unclamped.extent[0] >= xenos::TEXTURE_2D_MAX_WIDTH_HEIGHT
                && unclamped.offset[1] + unclamped.extent[1] >= xenos::TEXTURE_2D_MAX_WIDTH_HEIGHT
            {
                min_height =
                    min_height.min(crate::extent::vertex_max_y(&desc, regs, &viewport_regs, &vs.shader, &*self.memory));
            }
        }
        let min_height = min_height.max(1);

        mark!(self, 2);
        // Index data and ring space (submit first if this draw's data may not fit).
        let index_bytes: Option<(IndexFormat, Vec<u8>)> = match &host.indices {
            HostIndices::None => None,
            HostIndices::U16(v) => Some((IndexFormat::Uint16, as_bytes(v).to_vec())),
            HostIndices::U32(v) => Some((IndexFormat::Uint32, as_bytes(v).to_vec())),
        };
        let index_len = index_bytes.as_ref().map_or(0, |(_, b)| b.len() as u64 + 4);
        let gathered_len = match direct {
            Some((ib, None)) => gathered_index_range(&ib).1 + 4,
            _ => 0,
        };
        if index_len.max(gathered_len) > INDEX_RING_SIZE {
            return Err(Skip::Warn(format!("draw with {} bytes of indices skipped", index_len.max(gathered_len))));
        }
        // Each float constant binding spans a full stage's constants past its offset.
        let constant_space = BOOL_LOOP_SIZE + FETCH_SIZE + SYSTEM_SIZE + 3 * 256;
        if self.constants.remaining() < constant_space
            || self.indices.remaining() < index_len
            || self.gathered_indices.remaining() < gathered_len
            || !self.floats.has_room(2, FLOAT_CONSTANTS_SIZE)
        {
            self.flush();
        }

        mark!(self, 3);
        // Guest memory the vertex shader reads.
        let regs = &self.regs;
        let mut ranges = std::mem::take(&mut self.vertex_ranges);
        ranges.clear();
        ranges.extend(vs.translation.vertex_bindings.iter().map(|b| {
            let f = VertexFetchConstant::read(regs, b.fetch_constant);
            (f.address_bytes(), f.size.saturating_mul(4))
        }));
        let index_load = if expansion != VertexExpansion::None && desc.source_select == SourceSelect::Dma {
            desc.index_buffer.inspect(|ib| ranges.push((ib.guest_base, ib.size_bytes())))
        } else {
            None
        };
        if let Some((ib, ..)) = direct {
            ranges.push((ib.guest_base, ib.size_bytes()));
        }
        for &(addr, len) in &ranges {
            self.sync_shared(addr, len);
        }
        self.vertex_ranges = ranges;

        mark!(self, 4);
        // Textures.
        let textures = self.texture_set(&vs, ps.as_deref());

        mark!(self, 5);
        // Render targets (encode the previous pass first if they change).
        let reuse = self.pass.as_ref().is_some_and(|p| p.setup == setup && p.targets.height >= min_height);
        if !reuse {
            self.encode_pass();
            // A traced pass's targets are written out before later passes can change them.
            if !self.trace_dumps.is_empty() {
                self.flush();
            }
            let encoder = self.encoder.get_or_insert_with(|| self.device.create_command_encoder("xenos"));
            let targets = self.targets.targets(&self.device, encoder, &setup, min_height).ok_or(Skip::NoTargets)?;
            self.pass = Some(Pass { setup: setup.clone(), targets, draws: Vec::new() });
        }
        let (target_width, target_height) = {
            let targets = &self.pass.as_ref().expect("set above").targets;
            (targets.width, targets.height)
        };

        let viewport = viewport.crop_to_render_target(target_width, target_height, true);
        let scissor = guest_scissor.clamp_to(target_width, target_height);
        if viewport.xy_extent[0] == 0 || viewport.xy_extent[1] == 0 || scissor.extent[0] == 0 || scissor.extent[1] == 0
        {
            return Err(Skip::EmptyViewport);
        }

        mark!(self, 6);
        // Constants.
        let regs = &self.regs;
        let mut system = SystemConstants::default();
        let vte = PaClVteCntl::read(regs);
        let mut f = 0;
        if vte.vtx_xy_fmt {
            f |= flags::XY_DIVIDED_BY_W;
        }
        if vte.vtx_z_fmt {
            f |= flags::Z_DIVIDED_BY_W;
        }
        if vte.vtx_w0_fmt {
            f |= flags::W_NOT_RECIPROCAL;
        }
        if let Some(ib) = index_load {
            f |= flags::VERTEX_INDEX_LOAD;
            if ib.format == xenos::IndexFormat::Int32 {
                f |= flags::VERTEX_INDEX_LOAD_32BIT;
            }
            system.vertex_index_load_address = ib.guest_base;
            system.vertex_index_endian = ib.endian.raw();
        }
        if let Some((ib, ..)) = direct {
            system.vertex_index_endian = ib.endian.raw();
        }
        if rasterizer.polygonal {
            f |= flags::PRIMITIVE_POLYGONAL;
        } else if desc.primitive_type.is_line(false) {
            f |= flags::PRIMITIVE_LINE;
        } else if desc.primitive_type == xenos::PrimitiveType::PointList {
            f |= flags::PRIMITIVE_POINT;
        }
        if let Some(r) = program.param_gen_interpolator {
            f |= flags::PS_PARAM_GEN;
            system.ps_param_gen_register = r;
        }
        if let Some(test) = output.alpha_test {
            f |= flags::ALPHA_TEST;
            system.alpha_test_function = test.func.raw();
            system.alpha_test_reference = test.reference;
        } else {
            system.alpha_test_function = compare_function::ALWAYS;
        }
        system.flags = f;
        system.vertex_base_index = desc.index_params.offset as i32;
        system.vertex_index_min = desc.index_params.min;
        system.vertex_index_max = desc.index_params.max;
        system.ndc_scale = viewport.ndc_scale;
        system.ndc_offset = viewport.ndc_offset;
        system.point_constant_diameter = rasterizer.point_size;
        system.point_vertex_diameter_min = rasterizer.point_size_min_max[0];
        system.point_vertex_diameter_max = rasterizer.point_size_min_max[1];
        system.point_screen_diameter_to_ndc_radius =
            [1.0 / viewport.xy_extent[0] as f32, 1.0 / viewport.xy_extent[1] as f32];
        for (i, binding) in setup.color.iter().enumerate() {
            if let Some(b) = binding {
                system.color_exp_bias[i] = (b.exp_bias as f32).exp2();
                use xenos_state::xenos::ColorRenderTargetFormat as F;
                let range = match b.guest_color_format {
                    Some(F::K2101010Float | F::K2101010FloatAs16161616) => flags::COLOR_RANGE_7E3,
                    Some(F::K1616 | F::K16161616) => flags::COLOR_RANGE_FIXED16,
                    _ => flags::COLOR_RANGE_NONE,
                };
                system.flags |= range << (flags::COLOR_RANGE_SHIFT + 2 * i as u32);
            }
        }
        if crate::debug::tracing(self.frame) {
            self.trace_draw(&vs, ps.as_deref(), &setup, &system);
        }
        let float_reads = [
            vs.translation.info.float_constants_read(),
            ps.as_ref().map_or(0, |p| p.translation.info.float_constants_read()),
        ];
        let constant_offsets = self.push_constants(&system, float_reads);

        mark!(self, 7);
        // Pipeline.
        let pipeline_layout = textures.pipeline_layout.clone();
        let textures_group = textures.group.clone();
        let direct_format = direct.map(|(ib, ..)| match ib.format {
            xenos::IndexFormat::Int16 => IndexFormat::Uint16,
            xenos::IndexFormat::Int32 => IndexFormat::Uint32,
        });
        let strip_index_format = match (direct_format, &host.indices, host.topology.is_strip()) {
            (_, _, false) => IndexFormat::Undefined,
            (Some(format), ..) => format,
            (None, HostIndices::U16(_), true) => IndexFormat::Uint16,
            (None, HostIndices::U32(_), true) => IndexFormat::Uint32,
            (None, HostIndices::None, true) => IndexFormat::Undefined,
        };
        let key = self.pipeline_key(
            &vs,
            ps.as_deref(),
            textures.layout_id,
            &host,
            strip_index_format,
            &rasterizer,
            &output,
            &depth_stencil,
            &self.pass.as_ref().expect("set above").targets,
        );
        if ps.is_none() && key.color_slots != 0 {
            return Err(Skip::Warn("color targets bound without a pixel shader".into()));
        }
        let pipeline = self.pipeline(key, &vs, ps.as_deref(), &pipeline_layout).ok_or(Skip::NoPipeline)?;

        mark!(self, 8);
        // Indices.
        let indices = match (direct, index_bytes) {
            (Some((ib, None)), _) => {
                let (start, len) = gathered_index_range(&ib);
                let offset = self.gathered_indices.reserve(len).expect("index ring space was checked");
                let encoder = self.encoder.get_or_insert_with(|| self.device.create_command_encoder("xenos"));
                if !self.shared.gather(encoder, start, len as u32, &self.gathered_indices.buffer, offset) {
                    return Err(Skip::Warn("index buffer not in shared memory while recording a draw".into()));
                }
                let format = direct_format.expect("set with direct");
                Some(DrawIndices::Gathered { format, offset: offset + u64::from(ib.guest_base - start) })
            }
            (Some((ib, Some((buffer, offset)))), _) => {
                // Its blocks keep their slots while the batch reads them, unless syncing this
                // draw's other data had to submit the batch and then needed the slots.
                if self.shared.contiguous(ib.guest_base, ib.size_bytes()) != Some((buffer, offset)) {
                    return Err(Skip::Warn("index buffer moved in shared memory while recording a draw".into()));
                }
                Some(DrawIndices::Shared { buffer, format: direct_format.expect("set with direct"), offset })
            }
            (None, Some((format, bytes))) => {
                let offset = self.indices.push(&bytes).expect("index ring space was checked");
                Some(DrawIndices::Ring { format, offset })
            }
            (None, None) => None,
        };

        let [r, g, b, a] = output.blend_constant.map(f64::from);
        let draw = RecordedDraw {
            pipeline,
            textures: textures_group,
            constant_offsets,
            viewport: [
                viewport.xy_offset[0] as f32,
                viewport.xy_offset[1] as f32,
                viewport.xy_extent[0] as f32,
                viewport.xy_extent[1] as f32,
                viewport.z_min,
                viewport.z_max,
            ],
            scissor: [scissor.offset[0], scissor.offset[1], scissor.extent[0], scissor.extent[1]],
            blend_constant: Color { r, g, b, a },
            stencil_reference: u32::from(depth_stencil.stencil_reference),
            indices,
            count: host.vertex_count,
        };
        self.pass.as_mut().expect("pass created above").draws.push(draw);
        mark!(self, DRAW_STAGES.len());
        Ok(())
    }

    /// Pushes the constants this draw needs; returns the dynamic offsets in binding order.
    /// `float_reads` is how many float constants the vertex and the pixel shader read: only those
    /// are uploaded (a binding still spans all 256; the rest are never read). Blocks equal to the
    /// last ones pushed are reused: titles rewrite most constants before every draw, mostly with
    /// the values already there.
    fn push_constants(&mut self, system: &SystemConstants, float_reads: [u32; 2]) -> [u32; 5] {
        let dirty_bits = self.regs.take_dirty();
        let cache = &mut self.constant_cache;
        let ring = &mut self.constants;
        let mut scratch = std::mem::take(&mut cache.scratch);
        // Like Xenia's renderers, pixel shaders read c256.. regardless of SQ_PS_CONST (titles
        // leave the SQ_*_CONST windows unset).
        let mut float = [0u32; 2];
        let mut values = std::mem::take(&mut cache.values);
        for stage in 0..2 {
            let count = float_reads[stage] as usize;
            let slot = &mut cache.float[stage];
            if count == 0 {
                // Nothing read: any offset will do.
                float[stage] = slot.as_ref().map_or(0, |b| b.offset);
                continue;
            }
            let base = stage * 256;
            let Some(block) = slot.as_mut() else {
                // The stage's first block in this submission: all of it is sent.
                scratch.clear();
                self.regs.float_constants_window(base as u32, count as u32, &mut scratch);
                let offset = self.floats.push(stage as u32, None, &scratch, FLOAT_CONSTANTS_SIZE);
                *slot = Some(ConstantBlock { offset, data: scratch.clone() });
                self.regs.clear_float_written(stage, 256);
                float[stage] = offset;
                self.stats.float_vec4s_changed[stage] += count as u64;
                self.stats.float_upload_bytes[stage] += count as u64 * 16 + 48;
                continue;
            };
            // Vec4s the block doesn't have, and the written ones that differ from it: only those
            // are compared and sent (see float_patch); the title rewrites constants before every
            // draw, but only ~5% of them change.
            let have = block.data.len() / 16;
            let written = self.regs.float_written(stage);
            let mut mask = [0u32; 8];
            values.clear();
            for i in 0..count {
                let changed = if i >= have {
                    true
                } else if written[i / 64] >> (i % 64) & 1 != 0 {
                    self.regs.float_vec4(base + i) != block.data[i * 16..i * 16 + 16]
                } else {
                    false
                };
                if changed {
                    mask[i / 32] |= 1 << (i % 32);
                    values.extend_from_slice(&self.regs.float_vec4(base + i));
                }
            }
            if values.is_empty() {
                // The block has these values (beyond `count` it keeps its own, still compared
                // when a draw reads them).
                self.regs.clear_float_written(stage, count);
                self.stats.constant_blocks_reused += 1;
                float[stage] = block.offset;
                continue;
            }
            let offset = self.floats.push_masked(stage as u32, count, mask, &values, FLOAT_CONSTANTS_SIZE);
            block.offset = offset;
            block.data.resize(count * 16, 0);
            let mut v = values.chunks(16);
            for i in 0..count {
                if mask[i / 32] >> (i % 32) & 1 != 0 {
                    block.data[i * 16..i * 16 + 16].copy_from_slice(v.next().expect("one value per bit"));
                }
            }
            // The new block is `count` long: what follows is sent when read.
            self.regs.clear_float_written(stage, 256);
            float[stage] = offset;
            let changed = values.len() / 16;
            self.stats.float_vec4s_changed[stage] += changed as u64;
            self.stats.float_upload_bytes[stage] += changed as u64 * 16 + 48;
            self.stats.float_uploads_dynamic[stage] += u64::from(count == FLOAT_CONSTANTS_SIZE as usize / 16);
        }
        cache.values = values;
        let bool_loop = match &cache.bool_loop {
            Some(b) if dirty_bits & dirty::BOOL_LOOP == 0 => b.offset,
            _ => ConstantBlock::push(
                &mut cache.bool_loop,
                ring,
                as_bytes(self.regs.bool_loop_constants()),
                BOOL_LOOP_SIZE,
                &mut self.stats,
            ),
        };
        let fetch = match &cache.fetch {
            Some(b) if dirty_bits & dirty::FETCH == 0 => b.offset,
            _ => ConstantBlock::push(
                &mut cache.fetch,
                ring,
                as_bytes(self.regs.fetch_constants()),
                FETCH_SIZE,
                &mut self.stats,
            ),
        };
        let system = ConstantBlock::push(&mut cache.system, ring, system.as_bytes(), SYSTEM_SIZE, &mut self.stats);
        cache.scratch = scratch;
        [float[0], bool_loop, fetch, system, float[1]]
    }

    /// `--render-debug trace-frame=N`: logs the draw and writes its shaders out (once each).
    fn trace_draw(
        &mut self,
        vs: &CachedShader,
        ps: Option<&CachedShader>,
        setup: &RenderTargetSetup,
        system: &SystemConstants,
    ) {
        let targets: Vec<String> = setup
            .color
            .iter()
            .flatten()
            .map(|b| format!("{:?}@{} bias {}", b.guest_color_format, b.key.base_tiles, b.exp_bias))
            .collect();
        // The fetch constants of the textures sampled (6 words each; the format is in word 1).
        let fetch = self.regs.fetch_constants();
        let mut textures = Vec::new();
        for shader in [Some(vs), ps].into_iter().flatten() {
            for b in &shader.translation.texture_bindings {
                let first = b.fetch_constant as usize * 6;
                let words: Vec<String> = fetch[first..first + 6].iter().map(|w| format!("{w:08x}")).collect();
                textures.push(format!("tf{} {}", b.fetch_constant, words.join(" ")));
            }
        }
        // Blend state, the first pixel shader constants, and each vertex stream's fetch constant
        // with its first bytes.
        let regs = &self.regs;
        let blend = format!(
            "blend {:08x} colorcontrol {:08x} mask {:08x}",
            regs.reg(xenos_state::registers::RB_BLENDCONTROL0),
            regs.reg(xenos_state::registers::RB_COLORCONTROL),
            regs.reg(xenos_state::registers::RB_COLOR_MASK)
        );
        let mut ps_consts = Vec::new();
        regs.float_constants_window(256, 8, &mut ps_consts);
        let ps_consts: Vec<String> = ps_consts
            .chunks(16)
            .map(|c| {
                let v: Vec<String> =
                    c.chunks(4).map(|f| format!("{}", f32::from_le_bytes(f.try_into().unwrap()))).collect();
                format!("({})", v.join(","))
            })
            .collect();
        let mut streams = Vec::new();
        for b in &vs.translation.vertex_bindings {
            let f = VertexFetchConstant::read(regs, b.fetch_constant);
            let mut head = [0u8; 16];
            self.memory.read(f.address_bytes(), &mut head);
            let hex: String = head.iter().map(|b| format!("{b:02x}")).collect();
            streams.push(format!("vf{} @{:#x} {:?}: {hex}", b.fetch_constant, f.address_bytes(), f.endian));
        }
        log::info!(
            "trace frame {} draw {}: {blend}; ps c0-7 {}; vertices [{}]",
            self.frame,
            self.stats.draws,
            ps_consts.join(" "),
            streams.join(", ")
        );
        log::info!(
            "trace frame {} draw {}: vs {:016x} ps {:?} targets [{}] flags {:#x} textures [{}]",
            self.frame,
            self.stats.draws,
            vs.id,
            ps.map(|p| format!("{:016x}", p.id)),
            targets.join(", "),
            system.flags,
            textures.join(", ")
        );
        #[cfg(not(target_os = "emscripten"))]
        {
            let dir = std::path::Path::new("target/shader-dump");
            let _ = std::fs::create_dir_all(dir);
            let codes =
                [(Stage::Vertex, vs, self.active.vertex.clone()), (Stage::Pixel, vs, self.active.pixel.clone())];
            for (stage, _, code) in codes {
                let shader = if stage == Stage::Vertex { Some(vs) } else { ps };
                let (Some(shader), Some((code, _))) = (shader, code) else { continue };
                let base = dir.join(format!("{}-{:016x}", if stage == Stage::Vertex { "vs" } else { "ps" }, shader.id));
                if base.with_extension("wgsl").exists() {
                    continue;
                }
                let _ = std::fs::write(base.with_extension("wgsl"), &shader.translation.wgsl);
                let disasm = xenos_shader::disassemble(stage, &code).unwrap_or_else(|e| format!("{e:?}"));
                let _ = std::fs::write(base.with_extension("txt"), disasm);
            }
        }
    }

    /// The textures the shaders' fetches bind under the current fetch constants. Draws mostly
    /// repeat recent sets, so they're remembered by the shaders and the fetch constant words read
    /// until the texture cache changes (`forget_texture_sets`).
    fn texture_set(&mut self, vs: &CachedShader, ps: Option<&CachedShader>) -> Rc<TextureSet> {
        let fetch = self.regs.fetch_constants();
        let key = &mut self.texture_set_key;
        key.clear();
        for id in [vs.id, ps.map_or(0, |p| p.id)] {
            key.extend([id as u32, (id >> 32) as u32]);
        }
        for shader in [Some(vs), ps].into_iter().flatten() {
            for binding in &shader.translation.texture_bindings {
                let first = binding.fetch_constant as usize * 6;
                key.extend_from_slice(&fetch[first..first + 6]);
            }
        }
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            key.hash(&mut h);
            h.finish()
        };
        let memo = !crate::debug::NO_TEXTURE_SET_MEMO.load(std::sync::atomic::Ordering::Relaxed);
        if memo
            && let Some(set) = self.texture_sets.get(&hash)
            && set.key == *key
        {
            return set.clone();
        }
        let fetch: [u32; 192] = *fetch;
        let texture_memory = TextureMemory(self.memory.clone());
        let mut layout_key = Vec::new();
        let mut views: Vec<(u32, BindingKind)> = Vec::new();
        for shader in [Some(vs), ps].into_iter().flatten() {
            for binding in &shader.translation.texture_bindings {
                let uploaded = self.shared.uploaded;
                let mut source = SharedSource {
                    device: &self.device,
                    shared: &mut self.shared,
                    memory: &*self.memory,
                    encoder: &mut self.encoder,
                    queued: &mut self.queued,
                };
                let bound = self.textures.bind_with(&self.device, &texture_memory, &fetch, binding, Some(&mut source));
                self.stats.memory_upload_bytes += self.shared.uploaded - uploaded;
                push_texture(&mut layout_key, &mut views, binding, bound);
            }
        }
        let (layout, pipeline_layout, layout_id) = self.texture_layout(&layout_key);
        let group = self.texture_group(&layout, &views);
        let bound_views = views
            .iter()
            .filter_map(|(_, kind)| match kind {
                BindingKind::View(v) => Some(v.raw() as usize),
                BindingKind::Sampler(_) => None,
            })
            .collect();
        let set = Rc::new(TextureSet {
            key: self.texture_set_key.clone(),
            views: bound_views,
            layout_id,
            pipeline_layout,
            group,
        });
        if self.texture_sets.len() >= MAX_TEXTURE_SETS {
            self.texture_sets.clear();
        }
        self.texture_sets.insert(hash, set.clone());
        set
    }

    /// The texture cache may bind differently now (textures dropped, reloaded or revalidated).
    fn forget_texture_sets(&mut self) {
        self.texture_sets.clear();
    }

    fn texture_layout(&mut self, key: &[(u32, SlotKind)]) -> (BindGroupLayout, PipelineLayout, u32) {
        if let Some(l) = self.texture_layouts.get(key) {
            return l.clone();
        }
        let both = ShaderStage::Vertex | ShaderStage::Fragment;
        let entries: Vec<BindGroupLayoutEntry> = key
            .iter()
            .map(|&(binding, kind)| BindGroupLayoutEntry {
                binding,
                visibility: both,
                ty: match kind {
                    SlotKind::Texture { dimension, filterable } => BindingType::Texture {
                        sample_type: if filterable {
                            TextureSampleType::Float
                        } else {
                            TextureSampleType::UnfilterableFloat
                        },
                        view_dimension: dimension,
                        multisampled: false,
                    },
                    SlotKind::Sampler { filtering } => BindingType::Sampler(if filtering {
                        SamplerBindingType::Filtering
                    } else {
                        SamplerBindingType::NonFiltering
                    }),
                },
            })
            .collect();
        let group = self.device.create_bind_group_layout("xenos textures", &entries);
        let pipeline =
            self.device.create_pipeline_layout("xenos", &[&self.constants_layout, &self.shared_layout, &group]);
        let id = self.texture_layouts.len() as u32;
        self.texture_layouts.insert(key.to_vec(), (group.clone(), pipeline.clone(), id));
        (group, pipeline, id)
    }

    fn texture_group(&mut self, layout: &BindGroupLayout, views: &[(u32, BindingKind)]) -> BindGroup {
        // The same textures are bound by many draws; bind group creation is expensive (in
        // browsers especially). Cached groups keep their views alive, so the handles in the key
        // can't be reused by other objects while the entry exists.
        let mut key = Vec::with_capacity(views.len() + 1);
        key.push(layout.raw() as usize as u64);
        key.extend(views.iter().map(|(binding, kind)| {
            (match kind {
                BindingKind::View(v) => v.raw() as usize as u64,
                BindingKind::Sampler(s) => s.raw() as usize as u64,
            }) ^ (u64::from(*binding) << 48)
        }));
        let frame = self.frame;
        if let Some((group, used, _)) = self.texture_groups.get_mut(&key) {
            *used = frame;
            return group.clone();
        }
        let entries: Vec<BindGroupEntry> = views
            .iter()
            .map(|(binding, kind)| BindGroupEntry {
                binding: *binding,
                resource: match kind {
                    BindingKind::View(v) => BindingResource::TextureView(v),
                    BindingKind::Sampler(s) => BindingResource::Sampler(s),
                },
            })
            .collect();
        let group = self.device.create_bind_group("xenos textures", layout, &entries);
        let held = views.iter().map(|(_, kind)| kind.clone()).collect();
        self.texture_groups.insert(key, (group.clone(), frame, held));
        group
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_key(
        &self,
        vs: &CachedShader,
        ps: Option<&CachedShader>,
        texture_layout: u32,
        host: &HostDraw,
        strip_index_format: IndexFormat,
        rasterizer: &RasterizerState,
        output: &OutputMergerState,
        depth_stencil: &GuestDepthStencil,
        targets: &DrawTargets,
    ) -> PipelineKey {
        // Every bound slot gets a target (write mask 0 where the draw doesn't write it), so the
        // pipeline matches the pass.
        let colors = std::array::from_fn(|i| {
            let (_, format) = targets.colors[i].as_ref()?;
            let state = output.targets[i];
            Some(ColorKey {
                format: *format,
                blend: state.and_then(|s| s.blend).filter(|_| targets.blendable[i]).map(|(c, a)| {
                    [c, a]
                        .map(|x| (blend_operation(x.operation), blend_factor(x.src_factor), blend_factor(x.dst_factor)))
                }),
                write_mask: state.map_or(0, |s| s.write_mask),
            })
        });
        let depth = targets.depth.as_ref().map(|(_, format)| {
            let stencil = has_stencil(*format) && depth_stencil.stencil_enabled;
            let face = |s: xenos_state::depth_stencil::StencilFaceState| {
                if stencil {
                    (compare(s.compare), stencil_op(s.fail_op), stencil_op(s.depth_fail_op), stencil_op(s.pass_op))
                } else {
                    (CompareFunction::Always, StencilOperation::Keep, StencilOperation::Keep, StencilOperation::Keep)
                }
            };
            let (bias, slope) = if rasterizer.polygonal {
                (rasterizer.depth_bias.constant, rasterizer.depth_bias.slope_scale)
            } else {
                (0, 0.0)
            };
            DepthKey {
                format: *format,
                write: depth_stencil.depth_write_enabled,
                compare: compare(depth_stencil.depth_compare),
                stencil: [face(depth_stencil.stencil_front), face(depth_stencil.stencil_back)],
                read_mask: if stencil { u32::from(depth_stencil.stencil_read_mask) } else { 0xFF },
                write_mask: if stencil { u32::from(depth_stencil.stencil_write_mask) } else { 0 },
                bias,
                slope_scale_bits: slope.to_bits(),
            }
        });
        PipelineKey {
            vertex: vs.id,
            pixel: ps.map(|p| p.id),
            texture_layout,
            topology: topology(host.topology),
            strip_index_format,
            cull: match rasterizer.cull_mode {
                GuestCull::None => CullMode::None,
                GuestCull::Front => CullMode::Front,
                GuestCull::Back => CullMode::Back,
            },
            front_face: match rasterizer.front_face {
                GuestFrontFace::Ccw => FrontFace::CCW,
                GuestFrontFace::Cw => FrontFace::CW,
            },
            colors,
            depth,
            color_slots: color_slots(targets),
            sample_count: targets.sample_count,
            sample_mask: targets.sample_mask,
            alpha_to_coverage: output.alpha_to_coverage.is_some() && targets.sample_count > 1,
        }
    }

    fn pipeline(
        &mut self,
        key: PipelineKey,
        vs: &CachedShader,
        ps: Option<&CachedShader>,
        layout: &PipelineLayout,
    ) -> Option<RenderPipeline> {
        match self.pipelines.get_mut(&key) {
            Some(PipelineEntry::Ready(p)) => return Some(p.clone()),
            Some(PipelineEntry::Failed) => return None,
            Some(PipelineEntry::Compiling(pending)) => {
                let result = pending.try_take()?;
                return self.compiled(key, result);
            }
            None => {}
        }
        let color_targets: Vec<ColorTargetState> = key.colors[..key.color_slots]
            .iter()
            .map(|c| match c {
                Some(c) => ColorTargetState {
                    format: c.format,
                    blend: c.blend.map(|[color, alpha]| BlendState {
                        color: BlendComponent { operation: color.0, src_factor: color.1, dst_factor: color.2 },
                        alpha: BlendComponent { operation: alpha.0, src_factor: alpha.1, dst_factor: alpha.2 },
                    }),
                    write_mask: ColorWriteMask(u64::from(c.write_mask)),
                },
                None => {
                    ColorTargetState { format: TextureFormat::Undefined, blend: None, write_mask: ColorWriteMask::None }
                }
            })
            .collect();
        let face = |f: (CompareFunction, StencilOperation, StencilOperation, StencilOperation)| StencilFaceState {
            compare: f.0,
            fail_op: f.1,
            depth_fail_op: f.2,
            pass_op: f.3,
        };
        let depth_stencil = key.depth.as_ref().map(|d| DepthStencilState {
            format: d.format,
            depth_write_enabled: d.write,
            depth_compare: d.compare,
            stencil_front: face(d.stencil[0]),
            stencil_back: face(d.stencil[1]),
            stencil_read_mask: d.read_mask,
            stencil_write_mask: d.write_mask,
            depth_bias: d.bias,
            depth_bias_slope_scale: f32::from_bits(d.slope_scale_bits),
            depth_bias_clamp: 0.0,
        });
        let fragment = ps.map(|ps| FragmentState {
            stage: ProgrammableStage::new(&ps.module, ps.translation.entry_point),
            targets: &color_targets,
        });
        let desc = RenderPipelineDescriptor {
            label: "xenos",
            layout: Some(layout),
            vertex: ProgrammableStage::new(&vs.module, vs.translation.entry_point),
            vertex_buffers: &[],
            primitive: PrimitiveState {
                topology: key.topology,
                strip_index_format: key.strip_index_format,
                front_face: key.front_face,
                cull_mode: key.cull,
                unclipped_depth: false,
            },
            depth_stencil,
            multisample: MultisampleState {
                count: key.sample_count,
                mask: key.sample_mask,
                alpha_to_coverage_enabled: key.alpha_to_coverage,
            },
            fragment,
        };
        self.stats.pipelines += 1;
        // Natively: created synchronously and checked with an error scope (wgpu-native has no
        // async creation, and aborts a submit that uses an invalid pipeline).
        #[cfg(not(target_os = "emscripten"))]
        let result = {
            let started = std::time::Instant::now();
            self.device.push_error_scope(ErrorFilter::Validation);
            let pipeline = self.device.create_render_pipeline(&desc);
            let result = match self.device.block_on(self.device.pop_error_scope()) {
                Ok(None) => Ok(pipeline),
                Ok(Some(e)) => Err(gpu_webgpu::Error::Invalid(e.to_string())),
                Err(e) => Err(e),
            };
            self.pipeline_time += started.elapsed();
            result
        };
        // On the web: compiled in the background (a synchronous pipeline would be compiled on
        // first use, stalling the GPU process); validation errors come back as a failed result.
        #[cfg(target_os = "emscripten")]
        let pending = self.device.create_render_pipeline_async(&desc);
        #[cfg(target_os = "emscripten")]
        let result = match pending.try_take() {
            Some(r) => r,
            None => {
                // Completes once the worker returns to the browser; draws wait until then.
                self.pipelines.insert(key, PipelineEntry::Compiling(pending));
                return None;
            }
        };
        self.compiled(key, result)
    }

    fn compiled(&mut self, key: PipelineKey, result: gpu_webgpu::Result<RenderPipeline>) -> Option<RenderPipeline> {
        match result {
            Ok(pipeline) => {
                self.pipelines.insert(key, PipelineEntry::Ready(pipeline.clone()));
                Some(pipeline)
            }
            Err(e) => {
                self.warn_once(format!("pipeline rejected: {e}"));
                self.pipelines.insert(key, PipelineEntry::Failed);
                None
            }
        }
    }
}

#[derive(Clone)]
enum BindingKind {
    View(TextureView),
    Sampler(Sampler),
}

fn push_texture(
    layout: &mut Vec<(u32, SlotKind)>,
    views: &mut Vec<(u32, BindingKind)>,
    binding: &TextureBinding,
    bound: xenos_texture_cache::BoundTexture,
) {
    let dimension = match binding.dimension {
        TextureBindingDimension::D2 => TextureViewDimension::D2,
        TextureBindingDimension::D3OrStacked => TextureViewDimension::D3,
        TextureBindingDimension::Cube => TextureViewDimension::Cube,
    };
    let filterable = bound.filterable;
    layout.push((binding.texture_binding, SlotKind::Texture { dimension, filterable }));
    views.push((binding.texture_binding, BindingKind::View(bound.view)));
    if let (Some(stacked), Some(view)) = (binding.stacked_texture_binding, bound.stacked_view) {
        layout.push((stacked, SlotKind::Texture { dimension: TextureViewDimension::D2Array, filterable }));
        views.push((stacked, BindingKind::View(view)));
    }
    layout.push((binding.sampler_binding, SlotKind::Sampler { filtering: filterable }));
    views.push((binding.sampler_binding, BindingKind::Sampler(bound.sampler)));
}
