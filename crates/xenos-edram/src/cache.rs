use std::cell::RefCell;
use std::collections::HashMap;

use gpu_webgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsage, ColorTargetState, CommandEncoder, CompareFunction,
    ComputePipeline, ComputePipelineDescriptor, DepthStencilState, Device, FeatureName, FragmentState, LoadOp, MapMode,
    MultisampleState, Pending, PipelineLayout, PrimitiveState, ProgrammableStage, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderStage,
    StencilFaceState, StencilOperation, StoreOp, Texture, TextureAspect, TextureDescriptor, TextureFormat,
    TextureSampleType, TextureUsage, TextureView, TextureViewDescriptor, TextureViewDimension, extent,
};
use xenos_state::edram::{
    EDRAM_TILE_COUNT, EDRAM_TILE_HEIGHT_SAMPLES, EDRAM_TILE_WIDTH_SAMPLES, RenderTargetKey, RenderTargetSetup,
};
use xenos_state::resolve::{ResolveCopyKind, ResolveEdramInfo, ResolveInfo};
use xenos_state::xenos::{ColorRenderTargetFormat, DepthRenderTargetFormat, MsaaSamples, TEXTURE_2D_MAX_WIDTH_HEIGHT};
use xenos_texture::Endian128;

use crate::dest::DestLayout;
use crate::format::{
    ColorTargetInfo, DEPTH_FORMAT_CODE_BASE, DepthTargetInfo, EdramOptions, host_color_format, host_depth_format,
};
use crate::ownership::{OwnedRange, OwnershipMap};
use crate::shaders::{self, STORE_DST_BUFFERS, SrcKind};
use crate::{EdramError, GuestMemoryAccess};

const TILE_DWORDS: u32 = EDRAM_TILE_WIDTH_SAMPLES * EDRAM_TILE_HEIGHT_SAMPLES;
const EDRAM_BYTES: u64 = (EDRAM_TILE_COUNT * TILE_DWORDS * 4) as u64;
/// Largest resolve output staged on the GPU (a 8192x8192 64bpp copy).
const MAX_RESOLVE_OUTPUT_BYTES: u64 = 512 << 20;
const READBACK_POOL_LIMIT: usize = 4;

/// A host texture standing in for an EDRAM range.
struct HostRt {
    key: RenderTargetKey,
    texture: Texture,
    attachment: TextureView,
    /// Color view, or the depth aspect for depth targets.
    read: TextureView,
    stencil: Option<TextureView>,
    format: TextureFormat,
    samples: u32,
    width: u32,
    height: u32,
    /// Frame ends since it was last bound or cleared, while owning no EDRAM tile.
    idle_frames: u32,
}

/// Frames a host render target may go unused while owning no EDRAM tile before
/// [`EdramCache::end_frame`] destroys it. Most targets are used every frame but lose their tiles
/// to later passes before it ends, so ownership alone would recreate them over and over.
const RT_MAX_IDLE_FRAMES: u32 = 60;

/// Render target heights are multiples of this many tile rows (see [`EdramCache::ensure_rt`]).
const RT_HEIGHT_GRANULARITY_TILES: u32 = 8;

impl HostRt {
    fn src_kind(&self) -> SrcKind {
        let multisampled = self.samples > 1;
        if self.key.is_depth { SrcKind::Depth { multisampled } } else { SrcKind::Color { multisampled } }
    }
}

fn format_code(key: &RenderTargetKey) -> u32 {
    if key.is_depth { DEPTH_FORMAT_CODE_BASE + key.format } else { key.format }
}

fn key_flags(key: &RenderTargetKey) -> u32 {
    key.is_depth as u32 | ((key.is_64bpp() as u32) << 1)
}

fn host_samples(msaa: MsaaSamples) -> u32 {
    if msaa == MsaaSamples::X1 { 1 } else { 4 }
}

/// Host attachments for a draw. Unbound slots are `None`; a slot can also be `None` when its EDRAM
/// base coincides with another bound target (the later color target, or depth aliasing a color
/// target, is dropped like Xenia does) — the pipeline must then leave that output unbound.
#[derive(Clone, Debug, Default)]
pub struct DrawTargets {
    pub colors: [Option<(TextureView, TextureFormat)>; 4],
    pub depth: Option<(TextureView, TextureFormat)>,
    /// Attachment size (all attachments of a draw share it). Viewport/scissor are in guest pixels.
    pub width: u32,
    pub height: u32,
    /// Host sample count: 1, or 4 for both 2x and 4x guest MSAA.
    pub sample_count: u32,
    /// Pipeline sample mask: 2x guest MSAA lives in host samples 0 and 3 (`0b1001`).
    pub sample_mask: u32,
    pub color_info: [Option<ColorTargetInfo>; 4],
    pub depth_info: Option<DepthTargetInfo>,
}

impl DrawTargets {
    pub fn is_empty(&self) -> bool {
        self.depth.is_none() && self.colors.iter().all(Option::is_none)
    }
}

/// Outcome of a resolve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveResult {
    /// Guest physical range `(start, length)` rewritten by the copy (invalidate textures there).
    pub written: Option<(u32, u32)>,
    pub cleared_depth: bool,
    pub cleared_color: bool,
}

struct PendingCopy {
    readback: Buffer,
    size: u64,
    map: Option<Pending<gpu_webgpu::Result<()>>>,
    map_result: Option<gpu_webgpu::Result<()>>,
    layout: DestLayout,
    width: u32,
    height: u32,
    words_per_pixel: usize,
}

/// A GPU copy of guest physical memory: blocks of `block_bytes` in slots, `chunk_bytes` of slots
/// per buffer of `chunks` (at most four; both sizes powers of two), located through `table` (per
/// block of physical memory, a u32 slot + 1, or 0 when the block isn't mirrored). Resolves into
/// it complete without a readback; guest memory itself is not updated. The destination's blocks
/// must be mirrored (writes elsewhere are dropped).
pub struct SharedMemoryTarget<'a> {
    pub chunks: &'a [Buffer],
    pub table: &'a Buffer,
    pub block_bytes: u32,
    pub chunk_bytes: u32,
}

enum CopyOutput {
    Readback(PendingCopy),
    /// Written straight into a [`SharedMemoryTarget`].
    Stored {
        written: (u32, u32),
    },
}

/// A resolve whose copy waits for its GPU readback; finish it with [`EdramCache::poll_resolve`].
pub struct PendingResolve {
    copy: Option<PendingCopy>,
    cleared_depth: bool,
    cleared_color: bool,
    result: Option<Result<ResolveResult, EdramError>>,
}

impl PendingResolve {
    /// Whether the guest memory has been written (or nothing had to be).
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TransferOutput {
    Color,
    Depth,
    StencilBit(u32),
}

#[derive(Default)]
struct Pipelines {
    transfer_layouts: HashMap<SrcKind, (BindGroupLayout, PipelineLayout)>,
    transfer_modules: HashMap<(SrcKind, bool), ShaderModule>,
    transfers: HashMap<(SrcKind, TextureFormat, u32, TransferOutput), RenderPipeline>,
    dumps: HashMap<SrcKind, (BindGroupLayout, ComputePipeline)>,
    clear: Option<(BindGroupLayout, PipelineLayout, ShaderModule)>,
    clears: HashMap<(TextureFormat, u32, bool), RenderPipeline>,
    resolve: Option<(BindGroupLayout, ComputePipeline)>,
    store: [Option<(BindGroupLayout, ComputePipeline)>; 2],
}

fn uniform_entry(visibility: ShaderStage) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: 0,
        visibility,
        // The operation's slot in the parameter arena is chosen with a dynamic offset, so bind
        // groups only depend on longer-lived resources and can be reused.
        ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: 0 },
    }
}

fn src_layout_entries(kind: SrcKind, visibility: ShaderStage) -> Vec<BindGroupLayoutEntry> {
    let multisampled = kind.multisampled();
    let texture = |binding, sample_type| BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Texture { sample_type, view_dimension: TextureViewDimension::D2, multisampled },
    };
    let mut entries = vec![uniform_entry(visibility)];
    match kind {
        SrcKind::Color { .. } => entries.push(texture(1, TextureSampleType::UnfilterableFloat)),
        SrcKind::Depth { .. } => {
            entries.push(texture(1, TextureSampleType::Depth));
            entries.push(texture(2, TextureSampleType::Uint));
        }
    }
    entries
}

fn stencil_replace() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Replace,
    }
}

fn depth_state(format: TextureFormat, write_depth: bool, stencil_write_mask: u32) -> DepthStencilState {
    DepthStencilState {
        stencil_front: stencil_replace(),
        stencil_back: stencil_replace(),
        stencil_read_mask: 0xFF,
        stencil_write_mask,
        ..DepthStencilState::depth(format, write_depth, CompareFunction::Always)
    }
}

impl Pipelines {
    fn transfer_layout(&mut self, device: &Device, kind: SrcKind) -> &(BindGroupLayout, PipelineLayout) {
        self.transfer_layouts.entry(kind).or_insert_with(|| {
            let bgl =
                device.create_bind_group_layout("edram transfer", &src_layout_entries(kind, ShaderStage::Fragment));
            let pl = device.create_pipeline_layout("edram transfer", &[&bgl]);
            (bgl, pl)
        })
    }

    fn transfer(
        &mut self,
        device: &Device,
        kind: SrcKind,
        format: TextureFormat,
        samples: u32,
        output: TransferOutput,
    ) -> RenderPipeline {
        if let Some(p) = self.transfers.get(&(kind, format, samples, output)) {
            return p.clone();
        }
        let layout = self.transfer_layout(device, kind).1.clone();
        let module = self
            .transfer_modules
            .entry((kind, samples > 1))
            .or_insert_with(|| {
                device.create_shader_module_wgsl("edram transfer", &shaders::transfer_module(kind, samples > 1))
            })
            .clone();
        let color_targets = [ColorTargetState::new(format)];
        let (entry, targets, depth_stencil): (&str, &[ColorTargetState], _) = match output {
            TransferOutput::Color => ("fs_color", &color_targets, None),
            TransferOutput::Depth => ("fs_depth", &[], Some(depth_state(format, true, 0xFF))),
            TransferOutput::StencilBit(bit) => ("fs_stencil", &[], Some(depth_state(format, false, 1 << bit))),
        };
        let bit = match output {
            TransferOutput::StencilBit(bit) => bit,
            _ => 0,
        };
        let constants = [("STENCIL_BIT", bit as f64)];
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "edram transfer",
            layout: Some(&layout),
            vertex: ProgrammableStage::new(&module, "vs"),
            vertex_buffers: &[],
            primitive: PrimitiveState::default(),
            depth_stencil,
            multisample: MultisampleState { count: samples, ..Default::default() },
            fragment: Some(FragmentState {
                stage: ProgrammableStage {
                    module: &module,
                    entry_point: entry,
                    constants: if entry == "fs_stencil" { &constants } else { &[] },
                },
                targets,
            }),
        });
        self.transfers.insert((kind, format, samples, output), pipeline.clone());
        pipeline
    }

    fn dump(&mut self, device: &Device, kind: SrcKind) -> &(BindGroupLayout, ComputePipeline) {
        self.dumps.entry(kind).or_insert_with(|| {
            let mut entries = src_layout_entries(kind, ShaderStage::Compute);
            entries.push(BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStage::Compute,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage,
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                },
            });
            let bgl = device.create_bind_group_layout("edram dump", &entries);
            let pl = device.create_pipeline_layout("edram dump", &[&bgl]);
            let module = device.create_shader_module_wgsl("edram dump", &shaders::dump_module(kind));
            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: "edram dump",
                layout: Some(&pl),
                stage: ProgrammableStage::new(&module, "main"),
            });
            (bgl, pipeline)
        })
    }

    fn clear(&mut self, device: &Device, format: TextureFormat, samples: u32, depth: bool) -> RenderPipeline {
        if let Some(p) = self.clears.get(&(format, samples, depth)) {
            return p.clone();
        }
        let (_, layout, module) = self.clear.get_or_insert_with(|| {
            let bgl = device.create_bind_group_layout("edram clear", &[uniform_entry(ShaderStage::Fragment)]);
            let pl = device.create_pipeline_layout("edram clear", &[&bgl]);
            let module = device.create_shader_module_wgsl("edram clear", &shaders::clear_module());
            (bgl, pl, module)
        });
        let color_targets = [ColorTargetState::new(format)];
        let (entry, targets, depth_stencil): (&str, &[ColorTargetState], _) = if depth {
            ("fs_depth", &[], Some(depth_state(format, true, 0xFF)))
        } else {
            ("fs_color", &color_targets, None)
        };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "edram clear",
            layout: Some(layout),
            vertex: ProgrammableStage::new(module, "vs"),
            vertex_buffers: &[],
            primitive: PrimitiveState::default(),
            depth_stencil,
            multisample: MultisampleState { count: samples, ..Default::default() },
            fragment: Some(FragmentState { stage: ProgrammableStage::new(module, entry), targets }),
        });
        self.clears.insert((format, samples, depth), pipeline.clone());
        pipeline
    }

    fn clear_layout(&self) -> &BindGroupLayout {
        &self.clear.as_ref().expect("clear pipeline created first").0
    }

    /// The store shader; `atomic` for 8/16-bit elements.
    fn store(&mut self, device: &Device, atomic: bool) -> &(BindGroupLayout, ComputePipeline) {
        self.store[atomic as usize].get_or_insert_with(|| {
            let storage = |binding, ty| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStage::Compute,
                ty: BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: 0 },
            };
            let bgl = device.create_bind_group_layout(
                "edram store",
                &[
                    uniform_entry(ShaderStage::Compute),
                    storage(1, BufferBindingType::ReadOnlyStorage),
                    storage(2, BufferBindingType::ReadOnlyStorage),
                    storage(3, BufferBindingType::Storage),
                    storage(4, BufferBindingType::Storage),
                    storage(5, BufferBindingType::Storage),
                    storage(6, BufferBindingType::Storage),
                ],
            );
            let pl = device.create_pipeline_layout("edram store", &[&bgl]);
            let module = device.create_shader_module_wgsl("edram store", &shaders::store_module(atomic));
            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: "edram store",
                layout: Some(&pl),
                stage: ProgrammableStage::new(&module, "main"),
            });
            (bgl, pipeline)
        })
    }

    fn resolve(&mut self, device: &Device) -> &(BindGroupLayout, ComputePipeline) {
        self.resolve.get_or_insert_with(|| {
            let storage = |binding, ty| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStage::Compute,
                ty: BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: 0 },
            };
            let bgl = device.create_bind_group_layout(
                "edram resolve",
                &[
                    uniform_entry(ShaderStage::Compute),
                    storage(1, BufferBindingType::ReadOnlyStorage),
                    storage(2, BufferBindingType::Storage),
                ],
            );
            let pl = device.create_pipeline_layout("edram resolve", &[&bgl]);
            let module = device.create_shader_module_wgsl("edram resolve", &shaders::resolve_module());
            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: "edram resolve",
                layout: Some(&pl),
                stage: ProgrammableStage::new(&module, "main"),
            });
            (bgl, pipeline)
        })
    }
}

/// Bytes of the uniform arena; slots are reused after it wraps (thousands of operations later).
const PARAMS_ARENA_BYTES: u64 = 4 << 20;
/// Uniform binding offsets must be multiples of 256.
const PARAMS_SLOT: u64 = 256;

/// Shader parameters of the EDRAM operations, written into slots of one long-lived buffer
/// (`Queue::write_buffer` is ordered after earlier submissions). Creating a buffer per operation
/// costs a fraction of a millisecond in browsers, and a frame does dozens.
struct ParamsArena {
    buffer: Buffer,
    next: u64,
}

/// One operation's parameters in the arena.
#[derive(Clone)]
struct Params {
    buffer: Buffer,
    offset: u64,
}

impl Params {
    /// The binding: one slot's worth at offset 0; `dynamic_offset` selects the slot.
    fn binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer { buffer: &self.buffer, offset: 0, size: Some(PARAMS_SLOT) }
    }

    fn dynamic_offset(&self) -> [u32; 1] {
        [self.offset as u32]
    }
}

/// A resource a cached bind group refers to, held so its handle (part of the key) can't be
/// reused by another object while the entry exists (never read otherwise).
#[allow(dead_code)]
enum Held {
    Buffer(Buffer),
    View(TextureView),
    Sampler(gpu_webgpu::Sampler),
}

/// Bind groups by layout and resources (see `cached_bind_group`).
type GroupCache = HashMap<Vec<u64>, (BindGroup, Vec<Held>)>;

/// Bind groups cached at most (the cache is cleared when full).
const MAX_CACHED_GROUPS: usize = 2048;

fn uniform(arena: &RefCell<ParamsArena>, device: &Device, words: &[u32]) -> Params {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(bytes.len().next_multiple_of(16).max(16), 0);
    let _ = PARAMS_SLOT.checked_sub(bytes.len() as u64).expect("EDRAM parameters larger than a slot");
    let mut arena = arena.borrow_mut();
    let offset = arena.next;
    arena.next = (offset + PARAMS_SLOT) % PARAMS_ARENA_BYTES;
    device.queue().write_buffer(&arena.buffer, offset, &bytes);
    Params { buffer: arena.buffer.clone(), offset }
}

fn whole(buffer: &Buffer) -> BindingResource<'_> {
    BindingResource::Buffer { buffer, offset: 0, size: None }
}

/// Bind group entries reading a host render target (bindings 1 and 2).
fn src_entries(rt: &HostRt) -> Vec<BindGroupEntry<'_>> {
    let mut entries = vec![BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&rt.read) }];
    if let Some(stencil) = &rt.stencil {
        entries.push(BindGroupEntry { binding: 2, resource: BindingResource::TextureView(stencil) });
    }
    entries
}

/// Pixel rectangle `[x0, y0, x1, y1)` excluded from transfers (it's about to be cleared).
type Cutout = [u32; 4];

/// Host texture formats, EDRAM ownership and resolves; see the crate documentation.
pub struct EdramCache {
    depth32_stencil8: bool,
    options: EdramOptions,
    max_dimension: u32,
    max_output_bytes: u64,
    rts: HashMap<RenderTargetKey, HostRt>,
    /// Textures of render targets replaced by taller ones, destroyed at the end of the frame
    /// (recorded work may still use them until it is submitted).
    outgrown: Vec<Texture>,
    ownership: OwnershipMap,
    /// EDRAM-layout copy of the tiles dumped for resolves.
    shadow: Buffer,
    resolve_out: Option<Buffer>,
    readback_pool: Vec<Buffer>,
    pipelines: Pipelines,
    params: RefCell<ParamsArena>,
    groups: RefCell<GroupCache>,
}

/// A bind group for `entries` under `layout`: a cached one with the same resources, or a new one.
/// Creating bind groups is expensive in browsers, and EDRAM operations repeat theirs.
fn cached_bind_group(
    cache: &RefCell<GroupCache>,
    device: &Device,
    label: &str,
    layout: &BindGroupLayout,
    entries: &[BindGroupEntry<'_>],
) -> BindGroup {
    let mut key = Vec::with_capacity(1 + entries.len() * 4);
    key.push(layout.raw() as usize as u64);
    for e in entries {
        key.push(u64::from(e.binding));
        match &e.resource {
            BindingResource::Buffer { buffer, offset, size } => {
                key.extend([buffer.raw() as usize as u64, *offset, size.unwrap_or(u64::MAX)]);
            }
            BindingResource::TextureView(v) => key.push(v.raw() as usize as u64),
            BindingResource::Sampler(s) => key.push(s.raw() as usize as u64),
        }
    }
    let mut cache = cache.borrow_mut();
    if let Some((group, _)) = cache.get(&key) {
        return group.clone();
    }
    if cache.len() >= MAX_CACHED_GROUPS {
        cache.clear();
    }
    let group = device.create_bind_group(label, layout, entries);
    let held = entries
        .iter()
        .map(|e| match &e.resource {
            BindingResource::Buffer { buffer, .. } => Held::Buffer((*buffer).clone()),
            BindingResource::TextureView(v) => Held::View((*v).clone()),
            BindingResource::Sampler(s) => Held::Sampler((*s).clone()),
        })
        .collect();
    cache.insert(key, (group.clone(), held));
    group
}

impl EdramCache {
    pub fn new(device: &Device) -> Self {
        Self::with_options(device, EdramOptions::default())
    }

    pub fn with_options(device: &Device, options: EdramOptions) -> Self {
        let limits = device.limits();
        let shadow = device.create_buffer(&BufferDescriptor {
            label: "edram shadow",
            size: EDRAM_BYTES,
            usage: BufferUsage::Storage | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        Self {
            depth32_stencil8: device.has_feature(FeatureName::Depth32FloatStencil8),
            options,
            max_dimension: limits.maxTextureDimension2D.min(TEXTURE_2D_MAX_WIDTH_HEIGHT),
            max_output_bytes: limits
                .maxStorageBufferBindingSize
                .min(limits.maxBufferSize)
                .min(MAX_RESOLVE_OUTPUT_BYTES),
            rts: HashMap::new(),
            outgrown: Vec::new(),
            ownership: OwnershipMap::default(),
            shadow,
            resolve_out: None,
            readback_pool: Vec::new(),
            pipelines: Pipelines::default(),
            params: RefCell::new(ParamsArena {
                buffer: device.create_buffer(&BufferDescriptor {
                    label: "edram params",
                    size: PARAMS_ARENA_BYTES,
                    usage: BufferUsage::Uniform | BufferUsage::CopyDst,
                    mapped_at_creation: false,
                }),
                next: 0,
            }),
            groups: RefCell::new(HashMap::new()),
        }
    }

    /// Current tile ownership (for diagnostics).
    pub fn ownership(&self) -> &[OwnedRange] {
        self.ownership.ranges()
    }

    /// Number of host render targets alive.
    pub fn render_target_count(&self) -> usize {
        self.rts.len()
    }

    /// Host format used for an EDRAM depth format on this device.
    pub fn depth_host_format(&self, format: DepthRenderTargetFormat) -> TextureFormat {
        host_depth_format(format, self.depth32_stencil8, self.options.d24s8_float_host)
    }

    /// Destroys host render targets that no longer own any EDRAM tile.
    pub fn trim(&mut self) {
        self.trim_unowned(0);
    }

    /// Frame boundary, with everything recorded submitted: destroys host render targets that
    /// own no EDRAM tile and were not used for [`RT_MAX_IDLE_FRAMES`] frames (their contents
    /// are unreachable; a later use starts a new one).
    pub fn end_frame(&mut self) {
        self.trim_unowned(RT_MAX_IDLE_FRAMES);
    }

    fn trim_unowned(&mut self, max_idle_frames: u32) {
        for texture in self.outgrown.drain(..) {
            texture.destroy();
        }
        let ranges = self.ownership.ranges();
        self.rts.retain(|key, rt| {
            if ranges.iter().any(|r| r.owner == *key) {
                rt.idle_frames = 0;
                return true;
            }
            rt.idle_frames += 1;
            let keep = rt.idle_frames <= max_idle_frames;
            if !keep {
                rt.texture.destroy();
            }
            keep
        });
    }

    fn rt_height(&self, pitch_tiles_at_32bpp: u32, msaa: MsaaSamples) -> u32 {
        let tile_h = EDRAM_TILE_HEIGHT_SAMPLES >> msaa.samples_y_log2();
        let rows = EDRAM_TILE_COUNT.div_ceil(pitch_tiles_at_32bpp.max(1));
        (rows * tile_h).min(self.max_dimension / tile_h * tile_h)
    }

    fn host_format(&self, key: &RenderTargetKey) -> Result<TextureFormat, EdramError> {
        if key.is_depth {
            Ok(host_depth_format(key.depth_format(), self.depth32_stencil8, self.options.d24s8_float_host))
        } else {
            host_color_format(key.color_format()).ok_or(EdramError::UnsupportedFormat(key.format))
        }
    }

    /// Height of a new or grown render target that must cover `min_height` rows: rounded up so
    /// small increases don't each cost a new texture, and at most [`EdramCache::rt_height`].
    fn grown_height(&self, key: &RenderTargetKey, min_height: u32) -> u32 {
        let tile_h = EDRAM_TILE_HEIGHT_SAMPLES >> key.msaa_samples.samples_y_log2();
        let max = self.rt_height(key.pitch_tiles_at_32bpp, key.msaa_samples);
        min_height.max(1).next_multiple_of(tile_h * RT_HEIGHT_GRANULARITY_TILES).min(max)
    }

    /// Makes sure the host render target for `key` exists and is at least `min_height` rows
    /// tall (clamped to what one EDRAM addressing period needs). Targets are made only as tall
    /// as their uses so far: a full-period texture is mostly never touched (a 720-row frame in
    /// a 2048-row texture; a 320x8192 depth buffer of which 256 rows are addressable before the
    /// end of EDRAM), and that cost ~200 MiB of GPU memory in Skyrim. A taller use replaces the
    /// texture, copying the tiles it owns over with the ownership-transfer passes encoded into
    /// `encoder` (partial copies of depth/stencil and multisampled textures aren't allowed).
    fn ensure_rt(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        key: RenderTargetKey,
        min_height: u32,
    ) -> Result<(), EdramError> {
        let height = self.grown_height(&key, min_height);
        if let Some(rt) = self.rts.get_mut(&key) {
            rt.idle_frames = 0;
            if rt.height >= height {
                return Ok(());
            }
        }
        if key.msaa_samples.count().is_none() {
            return Err(EdramError::InvalidMsaa(key.msaa_samples.raw()));
        }
        let format = self.host_format(&key)?;
        let width = key.width_pixels();
        if width == 0 || width > self.max_dimension {
            return Err(EdramError::PitchTooLarge { width, max: self.max_dimension });
        }
        let samples = host_samples(key.msaa_samples);
        let texture = device.create_texture(&TextureDescriptor {
            label: if key.is_depth { "edram depth" } else { "edram color" },
            size: extent(width, height, 1),
            sample_count: samples,
            format,
            usage: TextureUsage::RenderAttachment | TextureUsage::TextureBinding,
            ..Default::default()
        });
        let attachment = texture.create_view(None);
        let (read, stencil) = if key.is_depth {
            let view = |aspect| texture.create_view(Some(&TextureViewDescriptor { aspect, ..Default::default() }));
            (view(TextureAspect::DepthOnly), Some(view(TextureAspect::StencilOnly)))
        } else {
            (attachment.clone(), None)
        };
        let rt = HostRt { key, texture, attachment, read, stencil, format, samples, width, height, idle_frames: 0 };
        match self.rts.insert(key, rt) {
            None => log::debug!("edram: new {width}x{height}x{samples} {format:?} render target for {key:?}"),
            Some(old) => {
                log::debug!("edram: {key:?} grows from {} to {height} rows", old.height);
                let owned: Vec<OwnedRange> =
                    self.ownership.ranges().iter().filter(|r| r.owner == key).copied().collect();
                self.encode_transfers(device, encoder, key, &owned, None, Some(&old));
                self.outgrown.push(old.texture);
            }
        }
        Ok(())
    }

    /// Host attachments for the current draw's EDRAM setup, covering at least `min_height` rows.
    /// Ownership-transfer passes for tiles last written through other targets are encoded into
    /// `encoder`, which must be submitted before a later [`EdramCache::begin_resolve`] (that one
    /// submits its own work).
    pub fn targets(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        setup: &RenderTargetSetup,
        min_height: u32,
    ) -> Result<DrawTargets, EdramError> {
        let msaa = setup.msaa_samples;
        if msaa.count().is_none() {
            return Err(EdramError::InvalidMsaa(msaa.raw()));
        }
        // Color first so a depth buffer aliasing a color target's base is the one dropped.
        let mut bound: Vec<(Option<usize>, RenderTargetKey, i32, ColorRenderTargetFormat)> = Vec::new();
        for (i, binding) in setup.color.iter().enumerate() {
            if let Some(b) = binding {
                let guest_format = b.guest_color_format.unwrap_or(b.key.color_format());
                bound.push((Some(i), b.key, b.exp_bias, guest_format));
            }
        }
        if let Some(d) = &setup.depth {
            bound.push((None, d.key, 0, ColorRenderTargetFormat::K8888));
        }
        let mut seen = Vec::new();
        bound.retain(|(slot, key, ..)| {
            let base = key.base_tiles & (EDRAM_TILE_COUNT - 1);
            if seen.contains(&base) {
                log::debug!("edram: dropping {slot:?} aliasing EDRAM base {base}");
                return false;
            }
            seen.push(base);
            true
        });
        let mut out = DrawTargets { sample_count: 1, sample_mask: !0, ..Default::default() };
        if bound.is_empty() {
            return Ok(out);
        }
        let height = self.rt_height(setup.pitch_tiles_at_32bpp, msaa);
        let tile_h = EDRAM_TILE_HEIGHT_SAMPLES >> msaa.samples_y_log2();
        let rows_tiles = min_height.clamp(1, height).div_ceil(tile_h);
        // Attachments of one pass must be the same size: all grow to the tallest.
        let mut common = rows_tiles * tile_h;
        for (_, key, ..) in &bound {
            self.host_format(key)?;
            common = common.max(self.rts.get(key).map_or(0, |rt| rt.height));
        }
        for (_, key, ..) in &bound {
            self.ensure_rt(device, encoder, *key, common)?;
        }

        // Each target owns from its base up to the rows used, but not past the next target's base.
        let mut order: Vec<usize> = (0..bound.len()).collect();
        order.sort_by_key(|&i| bound[i].1.base_tiles & (EDRAM_TILE_COUNT - 1));
        let mut claims = Vec::new();
        for (n, &i) in order.iter().enumerate() {
            let key = bound[i].1;
            let base = key.base_tiles & (EDRAM_TILE_COUNT - 1);
            let next = if n + 1 < order.len() {
                bound[order[n + 1]].1.base_tiles & (EDRAM_TILE_COUNT - 1)
            } else {
                (bound[order[0]].1.base_tiles & (EDRAM_TILE_COUNT - 1)) + EDRAM_TILE_COUNT
            };
            let length = (rows_tiles * key.pitch_tiles()).min(next - base).min(EDRAM_TILE_COUNT);
            let previous = self.ownership.claim(base, length, key);
            if !previous.is_empty() {
                claims.push((key, previous));
            }
        }
        for (key, previous) in claims {
            self.encode_transfers(device, encoder, key, &previous, None, None);
        }

        let any = &self.rts[&bound[0].1];
        out.width = any.width;
        out.height = any.height;
        out.sample_count = any.samples;
        out.sample_mask = if msaa == MsaaSamples::X2 { 0b1001 } else { !0 };
        for (slot, key, exp_bias, guest_format) in bound {
            let rt = &self.rts[&key];
            match slot {
                Some(i) => {
                    out.colors[i] = Some((rt.attachment.clone(), rt.format));
                    out.color_info[i] = ColorTargetInfo::new(guest_format, exp_bias);
                }
                None => {
                    out.depth = Some((rt.attachment.clone(), rt.format));
                    out.depth_info = Some(DepthTargetInfo {
                        guest_format: key.depth_format(),
                        host_format: rt.format,
                        float24: key.depth_format() == DepthRenderTargetFormat::D24FS8,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Copies tiles `previous` (owned by other targets, or by `dest_key` itself in `outgrown`,
    /// the texture it replaces) into `dest`'s texture.
    fn encode_transfers(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        dest_key: RenderTargetKey,
        previous: &[OwnedRange],
        cutout: Option<Cutout>,
        outgrown: Option<&HostRt>,
    ) {
        struct Draw {
            pipelines: Vec<(RenderPipeline, u32)>,
            bind_group: BindGroup,
            params: Params,
            scissor: [u32; 4],
        }
        let dest = &self.rts[&dest_key];
        let (dest_format, dest_samples, dest_width, dest_height) = (dest.format, dest.samples, dest.width, dest.height);
        let pitch = dest_key.pitch_tiles().max(1);
        let tile_h = EDRAM_TILE_HEIGHT_SAMPLES >> dest_key.msaa_samples.samples_y_log2();
        let mut draws = Vec::new();
        for range in previous {
            // `dest_key`'s own tiles come from its previous texture (when it grows).
            let src = if range.owner == dest_key { outgrown } else { self.rts.get(&range.owner) };
            let Some(src) = src else {
                continue;
            };
            let kind = src.src_kind();
            let rel_start = range.start.wrapping_sub(dest_key.base_tiles) & (EDRAM_TILE_COUNT - 1);
            let rel_end = rel_start + (range.end - range.start);
            let (y0, y1) = if rel_end > EDRAM_TILE_COUNT {
                (0, dest_height)
            } else {
                ((rel_start / pitch) * tile_h, ((rel_end - 1) / pitch + 1) * tile_h)
            };
            let y1 = y1.min(dest_height);
            if y0 >= y1 {
                continue;
            }
            let cut = cutout.unwrap_or([0; 4]);
            let params = uniform(
                &self.params,
                device,
                &[
                    dest_key.base_tiles & (EDRAM_TILE_COUNT - 1),
                    dest_key.pitch_tiles(),
                    dest_key.msaa_samples.raw(),
                    key_flags(&dest_key),
                    format_code(&dest_key),
                    src.key.base_tiles & (EDRAM_TILE_COUNT - 1),
                    src.key.pitch_tiles(),
                    src.key.msaa_samples.raw(),
                    key_flags(&src.key),
                    format_code(&src.key),
                    src.width,
                    src.height,
                    range.start,
                    range.end - range.start,
                    0,
                    0,
                    cut[0],
                    cut[1],
                    cut[2],
                    cut[3],
                ],
            );
            let outputs: Vec<TransferOutput> = if dest_key.is_depth {
                std::iter::once(TransferOutput::Depth).chain((0..8).map(TransferOutput::StencilBit)).collect()
            } else {
                vec![TransferOutput::Color]
            };
            let mut pipelines = Vec::new();
            for output in outputs {
                let pipeline = self.pipelines.transfer(device, kind, dest_format, dest_samples, output);
                let reference = if matches!(output, TransferOutput::StencilBit(_)) { 0xFF } else { 0 };
                pipelines.push((pipeline, reference));
            }
            let layout = &self.pipelines.transfer_layout(device, kind).0;
            let mut entries = vec![BindGroupEntry { binding: 0, resource: params.binding() }];
            entries.extend(src_entries(src));
            let bind_group = cached_bind_group(&self.groups, device, "edram transfer", layout, &entries);
            draws.push(Draw { pipelines, bind_group, params, scissor: [0, y0, dest_width, y1 - y0] });
        }
        if draws.is_empty() {
            return;
        }
        let dest = &self.rts[&dest_key];
        let color;
        let depth;
        let (colors, depth_attachment): (&[Option<RenderPassColorAttachment<'_>>], _) = if dest_key.is_depth {
            depth = RenderPassDepthStencilAttachment {
                view: &dest.attachment,
                depth_load_op: LoadOp::Load,
                depth_store_op: StoreOp::Store,
                depth_clear_value: 0.0,
                depth_read_only: false,
                stencil_load_op: LoadOp::Load,
                stencil_store_op: StoreOp::Store,
                stencil_clear_value: 0,
                stencil_read_only: false,
            };
            (&[], Some(&depth))
        } else {
            color = [Some(RenderPassColorAttachment::load(&dest.attachment))];
            (&color, None)
        };
        let mut pass = encoder.begin_render_pass("edram transfer", colors, depth_attachment);
        for draw in &draws {
            let [x, y, w, h] = draw.scissor;
            pass.set_scissor_rect(x, y, w, h);
            pass.set_bind_group(0, &draw.bind_group, &draw.params.dynamic_offset());
            for (pipeline, reference) in &draw.pipelines {
                pass.set_pipeline(pipeline);
                pass.set_stencil_reference(*reference);
                pass.draw(3, 1, 0, 0);
            }
        }
        pass.end();
    }

    /// Starts a resolve: copies the source rectangle out of EDRAM (converted to the destination
    /// format on the GPU, staged for readback) and performs the requested depth/color clears.
    /// Submits its own command buffer, so any earlier rendering into EDRAM targets must already be
    /// submitted. Nothing is done when an error is returned.
    pub fn begin_resolve(&mut self, device: &Device, info: &ResolveInfo) -> Result<PendingResolve, EdramError> {
        self.begin_resolve_to(device, info, None)
    }

    /// [`EdramCache::begin_resolve`], storing the copy into `shared` (a GPU copy of guest memory)
    /// when the destination lies within one of its buffers: the resolve is then complete when
    /// this returns (its result says what was written), and guest memory is left unchanged.
    /// Destinations spanning two buffers fall back to the readback.
    pub fn begin_resolve_to(
        &mut self,
        device: &Device,
        info: &ResolveInfo,
        shared: Option<&SharedMemoryTarget<'_>>,
    ) -> Result<PendingResolve, EdramError> {
        if let Some(reason) = device.lost_reason() {
            return Err(EdramError::DeviceLost(reason));
        }
        let mut pending = PendingResolve { copy: None, cleared_depth: false, cleared_color: false, result: None };
        if info.rect.is_empty() {
            pending.result = Some(Ok(ResolveResult { written: None, cleared_depth: false, cleared_color: false }));
            return Ok(pending);
        }
        let copy = self.plan_copy(info)?;
        let clears = self.plan_clears(info)?;
        let mut encoder = device.create_command_encoder("edram resolve");
        for clear in &clears {
            let tile_h = EDRAM_TILE_HEIGHT_SAMPLES >> clear.key.msaa_samples.samples_y_log2();
            self.ensure_rt(device, &mut encoder, clear.key, clear.rect[3].next_multiple_of(tile_h))?;
        }

        let mut stored = None;
        if let Some(plan) = copy {
            match self.encode_copy(device, &mut encoder, info, plan, shared) {
                CopyOutput::Readback(copy) => pending.copy = Some(copy),
                CopyOutput::Stored { written } => stored = Some(written),
            }
        }
        for clear in &clears {
            let previous = self.ownership.claim(clear.start, clear.length, clear.key);
            self.encode_transfers(device, &mut encoder, clear.key, &previous, Some(clear.rect), None);
            self.encode_clear(device, &mut encoder, clear);
            if clear.key.is_depth {
                pending.cleared_depth = true;
            } else {
                pending.cleared_color = true;
            }
        }
        device.queue().submit([encoder.finish("edram resolve")]);
        match &mut pending.copy {
            Some(copy) => copy.map = Some(copy.readback.map_async(MapMode::Read, 0, Some(copy.size))),
            None => {
                pending.result = Some(Ok(ResolveResult {
                    written: stored,
                    cleared_depth: pending.cleared_depth,
                    cleared_color: pending.cleared_color,
                }))
            }
        }
        Ok(pending)
    }

    /// Finishes a resolve once its readback is mapped: converts to the destination layout, writes
    /// guest memory and returns the outcome. `None` while the GPU is still busy (on the web, return
    /// to the browser event loop between polls). Repeated calls after completion return the same
    /// outcome.
    pub fn poll_resolve(
        &mut self,
        device: &Device,
        pending: &mut PendingResolve,
        memory: &dyn GuestMemoryAccess,
    ) -> Option<Result<ResolveResult, EdramError>> {
        if let Some(result) = &pending.result {
            return Some(result.clone());
        }
        let copy = pending.copy.as_mut()?;
        if copy.map_result.is_none() {
            device.poll();
            copy.map_result = copy.map.as_ref().and_then(Pending::try_take);
        }
        let map_result = copy.map_result.take()?;
        let copy = pending.copy.take().expect("checked above");
        let result = match map_result {
            Err(e) => Err(EdramError::Readback(e.to_string())),
            Ok(()) => {
                let mut bytes = vec![0u8; copy.size as usize];
                let read = copy.readback.read_mapped(0, &mut bytes);
                copy.readback.unmap();
                if self.readback_pool.len() < READBACK_POOL_LIMIT {
                    self.readback_pool.push(copy.readback.clone());
                }
                match read {
                    Err(e) => Err(EdramError::Readback(e.to_string())),
                    Ok(()) => {
                        let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                        let dropped = copy.layout.write(&words, copy.words_per_pixel, copy.width, copy.height, memory);
                        if dropped != 0 {
                            log::warn!("edram: {dropped} resolved elements fell outside the destination extent");
                        }
                        Ok(ResolveResult {
                            written: Some((copy.layout.extent_start, copy.layout.extent_len)),
                            cleared_depth: pending.cleared_depth,
                            cleared_color: pending.cleared_color,
                        })
                    }
                }
            }
        };
        pending.result = Some(result.clone());
        Some(result)
    }

    /// Native only: waits for the readback (blocking the thread) and finishes the resolve.
    #[cfg(not(target_os = "emscripten"))]
    pub fn wait_resolve(
        &mut self,
        device: &Device,
        pending: &mut PendingResolve,
        memory: &dyn GuestMemoryAccess,
    ) -> Result<ResolveResult, EdramError> {
        if let Some(copy) = pending.copy.as_mut()
            && copy.map_result.is_none()
            && let Some(map) = copy.map.take()
        {
            copy.map_result = Some(device.block_on(map));
        }
        self.poll_resolve(device, pending, memory)
            .unwrap_or_else(|| Err(EdramError::Readback("readback never completed".into())))
    }
}

struct CopyPlan {
    kind: ResolveCopyKind,
    source: ResolveEdramInfo,
    words_per_pixel: u32,
    dest_bpp: u32,
    mode: u32,
    out_bytes: u64,
}

struct ClearPlan {
    key: RenderTargetKey,
    /// Absolute first tile and tile count claimed by the clear.
    start: u32,
    length: u32,
    rect: Cutout,
    value: [u32; 2],
}

impl EdramCache {
    fn plan_copy(&self, info: &ResolveInfo) -> Result<Option<CopyPlan>, EdramError> {
        if info.copy_dest_extent_length == 0 {
            // Unusable destination: Xenia drops the copy and keeps the clears.
            return Ok(None);
        }
        let Some(&source) = info.source_edram() else {
            return Ok(None);
        };
        if source.msaa_samples.count().is_none() {
            return Err(EdramError::InvalidMsaa(source.msaa_samples.raw()));
        }
        let (kind, _) = info.copy_kind(1, 1).ok_or(EdramError::UnsupportedCopy(info.dest_format))?;
        use ResolveCopyKind as K;
        let (mode, words_per_pixel, dest_bpp) = match kind {
            K::Fast32bpp1x2xMsaa | K::Fast32bpp4xMsaa => (0, 1, 32),
            K::Fast64bpp1x2xMsaa | K::Fast64bpp4xMsaa => (1, 2, 64),
            K::Full8bpp => (2, 1, 8),
            K::Full16bpp => (2, 1, 16),
            K::Full32bpp => (2, 1, 32),
            K::Full64bpp => (2, 2, 64),
            K::Full128bpp => (2, 4, 128),
            K::Dxt3aAs1111 => (3, 1, 64),
        };
        let out_bytes = info.rect.width as u64 * info.rect.height as u64 * words_per_pixel as u64 * 4;
        if out_bytes > self.max_output_bytes {
            return Err(EdramError::ResolveTooLarge { bytes: out_bytes, max: self.max_output_bytes });
        }
        Ok(Some(CopyPlan { kind, source, words_per_pixel, dest_bpp, mode, out_bytes }))
    }

    /// Xenia's PrepareHostRenderTargetsResolveClear: the clear rectangle in render target pixels and
    /// the tiles depth and color clears claim (kept from overlapping each other).
    fn plan_clears(&self, info: &ResolveInfo) -> Result<Vec<ClearPlan>, EdramError> {
        let depth = info.depth_edram.filter(|_| info.is_clearing_depth());
        let color = info.color_edram.filter(|_| info.is_clearing_color());
        let (pitch32, offset32, msaa) = if let Some(d) = depth {
            (d.pitch_tiles, d.base_tiles.wrapping_sub(info.depth_original_base), d.msaa_samples)
        } else if let Some(c) = color {
            let s = c.format_is_64bpp as u32;
            (c.pitch_tiles >> s, c.base_tiles.wrapping_sub(info.color_original_base) >> s, c.msaa_samples)
        } else {
            return Ok(Vec::new());
        };
        if msaa.count().is_none() {
            return Err(EdramError::InvalidMsaa(msaa.raw()));
        }
        if pitch32 == 0 {
            return Ok(Vec::new());
        }
        let (sx, sy) = (msaa.samples_x_log2(), msaa.samples_y_log2());
        let pitch_pixels = pitch32 * (EDRAM_TILE_WIDTH_SAMPLES >> sx);
        if pitch_pixels > self.max_dimension {
            return Err(EdramError::PitchTooLarge { width: pitch_pixels, max: self.max_dimension });
        }
        let height = self.rt_height(pitch32, msaa);
        let offset_rows = offset32 / pitch32;
        let x = ((offset32 - offset_rows * pitch32) * (EDRAM_TILE_WIDTH_SAMPLES >> sx)
            + (info.edram_offset_x_div_8 << 3))
            .min(pitch_pixels);
        let y = (offset_rows * (EDRAM_TILE_HEIGHT_SAMPLES >> sy) + (info.edram_offset_y_div_8 << 3)).min(height);
        let w = info.rect.width.min(pitch_pixels - x);
        let h = info.rect.height.min(height - y);
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        let start32 = ((y << sy) / EDRAM_TILE_HEIGHT_SAMPLES) * pitch32 + (x << sx) / EDRAM_TILE_WIDTH_SAMPLES;
        let length32 = (((y + h - 1) << sy) / EDRAM_TILE_HEIGHT_SAMPLES) * pitch32
            + ((x + w - 1) << sx) / EDRAM_TILE_WIDTH_SAMPLES
            + 1
            - start32;
        let span = |is_64: bool| {
            let s = is_64 as u32;
            let start = (start32 << s).min(EDRAM_TILE_COUNT);
            (start, ((start32 + length32) << s).min(EDRAM_TILE_COUNT) - start)
        };
        let (depth_start, mut depth_len) = depth.map_or((0, 0), |_| span(false));
        let (color_start, mut color_len) = color.map_or((0, 0), |c| span(c.format_is_64bpp));
        let depth_wrapped = info.depth_original_base.wrapping_add(depth_start) & (EDRAM_TILE_COUNT - 1);
        let color_wrapped = info.color_original_base.wrapping_add(color_start) & (EDRAM_TILE_COUNT - 1);
        if depth_len != 0 && color_len != 0 {
            let distance = |from: u32, to: u32| if to < from { EDRAM_TILE_COUNT } else { 0 } + to - from;
            depth_len = depth_len.min(distance(depth_wrapped, color_wrapped));
            color_len = color_len.min(distance(color_wrapped, depth_wrapped));
        }
        let rect = [x, y, x + w, y + h];
        let mut plans = Vec::new();
        if let Some(d) = depth.filter(|_| depth_len != 0) {
            plans.push(ClearPlan {
                key: RenderTargetKey {
                    base_tiles: info.depth_original_base & (EDRAM_TILE_COUNT - 1),
                    pitch_tiles_at_32bpp: pitch32,
                    msaa_samples: msaa,
                    is_depth: true,
                    format: d.format,
                },
                start: depth_wrapped,
                length: depth_len,
                rect,
                value: [info.depth_clear, 0],
            });
        }
        if let Some(c) = color.filter(|_| color_len != 0) {
            let format = ColorRenderTargetFormat::from_raw(c.format).storage_format();
            if host_color_format(format).is_none() {
                return Err(EdramError::UnsupportedFormat(c.format));
            }
            plans.push(ClearPlan {
                key: RenderTargetKey {
                    base_tiles: info.color_original_base & (EDRAM_TILE_COUNT - 1),
                    pitch_tiles_at_32bpp: pitch32,
                    msaa_samples: msaa,
                    is_depth: false,
                    format: format.raw(),
                },
                start: color_wrapped,
                length: color_len,
                rect,
                value: info.color_clear_value(),
            });
        }
        Ok(plans)
    }

    fn encode_clear(&mut self, device: &Device, encoder: &mut CommandEncoder, clear: &ClearPlan) {
        let rt = &self.rts[&clear.key];
        let (format, samples) = (rt.format, rt.samples);
        let pipeline = self.pipelines.clear(device, format, samples, clear.key.is_depth);
        let params = uniform(&self.params, device, &[clear.value[0], clear.value[1], format_code(&clear.key), 0]);
        let bind_group = cached_bind_group(
            &self.groups,
            device,
            "edram clear",
            self.pipelines.clear_layout(),
            &[BindGroupEntry { binding: 0, resource: params.binding() }],
        );
        let rt = &self.rts[&clear.key];
        let color = [Some(RenderPassColorAttachment::load(&rt.attachment))];
        let depth = RenderPassDepthStencilAttachment {
            view: &rt.attachment,
            depth_load_op: LoadOp::Load,
            depth_store_op: StoreOp::Store,
            depth_clear_value: 0.0,
            depth_read_only: false,
            stencil_load_op: LoadOp::Load,
            stencil_store_op: StoreOp::Store,
            stencil_clear_value: 0,
            stencil_read_only: false,
        };
        let mut pass = if clear.key.is_depth {
            encoder.begin_render_pass("edram clear", &[], Some(&depth))
        } else {
            encoder.begin_render_pass("edram clear", &color, None)
        };
        let [x0, y0, x1, y1] = clear.rect;
        pass.set_scissor_rect(x0, y0, x1.min(rt.width) - x0, y1.min(rt.height) - y0);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &params.dynamic_offset());
        pass.set_stencil_reference(clear.value[0] & 0xFF);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    fn encode_copy(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        info: &ResolveInfo,
        plan: CopyPlan,
        shared: Option<&SharedMemoryTarget<'_>>,
    ) -> CopyOutput {
        let source = plan.source;
        // Dump the source tiles from their owners into the EDRAM-layout buffer.
        let (base, row_length, rows, pitch) = info.copy_edram_tile_span().expect("copy has a source");
        let spans: Vec<(u32, u32)> = if row_length >= pitch {
            vec![(base, (rows * pitch).min(EDRAM_TILE_COUNT))]
        } else {
            (0..rows).map(|r| (base + r * pitch, row_length)).collect()
        };
        let mut dumps = Vec::new();
        for (start, length) in spans {
            for range in self.ownership.owners_in(start, length) {
                let Some(rt) = self.rts.get(&range.owner) else {
                    continue;
                };
                let kind = rt.src_kind();
                let count = (range.end - range.start) * TILE_DWORDS;
                let params = uniform(
                    &self.params,
                    device,
                    &[
                        range.start * TILE_DWORDS,
                        count,
                        rt.key.base_tiles & (EDRAM_TILE_COUNT - 1),
                        rt.key.pitch_tiles(),
                        rt.key.msaa_samples.raw(),
                        key_flags(&rt.key),
                        format_code(&rt.key),
                        rt.width,
                        rt.height,
                        0,
                        0,
                        0,
                    ],
                );
                let (layout, pipeline) = self.pipelines.dump(device, kind);
                let mut entries = vec![
                    BindGroupEntry { binding: 0, resource: params.binding() },
                    BindGroupEntry { binding: 3, resource: whole(&self.shadow) },
                ];
                entries.extend(src_entries(rt));
                let bind_group = cached_bind_group(&self.groups, device, "edram dump", layout, &entries);
                dumps.push((pipeline.clone(), bind_group, params, count.div_ceil(256)));
            }
        }

        let out_size = plan.out_bytes.max(4);
        if self.resolve_out.as_ref().is_none_or(|b| b.size() < out_size) {
            self.resolve_out = Some(device.create_buffer(&BufferDescriptor {
                label: "edram resolve output",
                size: out_size.next_multiple_of(1 << 16),
                usage: BufferUsage::Storage | BufferUsage::CopySrc,
                mapped_at_creation: false,
            }));
        }
        let out = self.resolve_out.clone().expect("created above");
        let dest_info = info.copy_dest_info;
        let flags = source.is_depth as u32
            | ((source.format_is_64bpp as u32) << 1)
            | ((dest_info.copy_dest_swap as u32) << 2)
            | ((source.decode_pwl_gamma as u32) << 3);
        let params = uniform(
            &self.params,
            device,
            &[
                info.rect.width,
                info.rect.height,
                info.edram_offset_x_div_8 << 3,
                info.edram_offset_y_div_8 << 3,
                source.base_tiles & (EDRAM_TILE_COUNT - 1),
                source.pitch_tiles,
                source.msaa_samples.raw(),
                flags,
                source.format,
                plan.mode,
                info.copy_sample_select.raw(),
                info.dest_format.raw(),
                dest_info.copy_dest_number.raw(),
                (dest_info.copy_dest_exp_bias.clamp(-32, 31) as f32).exp2().to_bits(),
                plan.words_per_pixel,
                plan.dest_bpp,
                info.copy_func,
                info.copy_ref,
                info.copy_mask,
                0,
            ],
        );
        let (layout, pipeline) = self.pipelines.resolve(device);
        // The output buffer is bound whole (the shaders don't take its length), so the group
        // doesn't depend on this resolve's size.
        let resolve_params = params;
        let bind_group = cached_bind_group(
            &self.groups,
            device,
            "edram resolve",
            layout,
            &[
                BindGroupEntry { binding: 0, resource: resolve_params.binding() },
                BindGroupEntry { binding: 1, resource: whole(&self.shadow) },
                BindGroupEntry { binding: 2, resource: whole(&out) },
            ],
        );
        let pipeline = pipeline.clone();
        let is_1111 = plan.kind == ResolveCopyKind::Dxt3aAs1111;
        let layout = DestLayout {
            base: info.copy_dest_base,
            pitch_aligned: info.dest_pitch_aligned,
            height_aligned: info.dest_height_aligned,
            is_3d: dest_info.copy_dest_array,
            slice: dest_info.copy_dest_slice,
            offset_x: info.dest_offset_x_div_8 << 3,
            offset_y: info.dest_offset_y_div_8 << 3,
            bytes_per_element: plan.dest_bpp / 8,
            endian: Endian128::from_bits(dest_info.copy_dest_endian.raw()),
            extent_start: info.copy_dest_extent_start,
            extent_len: info.copy_dest_extent_length,
            block_1111: is_1111,
        };
        let store_target = shared.filter(|t| layout.extent_len != 0 && t.chunks.len() == STORE_DST_BUFFERS as usize);
        let store = store_target.map(|target| {
            let params = uniform(
                &self.params,
                device,
                &[
                    layout.base & 0x1FFF_FFFF,
                    layout.pitch_aligned,
                    layout.height_aligned,
                    layout.is_3d as u32,
                    layout.slice,
                    layout.offset_x,
                    layout.offset_y,
                    layout.bytes_per_element,
                    layout.endian as u32,
                    info.rect.width,
                    info.rect.height,
                    plan.words_per_pixel,
                    is_1111 as u32,
                    (target.block_bytes / 4).ilog2() | ((target.chunk_bytes / 4).ilog2() << 8),
                    layout.extent_start & 0x1FFF_FFFF,
                    layout.extent_len,
                ],
            );
            let (bgl, store_pipeline) = self.pipelines.store(device, layout.bytes_per_element < 4);
            let bind_group = cached_bind_group(
                &self.groups,
                device,
                "edram store",
                bgl,
                &[
                    BindGroupEntry { binding: 0, resource: params.binding() },
                    BindGroupEntry { binding: 1, resource: whole(&out) },
                    BindGroupEntry { binding: 2, resource: whole(target.table) },
                    BindGroupEntry { binding: 3, resource: whole(&target.chunks[0]) },
                    BindGroupEntry { binding: 4, resource: whole(&target.chunks[1]) },
                    BindGroupEntry { binding: 5, resource: whole(&target.chunks[2]) },
                    BindGroupEntry { binding: 6, resource: whole(&target.chunks[3]) },
                ],
            );
            let (gx, gy) =
                if is_1111 { (info.rect.width / 4, info.rect.height / 4) } else { (info.rect.width, info.rect.height) };
            (store_pipeline.clone(), bind_group, params, gx.div_ceil(8), gy.div_ceil(8))
        });
        {
            let mut pass = encoder.begin_compute_pass("edram resolve");
            for (pipeline, bind_group, params, groups) in &dumps {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &params.dynamic_offset());
                pass.dispatch_workgroups(*groups, 1, 1);
            }
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &resolve_params.dynamic_offset());
            pass.dispatch_workgroups(info.rect.width.div_ceil(8), info.rect.height.div_ceil(8), 1);
            if let Some((store_pipeline, bind_group, params, gx, gy)) = &store {
                pass.set_pipeline(store_pipeline);
                pass.set_bind_group(0, bind_group, &params.dynamic_offset());
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
            pass.end();
        }
        if store.is_some() {
            return CopyOutput::Stored { written: (layout.extent_start & 0x1FFF_FFFF, layout.extent_len) };
        }
        let readback = match self.readback_pool.iter().position(|b| b.size() >= out_size) {
            Some(i) => self.readback_pool.swap_remove(i),
            None => device.create_buffer(&BufferDescriptor {
                label: "edram readback",
                size: out_size.next_multiple_of(1 << 16),
                usage: BufferUsage::MapRead | BufferUsage::CopyDst,
                mapped_at_creation: false,
            }),
        };
        encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out_size);

        CopyOutput::Readback(PendingCopy {
            readback,
            size: out_size,
            map: None,
            map_result: None,
            layout,
            width: info.rect.width,
            height: info.rect.height,
            words_per_pixel: plan.words_per_pixel as usize,
        })
    }
}
