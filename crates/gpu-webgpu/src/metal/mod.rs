//! Native Metal backend with the same API as the webgpu.h one (feature `metal`, Apple targets).
//!
//! Shaders stay WGSL: naga translates each pipeline stage to MSL with options chosen here
//! (bounds checks, invariance, safe math) and Metal compiles it, so the code the GPU runs can be
//! read (`MSL_DUMP_DIR`) and stepped through in Xcode's shader debugger.
//!
//! Ordering follows WebGPU: `Queue::write_buffer`/`write_texture` copy the data into an upload
//! ring right away and record blits in a command buffer that is committed ahead of the next
//! submission's. Each submission gets a serial; the completion handler of its last command buffer
//! publishes the serial, and polling resolves the `Pending`s waiting for it (maps, work-done) in
//! submission order. Resources are created with Metal's default hazard tracking, so Metal orders
//! their reads and writes across command buffers like WebGPU does.

mod command;
mod conv;
mod resource;
mod shader;
mod surface;
pub mod timing;
mod upscale;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

pub use command::{CommandBuffer, CommandEncoder, ComputePass, RenderPass};
pub use resource::{BindGroup, BindGroupLayout, Buffer, PipelineLayout, Sampler, Texture, TextureView};
pub use shader::{ComputePipeline, RenderPipeline, ShaderModule};
pub use surface::{Surface, SurfaceFrame};
pub use upscale::SpatialScaler;

use crate::pending::{Completer, Pending};
use crate::{
    AdapterInfo, AdapterType, BackendType, DeviceDescriptor, Error, ErrorFilter, ErrorType, FeatureName, GpuError,
    Limits, RequestAdapterOptions, Result,
};

/// Web only (see the webgpu.h backend); kept so callers compile unchanged.
pub static UNSHARED_UPLOADS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) type Id<T> = Retained<ProtocolObject<T>>;
pub(crate) type MtlBuffer = Id<dyn MTLBuffer>;
pub(crate) type MtlTexture = Id<dyn MTLTexture>;
pub(crate) type MtlCommandBuffer = Id<dyn MTLCommandBuffer>;

/// Uncaptured errors kept for [`Device::take_errors`]; older ones are only logged.
const MAX_KEPT_ERRORS: usize = 64;
/// Size of an upload ring chunk; larger uploads get a chunk of their own.
const UPLOAD_CHUNK: u64 = 4 << 20;
/// Upload offsets are aligned for any texel block (and buffer copies' 4-byte rule).
const UPLOAD_ALIGN: u64 = 16;
/// wgpu-native's 16-bit norm texture formats feature (`NATIVE_TEXTURE_FORMAT_16BIT_NORM`).
const FEATURE_16BIT_NORM: FeatureName = FeatureName(0x0003_000B);
/// wgpu-native's vertex-writable storage feature.
const FEATURE_VERTEX_WRITABLE_STORAGE: FeatureName = FeatureName(0x0003_0005);

pub(crate) fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

/// Sets a Metal label when there is one (Xcode's GPU capture shows them).
macro_rules! set_label {
    ($obj:expr, $label:expr) => {
        if !$label.is_empty() {
            $obj.setLabel(Some(&crate::metal::ns($label)));
        }
    };
}
pub(crate) use set_label;

thread_local! {
    /// Devices created on this thread, polled by [`Instance::process_events`].
    static DEVICES: RefCell<Vec<Weak<Shared>>> = const { RefCell::new(Vec::new()) };
}

/// Entry point: adapters, surfaces and event processing.
#[derive(Clone, Debug, Default)]
pub struct Instance {
    _thread: std::marker::PhantomData<*const ()>,
}

impl PartialEq for Instance {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Instance {
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    pub fn raw(&self) -> *const c_void {
        std::ptr::null()
    }

    /// Runs the completions of finished GPU work for this thread's devices.
    pub fn process_events(&self) {
        let devices: Vec<Rc<Shared>> =
            DEVICES.with_borrow_mut(|d| {
                d.retain(|w| w.strong_count() > 0);
                d.iter().filter_map(Weak::upgrade).collect()
            });
        for d in devices {
            d.poll();
        }
    }

    /// The system default GPU (Apple platforms have one).
    pub fn request_adapter(&self, _options: &RequestAdapterOptions<'_>) -> Pending<Result<Adapter>> {
        Pending::ready(match MTLCreateSystemDefaultDevice() {
            Some(raw) => Ok(Adapter::new(raw)),
            None => Err(Error::RequestAdapter("no Metal device".into())),
        })
    }

    /// Waits for `pending`, processing events in a loop.
    pub fn block_on<T>(&self, pending: Pending<T>) -> T {
        loop {
            if let Some(v) = pending.try_take() {
                return v;
            }
            self.process_events();
            if pending.is_ready() {
                continue;
            }
            let devices: Vec<Rc<Shared>> = DEVICES.with_borrow(|d| d.iter().filter_map(Weak::upgrade).collect());
            if !devices.iter().any(|d| d.wait_oldest()) {
                std::thread::yield_now();
            }
        }
    }
}

/// A physical GPU.
#[derive(Clone)]
pub struct Adapter {
    raw: Id<dyn MTLDevice>,
    features: Rc<Vec<FeatureName>>,
    limits: Limits,
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Adapter({})", self.raw.name())
    }
}

impl Adapter {
    fn new(raw: Id<dyn MTLDevice>) -> Self {
        let apple7 = raw.supportsFamily(MTLGPUFamily::Apple7);
        let mac2 = raw.supportsFamily(MTLGPUFamily::Mac2);
        let mut features = vec![
            FeatureName::CoreFeaturesAndLimits,
            FeatureName::DepthClipControl,
            FeatureName::Depth32FloatStencil8,
            FeatureName::IndirectFirstInstance,
            FeatureName::RG11B10UfloatRenderable,
            FeatureName::BGRA8UnormStorage,
            FeatureName::Float32Blendable,
            FeatureName::DualSourceBlending,
            FeatureName::TextureFormatsTier1,
            FEATURE_16BIT_NORM,
            FEATURE_VERTEX_WRITABLE_STORAGE,
        ];
        if apple7 || mac2 {
            features.push(FeatureName::Float32Filterable);
        }
        if raw.supportsBCTextureCompression() {
            features.push(FeatureName::TextureCompressionBC);
            features.push(FeatureName::TextureCompressionBCSliced3D);
        }
        if raw.supportsFamily(MTLGPUFamily::Apple2) {
            features.push(FeatureName::TextureCompressionETC2);
            features.push(FeatureName::TextureCompressionASTC);
        }
        let max_buffer = (raw.maxBufferLength() as u64).min(u64::from(u32::MAX) & !3);
        let textures_per_stage = if cfg!(target_os = "macos") { 128 } else { 96 };
        let limits = Limits {
            nextInChain: std::ptr::null_mut(),
            maxTextureDimension1D: 16384,
            maxTextureDimension2D: 16384,
            maxTextureDimension3D: 2048,
            maxTextureArrayLayers: 2048,
            maxBindGroups: 8,
            maxBindGroupsPlusVertexBuffers: 24,
            maxBindingsPerBindGroup: 1000,
            maxDynamicUniformBuffersPerPipelineLayout: 12,
            maxDynamicStorageBuffersPerPipelineLayout: 8,
            maxSampledTexturesPerShaderStage: textures_per_stage,
            maxSamplersPerShaderStage: 16,
            maxStorageBuffersPerShaderStage: 24,
            maxStorageTexturesPerShaderStage: 8,
            maxUniformBuffersPerShaderStage: 12,
            maxUniformBufferBindingSize: max_buffer,
            maxStorageBufferBindingSize: max_buffer,
            minUniformBufferOffsetAlignment: 256,
            minStorageBufferOffsetAlignment: 256,
            maxVertexBuffers: 8,
            maxBufferSize: raw.maxBufferLength() as u64,
            maxVertexAttributes: 31,
            maxVertexBufferArrayStride: 2048,
            maxInterStageShaderVariables: 31,
            maxColorAttachments: 8,
            maxColorAttachmentBytesPerSample: 64,
            maxComputeWorkgroupStorageSize: 32768,
            maxComputeInvocationsPerWorkgroup: 1024,
            maxComputeWorkgroupSizeX: 1024,
            maxComputeWorkgroupSizeY: 1024,
            maxComputeWorkgroupSizeZ: 64,
            maxComputeWorkgroupsPerDimension: 65535,
            maxImmediateSize: 0,
        };
        Self { raw, features: Rc::new(features), limits }
    }

    pub fn raw(&self) -> *const c_void {
        Retained::as_ptr(&self.raw).cast()
    }

    pub fn info(&self) -> AdapterInfo {
        AdapterInfo {
            vendor: "Apple".into(),
            architecture: String::new(),
            device: self.raw.name().to_string(),
            description: format!("Metal ({})", self.raw.name()),
            backend: BackendType::Metal,
            adapter_type: if self.raw.hasUnifiedMemory() { AdapterType::IntegratedGPU } else { AdapterType::DiscreteGPU },
            vendor_id: 0x106B,
            device_id: self.raw.registryID() as u32,
        }
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    pub fn features(&self) -> Vec<FeatureName> {
        self.features.to_vec()
    }

    pub fn has_feature(&self, feature: FeatureName) -> bool {
        self.features.contains(&feature)
    }

    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Pending<Result<Device>> {
        if let Some(missing) = desc.required_features.iter().find(|f| !self.features.contains(f)) {
            return Pending::ready(Err(Error::RequestDevice(format!("feature {missing:?} is not supported"))));
        }
        let Some(queue) = self.raw.newCommandQueueWithMaxCommandBufferCount(4096) else {
            return Pending::ready(Err(Error::RequestDevice("cannot create a command queue".into())));
        };
        set_label!(queue, desc.label);
        let shared = Rc::new(Shared {
            raw: self.raw.clone(),
            queue,
            features: desc.required_features.to_vec(),
            limits: self.limits,
            tracker: Arc::new(Tracker { completed: AtomicU64::new(0), errors: Mutex::new(Vec::new()) }),
            submitted: Cell::new(0),
            in_flight: RefCell::new(VecDeque::new()),
            waiters: RefCell::new(VecDeque::new()),
            uploads: RefCell::new(Uploads::default()),
            errors: RefCell::new(Vec::new()),
            error_count: Cell::new(0),
            scopes: RefCell::new(Vec::new()),
            lost: RefCell::new(None),
            depth_stencil_states: RefCell::new(HashMap::new()),
        });
        DEVICES.with_borrow_mut(|d| d.push(Rc::downgrade(&shared)));
        log::info!("Metal device: {} (max buffer {} MiB)", self.raw.name(), self.raw.maxBufferLength() >> 20);
        Pending::ready(Ok(Device { queue: Queue { shared: shared.clone() }, shared }))
    }
}

/// Written by command buffer completion handlers (on Metal's threads).
struct Tracker {
    completed: AtomicU64,
    errors: Mutex<Vec<String>>,
}

struct Chunk {
    buffer: MtlBuffer,
    used: u64,
}

/// The command buffer committed ahead of the next submission: uploads and initialisation.
struct Batch {
    cb: MtlCommandBuffer,
    blit: Option<Id<dyn MTLBlitCommandEncoder>>,
}

impl Batch {
    fn new(shared: &Shared) -> Self {
        let cb = shared.new_command_buffer();
        cb.setLabel(Some(&ns("uploads")));
        Batch { cb, blit: None }
    }

    fn blit(&mut self) -> &ProtocolObject<dyn MTLBlitCommandEncoder> {
        if self.blit.is_none() {
            self.blit = Some(self.cb.blitCommandEncoder().expect("cannot create a Metal blit encoder"));
        }
        self.blit.as_deref().unwrap()
    }

    fn end_blit(&mut self) {
        if let Some(b) = self.blit.take() {
            b.endEncoding();
        }
    }
}

/// The upload ring and the blits recorded since the last submission.
#[derive(Default)]
struct Uploads {
    batch: Option<Batch>,
    /// Zeroes, the source of texture initialisation copies (grown as needed).
    zeros: Option<MtlBuffer>,
    current: Option<Chunk>,
    /// Chunks read by a submission, reusable once its serial completes.
    retired: VecDeque<(u64, Chunk)>,
}

/// Depth/stencil state cache key (the raw values of the WebGPU state).
pub(crate) type DepthStencilKey = [u32; 12];

pub(crate) struct Shared {
    pub(crate) raw: Id<dyn MTLDevice>,
    queue: Id<dyn MTLCommandQueue>,
    features: Vec<FeatureName>,
    limits: Limits,
    tracker: Arc<Tracker>,
    /// Serial of the last submission.
    submitted: Cell<u64>,
    /// The last command buffer of each submission still running.
    in_flight: RefCell<VecDeque<(u64, MtlCommandBuffer)>>,
    /// Callbacks to run once a serial completes, in submission order.
    waiters: RefCell<VecDeque<(u64, Box<dyn FnOnce()>)>>,
    uploads: RefCell<Uploads>,
    errors: RefCell<Vec<GpuError>>,
    error_count: Cell<u64>,
    scopes: RefCell<Vec<(ErrorFilter, Option<GpuError>)>>,
    lost: RefCell<Option<String>>,
    pub(crate) depth_stencil_states: RefCell<HashMap<DepthStencilKey, Id<dyn MTLDepthStencilState>>>,
}

impl Shared {
    fn completed(&self) -> u64 {
        self.tracker.completed.load(Ordering::Acquire)
    }

    /// Reports an error to the innermost matching error scope, or logs and keeps it.
    pub(crate) fn report(&self, ty: ErrorType, message: String) {
        let filter = match ty {
            ErrorType::OutOfMemory => ErrorFilter::OutOfMemory,
            ErrorType::Internal => ErrorFilter::Internal,
            _ => ErrorFilter::Validation,
        };
        let err = GpuError { ty, message };
        let mut scopes = self.scopes.borrow_mut();
        if let Some((_, slot)) = scopes.iter_mut().rev().find(|(f, _)| *f == filter) {
            if slot.is_none() {
                *slot = Some(err);
            }
            return;
        }
        drop(scopes);
        log::error!("uncaptured Metal {err}");
        self.error_count.set(self.error_count.get() + 1);
        let mut errors = self.errors.borrow_mut();
        if errors.len() < MAX_KEPT_ERRORS {
            errors.push(err);
        }
    }

    /// Runs what completed GPU work unblocks.
    fn poll(&self) {
        crate::memory::DRIVER_ALLOCATED.store(self.raw.currentAllocatedSize() as u64, std::sync::atomic::Ordering::Relaxed);
        let done = self.completed();
        self.in_flight.borrow_mut().retain(|(serial, _)| *serial > done);
        let gpu_errors = std::mem::take(&mut *self.tracker.errors.lock().unwrap());
        for e in gpu_errors {
            self.report(ErrorType::Internal, e);
        }
        loop {
            let next = {
                let mut waiters = self.waiters.borrow_mut();
                match waiters.front() {
                    Some((serial, _)) if *serial <= done => waiters.pop_front(),
                    _ => None,
                }
            };
            match next {
                Some((_, f)) => f(),
                None => break,
            }
        }
        let mut uploads = self.uploads.borrow_mut();
        while uploads.retired.len() > 4 && uploads.retired.front().is_some_and(|(s, _)| *s <= done) {
            uploads.retired.pop_front();
        }
    }

    /// Blocks until the oldest running submission finishes; `false` if none is running.
    fn wait_oldest(&self) -> bool {
        let oldest = self.in_flight.borrow().front().cloned();
        match oldest {
            Some((serial, cb)) => {
                cb.waitUntilCompleted();
                self.tracker.completed.fetch_max(serial, Ordering::AcqRel);
                self.poll();
                true
            }
            None => false,
        }
    }

    /// Runs `f` once everything submitted so far has completed (right away if it has).
    pub(crate) fn after_submitted(&self, f: Box<dyn FnOnce()>) {
        let serial = self.submitted.get();
        if serial <= self.completed() && self.waiters.borrow().is_empty() {
            f();
        } else {
            self.waiters.borrow_mut().push_back((serial, f));
        }
    }

    pub(crate) fn new_command_buffer(&self) -> MtlCommandBuffer {
        autoreleasepool(|_| self.queue.commandBuffer()).expect("cannot create a Metal command buffer")
    }

    /// Space for `len` bytes of upload data and the blit encoder to copy it with.
    fn upload<R>(&self, len: u64, f: impl FnOnce(&MtlBuffer, u64, &ProtocolObject<dyn MTLBlitCommandEncoder>) -> R) -> R {
        let mut uploads = self.uploads.borrow_mut();
        let u = &mut *uploads;
        let fits = u.current.as_ref().is_some_and(|c| c.used.next_multiple_of(UPLOAD_ALIGN) + len <= c.buffer.length() as u64);
        if !fits {
            let submitted = self.submitted.get();
            if let Some(full) = u.current.take() {
                // Read by the next submission.
                u.retired.push_back((submitted + 1, full));
            }
            let done = self.completed();
            let reuse = if len <= UPLOAD_CHUNK && u.retired.front().is_some_and(|(s, c)| *s <= done && c.buffer.length() as u64 == UPLOAD_CHUNK) {
                u.retired.pop_front().map(|(_, c)| c.buffer)
            } else {
                None
            };
            let buffer = reuse.unwrap_or_else(|| {
                let size = len.max(UPLOAD_CHUNK);
                let b = self
                    .raw
                    .newBufferWithLength_options(size as usize, MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined)
                    .expect("cannot allocate a Metal upload buffer");
                b.setLabel(Some(&ns("upload ring")));
                b
            });
            u.current = Some(Chunk { buffer, used: 0 });
        }
        let chunk = u.current.as_mut().unwrap();
        let offset = chunk.used.next_multiple_of(UPLOAD_ALIGN);
        chunk.used = offset + len;
        let batch = u.batch.get_or_insert_with(|| Batch::new(self));
        f(&chunk.buffer, offset, batch.blit())
    }

    /// Zeroes a new buffer (WebGPU guarantees zeroed contents), on the GPU ahead of the next
    /// submission; on the CPU when small (nothing can be using it yet) or `mapped` (the caller
    /// writes through the mapping before that submission).
    pub(crate) fn zero_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>, mapped: bool) {
        let len = buffer.length();
        if mapped || len <= 1 << 20 {
            unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, len) };
            return;
        }
        let mut uploads = self.uploads.borrow_mut();
        let batch = uploads.batch.get_or_insert_with(|| Batch::new(self));
        batch.blit().fillBuffer_range_value(buffer, objc2_foundation::NSRange::new(0, len), 0);
    }

    /// Zeroes every subresource of a new texture ahead of the next submission: render targets
    /// with clearing passes, others by copying from a zeroed buffer.
    pub(crate) fn zero_texture(&self, texture: &ProtocolObject<dyn MTLTexture>, info: &resource::TextureInfo) {
        let mut uploads = self.uploads.borrow_mut();
        let u = &mut *uploads;
        let batch = u.batch.get_or_insert_with(|| Batch::new(self));
        let levels = info.mip_level_count.max(1);
        let is_3d = info.dimension == crate::TextureDimension::D3;
        let layers = if is_3d { 1 } else { info.depth_or_array_layers.max(1) };
        if info.render {
            batch.end_blit();
            let depth = conv::has_depth(info.format);
            let stencil = conv::has_stencil(info.format);
            for level in 0..levels {
                let planes = if is_3d { (info.depth_or_array_layers >> level).max(1) } else { 1 };
                for layer in 0..layers {
                    for plane in 0..planes {
                        let d = MTLRenderPassDescriptor::new();
                        let attach = |a: &MTLRenderPassAttachmentDescriptor| {
                            a.setTexture(Some(texture));
                            a.setLevel(level as usize);
                            a.setSlice(layer as usize);
                            a.setDepthPlane(plane as usize);
                            a.setLoadAction(MTLLoadAction::Clear);
                            a.setStoreAction(MTLStoreAction::Store);
                        };
                        if depth || stencil {
                            if depth {
                                let a = d.depthAttachment();
                                attach(&a);
                                a.setClearDepth(0.0);
                            }
                            if stencil {
                                let a = d.stencilAttachment();
                                attach(&a);
                                a.setClearStencil(0);
                            }
                        } else {
                            let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
                            attach(&a);
                            a.setClearColor(MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0 });
                        }
                        let encoder = autoreleasepool(|_| batch.cb.renderCommandEncoderWithDescriptor(&d))
                            .expect("cannot create a Metal render encoder");
                        encoder.endEncoding();
                    }
                }
            }
            return;
        }
        let Some((block_bytes, block)) = crate::format_block_info(info.format) else { return };
        let row = |level: u32| ((info.width >> level).max(1).div_ceil(block) * block_bytes) as usize;
        let rows = |level: u32| (info.height >> level).max(1).div_ceil(block) as usize;
        let depth_at = |level: u32| if is_3d { (info.depth_or_array_layers >> level).max(1) as usize } else { 1 };
        let need = row(0) * rows(0) * depth_at(0);
        if u.zeros.as_ref().is_none_or(|z| z.length() < need) {
            let z = self
                .raw
                .newBufferWithLength_options(need.next_multiple_of(1 << 20), MTLResourceOptions::StorageModePrivate)
                .expect("cannot allocate a Metal buffer of zeroes");
            z.setLabel(Some(&ns("zeroes")));
            batch.blit().fillBuffer_range_value(&z, objc2_foundation::NSRange::new(0, z.length()), 0);
            u.zeros = Some(z);
        }
        let zeros = u.zeros.clone().unwrap();
        let blit = batch.blit();
        for level in 0..levels {
            let (w, h) = ((info.width >> level).max(1), (info.height >> level).max(1));
            for layer in 0..layers {
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &zeros,
                        0,
                        row(level),
                        if is_3d { row(level) * rows(level) } else { 0 },
                        MTLSize { width: w as usize, height: h as usize, depth: depth_at(level) },
                        texture,
                        layer as usize,
                        level as usize,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    )
                };
            }
        }
    }

    /// Commits the pending uploads and `buffers` as one submission.
    pub(crate) fn submit(&self, buffers: Vec<MtlCommandBuffer>) {
        autoreleasepool(|_| {
            let mut all = Vec::with_capacity(buffers.len() + 1);
            {
                let mut uploads = self.uploads.borrow_mut();
                if let Some(mut batch) = uploads.batch.take() {
                    batch.end_blit();
                    all.push(batch.cb);
                }
                // Everything staged so far is read by this submission.
                if let Some(c) = uploads.current.take() {
                    let serial = self.submitted.get() + 1;
                    uploads.retired.push_back((serial, c));
                }
            }
            all.extend(buffers);
            if all.is_empty() {
                all.push(self.new_command_buffer());
            }
            let serial = self.submitted.get() + 1;
            self.submitted.set(serial);
            let last = all.last().unwrap().clone();
            let tracker = self.tracker.clone();
            let block = block2::RcBlock::new(move |cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let cb = unsafe { cb.as_ref() };
                if cb.status() == MTLCommandBufferStatus::Error {
                    let message = cb.error().map(|e| e.to_string()).unwrap_or_else(|| "unknown".into());
                    tracker.errors.lock().unwrap().push(format!("command buffer failed: {message}"));
                }
                crate::counters::gpu_ran(cb.GPUStartTime(), cb.GPUEndTime());
                tracker.completed.fetch_max(serial, Ordering::AcqRel);
            });
            unsafe { last.addCompletedHandler(block2::RcBlock::as_ptr(&block)) };
            // Command buffers that failed get their errors reported too.
            for cb in &all[..all.len() - 1] {
                let tracker = self.tracker.clone();
                let block = block2::RcBlock::new(move |cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                    let cb = unsafe { cb.as_ref() };
                    if cb.status() == MTLCommandBufferStatus::Error {
                        let message = cb.error().map(|e| e.to_string()).unwrap_or_else(|| "unknown".into());
                        tracker.errors.lock().unwrap().push(format!("command buffer failed: {message}"));
                    }
                    crate::counters::gpu_ran(cb.GPUStartTime(), cb.GPUEndTime());
                });
                unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&block)) };
            }
            for cb in &all {
                cb.commit();
            }
            self.in_flight.borrow_mut().push_back((serial, last));
        });
    }
}

/// A logical device and its queue.
#[derive(Clone)]
pub struct Device {
    pub(crate) shared: Rc<Shared>,
    queue: Queue,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({:p})", Rc::as_ptr(&self.shared))
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.shared, &other.shared)
    }
}

impl Device {
    pub fn raw(&self) -> *const c_void {
        Retained::as_ptr(&self.shared.raw).cast()
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn limits(&self) -> Limits {
        self.shared.limits
    }

    pub fn features(&self) -> Vec<FeatureName> {
        self.shared.features.clone()
    }

    pub fn has_feature(&self, feature: FeatureName) -> bool {
        self.shared.features.contains(&feature)
    }

    pub fn push_error_scope(&self, filter: ErrorFilter) {
        self.shared.scopes.borrow_mut().push((filter, None));
    }

    /// Resolves to the first error captured by the innermost scope, if any.
    pub fn pop_error_scope(&self) -> Pending<Result<Option<GpuError>>> {
        Pending::ready(match self.shared.scopes.borrow_mut().pop() {
            Some((_, err)) => Ok(err),
            None => Err(Error::ErrorScope("error scope stack is empty".into())),
        })
    }

    /// Uncaptured errors reported since the last call (at most the first 64 are kept).
    pub fn take_errors(&self) -> Vec<GpuError> {
        std::mem::take(&mut *self.shared.errors.borrow_mut())
    }

    /// Total uncaptured errors reported over the device's lifetime.
    pub fn error_count(&self) -> u64 {
        self.shared.error_count.get()
    }

    pub fn lost_reason(&self) -> Option<String> {
        self.shared.lost.borrow().clone()
    }

    pub fn destroy(&self) {
        *self.shared.lost.borrow_mut() = Some("destroyed".into());
    }

    /// Runs the completions of finished GPU work (maps, work-done).
    pub fn poll(&self) {
        self.shared.poll();
    }

    /// Blocks until the oldest submission still running completes (or yields if none is),
    /// instead of polling in a loop while waiting for the GPU.
    pub fn wait_for_gpu_progress(&self) {
        if !self.shared.wait_oldest() {
            std::thread::yield_now();
        }
    }

    /// Waits for `pending`, blocking on GPU progress.
    pub fn block_on<T>(&self, pending: Pending<T>) -> T {
        loop {
            if let Some(v) = pending.try_take() {
                return v;
            }
            self.shared.poll();
            if pending.is_ready() {
                continue;
            }
            if !self.shared.wait_oldest() {
                std::thread::yield_now();
            }
        }
    }
}

/// The device's queue.
#[derive(Clone)]
pub struct Queue {
    shared: Rc<Shared>,
}

impl std::fmt::Debug for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Queue({:p})", Rc::as_ptr(&self.shared))
    }
}

impl Queue {
    pub fn raw(&self) -> *const c_void {
        Retained::as_ptr(&self.shared.queue).cast()
    }

    pub fn submit(&self, buffers: impl IntoIterator<Item = CommandBuffer>) {
        // Staged uploads first (see `staging`).
        let copies = crate::staging::take_copies();
        self.submit_raw(copies.into_iter().chain(buffers));
        crate::staging::submitted();
    }

    pub(crate) fn submit_raw(&self, buffers: impl IntoIterator<Item = CommandBuffer>) {
        crate::counters::add(&crate::counters::count::submits, 1);
        self.shared.submit(buffers.into_iter().map(|b| b.raw).collect());
    }

    /// `offset` and `data.len()` must be multiples of 4.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        crate::counters::add(&crate::counters::count::write_buffer_bytes, data.len() as u64);
        crate::counters::add(&crate::counters::count::write_buffers, 1);
        crate::counters::written(buffer.label(), data.len() as u64);
        if data.is_empty() {
            return;
        }
        if crate::staging::stage(self, buffer, offset, data) {
            return;
        }
        let Some(dst) = buffer.mtl() else { return };
        if offset + data.len() as u64 > buffer.size() {
            self.shared.report(
                ErrorType::Validation,
                format!("write_buffer {offset}+{} exceeds {} ({} bytes)", data.len(), buffer.label(), buffer.size()),
            );
            return;
        }
        self.shared.upload(data.len() as u64, |src, at, blit| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), src.contents().as_ptr().cast::<u8>().add(at as usize), data.len());
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                src,
                at as usize,
                &dst,
                offset as usize,
                data.len(),
            );
        });
    }

    /// Uploads `data` laid out with `bytes_per_row` (no 256-byte alignment needed here).
    pub fn write_texture(
        &self,
        dst: &crate::TexelCopyTextureInfo<'_>,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: Option<u32>,
        size: crate::Extent3d,
    ) {
        crate::counters::add(&crate::counters::count::write_textures, 1);
        crate::counters::add(&crate::counters::count::write_texture_bytes, data.len() as u64);
        if data.is_empty() || size.width == 0 || size.height == 0 || size.depthOrArrayLayers == 0 {
            return;
        }
        let Some(texture) = dst.texture.mtl() else { return };
        self.shared.upload(data.len() as u64, |src, at, blit| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    src.contents().as_ptr().cast::<u8>().add(at as usize),
                    data.len(),
                )
            };
            command::encode_buffer_to_texture(
                blit,
                src,
                at,
                bytes_per_row,
                rows_per_image,
                &texture,
                dst.texture,
                dst.mip_level,
                dst.origin,
                dst.aspect,
                size,
            );
        });
    }

    /// Resolves once all work submitted so far has finished on the GPU.
    pub fn on_submitted_work_done(&self) -> Pending<Result<()>> {
        let (pending, done): (_, Completer<Result<()>>) = Pending::pair();
        self.shared.after_submitted(Box::new(move || done.complete(Ok(()))));
        pending
    }
}
