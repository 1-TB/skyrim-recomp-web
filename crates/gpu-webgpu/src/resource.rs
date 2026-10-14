use std::ffi::c_void;
use std::ptr;

use crate::device::BindGroupEntry;
use crate::handle::{handle, label, string, sv, wbool};
use crate::pending::{Pending, complete};
use crate::{
    AddressMode, BlendFactor, BlendOperation, BufferBindingType, ColorWriteMask, CompareFunction, CullMode, Device,
    Extent3d, FilterMode, FrontFace, IndexFormat, MipmapFilterMode, PrimitiveTopology, SamplerBindingType, ShaderStage,
    StencilOperation, StorageTextureAccess, TextureAspect, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsage, TextureViewDimension, VertexFormat, VertexStepMode, ffi,
};
use crate::{Error, Result};

handle!(Texture, WGPUTexture, wgpuTextureAddRef, wgpuTextureRelease);
handle!(TextureView, WGPUTextureView, wgpuTextureViewAddRef, wgpuTextureViewRelease);
handle!(Sampler, WGPUSampler, wgpuSamplerAddRef, wgpuSamplerRelease);
handle!(ShaderModule, WGPUShaderModule, wgpuShaderModuleAddRef, wgpuShaderModuleRelease);
handle!(BindGroupLayout, WGPUBindGroupLayout, wgpuBindGroupLayoutAddRef, wgpuBindGroupLayoutRelease);
handle!(BindGroup, WGPUBindGroup, wgpuBindGroupAddRef, wgpuBindGroupRelease);
handle!(PipelineLayout, WGPUPipelineLayout, wgpuPipelineLayoutAddRef, wgpuPipelineLayoutRelease);
handle!(RenderPipeline, WGPURenderPipeline, wgpuRenderPipelineAddRef, wgpuRenderPipelineRelease);
handle!(ComputePipeline, WGPUComputePipeline, wgpuComputePipelineAddRef, wgpuComputePipelineRelease);

#[derive(Clone, Copy, Debug)]
pub struct TextureDescriptor<'a> {
    pub label: &'a str,
    pub size: Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub view_formats: &'a [TextureFormat],
}

impl Default for TextureDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: "",
            size: crate::extent(1, 1, 1),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Undefined,
            usage: TextureUsage::None,
            view_formats: &[],
        }
    }
}

/// Fields left at their defaults inherit from the texture.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextureViewDescriptor<'a> {
    pub label: &'a str,
    pub format: TextureFormat,
    pub dimension: TextureViewDimension,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
    pub aspect: TextureAspect,
    pub usage: TextureUsage,
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerDescriptor<'a> {
    pub label: &'a str,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: MipmapFilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<CompareFunction>,
    pub max_anisotropy: u16,
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: "",
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            max_anisotropy: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BindingType {
    Buffer { ty: BufferBindingType, has_dynamic_offset: bool, min_binding_size: u64 },
    Sampler(SamplerBindingType),
    Texture { sample_type: TextureSampleType, view_dimension: TextureViewDimension, multisampled: bool },
    StorageTexture { access: StorageTextureAccess, format: TextureFormat, view_dimension: TextureViewDimension },
}

#[derive(Clone, Copy, Debug)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStage,
    pub ty: BindingType,
}

impl BindGroupLayoutEntry {
    fn to_ffi(self) -> ffi::WGPUBindGroupLayoutEntry {
        let mut e =
            ffi::WGPUBindGroupLayoutEntry { binding: self.binding, visibility: self.visibility, ..Default::default() };
        match self.ty {
            BindingType::Buffer { ty, has_dynamic_offset, min_binding_size } => {
                e.buffer = ffi::WGPUBufferBindingLayout {
                    r#type: ty,
                    hasDynamicOffset: wbool(has_dynamic_offset),
                    minBindingSize: min_binding_size,
                    ..Default::default()
                }
            }
            BindingType::Sampler(ty) => e.sampler = ffi::WGPUSamplerBindingLayout { r#type: ty, ..Default::default() },
            BindingType::Texture { sample_type, view_dimension, multisampled } => {
                e.texture = ffi::WGPUTextureBindingLayout {
                    sampleType: sample_type,
                    viewDimension: view_dimension,
                    multisampled: wbool(multisampled),
                    ..Default::default()
                }
            }
            BindingType::StorageTexture { access, format, view_dimension } => {
                e.storageTexture = ffi::WGPUStorageTextureBindingLayout {
                    access,
                    format,
                    viewDimension: view_dimension,
                    ..Default::default()
                }
            }
        }
        e
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBufferLayout<'a> {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

/// A shader entry point; an empty `entry_point` selects the module's only one for the stage.
#[derive(Clone, Copy, Debug)]
pub struct ProgrammableStage<'a> {
    pub module: &'a ShaderModule,
    pub entry_point: &'a str,
    /// Pipeline-overridable constants (`override` declarations).
    pub constants: &'a [(&'a str, f64)],
}

impl<'a> ProgrammableStage<'a> {
    pub fn new(module: &'a ShaderModule, entry_point: &'a str) -> Self {
        Self { module, entry_point, constants: &[] }
    }

    fn constants_ffi(&self) -> Vec<ffi::WGPUConstantEntry> {
        self.constants
            .iter()
            .map(|&(key, value)| ffi::WGPUConstantEntry { key: sv(key), value, ..Default::default() })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    /// Required for strip topologies with indexed draws, `Undefined` otherwise.
    pub strip_index_format: IndexFormat,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub unclipped_depth: bool,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: IndexFormat::Undefined,
            front_face: FrontFace::CCW,
            cull_mode: CullMode::None,
            unclipped_depth: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }
}

impl StencilFaceState {
    fn to_ffi(self) -> ffi::WGPUStencilFaceState {
        ffi::WGPUStencilFaceState {
            compare: self.compare,
            failOp: self.fail_op,
            depthFailOp: self.depth_fail_op,
            passOp: self.pass_op,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub depth_bias: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}

impl DepthStencilState {
    /// Depth test/write with `compare`, stencil disabled.
    pub fn depth(format: TextureFormat, write: bool, compare: CompareFunction) -> Self {
        Self {
            format,
            depth_write_enabled: write,
            depth_compare: compare,
            stencil_front: StencilFaceState::default(),
            stencil_back: StencilFaceState::default(),
            stencil_read_mask: !0,
            stencil_write_mask: !0,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u32,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self { count: 1, mask: !0, alpha_to_coverage_enabled: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendComponent {
    pub operation: BlendOperation,
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
}

impl BlendComponent {
    pub const REPLACE: Self =
        Self { operation: BlendOperation::Add, src_factor: BlendFactor::One, dst_factor: BlendFactor::Zero };
    pub const OVER: Self = Self {
        operation: BlendOperation::Add,
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
    };

    fn to_ffi(self) -> ffi::WGPUBlendComponent {
        ffi::WGPUBlendComponent { operation: self.operation, srcFactor: self.src_factor, dstFactor: self.dst_factor }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            operation: BlendOperation::Add,
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
        },
        alpha: BlendComponent::OVER,
    };
}

/// A render target slot; `format: Undefined` leaves the slot unused.
#[derive(Clone, Copy, Debug)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWriteMask,
}

impl ColorTargetState {
    pub fn new(format: TextureFormat) -> Self {
        Self { format, blend: None, write_mask: ColorWriteMask::All }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FragmentState<'a> {
    pub stage: ProgrammableStage<'a>,
    pub targets: &'a [ColorTargetState],
}

#[derive(Clone, Copy, Debug)]
pub struct RenderPipelineDescriptor<'a> {
    pub label: &'a str,
    /// `None` derives the layout from the shaders (`get_bind_group_layout`).
    pub layout: Option<&'a PipelineLayout>,
    pub vertex: ProgrammableStage<'a>,
    pub vertex_buffers: &'a [VertexBufferLayout<'a>],
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<FragmentState<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ComputePipelineDescriptor<'a> {
    pub label: &'a str,
    pub layout: Option<&'a PipelineLayout>,
    pub stage: ProgrammableStage<'a>,
}

fn entry_point(s: &str) -> ffi::WGPUStringView {
    label(s)
}

impl Device {
    pub fn create_texture(&self, desc: &TextureDescriptor<'_>) -> Texture {
        crate::counters::add(&crate::counters::count::textures, 1);
        crate::counters::created("texture", desc.label);
        let raw = ffi::WGPUTextureDescriptor {
            label: label(desc.label),
            usage: desc.usage,
            dimension: desc.dimension,
            size: desc.size,
            format: desc.format,
            mipLevelCount: desc.mip_level_count,
            sampleCount: desc.sample_count,
            viewFormatCount: desc.view_formats.len(),
            viewFormats: desc.view_formats.as_ptr(),
            ..Default::default()
        };
        let texture = Texture::from_created(unsafe { ffi::wgpuDeviceCreateTexture(self.raw(), &raw) }, "texture")
            .expect("wgpuDeviceCreateTexture returned null");
        let s = desc.size;
        let bytes = crate::memory::texture_bytes(
            desc.format,
            s.width,
            s.height,
            s.depthOrArrayLayers,
            desc.mip_level_count,
            desc.sample_count,
        );
        crate::memory::track(texture.raw() as usize, "texture", desc.label, bytes);
        texture
    }

    pub fn create_sampler(&self, desc: &SamplerDescriptor<'_>) -> Sampler {
        crate::counters::add(&crate::counters::count::samplers, 1);
        let raw = ffi::WGPUSamplerDescriptor {
            label: label(desc.label),
            addressModeU: desc.address_mode_u,
            addressModeV: desc.address_mode_v,
            addressModeW: desc.address_mode_w,
            magFilter: desc.mag_filter,
            minFilter: desc.min_filter,
            mipmapFilter: desc.mipmap_filter,
            lodMinClamp: desc.lod_min_clamp,
            lodMaxClamp: desc.lod_max_clamp,
            compare: desc.compare.unwrap_or(CompareFunction::Undefined),
            maxAnisotropy: desc.max_anisotropy,
            ..Default::default()
        };
        Sampler::from_created(unsafe { ffi::wgpuDeviceCreateSampler(self.raw(), &raw) }, "sampler")
            .expect("wgpuDeviceCreateSampler returned null")
    }

    /// Compile errors surface through error scopes / the uncaptured-error log.
    pub fn create_shader_module_wgsl(&self, label_: &str, code: &str) -> ShaderModule {
        crate::counters::add(&crate::counters::count::shader_modules, 1);
        crate::counters::add(&crate::counters::count::shader_bytes, code.len() as u64);
        let mut source = ffi::WGPUShaderSourceWGSL { code: sv(code), ..Default::default() };
        let raw = ffi::WGPUShaderModuleDescriptor {
            nextInChain: ptr::from_mut(&mut source).cast::<ffi::WGPUChainedStruct>(),
            label: label(label_),
        };
        ShaderModule::from_created(unsafe { ffi::wgpuDeviceCreateShaderModule(self.raw(), &raw) }, "shader module")
            .expect("wgpuDeviceCreateShaderModule returned null")
    }

    pub fn create_bind_group_layout(&self, label_: &str, entries: &[BindGroupLayoutEntry]) -> BindGroupLayout {
        let entries: Vec<_> = entries.iter().map(|e| e.to_ffi()).collect();
        let raw = ffi::WGPUBindGroupLayoutDescriptor {
            label: label(label_),
            entryCount: entries.len(),
            entries: entries.as_ptr(),
            ..Default::default()
        };
        BindGroupLayout::from_created(unsafe { ffi::wgpuDeviceCreateBindGroupLayout(self.raw(), &raw) }, "layout")
            .expect("wgpuDeviceCreateBindGroupLayout returned null")
    }

    pub fn create_bind_group(
        &self,
        label_: &str,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        crate::counters::add(&crate::counters::count::bind_groups, 1);
        crate::counters::created("bind group", label_);
        let entries: Vec<_> = entries.iter().map(|e| e.to_ffi()).collect();
        let raw = ffi::WGPUBindGroupDescriptor {
            label: label(label_),
            layout: layout.raw(),
            entryCount: entries.len(),
            entries: entries.as_ptr(),
            ..Default::default()
        };
        BindGroup::from_created(unsafe { ffi::wgpuDeviceCreateBindGroup(self.raw(), &raw) }, "bind group")
            .expect("wgpuDeviceCreateBindGroup returned null")
    }

    pub fn create_pipeline_layout(&self, label_: &str, layouts: &[&BindGroupLayout]) -> PipelineLayout {
        let raws: Vec<_> = layouts.iter().map(|l| l.raw()).collect();
        let raw = ffi::WGPUPipelineLayoutDescriptor {
            label: label(label_),
            bindGroupLayoutCount: raws.len(),
            bindGroupLayouts: raws.as_ptr(),
            ..Default::default()
        };
        PipelineLayout::from_created(unsafe { ffi::wgpuDeviceCreatePipelineLayout(self.raw(), &raw) }, "layout")
            .expect("wgpuDeviceCreatePipelineLayout returned null")
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> RenderPipeline {
        crate::counters::add(&crate::counters::count::pipelines, 1);
        with_render_pipeline_descriptor(desc, |raw| {
            RenderPipeline::from_created(unsafe { ffi::wgpuDeviceCreateRenderPipeline(self.raw(), raw) }, "pipeline")
                .expect("wgpuDeviceCreateRenderPipeline returned null")
        })
    }

    /// Compiles the pipeline in the background; resolves to it, or to the validation or
    /// compilation error. Browsers compile synchronous pipelines on first use, stalling the GPU
    /// process; this lets the caller skip work until the pipeline is ready instead. wgpu-native
    /// (v29) doesn't implement it and panics: use [`Device::create_render_pipeline`] natively.
    pub fn create_render_pipeline_async(&self, desc: &RenderPipelineDescriptor<'_>) -> Pending<Result<RenderPipeline>> {
        crate::counters::add(&crate::counters::count::pipelines, 1);
        unsafe extern "C" fn done(
            status: ffi::WGPUCreatePipelineAsyncStatus,
            pipeline: ffi::WGPURenderPipeline,
            message: ffi::WGPUStringView,
            userdata: *mut c_void,
            _: *mut c_void,
        ) {
            let result = if status == ffi::WGPUCreatePipelineAsyncStatus::Success && !pipeline.is_null() {
                // The callback receives ownership of one reference.
                Ok(unsafe { RenderPipeline::from_raw(pipeline) })
            } else {
                if !pipeline.is_null() {
                    unsafe { ffi::wgpuRenderPipelineRelease(pipeline) };
                }
                Err(Error::Invalid(format!("pipeline creation failed (status {}): {}", status.0, string(message))))
            };
            unsafe { complete(userdata, result) };
        }
        let (pending, userdata) = Pending::new();
        let info = ffi::WGPUCreateRenderPipelineAsyncCallbackInfo {
            mode: ffi::WGPUCallbackMode::AllowProcessEvents,
            callback: Some(done),
            userdata1: userdata,
            ..Default::default()
        };
        with_render_pipeline_descriptor(desc, |raw| unsafe {
            ffi::wgpuDeviceCreateRenderPipelineAsync(self.raw(), raw, info);
        });
        pending
    }
}

/// Calls `f` with the webgpu.h form of `desc` (valid only during the call).
fn with_render_pipeline_descriptor<R>(
    desc: &RenderPipelineDescriptor<'_>,
    f: impl FnOnce(&ffi::WGPURenderPipelineDescriptor) -> R,
) -> R {
    {
        let vertex_constants = desc.vertex.constants_ffi();
        let attributes: Vec<Vec<ffi::WGPUVertexAttribute>> = desc
            .vertex_buffers
            .iter()
            .map(|b| {
                b.attributes
                    .iter()
                    .map(|a| ffi::WGPUVertexAttribute {
                        format: a.format,
                        offset: a.offset,
                        shaderLocation: a.shader_location,
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        let buffers: Vec<_> = desc
            .vertex_buffers
            .iter()
            .zip(&attributes)
            .map(|(b, attrs)| ffi::WGPUVertexBufferLayout {
                stepMode: b.step_mode,
                arrayStride: b.array_stride,
                attributeCount: attrs.len(),
                attributes: attrs.as_ptr(),
                ..Default::default()
            })
            .collect();
        let depth_stencil = desc.depth_stencil.map(|d| ffi::WGPUDepthStencilState {
            format: d.format,
            depthWriteEnabled: if d.depth_write_enabled {
                ffi::WGPUOptionalBool::True
            } else {
                ffi::WGPUOptionalBool::False
            },
            depthCompare: d.depth_compare,
            stencilFront: d.stencil_front.to_ffi(),
            stencilBack: d.stencil_back.to_ffi(),
            stencilReadMask: d.stencil_read_mask,
            stencilWriteMask: d.stencil_write_mask,
            depthBias: d.depth_bias,
            depthBiasSlopeScale: d.depth_bias_slope_scale,
            depthBiasClamp: d.depth_bias_clamp,
            ..Default::default()
        });
        let fragment_constants = desc.fragment.map(|f| f.stage.constants_ffi()).unwrap_or_default();
        let blends: Vec<Option<ffi::WGPUBlendState>> = desc
            .fragment
            .map(|f| {
                f.targets
                    .iter()
                    .map(|t| t.blend.map(|b| ffi::WGPUBlendState { color: b.color.to_ffi(), alpha: b.alpha.to_ffi() }))
                    .collect()
            })
            .unwrap_or_default();
        let targets: Vec<_> = desc
            .fragment
            .map(|f| {
                f.targets
                    .iter()
                    .zip(&blends)
                    .map(|(t, b)| ffi::WGPUColorTargetState {
                        format: t.format,
                        blend: b.as_ref().map_or(ptr::null(), ptr::from_ref),
                        writeMask: t.write_mask,
                        ..Default::default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let fragment = desc.fragment.map(|f| ffi::WGPUFragmentState {
            module: f.stage.module.raw(),
            entryPoint: entry_point(f.stage.entry_point),
            constantCount: fragment_constants.len(),
            constants: fragment_constants.as_ptr(),
            targetCount: targets.len(),
            targets: targets.as_ptr(),
            ..Default::default()
        });
        let raw = ffi::WGPURenderPipelineDescriptor {
            label: label(desc.label),
            layout: desc.layout.map_or(ptr::null_mut(), |l| l.raw()),
            vertex: ffi::WGPUVertexState {
                module: desc.vertex.module.raw(),
                entryPoint: entry_point(desc.vertex.entry_point),
                constantCount: vertex_constants.len(),
                constants: vertex_constants.as_ptr(),
                bufferCount: buffers.len(),
                buffers: buffers.as_ptr(),
                ..Default::default()
            },
            primitive: ffi::WGPUPrimitiveState {
                topology: desc.primitive.topology,
                stripIndexFormat: desc.primitive.strip_index_format,
                frontFace: desc.primitive.front_face,
                cullMode: desc.primitive.cull_mode,
                unclippedDepth: wbool(desc.primitive.unclipped_depth),
                ..Default::default()
            },
            depthStencil: depth_stencil.as_ref().map_or(ptr::null(), ptr::from_ref),
            multisample: ffi::WGPUMultisampleState {
                count: desc.multisample.count,
                mask: desc.multisample.mask,
                alphaToCoverageEnabled: wbool(desc.multisample.alpha_to_coverage_enabled),
                ..Default::default()
            },
            fragment: fragment.as_ref().map_or(ptr::null(), ptr::from_ref),
            ..Default::default()
        };
        f(&raw)
    }
}

impl Device {
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor<'_>) -> ComputePipeline {
        crate::counters::add(&crate::counters::count::pipelines, 1);
        let constants = desc.stage.constants_ffi();
        let raw = ffi::WGPUComputePipelineDescriptor {
            label: label(desc.label),
            layout: desc.layout.map_or(ptr::null_mut(), |l| l.raw()),
            compute: ffi::WGPUComputeState {
                module: desc.stage.module.raw(),
                entryPoint: entry_point(desc.stage.entry_point),
                constantCount: constants.len(),
                constants: constants.as_ptr(),
                ..Default::default()
            },
            ..Default::default()
        };
        ComputePipeline::from_created(unsafe { ffi::wgpuDeviceCreateComputePipeline(self.raw(), &raw) }, "pipeline")
            .expect("wgpuDeviceCreateComputePipeline returned null")
    }
}

impl Texture {
    pub fn create_view(&self, desc: Option<&TextureViewDescriptor<'_>>) -> TextureView {
        crate::counters::add(&crate::counters::count::views, 1);
        let raw = desc.map(|d| ffi::WGPUTextureViewDescriptor {
            label: label(d.label),
            format: d.format,
            dimension: d.dimension,
            baseMipLevel: d.base_mip_level,
            mipLevelCount: d.mip_level_count.unwrap_or(ffi::WGPU_MIP_LEVEL_COUNT_UNDEFINED),
            baseArrayLayer: d.base_array_layer,
            arrayLayerCount: d.array_layer_count.unwrap_or(ffi::WGPU_ARRAY_LAYER_COUNT_UNDEFINED),
            aspect: d.aspect,
            usage: d.usage,
            ..Default::default()
        });
        let ptr = raw.as_ref().map_or(ptr::null(), ptr::from_ref);
        TextureView::from_created(unsafe { ffi::wgpuTextureCreateView(self.raw(), ptr) }, "texture view")
            .expect("wgpuTextureCreateView returned null")
    }

    pub fn width(&self) -> u32 {
        unsafe { ffi::wgpuTextureGetWidth(self.raw()) }
    }

    pub fn height(&self) -> u32 {
        unsafe { ffi::wgpuTextureGetHeight(self.raw()) }
    }

    pub fn depth_or_array_layers(&self) -> u32 {
        unsafe { ffi::wgpuTextureGetDepthOrArrayLayers(self.raw()) }
    }

    pub fn mip_level_count(&self) -> u32 {
        unsafe { ffi::wgpuTextureGetMipLevelCount(self.raw()) }
    }

    pub fn sample_count(&self) -> u32 {
        unsafe { ffi::wgpuTextureGetSampleCount(self.raw()) }
    }

    pub fn format(&self) -> TextureFormat {
        unsafe { ffi::wgpuTextureGetFormat(self.raw()) }
    }

    pub fn usage(&self) -> TextureUsage {
        unsafe { ffi::wgpuTextureGetUsage(self.raw()) }
    }

    pub fn destroy(&self) {
        crate::memory::untrack(self.raw() as usize);
        unsafe { ffi::wgpuTextureDestroy(self.raw()) };
    }
}

impl RenderPipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        BindGroupLayout::from_created(
            unsafe { ffi::wgpuRenderPipelineGetBindGroupLayout(self.raw(), index) },
            "bind group layout",
        )
        .expect("wgpuRenderPipelineGetBindGroupLayout returned null")
    }
}

impl ComputePipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        BindGroupLayout::from_created(
            unsafe { ffi::wgpuComputePipelineGetBindGroupLayout(self.raw(), index) },
            "bind group layout",
        )
        .expect("wgpuComputePipelineGetBindGroupLayout returned null")
    }
}
