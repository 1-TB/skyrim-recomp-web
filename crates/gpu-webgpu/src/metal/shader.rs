//! Shader modules (WGSL parsed and validated by naga) and pipelines (one MSL translation and
//! Metal library per stage).
//!
//! Metal slots come from the shader, not the pipeline layout: each stage numbers the buffers,
//! textures and samplers its entry point uses, in (group, binding) order, and records where each
//! (group, binding) went. Bind groups are applied through that map at draw time, so pipelines
//! with different layouts (or none) need no further information.

use std::fmt::Write as _;
use std::rc::Rc;

use naga::back::msl;
use objc2::rc::Retained;
use objc2_metal::*;

use super::resource::rc_handle;
use super::{DepthStencilKey, Device, Id, Shared, conv, ns, set_label};
use crate::pending::Pending;
use crate::{
    BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipelineDescriptor, Error,
    ErrorType, ProgrammableStage, RenderPipelineDescriptor, Result, SamplerBindingType, ShaderStage,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

pub(crate) struct ShaderInner {
    label: String,
    /// The parsed module, or the parse/validation error.
    module: std::result::Result<(naga::Module, naga::valid::ModuleInfo), String>,
    /// Stages compiled from this module, by everything besides the module that the MSL
    /// depends on: pipelines differing only in render state (most of a title's) share their
    /// shaders, and compiling MSL is most of a pipeline's cost (~20 ms per stage).
    stages: std::cell::RefCell<std::collections::HashMap<StageKey, CompiledStage>>,
}

/// What a stage's MSL depends on besides its module (see [`StageRequest`]).
#[derive(Clone, PartialEq, Eq, Hash)]
struct StageKey {
    entry: String,
    constants: Vec<(String, u64)>,
    naga_stage: naga::ShaderStage,
    point: bool,
    sample_mask: u32,
    vertex_buffers: usize,
}

#[derive(Clone)]
pub struct ShaderModule {
    pub(crate) inner: Rc<ShaderInner>,
}

rc_handle!(ShaderModule);

fn validate(module: &naga::Module) -> std::result::Result<naga::valid::ModuleInfo, String> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(module)
        .map_err(|e| format!("{:?}", e.into_inner()))
}

impl Device {
    /// Parse and validation errors surface through error scopes / the uncaptured-error log.
    pub fn create_shader_module_wgsl(&self, label: &str, code: &str) -> ShaderModule {
        crate::counters::add(&crate::counters::count::shader_modules, 1);
        crate::counters::add(&crate::counters::count::shader_bytes, code.len() as u64);
        let module = naga::front::wgsl::parse_str(code)
            .map_err(|e| e.emit_to_string(code))
            .and_then(|m| validate(&m).map(|info| (m, info)));
        if let Err(e) = &module {
            self.shared.report(ErrorType::Validation, format!("shader module {label:?}: {e}"));
        }
        ShaderModule { inner: Rc::new(ShaderInner { label: label.to_string(), module, stages: Default::default() }) }
    }
}

/// Where one (group, binding) of a stage goes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slot {
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) index: usize,
}

/// A stage's Metal slot assignment.
#[derive(Clone, Debug, Default)]
pub(crate) struct StageBindings {
    pub(crate) buffers: Vec<Slot>,
    pub(crate) textures: Vec<Slot>,
    pub(crate) samplers: Vec<Slot>,
    /// Buffer slot of `_mslBufferSizes` and the bindings whose byte sizes it lists, in order.
    pub(crate) sizes: Option<(usize, Vec<(u32, u32)>)>,
    /// Bit per bind group the stage reads.
    pub(crate) groups: u32,
    /// Threadgroup memory per index (compute).
    pub(crate) workgroup_memory: Vec<usize>,
}

#[derive(Clone)]
struct CompiledStage {
    function: Id<dyn MTLFunction>,
    bindings: StageBindings,
    workgroup_size: [u32; 3],
    /// Bind group layout entries the stage implies (for `get_bind_group_layout`).
    entries: Vec<(u32, BindGroupLayoutEntry)>,
}

fn layout_entry(module: &naga::Module, var: &naga::GlobalVariable, visibility: ShaderStage) -> Option<BindGroupLayoutEntry> {
    let binding = var.binding.as_ref()?.binding;
    let dim = |d: naga::ImageDimension, arrayed: bool| match (d, arrayed) {
        (naga::ImageDimension::D1, _) => TextureViewDimension::D1,
        (naga::ImageDimension::D2, false) => TextureViewDimension::D2,
        (naga::ImageDimension::D2, true) => TextureViewDimension::D2Array,
        (naga::ImageDimension::D3, _) => TextureViewDimension::D3,
        (naga::ImageDimension::Cube, false) => TextureViewDimension::Cube,
        (naga::ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
    };
    let ty = match module.types[var.ty].inner {
        naga::TypeInner::Image { dim: d, arrayed, class } => match class {
            naga::ImageClass::Sampled { kind, multi } => BindingType::Texture {
                sample_type: match kind {
                    naga::ScalarKind::Sint => TextureSampleType::Sint,
                    naga::ScalarKind::Uint => TextureSampleType::Uint,
                    _ => TextureSampleType::Float,
                },
                view_dimension: dim(d, arrayed),
                multisampled: multi,
            },
            naga::ImageClass::Depth { multi } => BindingType::Texture {
                sample_type: TextureSampleType::Depth,
                view_dimension: dim(d, arrayed),
                multisampled: multi,
            },
            naga::ImageClass::Storage { access, .. } => BindingType::StorageTexture {
                access: if access.contains(naga::StorageAccess::LOAD | naga::StorageAccess::STORE) {
                    StorageTextureAccess::ReadWrite
                } else if access.contains(naga::StorageAccess::STORE) {
                    StorageTextureAccess::WriteOnly
                } else {
                    StorageTextureAccess::ReadOnly
                },
                format: TextureFormat::Undefined,
                view_dimension: dim(d, arrayed),
            },
            _ => return None,
        },
        naga::TypeInner::Sampler { comparison } => BindingType::Sampler(if comparison {
            SamplerBindingType::Comparison
        } else {
            SamplerBindingType::Filtering
        }),
        _ => BindingType::Buffer {
            ty: match var.space {
                naga::AddressSpace::Uniform => BufferBindingType::Uniform,
                naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE) => {
                    BufferBindingType::Storage
                }
                _ => BufferBindingType::ReadOnlyStorage,
            },
            has_dynamic_offset: false,
            min_binding_size: 0,
        },
    };
    Some(BindGroupLayoutEntry { binding, visibility, ty })
}

/// Adds `mask` as the fragment shader's sample mask output: Metal has no pipeline sample mask,
/// so the entry point's result gets a `@builtin(sample_mask)` member (ANDed into an existing one).
fn add_sample_mask(module: &mut naga::Module, ep_index: usize, mask: u32) -> std::result::Result<(), String> {
    use naga::{Binding, BuiltIn, Expression, Span, Statement, StructMember, Type, TypeInner};
    let u32_ty = module.types.insert(
        Type { name: None, inner: TypeInner::Scalar(naga::Scalar::U32) },
        Span::UNDEFINED,
    );
    let ep = &module.entry_points[ep_index];
    let old = ep.function.result.clone();
    // Members of the new result: the old ones (or the old single value) plus the mask.
    let (mut members, existing_mask, single) = match &old {
        Some(r) => match &module.types[r.ty].inner {
            TypeInner::Struct { members, .. } => {
                let at = members.iter().position(|m| m.binding == Some(Binding::BuiltIn(BuiltIn::SampleMask)));
                (members.clone(), at, false)
            }
            _ => (
                vec![StructMember { name: Some("value".into()), ty: r.ty, binding: r.binding.clone(), offset: 0 }],
                None,
                true,
            ),
        },
        None => (Vec::new(), None, false),
    };
    if existing_mask.is_none() {
        members.push(StructMember {
            name: Some("xe_sample_mask".into()),
            ty: u32_ty,
            binding: Some(Binding::BuiltIn(BuiltIn::SampleMask)),
            offset: 0,
        });
    }
    let mut offset = 0;
    for m in &mut members {
        m.offset = offset;
        offset += 16;
    }
    let new_ty = module.types.insert(
        Type { name: Some("XeMaskedOutput".into()), inner: TypeInner::Struct { members: members.clone(), span: offset } },
        Span::UNDEFINED,
    );
    let ep = &mut module.entry_points[ep_index];
    let f = &mut ep.function;
    f.result = Some(naga::FunctionResult { ty: new_ty, binding: None });

    // Rewrites every `return value;` into `return XeMaskedOutput(..., mask);`.
    fn rewrite(
        block: &mut naga::Block,
        f_exprs: &mut naga::Arena<Expression>,
        new_ty: naga::Handle<Type>,
        n_old: u32,
        existing_mask: Option<usize>,
        single: bool,
        mask: u32,
    ) {
        let mut i = 0;
        while i < block.len() {
            let replace = match &mut block[i] {
                Statement::Block(b) => {
                    rewrite(b, f_exprs, new_ty, n_old, existing_mask, single, mask);
                    None
                }
                Statement::If { accept, reject, .. } => {
                    rewrite(accept, f_exprs, new_ty, n_old, existing_mask, single, mask);
                    rewrite(reject, f_exprs, new_ty, n_old, existing_mask, single, mask);
                    None
                }
                Statement::Switch { cases, .. } => {
                    for c in cases {
                        rewrite(&mut c.body, f_exprs, new_ty, n_old, existing_mask, single, mask);
                    }
                    None
                }
                Statement::Loop { body, continuing, .. } => {
                    rewrite(body, f_exprs, new_ty, n_old, existing_mask, single, mask);
                    rewrite(continuing, f_exprs, new_ty, n_old, existing_mask, single, mask);
                    None
                }
                Statement::Return { value } => Some(*value),
                _ => None,
            };
            if let Some(value) = replace {
                let literal = f_exprs.append(Expression::Literal(naga::Literal::U32(mask)), naga::Span::UNDEFINED);
                let start = f_exprs.len();
                let mut components = Vec::new();
                if let Some(v) = value {
                    if single {
                        components.push(v);
                    } else {
                        for m in 0..n_old {
                            let c = f_exprs.append(Expression::AccessIndex { base: v, index: m }, naga::Span::UNDEFINED);
                            components.push(c);
                        }
                    }
                }
                match existing_mask {
                    Some(at) => {
                        let combined = f_exprs.append(
                            Expression::Binary { op: naga::BinaryOperator::And, left: components[at], right: literal },
                            naga::Span::UNDEFINED,
                        );
                        components[at] = combined;
                    }
                    None => components.push(literal),
                }
                let compose = f_exprs.append(Expression::Compose { ty: new_ty, components }, naga::Span::UNDEFINED);
                let range = f_exprs.range_from(start);
                let mut replacement = naga::Block::new();
                replacement.push(Statement::Emit(range), naga::Span::UNDEFINED);
                replacement.push(Statement::Return { value: Some(compose) }, naga::Span::UNDEFINED);
                block[i] = Statement::Block(replacement);
            }
            i += 1;
        }
    }
    let n_old = if single { 1 } else { members.len() as u32 - u32::from(existing_mask.is_none()) };
    let mut body = std::mem::take(&mut f.body);
    if old.is_none() && !matches!(body.last(), Some(Statement::Return { .. })) {
        body.push(Statement::Return { value: None }, naga::Span::UNDEFINED);
    }
    rewrite(&mut body, &mut f.expressions, new_ty, n_old, existing_mask, single, mask);
    f.body = body;
    Ok(())
}

/// The shared memory of the MSL compile: one set of options for every library.
fn compile_options() -> Retained<MTLCompileOptions> {
    let o = MTLCompileOptions::new();
    o.setLanguageVersion(MTLLanguageVersion::Version3_0);
    // The translator marks positions `@invariant`; Metal only honors it with this set.
    o.setPreserveInvariance(true);
    // Relaxed, not fast: it may reassociate and fuse (pixel shaders are ALU-bound: +9% fps on
    // an M2 over safe math) but honors NaN and Inf, which the guest's shaders rely on (SM3
    // 0 * inf = 0 selects, alpha tests on their results); fast math assumes neither occurs.
    o.setMathMode(MTLMathMode::Relaxed);
    o
}

struct StageRequest<'a> {
    stage: &'a ProgrammableStage<'a>,
    naga_stage: naga::ShaderStage,
    visibility: ShaderStage,
    topology: MTLPrimitiveTopologyClass,
    sample_mask: u32,
    vertex_buffers: &'a [crate::VertexBufferLayout<'a>],
    label: &'a str,
}

/// [`compile_stage_uncached`] once per module and [`StageKey`].
fn compile_stage(shared: &Shared, req: &StageRequest<'_>) -> std::result::Result<CompiledStage, String> {
    let key = StageKey {
        entry: req.stage.entry_point.to_string(),
        constants: req.stage.constants.iter().map(|&(k, v)| (k.to_string(), v.to_bits())).collect(),
        naga_stage: req.naga_stage,
        point: req.topology == MTLPrimitiveTopologyClass::Point,
        sample_mask: req.sample_mask,
        vertex_buffers: req.vertex_buffers.len(),
    };
    let shader = &req.stage.module.inner;
    if let Some(stage) = shader.stages.borrow().get(&key) {
        return Ok(stage.clone());
    }
    let stage = compile_stage_uncached(shared, req)?;
    shader.stages.borrow_mut().insert(key, stage.clone());
    Ok(stage)
}

fn compile_stage_uncached(shared: &Shared, req: &StageRequest<'_>) -> std::result::Result<CompiledStage, String> {
    let shader = &req.stage.module.inner;
    let (module, info) = shader.module.as_ref().map_err(|e| format!("invalid shader module {:?}: {e}", shader.label))?;
    let entry = if req.stage.entry_point.is_empty() {
        let mut eps = module.entry_points.iter().filter(|e| e.stage == req.naga_stage);
        match (eps.next(), eps.next()) {
            (Some(e), None) => e.name.clone(),
            _ => return Err(format!("{:?} has no single {:?} entry point", shader.label, req.naga_stage)),
        }
    } else {
        req.stage.entry_point.to_string()
    };
    let mut constants = naga::back::PipelineConstants::default();
    for &(k, v) in req.stage.constants {
        constants.insert(k.to_string(), v);
    }
    let (module, info) =
        naga::back::pipeline_constants::process_overrides(module, info, Some((req.naga_stage, &entry)), &constants)
            .map_err(|e| format!("overrides: {e:?}"))?;
    let ep_index = module
        .entry_points
        .iter()
        .position(|e| e.stage == req.naga_stage && e.name == entry)
        .ok_or_else(|| format!("entry point {entry:?} not found in {:?}", shader.label))?;
    let (module, info) = if req.naga_stage == naga::ShaderStage::Fragment && req.sample_mask != !0 {
        let mut m = module.into_owned();
        add_sample_mask(&mut m, ep_index, req.sample_mask)?;
        let i = validate(&m).map_err(|e| format!("sample mask rewrite: {e}"))?;
        (std::borrow::Cow::Owned(m), std::borrow::Cow::Owned(i))
    } else {
        (module, info)
    };
    let ep = &module.entry_points[ep_index];
    let ep_info = info.get_entry_point(ep_index);

    // Slots for the resources the entry point uses, in (group, binding) order.
    let mut used: Vec<(naga::ResourceBinding, naga::Handle<naga::GlobalVariable>)> = module
        .global_variables
        .iter()
        .filter(|(h, v)| v.binding.is_some() && !ep_info[*h].is_empty())
        .map(|(h, v)| (v.binding.unwrap(), h))
        .collect();
    used.sort_by_key(|(b, _)| (b.group, b.binding));
    let mut bindings = StageBindings::default();
    let mut map = msl::EntryPointResources::default();
    let mut entries = Vec::new();
    for (rb, h) in &used {
        let var = &module.global_variables[*h];
        let mut target = msl::BindTarget::default();
        let slot = |list: &mut Vec<Slot>| {
            let index = list.len();
            list.push(Slot { group: rb.group, binding: rb.binding, index });
            index as u8
        };
        match module.types[var.ty].inner {
            naga::TypeInner::Image { class, .. } => {
                target.texture = Some(slot(&mut bindings.textures));
                target.mutable = matches!(class, naga::ImageClass::Storage { access, .. } if access.contains(naga::StorageAccess::STORE));
            }
            naga::TypeInner::Sampler { .. } => {
                target.sampler = Some(msl::BindSamplerTarget::Resource(slot(&mut bindings.samplers)));
            }
            naga::TypeInner::BindingArray { .. } => return Err("binding arrays are not supported".into()),
            _ => {
                target.buffer = Some(slot(&mut bindings.buffers));
                target.mutable =
                    matches!(var.space, naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE));
            }
        }
        bindings.groups |= 1 << rb.group;
        map.resources.insert(*rb, target);
        if let Some(e) = layout_entry(&module, var, req.visibility) {
            entries.push((rb.group, e));
        }
    }
    let max_samplers = 16;
    if bindings.samplers.len() > max_samplers {
        return Err(format!("{} samplers in one stage (Metal allows {max_samplers})", bindings.samplers.len()));
    }
    // `_mslBufferSizes` lists every global containing a runtime-sized array, in declaration order.
    let sized: Vec<(u32, u32)> = module
        .global_variables
        .iter()
        .filter(|(_, v)| module.types[v.ty].inner.needs_host_buffer_byte_size(&module.types))
        .filter_map(|(_, v)| v.binding.map(|b| (b.group, b.binding)))
        .collect();
    if !sized.is_empty() {
        let slot = bindings.buffers.len();
        map.sizes_buffer = Some(slot as u8);
        bindings.sizes = Some((slot, sized));
    }
    let resource_buffers = bindings.buffers.len() + usize::from(bindings.sizes.is_some());
    if resource_buffers + req.vertex_buffers.len() > 31 {
        return Err(format!("{resource_buffers} buffers plus {} vertex buffers exceed Metal's 31", req.vertex_buffers.len()));
    }
    for (h, var) in module.global_variables.iter() {
        if var.space == naga::AddressSpace::WorkGroup && !ep_info[h].is_empty() {
            bindings.workgroup_memory.push(module.types[var.ty].inner.size(module.to_ctx()) as usize);
        }
    }

    let bounds = naga::proc::BoundsCheckPolicy::Restrict;
    let options = msl::Options {
        lang_version: (3, 0),
        per_entry_point_map: msl::EntryPointResourceMap::from([(entry.clone(), map)]),
        inline_samplers: Vec::new(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: bounds,
            buffer: bounds,
            image_load: bounds,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: true,
        // Every shader here is ours (the renderer's, or Xenos microcode translated with
        // bounded loops: counts of at most 255 and an iteration limit on the dispatch loop).
        // Bounding each loop with a counter kept the compiler from unrolling constant loops.
        force_loop_bounding: false,
        task_dispatch_limits: None,
        mesh_shader_primitive_indices_clamp: false,
        emit_int_div_checks: true,
        ray_query_initialization_tracking: false,
    };
    let pipeline_options = msl::PipelineOptions {
        entry_point: Some((req.naga_stage, entry.clone())),
        allow_and_force_point_size: req.topology == MTLPrimitiveTopologyClass::Point,
        // Vertex attributes come through Metal's stage-in (the vertex descriptor).
        vertex_pulling_transform: false,
        vertex_buffer_mappings: Vec::new(),
        binding_array_length_map: Default::default(),
    };
    let (source, translation) =
        msl::write_string(&module, &info, &options, &pipeline_options).map_err(|e| format!("MSL generation: {e:?}"))?;
    // naga's MSL writer declares depth outputs `[[depth(any)]]` whatever the entry point's
    // `@early_depth_test` says; Metal keeps early depth testing for a declared direction.
    let source = match ep.early_depth_test {
        Some(naga::EarlyDepthTest::Allow { conservative: naga::ConservativeDepth::LessEqual }) => {
            source.replace("[[depth(any)]]", "[[depth(less)]]")
        }
        Some(naga::EarlyDepthTest::Allow { conservative: naga::ConservativeDepth::GreaterEqual }) => {
            source.replace("[[depth(any)]]", "[[depth(greater)]]")
        }
        _ => source,
    };
    let name = translation.entry_point_names.first().cloned().unwrap_or_else(|| Err(msl::EntryPointError::MissingBinding(Default::default())))
        .map_err(|e| format!("MSL entry point: {e:?}"))?;
    dump_msl(req.label, &shader.label, &entry, &source);
    let library = shared
        .raw
        .newLibraryWithSource_options_error(&ns(&source), Some(&compile_options()))
        .map_err(|e| {
            log::debug!("MSL that failed to compile:\n{source}");
            format!("Metal compile of {:?} ({entry}): {}", shader.label, e.localizedDescription())
        })?;
    set_label!(library, &shader.label);
    let function = library.newFunctionWithName(&ns(&name)).ok_or_else(|| format!("function {name} missing from the library"))?;
    Ok(CompiledStage { function, bindings, workgroup_size: ep.workgroup_size, entries })
}

/// Writes the MSL next to the shader dumps when [`crate::MSL_DUMP_DIR`] is set.
fn dump_msl(pipeline: &str, module: &str, entry: &str, source: &str) {
    let Some(dir) = crate::MSL_DUMP_DIR.lock().unwrap().clone() else { return };
    let clean = |s: &str| s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>();
    let mut name = format!("{}_{}_{}", clean(pipeline), clean(module), clean(entry));
    name.truncate(150);
    let mut hash = 0xcbf29ce484222325u64;
    for b in source.bytes() {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    let _ = write!(name, "_{:08x}.metal", hash as u32);
    let path = dir.join(name);
    if !path.exists() {
        let _ = std::fs::create_dir_all(&dir);
        if let Err(e) = std::fs::write(&path, source) {
            log::warn!("cannot write {}: {e}", path.display());
        }
    }
}

/// Render state applied with the pipeline.
pub(crate) struct RenderState {
    pub(crate) raw: Id<dyn MTLRenderPipelineState>,
    pub(crate) depth_stencil: Option<Id<dyn MTLDepthStencilState>>,
    pub(crate) vs: StageBindings,
    pub(crate) fs: StageBindings,
    /// First Metal buffer slot of the vertex buffers.
    pub(crate) vertex_buffer_base: usize,
    pub(crate) primitive: MTLPrimitiveType,
    pub(crate) cull: MTLCullMode,
    pub(crate) winding: MTLWinding,
    pub(crate) depth_clip: MTLDepthClipMode,
    pub(crate) depth_bias: (f32, f32, f32),
}

pub(crate) struct RenderPipelineInner {
    /// `None` when creation failed (reported); draws with it are skipped.
    pub(crate) state: Option<RenderState>,
    entries: Vec<(u32, BindGroupLayoutEntry)>,
}

#[derive(Clone)]
pub struct RenderPipeline {
    pub(crate) inner: Rc<RenderPipelineInner>,
}

rc_handle!(RenderPipeline);

pub(crate) struct ComputeState {
    pub(crate) raw: Id<dyn MTLComputePipelineState>,
    pub(crate) bindings: StageBindings,
    pub(crate) workgroup_size: MTLSize,
}

pub(crate) struct ComputePipelineInner {
    pub(crate) state: Option<ComputeState>,
    entries: Vec<(u32, BindGroupLayoutEntry)>,
}

#[derive(Clone)]
pub struct ComputePipeline {
    pub(crate) inner: Rc<ComputePipelineInner>,
}

rc_handle!(ComputePipeline);

fn derived_layout(entries: &[(u32, BindGroupLayoutEntry)], index: u32) -> BindGroupLayout {
    let mut out: Vec<BindGroupLayoutEntry> = Vec::new();
    for (g, e) in entries.iter().filter(|(g, _)| *g == index) {
        let _ = g;
        match out.iter_mut().find(|o| o.binding == e.binding) {
            Some(o) => o.visibility = o.visibility | e.visibility,
            None => out.push(*e),
        }
    }
    out.sort_by_key(|e| e.binding);
    BindGroupLayout { inner: Rc::new(super::resource::LayoutInner { entries: out }) }
}

impl RenderPipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        derived_layout(&self.inner.entries, index)
    }
}

impl ComputePipeline {
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        derived_layout(&self.inner.entries, index)
    }
}

fn depth_stencil_state(shared: &Shared, ds: &crate::DepthStencilState) -> Id<dyn MTLDepthStencilState> {
    let key: DepthStencilKey = [
        ds.depth_compare.0,
        u32::from(ds.depth_write_enabled),
        ds.stencil_front.compare.0,
        ds.stencil_front.fail_op.0 | ds.stencil_front.depth_fail_op.0 << 8 | ds.stencil_front.pass_op.0 << 16,
        ds.stencil_back.compare.0,
        ds.stencil_back.fail_op.0 | ds.stencil_back.depth_fail_op.0 << 8 | ds.stencil_back.pass_op.0 << 16,
        ds.stencil_read_mask,
        ds.stencil_write_mask,
        u32::from(conv::has_stencil(ds.format)),
        u32::from(conv::has_depth(ds.format)),
        0,
        0,
    ];
    if let Some(s) = shared.depth_stencil_states.borrow().get(&key) {
        return s.clone();
    }
    let d = MTLDepthStencilDescriptor::new();
    if conv::has_depth(ds.format) {
        d.setDepthCompareFunction(conv::compare(ds.depth_compare));
        d.setDepthWriteEnabled(ds.depth_write_enabled);
    }
    if conv::has_stencil(ds.format) {
        let face = |f: &crate::StencilFaceState| {
            let s = MTLStencilDescriptor::new();
            s.setStencilCompareFunction(conv::compare(f.compare));
            s.setStencilFailureOperation(conv::stencil_op(f.fail_op));
            s.setDepthFailureOperation(conv::stencil_op(f.depth_fail_op));
            s.setDepthStencilPassOperation(conv::stencil_op(f.pass_op));
            s.setReadMask(ds.stencil_read_mask);
            s.setWriteMask(ds.stencil_write_mask);
            s
        };
        d.setFrontFaceStencil(Some(&face(&ds.stencil_front)));
        d.setBackFaceStencil(Some(&face(&ds.stencil_back)));
    }
    let state = shared.raw.newDepthStencilStateWithDescriptor(&d).expect("cannot create a Metal depth/stencil state");
    shared.depth_stencil_states.borrow_mut().insert(key, state.clone());
    state
}

fn build_render(shared: &Shared, desc: &RenderPipelineDescriptor<'_>) -> std::result::Result<(RenderState, Vec<(u32, BindGroupLayoutEntry)>), String> {
    let (primitive, topology) = conv::primitive(desc.primitive.topology);
    let mask = if desc.multisample.count > 1 { desc.multisample.mask | !((1u32 << desc.multisample.count) - 1) } else { !0 };
    let vs = compile_stage(
        shared,
        &StageRequest {
            stage: &desc.vertex,
            naga_stage: naga::ShaderStage::Vertex,
            visibility: ShaderStage::Vertex,
            topology,
            sample_mask: !0,
            vertex_buffers: desc.vertex_buffers,
            label: desc.label,
        },
    )?;
    let fs = match &desc.fragment {
        Some(f) => Some(compile_stage(
            shared,
            &StageRequest {
                stage: &f.stage,
                naga_stage: naga::ShaderStage::Fragment,
                visibility: ShaderStage::Fragment,
                topology,
                sample_mask: mask,
                vertex_buffers: &[],
                label: desc.label,
            },
        )?),
        None => None,
    };
    let d = MTLRenderPipelineDescriptor::new();
    set_label!(d, desc.label);
    d.setVertexFunction(Some(&vs.function));
    // A depth-only pipeline with a partial sample mask still needs a fragment function to apply it.
    let mask_only = if fs.is_none() && mask != !0 { Some(mask_only_function(shared, mask)?) } else { None };
    if let Some(f) = fs.as_ref().map(|f| &f.function).or(mask_only.as_ref()) {
        d.setFragmentFunction(Some(f));
    }
    let vertex_buffer_base = vs.bindings.buffers.len() + usize::from(vs.bindings.sizes.is_some());
    if !desc.vertex_buffers.is_empty() {
        let vd = MTLVertexDescriptor::new();
        for (i, b) in desc.vertex_buffers.iter().enumerate() {
            let slot = vertex_buffer_base + i;
            let layout = unsafe { vd.layouts().objectAtIndexedSubscript(slot) };
            if b.array_stride == 0 {
                // Every vertex reads the same element.
                let size = b.attributes.iter().map(|a| a.offset + conv::vertex_format(a.format).1).max().unwrap_or(4);
                unsafe { layout.setStride(size.next_multiple_of(4) as usize); }
                layout.setStepFunction(MTLVertexStepFunction::Constant);
                unsafe { layout.setStepRate(0); }
            } else {
                unsafe { layout.setStride(b.array_stride as usize); }
                layout.setStepFunction(conv::step_function(b.step_mode));
                unsafe { layout.setStepRate(1); }
            }
            for a in b.attributes {
                let attr = unsafe { vd.attributes().objectAtIndexedSubscript(a.shader_location as usize) };
                attr.setFormat(conv::vertex_format(a.format).0);
                unsafe { attr.setOffset(a.offset as usize); }
                unsafe { attr.setBufferIndex(slot); }
            }
        }
        d.setVertexDescriptor(Some(&vd));
    }
    if let Some(f) = &desc.fragment {
        for (i, t) in f.targets.iter().enumerate() {
            if t.format == TextureFormat::Undefined {
                continue;
            }
            let ca = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
            ca.setPixelFormat(conv::pixel_format(t.format).ok_or_else(|| format!("color format {:?}", t.format))?);
            ca.setWriteMask(conv::write_mask(t.write_mask));
            if let Some(b) = t.blend {
                ca.setBlendingEnabled(true);
                ca.setRgbBlendOperation(conv::blend_op(b.color.operation));
                ca.setSourceRGBBlendFactor(conv::blend_factor(b.color.src_factor));
                ca.setDestinationRGBBlendFactor(conv::blend_factor(b.color.dst_factor));
                ca.setAlphaBlendOperation(conv::blend_op(b.alpha.operation));
                ca.setSourceAlphaBlendFactor(conv::blend_factor(b.alpha.src_factor));
                ca.setDestinationAlphaBlendFactor(conv::blend_factor(b.alpha.dst_factor));
            }
        }
    }
    let mut depth_stencil = None;
    let mut depth_bias = (0.0, 0.0, 0.0);
    if let Some(ds) = &desc.depth_stencil {
        let pf = conv::pixel_format(ds.format).ok_or_else(|| format!("depth format {:?}", ds.format))?;
        if conv::has_depth(ds.format) {
            d.setDepthAttachmentPixelFormat(pf);
        }
        if conv::has_stencil(ds.format) {
            d.setStencilAttachmentPixelFormat(pf);
        }
        depth_stencil = Some(depth_stencil_state(shared, ds));
        depth_bias = (ds.depth_bias as f32, ds.depth_bias_slope_scale, ds.depth_bias_clamp);
    }
    d.setRasterSampleCount(desc.multisample.count.max(1) as usize);
    d.setAlphaToCoverageEnabled(desc.multisample.alpha_to_coverage_enabled);
    unsafe { d.setInputPrimitiveTopology(topology) };
    let raw = shared
        .raw
        .newRenderPipelineStateWithDescriptor_error(&d)
        .map_err(|e| format!("Metal render pipeline: {}", e.localizedDescription()))?;
    let mut entries = vs.entries;
    let fs_bindings = match fs {
        Some(f) => {
            entries.extend(f.entries);
            f.bindings
        }
        None => StageBindings::default(),
    };
    Ok((
        RenderState {
            raw,
            depth_stencil,
            vs: vs.bindings,
            fs: fs_bindings,
            vertex_buffer_base,
            primitive,
            cull: conv::cull(desc.primitive.cull_mode),
            winding: conv::winding(desc.primitive.front_face),
            depth_clip: if desc.primitive.unclipped_depth { MTLDepthClipMode::Clamp } else { MTLDepthClipMode::Clip },
            depth_bias,
        },
        entries,
    ))
}

/// A fragment function that only outputs `mask` as the coverage mask.
fn mask_only_function(shared: &Shared, mask: u32) -> std::result::Result<Id<dyn MTLFunction>, String> {
    let source = format!(
        "#include <metal_stdlib>\nstruct XeMaskOnly {{ uint mask [[sample_mask]]; }};\n\
         fragment XeMaskOnly xe_mask_only() {{ return XeMaskOnly {{ {mask}u }}; }}\n"
    );
    let library = shared
        .raw
        .newLibraryWithSource_options_error(&ns(&source), Some(&compile_options()))
        .map_err(|e| format!("mask-only fragment: {}", e.localizedDescription()))?;
    library.newFunctionWithName(&ns("xe_mask_only")).ok_or_else(|| "mask-only function missing".to_string())
}

impl Device {
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> RenderPipeline {
        crate::counters::add(&crate::counters::count::pipelines, 1);
        match self.try_render_pipeline(desc) {
            Ok(p) => p,
            Err(e) => {
                self.shared.report(ErrorType::Validation, format!("render pipeline {:?}: {e}", desc.label));
                RenderPipeline { inner: Rc::new(RenderPipelineInner { state: None, entries: Vec::new() }) }
            }
        }
    }

    fn try_render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> std::result::Result<RenderPipeline, String> {
        let (state, entries) = objc2::rc::autoreleasepool(|_| build_render(&self.shared, desc))?;
        Ok(RenderPipeline { inner: Rc::new(RenderPipelineInner { state: Some(state), entries }) })
    }

    /// Compiles synchronously; failures come back as the result instead of an error report.
    pub fn create_render_pipeline_async(&self, desc: &RenderPipelineDescriptor<'_>) -> Pending<Result<RenderPipeline>> {
        crate::counters::add(&crate::counters::count::pipelines, 1);
        Pending::ready(self.try_render_pipeline(desc).map_err(|e| Error::Invalid(format!("pipeline creation failed: {e}"))))
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor<'_>) -> ComputePipeline {
        crate::counters::add(&crate::counters::count::pipelines, 1);
        let built = objc2::rc::autoreleasepool(|_| -> std::result::Result<_, String> {
            let cs = compile_stage(
                &self.shared,
                &StageRequest {
                    stage: &desc.stage,
                    naga_stage: naga::ShaderStage::Compute,
                    visibility: ShaderStage::Compute,
                    topology: MTLPrimitiveTopologyClass::Unspecified,
                    sample_mask: !0,
                    vertex_buffers: &[],
                    label: desc.label,
                },
            )?;
            let d = MTLComputePipelineDescriptor::new();
            set_label!(d, desc.label);
            d.setComputeFunction(Some(&cs.function));
            let raw = self
                .shared
                .raw
                .newComputePipelineStateWithDescriptor_options_reflection_error(&d, MTLPipelineOption::None, None)
                .map_err(|e| format!("Metal compute pipeline: {}", e.localizedDescription()))?;
            let [x, y, z] = cs.workgroup_size;
            Ok((
                ComputeState {
                    raw,
                    bindings: cs.bindings,
                    workgroup_size: MTLSize { width: x as usize, height: y as usize, depth: z as usize },
                },
                cs.entries,
            ))
        });
        match built {
            Ok((state, entries)) => ComputePipeline { inner: Rc::new(ComputePipelineInner { state: Some(state), entries }) },
            Err(e) => {
                self.shared.report(ErrorType::Validation, format!("compute pipeline {:?}: {e}", desc.label));
                ComputePipeline { inner: Rc::new(ComputePipelineInner { state: None, entries: Vec::new() }) }
            }
        }
    }
}
