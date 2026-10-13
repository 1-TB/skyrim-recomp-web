//! WGSL generation.
//!
//! The microcode is translated into `fn xe_ucode()` operating on private variables (registers,
//! predicate, address registers, loop and call stacks, exports). Shaders without loops, calls and
//! jumps are emitted as straight-line code; otherwise control flow uses a dispatch loop over a
//! `switch` on the control flow instruction index (`xe_pc`), with a case per branch target that
//! falls through to the next one, like Xenia's SPIR-V translator.

mod alu;
mod fetch;
pub(crate) mod prelude;
mod structure;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::analysis::ShaderInfo;
use crate::bindings::{self, TEXTURE_GROUP, flags};
use crate::parsed::{
    AddressingMode, AnisoFilter, CfCondition, FetchDimension, InstructionOperand, InstructionResult, ParsedCf,
    ParsedExec, StorageSource, StorageTarget, SwizzleSource, TextureFilter,
};
use crate::program::{Instruction, Program};
use crate::{ShaderStage, TranslationOptions, VertexExpansion};

/// Kind of WGSL texture declared for a texture binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TextureBindingDimension {
    /// `texture_2d<f32>` (tfetch1D and tfetch2D; 1D textures are 2D textures of height 1).
    D2,
    /// `texture_3d<f32>` at `texture_binding` plus `texture_2d_array<f32>` at
    /// `stacked_texture_binding` (tfetch3D; the fetch constant's dimension selects one at runtime,
    /// the other must be bound to any compatible dummy view).
    D3OrStacked,
    /// `texture_cube<f32>`.
    Cube,
}

/// A texture + sampler pair in `@group(2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureBinding {
    /// Texture fetch constant 0..31 (dwords `6 * fetch_constant ..` of the fetch constants).
    pub fetch_constant: u32,
    pub dimension: TextureBindingDimension,
    pub texture_binding: u32,
    pub stacked_texture_binding: Option<u32>,
    pub sampler_binding: u32,
    /// Filter overrides from the fetch instruction (`UseFetchConst` = take from the fetch constant).
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub mip_filter: TextureFilter,
    pub aniso_filter: AnisoFilter,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TextureKey {
    fetch_constant: u32,
    dimension: TextureBindingDimension,
    mag: TextureFilter,
    min: TextureFilter,
    mip: TextureFilter,
    aniso: AnisoFilter,
}

/// Value written by a store.
#[derive(Clone, Copy)]
enum Value<'a> {
    /// A vec4<f32> expression (identifier).
    Vec(&'a str),
    /// An f32 replicated into every component.
    Scalar(&'a str),
    /// Only constant 0/1 components are written.
    None,
}

pub(crate) struct Output {
    pub wgsl: String,
    pub entry_point: &'static str,
    pub texture_bindings: Vec<TextureBinding>,
    pub register_count: u32,
    pub vs_output_interpolators: u32,
    pub ps_input_interpolators: u32,
    pub uses_shared_memory: bool,
    pub structured_control_flow: bool,
    pub warnings: Vec<String>,
}

/// Maximum iterations of the control flow dispatch loop before the shader gives up (protects the
/// host GPU from hangs caused by malformed microcode).
const DISPATCH_ITERATION_LIMIT: u32 = 1 << 20;

pub(crate) struct Generator<'a> {
    program: &'a Program,
    info: &'a ShaderInfo,
    options: &'a TranslationOptions,
    stage: ShaderStage,
    body: String,
    indent: usize,
    register_count: u32,
    texture_keys: BTreeMap<TextureKey, usize>,
    texture_bindings: Vec<TextureBinding>,
    next_texture_binding: u32,
    uses_shared_memory: bool,
    warnings: BTreeSet<String>,
    temp_counter: u32,
    structured: bool,
}

/// Formats an f32 as a WGSL literal that round-trips exactly.
pub(crate) fn lit(v: f32) -> String {
    if v.is_finite() {
        if v == v.trunc() && v.abs() < 1e9 {
            if v == 0.0 && v.is_sign_negative() {
                return "(-0.0)".to_string();
            }
            return if v < 0.0 { format!("({v:.1})") } else { format!("{v:.1}") };
        }
        let s = format!("{v:?}");
        if s.parse::<f32>() == Ok(v) && v.abs() < 1e30 && v.abs() > 1e-30 {
            return if v < 0.0 { format!("({s}f)") } else { format!("{s}f") };
        }
    }
    format!("bitcast<f32>({:#010x}u)", v.to_bits())
}

const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];

impl<'a> Generator<'a> {
    pub(crate) fn new(program: &'a Program, info: &'a ShaderInfo, options: &'a TranslationOptions) -> Self {
        let mut register_count = info.register_static_address_bound.max(1);
        if info.uses_register_dynamic_addressing {
            register_count = 64;
        }
        Generator {
            program,
            info,
            options,
            stage: program.stage,
            body: String::new(),
            indent: 1,
            register_count,
            texture_keys: BTreeMap::new(),
            texture_bindings: Vec::new(),
            next_texture_binding: options.texture_binding_base,
            uses_shared_memory: false,
            warnings: BTreeSet::new(),
            temp_counter: 0,
            structured: false,
        }
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.body.push_str("    ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn open(&mut self, text: &str) {
        self.line(text);
        self.indent += 1;
    }

    fn close(&mut self) {
        self.indent -= 1;
        self.line("}");
    }

    fn warn(&mut self, message: String) {
        self.warnings.insert(message);
    }

    fn temp(&mut self, prefix: &str) -> String {
        self.temp_counter += 1;
        format!("{prefix}{}", self.temp_counter)
    }

    // ---- Operands -------------------------------------------------------------------------

    fn register_index(&self, index: u32, addressing: AddressingMode) -> String {
        match addressing {
            AddressingMode::Absolute => format!("{}", index.min(self.register_count - 1)),
            AddressingMode::LoopRelative => format!("(u32({index} + xe_aL.x) & 63u)"),
            AddressingMode::AddressRegisterRelative => format!("(u32({index} + xe_a0) & 63u)"),
        }
    }

    /// Unswizzled vec4 storage of an operand.
    fn operand_storage(&self, op: &InstructionOperand) -> String {
        match op.source {
            StorageSource::Register => format!("xe_r[{}]", self.register_index(op.index, op.addressing)),
            StorageSource::ConstantFloat => match op.addressing {
                AddressingMode::Absolute => format!("xe_float_constants[{}u]", op.index & 255),
                AddressingMode::AddressRegisterRelative => {
                    format!("xe_float_constant({} + xe_a0)", op.index)
                }
                AddressingMode::LoopRelative => {
                    format!("xe_float_constant({} + xe_aL.x)", op.index)
                }
            },
            StorageSource::VertexFetchConstant | StorageSource::TextureFetchConstant => "vec4<f32>(0.0)".to_string(),
        }
    }

    fn apply_modifiers(op: &InstructionOperand, mut e: String) -> String {
        if op.is_absolute_value {
            e = format!("abs({e})");
        }
        if op.is_negated {
            e = format!("(-{e})");
        }
        e
    }

    /// Full 4-component swizzled operand with modifiers.
    fn operand_vec4(&self, op: &InstructionOperand) -> String {
        let storage = self.operand_storage(op);
        let e = if op.is_standard_swizzle() {
            storage
        } else {
            let swizzle: String = (0..4).map(|i| op.component(i).char()).collect();
            format!("{storage}.{swizzle}")
        };
        Self::apply_modifiers(op, e)
    }

    /// One component (operand component slot `i`) of an operand with modifiers.
    fn operand_scalar(&self, op: &InstructionOperand, i: u32) -> String {
        let storage = self.operand_storage(op);
        let c = op.component(i);
        let e = match c.component() {
            Some(c) => format!("{storage}.{}", COMPONENTS[c as usize]),
            None => "0.0".to_string(),
        };
        Self::apply_modifiers(op, e)
    }

    // ---- Results --------------------------------------------------------------------------

    fn store(&mut self, result: &InstructionResult, value: Value) {
        let mask = result.used_write_mask();
        if mask == 0 {
            return;
        }
        let has_value_components = (0..4).any(|i| mask & (1 << i) != 0 && result.components[i].component().is_some());
        let mut clamped = String::new();
        if result.is_clamped && has_value_components {
            match value {
                Value::Vec(v) => {
                    clamped = self.temp("xe_c");
                    self.line(&format!("let {clamped} = clamp({v}, vec4<f32>(0.0), vec4<f32>(1.0));"));
                }
                Value::Scalar(v) => {
                    clamped = self.temp("xe_c");
                    self.line(&format!("let {clamped} = clamp({v}, 0.0, 1.0);"));
                }
                Value::None => {}
            }
        }
        let value = match value {
            Value::Vec(_) if !clamped.is_empty() => Value::Vec(&clamped),
            Value::Scalar(_) if !clamped.is_empty() => Value::Scalar(&clamped),
            other => other,
        };
        let component_expr = |i: usize| -> String {
            match result.components[i] {
                SwizzleSource::Zero => "0.0".to_string(),
                SwizzleSource::One => "1.0".to_string(),
                c => match value {
                    Value::Vec(v) => format!("{v}.{}", c.char()),
                    Value::Scalar(v) => v.to_string(),
                    Value::None => "0.0".to_string(),
                },
            }
        };
        let target = match result.target {
            StorageTarget::None => return,
            StorageTarget::Register => {
                let index = self.register_index(result.index, result.addressing);
                if result.addressing == AddressingMode::Absolute {
                    format!("xe_r[{index}]")
                } else {
                    let t = self.temp("xe_i");
                    self.line(&format!("let {t} = {index};"));
                    format!("xe_r[{t}]")
                }
            }
            StorageTarget::Interpolator => format!("xe_o[{}]", result.index & 15),
            StorageTarget::Position => "xe_out_position".to_string(),
            StorageTarget::PointSizeEdgeFlagKillVertex => "xe_out_point".to_string(),
            StorageTarget::ExportAddress => "xe_export_address".to_string(),
            StorageTarget::ExportData => format!("xe_export_data[{}]", result.index.min(4)),
            StorageTarget::Color => format!("xe_out_color[{}]", result.index & 3),
            StorageTarget::Depth => {
                let v = component_expr(0);
                if result.is_clamped {
                    self.line(&format!("xe_out_depth = {v};"));
                } else {
                    self.line(&format!("xe_out_depth = clamp({v}, 0.0, 1.0);"));
                }
                return;
            }
        };
        for (i, name) in COMPONENTS.iter().enumerate() {
            if mask & (1 << i) == 0 {
                continue;
            }
            let mut v = component_expr(i);
            if result.target == StorageTarget::PointSizeEdgeFlagKillVertex && i == 0 {
                v = format!("xe_clamp_point_size({v})");
            }
            self.line(&format!("{target}.{name} = {v};"));
        }
    }

    // ---- Control flow ---------------------------------------------------------------------

    fn condition_expr(condition: &CfCondition) -> Option<String> {
        match condition {
            CfCondition::Always => None,
            CfCondition::Bool { index, value } => Some(format!("{}xe_bool({index}u)", if *value { "" } else { "!" })),
            CfCondition::Predicate(value) => Some(if *value { "xe_p0".to_string() } else { "!xe_p0".to_string() }),
        }
    }

    fn predicated(&mut self, is_predicated: bool, condition: bool) -> bool {
        if is_predicated {
            self.open(&format!("if ({}xe_p0) {{", if condition { "" } else { "!" }));
        }
        is_predicated
    }

    fn emit_exec(&mut self, exec: &ParsedExec, entry_index: usize) -> bool {
        let condition = Self::condition_expr(&exec.condition);
        if let Some(c) = &condition {
            self.open(&format!("if ({c}) {{"));
        }
        let program: &'a Program = self.program;
        for instruction in &program.cf[entry_index].instructions {
            self.line(&format!("// {}", one_line_disassembly(&instruction.instruction)));
            match &instruction.instruction {
                Instruction::Alu(alu) => self.emit_alu(alu),
                Instruction::VertexFetch(vf) => self.emit_vertex_fetch(vf),
                Instruction::TextureFetch(tf) => self.emit_texture_fetch(tf),
            }
        }
        if exec.is_end {
            self.line("return;");
        }
        if condition.is_some() {
            self.close();
            false
        } else {
            exec.is_end
        }
    }

    /// Emits one control flow instruction. Returns whether control never continues to the next
    /// instruction.
    fn emit_cf(&mut self, entry_index: usize, dispatch: bool) -> bool {
        let program: &'a Program = self.program;
        let entry = &program.cf[entry_index];
        let index = entry.index;
        let mut cf_disasm = String::new();
        crate::disasm::control_flow(&mut cf_disasm, &entry.cf, self.stage);
        self.line(&format!("// {:4}.{} {}", index / 2, index & 1, cf_disasm.trim()));
        match entry.cf {
            ParsedCf::Nop | ParsedCf::Alloc { .. } | ParsedCf::MarkVsFetchDone => false,
            ParsedCf::Exec(exec) => self.emit_exec(&exec, entry_index),
            ParsedCf::LoopStart { loop_constant, is_repeat, skip_address } => {
                debug_assert!(dispatch);
                self.open("{");
                self.line(&format!("let xe_lc = xe_loop_constant({loop_constant}u);"));
                self.open("if ((xe_lc & 0xFFu) == 0u) {");
                self.line(&format!("xe_pc = {skip_address}u;"));
                self.line("break;");
                self.close();
                self.line("xe_loop_count = vec4<u32>(xe_lc & 0xFFu, xe_loop_count.xyz);");
                if is_repeat {
                    self.line("xe_aL = vec4<i32>(xe_aL.x, xe_aL.xyz);");
                } else {
                    self.line("xe_aL = vec4<i32>(i32((xe_lc >> 8u) & 0xFFu), xe_aL.xyz);");
                }
                self.close();
                false
            }
            ParsedCf::LoopEnd { loop_constant, body_address, predicated_break } => {
                debug_assert!(dispatch);
                self.open("{");
                self.line("let xe_remaining = xe_loop_count.x - 1u;");
                match predicated_break {
                    Some(c) => self.line(&format!(
                        "let xe_continue = xe_remaining != 0u && xe_p0 != {};",
                        if c { "true" } else { "false" }
                    )),
                    None => self.line("let xe_continue = xe_remaining != 0u;"),
                }
                self.open("if (xe_continue) {");
                self.line("xe_loop_count.x = xe_remaining;");
                self.line(&format!(
                    "xe_aL.x = xe_aL.x + extractBits(bitcast<i32>(xe_loop_constant({loop_constant}u)), 16u, 8u);"
                ));
                self.line(&format!("xe_pc = {body_address}u;"));
                self.line("break;");
                self.close();
                self.line("xe_loop_count = vec4<u32>(xe_loop_count.yzw, 0u);");
                self.line("xe_aL = vec4<i32>(xe_aL.yzw, 0);");
                self.close();
                false
            }
            ParsedCf::Call { target, condition } => {
                debug_assert!(dispatch);
                let mut c = "xe_call_depth < 4u".to_string();
                if let Some(cond) = Self::condition_expr(&condition) {
                    c = format!("({cond}) && {c}");
                }
                self.open(&format!("if ({c}) {{"));
                self.line(&format!("xe_call_stack[xe_call_depth] = {}u;", index + 1));
                self.line("xe_call_depth = xe_call_depth + 1u;");
                self.line(&format!("xe_pc = {target}u;"));
                self.line("break;");
                self.close();
                false
            }
            ParsedCf::Return => {
                debug_assert!(dispatch);
                self.open("if (xe_call_depth != 0u) {");
                self.line("xe_call_depth = xe_call_depth - 1u;");
                self.line("xe_pc = xe_call_stack[xe_call_depth];");
                self.line("break;");
                self.close();
                false
            }
            ParsedCf::Jump { target, condition } => {
                debug_assert!(dispatch);
                match Self::condition_expr(&condition) {
                    Some(c) => {
                        self.open(&format!("if ({c}) {{"));
                        self.line(&format!("xe_pc = {target}u;"));
                        self.line("break;");
                        self.close();
                        false
                    }
                    None => {
                        self.line(&format!("xe_pc = {target}u;"));
                        self.line("break;");
                        true
                    }
                }
            }
        }
    }

    fn emit_nodes(&mut self, nodes: &[structure::Node]) -> bool {
        for node in nodes {
            let terminated = match node {
                structure::Node::Cf(i) => self.emit_cf(*i, false),
                structure::Node::If { condition, inverted, body, else_body } => {
                    let c = Self::condition_expr(condition).unwrap_or_else(|| "true".to_string());
                    let c = if *inverted { format!("!({c})") } else { c };
                    self.open(&format!("if ({c}) {{"));
                    let then_terminated = self.emit_nodes(body);
                    let mut else_terminated = false;
                    if !else_body.is_empty() {
                        self.indent -= 1;
                        self.open("} else {");
                        else_terminated = self.emit_nodes(else_body);
                    }
                    self.close();
                    then_terminated && else_terminated
                }
                structure::Node::Return { condition } => match Self::condition_expr(condition) {
                    None => {
                        self.line("return;");
                        true
                    }
                    Some(c) => {
                        self.open(&format!("if ({c}) {{"));
                        self.line("return;");
                        self.close();
                        false
                    }
                },
                structure::Node::Loop { start, end, body } => {
                    self.emit_structured_loop(*start, *end, body);
                    false
                }
            };
            if terminated {
                return true;
            }
        }
        false
    }

    fn emit_structured_loop(&mut self, start: usize, end: usize, body: &[structure::Node]) {
        let program: &'a Program = self.program;
        let ParsedCf::LoopStart { loop_constant, is_repeat, .. } = program.cf[start].cf else { unreachable!() };
        let ParsedCf::LoopEnd { loop_constant: end_constant, predicated_break, .. } = program.cf[end].cf else {
            unreachable!()
        };
        let mut text = String::new();
        crate::disasm::control_flow(&mut text, &program.cf[start].cf, self.stage);
        self.line(&format!("// {:4}.{} {}", start / 2, start & 1, text.trim()));
        self.open("{");
        self.line(&format!("let xe_lc = xe_loop_constant({loop_constant}u);"));
        self.open("if ((xe_lc & 0xFFu) != 0u) {");
        self.line("xe_loop_count = vec4<u32>(xe_lc & 0xFFu, xe_loop_count.xyz);");
        if is_repeat {
            self.line("xe_aL = vec4<i32>(xe_aL.x, xe_aL.xyz);");
        } else {
            self.line("xe_aL = vec4<i32>(i32((xe_lc >> 8u) & 0xFFu), xe_aL.xyz);");
        }
        self.open("loop {");
        let terminated = self.emit_nodes(body);
        if !terminated {
            let mut text = String::new();
            crate::disasm::control_flow(&mut text, &program.cf[end].cf, self.stage);
            self.line(&format!("// {:4}.{} {}", end / 2, end & 1, text.trim()));
            self.line("let xe_remaining = xe_loop_count.x - 1u;");
            match predicated_break {
                Some(c) => self.line(&format!("let xe_continue = xe_remaining != 0u && xe_p0 != {c};")),
                None => self.line("let xe_continue = xe_remaining != 0u;"),
            }
            self.open("if (xe_continue) {");
            self.line("xe_loop_count.x = xe_remaining;");
            self.line(&format!(
                "xe_aL.x = xe_aL.x + extractBits(bitcast<i32>(xe_loop_constant({end_constant}u)), 16u, 8u);"
            ));
            self.line("continue;");
            self.close();
            self.line("xe_loop_count = vec4<u32>(xe_loop_count.yzw, 0u);");
            self.line("xe_aL = vec4<i32>(xe_aL.yzw, 0);");
            self.line("break;");
        }
        self.close();
        self.close();
        self.close();
    }

    fn emit_ucode(&mut self) {
        let cf_count = self.program.cf.len();
        if let Some(nodes) = structure::structure(self.program) {
            self.structured = true;
            self.emit_nodes(&nodes);
            return;
        }
        let mut cases: BTreeSet<u32> =
            self.program.labels.iter().copied().filter(|&l| (l as usize) < cf_count).collect();
        cases.insert(0);
        for entry in &self.program.cf {
            if matches!(entry.cf, ParsedCf::Call { .. }) && ((entry.index + 1) as usize) < cf_count {
                cases.insert(entry.index + 1);
            }
        }
        self.line("var xe_pc = 0u;");
        self.line("var xe_iterations = 0u;");
        self.open("loop {");
        self.line("xe_iterations = xe_iterations + 1u;");
        self.open(&format!("if (xe_iterations > {DISPATCH_ITERATION_LIMIT}u) {{"));
        self.line("return;");
        self.close();
        self.open("switch (xe_pc) {");
        let case_list: Vec<u32> = cases.iter().copied().collect();
        for (n, &start) in case_list.iter().enumerate() {
            let end = case_list.get(n + 1).copied().unwrap_or(cf_count as u32);
            self.open(&format!("case {start}u: {{"));
            let mut terminated = false;
            for i in start..end {
                if self.emit_cf(i as usize, true) {
                    terminated = true;
                    break;
                }
            }
            if !terminated {
                if (end as usize) < cf_count {
                    self.line(&format!("xe_pc = {end}u;"));
                } else {
                    self.line("return;");
                }
            }
            self.close();
        }
        self.open("default: {");
        self.line("return;");
        self.close();
        self.close();
        self.close();
    }

    // ---- Textures -------------------------------------------------------------------------

    fn texture_binding(
        &mut self,
        fetch_constant: u32,
        dimension: FetchDimension,
        attributes: &crate::parsed::TextureFetchAttributes,
    ) -> TextureBinding {
        let dimension = match dimension {
            FetchDimension::D1 | FetchDimension::D2 => TextureBindingDimension::D2,
            FetchDimension::D3OrStacked => TextureBindingDimension::D3OrStacked,
            FetchDimension::Cube => TextureBindingDimension::Cube,
        };
        let key = TextureKey {
            fetch_constant,
            dimension,
            mag: attributes.mag_filter,
            min: attributes.min_filter,
            mip: attributes.mip_filter,
            aniso: attributes.aniso_filter,
        };
        if let Some(&i) = self.texture_keys.get(&key) {
            return self.texture_bindings[i];
        }
        let texture_binding = self.next_texture_binding;
        let stacked_texture_binding =
            if dimension == TextureBindingDimension::D3OrStacked { Some(texture_binding + 1) } else { None };
        let sampler_binding = texture_binding + 1 + stacked_texture_binding.is_some() as u32;
        self.next_texture_binding = sampler_binding + 1;
        let binding = TextureBinding {
            fetch_constant,
            dimension,
            texture_binding,
            stacked_texture_binding,
            sampler_binding,
            mag_filter: attributes.mag_filter,
            min_filter: attributes.min_filter,
            mip_filter: attributes.mip_filter,
            aniso_filter: attributes.aniso_filter,
        };
        self.texture_keys.insert(key, self.texture_bindings.len());
        self.texture_bindings.push(binding);
        binding
    }

    // ---- Module assembly ------------------------------------------------------------------

    pub(crate) fn generate(mut self) -> Output {
        self.emit_ucode();
        let mut m = String::new();
        m.push_str("// Translated from Xenos microcode by xenos-shader.\n");
        m.push_str("diagnostic(off, derivative_uniformity);\n\n");
        m.push_str(&bindings::common_declarations(self.stage));
        m.push_str(prelude::PRELUDE);
        m.push_str(&prelude::texture_format_table());
        m.push('\n');

        for b in &self.texture_bindings {
            let (kind, name) = match b.dimension {
                TextureBindingDimension::D2 => ("texture_2d<f32>", "2d"),
                TextureBindingDimension::D3OrStacked => ("texture_3d<f32>", "3d"),
                TextureBindingDimension::Cube => ("texture_cube<f32>", "cube"),
            };
            writeln!(
                m,
                "@group({TEXTURE_GROUP}) @binding({}) var xe_texture_{}: {kind}; // tf{} {name}",
                b.texture_binding, b.texture_binding, b.fetch_constant
            )
            .unwrap();
            if let Some(s) = b.stacked_texture_binding {
                writeln!(
                    m,
                    "@group({TEXTURE_GROUP}) @binding({s}) var xe_texture_{s}: texture_2d_array<f32>; // tf{} stacked",
                    b.fetch_constant
                )
                .unwrap();
            }
            writeln!(
                m,
                "@group({TEXTURE_GROUP}) @binding({}) var xe_sampler_{}: sampler; // tf{}",
                b.sampler_binding, b.sampler_binding, b.fetch_constant
            )
            .unwrap();
        }
        if !self.texture_bindings.is_empty() {
            m.push('\n');
        }

        writeln!(m, "var<private> xe_r: array<vec4<f32>, {}>;", self.register_count).unwrap();
        m.push_str(
            "var<private> xe_ps: f32;
var<private> xe_p0: bool;
var<private> xe_a0: i32;
var<private> xe_aL: vec4<i32>;
var<private> xe_loop_count: vec4<u32>;
var<private> xe_call_stack: array<u32, 4>;
var<private> xe_call_depth: u32;
var<private> xe_vfetch_address: i32;
var<private> xe_vfetch_end: i32;
var<private> xe_tfetch_lod: f32;
var<private> xe_tfetch_gradients_h: vec3<f32>;
var<private> xe_tfetch_gradients_v: vec3<f32>;
var<private> xe_export_address: vec4<f32>;
var<private> xe_export_data: array<vec4<f32>, 5>;
",
        );
        match self.stage {
            ShaderStage::Vertex => m.push_str(
                "var<private> xe_out_position: vec4<f32>;
var<private> xe_o: array<vec4<f32>, 16>;
var<private> xe_out_point: vec3<f32> = vec3<f32>(-1.0, 0.0, 0.0);
",
            ),
            ShaderStage::Pixel => m.push_str(
                "var<private> xe_out_color: array<vec4<f32>, 4>;
var<private> xe_out_depth: f32;
",
            ),
        }
        m.push_str("\nfn xe_ucode() {\n");
        m.push_str(&self.body);
        m.push_str("}\n\n");

        let (entry_point, vs_outputs, ps_inputs) = match self.stage {
            ShaderStage::Vertex => {
                let outputs = self.vertex_entry(&mut m);
                ("xe_vs_main", outputs, 0)
            }
            ShaderStage::Pixel => {
                let inputs = self.pixel_entry(&mut m);
                ("xe_ps_main", 0, inputs)
            }
        };
        Output {
            wgsl: m,
            entry_point,
            texture_bindings: self.texture_bindings,
            register_count: self.register_count,
            vs_output_interpolators: vs_outputs,
            ps_input_interpolators: ps_inputs,
            uses_shared_memory: self.uses_shared_memory || self.stage == ShaderStage::Vertex,
            structured_control_flow: self.structured,
            warnings: self.warnings.into_iter().collect(),
        }
    }

    fn vertex_entry(&mut self, m: &mut String) -> u32 {
        let mut outputs = self.options.vs_output_interpolators & 0xFFFF;
        if let Some(location) = self.options.vs_point_size_location
            && location < 16
            && outputs & (1 << location) != 0
        {
            outputs &= !(1 << location);
            self.warn(format!("interpolator {location} not output: its location holds the point size"));
        }
        m.push_str("struct XeVertexOutput {\n    @builtin(position) position: vec4<f32>,\n");
        for i in 0..16 {
            if outputs & (1 << i) != 0 {
                writeln!(m, "    @location({i}) o{i}: vec4<f32>,").unwrap();
            }
        }
        if let Some(location) = self.options.vs_point_size_location {
            writeln!(m, "    @location({location}) point_size: f32,").unwrap();
        }
        m.push_str("}\n\n");
        let writes_kill = self.info.writes_point_size_edge_flag_kill_vertex & 0b100 != 0;
        // One guest vertex: `position` is the index position the guest vertex index is taken from.
        write!(
            m,
            "fn xe_vertex(position: u32) -> XeVertexOutput {{
    var index = position;
    if ((xe_system.flags & {load}u) != 0u) {{
        if ((xe_system.flags & {load32}u) != 0u) {{
            index = xe_shared_load((xe_system.vertex_index_load_address >> 2u) + index);
        }} else {{
            let address = xe_system.vertex_index_load_address + index * 2u;
            index = extractBits(xe_shared_load(address >> 2u), (address & 2u) * 8u, 16u);
        }}
    }}
    index = xe_swap32(index, xe_system.vertex_index_endian);
    let vertex_index = clamp(bitcast<u32>(bitcast<i32>(index) + xe_system.vertex_base_index),
                             xe_system.vertex_index_min, xe_system.vertex_index_max);
    xe_r[0].x = f32(bitcast<i32>(vertex_index));
    xe_ucode();
    var out: XeVertexOutput;
    let flags = xe_system.flags;
    var w = xe_out_position.w;
    if ((flags & {w_not_reciprocal}u) == 0u) {{
        w = 1.0 / w;
    }}
    var xy = xe_out_position.xy;
    if ((flags & {xy_div}u) != 0u) {{
        xy = xy * w;
    }}
    var z = xe_out_position.z;
    if ((flags & {z_div}u) != 0u) {{
        z = z * w;
    }}
    let xyz = vec3<f32>(xy, z) * xe_system.ndc_scale + xe_system.ndc_offset * w;
",
            load = flags::VERTEX_INDEX_LOAD,
            load32 = flags::VERTEX_INDEX_LOAD_32BIT,
            w_not_reciprocal = flags::W_NOT_RECIPROCAL,
            xy_div = flags::XY_DIVIDED_BY_W,
            z_div = flags::Z_DIVIDED_BY_W,
        )
        .unwrap();
        if writes_kill {
            // The kill flag is bits 0:30 of oPts.z; a NaN W culls the whole primitive.
            m.push_str(
                "    if ((bitcast<u32>(xe_out_point.z) & 0x7FFFFFFFu) != 0u) {
        w = bitcast<f32>(0x7FC00000u | (flags & 0u));
    }
",
            );
        }
        m.push_str("    out.position = vec4<f32>(xyz, w);\n");
        for i in 0..16 {
            if outputs & (1 << i) != 0 {
                writeln!(m, "    out.o{i} = xe_o[{i}];").unwrap();
            }
        }
        if self.options.vs_point_size_location.is_some() {
            m.push_str("    out.point_size = xe_out_point.x;\n");
        }
        m.push_str("    return out;\n}\n\n");

        match self.options.vs_expansion {
            VertexExpansion::None => m.push_str(
                "@vertex
fn xe_vs_main(@builtin(vertex_index) xe_host_vertex_index: u32) -> XeVertexOutput {
    return xe_vertex(xe_host_vertex_index);
}
",
            ),
            VertexExpansion::PointSprite => m.push_str(
                "@vertex
fn xe_vs_main(@builtin(vertex_index) xe_host_vertex_index: u32) -> XeVertexOutput {
    var out = xe_vertex(xe_host_vertex_index >> 2u);
    var diameter = xe_system.point_constant_diameter;
    if (xe_out_point.x >= 0.0) {
        diameter = vec2<f32>(clamp(xe_out_point.x, xe_system.point_vertex_diameter_min,
                                   xe_system.point_vertex_diameter_max));
    }
    // Strip corners (-,-) (+,-) (-,+) (+,+).
    let corner = xe_host_vertex_index & 3u;
    let direction = vec2<f32>(select(-1.0, 1.0, (corner & 1u) != 0u), select(-1.0, 1.0, (corner & 2u) != 0u));
    out.position = vec4<f32>(out.position.xy + direction * diameter * xe_system.point_screen_diameter_to_ndc_radius
                             * out.position.w, out.position.zw);
    return out;
}
",
            ),
            VertexExpansion::RectangleList => {
                m.push_str("fn xe_reset() {\n");
                writeln!(m, "    xe_r = array<vec4<f32>, {}>();", self.register_count).unwrap();
                m.push_str(
                    "    xe_ps = 0.0;
    xe_p0 = false;
    xe_a0 = 0;
    xe_aL = vec4<i32>();
    xe_loop_count = vec4<u32>();
    xe_call_depth = 0u;
    xe_out_position = vec4<f32>();
    xe_o = array<vec4<f32>, 16>();
    xe_out_point = vec3<f32>(-1.0, 0.0, 0.0);
}

@vertex
fn xe_vs_main(@builtin(vertex_index) xe_host_vertex_index: u32) -> XeVertexOutput {
    let first_position = (xe_host_vertex_index >> 2u) * 3u;
    var v: array<XeVertexOutput, 3>;
    for (var i = 0u; i < 3u; i++) {
        xe_reset();
        v[i] = xe_vertex(first_position + i);
    }
    // Squared lengths of edges 12, 20 and 01; the longest is the diagonal, and the vertex
    // opposite it starts the strip.
    let e12 = v[2].position.xy - v[1].position.xy;
    let e20 = v[0].position.xy - v[2].position.xy;
    let e01 = v[1].position.xy - v[0].position.xy;
    let l12 = dot(e12, e12);
    let l20 = dot(e20, e20);
    let l01 = dot(e01, e01);
    var first = 2u;
    if (l12 > l20 && l12 > l01) {
        first = 0u;
    } else if (l20 > l01) {
        first = 1u;
    }
    let corner = xe_host_vertex_index & 3u;
    if (corner < 3u) {
        return v[(first + corner) % 3u];
    }
    let a = v[first];
    let b = v[(first + 1u) % 3u];
    let c = v[(first + 2u) % 3u];
    var out: XeVertexOutput;
    out.position = b.position + c.position - a.position;
",
                );
                for i in 0..16 {
                    if outputs & (1 << i) != 0 {
                        writeln!(m, "    out.o{i} = b.o{i} + c.o{i} - a.o{i};").unwrap();
                    }
                }
                if self.options.vs_point_size_location.is_some() {
                    m.push_str("    out.point_size = a.point_size;\n");
                }
                m.push_str("    return out;\n}\n");
            }
        }
        self.uses_shared_memory = true;
        outputs
    }

    fn pixel_entry(&mut self, m: &mut String) -> u32 {
        let interpolator_count = match self.options.ps_interpolator_count {
            Some(n) => n.min(16),
            None => self.register_count.min(16),
        };
        let inputs = if interpolator_count == 0 { 0 } else { (1u32 << interpolator_count) - 1 };
        m.push_str("struct XePixelInput {\n    @builtin(position) frag_coord: vec4<f32>,\n");
        if self.options.ps_param_gen {
            m.push_str("    @builtin(front_facing) front_facing: bool,\n");
        }
        for i in 0..interpolator_count {
            writeln!(m, "    @location({i}) i{i}: vec4<f32>,").unwrap();
        }
        m.push_str("}\n\n");
        let colors = self.info.writes_color_targets & 0xF;
        let has_output = colors != 0 || self.info.writes_depth;
        if has_output {
            m.push_str("struct XePixelOutput {\n");
            for i in 0..4 {
                if colors & (1 << i) != 0 {
                    writeln!(m, "    @location({i}) color{i}: vec4<f32>,").unwrap();
                }
            }
            if self.info.writes_depth {
                m.push_str("    @builtin(frag_depth) depth: f32,\n");
            }
            m.push_str("}\n\n");
        }
        m.push_str("@fragment\n");
        if has_output {
            m.push_str("fn xe_ps_main(input: XePixelInput) -> XePixelOutput {\n");
        } else {
            m.push_str("fn xe_ps_main(input: XePixelInput) {\n");
        }
        for i in 0..interpolator_count.min(self.register_count) {
            writeln!(m, "    xe_r[{i}] = input.i{i};").unwrap();
        }
        if self.options.ps_param_gen {
            write!(
                m,
                "    if ((xe_system.flags & {param_gen}u) != 0u) {{
        let flags = xe_system.flags;
        let pixel = abs(floor(input.frag_coord.xy)) + xe_system.ps_param_gen_offset;
        var x = pixel.x;
        if ((flags & {polygonal}u) != 0u && !input.front_facing) {{
            x = bitcast<f32>(bitcast<u32>(x) ^ 0x80000000u);
        }}
        var y = pixel.y;
        if ((flags & {point}u) != 0u) {{
            y = bitcast<f32>(bitcast<u32>(y) ^ 0x80000000u);
        }}
        let line_bit = select(0u, 0x80000000u, (flags & {line}u) != 0u);
        let gen_register = xe_system.ps_param_gen_register;
        if (gen_register < {count}u) {{
            xe_r[gen_register] = vec4<f32>(x, y, bitcast<f32>(line_bit), 0.0);
        }}
    }}
",
                param_gen = flags::PS_PARAM_GEN,
                polygonal = flags::PRIMITIVE_POLYGONAL,
                point = flags::PRIMITIVE_POINT,
                line = flags::PRIMITIVE_LINE,
                count = self.register_count,
            )
            .unwrap();
        }
        if self.info.writes_depth {
            m.push_str("    xe_out_depth = input.frag_coord.z;\n");
        }
        m.push_str("    xe_ucode();\n");
        if colors & 1 != 0 {
            write!(
                m,
                "    if ((xe_system.flags & {alpha}u) != 0u &&
        !xe_compare(xe_system.alpha_test_function, xe_out_color[0].w, xe_system.alpha_test_reference)) {{
        discard;
    }}
",
                alpha = flags::ALPHA_TEST
            )
            .unwrap();
        }
        if has_output {
            m.push_str("    var out: XePixelOutput;\n");
            for i in 0..4 {
                if colors & (1 << i) != 0 {
                    writeln!(
                        m,
                        "    out.color{i} = xe_color_range(xe_out_color[{i}] * xe_system.color_exp_bias[{i}], \
                         (xe_system.flags >> {}u) & 3u);",
                        flags::COLOR_RANGE_SHIFT + 2 * i
                    )
                    .unwrap();
                }
            }
            if self.info.writes_depth {
                m.push_str("    out.depth = xe_out_depth;\n");
            }
            m.push_str("    return out;\n");
        }
        m.push_str("}\n");
        inputs
    }
}

fn one_line_disassembly(instruction: &Instruction) -> String {
    let mut s = String::new();
    match instruction {
        Instruction::Alu(a) => crate::disasm::alu(&mut s, a),
        Instruction::VertexFetch(v) => crate::disasm::vertex_fetch(&mut s, v),
        Instruction::TextureFetch(t) => crate::disasm::texture_fetch(&mut s, t),
    }
    s.split('\n').map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::lit;

    #[test]
    fn float_literals() {
        assert_eq!(lit(1.0), "1.0");
        assert_eq!(lit(-2.0), "(-2.0)");
        assert_eq!(lit(0.5), "0.5f");
        assert_eq!(lit(-0.0), "(-0.0)");
        assert_eq!(lit(f32::MAX), format!("bitcast<f32>({:#010x}u)", f32::MAX.to_bits()));
        assert_eq!(lit(1.5 / 1024.0), "0.0014648438f");
    }
}
