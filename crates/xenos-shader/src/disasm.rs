//! Disassembler producing the same text as Xenia's `shader_translator_disasm.cc` /
//! `Shader::AnalyzeUcode` (the XNA-assembler-compatible syntax).

use std::fmt::Write;

use crate::ShaderStage;
use crate::parsed::{
    AddressingMode, AllocType, AnisoFilter, CfCondition, InstructionOperand, InstructionResult, ParsedAlu, ParsedCf,
    ParsedExec, ParsedTextureFetch, ParsedVertexFetch, StorageSource, StorageTarget, SwizzleSource, TextureFilter,
};
use crate::program::{Instruction, Program};
use crate::ucode::vertex_format;

fn result_operand(out: &mut String, result: &InstructionResult) {
    let uses_index = match result.target {
        StorageTarget::Register => {
            out.push('r');
            true
        }
        StorageTarget::Interpolator => {
            out.push('o');
            true
        }
        StorageTarget::Position => {
            out.push_str("oPos");
            false
        }
        StorageTarget::PointSizeEdgeFlagKillVertex => {
            out.push_str("oPts");
            false
        }
        StorageTarget::ExportAddress => {
            out.push_str("eA");
            false
        }
        StorageTarget::ExportData => {
            out.push_str("eM");
            true
        }
        StorageTarget::Color => {
            out.push_str("oC");
            true
        }
        StorageTarget::Depth => {
            out.push_str("oDepth");
            false
        }
        StorageTarget::None => false,
    };
    if uses_index {
        match result.addressing {
            AddressingMode::Absolute => write!(out, "{}", result.index).unwrap(),
            AddressingMode::AddressRegisterRelative => write!(out, "[{}+a0]", result.index).unwrap(),
            AddressingMode::LoopRelative => write!(out, "[{}+aL]", result.index).unwrap(),
        }
    }
    if result.original_write_mask == 0 {
        out.push_str("._");
    } else if result.original_write_mask != 0b1111
        || result.components != [SwizzleSource::X, SwizzleSource::Y, SwizzleSource::Z, SwizzleSource::W]
    {
        out.push('.');
        for i in 0..4 {
            if result.original_write_mask & (1 << i) != 0 {
                out.push(result.components[i].char());
            } else {
                out.push('_');
            }
        }
    }
}

fn source_operand(out: &mut String, op: &InstructionOperand) {
    if op.is_negated {
        out.push('-');
    }
    match op.source {
        StorageSource::Register => out.push('r'),
        StorageSource::ConstantFloat => out.push('c'),
        StorageSource::VertexFetchConstant | StorageSource::TextureFetchConstant => {}
    }
    if op.is_absolute_value {
        out.push_str("_abs");
    }
    match op.addressing {
        AddressingMode::Absolute => {
            if op.is_absolute_value {
                write!(out, "[{}]", op.index).unwrap();
            } else {
                write!(out, "{}", op.index).unwrap();
            }
        }
        AddressingMode::AddressRegisterRelative => write!(out, "[{}+a0]", op.index).unwrap(),
        AddressingMode::LoopRelative => write!(out, "[{}+aL]", op.index).unwrap(),
    }
    if !op.is_standard_swizzle() {
        out.push('.');
        for j in 0..op.component_count.min(4) {
            out.push(op.components[j as usize].char());
        }
    }
}

fn predicate_prefix(out: &mut String, is_predicated: bool, condition: bool) {
    if is_predicated {
        out.push_str(if condition { " (p0) " } else { "(!p0) " });
    } else {
        out.push_str("      ");
    }
}

pub fn exec(out: &mut String, instr: &ParsedExec) {
    match instr.condition {
        CfCondition::Always => write!(out, "      {}", instr.name).unwrap(),
        CfCondition::Predicate(c) => {
            out.push_str(if c { " (p0) " } else { "(!p0) " });
            out.push_str(instr.name);
        }
        CfCondition::Bool { index, value } => {
            write!(out, "      {} {}b{}", instr.name, if value { "" } else { "!" }, index).unwrap()
        }
    }
    if instr.is_yield {
        if matches!(instr.condition, CfCondition::Bool { .. }) {
            out.push(',');
        }
        out.push_str(" Yield=true");
    }
    if !instr.is_predicate_clean {
        out.push_str("    // PredicateClean=false");
    }
    out.push('\n');
}

pub fn control_flow(out: &mut String, cf: &ParsedCf, stage: ShaderStage) {
    match cf {
        ParsedCf::Nop => out.push_str("      cnop\n"),
        ParsedCf::Exec(e) => exec(out, e),
        ParsedCf::LoopStart { loop_constant, is_repeat, skip_address } => {
            write!(out, "      loop i{loop_constant}, L{skip_address}").unwrap();
            if *is_repeat {
                out.push_str(", Repeat=true");
            }
            out.push('\n');
        }
        ParsedCf::LoopEnd { loop_constant, body_address, predicated_break } => {
            match predicated_break {
                Some(c) => out.push_str(if *c { " (p0) " } else { "(!p0) " }),
                None => out.push_str("      "),
            }
            writeln!(out, "endloop i{loop_constant}, L{body_address}").unwrap();
        }
        ParsedCf::Call { target, condition } | ParsedCf::Jump { target, condition } => {
            let (plain, cond) = if matches!(cf, ParsedCf::Call { .. }) { ("call", "ccall") } else { ("jmp", "cjmp") };
            match condition {
                CfCondition::Always => write!(out, "      {plain} ").unwrap(),
                CfCondition::Predicate(c) => {
                    out.push_str(if *c { " (p0) " } else { "(!p0) " });
                    write!(out, "{plain} ").unwrap();
                }
                CfCondition::Bool { index, value } => {
                    write!(out, "      {cond} {}b{index}, ", if *value { "" } else { "!" }).unwrap()
                }
            }
            writeln!(out, "L{target}").unwrap();
        }
        ParsedCf::Return => out.push_str("      ret\n"),
        ParsedCf::Alloc { alloc_type, size } => {
            out.push_str("      alloc ");
            match alloc_type {
                AllocType::None => {}
                AllocType::VsPosition => out.push_str("position"),
                AllocType::InterpolatorsOrColors => {
                    out.push_str(if stage == ShaderStage::Vertex { "interpolators" } else { "colors" })
                }
                AllocType::Memory => write!(out, "export = {size}").unwrap(),
            }
            out.push('\n');
        }
        // Xenia prints nothing (not even a line break) for this instruction.
        ParsedCf::MarkVsFetchDone => out.push_str("      mark_vs_fetch_done\n"),
    }
}

pub fn vertex_fetch(out: &mut String, instr: &ParsedVertexFetch) {
    out.push_str("   ");
    predicate_prefix(out, instr.is_predicated, instr.predicate_condition);
    out.push_str(if instr.is_mini_fetch { "vfetch_mini" } else { "vfetch_full" });
    out.push(' ');
    result_operand(out, &instr.result);
    let a = &instr.attributes;
    if !instr.is_mini_fetch {
        out.push_str(", ");
        source_operand(out, &instr.index_operand);
        write!(out, ", vf{}", 95u32.wrapping_sub(instr.fetch_constant) as i32).unwrap();
        if a.is_index_rounded {
            out.push_str(", RoundIndex=true");
        }
    }
    if a.exp_adjust != 0 {
        write!(out, ", ExpAdjust={}", a.exp_adjust).unwrap();
    }
    if a.offset != 0 {
        write!(out, ", Offset={}", a.offset).unwrap();
    }
    if a.data_format != 0 {
        // Xenia indexes a table with null names for unknown formats.
        write!(out, ", DataFormat={}", vertex_format::name(a.data_format).unwrap_or("(null)")).unwrap();
    }
    if !instr.is_mini_fetch && a.stride != 0 {
        write!(out, ", Stride={}", a.stride).unwrap();
    }
    if a.is_signed {
        out.push_str(", Signed=true");
    }
    if a.is_integer {
        out.push_str(", NumFormat=integer");
    }
    if a.prefetch_count != 0 {
        write!(out, ", PrefetchCount={}", a.prefetch_count + 1).unwrap();
    }
    out.push('\n');
}

pub fn texture_fetch(out: &mut String, instr: &ParsedTextureFetch) {
    out.push_str("   ");
    predicate_prefix(out, instr.is_predicated, instr.predicate_condition);
    out.push_str(instr.name);
    out.push(' ');
    let has_result = instr.has_result();
    if has_result {
        result_operand(out, &instr.result);
        out.push_str(", ");
    }
    source_operand(out, &instr.src);
    if let Some(fc) = instr.fetch_constant {
        if has_result {
            out.push_str(", ");
        }
        write!(out, "tf{fc}").unwrap();
    }
    let a = &instr.attributes;
    if !a.fetch_valid_only {
        out.push_str(", FetchValidOnly=false");
    }
    if a.unnormalized_coordinates {
        out.push_str(", UnnormalizedTextureCoords=true");
    }
    let filter = |out: &mut String, name: &str, f: TextureFilter| {
        if f != TextureFilter::UseFetchConst {
            write!(out, ", {name}={}", f.name()).unwrap();
        }
    };
    filter(out, "MagFilter", a.mag_filter);
    filter(out, "MinFilter", a.min_filter);
    filter(out, "MipFilter", a.mip_filter);
    if a.aniso_filter != AnisoFilter::UseFetchConst {
        write!(out, ", AnisoFilter={}", a.aniso_filter.name()).unwrap();
    }
    filter(out, "VolMagFilter", a.vol_mag_filter);
    filter(out, "VolMinFilter", a.vol_min_filter);
    if !a.use_computed_lod {
        out.push_str(", UseComputedLOD=false");
    }
    if a.use_register_lod {
        out.push_str(", UseRegisterLOD=true");
    }
    if a.use_register_gradients {
        out.push_str(", UseRegisterGradients=true");
    }
    if a.lod_bias != 0.0 {
        write!(out, ", LODBias={}", a.lod_bias).unwrap();
    }
    let component_count = instr.dimension.component_count();
    if a.offset[0] != 0.0 {
        write!(out, ", OffsetX={}", a.offset[0]).unwrap();
    }
    if component_count > 1 && a.offset[1] != 0.0 {
        write!(out, ", OffsetY={}", a.offset[1]).unwrap();
    }
    if component_count > 2 && a.offset[2] != 0.0 {
        write!(out, ", OffsetZ={}", a.offset[2]).unwrap();
    }
    out.push('\n');
}

pub fn alu(out: &mut String, instr: &ParsedAlu) {
    let vector_nop = instr.is_vector_op_default_nop();
    let scalar_nop = instr.is_scalar_op_default_nop();
    if vector_nop && scalar_nop {
        out.push_str("   ");
        predicate_prefix(out, instr.is_predicated, instr.predicate_condition);
        out.push_str("nop\n");
        return;
    }
    if !vector_nop {
        out.push_str("   ");
        predicate_prefix(out, instr.is_predicated, instr.predicate_condition);
        out.push_str(instr.vector_name());
        if instr.vector_result.is_clamped {
            out.push_str("_sat");
        }
        out.push(' ');
        result_operand(out, &instr.vector_result);
        for i in 0..instr.vector_operand_count {
            out.push_str(", ");
            source_operand(out, &instr.vector_operands[i as usize]);
        }
        out.push('\n');
    }
    if !scalar_nop {
        out.push_str(if vector_nop { "   " } else { "              + " });
        predicate_prefix(out, instr.is_predicated, instr.predicate_condition);
        out.push_str(instr.scalar_name());
        if instr.scalar_result.is_clamped {
            out.push_str("_sat");
        }
        out.push(' ');
        result_operand(out, &instr.scalar_result);
        for i in 0..instr.scalar_operand_count {
            out.push_str(", ");
            source_operand(out, &instr.scalar_operands[i as usize]);
        }
        out.push('\n');
    }
}

/// Full program disassembly.
pub fn program(program: &Program) -> String {
    let mut out = String::new();
    for entry in &program.cf {
        if program.labels.contains(&entry.index) {
            writeln!(out, "                label L{}", entry.index).unwrap();
        }
        write!(out, "/* {:4}.{} */ ", entry.index / 2, entry.index & 1).unwrap();
        control_flow(&mut out, &entry.cf, program.stage);
        for instr in &entry.instructions {
            write!(out, "/* {:4}   */ ", instr.address).unwrap();
            if instr.serialize {
                out.push_str("         serialize\n             ");
            }
            match &instr.instruction {
                Instruction::Alu(a) => alu(&mut out, a),
                Instruction::VertexFetch(v) => vertex_fetch(&mut out, v),
                Instruction::TextureFetch(t) => texture_fetch(&mut out, t),
            }
        }
    }
    out
}
