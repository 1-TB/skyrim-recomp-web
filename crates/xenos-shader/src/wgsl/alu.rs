//! ALU (vector + scalar) instruction translation.

use super::{Generator, Value};
use crate::ShaderStage;
use crate::parsed::ParsedAlu;
use crate::ucode::{self, scalar_op as s, vector_op as v};

fn comparison(op: u32) -> &'static str {
    match op {
        v::SEQ | v::CNDEQ | v::SETP_EQ_PUSH | v::KILL_EQ => "==",
        v::SGT | v::CNDGT | v::SETP_GT_PUSH | v::KILL_GT => ">",
        v::SGE | v::CNDGE | v::SETP_GE_PUSH | v::KILL_GE => ">=",
        _ => "!=",
    }
}

impl Generator<'_> {
    pub(super) fn emit_alu(&mut self, alu: &ParsedAlu) {
        if alu.is_nop() {
            return;
        }
        let predicated = self.predicated(alu.is_predicated, alu.predicate_condition);
        self.open("{");

        let vop = alu.vector_opcode;
        let vector_needed = alu.vector_result.used_result_components() != 0 || ucode::vector_changes_state(vop);
        if vector_needed {
            let count = if vop == v::CUBE { 1 } else { alu.vector_operand_count };
            for (i, operand) in alu.vector_operands[..count as usize].iter().enumerate() {
                let e = self.operand_vec4(operand);
                self.line(&format!("let xe_v{i} = {e};"));
            }
        }
        let sop = alu.scalar_opcode;
        match alu.scalar_operand_count {
            1 => {
                let a = self.operand_scalar(&alu.scalar_operands[0], 0);
                self.line(&format!("let xe_sa = {a};"));
                if alu.scalar_operands[0].component_count == 2 {
                    let b = self.operand_scalar(&alu.scalar_operands[0], 1);
                    self.line(&format!("let xe_sb = {b};"));
                }
            }
            2 => {
                let a = self.operand_scalar(&alu.scalar_operands[0], 0);
                let b = self.operand_scalar(&alu.scalar_operands[1], 0);
                self.line(&format!("let xe_sa = {a};"));
                self.line(&format!("let xe_sb = {b};"));
            }
            _ => {}
        }

        if vector_needed {
            self.emit_vector_op(vop);
        }
        let scalar_written = alu.scalar_result.used_write_mask() != 0;
        self.emit_scalar_op(sop, scalar_written);

        self.store(&alu.vector_result, if vector_needed { Value::Vec("xe_vr") } else { Value::None });
        self.store(&alu.scalar_result, Value::Scalar("xe_sr"));

        self.close();
        if predicated {
            self.close();
        }
    }

    fn kill(&mut self, name: &str, condition: &str) {
        self.line(&format!("let {name} = {condition};"));
        if self.stage == ShaderStage::Pixel {
            self.open(&format!("if ({name}) {{"));
            self.line("discard;");
            self.close();
        }
    }

    fn emit_vector_op(&mut self, op: u32) {
        let cmp = comparison(op);
        let e = match op {
            v::ADD => "xe_v0 + xe_v1".to_string(),
            v::MUL => "xe_mul(xe_v0, xe_v1)".to_string(),
            v::MAX => "xe_max(xe_v0, xe_v1)".to_string(),
            v::MIN => "xe_min(xe_v0, xe_v1)".to_string(),
            v::SEQ | v::SGT | v::SGE | v::SNE => {
                format!("select(vec4<f32>(0.0), vec4<f32>(1.0), xe_v0 {cmp} xe_v1)")
            }
            v::FRC => "xe_v0 - floor(xe_v0)".to_string(),
            v::TRUNC => "trunc(xe_v0)".to_string(),
            v::FLOOR => "floor(xe_v0)".to_string(),
            v::MAD => "xe_mul(xe_v0, xe_v1) + xe_v2".to_string(),
            v::CNDEQ | v::CNDGE | v::CNDGT => format!("select(xe_v2, xe_v1, xe_v0 {cmp} vec4<f32>(0.0))"),
            v::DP4 => "vec4<f32>(xe_dp4(xe_v0, xe_v1))".to_string(),
            v::DP3 => "vec4<f32>(xe_dp3(xe_v0, xe_v1))".to_string(),
            v::DP2ADD => "vec4<f32>(xe_dp2add(xe_v0, xe_v1, xe_v2.x))".to_string(),
            v::CUBE => "xe_cube(xe_v0)".to_string(),
            v::MAX4 => "vec4<f32>(xe_max4(xe_v0))".to_string(),
            v::SETP_EQ_PUSH | v::SETP_NE_PUSH | v::SETP_GT_PUSH | v::SETP_GE_PUSH => {
                self.line(&format!("xe_p0 = xe_v0.w == 0.0 && xe_v1.w {cmp} 0.0;"));
                format!("vec4<f32>(select(xe_v0.x + 1.0, 0.0, xe_v0.x == 0.0 && xe_v1.x {cmp} 0.0))")
            }
            v::KILL_EQ | v::KILL_GT | v::KILL_GE | v::KILL_NE => {
                self.kill("xe_vkill", &format!("any(xe_v0 {cmp} xe_v1)"));
                "vec4<f32>(select(0.0, 1.0, xe_vkill))".to_string()
            }
            v::DST => "vec4<f32>(1.0, xe_mul1(xe_v0.y, xe_v1.y), xe_v0.z, xe_v1.w)".to_string(),
            v::MAXA => {
                self.line("xe_a0 = i32(clamp(floor(xe_v0.w + 0.5), -256.0, 255.0));");
                "xe_max(xe_v0, xe_v1)".to_string()
            }
            _ => {
                self.warn(format!("invalid ALU vector opcode {op} (result is zero)"));
                "vec4<f32>(0.0)".to_string()
            }
        };
        self.line(&format!("let xe_vr = {e};"));
    }

    /// Emits the scalar operation into `xe_sr`, updating ps (except for retain_prev, whose result
    /// is only materialized when written).
    fn emit_scalar_op(&mut self, op: u32, written: bool) {
        let e = match op {
            s::ADDS | s::ADDSC0 | s::ADDSC1 => "xe_sa + xe_sb".to_string(),
            s::ADDS_PREV => "xe_sa + xe_ps".to_string(),
            s::MULS | s::MULSC0 | s::MULSC1 => "xe_mul1(xe_sa, xe_sb)".to_string(),
            s::MULS_PREV => "xe_mul1(xe_sa, xe_ps)".to_string(),
            s::MULS_PREV2 => "select(xe_mul1(xe_sa, xe_ps), -XE_FLT_MAX, xe_ps == -XE_FLT_MAX || \
                              !xe_isfinite(xe_ps) || !xe_isfinite(xe_sb) || xe_sb <= 0.0)"
                .to_string(),
            s::MAXS => "xe_max1(xe_sa, xe_sb)".to_string(),
            s::MINS => "xe_min1(xe_sa, xe_sb)".to_string(),
            s::SEQS => "select(0.0, 1.0, xe_sa == 0.0)".to_string(),
            s::SGTS => "select(0.0, 1.0, xe_sa > 0.0)".to_string(),
            s::SGES => "select(0.0, 1.0, xe_sa >= 0.0)".to_string(),
            s::SNES => "select(0.0, 1.0, xe_sa != 0.0)".to_string(),
            s::FRCS => "xe_sa - floor(xe_sa)".to_string(),
            s::TRUNCS => "trunc(xe_sa)".to_string(),
            s::FLOORS => "floor(xe_sa)".to_string(),
            s::EXP => "xe_rfp(exp2(xe_sa))".to_string(),
            s::LOGC => "xe_inf_to_max(xe_rfp(log2(xe_sa)))".to_string(),
            s::LOG => "xe_rfp(log2(xe_sa))".to_string(),
            s::RCPC => "xe_inf_to_max(xe_rfp(1.0 / xe_sa))".to_string(),
            s::RCPF => "xe_inf_to_zero(xe_rfp(1.0 / xe_sa))".to_string(),
            s::RCP => "xe_rfp(1.0 / xe_sa)".to_string(),
            s::RSQC => "xe_inf_to_max(xe_rfp(inverseSqrt(xe_sa)))".to_string(),
            s::RSQF => "xe_inf_to_zero(xe_rfp(inverseSqrt(xe_sa)))".to_string(),
            s::RSQ => "xe_rfp(inverseSqrt(xe_sa))".to_string(),
            s::MAXAS | s::MAXASF => {
                let round = if op == s::MAXAS { " + 0.5" } else { "" };
                self.line(&format!("xe_a0 = i32(clamp(floor(xe_sa{round}), 0.0, 255.0));"));
                "xe_max1(xe_sa, xe_sb)".to_string()
            }
            s::SUBS | s::SUBSC0 | s::SUBSC1 => "xe_sa - xe_sb".to_string(),
            s::SUBS_PREV => "xe_sa - xe_ps".to_string(),
            s::SETP_EQ | s::SETP_NE | s::SETP_GT | s::SETP_GE => {
                let cmp = match op {
                    s::SETP_EQ => "==",
                    s::SETP_NE => "!=",
                    s::SETP_GT => ">",
                    _ => ">=",
                };
                self.line(&format!("xe_p0 = xe_sa {cmp} 0.0;"));
                "select(1.0, 0.0, xe_p0)".to_string()
            }
            s::SETP_INV => {
                self.line("xe_p0 = xe_sa == 1.0;");
                "select(select(xe_sa, 1.0, xe_sa == 0.0), 0.0, xe_p0)".to_string()
            }
            s::SETP_POP => {
                self.line("let xe_counter = xe_sa - 1.0;");
                self.line("xe_p0 = xe_counter <= 0.0;");
                "select(xe_counter, 0.0, xe_p0)".to_string()
            }
            s::SETP_CLR => {
                self.line("xe_p0 = false;");
                "XE_FLT_MAX".to_string()
            }
            s::SETP_RSTR => {
                self.line("xe_p0 = xe_sa == 0.0;");
                "select(xe_sa, 0.0, xe_p0)".to_string()
            }
            s::KILLS_EQ | s::KILLS_GT | s::KILLS_GE | s::KILLS_NE | s::KILLS_ONE => {
                let condition = match op {
                    s::KILLS_EQ => "xe_sa == 0.0",
                    s::KILLS_GT => "xe_sa > 0.0",
                    s::KILLS_GE => "xe_sa >= 0.0",
                    s::KILLS_NE => "xe_sa != 0.0",
                    _ => "xe_sa == 1.0",
                };
                self.kill("xe_skill", condition);
                "select(0.0, 1.0, xe_skill)".to_string()
            }
            s::SQRT => "xe_rfp(sqrt(xe_sa))".to_string(),
            s::SIN => "sin(xe_sa)".to_string(),
            s::COS => "cos(xe_sa)".to_string(),
            s::RETAIN_PREV => {
                if written {
                    self.line("let xe_sr = xe_ps;");
                }
                return;
            }
            _ => {
                self.warn(format!("invalid ALU scalar opcode {op} (treated as retain_prev)"));
                if written {
                    self.line("let xe_sr = xe_ps;");
                }
                return;
            }
        };
        self.line(&format!("let xe_sr = {e};"));
        self.line("xe_ps = xe_sr;");
    }
}
