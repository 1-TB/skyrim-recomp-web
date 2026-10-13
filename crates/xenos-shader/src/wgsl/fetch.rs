//! Vertex and texture fetch translation.

use super::{Generator, TextureBindingDimension, Value, lit};
use crate::ShaderStage;
use crate::parsed::{FetchDimension, ParsedTextureFetch, ParsedVertexFetch, TextureFilter};
use crate::ucode::{FetchOpcode, vertex_format as vfmt};

/// Texel-space offset added to texture coordinates to resolve ties between texels when point
/// sampling (Xenia's `kTextureCoordEpsilon`).
const TEXTURE_COORD_EPSILON: f32 = 1.5 / 1024.0;

enum Lod {
    Level(String),
    Bias(String),
    Grad(String, String),
}

impl Generator<'_> {
    pub(super) fn emit_vertex_fetch(&mut self, vf: &ParsedVertexFetch) {
        self.uses_shared_memory = true;
        let predicated = self.predicated(vf.is_predicated, vf.predicate_condition);
        self.open("{");
        let a = vf.attributes;
        let word_0 = vf.fetch_constant * 2;
        let word_1 = word_0 + 1;
        if !vf.is_mini_fetch {
            self.line(&format!("xe_vfetch_address = i32((xe_fetch_word({word_0}u) >> 2u) & 0x07FFFFFFu);"));
            self.line(&format!(
                "xe_vfetch_end = xe_vfetch_address + i32((xe_fetch_word({word_1}u) >> 2u) & 0xFFFFFFu);"
            ));
            if a.stride != 0 {
                let index = self.operand_scalar(&vf.index_operand, 0);
                let index = if a.is_index_rounded { format!("{index} + 0.5") } else { index };
                self.line(&format!("xe_vfetch_address = xe_vfetch_address + i32(floor({index})) * {};", a.stride));
            }
        }
        let used = vf.result.used_result_components();
        let format_components = vfmt::component_count(a.data_format);
        let used_format = used & ((1u32 << format_components) - 1);
        let needed = vfmt::needed_words(a.data_format, used);
        if format_components == 0 && used != 0 {
            self.warn(format!("unknown vertex format {} (fetches zero)", a.data_format));
        }
        if needed == 0 || used_format == 0 {
            self.store(&vf.result, Value::Vec("vec4<f32>(0.0)"));
            self.close();
            if predicated {
                self.close();
            }
            return;
        }
        self.line(&format!("let xe_endian = xe_fetch_word({word_1}u) & 3u;"));
        for w in 0..4 {
            if needed & (1 << w) != 0 {
                let offset = a.offset + w;
                self.line(&format!(
                    "let xe_w{w} = xe_vfetch_word(xe_vfetch_address + ({offset}), xe_vfetch_end, xe_endian);"
                ));
            }
        }
        let mut components = ["0.0".to_string(), "0.0".to_string(), "0.0".to_string(), "0.0".to_string()];
        // (word, bit offset, width) of packed fixed-point formats.
        let packed: Option<&[(u32, u32, u32)]> = match a.data_format {
            vfmt::K_8_8_8_8 => Some(&[(0, 0, 8), (0, 8, 8), (0, 16, 8), (0, 24, 8)]),
            vfmt::K_2_10_10_10 => Some(&[(0, 0, 10), (0, 10, 10), (0, 20, 10), (0, 30, 2)]),
            vfmt::K_10_11_11 => Some(&[(0, 0, 11), (0, 11, 11), (0, 22, 10)]),
            vfmt::K_11_11_10 => Some(&[(0, 0, 10), (0, 10, 11), (0, 21, 11)]),
            vfmt::K_16_16 => Some(&[(0, 0, 16), (0, 16, 16)]),
            vfmt::K_16_16_16_16 => Some(&[(0, 0, 16), (0, 16, 16), (1, 0, 16), (1, 16, 16)]),
            _ => None,
        };
        for (i, component) in components.iter_mut().enumerate() {
            if used_format & (1 << i) == 0 {
                continue;
            }
            let i = i as u32;
            *component = if let Some(packed) = packed {
                let (word, offset, width) = packed[i as usize];
                let raw = if a.is_signed {
                    format!("f32(extractBits(bitcast<i32>(xe_w{word}), {offset}u, {width}u))")
                } else {
                    format!("f32(extractBits(xe_w{word}, {offset}u, {width}u))")
                };
                if a.is_integer {
                    raw
                } else if a.is_signed {
                    let half = ((1u64 << (width - 1)) - 1) as f32;
                    if a.signed_rf_mode_no_zero {
                        format!("(({raw} + 0.5) / {})", lit(half + 0.5))
                    } else {
                        format!("max({raw} / {}, -1.0)", lit(half))
                    }
                } else {
                    format!("({raw} / {})", lit(((1u64 << width) - 1) as f32))
                }
            } else {
                match a.data_format {
                    vfmt::K_16_16_FLOAT | vfmt::K_16_16_16_16_FLOAT => {
                        let word = i / 2;
                        if i & 1 == 0 {
                            format!("xe_half_to_f32(xe_w{word} & 0xFFFFu)")
                        } else {
                            format!("xe_half_to_f32(xe_w{word} >> 16u)")
                        }
                    }
                    vfmt::K_32 | vfmt::K_32_32 | vfmt::K_32_32_32_32 => {
                        let raw =
                            if a.is_signed { format!("f32(bitcast<i32>(xe_w{i}))") } else { format!("f32(xe_w{i})") };
                        if a.is_integer {
                            raw
                        } else if a.is_signed {
                            if a.signed_rf_mode_no_zero {
                                format!("(({raw} + 0.5) / {})", lit(2147483647.5))
                            } else {
                                format!("({raw} / {})", lit(2147483647.0))
                            }
                        } else {
                            format!("({raw} / {})", lit(4294967295.0))
                        }
                    }
                    _ => format!("bitcast<f32>(xe_w{i})"),
                }
            };
        }
        let mut value = format!("vec4<f32>({})", components.join(", "));
        if a.exp_adjust != 0 {
            value = format!("({value} * {})", lit(2f32.powi(a.exp_adjust)));
        }
        self.line(&format!("let xe_f = {value};"));
        self.store(&vf.result, Value::Vec("xe_f"));
        self.close();
        if predicated {
            self.close();
        }
    }

    pub(super) fn emit_texture_fetch(&mut self, tf: &ParsedTextureFetch) {
        let used = tf.result.used_result_components();
        let is_set = matches!(
            tf.opcode,
            FetchOpcode::SetTextureLod | FetchOpcode::SetTextureGradientsHorz | FetchOpcode::SetTextureGradientsVert
        );
        if !is_set && tf.result.used_write_mask() == 0 {
            return;
        }
        let predicated = self.predicated(tf.is_predicated, tf.predicate_condition);
        self.open("{");
        let coords: Vec<String> = (0..tf.src.component_count).map(|i| self.operand_scalar(&tf.src, i)).collect();
        let pixel = self.stage == ShaderStage::Pixel;
        match tf.opcode {
            FetchOpcode::SetTextureLod => self.line(&format!("xe_tfetch_lod = {};", coords[0])),
            FetchOpcode::SetTextureGradientsHorz => {
                self.line(&format!("xe_tfetch_gradients_h = vec3<f32>({});", coords.join(", ")))
            }
            FetchOpcode::SetTextureGradientsVert => {
                self.line(&format!("xe_tfetch_gradients_v = vec3<f32>({});", coords.join(", ")))
            }
            FetchOpcode::GetTextureGradients => {
                let value = if pixel {
                    self.line(&format!("let xe_g = vec2<f32>({}, {});", coords[0], coords[1]));
                    "vec4<f32>(dpdx(xe_g.x), dpdy(xe_g.x), dpdx(xe_g.y), dpdy(xe_g.y))"
                } else {
                    "vec4<f32>(0.0)"
                };
                self.line(&format!("let xe_tr = {value};"));
                self.store(&tf.result, Value::Vec("xe_tr"));
            }
            FetchOpcode::GetTextureBorderColorFrac => {
                self.warn("getBCF is not supported (returns 0)".to_string());
                self.store(&tf.result, Value::Vec("vec4<f32>(0.0)"));
            }
            FetchOpcode::GetTextureComputedLod => {
                if pixel && used != 0 {
                    self.emit_computed_lod(tf, &coords);
                    self.store(&tf.result, Value::Vec("xe_tr"));
                } else {
                    self.store(&tf.result, Value::Vec("vec4<f32>(0.0)"));
                }
            }
            FetchOpcode::GetTextureWeights => {
                if used != 0 {
                    self.emit_weights(tf, &coords);
                    self.store(&tf.result, Value::Vec("xe_tr"));
                } else {
                    self.store(&tf.result, Value::None);
                }
            }
            FetchOpcode::TextureFetch => {
                if used != 0 {
                    self.emit_sample(tf, &coords);
                    self.store(&tf.result, Value::Vec("xe_tr"));
                } else {
                    self.store(&tf.result, Value::None);
                }
            }
            FetchOpcode::VertexFetch | FetchOpcode::Unknown(_) => {
                self.warn(format!("unknown fetch opcode {:?} (ignored)", tf.opcode));
            }
        }
        self.close();
        if predicated {
            self.close();
        }
    }

    /// Normalized coordinate along an axis: `c + offset / size` or `(c + offset) / size`.
    fn normalized(coord: &str, offset: f32, size: &str, unnormalized: bool) -> String {
        if unnormalized {
            format!("(({coord} + {}) / {size})", lit(offset))
        } else {
            format!("({coord} + {} / {size})", lit(offset))
        }
    }

    fn emit_sample(&mut self, tf: &ParsedTextureFetch, coords: &[String]) {
        let fc = tf.fetch_constant.unwrap_or(0);
        let a = tf.attributes;
        let binding = self.texture_binding(fc, tf.dimension, &a);
        let texture = format!("xe_texture_{}", binding.texture_binding);
        let sampler = format!("xe_sampler_{}", binding.sampler_binding);
        let pixel = self.stage == ShaderStage::Pixel;
        let unnorm = a.unnormalized_coordinates;
        let offset = |i: usize| a.offset[i] + TEXTURE_COORD_EPSILON;

        // LOD: fetch constant bias + register LOD + instruction bias.
        let mut lod = format!("xe_tex_lod_bias({fc}u)");
        if a.use_register_lod {
            lod.push_str(" + xe_tfetch_lod");
        }
        if a.lod_bias != 0.0 {
            lod.push_str(&format!(" + {}", lit(a.lod_bias)));
        }
        self.line(&format!("let xe_lod = {lod};"));
        let use_computed = a.use_computed_lod && (pixel || a.use_register_gradients);
        let mode = if use_computed && a.use_register_gradients {
            self.line(&format!("let xe_gexp = xe_tex_grad_exp_adjust({fc}u);"));
            self.line("let xe_gh = xe_tfetch_gradients_h * exp2(xe_lod + xe_gexp.x);");
            self.line("let xe_gv = xe_tfetch_gradients_v * exp2(xe_lod + xe_gexp.y);");
            Lod::Grad("xe_gh".to_string(), "xe_gv".to_string())
        } else if use_computed {
            self.line(&format!("let xe_gexp = xe_tex_grad_exp_adjust({fc}u);"));
            Lod::Bias("xe_lod + max(xe_gexp.x, xe_gexp.y)".to_string())
        } else {
            Lod::Level("xe_lod".to_string())
        };
        // Gradient components for the sampled dimension, normalized like the coordinates.
        let grads = |n: usize, size: &str| -> (String, String) {
            let Lod::Grad(h, v) = &mode else { unreachable!() };
            let swizzle = ["x", "xy", "xyz"][n - 1];
            if unnorm {
                let size_swizzle = format!("{size}.{swizzle}");
                (format!("({h}.{swizzle} / {size_swizzle})"), format!("({v}.{swizzle} / {size_swizzle})"))
            } else {
                (format!("{h}.{swizzle}"), format!("{v}.{swizzle}"))
            }
        };
        let call = |texture: &str, coords: &str, layer: Option<&str>, grad: Option<(String, String)>| -> String {
            let layer = layer.map(|l| format!(", {l}")).unwrap_or_default();
            match &mode {
                Lod::Level(l) => format!("textureSampleLevel({texture}, {sampler}, {coords}{layer}, {l})"),
                Lod::Bias(b) => format!("textureSampleBias({texture}, {sampler}, {coords}{layer}, {b})"),
                Lod::Grad(..) => {
                    let (h, v) = grad.unwrap();
                    format!("textureSampleGrad({texture}, {sampler}, {coords}{layer}, {h}, {v})")
                }
            }
        };
        let is_grad = matches!(mode, Lod::Grad(..));
        match binding.dimension {
            TextureBindingDimension::D2 => {
                self.line(&format!("let xe_size = xe_tex_size_2d({fc}u);"));
                let u = Self::normalized(&coords[0], offset(0), "xe_size.x", unnorm);
                let v = if coords.len() > 1 {
                    Self::normalized(&coords[1], offset(1), "xe_size.y", unnorm)
                } else {
                    "(0.5 / xe_size.y)".to_string()
                };
                self.line(&format!("let xe_uv = vec2<f32>({u}, {v});"));
                let grad = if is_grad {
                    let (h, v) = grads(2, "xe_size");
                    if tf.dimension == FetchDimension::D1 && coords.len() == 1 {
                        Some((format!("vec2<f32>({h}.x, 0.0)"), format!("vec2<f32>({v}.x, 0.0)")))
                    } else {
                        Some((h, v))
                    }
                } else {
                    None
                };
                let sample = call(&texture, "xe_uv", None, grad);
                self.line(&format!("let xe_ts = {sample};"));
            }
            TextureBindingDimension::Cube => {
                self.line(&format!("let xe_size = xe_tex_size_2d({fc}u);"));
                let sc = Self::normalized(&coords[0], offset(0), "xe_size.x", unnorm);
                let tc = Self::normalized(&coords[1], offset(1), "xe_size.y", unnorm);
                let face = if a.offset[2] != 0.0 {
                    format!("{} + {}", coords[2], lit(a.offset[2]))
                } else {
                    coords[2].clone()
                };
                self.line(&format!("let xe_dir = xe_cube_direction(vec3<f32>({sc}, {tc}, {face}));"));
                // Register gradients of cube maps are in cube space.
                let grad = if let Lod::Grad(h, v) = &mode { Some((h.clone(), v.clone())) } else { None };
                let sample = call(&texture, "xe_dir", None, grad);
                self.line(&format!("let xe_ts = {sample};"));
            }
            TextureBindingDimension::D3OrStacked => {
                let stacked = format!("xe_texture_{}", binding.stacked_texture_binding.unwrap());
                self.line(&format!("let xe_size = xe_tex_size_3d({fc}u);"));
                let u = Self::normalized(&coords[0], offset(0), "xe_size.x", unnorm);
                let v = Self::normalized(&coords[1], offset(1), "xe_size.y", unnorm);
                self.line(&format!("let xe_uv = vec2<f32>({u}, {v});"));
                self.line("var xe_ts: vec4<f32>;");
                self.open(&format!("if (xe_tex_is_3d({fc}u)) {{"));
                let w = Self::normalized(&coords[2], offset(2), "xe_size.z", unnorm);
                self.line(&format!("let xe_uvw = vec3<f32>(xe_uv, {w});"));
                let grad = if is_grad { Some(grads(3, "xe_size")) } else { None };
                let sample = call(&texture, "xe_uvw", None, grad);
                self.line(&format!("xe_ts = {sample};"));
                self.indent -= 1;
                self.open("} else {");
                let layer = if unnorm {
                    format!("{} + {}", coords[2], lit(offset(2)))
                } else {
                    format!("{} * xe_size.z + {}", coords[2], lit(offset(2)))
                };
                self.line(&format!("let xe_layer = {layer};"));
                let linear = match (a.vol_mag_filter, a.vol_min_filter) {
                    (TextureFilter::UseFetchConst, _) | (_, TextureFilter::UseFetchConst) => {
                        format!("xe_tex_vol_linear({fc}u)")
                    }
                    (mag, min) => (mag == TextureFilter::Linear || min == TextureFilter::Linear).to_string(),
                };
                let grad2 = if is_grad { Some(grads(2, "xe_size")) } else { None };
                self.open(&format!("if ({linear}) {{"));
                self.line("let xe_lt = xe_layer - 0.5;");
                self.line("let xe_l0 = floor(xe_lt);");
                self.line("let xe_last = xe_size.z - 1.0;");
                let s0 = call(&stacked, "xe_uv", Some("i32(clamp(xe_l0, 0.0, xe_last))"), grad2.clone());
                let s1 = call(&stacked, "xe_uv", Some("i32(clamp(xe_l0 + 1.0, 0.0, xe_last))"), grad2.clone());
                self.line(&format!("xe_ts = mix({s0}, {s1}, xe_lt - xe_l0);"));
                self.indent -= 1;
                self.open("} else {");
                let s = call(&stacked, "xe_uv", Some("i32(clamp(floor(xe_layer), 0.0, xe_size.z - 1.0))"), grad2);
                self.line(&format!("xe_ts = {s};"));
                self.close();
                self.close();
            }
        }
        self.line(&format!("let xe_tr = xe_tex_post(xe_ts, {fc}u);"));
    }

    fn emit_computed_lod(&mut self, tf: &ParsedTextureFetch, coords: &[String]) {
        let fc = tf.fetch_constant.unwrap_or(0);
        let scale = |i: usize, size: &str| {
            if tf.attributes.unnormalized_coordinates { coords[i].clone() } else { format!("{} * {size}", coords[i]) }
        };
        match tf.dimension {
            FetchDimension::D1 | FetchDimension::D2 | FetchDimension::Cube => {
                self.line(&format!("let xe_size = xe_tex_size_2d({fc}u);"));
                let y = if coords.len() > 1 { scale(1, "xe_size.y") } else { "0.0".to_string() };
                self.line(&format!("let xe_texel = vec2<f32>({}, {y});", scale(0, "xe_size.x")));
            }
            FetchDimension::D3OrStacked => {
                self.line(&format!("let xe_size = xe_tex_size_3d({fc}u);"));
                let z = format!("select(0.0, {}, xe_tex_is_3d({fc}u))", scale(2, "xe_size.z"));
                self.line(&format!(
                    "let xe_texel = vec3<f32>({}, {}, {z});",
                    scale(0, "xe_size.x"),
                    scale(1, "xe_size.y")
                ));
            }
        }
        self.line("let xe_dx = dpdx(xe_texel);");
        self.line("let xe_dy = dpdy(xe_texel);");
        self.line("let xe_tr = vec4<f32>(0.5 * log2(max(dot(xe_dx, xe_dx), dot(xe_dy, xe_dy))), 0.0, 0.0, 0.0);");
    }

    fn emit_weights(&mut self, tf: &ParsedTextureFetch, coords: &[String]) {
        let fc = tf.fetch_constant.unwrap_or(0);
        let a = tf.attributes;
        let (count, size) = match tf.dimension {
            FetchDimension::D1 => (1, "xe_tex_size_2d"),
            FetchDimension::D2 | FetchDimension::Cube => (2, "xe_tex_size_2d"),
            FetchDimension::D3OrStacked => (3, "xe_tex_size_3d"),
        };
        self.line(&format!("let xe_size = {size}({fc}u);"));
        let mut c = ["0.0".to_string(), "0.0".to_string(), "0.0".to_string()];
        for (i, component) in c.iter_mut().enumerate().take(count.min(coords.len())) {
            let texel =
                if a.unnormalized_coordinates { coords[i].clone() } else { format!("{} * xe_size[{i}]", coords[i]) };
            *component = format!("fract({texel} + {})", lit(a.offset[i] + TEXTURE_COORD_EPSILON - 0.5));
        }
        self.line(&format!("let xe_tr = vec4<f32>({}, {}, {}, 0.0);", c[0], c[1], c[2]));
    }
}
