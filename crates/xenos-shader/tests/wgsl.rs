mod common;

use common::*;
use xenos_shader::bindings::{self, SYSTEM_CONSTANTS_WGSL, SystemConstants};
use xenos_shader::{ShaderStage, TextureBindingDimension, Translation, TranslationOptions, VertexExpansion, translate};

fn translate_ok(stage: ShaderStage, cfs: &[Cf]) -> Translation {
    translate_with(stage, cfs, &TranslationOptions::default())
}

fn translate_with(stage: ShaderStage, cfs: &[Cf], options: &TranslationOptions) -> Translation {
    let ucode = assemble(cfs);
    let t = translate(stage, &ucode, options).unwrap();
    validate_wgsl(&t.wgsl);
    t
}

const STAGES: [ShaderStage; 2] = [ShaderStage::Vertex, ShaderStage::Pixel];

fn output_export(stage: ShaderStage) -> Alu {
    match stage {
        ShaderStage::Vertex => Alu::vector(2, 62, 0b1111, &[r(1), r(1)]).export(),
        ShaderStage::Pixel => Alu::vector(2, 0, 0b1111, &[r(1), r(1)]).export(),
    }
}

#[test]
fn every_vector_opcode() {
    for stage in STAGES {
        for op in 0..32 {
            let mut plain = Alu::vector(op, 1, 0b1111, &[r(2).swz("wzyx"), c(3).neg(), r(4).abs()]);
            plain.vector_clamp = op % 2 == 0;
            let mut relative = Alu::vector(op, 5, 0b0101, &[c(10), c(11).swz("yyyy"), r(6).rel()]);
            relative.const_0_rel = true;
            relative.const_1_rel = true;
            relative.const_a0 = op % 3 == 0;
            relative.vector_dest_rel = true;
            let state_only = Alu::vector(op, 7, 0b0000, &[r(1), r(2), r(3)]).pred(op % 2 == 1);
            let t = translate_ok(
                stage,
                &[exece(vec![plain.into(), relative.into(), state_only.into(), output_export(stage).into()])],
            );
            if op >= 30 {
                assert!(!t.warnings.is_empty(), "invalid opcode {op} should warn");
            }
        }
    }
}

#[test]
fn every_scalar_opcode() {
    for stage in STAGES {
        for op in 0..64 {
            let mut a = Alu::scalar(op, 1, 0b1001, r(2).swz("zwxy").neg());
            a.scalar_clamp = op % 2 == 0;
            // Co-issued with a three-operand vector op (scalar b component = Z).
            let mut b = Alu::vector(11, 3, 0b0011, &[r(1), c(2), c(5).abs()]);
            b.scalar_op = op;
            b.scalar_dest = 4;
            b.scalar_mask = 0b0100;
            b.scalar_dest_rel = true;
            b.const_1_rel = true;
            let mut constant = Alu::scalar(op, 6, 0b0010, c(7).swz("yxwz"));
            constant.const_0_rel = true;
            constant.const_a0 = true;
            constant.abs_constants = true;
            let t = translate_ok(
                stage,
                &[exece(vec![a.into(), b.into(), constant.pred(true).into(), output_export(stage).into()])],
            );
            if op == 41 || op > 50 {
                assert!(!t.warnings.is_empty(), "invalid scalar opcode {op} should warn");
            } else {
                assert!(t.warnings.is_empty(), "{:?}", t.warnings);
            }
        }
    }
}

#[test]
fn every_vertex_format() {
    let formats = [6, 7, 16, 17, 25, 26, 31, 32, 33, 34, 35, 36, 37, 38, 57];
    for format in formats {
        for flags in 0..8u32 {
            for swizzle in ["xyzw", "zyx1", "w0__", "____", "1111"] {
                let full = VFetch {
                    dest: 1,
                    dest_swizzle: swizzle,
                    format,
                    signed: flags & 1 != 0,
                    integer: flags & 2 != 0,
                    no_zero: flags & 4 != 0,
                    exp_adjust: if flags & 1 != 0 { -5 } else { 7 },
                    round_index: flags & 2 != 0,
                    offset: (flags as i32) - 3,
                    stride: 8,
                    fetch_constant: 3 * flags + 1,
                    ..Default::default()
                };
                let mini = VFetch {
                    dest: 2,
                    dest_swizzle: "yxwz",
                    mini: true,
                    format,
                    offset: 5,
                    predicated: Some(true),
                    ..full
                };
                let t = translate_ok(
                    ShaderStage::Vertex,
                    &[exece(vec![full.into(), mini.into(), output_export(ShaderStage::Vertex).into()])],
                );
                assert!(t.warnings.is_empty());
                // The mini fetch always uses the binding of the full fetch.
                assert_eq!(t.vertex_bindings.len(), 1);
                let attributes = &t.vertex_bindings[0].attributes;
                assert_eq!(attributes.len(), if swizzle == "____" || swizzle == "1111" { 1 } else { 2 });
            }
        }
    }
}

#[test]
fn vertex_fetch_relative_and_unknown_format() {
    let full = VFetch { dest: 3, format: 0, ..Default::default() };
    let mut encoded = full.encode();
    encoded[0] |= 1 << 11 | 1 << 18; // aL-relative source and destination
    let t = translate_ok(
        ShaderStage::Vertex,
        &[exece(vec![Ins::Fetch(encoded), output_export(ShaderStage::Vertex).into()])],
    );
    assert!(t.warnings.iter().any(|w| w.contains("unknown vertex format")));
    assert_eq!(t.register_count, 64);
}

#[test]
fn every_texture_fetch_variant() {
    for stage in STAGES {
        for opcode in [1u32, 16, 17, 18, 19, 24, 25, 26] {
            for dimension in 0..4 {
                for variant in 0..6u32 {
                    let tf = TFetch {
                        opcode,
                        dest: 1,
                        dest_swizzle: ["xyzw", "wzyx", "x01_", "_y__", "zzzz", "0w1x"][variant as usize],
                        src: 2,
                        src_swizzle: ["xyz", "xy", "zyx", "x", "wzy", "yy"][variant as usize],
                        fetch_constant: (variant * 5 + dimension) % 32,
                        dimension,
                        unnormalized: variant % 2 == 1,
                        computed_lod: variant != 2,
                        register_lod: variant >= 3,
                        register_gradients: variant == 4 || variant == 5,
                        lod_bias_16ths: variant as i32 * 7 - 20,
                        offset_halves: [variant as i32 - 2, 3 - variant as i32, 1],
                        mag: variant % 4,
                        min: (variant + 1) % 4,
                        mip: (variant + 2) % 4,
                        aniso: if variant == 1 { 3 } else { 7 },
                        vol_mag: variant % 4,
                        vol_min: (variant + 3) % 4,
                        predicated: if variant == 3 { Some(false) } else { None },
                        ..Default::default()
                    };
                    translate_ok(stage, &[exece(vec![tf.into(), output_export(stage).into()])]);
                }
            }
        }
    }
}

#[test]
fn texture_binding_assignment() {
    let t = translate_ok(
        ShaderStage::Pixel,
        &[
            exec(vec![
                TFetch { fetch_constant: 4, dimension: 1, ..Default::default() }.into(),
                TFetch { fetch_constant: 4, dimension: 0, src_swizzle: "x", ..Default::default() }.into(),
                TFetch { fetch_constant: 2, dimension: 2, ..Default::default() }.into(),
                TFetch { fetch_constant: 7, dimension: 3, ..Default::default() }.into(),
                TFetch { fetch_constant: 4, dimension: 1, mag: 1, ..Default::default() }.into(),
                TFetch { opcode: 19, fetch_constant: 9, dimension: 1, ..Default::default() }.into(),
            ]),
            exece(vec![output_export(ShaderStage::Pixel).into()]),
        ],
    );
    let b = &t.texture_bindings;
    assert_eq!(b.len(), 4);
    assert_eq!(
        (b[0].fetch_constant, b[0].dimension, b[0].texture_binding, b[0].sampler_binding),
        (4, TextureBindingDimension::D2, 0, 1)
    );
    assert_eq!(
        (b[1].fetch_constant, b[1].dimension, b[1].texture_binding, b[1].stacked_texture_binding, b[1].sampler_binding),
        (2, TextureBindingDimension::D3OrStacked, 2, Some(3), 4)
    );
    assert_eq!(
        (b[2].fetch_constant, b[2].dimension, b[2].texture_binding, b[2].sampler_binding),
        (7, TextureBindingDimension::Cube, 5, 6)
    );
    assert_eq!((b[3].texture_binding, b[3].sampler_binding), (7, 8));
    assert_eq!(b[3].mag_filter, xenos_shader::parsed::TextureFilter::Linear);
    assert!(t.wgsl.contains("@group(2) @binding(3) var xe_texture_3: texture_2d_array<f32>;"));
    assert!(t.wgsl.contains("@group(2) @binding(5) var xe_texture_5: texture_cube<f32>;"));

    // A binding base shifts every texture and sampler (vertex and pixel shaders share group 2).
    let cfs = [
        exec(vec![
            TFetch { fetch_constant: 4, dimension: 1, ..Default::default() }.into(),
            TFetch { fetch_constant: 2, dimension: 2, ..Default::default() }.into(),
        ]),
        exece(vec![output_export(ShaderStage::Pixel).into()]),
    ];
    let shifted = translate_with(
        ShaderStage::Pixel,
        &cfs,
        &TranslationOptions { texture_binding_base: 10, ..Default::default() },
    );
    let b = &shifted.texture_bindings;
    assert_eq!((b[0].texture_binding, b[0].sampler_binding), (10, 11));
    assert_eq!((b[1].texture_binding, b[1].stacked_texture_binding, b[1].sampler_binding), (12, Some(13), 14));
    assert!(shifted.wgsl.contains("@group(2) @binding(10) var xe_texture_10: texture_2d<f32>;"));
}

#[test]
fn control_flow_patterns() {
    let body = |dest: u32| -> Ins { Alu::vector(0, dest, 0b1111, &[r(dest), c(1)]).into() };
    let pred_set: Ins = Alu::scalar(29, 0, 0, r(1)).into();
    let stage = ShaderStage::Pixel;
    let export: Ins = output_export(stage).into();
    let cases: Vec<(Vec<Cf>, bool)> = vec![
        // Straight line with conditional execs and an early conditional end.
        (
            vec![
                cexec(Cond::Bool(3, true), false, vec![body(1)]),
                cexec(Cond::Pred(false), false, vec![pred_set, body(1)]),
                cexec(Cond::Bool(4, false), true, vec![export]),
                cexec(Cond::Pred(true), true, vec![export]),
                exece(vec![export]),
            ],
            true,
        ),
        // if / else via jumps.
        (
            vec![
                exec(vec![pred_set]),
                Cf::Jmp { target: 4, cond: Cond::Pred(false) },
                exec(vec![body(1)]),
                Cf::Jmp { target: 5, cond: Cond::Always },
                exec(vec![body(2)]),
                exece(vec![export]),
            ],
            true,
        ),
        // Nested loops with a predicated break and a repeat loop, aL-relative accesses.
        (
            vec![
                Cf::LoopStart { loop_id: 0, skip: 6, repeat: false },
                exec(vec![Alu::vector(0, 1, 0b1111, &[r(1), c(0)]).into()]),
                Cf::LoopStart { loop_id: 1, skip: 5, repeat: true },
                exec(vec![pred_set, Alu::vector(0, 1, 0b1111, &[r(1).rel(), r(2)]).into()]),
                Cf::LoopEnd { loop_id: 1, body: 3, pred_break: Some(true) },
                Cf::LoopEnd { loop_id: 0, body: 1, pred_break: None },
                exece(vec![export]),
            ],
            true,
        ),
        // Jump inside a loop body to the loop end (continue).
        (
            vec![
                Cf::LoopStart { loop_id: 2, skip: 5, repeat: false },
                exec(vec![pred_set]),
                Cf::Jmp { target: 4, cond: Cond::Bool(7, true) },
                exec(vec![body(3)]),
                Cf::LoopEnd { loop_id: 2, body: 1, pred_break: Some(false) },
                exece(vec![export]),
            ],
            true,
        ),
        // Jump past the end of the program.
        (
            vec![
                exec(vec![pred_set, export]),
                Cf::Jmp { target: 40, cond: Cond::Pred(true) },
                exece(vec![body(1), export]),
            ],
            true,
        ),
        // Backward jump: needs the dispatch loop.
        (
            vec![exec(vec![body(1), pred_set]), Cf::Jmp { target: 0, cond: Cond::Pred(false) }, exece(vec![export])],
            false,
        ),
        // Subroutine calls.
        (
            vec![
                Cf::Call { target: 4, cond: Cond::Always },
                Cf::Call { target: 4, cond: Cond::Bool(9, true) },
                Cf::Call { target: 4, cond: Cond::Pred(false) },
                exece(vec![export]),
                exec(vec![body(2)]),
                Cf::Ret,
            ],
            false,
        ),
        // Overlapping jumps (not single-entry).
        (
            vec![
                Cf::Jmp { target: 3, cond: Cond::Bool(1, true) },
                exec(vec![body(1)]),
                Cf::Jmp { target: 4, cond: Cond::Bool(2, true) },
                exec(vec![body(2)]),
                exece(vec![export]),
            ],
            false,
        ),
        // Loop end without a matching start and a jump into a loop body.
        (
            vec![
                Cf::Jmp { target: 2, cond: Cond::Bool(1, true) },
                Cf::LoopStart { loop_id: 3, skip: 4, repeat: false },
                exec(vec![body(1)]),
                Cf::LoopEnd { loop_id: 3, body: 2, pred_break: None },
                exece(vec![export]),
            ],
            false,
        ),
    ];
    for (n, (cfs, structured)) in cases.iter().enumerate() {
        for stage in STAGES {
            let mut cfs = cfs.clone();
            if stage == ShaderStage::Vertex {
                for cf in &mut cfs {
                    if let Cf::Exec { instrs, .. } = cf {
                        for i in instrs.iter_mut() {
                            if let Ins::Alu(d) = i
                                && *d == output_export(ShaderStage::Pixel).encode()
                            {
                                *i = output_export(ShaderStage::Vertex).into();
                            }
                        }
                    }
                }
            }
            let t = translate_ok(stage, &cfs);
            assert_eq!(t.structured_control_flow, *structured, "case {n}:\n{}", t.wgsl);
        }
    }
}

#[test]
fn outputs_and_options() {
    // Vertex shader with interpolators, point size and vertex kill.
    let mut psize = Alu::vector(2, 63, 0b0101, &[r(1), r(1)]).export();
    psize.scalar_op = 50;
    let vs = [exece(vec![
        output_export(ShaderStage::Vertex).into(),
        Alu::vector(2, 3, 0b1111, &[r(2), r(2)]).export().into(),
        psize.into(),
    ])];
    let options =
        TranslationOptions { vs_output_interpolators: 0b1010, vs_point_size_location: Some(15), ..Default::default() };
    let t = translate_with(ShaderStage::Vertex, &vs, &options);
    assert_eq!(t.entry_point, "xe_vs_main");
    assert_eq!(t.interpolators_written, 0b1000);
    assert_eq!(t.vs_output_interpolators, 0b1010);
    assert!(t.writes_point_size && t.writes_vertex_kill);
    assert!(t.wgsl.contains("@location(15) point_size: f32"));
    assert!(t.wgsl.contains("@location(3) o3"));
    assert!(!t.wgsl.contains("@location(0) o0"));
    let options = TranslationOptions { vs_point_size_location: Some(1), ..Default::default() };
    let t = translate_with(ShaderStage::Vertex, &vs, &options);
    assert_eq!(t.vs_output_interpolators, 0xFFFD);
    assert!(t.warnings.iter().any(|w| w.contains("point size")));

    // Pixel shader with kill, depth, multiple render targets, no param gen.
    let ps = [exece(vec![
        Alu::scalar(35, 0, 0, r(0)).into(),
        Alu::vector(27, 0, 0, &[r(1), r(2)]).into(),
        Alu::vector(2, 1, 0b1111, &[r(3), r(3)]).export().into(),
        Alu::vector(2, 3, 0b0011, &[r(5), r(5)]).export().into(),
        Alu::vector(2, 61, 0b0001, &[r(4), r(4)]).export().into(),
    ])];
    let options = TranslationOptions { ps_param_gen: false, ps_interpolator_count: Some(3), ..Default::default() };
    let t = translate_with(ShaderStage::Pixel, &ps, &options);
    assert_eq!(t.entry_point, "xe_ps_main");
    assert!(t.kills_pixels && t.writes_depth);
    assert_eq!(t.color_targets_written, 0b1010);
    assert_eq!(t.ps_input_interpolators, 0b111);
    assert!(t.wgsl.contains("@builtin(frag_depth) depth: f32"));
    assert!(!t.wgsl.contains("front_facing"));

    // Depth-only pixel shader without outputs, and a memexport.
    let mut ea = Alu::vector(11, 32, 0b1111, &[r(0), c(255), c(9)]).export();
    ea.scalar_op = 50;
    let t = translate_ok(
        ShaderStage::Pixel,
        &[exece(vec![ea.into(), Alu::vector(2, 33, 0b1111, &[r(1), r(1)]).export().into()])],
    );
    assert!(t.uses_memexport);
    assert_eq!(t.info.memexport_stream_constants.iter().copied().collect::<Vec<_>>(), vec![9]);
    assert!(t.warnings.iter().any(|w| w.contains("memory export")));
    assert_eq!(t.color_targets_written, 0);
}

#[test]
fn analysis_reports_resources() {
    let mut rel = Alu::vector(0, 2, 0b1111, &[c(4), c(5)]);
    rel.const_0_rel = true;
    let t = translate_ok(
        ShaderStage::Vertex,
        &[
            cexec(Cond::Bool(130, true), false, vec![Alu::vector(0, 9, 0b1111, &[c(3), c(200)]).into()]),
            Cf::LoopStart { loop_id: 5, skip: 4, repeat: false },
            exec(vec![rel.into()]),
            Cf::LoopEnd { loop_id: 5, body: 2, pred_break: None },
            exece(vec![output_export(ShaderStage::Vertex).into()]),
        ],
    );
    let info = &t.info;
    assert_eq!(info.bool_constants[4], 1 << 2);
    assert_eq!(info.loop_constants, 1 << 5);
    assert!(info.float_dynamic_addressing);
    assert_eq!(info.register_static_address_bound, 10);
    assert!(info.uses_loops);
}

#[test]
fn system_constants_layout_matches_wgsl() {
    let source = format!("{SYSTEM_CONSTANTS_WGSL}\n@group(0) @binding(3) var<uniform> s: XeSystemConstants;\n");
    let module = validate_wgsl(&source);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).unwrap();
    let (handle, ty) = module.types.iter().find(|(_, t)| t.name.as_deref() == Some("XeSystemConstants")).unwrap();
    let naga::TypeInner::Struct { members, span } = &ty.inner else { panic!() };
    let rust_offsets = [
        ("flags", std::mem::offset_of!(SystemConstants, flags)),
        ("vertex_index_endian", std::mem::offset_of!(SystemConstants, vertex_index_endian)),
        ("vertex_base_index", std::mem::offset_of!(SystemConstants, vertex_base_index)),
        ("vertex_index_load_address", std::mem::offset_of!(SystemConstants, vertex_index_load_address)),
        ("vertex_index_min", std::mem::offset_of!(SystemConstants, vertex_index_min)),
        ("vertex_index_max", std::mem::offset_of!(SystemConstants, vertex_index_max)),
        ("alpha_test_function", std::mem::offset_of!(SystemConstants, alpha_test_function)),
        ("alpha_test_reference", std::mem::offset_of!(SystemConstants, alpha_test_reference)),
        ("ndc_scale", std::mem::offset_of!(SystemConstants, ndc_scale)),
        ("point_vertex_diameter_min", std::mem::offset_of!(SystemConstants, point_vertex_diameter_min)),
        ("ndc_offset", std::mem::offset_of!(SystemConstants, ndc_offset)),
        ("point_vertex_diameter_max", std::mem::offset_of!(SystemConstants, point_vertex_diameter_max)),
        ("point_constant_diameter", std::mem::offset_of!(SystemConstants, point_constant_diameter)),
        (
            "point_screen_diameter_to_ndc_radius",
            std::mem::offset_of!(SystemConstants, point_screen_diameter_to_ndc_radius),
        ),
        ("ps_param_gen_offset", std::mem::offset_of!(SystemConstants, ps_param_gen_offset)),
        ("ps_param_gen_register", std::mem::offset_of!(SystemConstants, ps_param_gen_register)),
        ("padding_0", std::mem::offset_of!(SystemConstants, padding_0)),
        ("color_exp_bias", std::mem::offset_of!(SystemConstants, color_exp_bias)),
    ];
    assert_eq!(members.len(), rust_offsets.len());
    for (member, (name, offset)) in members.iter().zip(rust_offsets) {
        assert_eq!(member.name.as_deref(), Some(name));
        assert_eq!(member.offset as usize, offset, "offset of {name}");
    }
    assert_eq!(*span as usize, std::mem::size_of::<SystemConstants>());
    assert_eq!(layouter[handle].size as usize, std::mem::size_of::<SystemConstants>());
    assert_eq!(SystemConstants::default().as_bytes().len(), 112);
}

#[test]
fn common_declarations_match_binding_constants() {
    let pixel = bindings::common_declarations(xenos_shader::ShaderStage::Pixel);
    assert!(pixel.contains("@group(0) @binding(4) var<storage, read> xe_float_constants: array<vec4<f32>, 256>;"));
    let decl = bindings::common_declarations(xenos_shader::ShaderStage::Vertex);
    assert!(decl.contains("@group(0) @binding(0) var<storage, read> xe_float_constants: array<vec4<f32>, 256>;"));
    assert!(decl.contains("@group(0) @binding(1) var<uniform> xe_bool_loop: XeBoolLoopConstants;"));
    assert!(decl.contains("@group(0) @binding(2) var<uniform> xe_fetch: XeFetchConstants;"));
    assert!(decl.contains("@group(0) @binding(3) var<uniform> xe_system: XeSystemConstants;"));
    for n in 0..4 {
        assert!(
            decl.contains(&format!("@group(1) @binding({n}) var<storage, read> xe_shared_memory_{n}: array<u32>;"))
        );
    }
    assert!(decl.contains("@group(1) @binding(4) var<storage, read> xe_shared_table: array<u32>;"));
    assert!(decl.contains("fn xe_shared_load(i: u32) -> u32"));
    let module = validate_wgsl(&decl);
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).unwrap();
    for (handle, ty) in module.types.iter() {
        match ty.name.as_deref() {
            Some("XeBoolLoopConstants") => {
                assert_eq!(layouter[handle].size as usize, std::mem::size_of::<xenos_shader::BoolLoopConstants>())
            }
            Some("XeFetchConstants") => assert_eq!(layouter[handle].size as u64, bindings::FETCH_CONSTANTS_SIZE),
            _ => {}
        }
    }
}

#[test]
fn vertex_expansion() {
    let vs = [exece(vec![
        output_export(ShaderStage::Vertex).into(),
        Alu::vector(2, 0, 0b1111, &[r(2), r(2)]).export().into(),
    ])];
    for (expansion, marker) in [
        (VertexExpansion::None, "return xe_vertex(xe_host_vertex_index);"),
        (VertexExpansion::PointSprite, "point_screen_diameter_to_ndc_radius"),
        (VertexExpansion::RectangleList, "out.o0 = b.o0 + c.o0 - a.o0;"),
    ] {
        for point_size in [None, Some(15)] {
            let options = TranslationOptions {
                vs_expansion: expansion,
                vs_point_size_location: point_size,
                ..Default::default()
            };
            let t = translate_with(ShaderStage::Vertex, &vs, &options);
            assert!(t.wgsl.contains(marker), "{expansion:?}:\n{}", t.wgsl);
        }
    }
}
