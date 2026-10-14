//! Builds the recompiled title and links it into the crate.
//!
//! Environment:
//!   XENON_RECOMP_OUT  directory XenonRecomp wrote (ppc_recomp.*.cpp, ppc_config.h, ...)
//!   XENON_RECOMP_DIR  XenonRecomp checkout (simde headers, export name tables)
//!   SKYRIM_XEX        the decrypted, decompressed XEX that was recompiled
//!   SKYRT_GUEST_BUILD_TYPE  CMake build type for the guest code (default Release)
//!   EMSDK             Emscripten SDK root (wasm32-unknown-emscripten builds)
//!
//! Generates `skyrt_imports.cpp`, which defines every import thunk as a call
//! into the Rust kernel (`skyrt_kernel_call(module, ordinal, ctx, base)`),
//! asserts the PPCContext layout runtime-core assumes, and exports the
//! function table and image constants.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use runtime_core::context::offsets;

fn fail(msg: impl AsRef<str>) -> ! {
    eprintln!("\nguest-sys: {}\n", msg.as_ref());
    std::process::exit(1);
}

fn env_path(name: &str, what: &str) -> PathBuf {
    match env::var_os(name) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => fail(format!("{name} is not set; it must point to {what}")),
    }
}

fn main() {
    for v in ["XENON_RECOMP_OUT", "XENON_RECOMP_DIR", "SKYRIM_XEX", "SKYRT_GUEST_BUILD_TYPE", "EMSDK", "CXX"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir =
        manifest_dir.join("../../cpp").canonicalize().unwrap_or_else(|e| fail(format!("cpp/ directory: {e}")));

    let recomp_out = env_path("XENON_RECOMP_OUT", "the XenonRecomp output directory");
    let xenon_dir = env_path("XENON_RECOMP_DIR", "the XenonRecomp checkout");
    let xex_path = env_path("SKYRIM_XEX", "the decrypted, decompressed default.xex that was recompiled");
    println!("cargo:rerun-if-changed={}", recomp_out.display());
    println!("cargo:rerun-if-changed={}", xex_path.display());
    println!("cargo:rerun-if-changed={}", cpp_dir.display());

    check_generated(&recomp_out);
    let declared = declared_symbols(&recomp_out);
    let names = export_names(&xenon_dir);

    let data = fs::read(&xex_path).unwrap_or_else(|e| fail(format!("reading {}: {e}", xex_path.display())));
    let xex = xex::Xex::parse(&data).unwrap_or_else(|e| fail(format!("{}: {e}", xex_path.display())));
    let libraries = xex.import_libraries().unwrap_or_else(|e| fail(format!("import table: {e}")));
    let lib_names: Vec<String> = libraries.iter().map(|l| l.name.clone()).collect();
    let ids = kernel_hle::assign_module_ids(&lib_names);
    let module_of: HashMap<&str, u32> = lib_names.iter().map(String::as_str).zip(ids.iter().map(|m| m.0)).collect();
    let imports = xex.imports().unwrap_or_else(|e| fail(format!("import table: {e}")));

    let shim = out_dir.join("skyrt_imports.cpp");
    let (cpp, rust) = generate(&imports, &module_of, &names, &declared);
    fs::write(&shim, cpp).unwrap();
    fs::write(out_dir.join("imports.rs"), rust).unwrap();

    let simde = xenon_dir.join("thirdparty/simde");
    if !simde.join("x86/avx.h").exists() {
        fail(format!("{} has no simde headers; clone XenonRecomp with --recursive", xenon_dir.display()));
    }
    let overlay = overlay_context_header(&recomp_out, &cpp_dir, &out_dir);
    let lib_dir = build_cmake(&cpp_dir, &out_dir, &overlay, &simde, &shim);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=skyrim_guest");
    if !env::var("TARGET").unwrap().contains("emscripten") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// The output must come from our forked header with every register kept in
/// the context, which is the layout runtime-core mirrors.
fn check_generated(dir: &Path) {
    let ctx = fs::read_to_string(dir.join("ppc_context.h"))
        .unwrap_or_else(|e| fail(format!("{}/ppc_context.h: {e}", dir.display())));
    if !ctx.contains("#define SKYRT_PPC_CONTEXT_FORK 3") {
        fail("ppc_context.h was not generated from the current cpp/ppc_context.h; rerun `cargo xtask recompile`");
    }
    let config = fs::read_to_string(dir.join("ppc_config.h"))
        .unwrap_or_else(|e| fail(format!("{}/ppc_config.h: {e}", dir.display())));
    // Output of an unpatched recompiler would address the physical views
    // without folding them (see skyrt_ea in cpp/ppc_context.h).
    // Without the write marks (PPC_SKYRT_DIRTY) the renderer would miss guest stores.
    if !config.contains("#define PPC_SKYRT_EA") || !config.contains("#define PPC_SKYRT_DIRTY") {
        fail(
            "the output was not produced by the skyrt-patched XenonRecomp; run `cargo xtask setup-xenonrecomp` and `cargo xtask recompile`",
        );
    }
    let flags: Vec<&str> = config
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define PPC_CONFIG_"))
        .filter(|f| !f.starts_with("H_INCLUDED"))
        .collect();
    if !flags.is_empty() {
        fail(format!(
            "the recompiler config enables {flags:?}; set every *_as_local / skip_* option to false \
             (runtime-core's PPCContext mirrors the all-in-context layout)"
        ));
    }
}

/// A copy of the recompiler output (symlinks) whose ppc_context.h is cpp/ppc_context.h.
///
/// XenonRecomp copies the header into its output when it runs, and the generated sources include
/// it from there; building against the current file instead lets the header's helpers (memory
/// access, view folding, dirty marking) change without recompiling the title, as long as the
/// macros the generated code uses stay the same (`SKYRT_PPC_CONTEXT_FORK`, checked above).
fn overlay_context_header(recomp_out: &Path, cpp_dir: &Path, out_dir: &Path) -> PathBuf {
    // Files are only replaced when they change, so unchanged sources aren't rebuilt.
    let overlay = out_dir.join("recomp-overlay");
    fs::create_dir_all(&overlay).unwrap_or_else(|e| fail(format!("{}: {e}", overlay.display())));
    let entries = fs::read_dir(recomp_out).unwrap_or_else(|e| fail(format!("{}: {e}", recomp_out.display())));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| fail(format!("{}: {e}", recomp_out.display())));
        let name = entry.file_name();
        if name == "ppc_context.h" {
            continue;
        }
        let target = recomp_out.join(&name).canonicalize().unwrap_or_else(|e| fail(format!("{name:?}: {e}")));
        let link = overlay.join(&name);
        if fs::read_link(&link).is_ok_and(|t| t == target) {
            continue;
        }
        let _ = fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link)
            .unwrap_or_else(|e| fail(format!("linking {}: {e}", target.display())));
    }
    // XenonRecomp prepends these two lines to its copy.
    let header = cpp_dir.join("ppc_context.h");
    let text = fs::read_to_string(&header).unwrap_or_else(|e| fail(format!("{}: {e}", header.display())));
    let text = format!("#pragma once\n#include \"ppc_config.h\"\n\n{text}");
    let copy = overlay.join("ppc_context.h");
    if fs::read_to_string(&copy).ok().as_deref() != Some(text.as_str()) {
        fs::write(&copy, text).unwrap_or_else(|e| fail(format!("{}: {e}", copy.display())));
    }
    overlay
}

fn declared_symbols(dir: &Path) -> HashSet<String> {
    let text = fs::read_to_string(dir.join("ppc_recomp_shared.h"))
        .unwrap_or_else(|e| fail(format!("{}/ppc_recomp_shared.h: {e}", dir.display())));
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("PPC_EXTERN_FUNC(")?.strip_suffix(");").map(str::to_string))
        .collect()
}

/// (module id, ordinal) -> name, from XenonRecomp's copy of Xenia's tables.
fn export_names(xenon_dir: &Path) -> HashMap<(u32, u16), String> {
    let mut names = HashMap::new();
    for (file, module) in [("xboxkrnl_table.inc", 0u32), ("xam_table.inc", 1u32)] {
        let path = xenon_dir.join("XenonUtils/xbox").join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(&path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
        for line in text.lines().filter(|l| l.starts_with("XE_EXPORT(")) {
            let parts: Vec<&str> = line["XE_EXPORT(".len()..].split(',').map(str::trim).collect();
            if parts.len() < 3 {
                continue;
            }
            if let Ok(ordinal) = u16::from_str_radix(parts[1].trim_start_matches("0x"), 16) {
                names.insert((module, ordinal), parts[2].to_string());
            }
        }
    }
    names
}

fn generate(
    imports: &[xex::Import],
    module_of: &HashMap<&str, u32>,
    names: &HashMap<(u32, u16), String>,
    declared: &HashSet<String>,
) -> (String, String) {
    let mut cpp = String::new();
    let mut rust = String::new();
    writeln!(cpp, "// Generated by crates/guest-sys/build.rs.").unwrap();
    writeln!(cpp, "#include \"ppc_recomp_shared.h\"").unwrap();
    writeln!(cpp, "#include <cstddef>\n#include <new>\n").unwrap();
    writeln!(
        cpp,
        "extern \"C\" void skyrt_kernel_call(uint32_t module, uint32_t ordinal, PPCContext* ctx, uint8_t* base);\n"
    )
    .unwrap();

    // Layout runtime-core relies on.
    let checks = [
        ("r3", offsets::R3),
        ("r0", offsets::R0),
        ("r1", offsets::R1),
        ("r2", offsets::R2),
        ("r4", offsets::R4),
        ("r13", offsets::R13),
        ("r31", offsets::R31),
        ("lr", offsets::LR),
        ("ctr", offsets::CTR),
        ("xer", offsets::XER),
        ("reserved", offsets::RESERVED),
        ("msr", offsets::MSR),
        ("cr0", offsets::CR0),
        ("fpscr", offsets::FPSCR),
        ("f0", offsets::F0),
        ("f1", offsets::F1),
        ("v0", offsets::V0),
        ("v127", offsets::V127),
    ];
    for (field, off) in checks {
        writeln!(cpp, "static_assert(offsetof(PPCContext, {field}) == {off}, \"PPCContext.{field} moved\");").unwrap();
    }
    writeln!(cpp, "static_assert(sizeof(PPCContext) == {}, \"PPCContext size changed\");", offsets::SIZE).unwrap();
    writeln!(cpp, "static_assert(alignof(PPCContext) == {}, \"PPCContext alignment changed\");\n", offsets::ALIGN)
        .unwrap();

    writeln!(rust, "// Generated by crates/guest-sys/build.rs.").unwrap();
    writeln!(rust, "pub static IMPORT_NAMES: &[kernel_hle::dispatch::ImportName] = &[").unwrap();

    let mut defined = BTreeMap::new();
    for imp in imports.iter() {
        let Some(&module) = module_of.get(imp.library.as_str()) else { continue };
        let name = names.get(&(module, imp.ordinal));
        if let Some(n) = name {
            writeln!(
                rust,
                "    kernel_hle::dispatch::ImportName {{ module: {module}, ordinal: {:#05X}, name: {n:?} }},",
                imp.ordinal
            )
            .unwrap();
        }
        let Some(thunk) = imp.thunk_addr else { continue };
        // XenonRecomp names named thunks `__imp__<Name>`; thunks for ordinals it
        // has no name for are recompiled as ordinary `sub_<ADDR>` stubs, which a
        // strong definition here overrides.
        let symbol = match name {
            Some(n) if declared.contains(&format!("__imp__{n}")) => format!("__imp__{n}"),
            _ => format!("sub_{thunk:X}"),
        };
        if !declared.contains(&symbol) {
            println!(
                "cargo:warning=import {}#{:#05X} thunk {thunk:#010X} has no recompiled symbol ({symbol}); calls to it cannot be routed",
                imp.library, imp.ordinal
            );
            continue;
        }
        defined.insert(symbol, (module, imp.ordinal));
    }
    writeln!(rust, "];").unwrap();

    for (symbol, (module, ordinal)) in &defined {
        writeln!(cpp, "PPC_FUNC({symbol}) {{ skyrt_kernel_call({module}u, {ordinal:#05X}u, &ctx, base); }}").unwrap();
    }

    cpp.push_str(
        r#"
extern "C" void skyrt_init_context(PPCContext* ctx) {
    new (ctx) PPCContext();
    ctx->fpscr.loadFromHost();
}

extern "C" const PPCFuncMapping* skyrt_func_mappings(size_t* count) {
    size_t n = 0;
    while (PPCFuncMappings[n].guest != 0) ++n;
    *count = n;
    return PPCFuncMappings;
}

extern "C" void skyrt_ppc_config(uint64_t* out) {
    out[0] = PPC_IMAGE_BASE;
    out[1] = PPC_IMAGE_SIZE;
    out[2] = PPC_CODE_BASE;
    out[3] = PPC_CODE_SIZE;
}
"#,
    );
    (cpp, rust)
}

fn build_cmake(cpp_dir: &Path, out_dir: &Path, recomp_out: &Path, simde: &Path, shim: &Path) -> PathBuf {
    let build_dir = out_dir.join("cmake-build");
    let target = env::var("TARGET").unwrap();
    let build_type = env::var("SKYRT_GUEST_BUILD_TYPE").unwrap_or_else(|_| "Release".into());

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(cpp_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-G")
        .arg("Ninja")
        .arg(format!("-DCMAKE_BUILD_TYPE={build_type}"))
        .arg(format!("-DXENON_RECOMP_OUT={}", recomp_out.display()))
        .arg(format!("-DSIMDE_DIR={}", simde.display()))
        .arg(format!("-DSKYRT_SHIM={}", shim.display()));
    if target.contains("emscripten") {
        let emsdk = env_path("EMSDK", "the Emscripten SDK (source emsdk_env.sh)");
        let toolchain = emsdk.join("upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake");
        configure.arg(format!("-DCMAKE_TOOLCHAIN_FILE={}", toolchain.display()));
    } else {
        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".into());
        configure.arg(format!("-DCMAKE_CXX_COMPILER={cxx}"));
    }
    run(&mut configure, "cmake configure");

    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "4".into());
    run(Command::new("cmake").arg("--build").arg(&build_dir).arg("--parallel").arg(jobs), "cmake build");
    build_dir
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|e| fail(format!("{what}: could not start: {e}")));
    if !status.success() {
        fail(format!("{what} failed ({status})"));
    }
}
