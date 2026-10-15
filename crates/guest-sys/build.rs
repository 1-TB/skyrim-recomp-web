//! Builds the recompiled title and links it into the crate.
//!
//! Environment:
//!   XENON_RECOMP_OUT  directory XenonRecomp wrote (ppc_recomp.*.cpp, ppc_config.h, ...)
//!   XENON_RECOMP_DIR  XenonRecomp checkout (simde headers, export name tables)
//!   SKYRIM_XEX        the decrypted, decompressed XEX that was recompiled
//!   SKYRT_GUEST_BUILD_TYPE  CMake build type for the guest code (default Release)
//!   SKYRT_CHECK_FRAMES      1: report scalar guest accesses straddling non-adjacent linear
//!                           frames (debugging the compact memory layout; native only)
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
    for v in ["XENON_RECOMP_OUT", "XENON_RECOMP_DIR", "SKYRIM_XEX", "SKYRT_GUEST_BUILD_TYPE", "SKYRT_CHECK_FRAMES", "EMSDK", "CXX"] {
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
    let (mut cpp, rust) = generate(&imports, &module_of, &names, &declared);
    cpp.push_str(&title_overrides(xex.execution_info().map(|e| (e.title_id, e.version)), &declared));
    fs::write(&shim, cpp).unwrap();
    fs::write(out_dir.join("imports.rs"), rust).unwrap();

    let simde = xenon_dir.join("thirdparty/simde");
    if !simde.join("x86/avx.h").exists() {
        fail(format!("{} has no simde headers; clone XenonRecomp with --recursive", xenon_dir.display()));
    }
    let overlay = overlay_context_header(&recomp_out, &cpp_dir, &out_dir);
    let dispatch_cpp = out_dir.join("skyrt_dispatch.cpp");
    let text = dispatch(&recomp_out);
    if fs::read_to_string(&dispatch_cpp).ok().as_deref() != Some(text.as_str()) {
        fs::write(&dispatch_cpp, text).unwrap();
    }
    let lib_dir = build_cmake(&cpp_dir, &out_dir, &overlay, &simde, &shim, &dispatch_cpp);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=skyrim_guest");
    if !env::var("TARGET").unwrap().contains("emscripten") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// Host replacements for guest functions of known titles, wrapping the recompiled function
/// (a strong definition of `sub_X` overrides the recompiler's weak alias; `__imp__sub_X` stays
/// the recompiled body).
fn title_overrides(title: Option<(u32, u32)>, declared: &HashSet<String>) -> String {
    let mut cpp = String::new();
    // Skyrim, disc version. D3D's wait for ring-buffer space calls this between checks of the
    // read pointer (it spins four iterations and checks for a hang; nonzero = keep waiting).
    // Spinning there kept the render thread busy whenever the GPU was behind; instead sleep
    // until the command processor writes the read pointer back (or 1 ms passes).
    const SKYRIM_RING_WAIT: &str = "sub_82C8B818";
    let skyrim = title == Some((0x4253_07E6, 0x0000_0A02));
    // zlib's deflate in the title: deflateInit2_, deflate, deflateEnd. The title compresses
    // buffers with it while playing (one call took ~330 ms natively, ~560 ms in the browser,
    // and the frame waits for it); kernel-hle's zlib module does it on the host. Each override
    // falls back to the recompiled function for streams the host doesn't handle.
    // Inflate (the loader's BSA reads) goes to zlib-rs the same way; `inflate`'s fallback is
    // followed by a check hook so `SKYRT_ZLIB_VERIFY` can compare every call with the host's.
    const SKYRIM_ZLIB: [(&str, &str, &str); 7] = [
        ("sub_82CA7688", "skyrt_deflate_init(ctx.r3.u32, ctx.r4.s32, ctx.r5.s32, ctx.r6.s32, ctx.r8.s32)", ""),
        ("sub_82CA5798", "skyrt_deflate(ctx.r3.u32, ctx.r4.s32)", ""),
        ("sub_82CA6360", "skyrt_deflate_end(ctx.r3.u32)", ""),
        ("sub_823359D0", "skyrt_inflate_init(ctx.r3.u32)", ""),
        ("sub_82335CF0", "skyrt_inflate(ctx.r3.u32, ctx.r4.s32)", "skyrt_inflate_verify(strm, ctx.r3.s32);"),
        ("sub_82334C38", "skyrt_inflate_reset(ctx.r3.u32)", ""),
        ("sub_82CA5640", "skyrt_inflate_end(ctx.r3.u32)", ""),
    ];
    if skyrim && SKYRIM_ZLIB.iter().all(|(f, _, _)| declared.contains(*f)) {
        cpp.push_str(
            r#"
extern "C" int32_t skyrt_deflate_init(uint32_t strm, int32_t level, int32_t method, int32_t window_bits, int32_t strategy);
extern "C" int32_t skyrt_deflate(uint32_t strm, int32_t flush);
extern "C" int32_t skyrt_deflate_end(uint32_t strm);
extern "C" int32_t skyrt_inflate_init(uint32_t strm);
extern "C" int32_t skyrt_inflate(uint32_t strm, int32_t flush);
extern "C" void skyrt_inflate_verify(uint32_t strm, int32_t ret);
extern "C" int32_t skyrt_inflate_reset(uint32_t strm);
extern "C" int32_t skyrt_inflate_end(uint32_t strm);
"#,
        );
        for (f, host, after) in SKYRIM_ZLIB {
            writeln!(
                cpp,
                r#"
extern "C" PPC_FUNC(__imp__{f});
PPC_FUNC({f}) {{
    uint32_t strm = ctx.r3.u32;
    int32_t r = {host};
    if (r == INT32_MIN) {{
        __imp__{f}(ctx, base);
        {after}
        return;
    }}
    (void)strm;
    ctx.r3.s64 = r;
}}"#
            )
            .unwrap();
        }
    }
    // The title's Direct3D entry points (d3d-hle's list): each calls kernel-hle's d3d hook, which
    // does nothing unless `--gpu` asks for probing, then runs the original.
    if title == Some(d3d_hle::TITLE) {
        let hooks: Vec<(usize, &d3d_hle::Hook)> =
            d3d_hle::HOOKS.iter().enumerate().filter(|(_, h)| declared.contains(h.function)).collect();
        if hooks.len() != d3d_hle::HOOKS.len() {
            println!("cargo:warning=some D3D functions to hook are missing from the recompiled output");
        }
        cpp.push_str(
            "\nextern \"C\" void skyrt_d3d_enter(uint32_t id, PPCContext* ctx);\n\
             extern \"C\" void skyrt_d3d_leave(uint32_t id, PPCContext* ctx);\n",
        );
        for (id, hook) in hooks {
            writeln!(
                cpp,
                r#"
// {name}
extern "C" PPC_FUNC(__imp__{f});
PPC_FUNC({f}) {{
    skyrt_d3d_enter({id}, &ctx);
    __imp__{f}(ctx, base);
    skyrt_d3d_leave({id}, &ctx);
}}"#,
                name = hook.name,
                f = hook.function,
            )
            .unwrap();
        }
    }
    if skyrim && declared.contains(SKYRIM_RING_WAIT) {
        writeln!(
            cpp,
            r#"
extern "C" uint32_t skyrt_gpu_progress(void);
extern "C" void skyrt_gpu_wait_progress(uint32_t seen, uint32_t timeout_us);
extern "C" PPC_FUNC(__imp__{f});
PPC_FUNC({f}) {{
    uint32_t seen = skyrt_gpu_progress();
    __imp__{f}(ctx, base);
    if (ctx.r3.u32 != 0) skyrt_gpu_wait_progress(seen, 1000);
}}"#,
            f = SKYRIM_RING_WAIT
        )
        .unwrap();
    }
    cpp
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
    // Registers the recompiler may keep in C++ locals: cpp/ppc_context.h keeps every field in
    // PPCContext regardless, so runtime-core's layout holds, and the kernel only reads the
    // argument registers, r1, r13 and LR, which stay in the context. Skipping LR or MSR would
    // drop state the runtime and the skyrt patch rely on (blrl, guest backtraces).
    const ALLOWED: [&str; 6] =
        ["CTR_AS_LOCAL", "XER_AS_LOCAL", "RESERVED_AS_LOCAL", "CR_AS_LOCAL", "NON_ARGUMENT_AS_LOCAL", "NON_VOLATILE_AS_LOCAL"];
    let flags: Vec<&str> = config
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define PPC_CONFIG_"))
        .filter(|f| !f.starts_with("H_INCLUDED") && !ALLOWED.contains(f))
        .collect();
    if !flags.is_empty() {
        fail(format!("the recompiler config enables {flags:?}; set skip_lr and skip_msr to false"));
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

/// `skyrt_dispatch.cpp`: indirect calls into recompiled code (`PPC_CALL_INDIRECT_FUNC` and
/// runtime-core's `Runtime::call`), from XenonRecomp's function list (ppc_func_mapping.cpp, which
/// isn't compiled).
///
/// XenonRecomp's own scheme stores a function pointer per guest instruction in a table after the
/// image. On wasm that table is ~38 MiB of touched guest memory, and every function whose address
/// is taken also takes an entry in each instance's funcref table, which Safari spends ~5 MiB on
/// per thread for 87k functions. Here each function has an id (its index in the address-ordered
/// list): a bitmap per 64 instructions says which start a function, the id is the block's first
/// id plus a popcount, and a switch on it makes direct calls, which are code and shared.
fn dispatch(recomp_out: &Path) -> String {
    let config = fs::read_to_string(recomp_out.join("ppc_config.h"))
        .unwrap_or_else(|e| fail(format!("{}/ppc_config.h: {e}", recomp_out.display())));
    let define = |name: &str| -> u32 {
        config
            .lines()
            .find_map(|l| l.trim().strip_prefix("#define ")?.strip_prefix(name)?.trim().strip_prefix("0x"))
            .and_then(|v| u32::from_str_radix(v.trim_end_matches("ull"), 16).ok())
            .unwrap_or_else(|| fail(format!("ppc_config.h: no {name}")))
    };
    let (code_base, code_size) = (define("PPC_CODE_BASE"), define("PPC_CODE_SIZE"));

    let path = recomp_out.join("ppc_func_mapping.cpp");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    let mut functions: Vec<(u32, &str)> = text
        .lines()
        .filter_map(|l| {
            let (addr, name) = l.trim().strip_prefix("{ 0x")?.strip_suffix(" },")?.split_once(", ")?;
            Some((u32::from_str_radix(addr, 16).ok()?, name))
        })
        .filter(|&(addr, _)| addr != 0)
        .collect();
    functions.sort_unstable();
    functions.dedup_by_key(|f| f.0);
    if functions.is_empty() {
        fail(format!("{}: no functions", path.display()));
    }
    for &(addr, name) in &functions {
        if addr % 4 != 0 || addr.wrapping_sub(code_base) >= code_size {
            fail(format!("{name} at {addr:#010X} is outside the code range"));
        }
    }

    let blocks = (code_size / 4).div_ceil(64) as usize;
    let mut first = vec![0u32; blocks];
    let mut starts = vec![0u64; blocks];
    let mut next = 0;
    for (block, (first, starts)) in first.iter_mut().zip(&mut starts).enumerate() {
        *first = next as u32;
        while let Some(&(addr, _)) = functions.get(next) {
            let index = ((addr - code_base) / 4) as usize;
            if index / 64 != block {
                break;
            }
            *starts |= 1 << (index % 64);
            next += 1;
        }
    }

    let mut cpp = String::new();
    writeln!(cpp, "// Generated by crates/guest-sys/build.rs (fn dispatch).").unwrap();
    writeln!(cpp, "#include \"ppc_recomp_shared.h\"\n#include <cstddef>\n").unwrap();
    writeln!(cpp, "static_assert(PPC_CODE_BASE == {code_base:#X}u && PPC_CODE_SIZE == {code_size:#X}u);\n").unwrap();
    writeln!(cpp, "static const uint32_t skyrt_func_addr_list[{}] = {{", functions.len()).unwrap();
    for chunk in functions.chunks(8) {
        let line: Vec<String> = chunk.iter().map(|(a, _)| format!("{a:#X}")).collect();
        writeln!(cpp, "    {},", line.join(", ")).unwrap();
    }
    writeln!(cpp, "}};\n").unwrap();
    writeln!(cpp, "// Per 64 instructions of code: the id of the first function starting there,").unwrap();
    writeln!(cpp, "// and which instructions start one.").unwrap();
    writeln!(cpp, "static const uint32_t skyrt_func_first[{blocks}] = {{").unwrap();
    for chunk in first.chunks(16) {
        let line: Vec<String> = chunk.iter().map(u32::to_string).collect();
        writeln!(cpp, "    {},", line.join(", ")).unwrap();
    }
    writeln!(cpp, "}};\nstatic const uint64_t skyrt_func_starts[{blocks}] = {{").unwrap();
    for chunk in starts.chunks(8) {
        let line: Vec<String> = chunk.iter().map(|s| format!("{s:#X}ull")).collect();
        writeln!(cpp, "    {},", line.join(", ")).unwrap();
    }
    writeln!(cpp, "}};\n").unwrap();
    writeln!(
        cpp,
        r#"extern "C" const uint32_t* skyrt_func_addrs(size_t* count) {{
    *count = {count};
    return skyrt_func_addr_list;
}}
"#,
        count = functions.len(),
    )
    .unwrap();
    // Switches over GROUP ids each: one switch over all of them is a single function too big for
    // wasm-opt (it didn't finish in an hour) and for Safari's optimizing tier (over 100 KB).
    const GROUP: usize = 1024;
    let groups = functions.len().div_ceil(GROUP);
    for (group, chunk) in functions.chunks(GROUP).enumerate() {
        writeln!(
            cpp,
            "__attribute__((noinline)) static void skyrt_call_group_{group}(uint32_t id, PPCContext& __restrict ctx, uint8_t* base) {{\n    switch (id) {{"
        )
        .unwrap();
        for (i, (_, name)) in chunk.iter().enumerate() {
            writeln!(cpp, "    case {}: {name}(ctx, base); return;", group * GROUP + i).unwrap();
        }
        writeln!(cpp, "    }}\n}}\n").unwrap();
    }
    writeln!(
        cpp,
        r#"extern "C" [[noreturn]] void skyrt_missing_function(uint32_t addr);

extern "C" void skyrt_call_func(uint32_t addr, PPCContext& __restrict ctx, uint8_t* base) {{
    uint32_t index = (addr - {code_base:#X}u) >> 2;
    if ((addr & 3) == 0 && index < {limit}u) {{
        uint64_t starts = skyrt_func_starts[index >> 6];
        uint64_t bit = 1ull << (index & 63);
        if (starts & bit) {{
            uint32_t id = skyrt_func_first[index >> 6] + (uint32_t)__builtin_popcountll(starts & (bit - 1));
            switch (id / {GROUP}) {{"#,
        limit = blocks * 64,
    )
    .unwrap();
    for group in 0..groups {
        writeln!(cpp, "            case {group}: skyrt_call_group_{group}(id, ctx, base); return;").unwrap();
    }
    writeln!(cpp, "            }}\n        }}\n    }}\n    skyrt_missing_function(addr);\n}}").unwrap();
    cpp
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

fn build_cmake(cpp_dir: &Path, out_dir: &Path, recomp_out: &Path, simde: &Path, shim: &Path, dispatch: &Path) -> PathBuf {
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
        .arg(format!("-DSKYRT_SHIM={}", shim.display()))
        .arg(format!("-DSKYRT_DISPATCH={}", dispatch.display()))
        .arg(format!(
            "-DSKYRT_CHECK_FRAMES={}",
            if env::var("SKYRT_CHECK_FRAMES").as_deref() == Ok("1") && !target.contains("emscripten") { "ON" } else { "OFF" }
        ));
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
