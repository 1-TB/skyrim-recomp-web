//! `cargo xtask <command>`
//!
//!   setup-xenonrecomp              clone XenonRecomp at the pinned commit, apply tools/xenonrecomp/skyrt.patch, build
//!   decrypt --in IN.xex --out OUT.xex  write the decrypted, decompressed XEX the runtime loads
//!   switches --xex FILE.xex --out FILE.toml   find jump tables for XenonRecomp
//!   scan --xex FILE.xex            find save/restore helper and longjmp candidates for the config
//!   recompile --config FILE.toml   run XenonRecomp with cpp/ppc_context.h
//!   synth --out DIR                write the synthetic test title
//!   e2e [--web]                    synth -> recompile -> build -> run, checking output
//!   build-web [--release]          build host-web for wasm32-unknown-emscripten into web-dist/
//!   serve [--dir DIR] [--port N]   static server with COOP/COEP headers
//!
//! Tool locations: XENON_RECOMP_DIR (XenonRecomp checkout, default
//! ~/tools/XenonRecomp) and XENON_RECOMP_BIN (default <dir>/build/XenonRecomp/XenonRecomp).

mod ppc;
mod scan;
mod serve;
mod switches;
mod synth;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::{env, fs};

type Result<T> = std::result::Result<T, String>;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn xenon_dir() -> PathBuf {
    env::var_os("XENON_RECOMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env::var("HOME").unwrap_or_default()).join("tools/XenonRecomp"))
}

fn xenon_bin() -> PathBuf {
    env::var_os("XENON_RECOMP_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| xenon_dir().join("build/XenonRecomp/XenonRecomp"))
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("recompile") => match flag_value(&args, "--config") {
            Some(c) => recompile(Path::new(&c)).map(|out| {
                println!("\nexport XENON_RECOMP_OUT={}", out.out_dir.display());
                println!("export XENON_RECOMP_DIR={}", xenon_dir().display());
                println!("export SKYRIM_XEX={}", out.xex.display());
            }),
            None => Err("recompile needs --config FILE.toml".into()),
        },
        Some("setup-xenonrecomp") => setup_xenonrecomp(),
        Some("decrypt") => match (flag_value(&args, "--in"), flag_value(&args, "--out")) {
            (Some(i), Some(o)) => decrypt(Path::new(&i), Path::new(&o)),
            _ => Err("decrypt needs --in IN.xex --out OUT.xex".into()),
        },
        Some("switches") => match (flag_value(&args, "--xex"), flag_value(&args, "--out")) {
            (Some(x), Some(o)) => switches::run(Path::new(&x), Path::new(&o)),
            _ => Err("switches needs --xex FILE.xex --out FILE.toml".into()),
        },
        Some("scan") => match flag_value(&args, "--xex") {
            Some(x) => scan::scan(Path::new(&x)),
            None => Err("scan needs --xex FILE.xex".into()),
        },
        Some("synth") => {
            let out = flag_value(&args, "--out").map(PathBuf::from).unwrap_or_else(|| repo_root().join("target/synth"));
            synth::write(&out)
                .map_err(|e| e.to_string())
                .map(|()| println!("wrote synthetic title to {}", out.display()))
        }
        Some("e2e") => e2e(args.iter().any(|a| a == "--web")),
        Some("build-web") => build_web(args.iter().any(|a| a == "--release")).map(|_| ()),
        Some("serve") => {
            let dir = flag_value(&args, "--dir").map(PathBuf::from).unwrap_or_else(|| repo_root().join("web-dist"));
            let port = flag_value(&args, "--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
            serve::serve(&dir, port)
        }
        _ => {
            Err("usage: cargo xtask <setup-xenonrecomp|decrypt|scan|switches|recompile|synth|e2e|build-web|serve> ..."
                .into())
        }
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Recompiled {
    out_dir: PathBuf,
    xex: PathBuf,
}

/// Value of `key = "..."` in the [main] table (the only keys xtask needs).
fn toml_string(text: &str, key: &str) -> Option<String> {
    let mut in_main = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_main = line == "[main]";
            continue;
        }
        if !in_main {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        if k.trim() == key {
            let v = v.split('#').next().unwrap_or("").trim();
            return Some(v.trim_matches('"').to_string());
        }
    }
    None
}

fn recompile(config: &Path) -> Result<Recompiled> {
    let config = config.canonicalize().map_err(|e| format!("{}: {e}", config.display()))?;
    let dir = config.parent().unwrap();
    let text = fs::read_to_string(&config).map_err(|e| e.to_string())?;
    let file = toml_string(&text, "file_path").ok_or("config has no [main] file_path")?;
    let out = toml_string(&text, "out_directory_path").ok_or("config has no [main] out_directory_path")?;
    // With a title update patch, XenonRecomp recompiles the patched image.
    let xex = match toml_string(&text, "patched_file_path").filter(|p| !p.is_empty()) {
        Some(p) => dir.join(p),
        None => dir.join(file),
    };
    let out_dir = dir.join(out);

    if xex.exists() {
        let data = fs::read(&xex).map_err(|e| format!("{}: {e}", xex.display()))?;
        let parsed = xex::Xex::parse(&data).map_err(|e| format!("{}: {e}", xex.display()))?;
        println!(
            "{}: image {:#010X}+{:#X}, entry {:#010X}, {} imports",
            xex.display(),
            parsed.image_base(),
            parsed.image_size(),
            parsed.entry_point().unwrap_or(0),
            parsed.imports().map(|i| i.len()).unwrap_or(0)
        );
    } else if toml_string(&text, "patch_file_path").is_none() {
        return Err(format!("{} does not exist", xex.display()));
    }

    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    // The build globs ppc_recomp.*.cpp; a previous run may have written more
    // files than this one will.
    for entry in fs::read_dir(&out_dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("ppc_recomp.") && name.ends_with(".cpp") {
            fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let header = repo_root().join("cpp/ppc_context.h");
    let bin = xenon_bin();
    println!("running {} {} {}", bin.display(), config.display(), header.display());
    let status = Command::new(&bin)
        .arg(&config)
        .arg(&header)
        .status()
        .map_err(|e| format!("could not run {}: {e} (set XENON_RECOMP_BIN)", bin.display()))?;
    if !status.success() {
        return Err(format!("XenonRecomp failed ({status})"));
    }
    if !out_dir.join("ppc_recomp_shared.h").exists() {
        return Err(format!("XenonRecomp produced no output in {}", out_dir.display()));
    }
    Ok(Recompiled { out_dir, xex })
}

fn guest_env(cmd: &mut Command, r: &Recompiled) {
    cmd.env("XENON_RECOMP_OUT", &r.out_dir).env("XENON_RECOMP_DIR", xenon_dir()).env("SKYRIM_XEX", &r.xex);
}

fn cargo() -> Command {
    Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
}

fn e2e(web: bool) -> Result<()> {
    let root = repo_root();
    let dir = root.join("target/e2e");
    let _ = fs::remove_dir_all(&dir);
    synth::write(&dir).map_err(|e| e.to_string())?;
    // Round-trip the title through encryption so the decrypt tool's AES and
    // basic-decompression paths are exercised; the runtime then loads the
    // decrypted copy, whose image must match the original byte for byte.
    let plain = dir.join("default.xex");
    let encrypted = dir.join("encrypted.xex");
    run_tool(xex_decrypt_tool()?.arg("--encrypt").arg(&plain).arg(&encrypted), "xex-decrypt --encrypt")?;
    let decrypted = dir.join("game/default.xex");
    decrypt(&encrypted, &decrypted)?;
    let image = |p: &Path| -> Result<Vec<u8>> {
        let data = fs::read(p).map_err(|e| e.to_string())?;
        Ok(xex::Xex::parse(&data).map_err(|e| e.to_string())?.image().to_vec())
    };
    if image(&plain)? != image(&decrypted)? {
        return Err("decrypted image differs from the original".into());
    }
    let r = recompile(&dir.join("synthetic.toml"))?;

    let mut build = cargo();
    build.current_dir(&root).args(["build", "-p", "host-native"]);
    guest_env(&mut build, &r);
    if !build.status().map_err(|e| e.to_string())?.success() {
        return Err("building host-native failed".into());
    }
    let exe = root.join("target/debug/skyrim-native");
    let output = Command::new(&exe)
        .arg("--game-dir")
        .arg(dir.join("game"))
        .args(["--stub-return", &format!("{:#X}", synth::STUB_RETURN), "--log", "info"])
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    check_run("native", &String::from_utf8_lossy(&output.stderr), output.status.code())?;

    if web {
        let dist = build_web_with(false, Some(&r))?;
        let output = Command::new("node")
            .arg(dist.join("host-web.js"))
            .arg("--node-dir")
            .arg(dir.join("game"))
            .args(["--stub-return", &format!("{:#X}", synth::STUB_RETURN), "--log", "info"])
            .output()
            .map_err(|e| format!("node: {e}"))?;
        let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&output.stderr));
        check_run("web (node)", &log, output.status.code())?;
    }
    Ok(())
}

fn check_run(label: &str, log: &str, code: Option<i32>) -> Result<()> {
    let guest: Vec<&str> = log.lines().filter_map(|l| l.split_once("guest: ").map(|(_, m)| m)).collect();
    let mut missing = Vec::new();
    let mut pos = 0;
    for want in synth::expected_lines() {
        match guest[pos..].iter().position(|l| *l == want) {
            Some(i) => pos += i + 1,
            None => missing.push(want),
        }
    }
    let alloc_ok = guest.iter().any(|l| {
        l.strip_prefix("synth: alloc status=00000000 base=")
            .and_then(|b| u32::from_str_radix(b, 16).ok())
            .is_some_and(|b| (0x4000_0000..0x7000_0000).contains(&b))
    });
    let unlogged: Vec<&str> = ["NtDeviceIoControlFile", "xboxkrnl_3FF", "XamUserCheckPrivilege"]
        .into_iter()
        .filter(|n| !log.lines().any(|l| l.contains("unimplemented import") && l.contains(n)))
        .collect();
    let ok = missing.is_empty() && alloc_ok && unlogged.is_empty() && code == Some(synth::EXIT_CODE);
    if !ok {
        eprintln!("{log}");
        return Err(format!(
            "{label}: e2e failed (exit {code:?}, missing {missing:?}, alloc ok {alloc_ok}, \
             unimplemented-import logs missing {unlogged:?})"
        ));
    }
    println!("{label}: e2e passed ({} guest lines, exit {})", guest.len(), synth::EXIT_CODE);
    Ok(())
}

fn build_web(release: bool) -> Result<PathBuf> {
    build_web_with(release, None)
}

/// Builds host-web and assembles web-dist/ (or target/e2e/web for tests).
fn build_web_with(release: bool, e2e: Option<&Recompiled>) -> Result<PathBuf> {
    let root = repo_root();
    if env::var_os("EMSDK").is_none() {
        return Err("EMSDK is not set; run `source ~/emsdk/emsdk_env.sh` first".into());
    }
    let mut build = cargo();
    build.current_dir(&root).args([
        "build",
        "-p",
        "host-web",
        "--target",
        "wasm32-unknown-emscripten",
        "-Zbuild-std=std,panic_unwind",
    ]);
    if release {
        build.arg("--release");
    }
    if let Some(r) = e2e {
        guest_env(&mut build, r);
    }
    if !build.status().map_err(|e| e.to_string())?.success() {
        return Err("building host-web failed".into());
    }
    let profile = if release { "release" } else { "debug" };
    let out = root.join("target/wasm32-unknown-emscripten").join(profile);
    let dist = if e2e.is_some() { root.join("target/e2e/web") } else { root.join("web-dist") };
    fs::create_dir_all(&dist).map_err(|e| e.to_string())?;
    for name in ["host-web.js", "host_web.wasm"] {
        fs::copy(out.join(name), dist.join(name)).map_err(|e| format!("{name}: {e}"))?;
    }
    for entry in fs::read_dir(root.join("hosts/host-web/web")).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        fs::copy(entry.path(), dist.join(entry.file_name())).map_err(|e| e.to_string())?;
    }
    println!("web build in {}", dist.display());
    Ok(dist)
}

/// Builds tools/xex-decrypt against the XenonRecomp checkout (once) and
/// returns a command for it.
fn xex_decrypt_tool() -> Result<Command> {
    let root = repo_root();
    let build = root.join("target/xex-decrypt");
    let bin = build.join("xex-decrypt");
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(root.join("tools/xex-decrypt"))
        .arg("-B")
        .arg(&build)
        .args(["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
        .arg(format!("-DCMAKE_C_COMPILER={}", env::var("CC").unwrap_or_else(|_| "clang".into())))
        .arg(format!("-DCMAKE_CXX_COMPILER={}", env::var("CXX").unwrap_or_else(|_| "clang++".into())))
        .arg(format!("-DXENON_RECOMP_DIR={}", xenon_dir().display()));
    if !build.join("build.ninja").exists() {
        run_tool(&mut configure, "configuring xex-decrypt")?;
    }
    run_tool(Command::new("cmake").arg("--build").arg(&build), "building xex-decrypt")?;
    Ok(Command::new(bin))
}

fn run_tool(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd.status().map_err(|e| format!("{what}: {e}"))?;
    if status.success() { Ok(()) } else { Err(format!("{what} failed ({status})")) }
}

fn decrypt(input: &Path, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    run_tool(xex_decrypt_tool()?.arg(input).arg(output), "xex-decrypt")?;
    let data = fs::read(output).map_err(|e| e.to_string())?;
    let xex = xex::Xex::parse(&data).map_err(|e| format!("{}: {e}", output.display()))?;
    let sections = xex.sections().map_err(|e| e.to_string())?;
    let imports = xex.imports().map_err(|e| e.to_string())?;
    println!(
        "{}: image {:#010X}+{:#X}, entry {:#010X}, {} sections, {} imports",
        output.display(),
        xex.image_base(),
        xex.image_size(),
        xex.entry_point().unwrap_or(0),
        sections.len(),
        imports.len()
    );
    Ok(())
}

/// Upstream commit the skyrt patch applies to.
const XENON_RECOMP_COMMIT: &str = "ddd128bcca99fe8bfbb99bea583c972351fa6ace";

/// Clones XenonRecomp into XENON_RECOMP_DIR (if absent), checks out the pinned
/// commit, applies tools/xenonrecomp/skyrt.patch (if not already applied) and
/// builds XenonRecomp and XenonAnalyse.
fn setup_xenonrecomp() -> Result<()> {
    let dir = xenon_dir();
    let git = |args: &[&str]| -> Result<std::process::Output> {
        Command::new("git").arg("-C").arg(&dir).args(args).output().map_err(|e| format!("git: {e}"))
    };
    if !dir.join(".git").exists() {
        run_tool(
            Command::new("git")
                .args(["clone", "--recursive", "https://github.com/hedge-dev/XenonRecomp.git"])
                .arg(&dir),
            "cloning XenonRecomp",
        )?;
    }
    let head = String::from_utf8_lossy(&git(&["rev-parse", "HEAD"])?.stdout).trim().to_string();
    if head != XENON_RECOMP_COMMIT {
        let dirty = !git(&["status", "--porcelain", "--untracked-files=no"])?.stdout.is_empty();
        if dirty {
            return Err(format!("{} has local changes at {head}; expected {XENON_RECOMP_COMMIT}", dir.display()));
        }
        run_tool(Command::new("git").arg("-C").arg(&dir).args(["fetch", "origin", XENON_RECOMP_COMMIT]), "fetching")?;
        run_tool(Command::new("git").arg("-C").arg(&dir).args(["checkout", XENON_RECOMP_COMMIT]), "checkout")?;
        run_tool(
            Command::new("git").arg("-C").arg(&dir).args(["submodule", "update", "--init", "--recursive"]),
            "submodules",
        )?;
    }
    let patch = repo_root().join("tools/xenonrecomp/skyrt.patch");
    let applied = git(&["apply", "--check", "-R", patch.to_str().unwrap()])?.status.success();
    if applied {
        println!("skyrt patch already applied");
    } else {
        // An older revision of the patch: drop it from the tracked files first.
        let applies = git(&["apply", "--check", patch.to_str().unwrap()])?.status.success();
        if !applies {
            println!("replacing an older skyrt patch in {}", dir.display());
            run_tool(Command::new("git").arg("-C").arg(&dir).args(["checkout", "--", "."]), "resetting XenonRecomp")?;
        }
        run_tool(Command::new("git").arg("-C").arg(&dir).arg("apply").arg(&patch), "applying skyrt.patch")?;
        println!("applied {}", patch.display());
    }
    let build = dir.join("build");
    if !build.join("build.ninja").exists() {
        run_tool(
            Command::new("cmake").arg("-S").arg(&dir).arg("-B").arg(&build).args([
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
            ]),
            "configuring XenonRecomp",
        )?;
    }
    run_tool(Command::new("cmake").arg("--build").arg(&build), "building XenonRecomp")?;
    println!("XenonRecomp ready: {}", xenon_bin().display());
    Ok(())
}
