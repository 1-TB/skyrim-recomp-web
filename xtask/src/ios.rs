//! `cargo xtask build-ios`: the iOS app (hosts/host-ios) for a device.
//!
//! Builds the host-ios static library for aarch64-apple-ios and puts everything the Xcode
//! project in hosts/host-ios/xcode/ links into target/ios/: the library and local.xcconfig with
//! the signing team and bundle id. The app renders on Metal directly; with `--wgpu` it links
//! wgpu-native instead, embedded as a framework (its release ships a bare dylib whose install
//! name is the CI machine's path). Then generates the project with XcodeGen, and with
//! `--install` builds, signs and installs the app on the connected device (`--launch` also
//! starts it). With `--simulator` everything is built for the iOS Simulator instead (in
//! target/ios-sim/, unsigned) and `--install`/`--launch` target a simulator device.
//!
//! The guest is the one in the environment (XENON_RECOMP_OUT etc., as for the other hosts), or
//! with `--synthetic` the synthetic test title (`cargo xtask e2e` writes it).
//!
//! Environment: WGPU_NATIVE_DIR (with `--wgpu`; default ~/tools/wgpu-native/ios-aarch64, an
//! unpacked wgpu-ios-aarch64-release.zip), SKYRT_IOS_TEAM (default: the first team signed in to Xcode),
//! SKYRT_IOS_BUNDLE_ID (default dev.skyrt.recomp.<team>), SKYRT_IOS_DEVICE (a devicectl device
//! name or UDID; default the only connected physical device), SKYRT_IOS_SIMULATOR (simulator
//! name or UDID; default "iPhone 17 Pro Max").

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use crate::{Result, cargo, guest_env, recompile, repo_root, run_tool};

/// Matches project.yml and the guest C++ build (guest-sys build.rs reads the same variable).
const DEPLOYMENT_TARGET: &str = "17.0";
const TARGET: &str = "aarch64-apple-ios";
const SIMULATOR_TARGET: &str = "aarch64-apple-ios-sim";

pub fn build_ios(args: &[String]) -> Result<()> {
    let has = |f: &str| args.iter().any(|a| a == f);
    let release = has("--release");
    let launch = has("--launch");
    let install = has("--install") || launch;
    let use_wgpu = has("--wgpu");
    let simulator = has("--simulator");
    if simulator && use_wgpu {
        return Err("--wgpu is for devices only (no wgpu-native simulator framework is wired up)".into());
    }
    let target = if simulator { SIMULATOR_TARGET } else { TARGET };
    let root = repo_root();
    let out = root.join(if simulator { "target/ios-sim" } else { "target/ios" });
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;

    let wgpu = env::var_os("WGPU_NATIVE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("tools/wgpu-native/ios-aarch64"));
    let mut build = cargo();
    build
        .current_dir(&root)
        .args(["build", "-p", "host-ios", "--target", target])
        .env("IPHONEOS_DEPLOYMENT_TARGET", DEPLOYMENT_TARGET);
    if use_wgpu {
        build.env("WGPU_NATIVE_DIR", &wgpu).arg("--no-default-features");
    }
    if release {
        build.arg("--release");
    }
    if has("--synthetic") {
        let config = root.join("target/e2e/synthetic.toml");
        if !config.is_file() {
            return Err("no synthetic title in target/e2e; run `cargo xtask e2e` first".into());
        }
        guest_env(&mut build, &recompile(&config)?);
    } else if env::var_os("XENON_RECOMP_OUT").is_none() {
        return Err(
            "XENON_RECOMP_OUT is not set: `source generated/env.sh` (or pass --synthetic for the test title)".into()
        );
    }
    if !build.status().map_err(|e| e.to_string())?.success() {
        return Err("building host-ios failed".into());
    }
    let profile = if release { "release" } else { "debug" };
    let lib = root.join("target").join(target).join(profile).join("libhost_ios.a");
    copy(&lib, &out.join("libhost_ios.a"))?;
    let framework_dir = out.join("wgpu_native.framework");
    if use_wgpu {
        framework(&wgpu, &framework_dir)?;
    } else if framework_dir.exists() {
        fs::remove_dir_all(&framework_dir).map_err(|e| e.to_string())?;
    }

    let team = match env::var("SKYRT_IOS_TEAM") {
        Ok(t) => t,
        Err(_) => xcode_team()?,
    };
    let bundle =
        env::var("SKYRT_IOS_BUNDLE_ID").unwrap_or_else(|_| format!("dev.skyrt.recomp.{}", team.to_lowercase()));
    let xcconfig = format!(
        "// Written by `cargo xtask build-ios`.\nDEVELOPMENT_TEAM = {team}\nPRODUCT_BUNDLE_IDENTIFIER = {bundle}\n"
    );
    fs::write(out.join("local.xcconfig"), xcconfig).map_err(|e| e.to_string())?;
    let xcode = root.join("hosts/host-ios/xcode");
    // The spec as built: without wgpu-native's framework unless it is used.
    let spec = fs::read_to_string(xcode.join("project.yml")).map_err(|e| e.to_string())?;
    let spec = if use_wgpu { spec } else { without_wgpu_framework(&spec)? };
    let generated = xcode.join("project.gen.yml");
    fs::write(&generated, spec).map_err(|e| e.to_string())?;
    run_tool(
        Command::new("xcodegen").arg("generate").arg("--quiet").arg("--spec").arg(&generated),
        "xcodegen (brew install xcodegen)",
    )?;
    let project = xcode.join("SkyrimRecomp.xcodeproj");
    println!("library: {}\nproject: {} (team {team}, bundle id {bundle})", lib.display(), project.display());
    if !install {
        return Ok(());
    }

    // The C side is one file; the configuration only matters for its optimization.
    let derived = out.join("derived");
    if simulator {
        return run_simulator(&project, &out, &derived, &bundle, launch);
    }
    run_tool(
        Command::new("xcodebuild")
            .arg("-project")
            .arg(&project)
            .args(["-scheme", "Skyrim", "-configuration", "Release", "-destination", "generic/platform=iOS"])
            .arg("-derivedDataPath")
            .arg(&derived)
            .args(["-allowProvisioningUpdates", "-quiet", "build"]),
        "xcodebuild",
    )?;
    let app = derived.join("Build/Products/Release-iphoneos/Skyrim.app");
    let device = match env::var("SKYRT_IOS_DEVICE") {
        Ok(d) => d,
        Err(_) => connected_device()?,
    };
    run_tool(
        Command::new("xcrun").args(["devicectl", "device", "install", "app", "--device", &device]).arg(&app),
        "installing the app",
    )?;
    if launch {
        run_tool(
            Command::new("xcrun").args([
                "devicectl",
                "device",
                "process",
                "launch",
                "--terminate-existing",
                "--device",
                &device,
                &bundle,
            ]),
            "launching the app",
        )?;
    }
    Ok(())
}

/// Drops the `wgpu_native.framework` dependency (its line and the indented lines below it).
fn without_wgpu_framework(spec: &str) -> Result<String> {
    let mut out = String::new();
    let mut skip_indent = None;
    let mut found = false;
    for line in spec.lines() {
        let indent = line.len() - line.trim_start().len();
        if let Some(i) = skip_indent {
            if indent > i {
                continue;
            }
            skip_indent = None;
        }
        if line.trim_start().starts_with("- framework:") && line.contains("wgpu_native.framework") {
            skip_indent = Some(indent);
            found = true;
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !found {
        return Err("project.yml has no wgpu_native.framework dependency to drop".into());
    }
    Ok(out)
}

/// Builds the app for the simulator (unsigned, linking the library in `out`), boots the simulator
/// device and installs the app (`launch`: and starts it).
fn run_simulator(project: &Path, out: &Path, derived: &Path, bundle: &str, launch: bool) -> Result<()> {
    let device = env::var("SKYRT_IOS_SIMULATOR").unwrap_or_else(|_| "iPhone 17 Pro Max".to_string());
    run_tool(
        Command::new("xcodebuild")
            .arg("-project")
            .arg(project)
            .args(["-scheme", "Skyrim", "-configuration", "Release", "-sdk", "iphonesimulator"])
            .arg("-destination")
            .arg(format!("platform=iOS Simulator,name={device}"))
            .arg("-derivedDataPath")
            .arg(derived)
            .arg("SUPPORTED_PLATFORMS=iphonesimulator")
            .arg(format!("LIBRARY_SEARCH_PATHS={}", out.display()))
            .args(["CODE_SIGNING_ALLOWED=NO", "-quiet", "build"]),
        "xcodebuild (simulator)",
    )?;
    let app = derived.join("Build/Products/Release-iphonesimulator/Skyrim.app");
    // Booting an already booted device fails; that is fine.
    let _ = Command::new("xcrun").args(["simctl", "boot", &device]).status();
    // Simulator.app ships inside Xcode, where `open -a Simulator` does not look.
    if let Ok(dev) = Command::new("xcode-select").arg("-p").output() {
        let app = PathBuf::from(String::from_utf8_lossy(&dev.stdout).trim()).join("Applications/Simulator.app");
        let _ = Command::new("open").arg(app).status();
    }
    run_tool(Command::new("xcrun").args(["simctl", "install", &device]).arg(&app), "installing in the simulator")?;
    let data = Command::new("xcrun")
        .args(["simctl", "get_app_container", &device, bundle, "data"])
        .output()
        .map_err(|e| e.to_string())?;
    println!("simulator app data: {}", String::from_utf8_lossy(&data.stdout).trim());
    if launch {
        run_tool(
            Command::new("xcrun").args(["simctl", "launch", "--terminate-running-process", &device, bundle]),
            "launching in the simulator",
        )?;
    }
    Ok(())
}

fn home() -> PathBuf {
    env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

fn copy(from: &Path, to: &Path) -> Result<()> {
    fs::copy(from, to).map(|_| ()).map_err(|e| format!("{} -> {}: {e}", from.display(), to.display()))
}

/// Wraps wgpu-native's iOS dylib as `wgpu_native.framework`: iOS loads embedded libraries from
/// frameworks in the app's Frameworks folder, by an `@rpath` install name.
fn framework(wgpu: &Path, dir: &Path) -> Result<()> {
    let dylib = wgpu.join("lib/libwgpu_native.dylib");
    if !dylib.is_file() {
        return Err(format!(
            "{} not found: unpack wgpu-ios-aarch64-release.zip there or set WGPU_NATIVE_DIR",
            dylib.display()
        ));
    }
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let binary = dir.join("wgpu_native");
    copy(&dylib, &binary)?;
    run_tool(
        Command::new("install_name_tool").args(["-id", "@rpath/wgpu_native.framework/wgpu_native"]).arg(&binary),
        "install_name_tool",
    )?;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key>
	<string>wgpu_native</string>
	<key>CFBundleIdentifier</key>
	<string>org.gfx-rs.wgpu-native</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundlePackageType</key>
	<string>FMWK</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>CFBundleSupportedPlatforms</key>
	<array>
		<string>iPhoneOS</string>
	</array>
	<key>MinimumOSVersion</key>
	<string>{DEPLOYMENT_TARGET}</string>
</dict>
</plist>
"#
    );
    fs::write(dir.join("Info.plist"), plist).map_err(|e| e.to_string())
}

/// The first development team signed in to Xcode (Settings › Accounts), from Xcode's defaults.
fn xcode_team() -> Result<String> {
    let out = Command::new("defaults")
        .args(["read", "com.apple.dt.Xcode", "IDEProvisioningTeamByIdentifier"])
        .output()
        .map_err(|e| format!("defaults: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .find_map(|l| l.trim().strip_prefix("teamID = "))
        .map(|t| t.trim_end_matches(';').trim_matches('"').to_string())
        .ok_or_else(|| "no team signed in to Xcode (Settings › Accounts); or set SKYRT_IOS_TEAM".into())
}

/// The one connected physical iOS device's UDID (`devicectl list devices`).
fn connected_device() -> Result<String> {
    let out =
        Command::new("xcrun").args(["devicectl", "list", "devices"]).output().map_err(|e| format!("devicectl: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let devices: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("physical") && (l.contains("available") || l.contains("connected")))
        .filter_map(|l| l.split_whitespace().find(|w| w.len() >= 25 && w.contains('-')))
        .collect();
    match devices.as_slice() {
        [one] => Ok(one.to_string()),
        [] => Err("no connected iOS device (or set SKYRT_IOS_DEVICE)".into()),
        _ => Err(format!("several devices connected ({}); set SKYRT_IOS_DEVICE", devices.join(", "))),
    }
}
