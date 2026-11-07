//! iOS host: the Xcode project in `xcode/` links this static library and its `main` calls
//! [`skyrt_ios_main`], which hands the main thread to UIKit (see app.rs) and boots the title on
//! another thread once the window exists.
//!
//! Everything lives in the app's Documents folder, which Finder (USB file sharing) and the Files
//! app can see:
//!
//! * `game/`: the title files, as `extract-xiso` writes them, with the decrypted `default.xex`
//!   (the app waits until it appears, so copy it last);
//! * `content/`: saves and other title content;
//! * `logs/`: one log per run (stderr is copied there; Xcode's console still shows it);
//! * `settings.txt`: the choices made in the settings panel shown at launch (settings.rs);
//! * `args.txt` (optional): extra command-line options, as for `skyrim-native`, one or more per
//!   line; lines starting with `#` are ignored. They come after the settings' options, so they
//!   override them. `--autoplay` starts the title with the saved settings without showing the
//!   settings panel. `--dump-frames DIR --dump-every N` writes every
//!   Nth presented frame to `DIR` (in Documents) as PNG.

mod app;
#[path = "../../host-native/src/audio_out.rs"]
mod audio_out;
#[path = "../../common/boot.rs"]
mod boot;
#[path = "../../host-native/src/crash.rs"]
mod crash;
#[path = "../../common/gpu.rs"]
mod gpu;
mod pads;
mod settings;
mod status;
mod touch;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The app's Documents folder (`$HOME` is the app's container on iOS).
fn documents() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp")).join("Documents")
}

/// Entry point, called from `main` in `xcode/main.c`. Never returns: UIKit owns the main thread.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_ios_main() {
    let docs = documents();
    for dir in ["game", "content", "logs"] {
        let _ = std::fs::create_dir_all(docs.join(dir));
    }
    // Relative paths (the renderer's trace output, target/shader-dump) land in Documents too;
    // the process starts in / otherwise, which the app can't write.
    let _ = std::env::set_current_dir(&docs);
    tee_stderr(&docs.join("logs"));
    let mut extra = read_args(&docs.join("args.txt"));
    // `--autoplay`: start with the saved settings, without the settings panel (scripted runs).
    let autoplay = extra.iter().any(|a| a == "--autoplay");
    extra.retain(|a| a != "--autoplay");
    // The log level is needed now; the other options are read once Play is tapped.
    let log_level = extra.iter().position(|a| a == "--log").and_then(|i| extra.get(i + 1)).cloned();
    if let Err(e) = kernel_hle::logging::init(log_level.as_deref().unwrap_or("info")) {
        eprintln!("{e}");
    }
    log::info!("documents: {}", docs.display());
    crash::install();
    runtime_core::thread::set_thread_start_hook(crash::install_alt_stack);
    let settings_path = docs.join("settings.txt");
    let result = app::run(settings_path, autoplay, move |window, settings| {
        let mut args = vec![
            "--game-dir".to_string(),
            docs.join("game").display().to_string(),
            "--content".to_string(),
            docs.join("content").display().to_string(),
        ];
        args.extend(settings.args());
        args.extend(extra);
        let mut opts = match boot::parse(args.into_iter(), &["--dump-frames", "--dump-every"]) {
            Ok(o) => o,
            Err(msg) => {
                // The usage text is no help on a phone; show what was wrong with args.txt.
                let first = msg.lines().next().unwrap_or_default().to_string();
                eprintln!("args.txt: {msg}");
                status::set(format!("args.txt: {first}"));
                let mut o = boot::Options { log: "info".into(), ..Default::default() };
                o.game_dir = Some(docs.join("game"));
                o.content = Some(docs.join("content"));
                o
            }
        };
        opts.vblank_hz.get_or_insert(boot::NATIVE_VBLANK_HZ);
        let spawned =
            std::thread::Builder::new().name("boot".into()).stack_size(16 << 20).spawn(move || start(opts, window));
        if let Err(e) = spawned {
            log::error!("starting the title: {e}");
            status::set(format!("Could not start: {e}"));
        }
    });
    // run_app hands the thread to UIApplicationMain, which only returns on failure.
    if let Err(e) = result {
        log::error!("window: {e}");
    }
}

/// The boot thread: waits for the game files, then runs the title until it exits.
fn start(opts: boot::Options, window: xenos_render::native::Window) {
    let game_root = opts.game_dir.clone().expect("set by skyrt_ios_main");
    let xex = opts.xex.clone().unwrap_or_else(|| game_root.join("default.xex"));
    if !xex.is_file() {
        status::set(
            "Copy your game files into the Skyrim app's \"game\" folder (Finder or the Files app): \
             the Data folder first, then the decrypted default.xex.\n\nThe game starts once default.xex is there."
                .to_string(),
        );
        log::info!("waiting for {}", xex.display());
        while !xex.is_file() {
            std::thread::sleep(Duration::from_secs(2));
        }
        // Let a copy that is still writing the file finish.
        std::thread::sleep(Duration::from_secs(3));
    }
    status::set("Starting…".to_string());
    let content = opts.content.clone().expect("set by skyrt_ios_main");
    let game = vfs::GameRoot::Dir(game_root);
    // `--dump-frames DIR` (in Documents) and `--dump-every N` write presented frames as PNG.
    let mut render = xenos_render::native::NativeOptions { dump_dir: None, dump_every: 60, window: Some(window) };
    for (flag, value) in &opts.extra {
        match flag.as_str() {
            "--dump-frames" => render.dump_dir = Some(documents().join(value)),
            "--dump-every" => match value.parse() {
                Ok(n) => render.dump_every = n,
                Err(e) => log::warn!("--dump-every {value}: {e}"),
            },
            _ => {}
        }
    }
    let on_kernel = move |kernel: &'static kernel_hle::Kernel| on_kernel(kernel, render);
    match boot::run(&opts, &game, &content, on_kernel) {
        Ok(code) => {
            log::info!("title exited with {code:#X}");
            status::set(format!("The game exited (code {code:#X}). Close the app to start it again."));
        }
        Err(e) => {
            log::error!("{e}");
            status::set(format!("The game could not start:\n{e}"));
        }
    }
}

fn on_kernel(kernel: &'static kernel_hle::Kernel, render: xenos_render::native::NativeOptions) {
    crash::set_kernel(kernel);
    app::attach(kernel);
    audio_out::install(kernel);
    gpu::start_memory_report();
    // The command processor waits for the renderer at fences, as in the browser on phones. The
    // GPU memory budgets stay the desktop ones: the browser's phone budgets (a 56 MiB texture
    // cache, new textures loaded at 16 MiB/s) are sized for a tab the system closes early, and
    // in a native app they left busy scenes drawn without textures. `--render-debug handheld`
    // in args.txt restores them.
    kernel_hle::gpu::set_sync_default(true);
    // A native app has room a browser tab doesn't (an iPhone 18 Pro Max leaves ~2 GiB free at
    // 1.5 GiB used): twice the desktop texture cache, so fewer textures are dropped and reloaded.
    xenos_render::default_texture_budget_mib(1024);
    let memory = std::sync::Arc::new(gpu::GuestPhysical(kernel));
    match xenos_render::native::start(memory, render, gpu::render_targets()) {
        Ok(handle) => {
            gpu::install(kernel, handle);
            status::clear_when_presenting(kernel);
        }
        Err(e) => {
            log::error!("starting the renderer: {e}");
            status::set(format!("The renderer could not start: {e}"));
        }
    }
}

/// Options from `args.txt`: whitespace-separated, `#` lines skipped. Quoting isn't supported;
/// paths in the Documents folder have no spaces unless the user adds some.
fn read_args(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| l.split_whitespace().map(String::from).collect::<Vec<_>>())
        .collect()
}

/// Copies everything written to stderr (the logger, panics, crash reports) into a new file in
/// `dir`, and still to the original stderr, which Xcode's console shows while debugging.
fn tee_stderr(dir: &Path) {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let Ok(mut file) = std::fs::File::create(dir.join(format!("run-{stamp}.log"))) else { return };
    let mut fds = [0; 2];
    // SAFETY: plain descriptor plumbing on descriptors this function owns, plus fd 2.
    let original = unsafe {
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return;
        }
        let original = libc::dup(2);
        if original < 0 || libc::dup2(fds[1], 2) < 0 {
            return;
        }
        libc::close(fds[1]);
        original
    };
    let read_end = fds[0];
    let _ = std::thread::Builder::new().name("log-tee".into()).spawn(move || {
        use std::os::fd::FromRawFd;
        // SAFETY: both descriptors were just created for this thread alone.
        let (mut pipe, mut console) =
            unsafe { (std::fs::File::from_raw_fd(read_end), std::fs::File::from_raw_fd(original)) };
        let mut buf = vec![0u8; 64 << 10];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    let _ = console.write_all(&buf[..n]);
                    // Flushed per write: the system may end the app at any moment (memory).
                    let _ = file.write_all(&buf[..n]).and_then(|()| file.flush());
                }
            }
        }
    });
}
