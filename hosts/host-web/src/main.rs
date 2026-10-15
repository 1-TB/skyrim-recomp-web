//! Browser host. The page (web/index.html) copies the user's game files into
//! the origin-private file system under `game/`; this binary mounts OPFS and
//! boots `game/default.xex` from it.
//!
//! Extra options: `--node-dir DIR` mounts a host directory through the WasmFS
//! Node backend instead of OPFS (headless runs under Node.js); `--stream-game URL`
//! reads the game files from the server instead (`URL` + `manifest`, `files/...`;
//! see `cargo xtask serve --game-dir`), keeping only content in OPFS.

#[cfg(target_os = "emscripten")]
#[path = "../../common/audio.rs"]
mod audio;
#[cfg(target_os = "emscripten")]
#[path = "../../common/boot.rs"]
mod boot;
#[cfg(target_os = "emscripten")]
#[path = "../../common/gpu.rs"]
mod gpu;
#[cfg(target_os = "emscripten")]
mod stream;
#[cfg(target_os = "emscripten")]
mod text_input;

/// The memory maximum build.rs links with (`-sMAXIMUM_MEMORY`) must hold the linear layout.
#[cfg(target_os = "emscripten")]
const _: () = assert!(runtime_core::memory::linear_map::SIZE <= 1538 << 20);

#[cfg(target_os = "emscripten")]
fn main() {
    use std::path::PathBuf;

    let wasi_clock = std::env::args().any(|a| a == "--wasi-clock");
    clock::WASI.store(wasi_clock, std::sync::atomic::Ordering::Relaxed);
    let args = std::env::args().skip(1).filter(|a| a != "--wasi-clock");
    let opts = match boot::parse(args, &["--node-dir", "--stream-game", "--stream-cache-mib"]) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!(
                "host-web options:\n  --node-dir DIR           (Node.js only) use DIR instead of OPFS\n  \
                 --stream-game URL        read the game files from the server at URL\n  \
                 --stream-cache-mib N     memory for small streamed reads (default 16)\n  \
                 --wasi-clock             read clocks through Emscripten's WASI import (as before)\n{msg}"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = kernel_hle::logging::init(&opts.log) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let extra = |flag: &str| opts.extra.iter().find(|(k, _)| k == flag).map(|(_, v)| v.clone());
    let node_dir = extra("--node-dir");
    let stream_url = extra("--stream-game");
    let stream_cache = match extra("--stream-cache-mib").map(|v| v.parse::<u64>()) {
        None => None,
        Some(Ok(mib)) => Some(mib),
        Some(Err(e)) => {
            eprintln!("--stream-cache-mib: {e}");
            std::process::exit(2);
        }
    };
    // Node.js has no canvas or WebGPU: headless runs only execute the title.
    let render = node_dir.is_none();
    let (game_root, content): (PathBuf, PathBuf) = match node_dir {
        Some(dir) => match vfs::opfs::mount_node_dir("/host", &dir) {
            Ok(p) => (p.clone(), p.join("content")),
            Err(e) => {
                log::error!("mounting {dir} via the Node backend: {e}");
                std::process::exit(1);
            }
        },
        None => match vfs::opfs::mount_opfs("/opfs") {
            Ok(p) => (opts.game_dir.clone().unwrap_or_else(|| p.join("game")), p.join("content")),
            Err(e) => {
                log::error!("mounting OPFS: {e}");
                std::process::exit(1);
            }
        },
    };
    let game = match stream_url {
        Some(url) => match stream::open(&url, stream_cache) {
            Ok(tree) => vfs::GameRoot::Remote(tree),
            Err(e) => {
                log::error!("{e}");
                std::process::exit(1);
            }
        },
        None => vfs::GameRoot::Dir(game_root),
    };
    let on_kernel = move |kernel| {
        text_input::install();
        if !render {
            return;
        }
        audio::install(kernel);
        gpu::start_memory_report();
        unsafe extern "C" {
            /// lib/device.js.
            fn skyrt_handheld() -> i32;
        }
        // SAFETY: no arguments; proxied to the page's thread.
        if unsafe { skyrt_handheld() } != 0 {
            // Browsers on phones and tablets close tabs long before desktop ones.
            log::info!("phone or tablet: smaller GPU memory budgets; the GPU waits for the renderer at fences");
            xenos_render::handheld_budgets();
            kernel_hle::gpu::set_sync_default(true);
        }
        let memory = std::sync::Arc::new(gpu::GuestPhysical(kernel));
        match xenos_render::web::start(memory, "#canvas", gpu::render_targets()) {
            Ok(handle) => gpu::install(kernel, handle),
            Err(e) => log::error!("starting the renderer: {e}"),
        }
    };
    match boot::run(&opts, &game, &content, on_kernel) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            log::error!("{e}");
            std::process::exit(1);
        }
    }
}

/// `clock_gettime` (and libc's internal `__clock_gettime`, behind `time`, `gettimeofday`...)
/// without the WASI import Emscripten's libc goes through: that import takes its (ignored)
/// precision as an i64, which JavaScriptCore boxes in a heap BigInt on every call. The guest
/// threads read the clock millions of times a minute, and in WebKit those BigInts were never
/// collected on workers that don't return to their event loop: Safari's memory grew by
/// megabytes a second until iOS closed the tab. Same clocks as Emscripten's: the page's
/// `Date.now()` for CLOCK_REALTIME, `performance.timeOrigin + performance.now()` (shared by
/// all threads) otherwise.
#[cfg(target_os = "emscripten")]
mod clock {
    #[repr(C)]
    pub struct Timespec {
        tv_sec: i64,
        tv_nsec: i32,
        _pad: i32,
    }

    unsafe extern "C" {
        fn emscripten_get_now() -> f64;
        fn emscripten_date_now() -> f64;
    }

    #[link(wasm_import_module = "wasi_snapshot_preview1")]
    unsafe extern "C" {
        #[link_name = "clock_time_get"]
        fn wasi_clock_time_get(clk: u32, precision: u64, time: *mut u64) -> u16;
    }

    /// `--wasi-clock`: through the WASI import after all (for comparisons).
    pub static WASI: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    const CLOCK_REALTIME: i32 = 0;
    const CLOCK_THREAD_CPUTIME_ID: i32 = 3;
    const EINVAL: i32 = 28;

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __clock_gettime(clk: i32, ts: *mut Timespec) -> i32 {
        if !(0..=CLOCK_THREAD_CPUTIME_ID).contains(&clk) || ts.is_null() {
            // SAFETY: errno is thread-local and always valid.
            unsafe { *__errno_location() = EINVAL };
            return -1;
        }
        let ns = if WASI.load(std::sync::atomic::Ordering::Relaxed) {
            let mut t = 0u64;
            // SAFETY: a valid out pointer.
            unsafe { wasi_clock_time_get(clk as u32, 1, &mut t) };
            t as i64
        } else {
            // SAFETY: imports without arguments.
            let ms = unsafe { if clk == CLOCK_REALTIME { emscripten_date_now() } else { emscripten_get_now() } };
            (ms * 1e6) as i64
        };
        // SAFETY: the caller passes a valid timespec.
        unsafe {
            *ts = Timespec { tv_sec: ns.div_euclid(1_000_000_000), tv_nsec: ns.rem_euclid(1_000_000_000) as i32, _pad: 0 };
        }
        0
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn clock_gettime(clk: i32, ts: *mut Timespec) -> i32 {
        // SAFETY: as for `__clock_gettime`.
        unsafe { __clock_gettime(clk, ts) }
    }

    unsafe extern "C" {
        fn __errno_location() -> *mut i32;
    }
}

/// Controller state from the page (Gamepad API or keyboard), called from the
/// browser's main thread: XINPUT_GAMEPAD fields for port `user` (0-3).
#[cfg(target_os = "emscripten")]
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn skyrt_set_pad(user: u32, buttons: u32, lt: u32, rt: u32, lx: i32, ly: i32, rx: i32, ry: i32) {
    let Some(k) = kernel_hle::Kernel::try_global() else { return };
    let clamp = |v: i32| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    k.input.update(
        user as usize,
        kernel_hle::input::Gamepad {
            buttons: buttons as u16,
            left_trigger: lt.min(255) as u8,
            right_trigger: rt.min(255) as u8,
            thumb_lx: clamp(lx),
            thumb_ly: clamp(ly),
            thumb_rx: clamp(rx),
            thumb_ry: clamp(ry),
        },
    );
}

/// The page canvas's size on screen in device pixels (the renderer sizes its output to it).
#[cfg(target_os = "emscripten")]
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_set_display_size(width: u32, height: u32) {
    xenos_render::web::set_display_size(width, height);
}

/// Plugs a controller into port `user` or pulls it out.
#[cfg(target_os = "emscripten")]
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_set_pad_connected(user: u32, connected: u32) {
    let Some(k) = kernel_hle::Kernel::try_global() else { return };
    if connected != 0 { k.input.connect(user as usize) } else { k.input.disconnect(user as usize) }
}

/// Frames the title has presented so far (for test scripts that wait for rendering to advance).
/// A double: the count passes through JS as a number.
#[cfg(target_os = "emscripten")]
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_swap_count() -> f64 {
    kernel_hle::Kernel::try_global().map_or(0.0, |k| k.gpu.swap_count() as f64)
}

/// Size of the wasm linear memory in bytes (the page's `HEAPU8` view can be stale after growth).
#[cfg(target_os = "emscripten")]
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size(0) as f64) * 65536.0
}

/// Host file-read statistics for the page's crash breadcrumbs: `what` 0 = bytes read,
/// 1 = of those through files opened read-only (Blob reads in WasmFS), 2 = open files.
#[cfg(target_os = "emscripten")]
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_io_stats(what: i32) -> f64 {
    let (read, read_only, open) = vfs::stats::snapshot();
    match what {
        0 => read as f64,
        1 => read_only as f64,
        _ => open as f64,
    }
}

#[cfg(not(target_os = "emscripten"))]
fn main() {
    eprintln!("host-web targets wasm32-unknown-emscripten (cargo xtask build-web); use skyrim-native on this host");
    std::process::exit(2);
}
