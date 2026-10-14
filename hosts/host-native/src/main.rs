mod audio_out;
#[path = "../../common/boot.rs"]
mod boot;
mod crash;
#[path = "../../common/gpu.rs"]
mod gpu;
mod window;

use std::sync::Arc;

const NATIVE_USAGE: &str = "\
By default the game runs in a window with sound, keyboard, mouse and gamepad input.
  --dump-frames DIR        render headlessly with WebGPU and write frames to DIR as PNG
  --dump-every N           write every Nth presented frame (default 60)
  --dump-audio FILE        write the played audio to FILE (raw 48 kHz stereo signed 16-bit little-endian)
                           instead of playing it
  --no-window              run the title without a window or renderer
  --window                 open the window even with --dump-frames";

fn main() {
    // Flags without a value are rewritten to carry one, since boot::parse takes `--flag VALUE`.
    let args = std::env::args()
        .skip(1)
        .flat_map(|a| if a == "--no-window" || a == "--window" { vec![a, "1".into()] } else { vec![a] });
    let opts = match boot::parse(args, &["--dump-frames", "--dump-every", "--dump-audio", "--no-window", "--window"]) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("usage: skyrim-native --game-dir DIR [options]\n{msg}\n{NATIVE_USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = kernel_hle::logging::init(&opts.log) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let Some(game_root) = opts.game_dir.clone() else {
        eprintln!("--game-dir is required\n{}", boot::USAGE);
        std::process::exit(2);
    };
    crash::install();
    runtime_core::thread::set_thread_start_hook(crash::install_alt_stack);
    let content = game_root.parent().unwrap_or(std::path::Path::new(".")).join("content");
    let mut render = xenos_render::native::NativeOptions { dump_dir: None, dump_every: 60, window: None };
    let mut audio_dump: Option<std::path::PathBuf> = None;
    let mut no_window = false;
    let mut force_window = false;
    for (flag, value) in &opts.extra {
        match flag.as_str() {
            "--dump-frames" => render.dump_dir = Some(value.into()),
            "--dump-audio" => audio_dump = Some(value.into()),
            "--no-window" => no_window = true,
            "--window" => force_window = true,
            "--dump-every" => match value.parse() {
                Ok(n) => render.dump_every = n,
                Err(e) => {
                    eprintln!("--dump-every: {e}");
                    std::process::exit(2);
                }
            },
            _ => unreachable!("only registered flags are collected"),
        }
    }
    let windowed = !no_window && (render.dump_dir.is_none() || force_window);
    let game = vfs::GameRoot::Dir(game_root);
    let start = move |render: xenos_render::native::NativeOptions| {
        let on_kernel = move |kernel: &'static kernel_hle::Kernel| on_kernel(kernel, render, audio_dump);
        match boot::run(&opts, &game, &content, on_kernel) {
            Ok(code) => exit(code),
            Err(e) => {
                log::error!("{e}");
                exit(1);
            }
        }
    };
    if !windowed {
        start(render);
        return;
    }
    // The window's event loop needs the main thread; the title boots on another one.
    let result = window::run(move |window| {
        render.window = Some(window);
        let spawned = std::thread::Builder::new().name("boot".into()).stack_size(16 << 20).spawn(move || start(render));
        if let Err(e) = spawned {
            log::error!("starting the title: {e}");
            exit(1);
        }
    });
    if let Err(e) = result {
        log::error!("window: {e}");
        exit(1);
    }
}

/// Ends the process at once. A normal exit runs the GPU driver's exit handlers while the
/// renderer thread may still be using it, which crashes.
pub(crate) fn exit(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: _exit only ends the process.
    unsafe { libc::_exit(code) }
}

fn on_kernel(
    kernel: &'static kernel_hle::Kernel,
    render: xenos_render::native::NativeOptions,
    audio_dump: Option<std::path::PathBuf>,
) {
    crash::set_kernel(kernel);
    let windowed = render.window.is_some();
    if windowed {
        window::attach(kernel);
        if audio_dump.is_none() {
            audio_out::install(kernel);
        }
    }
    if let Some(path) = &audio_dump {
        match std::fs::File::create(path) {
            Ok(file) => {
                let out = std::sync::Mutex::new(std::io::BufWriter::new(file));
                kernel.audio.set_output(Arc::new(move |_client, frame| dump_audio(&out, frame)));
            }
            Err(e) => log::error!("{}: {e}", path.display()),
        }
    }
    if render.dump_dir.is_none() && !windowed {
        return;
    }
    gpu::start_memory_report();
    let memory = Arc::new(gpu::GuestPhysical(kernel));
    match xenos_render::native::start(memory, render, gpu::render_targets()) {
        Ok(handle) => gpu::install(kernel, handle),
        Err(e) => log::error!("starting the renderer: {e}"),
    }
}

/// Appends one played frame (5.1, channel-major) downmixed to stereo s16le.
fn dump_audio(
    out: &std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    frame: &[f32; kernel_hle::audio::FRAME_SAMPLES],
) {
    use kernel_hle::audio::CHANNEL_SAMPLES;
    use std::io::Write;
    let c = |ch: usize, i: usize| frame[ch * CHANNEL_SAMPLES + i];
    let k = std::f32::consts::FRAC_1_SQRT_2;
    let mut bytes = Vec::with_capacity(CHANNEL_SAMPLES * 4);
    for i in 0..CHANNEL_SAMPLES {
        let center = c(2, i) * k + c(3, i) * 0.5;
        for v in [c(0, i) + center + c(4, i) * k, c(1, i) + center + c(5, i) * k] {
            bytes.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
    }
    let mut out = out.lock().unwrap();
    if let Err(e) = out.write_all(&bytes).and_then(|()| out.flush()) {
        log::error!("audio dump: {e}");
    }
}
