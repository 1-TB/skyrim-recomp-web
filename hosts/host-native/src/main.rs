#[path = "../../common/boot.rs"]
mod boot;
mod crash;
#[path = "../../common/gpu.rs"]
mod gpu;

use std::sync::Arc;

const NATIVE_USAGE: &str = "\
  --dump-frames DIR        render headlessly with WebGPU and write frames to DIR as PNG
  --dump-every N           write every Nth presented frame (default 60)
  --dump-audio FILE        write the played audio to FILE (raw 48 kHz stereo signed 16-bit little-endian)";

fn main() {
    let opts = match boot::parse(std::env::args().skip(1), &["--dump-frames", "--dump-every", "--dump-audio"]) {
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
    let xex = opts.xex.clone().unwrap_or_else(|| game_root.join("default.xex"));
    crash::install();
    runtime_core::thread::set_thread_start_hook(crash::install_alt_stack);
    let content = game_root.parent().unwrap_or(std::path::Path::new(".")).join("content");
    let mut render = xenos_render::native::NativeOptions { dump_dir: None, dump_every: 60 };
    let mut audio_dump: Option<std::path::PathBuf> = None;
    for (flag, value) in &opts.extra {
        match flag.as_str() {
            "--dump-frames" => render.dump_dir = Some(value.into()),
            "--dump-audio" => audio_dump = Some(value.into()),
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
    let on_kernel = move |kernel: &'static kernel_hle::Kernel| {
        crash::set_kernel(kernel);
        if let Some(path) = &audio_dump {
            match std::fs::File::create(path) {
                Ok(file) => {
                    let out = std::sync::Mutex::new(std::io::BufWriter::new(file));
                    kernel.audio.set_output(Arc::new(move |_client, frame| dump_audio(&out, frame)));
                }
                Err(e) => log::error!("{}: {e}", path.display()),
            }
        }
        if render.dump_dir.is_none() {
            return;
        }
        let memory = Arc::new(gpu::GuestPhysical(kernel));
        match xenos_render::native::start(memory, render, gpu::render_targets()) {
            Ok(handle) => gpu::install(kernel, handle),
            Err(e) => log::error!("starting the renderer: {e}"),
        }
    };
    match boot::run(&opts, &game_root, &xex, &content, on_kernel) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            log::error!("{e}");
            std::process::exit(1);
        }
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
