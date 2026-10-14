//! Command-line options and the boot sequence shared by host-native and host-web.

use std::path::{Path, PathBuf};
use std::time::Duration;

use kernel_hle::Kernel;
use kernel_hle::dispatch::{StubConfig, parse_num};
use runtime_core::Runtime;
use vfs::{GameRoot, Vfs};

pub const USAGE: &str = "\
options:
  --xex PATH               decrypted, decompressed default.xex (default: <game-dir>/default.xex)
  --game-dir DIR           title files, mounted read-only as game:\\ and d:\\
  --stub-return VALUE      r3 value for unimplemented imports (default 0)
  --stub MODULE:ORD=VALUE  per-import stub value, e.g. xboxkrnl:0xD9=0xC0000001 (repeatable)
  --mount DEVICE=DIR[:rw]  extra device, e.g. \\Device\\Harddisk0\\Partition1=/saves:rw (repeatable)
  --link NAME=DEVICE       symbolic link, e.g. cache:=\\Device\\Harddisk0\\Cache0 (repeatable)
  --log SPEC               error|warn|info|debug|trace (trace logs every import call), optionally
                           with per-module levels: info,kernel_hle::gpu=trace
  --run-for SECONDS        stop after this long and print the import summary
  --content DIR            where saves and other title content are stored (read-write)
  --gamertag NAME          name of the signed-in local profile (default Player)
  --vblank-hz N            vblank interrupts per second (default 60). Skyrim shows a frame every
                           second vblank, so 120 allows 60 fps and e.g. 1000 removes the cap
                           (physics may misbehave at very high frame rates)
  --no-signin              start with no profile signed in
  --no-pad                 start with no controller connected (default: one on port 1)
  --tap BUTTON@SECONDS     press a port-1 button at that time after boot (held for 200 ms
                           and two frames), e.g. START@20
                           (A B X Y START BACK LB RB LS RS UP DOWN LEFT RIGHT; repeatable)
  --resolution-scale N     draw at N (1, 2 or 3) times the console's resolution along each axis
                           (default 1)
  --render-debug MODE      renderer experiments for finding bottlenecks (repeatable):
                           scissor (shade one pixel per draw), nodraw (record but skip draw calls),
                           frames=N (let the renderer run N frames ahead of the GPU; default 2),
                           mirror=MIB (GPU memory for mirroring guest memory, 32-128; default
                           128, or 64 on phones and tablets in the browser), handheld (the
                           GPU memory budgets used on phones and tablets),
                           nopool (never reuse textures), nomemo (bind textures per draw),
                           cpu-resolve (read resolves back to guest memory), cpu-texture (convert
                           textures on the CPU), trace-frame=N (log frame N's draws and, natively,
                           write their shaders to target/shader-dump/), trace-every=K (and every
                           Kth frame after N)";

#[derive(Default)]
pub struct Options {
    pub xex: Option<PathBuf>,
    pub game_dir: Option<PathBuf>,
    pub stubs: StubConfig,
    pub log: String,
    pub run_for: Option<Duration>,
    pub mounts: Vec<(String, PathBuf, bool)>,
    pub links: Vec<(String, String)>,
    pub content: Option<PathBuf>,
    pub gamertag: Option<String>,
    pub no_signin: bool,
    pub no_pad: bool,
    pub vblank_hz: Option<u32>,
    /// Scripted presses: (button mask, seconds after boot).
    pub taps: Vec<(u16, f64)>,
    /// Host-specific flags the shared parser doesn't know.
    pub extra: Vec<(String, String)>,
}

pub fn parse(mut args: impl Iterator<Item = String>, extra_flags: &[&str]) -> Result<Options, String> {
    let mut o = Options { log: "info".into(), ..Default::default() };
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--xex" => o.xex = Some(value()?.into()),
            "--game-dir" => o.game_dir = Some(value()?.into()),
            "--stub-return" => o.stubs.default_return = parse_num(&value()?)?,
            "--stub" => {
                let (k, v) = StubConfig::parse_override(&value()?)?;
                o.stubs.overrides.insert(k, v);
            }
            "--mount" => {
                let spec = value()?;
                let (device, dir) = spec.split_once('=').ok_or("--mount expects DEVICE=DIR[:rw]")?;
                let (dir, rw) = match dir.strip_suffix(":rw") {
                    Some(d) => (d, true),
                    None => (dir, false),
                };
                o.mounts.push((device.to_string(), dir.into(), rw));
            }
            "--link" => {
                let spec = value()?;
                let (name, device) = spec.split_once('=').ok_or("--link expects NAME=DEVICE")?;
                o.links.push((name.to_string(), device.to_string()));
            }
            "--log" => o.log = value()?,
            "--content" => o.content = Some(value()?.into()),
            "--gamertag" => o.gamertag = Some(value()?),
            "--no-signin" => o.no_signin = true,
            "--no-pad" => o.no_pad = true,
            "--vblank-hz" => o.vblank_hz = Some(value()?.parse().map_err(|e| format!("--vblank-hz: {e}"))?),
            "--tap" => o.taps.push(parse_tap(&value()?)?),
            "--render-debug" => xenos_render::debug::enable(&value()?)?,
            "--resolution-scale" => {
                let scale = value()?.parse().map_err(|e| format!("--resolution-scale: {e}"))?;
                xenos_render::set_resolution_scale(scale)?;
            }
            "--run-for" => {
                let secs: f64 = value()?.parse().map_err(|e| format!("--run-for: {e}"))?;
                o.run_for = Some(Duration::from_secs_f64(secs));
            }
            f if extra_flags.contains(&f) => {
                let v = value()?;
                o.extra.push((flag, v));
            }
            other => return Err(format!("unknown option {other}\n{USAGE}")),
        }
    }
    Ok(o)
}

fn parse_tap(spec: &str) -> Result<(u16, f64), String> {
    use kernel_hle::input::buttons as b;
    let (name, at) = spec.split_once('@').ok_or("--tap expects BUTTON@SECONDS")?;
    let mask = match name.to_ascii_uppercase().as_str() {
        "A" => b::A,
        "B" => b::B,
        "X" => b::X,
        "Y" => b::Y,
        "START" => b::START,
        "BACK" => b::BACK,
        "LB" => b::LEFT_SHOULDER,
        "RB" => b::RIGHT_SHOULDER,
        "LS" => b::LEFT_THUMB,
        "RS" => b::RIGHT_THUMB,
        "UP" => b::DPAD_UP,
        "DOWN" => b::DPAD_DOWN,
        "LEFT" => b::DPAD_LEFT,
        "RIGHT" => b::DPAD_RIGHT,
        other => return Err(format!("--tap: unknown button {other}")),
    };
    let at: f64 = at.parse().map_err(|e| format!("--tap {spec}: {e}"))?;
    Ok((mask, at))
}

/// Plays `--tap` presses on port 1 from a host thread.
fn spawn_taps(kernel: &'static Kernel, taps: &[(u16, f64)]) {
    if taps.is_empty() {
        return;
    }
    let mut taps = taps.to_vec();
    taps.sort_by(|a, b| a.1.total_cmp(&b.1));
    let start = std::time::Instant::now();
    let spawned = std::thread::Builder::new().name("input-script".into()).spawn(move || {
        use kernel_hle::input::Gamepad;
        for (mask, at) in taps {
            let due = start + Duration::from_secs_f64(at);
            std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
            log::info!("input script: press {mask:#06X}");
            kernel.input.update(0, Gamepad { buttons: mask, ..Default::default() });
            // Hold for 200 ms and at least two presented frames, so the title
            // sees the press even when rendering is slow (titles poll input
            // once a frame); give up waiting for frames after 5 s.
            let pressed = std::time::Instant::now();
            let frames = kernel.gpu.swap_count();
            while pressed.elapsed() < Duration::from_millis(200)
                || (kernel.gpu.swap_count() < frames + 2 && pressed.elapsed() < Duration::from_secs(5))
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            kernel.input.update(0, Gamepad::default());
        }
    });
    if let Err(e) = spawned {
        log::error!("starting the input script: {e}");
    }
}

/// Loads the title (`--xex`, else the game root's `default.xex`), boots its main thread and
/// waits for it. Returns the exit code. `on_kernel` sees the kernel before any guest code runs.
/// Content goes to `--content`, else `default_content`.
pub fn run(
    o: &Options,
    game: &GameRoot,
    default_content: &Path,
    on_kernel: impl FnOnce(&'static Kernel),
) -> Result<i32, String> {
    let (xex_name, data) = match &o.xex {
        Some(path) => (path.display().to_string(), std::fs::read(path)),
        None => ("default.xex".to_string(), game.read("default.xex")),
    };
    log::info!("title {xex_name} (game root {game})");
    let data = data.map_err(|e| format!("{xex_name}: {e}"))?;
    let xex = xex::Xex::parse(&data).map_err(|e| format!("{xex_name}: {e}"))?;
    if let Some(info) = xex.execution_info() {
        log::info!("title id {:08X}, version {:08X}", info.title_id, info.version);
    }
    let rt = Runtime::load(guest_sys::module(), xex).map_err(|e| e.to_string())?;

    let vfs = Vfs::with_game(game);
    for (device, dir, rw) in &o.mounts {
        vfs.mount(device, dir, !rw);
    }
    for (name, device) in &o.links {
        vfs.add_symlink(name, device);
    }

    let kernel = Kernel::new(rt, vfs, o.stubs.clone(), guest_sys::IMPORT_NAMES)?;
    kernel.install_global();
    let content = o.content.clone().unwrap_or_else(|| default_content.to_path_buf());
    log::info!("title content in {}", content.display());
    kernel.content.set_root(content);
    if o.no_signin {
        kernel.users.sign_out(0);
    } else if let Some(name) = &o.gamertag {
        kernel.users.sign_in(0, kernel_hle::xam::user::Profile::offline(name));
    }
    if !o.no_pad {
        kernel.input.connect(0);
    }
    if let Some(hz) = o.vblank_hz {
        log::info!("vblank rate {hz} Hz");
        kernel.gpu.set_vblank_hz(hz);
    }
    on_kernel(kernel);
    spawn_taps(kernel, &o.taps);
    kernel.start_main_thread().map_err(|s| format!("starting main thread: {s:#010X}"))?;

    let Some(code) = wait_for_title(kernel, o.run_for) else {
        log::info!("--run-for limit reached; stopping");
        kernel.log_waiters();
        kernel.log_import_summary();
        return Ok(0);
    };
    log::info!("title exited with {code:#X}");
    kernel.log_import_summary();
    Ok(code as i32)
}

/// How long presenting may stop before the watchdog reports blocked threads.
const STALL_REPORT_AFTER: Duration = Duration::from_secs(10);

/// Waits for the title to exit (None once `limit` passes). Meanwhile, if the
/// title has presented frames and then stops for [`STALL_REPORT_AFTER`], logs
/// the blocked guest threads once per stall: a hung title is otherwise
/// indistinguishable from a slow one.
fn wait_for_title(kernel: &'static Kernel, limit: Option<Duration>) -> Option<u32> {
    let start = std::time::Instant::now();
    let mut last_swaps = 0;
    let mut last_change = start;
    let mut reported = false;
    loop {
        let step = match limit {
            Some(l) => (l.saturating_sub(start.elapsed())).min(Duration::from_secs(1)),
            None => Duration::from_secs(1),
        };
        if let Some(code) = kernel.wait_for_exit_timeout(step) {
            return Some(code);
        }
        if limit.is_some_and(|l| start.elapsed() >= l) {
            return None;
        }
        let swaps = kernel.gpu.swap_count();
        if swaps != last_swaps {
            last_swaps = swaps;
            last_change = std::time::Instant::now();
            reported = false;
        } else if swaps != 0 && !reported && last_change.elapsed() >= STALL_REPORT_AFTER {
            reported = true;
            log::warn!("no frame presented for {:?} (after swap #{swaps}); blocked threads:", last_change.elapsed());
            kernel.log_waiters();
        }
    }
}
