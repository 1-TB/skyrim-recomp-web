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
  --vblank-hz N            vblank interrupts per second (default 60 in the browser, 240 natively).
                           Skyrim shows a frame every second vblank, so 120 allows 60 fps, 240
                           allows 120 fps and e.g. 1000 removes the cap
                           (physics may misbehave at very high frame rates)
  --slow-calls MS          log kernel calls that take at least MS milliseconds, with their
                           guest callers (what a stalled title thread is waiting on)
  --no-signin              start with no profile signed in
  --no-pad                 start with no controller connected (default: one on port 1)
  --tap BUTTON@SECONDS     press a port-1 button at that time after boot (held for 200 ms
                           and two frames), e.g. START@20
                           (A B X Y START BACK LB RB LS RS UP DOWN LEFT RIGHT; repeatable)
  --rescale-at SECONDS:N   (testing) change the resolution scale to N while running, SECONDS
                           after start (repeatable)
  --cheat CHEAT@SECONDS    (Skyrim) a cheat at that time after boot: god, god=off, gold=N,
                           lockpicks=N, heal, carry=N (more carry weight), perks=N (perk
                           points), skills=N (every skill up N levels), speed=PERCENT (repeatable)
  --hold CONTROL[=VALUE]@START+DURATION
                           hold a port-1 control from START for DURATION seconds: a button,
                           LS=X,Y or RS=X,Y (sticks, -1..1, up positive) or LT=V / RT=V
                           (triggers, 0..1), e.g. LS=0,1@300+5 walks forward (repeatable)
  --gpu MODE               d3d (default for the executable the D3D hooks are built for: render
                           from the title's Direct3D calls; render targets are the D3D surfaces,
                           no EDRAM), xenos (render the command stream; other executables),
                           d3d-probe (count the Direct3D calls), d3d-verify (also check the
                           D3D device's register copy and the decoded call arguments against
                           the command stream at every draw)
  --gpu-sync-draws         (debugging) the command processor waits for the renderer after every draw
  --gpu-sync               the command processor waits for the renderer at fences, interrupts
                           and occlusion reports instead of going on (default on phones and
                           tablets in the browser); --no-gpu-sync: never
  --resolution-scale N     draw at N (1, 2 or 3) times the console's resolution along each axis
                           (default 1)
  --anisotropy N           at least Nx anisotropic filtering (1-16) where the title samples
                           trilinearly (default 1: the title's own)
  --no-msaa                draw the title's multisampled surfaces single-sampled (no
                           anti-aliasing; faster)
  --metalfx                scale frames up to the window with MetalFX (Apple GPUs) instead of
                           bicubic filtering
  --aspect MODE            how the 16:9 frame fills a window or screen of another shape: fit
                           (whole frame, black bars; default), fill (zoom in, cropping the
                           edges) or stretch
  --game-aspect W:H        draw the game itself wider than 16:9 (true widescreen: more of the
                           world to the sides, menus laid out for the shape), e.g. 19.5:9 for
                           a phone; 720 pixels high, 1280-1920 wide (default 16:9)
  --game-setting SECTION:NAME=VALUE
                           set one of the game's own settings, as its Skyrim.ini would (e.g.
                           Display:bDrawShadows=0; repeatable)
  --render-debug MODE      renderer experiments for finding bottlenecks (repeatable):
                           scissor (shade one pixel per draw), nodraw (record but skip draw calls),
                           frames=N (let the renderer run N frames ahead of the GPU; default 2),
                           mirror=MIB (GPU memory for mirroring guest memory, 32-128; default
                           128, or 64 on phones and tablets in the browser), handheld (the
                           GPU memory budgets used on phones and tablets), load-cap=MIB (guest
                           MiB of new textures loaded per second, the rest later; default
                           no cap, 16 on phones and tablets; 0 = no cap), upload-cap=MIB (MiB of uploads the GPU may not have consumed
                           yet before the renderer waits; default 256, 32 on phones and
                           tablets; 0 = no cap), unshared-upload (uploads copied out
                           of the shared wasm memory first; web only), staging / no-staging (uploads through a ring of mapped
                           staging buffers instead of writeBuffer, or not),
                           nopool (never reuse textures), nomemo (bind textures per draw),
                           cpu-resolve (read resolves back to guest memory), cpu-texture (convert
                           textures on the CPU), no-bc (no BC textures), single-sample (D3D surfaces
                           single-sampled), no-d32s8 (float24 depth
                           in depth24plus-stencil8), no-a2c (no alpha to coverage), d3d-surfaces
                           (with --gpu d3d, show all color surfaces), gpu-time (GPU time per pass,
                           with timestamp queries), trace-frame=N (log frame N's
                           draws and, natively, write their shaders to target/shader-dump/),
                           trace-every=K (and every Kth frame after N)";

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
    /// `--cheat`: (a kernel-hle cheat by name, seconds after boot).
    pub cheats: Vec<(String, f64)>,
    pub holds: Vec<(Held, f64, f64)>,
    /// `--gpu` as given; without it the D3D path is used when the hooks were built for the title.
    pub gpu: Option<String>,
    /// `--game-aspect`: the width the title draws at (720 pixels high).
    pub game_width: Option<u32>,
    /// `--game-setting`: (section, `name=value`) lines added to what the title reads as its
    /// Skyrim.ini.
    pub game_settings: Vec<(String, String)>,
    /// Host-specific flags the shared parser doesn't know.
    pub extra: Vec<(String, String)>,
}

/// Default `--vblank-hz` of the native hosts: Skyrim shows a frame every second vblank, so 240
/// lets it run at up to 120 fps (the browser page picks its own rate).
pub const NATIVE_VBLANK_HZ: u32 = 240;

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
            "--slow-calls" => {
                let ms: u32 = value()?.parse().map_err(|e| format!("--slow-calls: {e}"))?;
                kernel_hle::dispatch::SLOW_CALL_MS.store(ms, std::sync::atomic::Ordering::Relaxed);
            }
            "--tap" => o.taps.push(parse_tap(&value()?)?),
            "--cheat" => {
                let v = value()?;
                let (cheat, at) = v.split_once('@').ok_or(format!("--cheat {v}: not CHEAT@SECONDS"))?;
                let at: f64 = at.parse().map_err(|e| format!("--cheat {v}: {e}"))?;
                o.cheats.push((cheat.to_string(), at));
            }
            "--hold" => o.holds.push(parse_hold(&value()?)?),
            "--rescale-at" => {
                let v = value()?;
                let (at, scale) = v.split_once(':').ok_or(format!("--rescale-at {v}: not SECONDS:SCALE"))?;
                let at: f64 = at.parse().map_err(|e| format!("--rescale-at {v}: {e}"))?;
                let scale: u32 = scale.parse().map_err(|e| format!("--rescale-at {v}: {e}"))?;
                if !(1..=3).contains(&scale) {
                    return Err(format!("--rescale-at {v}: scale not 1, 2 or 3"));
                }
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs_f64(at));
                    log::info!("--rescale-at: resolution scale {scale}");
                    let _ = xenos_render::set_resolution_scale(scale);
                    xenos_render::reconfigure();
                });
            }
            "--render-debug" => xenos_render::debug::enable(&value()?)?,
            "--gpu-sync-draws" => kernel_hle::gpu::SYNC_DRAWS.store(true, std::sync::atomic::Ordering::Relaxed),
            "--gpu-sync" => kernel_hle::gpu::SYNC_EFFECTS.store(kernel_hle::gpu::SYNC_FORCED_ON, std::sync::atomic::Ordering::Relaxed),
            "--no-gpu-sync" => {
                kernel_hle::gpu::SYNC_EFFECTS.store(kernel_hle::gpu::SYNC_FORCED_OFF, std::sync::atomic::Ordering::Relaxed)
            }
            "--gpu" => {
                let mode = value()?;
                kernel_hle::d3d::set_gpu_mode(&mode)?;
                o.gpu = Some(mode);
            }
            "--resolution-scale" => {
                let scale = value()?.parse().map_err(|e| format!("--resolution-scale: {e}"))?;
                xenos_render::set_resolution_scale(scale)?;
            }
            "--metalfx" => xenos_render::set_metalfx(true),
            "--aspect" => xenos_render::set_aspect(value()?.parse()?),
            "--game-aspect" => {
                let v = value()?;
                let ratio = v
                    .split_once(':')
                    .and_then(|(w, h)| Some(w.parse::<f64>().ok()? / h.parse::<f64>().ok()?))
                    .filter(|r| r.is_finite() && *r > 0.0)
                    .ok_or(format!("--game-aspect {v}: not W:H"))?;
                o.game_width = Some(game_width(ratio));
            }
            "--game-setting" => {
                let v = value()?;
                let (section, setting) = v
                    .split_once(':')
                    .filter(|(section, setting)| !section.is_empty() && setting.contains('='))
                    .ok_or(format!("--game-setting {v}: not SECTION:NAME=VALUE"))?;
                o.game_settings.push((section.to_string(), setting.to_string()));
            }
            "--no-msaa" => xenos_render::set_single_sample(true),
            "--anisotropy" => {
                let n: u16 = value()?.parse().map_err(|e| format!("--anisotropy: {e}"))?;
                if !(1..=16).contains(&n) {
                    return Err(format!("--anisotropy {n}: not 1 to 16"));
                }
                xenos_render::set_min_anisotropy(n);
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

fn button(name: &str) -> Option<u16> {
    use kernel_hle::input::buttons as b;
    Some(match name.to_ascii_uppercase().as_str() {
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
        _ => return None,
    })
}

fn parse_tap(spec: &str) -> Result<(u16, f64), String> {
    let (name, at) = spec.split_once('@').ok_or("--tap expects BUTTON@SECONDS")?;
    let mask = button(name).ok_or_else(|| format!("--tap: unknown button {name}"))?;
    let at: f64 = at.parse().map_err(|e| format!("--tap {spec}: {e}"))?;
    Ok((mask, at))
}

/// What a `--hold` holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Held {
    Buttons(u16),
    LeftStick(f32, f32),
    RightStick(f32, f32),
    LeftTrigger(f32),
    RightTrigger(f32),
}

/// `--hold CONTROL[=VALUE]@START+DURATION`: a button, LS=X,Y / RS=X,Y (-1..1, up positive) or
/// LT=V / RT=V (0..1), held from START seconds after boot for DURATION seconds.
fn parse_hold(spec: &str) -> Result<(Held, f64, f64), String> {
    let usage = || format!("--hold {spec}: expected CONTROL[=VALUE]@START+DURATION");
    let (control, time) = spec.split_once('@').ok_or_else(usage)?;
    let (start, duration) = time.split_once('+').ok_or_else(usage)?;
    let number = |v: &str| v.trim().parse::<f32>().map_err(|e| format!("--hold {spec}: {e}"));
    let (name, value) = control.split_once('=').map_or((control, None), |(n, v)| (n, Some(v)));
    let stick = |v: Option<&str>| -> Result<(f32, f32), String> {
        let (x, y) = v.and_then(|v| v.split_once(',')).ok_or_else(usage)?;
        Ok((number(x)?.clamp(-1.0, 1.0), number(y)?.clamp(-1.0, 1.0)))
    };
    let trigger = |v: Option<&str>| -> Result<f32, String> { Ok(number(v.ok_or_else(usage)?)?.clamp(0.0, 1.0)) };
    let held = match (name.to_ascii_uppercase().as_str(), value) {
        ("LS", Some(_)) => {
            let (x, y) = stick(value)?;
            Held::LeftStick(x, y)
        }
        ("RS", Some(_)) => {
            let (x, y) = stick(value)?;
            Held::RightStick(x, y)
        }
        ("LT", _) => Held::LeftTrigger(trigger(value)?),
        ("RT", _) => Held::RightTrigger(trigger(value)?),
        (_, None) => Held::Buttons(button(name).ok_or_else(|| format!("--hold: unknown control {name}"))?),
        _ => return Err(usage()),
    };
    let start: f64 = start.parse().map_err(|e| format!("--hold {spec}: {e}"))?;
    let duration: f64 = duration.parse().map_err(|e| format!("--hold {spec}: {e}"))?;
    Ok((held, start, duration.max(0.0)))
}

/// Asks for the `--cheat`s as they come due, from a host thread.
fn spawn_cheat_script(cheats: &[(String, f64)]) {
    if cheats.is_empty() {
        return;
    }
    let mut cheats = cheats.to_vec();
    cheats.sort_by(|a, b| a.1.total_cmp(&b.1));
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        for (cheat, at) in cheats {
            std::thread::sleep(Duration::from_secs_f64(at).saturating_sub(start.elapsed()));
            if let Err(e) = kernel_hle::cheats::request_named(&cheat) {
                log::warn!("--cheat: {e}");
            }
        }
    });
}

/// Plays `--tap` presses and `--hold`s on port 1 from a host thread: each change of the
/// script's state is sent as it comes due. A tap is a 200 ms hold that also lasts at least two
/// presented frames, so the title sees it even when rendering is slow (titles poll input once
/// a frame).
fn spawn_input_script(kernel: &'static Kernel, taps: &[(u16, f64)], holds: &[(Held, f64, f64)]) {
    if taps.is_empty() && holds.is_empty() {
        return;
    }
    // (time, hold index or tap, starts): taps release once their frames have been shown.
    #[derive(Clone, Copy)]
    enum Edge {
        Hold(usize, bool),
        Tap(u16),
    }
    let mut edges: Vec<(f64, Edge)> = taps.iter().map(|&(mask, at)| (at, Edge::Tap(mask))).collect();
    for (i, &(_, start, duration)) in holds.iter().enumerate() {
        edges.push((start, Edge::Hold(i, true)));
        edges.push((start + duration, Edge::Hold(i, false)));
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let holds = holds.to_vec();
    let start = std::time::Instant::now();
    let spawned = std::thread::Builder::new().name("input-script".into()).spawn(move || {
        use kernel_hle::input::Gamepad;
        let mut active = vec![false; holds.len()];
        let state = |active: &[bool], extra: u16| {
            let axis = |v: f32| (v * 32767.0).round() as i16;
            let trigger = |v: f32| (v * 255.0).round() as u8;
            let mut pad = Gamepad { buttons: extra, ..Default::default() };
            for (h, _) in holds.iter().zip(active).filter(|(_, on)| **on) {
                match h.0 {
                    Held::Buttons(mask) => pad.buttons |= mask,
                    Held::LeftStick(x, y) => (pad.thumb_lx, pad.thumb_ly) = (axis(x), axis(y)),
                    Held::RightStick(x, y) => (pad.thumb_rx, pad.thumb_ry) = (axis(x), axis(y)),
                    Held::LeftTrigger(v) => pad.left_trigger = trigger(v),
                    Held::RightTrigger(v) => pad.right_trigger = trigger(v),
                }
            }
            pad
        };
        for (at, edge) in edges {
            let due = start + Duration::from_secs_f64(at);
            std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
            match edge {
                Edge::Hold(i, on) => {
                    active[i] = on;
                    log::info!("input script: {} {:?}", if on { "hold" } else { "release" }, holds[i].0);
                    kernel.input.update(0, state(&active, 0));
                }
                Edge::Tap(mask) => {
                    log::info!("input script: press {mask:#06X}");
                    kernel.input.update(0, state(&active, mask));
                    let pressed = std::time::Instant::now();
                    let frames = kernel.gpu.swap_count();
                    // Give up waiting for frames after 5 s.
                    while pressed.elapsed() < Duration::from_millis(200)
                        || (kernel.gpu.swap_count() < frames + 2 && pressed.elapsed() < Duration::from_secs(5))
                    {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    kernel.input.update(0, state(&active, 0));
                }
            }
        }
    });
    if let Err(e) = spawned {
        log::error!("starting the input script: {e}");
    }
}

/// Height of the title's frame.
const GAME_HEIGHT: u32 = 720;

/// The width a title frame `GAME_HEIGHT` high has at width:height `ratio`: a multiple of 8,
/// 1280 (16:9; narrower screens show the 16:9 frame with bars) to 1920 (8:3).
pub fn game_width(ratio: f64) -> u32 {
    ((ratio * f64::from(GAME_HEIGHT) / 8.0).round() as u32 * 8).clamp(1280, 1920)
}

/// Adds settings to what the title reads as its Skyrim.ini (the console title's file has only a
/// few; the others take built-in defaults, and a line in the file overrides one the way the PC
/// version's INI does).
///
/// `width`: Skyrim's renderer takes its frame size from `iSize W`/`iSize H` in `[Display]` when
/// `bIgnoreResolutionCheck` is set (otherwise 1280x720 on a widescreen console). Its camera keeps
/// the vertical field of view and widens the horizontal one, and its menus lay themselves out
/// for the frame's shape.
fn set_game_settings(vfs: &Vfs, width: Option<u32>, settings: &[(String, String)]) {
    let mut text = String::new();
    if let Some(width) = width {
        text += &format!("[Display]\r\nbIgnoreResolutionCheck=1\r\niSize W={width}\r\niSize H={GAME_HEIGHT}\r\n");
    }
    for (section, setting) in settings {
        text += &format!("[{section}]\r\n{setting}\r\n");
    }
    if text.is_empty() {
        return;
    }
    match vfs.append_on_read("game:\\Skyrim.ini", text.as_bytes()) {
        Ok(()) => {
            if let Some(width) = width {
                log::info!("game frame {width}x{GAME_HEIGHT}");
                kernel_hle::menus::set_wide_frame(true);
            }
            for (section, setting) in settings {
                log::info!("game setting [{section}] {setting}");
            }
        }
        Err(e) => log::warn!("--game-aspect / --game-setting: {e}"),
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
    // The image is built in the file's buffer (dropped after loading, see Kernel::new).
    let data = data.map_err(|e| format!("{xex_name}: {e}"))?;
    let xex = xex::Xex::parse_owned(data).map_err(|e| format!("{xex_name}: {e}"))?;
    if let Some(info) = xex.execution_info() {
        log::info!("title id {:08X}, version {:08X}", info.title_id, info.version);
    }
    // Rendering from the title's Direct3D calls is the default; the hooks it needs exist only
    // for the executable d3d-hle describes, so any other one renders the command stream.
    let title = xex.execution_info().map(|i| (i.title_id, i.version));
    let gpu = o.gpu.clone().unwrap_or_else(|| if title == Some(d3d_hle::TITLE) { "d3d" } else { "xenos" }.into());
    if o.gpu.is_none() {
        kernel_hle::d3d::set_gpu_mode(&gpu)?;
        if gpu == "xenos" {
            log::warn!("the D3D renderer needs title {:08X} version {:08X}; rendering the command stream", d3d_hle::TITLE.0, d3d_hle::TITLE.1);
        }
    }
    xenos_render::set_d3d_mode(gpu == "d3d");
    log::info!("GPU path: {gpu}");
    let rt = Runtime::load(guest_sys::module(), xex).map_err(|e| e.to_string())?;

    let vfs = Vfs::with_game(game);
    for (device, dir, rw) in &o.mounts {
        vfs.mount(device, dir, !rw);
    }
    for (name, device) in &o.links {
        vfs.add_symlink(name, device);
    }
    set_game_settings(&vfs, o.game_width.filter(|&w| w != 1280), &o.game_settings);

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
    spawn_input_script(kernel, &o.taps, &o.holds);
    spawn_cheat_script(&o.cheats);
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

/// Waits for the title to exit (None once `limit` passes), checking the memory report
/// (`gpu::memory_report_tick`). Meanwhile, if the title has presented frames and then stops
/// for [`STALL_REPORT_AFTER`], logs the blocked guest threads once per stall: a hung title is
/// otherwise indistinguishable from a slow one.
fn wait_for_title(kernel: &'static Kernel, limit: Option<Duration>) -> Option<u32> {
    let start = std::time::Instant::now();
    let mut last_swaps = 0;
    let mut last_change = start;
    let mut reported = false;
    loop {
        let step = match limit {
            Some(l) => (l.saturating_sub(start.elapsed())).min(crate::gpu::MEMORY_REPORT_TICK),
            None => crate::gpu::MEMORY_REPORT_TICK,
        };
        if let Some(code) = kernel.wait_for_exit_timeout(step) {
            return Some(code);
        }
        if limit.is_some_and(|l| start.elapsed() >= l) {
            return None;
        }
        crate::gpu::memory_report_tick();
        kernel.check_progress();
        let swaps = kernel.gpu.swap_count();
        if swaps != last_swaps {
            last_swaps = swaps;
            last_change = std::time::Instant::now();
            reported = false;
        } else if swaps != 0 && !reported && last_change.elapsed() >= STALL_REPORT_AFTER {
            reported = true;
            log::warn!("no frame presented for {:?} (after swap #{swaps}); blocked threads:", last_change.elapsed());
            kernel.log_waiters();
            kernel.log_command_processor(kernel.rt.mem());
        }
    }
}
