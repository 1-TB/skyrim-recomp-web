//! Renderer experiments for finding where frame time goes (`--render-debug`): each removes one
//! kind of GPU work while everything else runs as usual.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Every draw's scissor is one pixel: vertex work stays, pixel shading is (almost) gone.
pub(crate) static SCISSOR_ONE_PIXEL: AtomicBool = AtomicBool::new(false);
/// Draw calls are left out of the passes (all state is still recorded and set).
pub(crate) static SKIP_DRAW_CALLS: AtomicBool = AtomicBool::new(false);

/// Textures are never reused (always freshly created, zeroed).
pub(crate) static NO_TEXTURE_POOL: AtomicBool = AtomicBool::new(false);
/// Every draw binds its textures through the texture cache (no remembered texture sets).
pub(crate) static NO_TEXTURE_SET_MEMO: AtomicBool = AtomicBool::new(false);

/// Resolves are read back into guest memory instead of stored into its GPU copy.
pub(crate) static CPU_RESOLVES: AtomicBool = AtomicBool::new(false);
/// Textures in guest memory are converted on the CPU (not raw-copied and converted on the GPU).
pub(crate) static CPU_TEXTURES: AtomicBool = AtomicBool::new(false);

/// Frame (counted by the renderer from 0) whose draws are logged, with their shaders written to
/// `target/shader-dump/` (native only); 0 = none.
pub(crate) static TRACE_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With [`TRACE_FRAME`], also every this many frames after it (0 = only that frame).
pub(crate) static TRACE_EVERY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether frame `frame`'s draws are traced (`trace-frame=N`, `trace-every=K`).
pub(crate) fn tracing(frame: u64) -> bool {
    let first = TRACE_FRAME.load(Ordering::Relaxed);
    let every = TRACE_EVERY.load(Ordering::Relaxed);
    first != 0 && (frame == first || (every != 0 && frame > first && (frame - first).is_multiple_of(every)))
}

/// `staging` (1) / `no-staging` (2): uploads through mapped staging buffers or not; 0 = the
/// default (phones only).
pub(crate) static STAGING: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `load-cap=MIB`: guest MiB of new textures loaded per second (0 = no cap); `u64::MAX` = the
/// default (a cap only with the handheld budgets).
pub(crate) static LOAD_CAP_MIB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// `upload-cap=MIB`: upload bytes that may be on their way to the GPU before the renderer waits
/// (0 = no cap); `u64::MAX` = the default (see the executor's `UPLOAD_BUDGET`).
pub(crate) static UPLOAD_CAP_MIB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Frames the renderer may run ahead of the GPU (2 unless an experiment changes it).
pub(crate) static MAX_FRAMES_IN_FLIGHT: AtomicUsize = AtomicUsize::new(2);
/// Don't use BC texture compression even when the device has it (as on phones, where BC
/// textures are decompressed), to measure what that costs.
pub(crate) static NO_BC: AtomicBool = AtomicBool::new(false);
/// Don't use depth32float-stencil8 even when the device has it (float24 depth then lives in
/// depth24plus-stencil8, as on devices without it).
pub(crate) static NO_D32S8: AtomicBool = AtomicBool::new(false);
/// `--gpu d3d`: show the frame's color surfaces side by side instead of the front buffer.
pub(crate) static D3D_SURFACES: AtomicBool = AtomicBool::new(false);
/// Alpha to coverage is never enabled.
pub(crate) static NO_A2C: AtomicBool = AtomicBool::new(false);
/// `--gpu d3d`: multisampled D3D surfaces are drawn single-sampled (as with the handheld budgets).
pub(crate) static SINGLE_SAMPLE: AtomicBool = AtomicBool::new(false);
/// `no-skinned-alpha-test`: draws with skinned vertex shaders skip the alpha test.
pub(crate) static NO_SKINNED_ALPHA_TEST: AtomicBool = AtomicBool::new(false);
/// `alpha-view`: alpha-tested pixels show what the test saw instead of being discarded: red if
/// the alpha is NaN, green the alpha (0..1), blue if the test passes.
pub(crate) static ALPHA_VIEW: AtomicBool = AtomicBool::new(false);
/// `hash-pages`: guest memory is compared by content hash instead of by write marks (the GPU
/// mirror and the texture cache then notice every change, marked or not): for telling a lost
/// write mark from a renderer bug.
pub static HASH_PAGES: AtomicBool = AtomicBool::new(false);
/// `alpha-no-depth`: alpha-tested draws pass the depth test (to tell what culls their pixels).
pub(crate) static ALPHA_NO_DEPTH: AtomicBool = AtomicBool::new(false);
/// `no-carry` (`--gpu d3d`): surfaces sharing an EDRAM base don't inherit each other's contents.
pub(crate) static NO_CARRY: AtomicBool = AtomicBool::new(false);
/// `verify-loads`: textures loaded from the GPU copy of guest memory are read back at frame end
/// and compared with their CPU conversion; differences are logged (slow).
pub(crate) static VERIFY_LOADS: AtomicBool = AtomicBool::new(false);
/// `bias-scale=N`: every draw's depth bias (constant and slope) times N.
pub(crate) static BIAS_SCALE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
/// `alpha-depth-bias=N`: constant depth bias added to draws with the alpha test on (0 = none), to
/// test whether alpha-tested passes lose depth ties against earlier depth-only passes.
pub(crate) static ALPHA_DEPTH_BIAS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
/// `alpha-view-kill`: [`ALPHA_VIEW`]'s colors, but failing pixels are still dropped.
pub(crate) static ALPHA_VIEW_KILL: AtomicBool = AtomicBool::new(false);
/// `no-alpha-test`: the alpha test is never applied (pixels it would discard are drawn).
pub(crate) static NO_ALPHA_TEST: AtomicBool = AtomicBool::new(false);
/// `texture-mib=N`: the texture cache's budget (0 = the default 512 MiB): textures kept within
/// N MiB, N/2 MiB of pooled ones, and unused ones kept twice as long.
pub(crate) static TEXTURE_MIB: AtomicU64 = AtomicU64::new(0);
/// `solid-skinned` (native): draws with skinned vertex shaders write solid magenta instead of
/// running their pixel shaders (depth, stencil and blending as set), to tell lost geometry from
/// pixels the shader throws away.
pub(crate) static SOLID_SKINNED: AtomicBool = AtomicBool::new(false);
/// `vs-capture=N` (native): in frame N and every 600th after, skinned draws store the clip-space
/// positions their vertex shaders output, which are logged after the frame (0 = off). For telling a draw that renders
/// nothing because its vertices are wrong from one whose pixels are lost later.
pub(crate) static VS_CAPTURE_FRAME: AtomicU64 = AtomicU64::new(0);
/// `ps-probe=FRAME:XxY,XxY...` (native): in frame FRAME and every 300th after, every alpha-tested
/// draw records, at each listed host pixel it shades, its alpha, the alpha test's function and
/// reference, the sample mask it kept the pixel with and its depth; logged after the frame. For
/// comparing what the alpha test decides per draw on two GPUs (`vs-capture` is off meanwhile).
pub(crate) static PS_PROBE: std::sync::Mutex<Option<(u64, Vec<(u32, u32)>)>> = std::sync::Mutex::new(None);

/// Pass GPU times are measured with timestamp queries and logged with the frame reports.
pub(crate) static GPU_TIME: AtomicBool = AtomicBool::new(false);

/// Turns on experiment `mode`: `scissor`, `nodraw` or `frames=N` (frames in flight).
pub fn enable(mode: &str) -> Result<(), String> {
    if let Some(n) = mode.strip_prefix("trace-frame=") {
        let n: u64 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        TRACE_FRAME.store(n, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("trace-every=") {
        let n: u64 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        TRACE_EVERY.store(n, Ordering::Relaxed);
        return Ok(());
    }
    if mode == "handheld" {
        crate::handheld_budgets();
        return Ok(());
    }
    if mode == "unshared-upload" {
        gpu_webgpu::UNSHARED_UPLOADS.store(true, Ordering::Relaxed);
        return Ok(());
    }
    if mode == "staging" || mode == "no-staging" {
        STAGING.store(if mode == "staging" { 1 } else { 2 }, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("load-cap=") {
        let mib: u64 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        LOAD_CAP_MIB.store(mib, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("upload-cap=") {
        let mib: u64 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        UPLOAD_CAP_MIB.store(mib, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("mirror=") {
        let mib: u32 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        crate::set_mirror_limit(mib);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("texture-mib=") {
        let n: u64 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        TEXTURE_MIB.store(n, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("vs-capture=") {
        let n: u64 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        VS_CAPTURE_FRAME.store(n, Ordering::Relaxed);
        log::warn!("renderer experiment: capturing vertex shader positions in frame {n}");
        return Ok(());
    }
    if let Some(spec) = mode.strip_prefix("ps-probe=") {
        let bad = || format!("--render-debug {mode}: expected ps-probe=FRAME:XxY[,XxY...]");
        let (frame, points) = spec.split_once(':').ok_or_else(bad)?;
        let frame: u64 = frame.parse().map_err(|_| bad())?;
        let points = points
            .split(',')
            .map(|p| {
                let (x, y) = p.split_once('x').ok_or_else(bad)?;
                Ok((x.parse().map_err(|_| bad())?, y.parse().map_err(|_| bad())?))
            })
            .collect::<Result<Vec<(u32, u32)>, String>>()?;
        if points.is_empty() || points.len() > 8 {
            return Err(bad());
        }
        log::warn!("renderer experiment: probing pixels {points:?} from frame {frame}");
        *PS_PROBE.lock().unwrap() = Some((frame, points));
        // The probe needs the vertex capture's binding; both at once is not supported.
        VS_CAPTURE_FRAME.store(0, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("bias-scale=") {
        let n: u32 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        BIAS_SCALE.store(n, Ordering::Relaxed);
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("alpha-depth-bias=") {
        let n: i32 = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        ALPHA_DEPTH_BIAS.store(n, Ordering::Relaxed);
        log::warn!("renderer experiment: depth bias {n} on alpha-tested draws");
        return Ok(());
    }
    if let Some(n) = mode.strip_prefix("frames=") {
        let n: usize = n.parse().map_err(|e| format!("--render-debug {mode}: {e}"))?;
        MAX_FRAMES_IN_FLIGHT.store(n.clamp(1, 16), Ordering::Relaxed);
        log::warn!("renderer experiment: up to {n} frames in flight");
        return Ok(());
    }
    let flag = match mode {
        "scissor" => &SCISSOR_ONE_PIXEL,
        "nodraw" => &SKIP_DRAW_CALLS,
        "nopool" => &NO_TEXTURE_POOL,
        "nomemo" => &NO_TEXTURE_SET_MEMO,
        "cpu-resolve" => &CPU_RESOLVES,
        "cpu-texture" => &CPU_TEXTURES,
        "no-bc" => &NO_BC,
        "no-d32s8" => &NO_D32S8,
        "no-a2c" => &NO_A2C,
        "single-sample" => &SINGLE_SAMPLE,
        "solid-skinned" => &SOLID_SKINNED,
        "no-alpha-test" => &NO_ALPHA_TEST,
        "alpha-view" => &ALPHA_VIEW,
        "alpha-view-kill" => &ALPHA_VIEW_KILL,
        "hash-pages" => &HASH_PAGES,
        "verify-loads" => &VERIFY_LOADS,
        "no-carry" => &NO_CARRY,
        "alpha-no-depth" => &ALPHA_NO_DEPTH,
        "no-skinned-alpha-test" => &NO_SKINNED_ALPHA_TEST,
        "d3d-surfaces" => {
            D3D_SURFACES.store(true, Ordering::Relaxed);
            return Ok(());
        }
        "gpu-time" => {
            GPU_TIME.store(true, Ordering::Relaxed);
            return Ok(());
        }
        "float32-depth" => {
            xenos_shader::set_float24_depth(false);
            log::warn!("renderer experiment: float24 depth targets keep float32 precision (layers z-fight)");
            return Ok(());
        }
        "alpha-mask" => {
            xenos_shader::set_alpha_test_by_sample_mask(true);
            log::info!("alpha test: failing pixels clear the sample mask instead of discarding");
            return Ok(());
        }
        "alpha-discard" => {
            xenos_shader::set_alpha_test_by_sample_mask(false);
            log::info!("alpha test: failing pixels are discarded");
            return Ok(());
        }
        "msl-dump" => {
            // Relative like the other dumps (on iOS the working directory is Documents).
            let dir = std::path::PathBuf::from("target/shader-dump/msl");
            log::info!("Metal backend: writing generated MSL to {}", dir.display());
            *gpu_webgpu::MSL_DUMP_DIR.lock().unwrap() = Some(dir);
            return Ok(());
        }
        other => {
            return Err(format!(
                "--render-debug: unknown mode {other} (scissor, nodraw, nopool, nomemo, cpu-resolve, cpu-texture, no-bc, no-d32s8, no-a2c, single-sample, solid-skinned, no-alpha-test, no-skinned-alpha-test, alpha-view, alpha-view-kill, hash-pages, verify-loads, no-carry, alpha-no-depth, alpha-depth-bias=N, bias-scale=N, vs-capture=N, ps-probe=FRAME:XxY, texture-mib=N, d3d-surfaces, gpu-time, msl-dump, float32-depth, alpha-mask, alpha-discard, frames=N, mirror=MIB, load-cap=MIB, upload-cap=MIB, unshared-upload, staging, no-staging, handheld, trace-frame=N, trace-every=K)"
            ));
        }
    };
    flag.store(true, Ordering::Relaxed);
    log::warn!("renderer experiment {mode} enabled: the picture will be wrong");
    Ok(())
}

/// The experiments in effect, for log lines ("" if none).
pub fn describe() -> String {
    let mut parts = Vec::new();
    if SCISSOR_ONE_PIXEL.load(Ordering::Relaxed) {
        parts.push("scissor".to_string());
    }
    if SKIP_DRAW_CALLS.load(Ordering::Relaxed) {
        parts.push("nodraw".to_string());
    }
    if NO_TEXTURE_POOL.load(Ordering::Relaxed) {
        parts.push("nopool".to_string());
    }
    if NO_TEXTURE_SET_MEMO.load(Ordering::Relaxed) {
        parts.push("nomemo".to_string());
    }
    if CPU_RESOLVES.load(Ordering::Relaxed) {
        parts.push("cpu-resolve".to_string());
    }
    if CPU_TEXTURES.load(Ordering::Relaxed) {
        parts.push("cpu-texture".to_string());
    }
    if NO_A2C.load(Ordering::Relaxed) {
        parts.push("no-a2c".to_string());
    }
    if SINGLE_SAMPLE.load(Ordering::Relaxed) {
        parts.push("single-sample".to_string());
    }
    if SOLID_SKINNED.load(Ordering::Relaxed) {
        parts.push("solid-skinned".to_string());
    }
    if NO_ALPHA_TEST.load(Ordering::Relaxed) {
        parts.push("no-alpha-test".to_string());
    }
    if ALPHA_VIEW_KILL.load(Ordering::Relaxed) {
        parts.push("alpha-view-kill".to_string());
    }
    if ALPHA_VIEW.load(Ordering::Relaxed) {
        parts.push("alpha-view".to_string());
    }
    if NO_SKINNED_ALPHA_TEST.load(Ordering::Relaxed) {
        parts.push("no-skinned-alpha-test".to_string());
    }
    if NO_D32S8.load(Ordering::Relaxed) {
        parts.push("no-d32s8".to_string());
    }
    if NO_BC.load(Ordering::Relaxed) {
        parts.push("no-bc".to_string());
    }
    let frames = MAX_FRAMES_IN_FLIGHT.load(Ordering::Relaxed);
    if frames != 2 {
        parts.push(format!("frames={frames}"));
    }
    if parts.is_empty() { String::new() } else { format!(" [experiments: {}]", parts.join(", ")) }
}
