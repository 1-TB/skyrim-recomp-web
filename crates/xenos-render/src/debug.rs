//! Renderer experiments for finding where frame time goes (`--render-debug`): each removes one
//! kind of GPU work while everything else runs as usual.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
        "d3d-surfaces" => {
            D3D_SURFACES.store(true, Ordering::Relaxed);
            return Ok(());
        }
        "gpu-time" => {
            GPU_TIME.store(true, Ordering::Relaxed);
            return Ok(());
        }
        other => {
            return Err(format!(
                "--render-debug: unknown mode {other} (scissor, nodraw, nopool, nomemo, cpu-resolve, cpu-texture, no-bc, no-d32s8, no-a2c, d3d-surfaces, gpu-time, frames=N, mirror=MIB, load-cap=MIB, upload-cap=MIB, unshared-upload, staging, no-staging, handheld, trace-frame=N, trace-every=K)"
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
