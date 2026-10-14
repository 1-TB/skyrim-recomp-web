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

/// Frames the renderer may run ahead of the GPU (2 unless an experiment changes it).
pub(crate) static MAX_FRAMES_IN_FLIGHT: AtomicUsize = AtomicUsize::new(2);
/// Don't use BC texture compression even when the device has it (as on phones, where BC
/// textures are decompressed), to measure what that costs.
pub(crate) static NO_BC: AtomicBool = AtomicBool::new(false);

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
        other => {
            return Err(format!(
                "--render-debug: unknown mode {other} (scissor, nodraw, nopool, nomemo, cpu-resolve, cpu-texture, no-bc, frames=N, mirror=MIB, handheld, trace-frame=N, trace-every=K)"
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
    if NO_BC.load(Ordering::Relaxed) {
        parts.push("no-bc".to_string());
    }
    let frames = MAX_FRAMES_IN_FLIGHT.load(Ordering::Relaxed);
    if frames != 2 {
        parts.push(format!("frames={frames}"));
    }
    if parts.is_empty() { String::new() } else { format!(" [experiments: {}]", parts.join(", ")) }
}
