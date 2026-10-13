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

/// Frames the renderer may run ahead of the GPU (2 unless an experiment changes it).
pub(crate) static MAX_FRAMES_IN_FLIGHT: AtomicUsize = AtomicUsize::new(2);

/// Turns on experiment `mode`: `scissor`, `nodraw` or `frames=N` (frames in flight).
pub fn enable(mode: &str) -> Result<(), String> {
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
        other => {
            return Err(format!(
                "--render-debug: unknown mode {other} (scissor, nodraw, nopool, nomemo, cpu-resolve, cpu-texture, frames=N)"
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
    let frames = MAX_FRAMES_IN_FLIGHT.load(Ordering::Relaxed);
    if frames != 2 {
        parts.push(format!("frames={frames}"));
    }
    if parts.is_empty() { String::new() } else { format!(" [experiments: {}]", parts.join(", ")) }
}
