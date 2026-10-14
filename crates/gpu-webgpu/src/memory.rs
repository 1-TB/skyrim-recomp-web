//! Bytes of the buffers and textures created through this crate and not yet destroyed, per label,
//! for measuring the GPU memory the renderer holds (phones close tabs that use too much).
//!
//! Objects leave the tally on `destroy()`, or when the driver hands their address to a new object
//! (the old one was freed). Ones only dropped stay counted: in browsers, objects released in
//! workers are freed only by garbage collection, so they do still hold memory.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::TextureFormat;

struct Live {
    /// Raw handle address -> (kind, label, bytes).
    objects: BTreeMap<usize, (&'static str, String, u64)>,
    total: u64,
    peak: u64,
}

static LIVE: Mutex<Live> = Mutex::new(Live { objects: BTreeMap::new(), total: 0, peak: 0 });

pub(crate) fn track(raw: usize, kind: &'static str, label: &str, bytes: u64) {
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, _, old)) = live.objects.insert(raw, (kind, label.to_string(), bytes)) {
        live.total -= old;
    }
    live.total += bytes;
    live.peak = live.peak.max(live.total);
}

pub(crate) fn untrack(raw: usize) {
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, _, bytes)) = live.objects.remove(&raw) {
        live.total -= bytes;
    }
}

/// Bytes live now and the most ever live at once.
pub fn totals() -> (u64, u64) {
    let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    (live.total, live.peak)
}

/// One group of live objects with the same kind and label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveGroup {
    pub kind: &'static str,
    pub label: String,
    pub count: u64,
    pub bytes: u64,
}

/// Live objects grouped by kind and label, largest first.
pub fn live_by_label() -> Vec<LiveGroup> {
    let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    let mut groups: Vec<LiveGroup> = Vec::new();
    for (kind, label, bytes) in live.objects.values() {
        match groups.iter_mut().find(|g| g.kind == *kind && g.label == *label) {
            Some(g) => {
                g.count += 1;
                g.bytes += bytes;
            }
            None => groups.push(LiveGroup { kind, label: label.clone(), count: 1, bytes: *bytes }),
        }
    }
    groups.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    groups
}

/// Approximate size of a texture: all mips and samples, depth/stencil at their usual sizes.
pub(crate) fn texture_bytes(
    format: TextureFormat,
    width: u32,
    height: u32,
    layers: u32,
    mips: u32,
    samples: u32,
) -> u64 {
    use TextureFormat as F;
    let (block_bytes, block) = crate::format_block_info(format).unwrap_or(match format {
        F::Depth24Plus | F::Depth24PlusStencil8 => (4, 1),
        F::Depth32FloatStencil8 => (8, 1),
        _ => (4, 1),
    });
    let mut total = 0u64;
    for level in 0..mips.max(1) {
        let w = (width >> level).max(1).div_ceil(block) as u64;
        let h = (height >> level).max(1).div_ceil(block) as u64;
        total += w * h * u64::from(block_bytes);
    }
    total * u64::from(layers.max(1)) * u64::from(samples.max(1))
}
