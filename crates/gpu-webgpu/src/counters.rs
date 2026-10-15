//! Counts of the WebGPU operations made through this crate, for finding which ones a frame
//! spends its time on (browsers pay per call, and per object created).

use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        /// A snapshot of the counters (totals since start).
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct Counters {
            $(pub $name: u64,)*
        }

        pub(crate) mod count {
            $(pub(crate) static $name: super::AtomicU64 = super::AtomicU64::new(0);)*
        }

        /// The totals so far.
        pub fn snapshot() -> Counters {
            Counters { $($name: count::$name.load(Ordering::Relaxed),)* }
        }

        impl Counters {
            /// `self - earlier`, field by field.
            pub fn since(&self, earlier: &Counters) -> Counters {
                Counters { $($name: self.$name - earlier.$name,)* }
            }
        }
    };
}

#[allow(non_upper_case_globals)]
mod defs {
    use super::*;
    counters!(
        buffers,
        textures,
        views,
        bind_groups,
        samplers,
        pipelines,
        render_passes,
        compute_passes,
        submits,
        write_buffers,
        write_buffer_bytes,
        write_textures,
        write_texture_bytes,
        copies,
        set_pipelines,
        set_bind_groups,
        draws,
        dispatches,
        shader_modules,
        shader_bytes,
    );
}
pub(crate) use defs::count;
pub use defs::{Counters, snapshot};

/// Bytes uploaded so far (`write_buffer` plus `write_texture`).
pub fn upload_bytes() -> u64 {
    count::write_buffer_bytes.load(Ordering::Relaxed) + count::write_texture_bytes.load(Ordering::Relaxed)
}

/// Adds `n` to a counter.
#[inline]
pub(crate) fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Ordering::Relaxed);
}

/// Objects created per label (bind groups and textures), to find who creates them.
static CREATED_BY_LABEL: std::sync::Mutex<Vec<(&'static str, String, u64)>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn created(kind: &'static str, label: &str) {
    let mut v = CREATED_BY_LABEL.lock().unwrap_or_else(|e| e.into_inner());
    match v.iter_mut().find(|(k, l, _)| *k == kind && l == label) {
        Some(e) => e.2 += 1,
        None => v.push((kind, label.to_string(), 1)),
    }
}

/// Creation counts per (kind, label) since the last call, most first.
pub fn take_created_by_label() -> Vec<(&'static str, String, u64)> {
    let mut v = std::mem::take(&mut *CREATED_BY_LABEL.lock().unwrap_or_else(|e| e.into_inner()));
    v.sort_by(|a, b| b.2.cmp(&a.2));
    v
}

/// Calls and bytes of `Queue::write_buffer` per destination buffer label, to find what the
/// uploads are.
static WRITTEN_BY_LABEL: std::sync::Mutex<Vec<(String, u64, u64)>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn written(label: &str, bytes: u64) {
    let mut v = WRITTEN_BY_LABEL.lock().unwrap_or_else(|e| e.into_inner());
    match v.iter_mut().find(|(l, ..)| l == label) {
        Some(e) => {
            e.1 += 1;
            e.2 += bytes;
        }
        None => v.push((label.to_string(), 1, bytes)),
    }
}

/// (label, calls, bytes) uploaded per buffer label since the last call, most bytes first.
pub fn take_written_by_label() -> Vec<(String, u64, u64)> {
    let mut v = std::mem::take(&mut *WRITTEN_BY_LABEL.lock().unwrap_or_else(|e| e.into_inner()));
    v.sort_by(|a, b| b.2.cmp(&a.2));
    v
}
