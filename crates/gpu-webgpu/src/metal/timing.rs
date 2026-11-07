//! GPU time per pass label (a profiling aid, off by default), from Metal counter sample buffers.
//!
//! Once [`enable`]d on a device that samples timestamps at stage boundaries (every Apple GPU),
//! every render and compute pass begun on this thread samples the GPU timestamp when its vertex
//! stage starts and its fragment stage ends (compute: when the encoder starts and ends) into a
//! sample buffer of its command buffer. [`collect`] (called once per frame) reads the buffers of
//! command buffers that completed and adds each pass's duration to its label's total, which
//! [`take`] returns. On a tile-based GPU passes overlap a little, so the times are approximate.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommonCounterSetTimestamp, MTLComputePassDescriptor,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
    MTLRenderPassDescriptor, MTLStorageMode,
};

use super::{Id, MtlCommandBuffer};
use crate::Device;

/// Samples per buffer (two per pass); passes past this many in one command buffer go untimed.
const SAMPLES: usize = 1024;

/// Metal's value for a sample the GPU didn't write.
const COUNTER_ERROR: u64 = u64::MAX;

type SampleBuffer = Id<dyn MTLCounterSampleBuffer>;

/// A command buffer's passes so far.
struct Recording {
    buffer: SampleBuffer,
    labels: Vec<String>,
}

struct InFlight {
    cb: MtlCommandBuffer,
    buffer: SampleBuffer,
    labels: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LabelTime {
    /// Sum of the passes' GPU time.
    pub nanos: u64,
    pub passes: u64,
}

struct Profiler {
    device: Id<dyn MTLDevice>,
    counters: Id<dyn MTLCounterSet>,
    recording: HashMap<usize, Recording>,
    in_flight: Vec<InFlight>,
    free: Vec<SampleBuffer>,
    totals: HashMap<String, LabelTime>,
    /// Passes left untimed (more than `SAMPLES / 2` in one command buffer).
    untimed: u64,
    /// GPU and CPU timestamps when profiling began, to convert GPU ticks into nanoseconds.
    origin: (u64, u64),
    origin_time: Instant,
    nanos_per_tick: f64,
}

thread_local! {
    static PROFILER: RefCell<Option<Profiler>> = const { RefCell::new(None) };
}

/// Both clocks now: (CPU nanoseconds, GPU ticks).
fn timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0u64, 0u64);
    // SAFETY: both pointers are valid for writes.
    unsafe { device.sampleTimestamps_gpuTimestamp(std::ptr::NonNull::from(&mut cpu), std::ptr::NonNull::from(&mut gpu)) };
    (cpu, gpu)
}

/// Starts timing this thread's passes on `device`. `false` if its GPU can't sample timestamps
/// at stage boundaries.
pub fn enable(device: &Device) -> bool {
    let raw = device.shared.raw.clone();
    if !raw.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return false;
    }
    // SAFETY: an extern static of Metal.
    let wanted = unsafe { MTLCommonCounterSetTimestamp };
    let Some(counters) =
        raw.counterSets().and_then(|sets| sets.iter().find(|s| s.name().to_string() == wanted.to_string()))
    else {
        return false;
    };
    let origin = timestamps(&raw);
    PROFILER.with_borrow_mut(|p| {
        *p = Some(Profiler {
            device: raw,
            counters,
            recording: HashMap::new(),
            in_flight: Vec::new(),
            free: Vec::new(),
            totals: HashMap::new(),
            untimed: 0,
            origin,
            origin_time: Instant::now(),
            nanos_per_tick: 1.0,
        })
    });
    true
}

pub fn enabled() -> bool {
    PROFILER.with_borrow(Option::is_some)
}

impl Profiler {
    fn buffer(&mut self) -> Option<SampleBuffer> {
        if let Some(b) = self.free.pop() {
            return Some(b);
        }
        let d = MTLCounterSampleBufferDescriptor::new();
        d.setCounterSet(Some(&self.counters));
        d.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a count within Metal's limit of 32 KiB samples per buffer.
        unsafe { d.setSampleCount(SAMPLES) };
        match self.device.newCounterSampleBufferWithDescriptor_error(&d) {
            Ok(b) => Some(b),
            Err(e) => {
                log::warn!("gpu-time: counter sample buffer: {e}");
                None
            }
        }
    }

    /// The sample buffer and first sample index for a pass `label` in `cb`.
    fn pass(&mut self, cb: &MtlCommandBuffer, label: &str) -> Option<(SampleBuffer, usize)> {
        let key = super::Retained::as_ptr(cb) as *const () as usize;
        if !self.recording.contains_key(&key) {
            let buffer = self.buffer()?;
            self.recording.insert(key, Recording { buffer, labels: Vec::new() });
        }
        let rec = self.recording.get_mut(&key).expect("inserted above");
        let index = rec.labels.len() * 2;
        if index + 2 > SAMPLES {
            self.untimed += 1;
            return None;
        }
        rec.labels.push(label.to_string());
        Some((rec.buffer.clone(), index))
    }
}

/// Adds timestamp sampling for a render pass `label` in `cb` to `desc`, if profiling.
pub(crate) fn render_pass(cb: &MtlCommandBuffer, label: &str, desc: &MTLRenderPassDescriptor) {
    PROFILER.with_borrow_mut(|p| {
        let Some((buffer, index)) = p.as_mut().and_then(|p| p.pass(cb, label)) else { return };
        // SAFETY: index 0 of the attachment array always exists; indices are within the buffer.
        unsafe {
            let a = desc.sampleBufferAttachments().objectAtIndexedSubscript(0);
            a.setSampleBuffer(Some(&buffer));
            a.setStartOfVertexSampleIndex(index);
            a.setEndOfVertexSampleIndex(objc2_metal::MTLCounterDontSample);
            a.setStartOfFragmentSampleIndex(objc2_metal::MTLCounterDontSample);
            a.setEndOfFragmentSampleIndex(index + 1);
        }
    });
}

/// A compute pass descriptor sampling timestamps for a pass `label` in `cb`, if profiling.
pub(crate) fn compute_pass(cb: &MtlCommandBuffer, label: &str) -> Option<objc2::rc::Retained<MTLComputePassDescriptor>> {
    PROFILER.with_borrow_mut(|p| {
        let (buffer, index) = p.as_mut()?.pass(cb, label)?;
        let desc = MTLComputePassDescriptor::new();
        // SAFETY: as for render passes.
        unsafe {
            let a = desc.sampleBufferAttachments().objectAtIndexedSubscript(0);
            a.setSampleBuffer(Some(&buffer));
            a.setStartOfEncoderSampleIndex(index);
            a.setEndOfEncoderSampleIndex(index + 1);
        }
        Some(desc)
    })
}

/// `cb` was finished: its samples are read once it has completed.
pub(crate) fn finished(cb: &MtlCommandBuffer) {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p else { return };
        let key = super::Retained::as_ptr(cb) as *const () as usize;
        if let Some(rec) = p.recording.remove(&key) {
            p.in_flight.push(InFlight { cb: cb.clone(), buffer: rec.buffer, labels: rec.labels });
        }
    });
}

/// Reads the samples of command buffers that completed since the last call.
pub fn collect() {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p else { return };
        // Calibrate GPU ticks against the CPU clock once a second has passed (Apple GPUs tick
        // in nanoseconds, but that's not promised).
        if p.origin_time.elapsed().as_secs() >= 1 {
            let (cpu, gpu) = timestamps(&p.device);
            if gpu > p.origin.1 && cpu > p.origin.0 {
                p.nanos_per_tick = (cpu - p.origin.0) as f64 / (gpu - p.origin.1) as f64;
            }
        }
        let mut i = 0;
        while i < p.in_flight.len() {
            let status = p.in_flight[i].cb.status();
            if status != MTLCommandBufferStatus::Completed && status != MTLCommandBufferStatus::Error {
                i += 1;
                continue;
            }
            let entry = p.in_flight.swap_remove(i);
            if status == MTLCommandBufferStatus::Completed {
                let range = NSRange::new(0, entry.labels.len() * 2);
                // SAFETY: the range lies within the buffer and its command buffer completed.
                let data = autoreleasepool(|_| unsafe { entry.buffer.resolveCounterRange(range) });
                if let Some(data) = data {
                    let bytes = data.to_vec();
                    for (label, pair) in entry.labels.iter().zip(bytes.chunks_exact(16)) {
                        let begin = u64::from_le_bytes(pair[..8].try_into().expect("8 bytes"));
                        let end = u64::from_le_bytes(pair[8..].try_into().expect("8 bytes"));
                        if begin == COUNTER_ERROR || end == COUNTER_ERROR || end < begin {
                            continue;
                        }
                        let t = p.totals.entry(label.clone()).or_default();
                        t.nanos += ((end - begin) as f64 * p.nanos_per_tick) as u64;
                        t.passes += 1;
                    }
                }
            }
            p.free.push(entry.buffer);
        }
    });
}

/// GPU time per pass label since the last call (largest first), and passes left untimed.
pub fn take() -> (Vec<(String, LabelTime)>, u64) {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p else { return (Vec::new(), 0) };
        let mut totals: Vec<(String, LabelTime)> = p.totals.drain().collect();
        totals.sort_by(|a, b| b.1.nanos.cmp(&a.1.nanos));
        (totals, std::mem::take(&mut p.untimed))
    })
}
