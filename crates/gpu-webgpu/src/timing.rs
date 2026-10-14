//! GPU time per pass label, measured with timestamp queries (a profiling aid, off by default).
//!
//! Once [`enable`]d on a device created with `FeatureName::TimestampQuery`, every render and
//! compute pass begun on this thread gets timestamp writes in a query set of its encoder. When
//! the encoder is finished the queries are resolved and copied to a mappable buffer; [`collect`]
//! (called once per frame, after everything recorded so far was submitted) maps those buffers
//! and adds each pass's duration to its label's total, which [`take`] returns.
//!
//! Browsers quantize timestamps (Chrome to 100 µs) unless started with developer features
//! (`--enable-webgpu-developer-features`), so short passes then read as 0 or 100 µs.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::{Buffer, BufferDescriptor, BufferUsage, CommandEncoder, Device, FeatureName, MapMode, Pending, ffi};

/// Queries per encoder (two per pass); passes past this many in one encoder go untimed.
const QUERIES: u32 = 1024;

/// The query set and buffers one encoder's timestamps use, reused once read back.
struct Slot {
    set: ffi::WGPUQuerySet,
    resolve: Buffer,
    readback: Buffer,
}

impl Drop for Slot {
    fn drop(&mut self) {
        unsafe {
            ffi::wgpuQuerySetDestroy(self.set);
            ffi::wgpuQuerySetRelease(self.set);
        }
        self.resolve.destroy();
        self.readback.destroy();
    }
}

/// An encoder's passes so far.
struct Recording {
    slot: Slot,
    labels: Vec<String>,
}

enum Stage {
    /// Finished; mapped at the next [`collect`] (by then it has been submitted).
    Finished,
    Mapping(Pending<crate::Result<()>>),
}

struct InFlight {
    slot: Slot,
    labels: Vec<String>,
    stage: Stage,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LabelTime {
    /// Sum of the passes' GPU time.
    pub nanos: u64,
    pub passes: u64,
}

struct Profiler {
    device: Device,
    recording: HashMap<usize, Recording>,
    in_flight: Vec<InFlight>,
    free: Vec<Slot>,
    totals: HashMap<String, LabelTime>,
    /// Passes left untimed (more than `QUERIES / 2` in one encoder).
    untimed: u64,
}

thread_local! {
    static PROFILER: RefCell<Option<Profiler>> = const { RefCell::new(None) };
}

/// Starts timing this thread's passes on `device`. `false` if the device lacks timestamp queries.
pub fn enable(device: &Device) -> bool {
    if !device.has_feature(FeatureName::TimestampQuery) {
        return false;
    }
    PROFILER.with_borrow_mut(|p| {
        *p = Some(Profiler {
            device: device.clone(),
            recording: HashMap::new(),
            in_flight: Vec::new(),
            free: Vec::new(),
            totals: HashMap::new(),
            untimed: 0,
        })
    });
    true
}

pub fn enabled() -> bool {
    PROFILER.with_borrow(Option::is_some)
}

impl Profiler {
    fn slot(&mut self) -> Slot {
        self.free.pop().unwrap_or_else(|| {
            let desc = ffi::WGPUQuerySetDescriptor {
                label: crate::handle::label("timestamps"),
                r#type: ffi::WGPUQueryType::Timestamp,
                count: QUERIES,
                ..Default::default()
            };
            let set = unsafe { ffi::wgpuDeviceCreateQuerySet(self.device.raw(), &desc) };
            assert!(!set.is_null(), "wgpuDeviceCreateQuerySet returned null");
            let buffer = |label, usage| {
                self.device.create_buffer(&BufferDescriptor {
                    label,
                    size: u64::from(QUERIES) * 8,
                    usage,
                    mapped_at_creation: false,
                })
            };
            Slot {
                set,
                resolve: buffer("timestamp resolve", BufferUsage::QueryResolve | BufferUsage::CopySrc),
                readback: buffer("timestamp readback", BufferUsage::MapRead | BufferUsage::CopyDst),
            }
        })
    }
}

/// A new encoder at `encoder`: a recording left by an earlier encoder at the same address (one
/// dropped without being finished) is discarded.
pub(crate) fn encoder_created(encoder: ffi::WGPUCommandEncoder) {
    PROFILER.with_borrow_mut(|p| {
        if let Some(p) = p
            && let Some(old) = p.recording.remove(&(encoder as usize))
        {
            p.free.push(old.slot);
        }
    });
}

/// Timestamp writes for a pass `label` begun on `encoder`, if profiling.
pub(crate) fn pass_writes(encoder: ffi::WGPUCommandEncoder, label: &str) -> Option<ffi::WGPUPassTimestampWrites> {
    PROFILER.with_borrow_mut(|p| {
        let p = p.as_mut()?;
        if !p.recording.contains_key(&(encoder as usize)) {
            let slot = p.slot();
            p.recording.insert(encoder as usize, Recording { slot, labels: Vec::new() });
        }
        let rec = p.recording.get_mut(&(encoder as usize)).expect("inserted above");
        let index = rec.labels.len() as u32 * 2;
        if index + 2 > QUERIES {
            p.untimed += 1;
            return None;
        }
        rec.labels.push(label.to_string());
        Some(ffi::WGPUPassTimestampWrites {
            querySet: rec.slot.set,
            beginningOfPassWriteIndex: index,
            endOfPassWriteIndex: index + 1,
            ..Default::default()
        })
    })
}

/// Before `encoder` is finished: resolves its timestamps into the readback buffer.
pub(crate) fn encoder_finishing(encoder: &mut CommandEncoder) {
    let rec = PROFILER.with_borrow_mut(|p| p.as_mut()?.recording.remove(&(encoder.raw() as usize)));
    let Some(rec) = rec else {
        return;
    };
    let count = rec.labels.len() as u32 * 2;
    unsafe { ffi::wgpuCommandEncoderResolveQuerySet(encoder.raw(), rec.slot.set, 0, count, rec.slot.resolve.raw(), 0) };
    encoder.copy_buffer_to_buffer(&rec.slot.resolve, 0, &rec.slot.readback, 0, u64::from(count) * 8);
    PROFILER.with_borrow_mut(|p| {
        if let Some(p) = p {
            p.in_flight.push(InFlight { slot: rec.slot, labels: rec.labels, stage: Stage::Finished });
        }
    });
}

/// Reads back the timestamps that arrived and starts mapping those finished since the last
/// call. Call when everything finished so far has been submitted (e.g. at the end of a frame).
pub fn collect() {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p else { return };
        let mut i = 0;
        while i < p.in_flight.len() {
            let entry = &mut p.in_flight[i];
            match &entry.stage {
                Stage::Finished => {
                    let size = entry.labels.len() as u64 * 16;
                    entry.stage = Stage::Mapping(entry.slot.readback.map_async(MapMode::Read, 0, Some(size)));
                    i += 1;
                }
                Stage::Mapping(pending) => {
                    let Some(result) = pending.try_take() else {
                        i += 1;
                        continue;
                    };
                    let entry = p.in_flight.swap_remove(i);
                    if result.is_ok() {
                        let mut bytes = vec![0u8; entry.labels.len() * 16];
                        if entry.slot.readback.read_mapped(0, &mut bytes).is_ok() {
                            for (label, pair) in entry.labels.iter().zip(bytes.chunks_exact(16)) {
                                let begin = u64::from_le_bytes(pair[..8].try_into().expect("8 bytes"));
                                let end = u64::from_le_bytes(pair[8..].try_into().expect("8 bytes"));
                                let t = p.totals.entry(label.clone()).or_default();
                                t.nanos += end.saturating_sub(begin);
                                t.passes += 1;
                            }
                        }
                    }
                    entry.slot.readback.unmap();
                    p.free.push(entry.slot);
                }
            }
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
