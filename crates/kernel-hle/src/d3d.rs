//! Hooks on the title's Direct3D functions (see the `d3d-hle` crate): guest-sys replaces each
//! function in [`d3d_hle::HOOKS`] with a call to [`skyrt_d3d_enter`] followed by the original.
//!
//! `--gpu xenos` (the default) leaves them inert. `--gpu d3d-probe` counts the calls per frame and
//! the threads they come from. `--gpu d3d-verify` also copies the device's register shadow at
//! every draw and puts a marker before the draw's packets, so that the command processor can
//! compare its registers with the copy when it gets there ([`verify_draw`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

use d3d_hle::verify::{DISCOVERY_DRAWS, Snapshot, Verifier};
use d3d_hle::{HOOKS, HookKind, device};

use crate::Kernel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GpuMode {
    /// The command processor renders everything (D3D hooks inert).
    Xenos = 0,
    /// Count and log the D3D calls.
    D3dProbe = 1,
    /// Probe, and check the device shadow layout against the command processor at every draw.
    D3dVerify = 2,
}

static MODE: AtomicU8 = AtomicU8::new(GpuMode::Xenos as u8);

/// Selects the GPU path from a `--gpu` value.
pub fn set_gpu_mode(value: &str) -> Result<(), String> {
    let mode = match value {
        "xenos" => GpuMode::Xenos,
        "d3d-probe" => GpuMode::D3dProbe,
        "d3d-verify" => GpuMode::D3dVerify,
        other => return Err(format!("--gpu: unknown mode {other} (xenos, d3d-probe, d3d-verify)")),
    };
    MODE.store(mode as u8, Ordering::Relaxed);
    Ok(())
}

pub fn gpu_mode() -> GpuMode {
    match MODE.load(Ordering::Relaxed) {
        1 => GpuMode::D3dProbe,
        2 => GpuMode::D3dVerify,
        _ => GpuMode::Xenos,
    }
}

/// Snapshots waiting for the command processor (it runs a frame or two behind); older ones are
/// dropped past this many.
const MAX_PENDING: usize = 16384;
/// Presents between reports.
const REPORT_PRESENTS: u64 = 300;

#[derive(Default)]
struct Probe {
    /// Calls per hook since the last report, and in total.
    calls: Vec<(u64, u64)>,
    /// Guest threads seen calling each hook.
    threads: Vec<BTreeSet<u32>>,
    /// Calls per guest thread since the last report.
    by_thread: BTreeMap<u32, u64>,
    /// The thread of the previous call, and calls from a different thread than the previous
    /// one: (from, to) -> count.
    last_thread: u32,
    switches: BTreeMap<(u32, u32), u64>,
    /// Distinct (r4, r5, r6) argument tuples of the first calls of each hook, for the log.
    logged: Vec<u32>,
    presents: u64,
    seq: u32,
    /// Markers that didn't fit in the command buffer (their draws aren't checked).
    no_room: u64,
}

#[derive(Default)]
struct Verify {
    pending: VecDeque<Snapshot>,
    verifier: Verifier,
    dropped: u64,
    /// Draw packets without a marker before them (emitted outside the hooked draw functions,
    /// or whose marker didn't fit), and the address of the last one.
    unmarked: u64,
    /// Unmarked draw packets by address (the first few hundred addresses).
    unmarked_at: BTreeMap<u32, u64>,
}

/// Command processor: a draw packet at `address` had no marker before it.
pub(crate) fn unmarked_draw(address: u32) {
    let mut guard = VERIFY.lock().unwrap();
    let v = guard.get_or_insert_with(Verify::default);
    v.unmarked += 1;
    if v.unmarked_at.len() < 256 || v.unmarked_at.contains_key(&address) {
        *v.unmarked_at.entry(address).or_default() += 1;
    }
}

static PROBE: Mutex<Option<Probe>> = Mutex::new(None);
static VERIFY: Mutex<Option<Verify>> = Mutex::new(None);

/// Called by the hook of `HOOKS[id]` before the original function, with its first argument
/// registers and return address.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_d3d_enter(id: u32, r3: u32, r4: u32, r5: u32, r6: u32, lr: u32) {
    let mode = gpu_mode();
    if mode == GpuMode::Xenos {
        return;
    }
    let Some(k) = Kernel::try_global() else { return };
    let Some(hook) = HOOKS.get(id as usize) else { return };
    let (thread, _) = crate::thread::current_caller();
    let mut guard = PROBE.lock().unwrap();
    let p = guard.get_or_insert_with(|| Probe {
        calls: vec![(0, 0); HOOKS.len()],
        threads: vec![BTreeSet::new(); HOOKS.len()],
        logged: vec![0; HOOKS.len()],
        ..Default::default()
    });
    let i = id as usize;
    p.calls[i].0 += 1;
    p.calls[i].1 += 1;
    *p.by_thread.entry(thread).or_default() += 1;
    if p.last_thread != thread {
        if p.last_thread != 0 {
            *p.switches.entry((p.last_thread, thread)).or_default() += 1;
        }
        p.last_thread = thread;
    }
    if p.threads[i].insert(thread) && p.threads[i].len() > 1 {
        log::warn!("d3d: {} ({}) called from a second thread {thread:#X}", hook.function, hook.name);
    }
    if p.logged[i] < 3 {
        p.logged[i] += 1;
        log::info!(
            "d3d: {} ({}) from {lr:#010X} on thread {thread:#X}: r3 {r3:#010X} r4 {r4:#010X} r5 {r5:#010X} r6 {r6:#010X}",
            hook.function,
            hook.name
        );
    }
    match hook.kind {
        HookKind::Present => {
            p.presents += 1;
            if p.presents.is_multiple_of(REPORT_PRESENTS) {
                report(p, REPORT_PRESENTS);
            }
        }
        HookKind::DrawLeaf if mode == GpuMode::D3dVerify => {
            p.seq = p.seq.wrapping_add(1);
            let seq = p.seq;
            if put_marker(k, r3, seq) {
                // The shadow is copied when the draw returns: the draw itself updates some
                // registers in it (the shader setup does) before writing them out.
                MARKED.with(|m| m.set(Some((seq, id, r3))));
            } else {
                p.no_room += 1;
            }
        }
        _ => {}
    }
}

thread_local! {
    /// The draw on this thread whose marker was written: (sequence number, hook, device).
    static MARKED: std::cell::Cell<Option<(u32, u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// Called by the hook of `HOOKS[id]` after the original function returned.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_d3d_leave(id: u32) {
    if gpu_mode() != GpuMode::D3dVerify {
        return;
    }
    let Some((seq, hook, dev)) = MARKED.with(|m| m.take()) else { return };
    if hook != id {
        MARKED.with(|m| m.set(Some((seq, hook, dev))));
        return;
    }
    let Some(k) = Kernel::try_global() else { return };
    let mem = k.mem();
    let whole = u64::from(seq) <= DISCOVERY_DRAWS;
    let snap = Snapshot::capture(seq, id, dev, whole, |a| mem.read_u32(a));
    let mut v = VERIFY.lock().unwrap();
    let v = v.get_or_insert_with(Verify::default);
    if v.pending.len() >= MAX_PENDING {
        v.pending.pop_front();
        v.dropped += 1;
    }
    v.pending.push_back(snap);
}

/// Writes the marker NOP (signature, `seq`) into the device's command buffer, if it fits within
/// the draw's own limit (the draw then makes space as it would have).
fn put_marker(k: &Kernel, dev: u32, seq: u32) -> bool {
    let mem = k.mem();
    let write = mem.read_u32(dev + device::COMMAND_WRITE);
    let limit = mem.read_u32(dev + device::COMMAND_LIMIT);
    let end = mem.read_u32(dev + device::COMMAND_END);
    if write == 0 || write + 12 > limit || write + 12 >= end {
        return false;
    }
    mem.write_u32(write + 4, d3d_hle::nop_header(2));
    mem.write_u32(write + 8, d3d_hle::MARKER_SIGNATURE);
    mem.write_u32(write + 12, seq);
    mem.write_u32(dev + device::COMMAND_WRITE, write + 12);
    true
}

fn report(p: &mut Probe, presents: u64) {
    let per_frame: Vec<String> = HOOKS
        .iter()
        .zip(&mut p.calls)
        .filter(|(_, c)| c.1 != 0)
        .map(|(h, c)| {
            let s = format!("{} {:.1}", h.function, c.0 as f64 / presents as f64);
            c.0 = 0;
            s
        })
        .collect();
    let threads: Vec<String> = std::mem::take(&mut p.by_thread).iter().map(|(t, n)| format!("{t:#X} {n}")).collect();
    let switches: Vec<String> =
        std::mem::take(&mut p.switches).iter().map(|((a, b), n)| format!("{a:#X}->{b:#X} {n}")).collect();
    if !switches.is_empty() {
        log::info!("d3d: calls changed threads: {}", switches.join(", "));
    }
    log::info!(
        "d3d: calls per frame over {presents} presents: {}; calls by thread {}{}",
        per_frame.join(", "),
        threads.join(", "),
        if p.no_room > 0 { format!("; {} draws without room for a marker", p.no_room) } else { String::new() }
    );
    if gpu_mode() == GpuMode::D3dVerify
        && let Some(v) = VERIFY.lock().unwrap().as_ref()
    {
        for line in v.verifier.report().lines() {
            log::info!("{line}");
        }
        if v.dropped > 0 {
            log::info!("d3d verify: {} snapshots dropped (the command processor fell behind)", v.dropped);
        }
        let mut at: Vec<(&u32, &u64)> = v.unmarked_at.iter().collect();
        at.sort_by(|a, b| b.1.cmp(a.1));
        let at: Vec<String> = at.iter().take(12).map(|(a, n)| format!("{a:#010X} {n}")).collect();
        log::info!(
            "d3d verify: {} draw packets without a marker ({} addresses; most at {}); markers that didn't fit: {}",
            v.unmarked,
            v.unmarked_at.len(),
            at.join(", "),
            p.no_room
        );
    }
}

/// Command processor: the draw after marker `seq` is about to execute with registers `read`.
pub(crate) fn verify_draw(
    seq: u32,
    written: &[(u32, u32)],
    read: impl Fn(u32) -> u32,
    last_writer: impl Fn(u32) -> (u32, u32),
) {
    let mut guard = VERIFY.lock().unwrap();
    let Some(v) = guard.as_mut() else { return };
    // Snapshots of markers whose draw never came (or that were replayed and already compared).
    while v.pending.front().is_some_and(|s| seq.wrapping_sub(s.seq) as i32 > 0) {
        v.pending.pop_front();
    }
    if let Some(snap) = v.pending.pop_front_if(|s| s.seq == seq) {
        v.verifier.compare(&snap, written, read, last_writer);
    }
}

/// Hook calls in total so far, by function (for tests and tools).
pub fn call_counts() -> BTreeMap<&'static str, u64> {
    let guard = PROBE.lock().unwrap();
    let Some(p) = guard.as_ref() else { return BTreeMap::new() };
    HOOKS.iter().zip(&p.calls).map(|(h, c)| (h.function, c.1)).collect()
}
