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

use d3d_hle::args::{CallRegs, ClearArgs, ResolveArgs, SurfaceHeader, TextureHeader};
use d3d_hle::verify::{ArgumentChecks, DISCOVERY_DRAWS, Snapshot, Verifier};
use d3d_hle::{HOOKS, HookKind, device};
use runtime_core::PPCContext;

use crate::Kernel;

/// A hooked call's registers, and guest memory for its pointer arguments.
struct Call<'a> {
    ctx: &'a PPCContext,
    k: &'a Kernel,
}

impl CallRegs for Call<'_> {
    fn gpr(&self, n: usize) -> u32 {
        self.ctx.gpr32(n)
    }

    fn fpr(&self, n: usize) -> f64 {
        self.ctx.fpr_f64(n)
    }

    fn read(&self, address: u32) -> u32 {
        self.k.mem().read_u32(address)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GpuMode {
    /// The command processor renders everything (D3D hooks inert).
    Xenos = 0,
    /// Count and log the D3D calls.
    D3dProbe = 1,
    /// Probe, and check the device shadow layout against the command processor at every draw.
    D3dVerify = 2,
    /// Render from the D3D calls (the renderer gets [`d3d_hle::command::Command`]s, not the
    /// command stream).
    D3d = 3,
}

static MODE: AtomicU8 = AtomicU8::new(GpuMode::Xenos as u8);

/// Selects the GPU path from a `--gpu` value.
pub fn set_gpu_mode(value: &str) -> Result<(), String> {
    let mode = match value {
        "xenos" => GpuMode::Xenos,
        "d3d-probe" => GpuMode::D3dProbe,
        "d3d-verify" => GpuMode::D3dVerify,
        "d3d" => GpuMode::D3d,
        other => return Err(format!("--gpu: unknown mode {other} (xenos, d3d, d3d-probe, d3d-verify)")),
    };
    MODE.store(mode as u8, Ordering::Relaxed);
    Ok(())
}

pub fn gpu_mode() -> GpuMode {
    match MODE.load(Ordering::Relaxed) {
        1 => GpuMode::D3dProbe,
        2 => GpuMode::D3dVerify,
        3 => GpuMode::D3d,
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
    /// Surfaces, clears and resolves logged so far (the first few of each).
    surfaces_logged: BTreeSet<u32>,
    clears_logged: u32,
    resolves_logged: u32,
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
    args: ArgumentChecks,
    /// Resolves whose copy draw has marker `seq`: the decoded call and its destination.
    resolves: BTreeMap<u32, (ResolveArgs, Option<TextureHeader>)>,
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

/// Called by the hook of `HOOKS[id]` before the original function, with the call's registers.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_d3d_enter(id: u32, ctx: &PPCContext) {
    let mode = gpu_mode();
    if mode == GpuMode::Xenos {
        return;
    }
    let Some(k) = Kernel::try_global() else { return };
    let Some(hook) = HOOKS.get(id as usize) else { return };
    let call = Call { ctx, k };
    let (r3, r4, r5, r6, lr) = (ctx.gpr32(3), ctx.gpr32(4), ctx.gpr32(5), ctx.gpr32(6), ctx.lr as u32);
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
    let mem = k.mem();
    match hook.kind {
        HookKind::SetRenderTarget | HookKind::SetDepthStencilSurface => {
            let surface = if hook.kind == HookKind::SetRenderTarget { r5 } else { r4 };
            if surface != 0 && p.surfaces_logged.len() < 24 && p.surfaces_logged.insert(surface) {
                let h = SurfaceHeader::read(surface, |a| mem.read_u32(a));
                log::info!("d3d: {} surface {surface:#010X}: {:08X?}", hook.name, h.words);
            }
        }
        HookKind::Clear => {
            if p.clears_logged < 16 {
                p.clears_logged += 1;
                log::info!("d3d: Clear {:X?}", ClearArgs::read(&call));
            }
        }
        HookKind::Resolve => {
            let args = ResolveArgs::read(&call);
            let dest = (args.dest_texture != 0).then(|| TextureHeader::read(args.dest_texture, |a| mem.read_u32(a)));
            if p.resolves_logged < 16 {
                p.resolves_logged += 1;
                log::info!("d3d: Resolve {args:X?}; destination {dest:08X?}");
            }
            if mode == GpuMode::D3dVerify {
                RESOLVE.with(|r| *r.borrow_mut() = Some((args, dest)));
            }
        }
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
                // A resolve's copy draw: check its arguments against the copy registers.
                if let Some(resolve) = RESOLVE.with(|r| r.borrow().clone()) {
                    VERIFY.lock().unwrap().get_or_insert_with(Verify::default).resolves.insert(seq, resolve);
                }
            } else {
                p.no_room += 1;
            }
        }
        _ => {}
    }
    if mode == GpuMode::D3d {
        drop(guard);
        d3d_enter(k, hook.kind, &call);
    }
}

/// `--gpu d3d`. The title's draws reach the renderer through the command stream, and D3D calls
/// must stay in order with them: each call is queued here and a marker with its id goes into
/// the command buffer; the command processor delivers the queued calls up to that id when it
/// gets there ([`take_calls`]). Draws get markers too, with a sequence number: the command
/// processor executes a draw only right after a marker with a new number, which leaves out the
/// draws D3D itself makes for clears, resolves and tiling (and tile replays of the title's own).
struct Calls {
    queue: VecDeque<(u32, d3d_hle::command::Command)>,
    next_call: u32,
    next_draw: u32,
}

static CALLS: Mutex<Calls> = Mutex::new(Calls { queue: VecDeque::new(), next_call: 0, next_draw: 0 });

/// Marker kinds of `--gpu d3d` (payload: signature, kind, value).
pub(crate) const MARKER_DRAW: u32 = 1;
pub(crate) const MARKER_CALL: u32 = 2;
/// A resolve's copy draw: runs in every tile it comes by in (each tile is resolved on its own).
pub(crate) const MARKER_RESOLVE_DRAW: u32 = 3;
/// A D3D call whose draws aren't the title's begins (tiling).
pub(crate) const MARKER_INTERNAL: u32 = 4;
/// First payload word of `--gpu d3d` markers ('SKDC').
pub(crate) const CALL_MARKER_SIGNATURE: u32 = 0x534B_4443;

thread_local! {
    /// Depth of D3D calls on this thread whose own draws are not the title's (Clear, Resolve,
    /// Begin/EndTiling).
    static INTERNAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Resolve calls this thread is in, with the call to deliver when each returns: its copy
    /// draws run (the renderer stores the copy into guest memory and does its clears), and the
    /// call itself comes after them.
    static RESOLVES: std::cell::RefCell<Vec<(u32, d3d_hle::command::Command)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn internal_call(kind: HookKind) -> bool {
    matches!(kind, HookKind::Clear | HookKind::Resolve | HookKind::BeginTiling | HookKind::EndTiling)
}

/// Bytes past the command buffer's soft limit that ending a segment takes (with some margin).
const SEGMENT_END_BYTES: u32 = 64;

/// Writes a `--gpu d3d` marker into the device's command buffer, if it fits.
fn put_call_marker(k: &Kernel, dev: u32, kind: u32, value: u32) -> bool {
    let mem = k.mem();
    let write = mem.read_u32(dev + device::COMMAND_WRITE);
    let end = mem.read_u32(dev + device::COMMAND_END);
    // Past the soft limit the title kicks the segment off before its next packets; up to
    // `end` it keeps room for that (an indirect buffer return and a few events, 60 bytes), of
    // which the marker may take some: it then lands before the kick and its draw after it.
    if write == 0 || write + 16 + SEGMENT_END_BYTES > end {
        return false;
    }
    mem.write_u32(write + 4, d3d_hle::nop_header(3));
    mem.write_u32(write + 8, CALL_MARKER_SIGNATURE);
    mem.write_u32(write + 12, kind);
    mem.write_u32(write + 16, value);
    mem.write_u32(dev + device::COMMAND_WRITE, write + 16);
    true
}

fn d3d_enter(k: &Kernel, kind: HookKind, call: &Call<'_>) {
    let dev = call.gpr(3);
    if kind == HookKind::DrawLeaf {
        let in_resolve = RESOLVES.with(|r| !r.borrow().is_empty());
        if INTERNAL.with(|d| d.get()) == 0 || in_resolve {
            let seq = {
                let mut c = CALLS.lock().unwrap();
                c.next_draw = c.next_draw.wrapping_add(1);
                c.next_draw
            };
            put_call_marker(k, dev, if in_resolve { MARKER_RESOLVE_DRAW } else { MARKER_DRAW }, seq);
        }
        return;
    }
    if internal_call(kind) {
        INTERNAL.with(|d| d.set(d.get() + 1));
        if kind != HookKind::Clear {
            put_call_marker(k, dev, MARKER_INTERNAL, 0);
        }
    }
    let Some(cmd) = d3d_command(k, kind, call) else { return };
    if kind == HookKind::Resolve {
        RESOLVES.with(|r| r.borrow_mut().push((dev, cmd)));
        return;
    }
    queue_call(k, dev, cmd);
}

/// Queues a D3D call and puts its marker into the command buffer.
fn queue_call(k: &Kernel, dev: u32, cmd: d3d_hle::command::Command) {
    let id = {
        let mut c = CALLS.lock().unwrap();
        c.next_call = c.next_call.wrapping_add(1);
        let id = c.next_call;
        c.queue.push_back((id, cmd));
        id
    };
    // Without room the call is delivered with the next marker (or at the next swap).
    put_call_marker(k, dev, MARKER_CALL, id);
}

/// `--gpu d3d`: the hook of an internal call returned.
fn d3d_leave(kind: HookKind) {
    if internal_call(kind) {
        INTERNAL.with(|d| d.set(d.get().saturating_sub(1)));
    }
    if kind == HookKind::Resolve
        && let Some((dev, cmd)) = RESOLVES.with(|r| r.borrow_mut().pop())
        && let Some(k) = Kernel::try_global()
    {
        queue_call(k, dev, cmd);
    }
}

/// Command processor: the D3D calls queued up to marker `id`.
pub(crate) fn take_calls(id: u32) -> Vec<d3d_hle::command::Command> {
    let mut c = CALLS.lock().unwrap();
    let mut out = Vec::new();
    while let Some((call, _)) = c.queue.front() {
        if call.wrapping_sub(id) as i32 > 0 {
            break;
        }
        out.push(c.queue.pop_front().expect("checked").1);
    }
    out
}

/// The renderer's command for a D3D call.
fn d3d_command(k: &Kernel, kind: HookKind, call: &Call<'_>) -> Option<d3d_hle::command::Command> {
    use d3d_hle::command::{Command, Surface};
    let mem = k.mem();
    let surface = |address: u32, depth: bool| {
        (address != 0).then(|| Surface { address, header: SurfaceHeader::read(address, |a| mem.read_u32(a)), depth })
    };
    let texture = |address: u32| (address != 0).then(|| TextureHeader::read(address, |a| mem.read_u32(a)));
    let cmd = match kind {
        HookKind::SetRenderTarget => {
            Command::SetRenderTarget { index: call.arg(1), surface: surface(call.arg(2), false) }
        }
        HookKind::SetDepthStencilSurface => Command::SetDepthStencil { surface: surface(call.arg(1), true) },
        HookKind::Clear => Command::Clear(ClearArgs::read(call)),
        HookKind::Resolve => {
            let args = ResolveArgs::read(call);
            let dest = texture(args.dest_texture);
            Command::Resolve { args, dest }
        }
        HookKind::Present => Command::Present { front: texture(call.arg(1)) },
        _ => return None,
    };
    Some(cmd)
}

thread_local! {
    /// The draw on this thread whose marker was written: (sequence number, hook, device).
    static MARKED: std::cell::Cell<Option<(u32, u32, u32)>> = const { std::cell::Cell::new(None) };
    /// The Resolve call this thread is in (its copy draws are checked against it).
    static RESOLVE: std::cell::RefCell<Option<(ResolveArgs, Option<TextureHeader>)>> =
        const { std::cell::RefCell::new(None) };
}

/// Called by the hook of `HOOKS[id]` after the original function returned.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_d3d_leave(id: u32, _ctx: &PPCContext) {
    if gpu_mode() == GpuMode::D3d
        && let Some(hook) = HOOKS.get(id as usize)
    {
        d3d_leave(hook.kind);
        return;
    }
    if gpu_mode() != GpuMode::D3dVerify {
        return;
    }
    if HOOKS.get(id as usize).is_some_and(|h| h.kind == HookKind::Resolve) {
        RESOLVE.with(|r| r.borrow_mut().take());
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
    // The bound surfaces against the RB registers the draw just wrote from the shadow.
    let surface = |offset: u32| {
        let a = mem.read_u32(dev + offset);
        (a != 0).then(|| SurfaceHeader::read(a, |x| mem.read_u32(x)))
    };
    let targets = std::array::from_fn(|i| surface(device::RENDER_TARGETS + 4 * i as u32));
    let depth = surface(device::DEPTH_STENCIL);
    let shadow = |r: u32| device::shadow_offset(r).map_or(0, |o| mem.read_u32(dev + o));
    let mut v = VERIFY.lock().unwrap();
    let v = v.get_or_insert_with(Verify::default);
    if id == 0 {
        v.args.check_surfaces(&targets, depth.as_ref(), shadow);
    }
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
        for line in v.verifier.report().lines().chain(v.args.report().lines()) {
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
    while v.resolves.first_key_value().is_some_and(|(&s, _)| seq.wrapping_sub(s) as i32 > 0) {
        v.resolves.pop_first();
    }
    if let Some((args, dest)) = v.resolves.remove(&seq) {
        v.args.check_resolve(&args, dest.as_ref(), &read);
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
