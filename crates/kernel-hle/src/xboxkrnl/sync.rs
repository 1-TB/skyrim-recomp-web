//! Dispatcher objects, waits, critical sections, spinlocks and IRQL.
#![allow(non_snake_case)]

use std::sync::Arc;

use runtime_core::layout::{dispatch_header as dh, kpcr, kthread, object_type};
use runtime_core::thread::init_list_head;

use crate::dispatch::{Call, Export, ModuleId};
use crate::objects::{KObject, ObjectType, Payload, SyncState, WaitOutcome};
use crate::status;
use crate::strings::read_object_attributes;
use crate::thread::{deadline_from_timeout, read_timeout};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x070, KeInitializeEvent),
    export!(K, 0x09D, KeSetEvent),
    export!(K, 0x08F, KeResetEvent),
    export!(K, 0x07F, KePulseEvent),
    export!(K, 0x0D1, NtCreateEvent),
    export!(K, 0x0F6, NtSetEvent),
    export!(K, 0x0CE, NtClearEvent),
    export!(K, 0x0E2, NtPulseEvent),
    export!(K, 0x0E6, NtQueryEvent),
    export!(K, 0x074, KeInitializeSemaphore),
    export!(K, 0x088, KeReleaseSemaphore),
    export!(K, 0x0D5, NtCreateSemaphore),
    export!(K, 0x0F3, NtReleaseSemaphore),
    export!(K, 0x072, KeInitializeMutant),
    export!(K, 0x087, KeReleaseMutant),
    export!(K, 0x0D4, NtCreateMutant),
    export!(K, 0x0F2, NtReleaseMutant),
    export!(K, 0x0B0, KeWaitForSingleObject),
    export!(K, 0x0AF, KeWaitForMultipleObjects),
    export!(K, 0x0FD, NtWaitForSingleObjectEx),
    export!(K, 0x0FE, NtWaitForMultipleObjectsEx),
    export!(K, 0x0FB, NtSignalAndWaitForSingleObjectEx),
    export!(K, 0x05A, KeDelayExecutionThread),
    export!(K, 0x101, NtYieldExecution),
    export!(K, 0x12E, RtlInitializeCriticalSection),
    export!(K, 0x12F, RtlInitializeCriticalSectionAndSpinCount),
    export!(K, 0x125, RtlEnterCriticalSection),
    export!(K, 0x130, RtlLeaveCriticalSection),
    export!(K, 0x141, RtlTryEnterCriticalSection),
    export!(K, 0x0B1, KfAcquireSpinLock),
    export!(K, 0x0B4, KfReleaseSpinLock),
    export!(K, 0x04D, KeAcquireSpinLockAtRaisedIrql),
    export!(K, 0x089, KeReleaseSpinLockFromRaisedIrql),
    export!(K, 0x0AE, KeTryToAcquireSpinLockAtRaisedIrql),
    export!(K, 0x085, KeRaiseIrqlToDpcLevel),
    export!(K, 0x0B3, KfLowerIrql),
    export!(K, 0x0B2, KfRaiseIrql),
    export!(K, 0x05F, KeEnterCriticalRegion),
    export!(K, 0x07D, KeLeaveCriticalRegion),
];

const WAIT_ALL: u32 = 0;
const WAIT_ANY: u32 = 1;
const MAX_WAIT_OBJECTS: u32 = 64;

// --- Helpers ----------------------------------------------------------------

fn init_header(c: &Call, ptr: u32, ty: u8, size: u32, signal: i32) {
    let mem = c.mem();
    mem.write_u8(ptr + dh::TYPE, ty);
    mem.write_u8(ptr + dh::ABSOLUTE, 0);
    mem.write_u8(ptr + dh::SIZE_FIELD, (size / 4) as u8);
    mem.write_u8(ptr + dh::INSERTED, 0);
    mem.write_i32(ptr + dh::SIGNAL_STATE, signal);
    init_list_head(mem, ptr + dh::WAIT_LIST_FLINK);
}

/// Resolves a handle, answering NtCurrentThread for the caller.
pub(crate) fn handle_object(c: &Call, handle: u32) -> Option<Arc<KObject>> {
    let current = crate::thread::current();
    c.kernel.objects.lookup(handle, current.as_ref().map(|t| &t.object))
}

/// Looks up a named object for Nt*Create*; returns Some(status) when the call
/// is fully handled (existing object opened, or type mismatch).
fn open_existing(c: &Call, handle_ptr: u32, attrs: u32, ty: ObjectType) -> Option<u64> {
    let name = read_object_attributes(c.mem(), attrs)?.name?;
    let existing = c.kernel.objects.find_named(&name)?;
    if existing.ty != ty {
        return Some(status::OBJECT_TYPE_MISMATCH as u64);
    }
    if handle_ptr != 0 {
        let h = c.kernel.objects.insert_handle(&existing);
        c.mem().write_u32(handle_ptr, h);
    }
    Some(status::OBJECT_NAME_EXISTS as u64)
}

fn create_with_handle(c: &Call, handle_ptr: u32, attrs: u32, ty: ObjectType, sync: SyncState) -> u64 {
    let name = read_object_attributes(c.mem(), attrs).and_then(|a| a.name);
    match c.kernel.objects.create(&c.kernel.rt.space, ty, Some(sync), Payload::None, name) {
        Ok(obj) => {
            let h = c.kernel.objects.insert_handle(&obj);
            if handle_ptr != 0 {
                c.mem().write_u32(handle_ptr, h);
            }
            status::SUCCESS as u64
        }
        Err(s) => s as u64,
    }
}

/// Runs a wait for the calling thread, delivering APCs if it was alertable and
/// one arrived (the call then returns STATUS_USER_APC, as on NT).
pub(crate) fn do_wait(
    c: &mut Call,
    objects: &[Arc<KObject>],
    wait_all: bool,
    alertable: bool,
    timeout: Option<i64>,
) -> u32 {
    let tid = c.tid();
    if alertable && c.kernel.deliver_apcs(c) > 0 {
        return status::USER_APC;
    }
    let ids: Vec<_> = objects.iter().map(|o| o.id).collect();
    let deadline = deadline_from_timeout(timeout);
    let mem = c.kernel.mem();
    let k = c.thread().block.kthread;
    mem.write_u8(k + kthread::THREAD_STATE, kthread::STATE_WAITING);
    let started = std::time::Instant::now();
    let site = c.guest_backtrace(6);
    let outcome = c.kernel.objects.wait(mem, tid, &ids, wait_all, deadline, alertable, site.clone());
    mem.write_u8(k + kthread::THREAD_STATE, kthread::STATE_RUNNING);
    if log::log_enabled!(log::Level::Debug) {
        let ptrs: Vec<String> = objects.iter().map(|o| format!("{:#010X}", o.guest_ptr)).collect();
        log::debug!(
            "wait [{}] timeout {timeout:?} from {:08X?} -> {outcome:?} after {:?}",
            ptrs.join(" "),
            site,
            started.elapsed()
        );
    }
    match outcome {
        WaitOutcome::Satisfied(s) => s,
        WaitOutcome::Timeout => status::TIMEOUT,
        WaitOutcome::Apc => {
            c.kernel.deliver_apcs(c);
            status::USER_APC
        }
    }
}

// --- Events -----------------------------------------------------------------

fn KeInitializeEvent(c: &mut Call) -> u64 {
    let (ptr, ty, state) = (c.arg(0), c.arg(1), c.arg(2));
    init_header(c, ptr, ty as u8, 0x10, (state != 0) as i32);
    let sync = SyncState::Event { manual_reset: ty == object_type::EVENT_NOTIFICATION as u32, signaled: state != 0 };
    c.kernel.objects.init_native(c.mem(), ptr, ObjectType::Event, sync);
    0
}

fn KeSetEvent(c: &mut Call) -> u64 {
    let obj = c.kernel.objects.native(c.mem(), c.arg(0));
    c.kernel.objects.set_event(c.mem(), &obj).unwrap_or(0) as u32 as u64
}

fn KeResetEvent(c: &mut Call) -> u64 {
    let obj = c.kernel.objects.native(c.mem(), c.arg(0));
    c.kernel.objects.reset_event(c.mem(), &obj).unwrap_or(0) as u32 as u64
}

fn KePulseEvent(c: &mut Call) -> u64 {
    let obj = c.kernel.objects.native(c.mem(), c.arg(0));
    c.kernel.objects.pulse_event(c.mem(), &obj).unwrap_or(0) as u32 as u64
}

fn NtCreateEvent(c: &mut Call) -> u64 {
    let (handle_ptr, attrs, ty, state) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3));
    if let Some(s) = open_existing(c, handle_ptr, attrs, ObjectType::Event) {
        return s;
    }
    // 0 = NotificationEvent (manual reset), 1 = SynchronizationEvent.
    create_with_handle(
        c,
        handle_ptr,
        attrs,
        ObjectType::Event,
        SyncState::Event { manual_reset: ty == 0, signaled: state != 0 },
    )
}

fn event_from_handle(c: &Call, handle: u32) -> Result<Arc<KObject>, u64> {
    let obj = handle_object(c, handle).ok_or(status::INVALID_HANDLE as u64)?;
    if obj.ty != ObjectType::Event {
        return Err(status::OBJECT_TYPE_MISMATCH as u64);
    }
    Ok(obj)
}

fn NtSetEvent(c: &mut Call) -> u64 {
    let prev_ptr = c.arg(1);
    let obj = match event_from_handle(c, c.arg(0)) {
        Ok(o) => o,
        Err(s) => return s,
    };
    let prev = c.kernel.objects.set_event(c.mem(), &obj).unwrap_or(0);
    log::debug!("NtSetEvent({:#010X}) from {:08X?} -> {prev}", obj.guest_ptr, c.guest_backtrace(6));
    if prev_ptr != 0 {
        c.mem().write_i32(prev_ptr, prev);
    }
    status::SUCCESS as u64
}

fn NtClearEvent(c: &mut Call) -> u64 {
    match event_from_handle(c, c.arg(0)) {
        Ok(obj) => {
            log::debug!("NtClearEvent({:#010X}) from {:08X?}", obj.guest_ptr, c.guest_backtrace(6));
            c.kernel.objects.reset_event(c.mem(), &obj);
            status::SUCCESS as u64
        }
        Err(s) => s,
    }
}

fn NtPulseEvent(c: &mut Call) -> u64 {
    let prev_ptr = c.arg(1);
    let obj = match event_from_handle(c, c.arg(0)) {
        Ok(o) => o,
        Err(s) => return s,
    };
    let prev = c.kernel.objects.pulse_event(c.mem(), &obj).unwrap_or(0);
    if prev_ptr != 0 {
        c.mem().write_i32(prev_ptr, prev);
    }
    status::SUCCESS as u64
}

fn NtQueryEvent(c: &mut Call) -> u64 {
    let out = c.arg(1);
    let obj = match event_from_handle(c, c.arg(0)) {
        Ok(o) => o,
        Err(s) => return s,
    };
    if let Some(SyncState::Event { manual_reset, signaled }) = c.kernel.objects.sync_state(&obj) {
        c.mem().write_u32(out, if manual_reset { 0 } else { 1 });
        c.mem().write_u32(out + 4, signaled as u32);
    }
    status::SUCCESS as u64
}

// --- Semaphores -------------------------------------------------------------

fn KeInitializeSemaphore(c: &mut Call) -> u64 {
    let (ptr, count, limit) = (c.arg(0), c.arg_i32(1), c.arg_i32(2));
    init_header(c, ptr, object_type::SEMAPHORE, 0x14, count);
    c.mem().write_i32(ptr + 0x10, limit);
    c.kernel.objects.init_native(c.mem(), ptr, ObjectType::Semaphore, SyncState::Semaphore { count, limit });
    0
}

fn KeReleaseSemaphore(c: &mut Call) -> u64 {
    let (ptr, adjustment) = (c.arg(0), c.arg_i32(2));
    let obj = c.kernel.objects.native(c.mem(), ptr);
    match c.kernel.objects.release_semaphore(c.mem(), &obj, adjustment) {
        Ok(prev) => prev as u32 as u64,
        Err(s) => {
            // NT raises STATUS_SEMAPHORE_LIMIT_EXCEEDED here; there is no SEH to
            // raise into, so report and leave the count unchanged.
            log::warn!("KeReleaseSemaphore({ptr:#010X}, {adjustment}): {s:#010X}");
            match c.kernel.objects.sync_state(&obj) {
                Some(SyncState::Semaphore { count, .. }) => count as u32 as u64,
                _ => 0,
            }
        }
    }
}

fn NtCreateSemaphore(c: &mut Call) -> u64 {
    let (handle_ptr, attrs, count, limit) = (c.arg(0), c.arg(1), c.arg_i32(2), c.arg_i32(3));
    if let Some(s) = open_existing(c, handle_ptr, attrs, ObjectType::Semaphore) {
        return s;
    }
    if limit <= 0 || count < 0 || count > limit {
        return status::INVALID_PARAMETER as u64;
    }
    create_with_handle(c, handle_ptr, attrs, ObjectType::Semaphore, SyncState::Semaphore { count, limit })
}

fn NtReleaseSemaphore(c: &mut Call) -> u64 {
    let (handle, count, prev_ptr) = (c.arg(0), c.arg_i32(1), c.arg(2));
    let Some(obj) = handle_object(c, handle) else { return status::INVALID_HANDLE as u64 };
    if obj.ty != ObjectType::Semaphore {
        return status::OBJECT_TYPE_MISMATCH as u64;
    }
    let r = c.kernel.objects.release_semaphore(c.mem(), &obj, count);
    log::debug!("NtReleaseSemaphore({:#010X}, {count}) from {:#010X} -> {r:?}", obj.guest_ptr, c.ctx.lr as u32);
    let prev = match (&r, c.kernel.objects.sync_state(&obj)) {
        (Ok(p), _) => *p,
        (Err(_), Some(SyncState::Semaphore { count, .. })) => count,
        _ => 0,
    };
    if prev_ptr != 0 {
        c.mem().write_i32(prev_ptr, prev);
    }
    match r {
        Ok(_) => status::SUCCESS as u64,
        Err(s) => s as u64,
    }
}

// --- Mutants ----------------------------------------------------------------

fn KeInitializeMutant(c: &mut Call) -> u64 {
    let (ptr, initial_owner) = (c.arg(0), c.arg(1) != 0);
    init_header(c, ptr, object_type::MUTANT, 0x20, if initial_owner { 0 } else { 1 });
    init_list_head(c.mem(), ptr + 0x10);
    let owner_thread = if initial_owner { c.thread().block.kthread } else { 0 };
    c.mem().write_u32(ptr + 0x18, owner_thread);
    c.mem().write_u8(ptr + 0x1C, 0);
    let sync =
        SyncState::Mutant { owner: initial_owner.then(|| c.tid()), recursion: initial_owner as u32, abandoned: false };
    c.kernel.objects.init_native(c.mem(), ptr, ObjectType::Mutant, sync);
    0
}

fn KeReleaseMutant(c: &mut Call) -> u64 {
    let (ptr, abandon) = (c.arg(0), c.arg(2) != 0);
    let obj = c.kernel.objects.native(c.mem(), ptr);
    let tid = c.tid();
    match c.kernel.objects.release_mutant(c.mem(), &obj, tid, abandon) {
        Ok(prev) => {
            if matches!(c.kernel.objects.sync_state(&obj), Some(SyncState::Mutant { owner: None, .. })) {
                c.mem().write_u32(ptr + 0x18, 0);
            }
            prev as u32 as u64
        }
        Err(s) => {
            log::warn!("KeReleaseMutant({ptr:#010X}) by thread {tid:#X}: {s:#010X}");
            0
        }
    }
}

fn NtCreateMutant(c: &mut Call) -> u64 {
    let (handle_ptr, attrs, initial_owner) = (c.arg(0), c.arg(1), c.arg(2) != 0);
    if let Some(s) = open_existing(c, handle_ptr, attrs, ObjectType::Mutant) {
        return s;
    }
    let sync =
        SyncState::Mutant { owner: initial_owner.then(|| c.tid()), recursion: initial_owner as u32, abandoned: false };
    create_with_handle(c, handle_ptr, attrs, ObjectType::Mutant, sync)
}

fn NtReleaseMutant(c: &mut Call) -> u64 {
    let (handle, prev_ptr) = (c.arg(0), c.arg(1));
    let Some(obj) = handle_object(c, handle) else { return status::INVALID_HANDLE as u64 };
    if obj.ty != ObjectType::Mutant {
        return status::OBJECT_TYPE_MISMATCH as u64;
    }
    match c.kernel.objects.release_mutant(c.mem(), &obj, c.tid(), false) {
        Ok(prev) => {
            if prev_ptr != 0 {
                c.mem().write_i32(prev_ptr, prev);
            }
            status::SUCCESS as u64
        }
        Err(s) => s as u64,
    }
}

// --- Waits ------------------------------------------------------------------

fn KeWaitForSingleObject(c: &mut Call) -> u64 {
    let (ptr, alertable, timeout_ptr) = (c.arg(0), c.arg(3) != 0, c.arg(4));
    if ptr == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let obj = c.kernel.objects.native(c.mem(), ptr);
    let timeout = read_timeout(c.mem(), timeout_ptr);
    do_wait(c, &[obj], false, alertable, timeout) as u64
}

fn KeWaitForMultipleObjects(c: &mut Call) -> u64 {
    let (count, array, wait_type, alertable, timeout_ptr) = (c.arg(0), c.arg(1), c.arg(2), c.arg(5) != 0, c.arg(6));
    if count == 0 || count > MAX_WAIT_OBJECTS || wait_type > WAIT_ANY {
        return status::INVALID_PARAMETER as u64;
    }
    let objs: Vec<_> = (0..count).map(|i| c.kernel.objects.native(c.mem(), c.mem().read_u32(array + i * 4))).collect();
    let timeout = read_timeout(c.mem(), timeout_ptr);
    do_wait(c, &objs, wait_type == WAIT_ALL, alertable, timeout) as u64
}

fn NtWaitForSingleObjectEx(c: &mut Call) -> u64 {
    let (handle, alertable, timeout_ptr) = (c.arg(0), c.arg(2) != 0, c.arg(3));
    let Some(obj) = handle_object(c, handle) else { return status::INVALID_HANDLE as u64 };
    let timeout = read_timeout(c.mem(), timeout_ptr);
    do_wait(c, &[obj], false, alertable, timeout) as u64
}

fn NtWaitForMultipleObjectsEx(c: &mut Call) -> u64 {
    let (count, handles, wait_type, alertable, timeout_ptr) = (c.arg(0), c.arg(1), c.arg(2), c.arg(4) != 0, c.arg(5));
    if count == 0 || count > MAX_WAIT_OBJECTS || wait_type > WAIT_ANY {
        return status::INVALID_PARAMETER as u64;
    }
    let mut objs = Vec::with_capacity(count as usize);
    for i in 0..count {
        match handle_object(c, c.mem().read_u32(handles + i * 4)) {
            Some(o) => objs.push(o),
            None => return status::INVALID_HANDLE as u64,
        }
    }
    let timeout = read_timeout(c.mem(), timeout_ptr);
    do_wait(c, &objs, wait_type == WAIT_ALL, alertable, timeout) as u64
}

fn NtSignalAndWaitForSingleObjectEx(c: &mut Call) -> u64 {
    let (signal, wait, alertable, timeout_ptr) = (c.arg(0), c.arg(1), c.arg(3) != 0, c.arg(4));
    let (Some(sig), Some(obj)) = (handle_object(c, signal), handle_object(c, wait)) else {
        return status::INVALID_HANDLE as u64;
    };
    let mem = c.kernel.mem();
    let r = match sig.ty {
        ObjectType::Event => c.kernel.objects.set_event(mem, &sig).map(|_| ()).ok_or(status::INVALID_HANDLE),
        ObjectType::Semaphore => c.kernel.objects.release_semaphore(mem, &sig, 1).map(|_| ()),
        ObjectType::Mutant => c.kernel.objects.release_mutant(mem, &sig, c.tid(), false).map(|_| ()),
        _ => Err(status::OBJECT_TYPE_MISMATCH),
    };
    if let Err(s) = r {
        return s as u64;
    }
    let timeout = read_timeout(mem, timeout_ptr);
    do_wait(c, &[obj], false, alertable, timeout) as u64
}

fn KeDelayExecutionThread(c: &mut Call) -> u64 {
    let (alertable, interval_ptr) = (c.arg(1) != 0, c.arg(2));
    let interval = read_timeout(c.mem(), interval_ptr).unwrap_or(0);
    if interval == 0 && !alertable {
        std::thread::yield_now();
        return status::SUCCESS as u64;
    }
    match do_wait(c, &[], false, alertable, Some(interval)) {
        status::TIMEOUT => status::SUCCESS as u64,
        s => s as u64,
    }
}

fn NtYieldExecution(_c: &mut Call) -> u64 {
    std::thread::yield_now();
    status::SUCCESS as u64
}

// --- Critical sections ------------------------------------------------------
//
// RTL_CRITICAL_SECTION (28 bytes): DISPATCHER_HEADER (a synchronization event;
// header.absolute holds spin count / 256), lock_count @0x10 (-1 when free),
// recursion_count @0x14, owning_thread (KTHREAD*) @0x18.

const CS_LOCK_COUNT: u32 = 0x10;
const CS_RECURSION: u32 = 0x14;
const CS_OWNER: u32 = 0x18;

fn init_critical_section(c: &Call, cs: u32, spin_div_256: u8) {
    init_critical_section_at(c.kernel, cs, spin_div_256);
}

/// RTL_CRITICAL_SECTION initialization for kernel-owned locks (no caller needed).
pub(crate) fn init_critical_section_at(kernel: &crate::Kernel, cs: u32, spin_div_256: u8) {
    let mem = kernel.mem();
    mem.write_u8(cs + dh::TYPE, object_type::EVENT_SYNCHRONIZATION);
    mem.write_u8(cs + dh::ABSOLUTE, spin_div_256);
    mem.write_u8(cs + dh::SIZE_FIELD, 0);
    mem.write_u8(cs + dh::INSERTED, 0);
    mem.write_i32(cs + dh::SIGNAL_STATE, 0);
    init_list_head(mem, cs + dh::WAIT_LIST_FLINK);
    mem.write_i32(cs + CS_LOCK_COUNT, -1);
    mem.write_i32(cs + CS_RECURSION, 0);
    mem.write_u32(cs + CS_OWNER, 0);
    kernel.objects.init_native(mem, cs, ObjectType::Event, SyncState::Event { manual_reset: false, signaled: false });
}

/// Spin count as RtlInitializeCriticalSectionAndSpinCount stores it.
pub(crate) fn spin_count_field(spin: u32) -> u8 {
    (spin.saturating_add(255) >> 8).min(255) as u8
}

fn RtlInitializeCriticalSection(c: &mut Call) -> u64 {
    init_critical_section(c, c.arg(0), 0);
    0
}

fn RtlInitializeCriticalSectionAndSpinCount(c: &mut Call) -> u64 {
    let spin = c.arg(1);
    init_critical_section(c, c.arg(0), spin_count_field(spin));
    status::SUCCESS as u64
}

/// RtlEnterCriticalSection for kernel and test use.
pub fn enter_critical_section(c: &mut Call, cs: u32) {
    let mem = c.kernel.mem();
    let me = c.thread().block.kthread;
    if mem.read_u32(cs + CS_OWNER) == me {
        mem.atomic_add_i32(cs + CS_LOCK_COUNT, 1);
        mem.write_i32(cs + CS_RECURSION, mem.read_i32(cs + CS_RECURSION) + 1);
        return;
    }
    let mut spin = mem.read_u8(cs + dh::ABSOLUTE) as u32 * 256;
    while spin > 0 {
        if mem.compare_exchange_u32(cs + CS_LOCK_COUNT, u32::MAX, 0) {
            mem.write_u32(cs + CS_OWNER, me);
            mem.write_i32(cs + CS_RECURSION, 1);
            return;
        }
        std::hint::spin_loop();
        spin -= 1;
    }
    if mem.atomic_add_i32(cs + CS_LOCK_COUNT, 1) != 0 {
        // Contended: block on the embedded event until the owner hands off.
        let ev = c.kernel.objects.native(mem, cs);
        do_wait(c, &[ev], false, false, None);
    }
    mem.write_u32(cs + CS_OWNER, me);
    mem.write_i32(cs + CS_RECURSION, 1);
}

/// RtlLeaveCriticalSection for kernel and test use.
pub fn leave_critical_section(c: &mut Call, cs: u32) {
    let mem = c.kernel.mem();
    let recursion = mem.read_i32(cs + CS_RECURSION) - 1;
    mem.write_i32(cs + CS_RECURSION, recursion);
    if recursion > 0 {
        mem.atomic_add_i32(cs + CS_LOCK_COUNT, -1);
        return;
    }
    mem.write_u32(cs + CS_OWNER, 0);
    if mem.atomic_add_i32(cs + CS_LOCK_COUNT, -1) != -1 {
        // Waiters queued: wake exactly one.
        let ev = c.kernel.objects.native(mem, cs);
        c.kernel.objects.set_event(mem, &ev);
    }
}

fn RtlEnterCriticalSection(c: &mut Call) -> u64 {
    let cs = c.arg(0);
    if cs == 0 {
        log::error!("RtlEnterCriticalSection(NULL)");
        return 0;
    }
    enter_critical_section(c, cs);
    0
}

fn RtlLeaveCriticalSection(c: &mut Call) -> u64 {
    let cs = c.arg(0);
    if cs == 0 {
        log::error!("RtlLeaveCriticalSection(NULL)");
        return 0;
    }
    leave_critical_section(c, cs);
    0
}

fn RtlTryEnterCriticalSection(c: &mut Call) -> u64 {
    let cs = c.arg(0);
    let mem = c.kernel.mem();
    let me = c.thread().block.kthread;
    if mem.compare_exchange_u32(cs + CS_LOCK_COUNT, u32::MAX, 0) {
        mem.write_u32(cs + CS_OWNER, me);
        mem.write_i32(cs + CS_RECURSION, 1);
        return 1;
    }
    if mem.read_u32(cs + CS_OWNER) == me {
        mem.atomic_add_i32(cs + CS_LOCK_COUNT, 1);
        mem.write_i32(cs + CS_RECURSION, mem.read_i32(cs + CS_RECURSION) + 1);
        return 1;
    }
    0
}

// --- Spinlocks and IRQL ------------------------------------------------------
//
// A KSPINLOCK holds the owner's PCR address (0 when free).

const IRQL_DISPATCH: u8 = 2;

fn pcr(c: &Call) -> u32 {
    c.ctx.gpr32(13)
}

fn set_irql(c: &Call, irql: u8) -> u8 {
    let p = pcr(c);
    let old = c.mem().read_u8(p + kpcr::CURRENT_IRQL);
    c.mem().write_u8(p + kpcr::CURRENT_IRQL, irql);
    old
}

fn acquire_spinlock(c: &Call, lock: u32) {
    let me = pcr(c);
    let mem = c.mem();
    let mut spins = 0u32;
    while !mem.compare_exchange_u32(lock, 0, me) {
        spins += 1;
        if spins.is_multiple_of(64) {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

fn KfAcquireSpinLock(c: &mut Call) -> u64 {
    let old = set_irql(c, IRQL_DISPATCH);
    acquire_spinlock(c, c.arg(0));
    old as u64
}

fn KfReleaseSpinLock(c: &mut Call) -> u64 {
    c.mem().atomic_store_u32(c.arg(0), 0);
    set_irql(c, c.arg(1) as u8);
    0
}

fn KeAcquireSpinLockAtRaisedIrql(c: &mut Call) -> u64 {
    acquire_spinlock(c, c.arg(0));
    0
}

fn KeReleaseSpinLockFromRaisedIrql(c: &mut Call) -> u64 {
    c.mem().atomic_store_u32(c.arg(0), 0);
    0
}

fn KeTryToAcquireSpinLockAtRaisedIrql(c: &mut Call) -> u64 {
    c.mem().compare_exchange_u32(c.arg(0), 0, pcr(c)) as u64
}

fn KeRaiseIrqlToDpcLevel(c: &mut Call) -> u64 {
    set_irql(c, IRQL_DISPATCH) as u64
}

fn KfLowerIrql(c: &mut Call) -> u64 {
    set_irql(c, c.arg(0) as u8);
    0
}

fn KfRaiseIrql(c: &mut Call) -> u64 {
    set_irql(c, c.arg(0) as u8) as u64
}

/// KTHREAD.KernelApcDisable (signed counter).
const KTHREAD_APC_DISABLE: u32 = 0xB0;

fn KeEnterCriticalRegion(c: &mut Call) -> u64 {
    let k = c.thread().block.kthread;
    c.mem().write_i32(k + KTHREAD_APC_DISABLE, c.mem().read_i32(k + KTHREAD_APC_DISABLE) - 1);
    0
}

fn KeLeaveCriticalRegion(c: &mut Call) -> u64 {
    let k = c.thread().block.kthread;
    c.mem().write_i32(k + KTHREAD_APC_DISABLE, c.mem().read_i32(k + KTHREAD_APC_DISABLE) + 1);
    0
}
