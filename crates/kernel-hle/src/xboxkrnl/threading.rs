//! Thread creation/control, TLS, APCs and clocks.
#![allow(non_snake_case)]

use std::sync::atomic::Ordering;

use runtime_core::layout::{kpcr, kthread};
use runtime_core::thread::creation_flags;
use runtime_core::timebase;

use super::sync::handle_object;
use crate::dispatch::{Call, Export, ModuleId};
use crate::objects::{ObjectType, UserApc};
use crate::status;
use crate::thread::{KThread, exit_current_thread};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x00D, ExCreateThread),
    export!(K, 0x019, ExTerminateThread),
    export!(K, 0x0F5, NtResumeThread),
    export!(K, 0x092, KeResumeThread),
    export!(K, 0x097, KeSetAffinityThread),
    export!(K, 0x081, KeQueryBasePriorityThread),
    export!(K, 0x099, KeSetBasePriorityThread),
    export!(K, 0x0A3, KeSetPriorityThread),
    export!(K, 0x066, KeGetCurrentProcessType),
    export!(K, 0x1DF, KiApcNormalRoutineNop),
    export!(K, 0x09A, KeSetCurrentProcessType),
    export!(K, 0x0E3, NtQueueApcThread),
    export!(K, 0x152, KeTlsAlloc),
    export!(K, 0x153, KeTlsFree),
    export!(K, 0x154, KeTlsGetValue),
    export!(K, 0x155, KeTlsSetValue),
    export!(K, 0x083, KeQueryPerformanceFrequency),
    export!(K, 0x084, KeQuerySystemTime),
    export!(K, 0x082, KeQueryInterruptTime),
];

pub const TLS_OUT_OF_INDEXES: u32 = u32::MAX;

fn ExCreateThread(c: &mut Call) -> u64 {
    let (handle_ptr, stack_size, tid_ptr, xapi_startup, start, context, flags) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6));
    let t = match c.kernel.create_thread(stack_size, flags, xapi_startup, start, context) {
        Ok(t) => t,
        Err(s) => return s as u64,
    };
    log::info!(
        "ExCreateThread: thread {:#X} entry {start:#010X} ctx {context:#010X} xapi {xapi_startup:#010X} flags {flags:#010X}",
        t.tid()
    );
    if handle_ptr != 0 {
        let h = if flags & creation_flags::RETURN_KTHREAD_PTR != 0 {
            t.block.kthread
        } else {
            c.kernel.objects.insert_handle(&t.object)
        };
        c.mem().write_u32(handle_ptr, h);
    }
    if tid_ptr != 0 {
        c.mem().write_u32(tid_ptr, t.tid());
    }
    status::SUCCESS as u64
}

fn ExTerminateThread(c: &mut Call) -> u64 {
    exit_current_thread(c.arg(0))
}

fn thread_from_handle(c: &Call, handle: u32) -> Result<std::sync::Arc<KThread>, u64> {
    let obj = handle_object(c, handle).ok_or(status::INVALID_HANDLE as u64)?;
    if obj.ty != ObjectType::Thread {
        return Err(status::OBJECT_TYPE_MISMATCH as u64);
    }
    c.kernel.thread_for_object(&obj).ok_or(status::INVALID_HANDLE as u64)
}

fn thread_from_kthread(c: &Call, kthread_ptr: u32) -> Option<std::sync::Arc<KThread>> {
    let obj = c.kernel.objects.by_guest_ptr(kthread_ptr)?;
    c.kernel.thread_for_object(&obj)
}

fn NtResumeThread(c: &mut Call) -> u64 {
    let count_ptr = c.arg(1);
    match thread_from_handle(c, c.arg(0)) {
        Ok(t) => {
            let prev = t.resume();
            c.mem().write_u8(t.block.kthread + kthread::SUSPEND_COUNT, prev.saturating_sub(1) as u8);
            if count_ptr != 0 {
                c.mem().write_u32(count_ptr, prev);
            }
            status::SUCCESS as u64
        }
        Err(s) => s,
    }
}

fn KeResumeThread(c: &mut Call) -> u64 {
    match thread_from_kthread(c, c.arg(0)) {
        Some(t) => {
            let prev = t.resume();
            c.mem().write_u8(t.block.kthread + kthread::SUSPEND_COUNT, prev.saturating_sub(1) as u8);
            prev as u64
        }
        None => 0,
    }
}

/// Moves the thread to the hardware thread its affinity names, as Xenia's
/// KeSetAffinityThread / XThread::SetAffinity do: the PCR and KTHREAD record
/// it and the previous processor goes out through the pointer (the return
/// value is a status, unlike NT). Like Xenia's default
/// (`ignore_thread_affinities`), host threads are not pinned.
fn KeSetAffinityThread(c: &mut Call) -> u64 {
    let (k, affinity, prev_ptr) = (c.arg(0), c.arg(1), c.arg(2));
    let Some(cpu) = runtime_core::thread::hardware_thread_for(affinity) else {
        return status::INVALID_PARAMETER as u64;
    };
    let Some(t) = thread_from_kthread(c, k) else { return status::INVALID_HANDLE as u64 };
    let mem = c.mem();
    let prev = mem.read_u8(t.block.pcr + kpcr::PRCB_CURRENT_CPU) as u32;
    log::debug!("thread {:#X} affinity {affinity:#X} (hardware thread {prev} -> {cpu})", t.tid());
    if prev_ptr != 0 {
        mem.write_u32(prev_ptr, 1 << prev);
    }
    mem.write_u8(k + kthread::CURRENT_CPU, cpu as u8);
    mem.write_u8(t.block.pcr + kpcr::PRCB_CURRENT_CPU, cpu as u8);
    status::SUCCESS as u64
}

fn KeQueryBasePriorityThread(c: &mut Call) -> u64 {
    let k = c.arg(0);
    let process_base =
        c.mem().read_u8(c.kernel.rt.threads.process() + runtime_core::layout::kprocess::DEFAULT_THREAD_PRIORITY) as i32;
    match thread_from_kthread(c, k) {
        Some(t) => (t.priority.load(Ordering::Relaxed) - process_base) as u32 as u64,
        None => 0,
    }
}

fn KeSetBasePriorityThread(c: &mut Call) -> u64 {
    let (k, increment) = (c.arg(0), c.arg_i32(1));
    let process_base =
        c.mem().read_u8(c.kernel.rt.threads.process() + runtime_core::layout::kprocess::DEFAULT_THREAD_PRIORITY) as i32;
    let Some(t) = thread_from_kthread(c, k) else { return 0 };
    let new = process_base + increment;
    let prev = t.priority.swap(new, Ordering::Relaxed);
    log::debug!("thread {:#X} base priority {new} (was {prev})", t.tid());
    c.mem().write_u8(k + kthread::BASE_PRIORITY, new as u8);
    c.mem().write_u8(k + kthread::PRIORITY, new as u8);
    (prev - process_base) as u32 as u64
}

fn KeSetPriorityThread(c: &mut Call) -> u64 {
    let (k, priority) = (c.arg(0), c.arg_i32(1));
    let Some(t) = thread_from_kthread(c, k) else { return 0 };
    let prev = t.priority.swap(priority, Ordering::Relaxed);
    c.mem().write_u8(k + kthread::PRIORITY, priority as u8);
    prev as u32 as u64
}

fn KeGetCurrentProcessType(c: &mut Call) -> u64 {
    let pcr = c.ctx.gpr32(13);
    let mem = c.mem();
    if mem.read_u32(pcr + kpcr::PRCB_DPC_ACTIVE) != 0 {
        return mem.read_u8(pcr + kpcr::PROCESSTYPE_VALUE_IN_DPC) as u64;
    }
    let k = mem.read_u32(pcr + kpcr::PRCB_CURRENT_THREAD);
    mem.read_u8(k + kthread::PROCESS_TYPE) as u64
}

fn KeSetCurrentProcessType(c: &mut Call) -> u64 {
    let ty = c.arg(0) as u8;
    let pcr = c.ctx.gpr32(13);
    let mem = c.mem();
    if mem.read_u32(pcr + kpcr::PRCB_DPC_ACTIVE) != 0 {
        mem.write_u8(pcr + kpcr::PROCESSTYPE_VALUE_IN_DPC, ty);
    } else {
        let k = mem.read_u32(pcr + kpcr::PRCB_CURRENT_THREAD);
        mem.write_u8(k + kthread::PROCESS_TYPE, ty);
    }
    0
}

/// User APCs run the next time the target thread enters an alertable wait.
fn NtQueueApcThread(c: &mut Call) -> u64 {
    let (handle, routine, context, arg1, arg2) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    match thread_from_handle(c, handle) {
        Ok(t) => {
            c.kernel.objects.queue_apc(t.tid(), UserApc { routine, context, arg1, arg2 });
            status::SUCCESS as u64
        }
        Err(s) => s,
    }
}

// --- TLS --------------------------------------------------------------------

fn KeTlsAlloc(c: &mut Call) -> u64 {
    let slot = {
        let mut slots = c.kernel.tls_slots.lock().unwrap();
        match slots.iter().position(|used| !used) {
            Some(i) => {
                slots[i] = true;
                i as u32
            }
            None => return TLS_OUT_OF_INDEXES as u64,
        }
    };
    // A fresh slot reads as NULL on every thread.
    let mem = c.kernel.mem();
    for t in c.kernel.threads.lock().unwrap().values() {
        t.block.tls_set(mem, slot, 0);
    }
    slot as u64
}

fn KeTlsFree(c: &mut Call) -> u64 {
    let slot = c.arg(0);
    let mut slots = c.kernel.tls_slots.lock().unwrap();
    match slots.get_mut(slot as usize) {
        Some(used) if *used => {
            *used = false;
            1
        }
        _ => 0,
    }
}

fn KeTlsGetValue(c: &mut Call) -> u64 {
    c.thread().block.tls_get(c.kernel.mem(), c.arg(0)).unwrap_or(0) as u64
}

fn KeTlsSetValue(c: &mut Call) -> u64 {
    c.thread().block.tls_set(c.kernel.mem(), c.arg(0), c.arg(1)) as u64
}

// --- Clocks -----------------------------------------------------------------

fn KeQueryPerformanceFrequency(_c: &mut Call) -> u64 {
    timebase::TIMEBASE_FREQUENCY
}

fn KeQuerySystemTime(c: &mut Call) -> u64 {
    let ptr = c.arg(0);
    let now = timebase::system_time();
    c.mem().write_u64(c.kernel.vars.ke_time_stamp_bundle + 8, now);
    if ptr != 0 {
        c.mem().write_u64(ptr, now);
    }
    0
}

fn KeQueryInterruptTime(_c: &mut Call) -> u64 {
    timebase::interrupt_time()
}

/// Normal routine for kernel APCs that only need their kernel routine.
fn KiApcNormalRoutineNop(_c: &mut Call) -> u64 {
    0
}
