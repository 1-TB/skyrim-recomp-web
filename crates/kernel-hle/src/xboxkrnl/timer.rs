//! Kernel timers (KTIMER): KeInitializeTimerEx, KeSetTimer(Ex), KeCancelTimer.
//!
//! An armed timer sits in a queue that a kernel thread services: at its due
//! time the timer becomes signaled (waking waiters as a notification or
//! synchronization event would) and its DPC, if any, runs on that thread with
//! the expiry's system time as the system arguments, as NT's
//! KiTimerExpiration delivers it. Periodic timers are re-armed one period
//! after each expiry. Xenia has no KeSetTimer; the semantics are NT's
//! (KeSetTimerEx clears the signal state and replaces any pending expiry,
//! KeCancelTimer leaves the signal state alone).
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::{Duration, Instant};

use runtime_core::PPCContext;
use runtime_core::layout::{dispatch_header as dh, object_type};
use runtime_core::thread::init_list_head;
use runtime_core::timebase;

use super::dpc::run_dpc;
use crate::Kernel;
use crate::dispatch::{Call, Export, ModuleId};
use crate::objects::{KObject, ObjectType, SyncState};
use crate::thread::deadline_from_timeout;

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x075, KeInitializeTimerEx),
    export!(K, 0x0A6, KeSetTimer),
    export!(K, 0x0A7, KeSetTimerEx),
    export!(K, 0x054, KeCancelTimer),
];

/// KTIMER layout (0x28 bytes, Xenia's X_KTIMER): DISPATCHER_HEADER, then these.
mod ktimer {
    /// Absolute system time of the next expiry.
    pub const DUE_TIME: u32 = 0x10;
    pub const DPC: u32 = 0x20;
    /// Milliseconds; 0 for a one-shot timer.
    pub const PERIOD: u32 = 0x24;
}

/// DISPATCHER_HEADER byte 2 holds a timer's process type.
const PROCESS_TYPE: u32 = dh::SIZE_FIELD;

struct Armed {
    obj: Arc<KObject>,
    due: Instant,
    period: Option<Duration>,
    dpc: u32,
}

/// The queue of armed timers, keyed by KTIMER address.
#[derive(Default)]
pub struct Timers {
    queue: Mutex<HashMap<u32, Armed>>,
    changed: Condvar,
    service: Once,
}

impl Kernel {
    /// Removes `ptr` from the queue; returns whether it was armed. Clears the
    /// header's Inserted flag.
    fn disarm_timer(&self, queue: &mut HashMap<u32, Armed>, ptr: u32) -> bool {
        let was = queue.remove(&ptr).is_some();
        self.mem().write_u8(ptr + dh::INSERTED, 0);
        was
    }

    /// KeSetTimerEx: (re)arms the timer at `ptr`; returns whether it was armed.
    pub(crate) fn set_timer(&'static self, ptr: u32, due_time: i64, period_ms: i32, dpc: u32) -> bool {
        let mem = self.mem();
        let obj = self.objects.native(mem, ptr);
        if obj.ty != ObjectType::Timer {
            log::warn!("KeSetTimer({ptr:#010X}): not a timer ({:?})", obj.ty);
            return false;
        }
        let due = deadline_from_timeout(Some(due_time)).unwrap_or_else(Instant::now);
        let absolute = if due_time < 0 {
            timebase::system_time().saturating_add(due_time.unsigned_abs())
        } else {
            due_time as u64
        };
        let period = (period_ms > 0).then(|| Duration::from_millis(period_ms as u64));
        let was = {
            let mut queue = self.timers.queue.lock().unwrap();
            let was = self.disarm_timer(&mut queue, ptr);
            self.objects.set_timer_signaled(mem, &obj, false);
            mem.write_u64(ptr + ktimer::DUE_TIME, absolute);
            mem.write_u32(ptr + ktimer::DPC, dpc);
            mem.write_u32(ptr + ktimer::PERIOD, period_ms.max(0) as u32);
            mem.write_u8(ptr + dh::INSERTED, 1);
            queue.insert(ptr, Armed { obj, due, period, dpc });
            was
        };
        self.timers.changed.notify_one();
        self.timers.service.call_once(|| self.spawn_kernel_thread("timers", timer_service));
        was
    }

    /// KeCancelTimer: returns whether the timer was armed.
    pub(crate) fn cancel_timer(&self, ptr: u32) -> bool {
        let mut queue = self.timers.queue.lock().unwrap();
        self.disarm_timer(&mut queue, ptr)
    }
}

/// Fires due timers: signals them under the queue lock (so a concurrent
/// KeSetTimer either precedes the expiry or clears it), then runs their DPCs
/// outside it, since a DPC routine may re-arm its own timer.
fn timer_service(k: &'static Kernel, ctx: &mut PPCContext) {
    let mem = k.mem();
    let mut queue = k.timers.queue.lock().unwrap();
    loop {
        let now = Instant::now();
        match queue.values().map(|a| a.due).min() {
            None => {
                queue = k.timers.changed.wait(queue).unwrap();
                continue;
            }
            Some(due) if due > now => {
                queue = k.timers.changed.wait_timeout(queue, due - now).unwrap().0;
                continue;
            }
            Some(_) => {}
        }
        let expired: Vec<u32> = queue.iter().filter(|(_, a)| a.due <= now).map(|(&p, _)| p).collect();
        let mut dpcs = Vec::new();
        for ptr in expired {
            let armed = queue.get_mut(&ptr).expect("expired timer is queued");
            k.objects.set_timer_signaled(mem, &armed.obj, true);
            if armed.dpc != 0 {
                dpcs.push((armed.dpc, mem.read_u8(ptr + PROCESS_TYPE)));
            }
            let period = armed.period;
            match period {
                Some(period) => {
                    // A stall (debugger, overloaded host) doesn't owe a burst of expiries.
                    armed.due = (armed.due + period).max(now);
                    let next = timebase::system_time() + period.as_nanos() as u64 / 100;
                    mem.write_u64(ptr + ktimer::DUE_TIME, next);
                }
                None => {
                    k.disarm_timer(&mut queue, ptr);
                }
            }
        }
        if dpcs.is_empty() {
            continue;
        }
        drop(queue);
        let time = timebase::system_time();
        for (dpc, process_type) in dpcs {
            run_dpc(k, ctx, dpc, time as u32, (time >> 32) as u32, Some(process_type));
        }
        queue = k.timers.queue.lock().unwrap();
    }
}

/// (KTIMER* Timer, TIMER_TYPE Type, u32 ProcessType). Type 0 is a
/// notification timer, 1 a synchronization timer.
fn KeInitializeTimerEx(c: &mut Call) -> u64 {
    let (ptr, ty, process_type) = (c.arg(0), c.arg(1), c.arg(2));
    // Re-initializing an armed timer would leave the queue pointing at reused memory.
    c.kernel.cancel_timer(ptr);
    let mem = c.mem();
    let notification = ty == 0;
    mem.write_u8(
        ptr + dh::TYPE,
        if notification { object_type::TIMER_NOTIFICATION } else { object_type::TIMER_SYNCHRONIZATION },
    );
    mem.write_u8(ptr + PROCESS_TYPE, process_type as u8);
    mem.write_u8(ptr + dh::INSERTED, 0);
    mem.write_i32(ptr + dh::SIGNAL_STATE, 0);
    init_list_head(mem, ptr + dh::WAIT_LIST_FLINK);
    mem.write_u64(ptr + ktimer::DUE_TIME, 0);
    mem.write_u32(ptr + ktimer::PERIOD, 0);
    c.kernel.objects.init_native(
        mem,
        ptr,
        ObjectType::Timer,
        SyncState::Event { manual_reset: notification, signaled: false },
    );
    0
}

/// (KTIMER* Timer, LARGE_INTEGER DueTime, KDPC* Dpc) -> BOOLEAN was armed.
/// The LARGE_INTEGER is passed by value in one register.
fn KeSetTimer(c: &mut Call) -> u64 {
    let (ptr, due, dpc) = (c.arg(0), c.arg64(1) as i64, c.arg(2));
    c.kernel.set_timer(ptr, due, 0, dpc) as u64
}

/// (KTIMER* Timer, LARGE_INTEGER DueTime, s32 Period (ms), KDPC* Dpc) -> BOOLEAN was armed.
fn KeSetTimerEx(c: &mut Call) -> u64 {
    let (ptr, due, period, dpc) = (c.arg(0), c.arg64(1) as i64, c.arg_i32(2), c.arg(3));
    c.kernel.set_timer(ptr, due, period, dpc) as u64
}

/// (KTIMER* Timer) -> BOOLEAN was armed.
fn KeCancelTimer(c: &mut Call) -> u64 {
    c.kernel.cancel_timer(c.arg(0)) as u64
}
