mod common;

use std::thread;
use std::time::{Duration, Instant};

use common::{RECORDER, guest, kernel, recorded_with_arg1, scratch, timeout_ms};
use kernel_hle::Kernel;
use kernel_hle::dispatch::ModuleId;
use kernel_hle::status;
use runtime_core::PPCContext;

const NOTIFICATION: u32 = 0;
const SYNCHRONIZATION: u32 = 1;
/// DISPATCHER_HEADER.Inserted.
const INSERTED: u32 = 3;

/// KeSetTimerEx with the LARGE_INTEGER due time in one register, as the guest passes it.
fn set_timer(k: &'static Kernel, ctx: &mut PPCContext, timer: u32, due_100ns: i64, period_ms: i32, dpc: u32) -> u64 {
    ctx.set_gpr(3, timer as u64);
    ctx.set_gpr(4, due_100ns as u64);
    ctx.set_gpr(5, period_ms as u32 as u64);
    ctx.set_gpr(6, dpc as u64);
    k.dispatch(ModuleId::XBOXKRNL, 0x0A7, ctx);
    ctx.gpr(3)
}

fn wait(k: &'static Kernel, ctx: &mut PPCContext, obj: u32, timeout_at: u32, ms: i64) -> u32 {
    let t = timeout_ms(k, timeout_at, ms);
    k.call_export("KeWaitForSingleObject", ctx, &[obj, 0, 0, 0, t]) as u32
}

#[test]
fn one_shot_notification_timer_signals_at_its_due_time_and_stays_signaled() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let timer = mem;
    k.call_export("KeInitializeTimerEx", &mut ctx, &[timer, NOTIFICATION, 1]);
    assert_eq!(k.mem().read_u8(timer), 8, "TimerNotificationObject");

    let start = Instant::now();
    assert_eq!(set_timer(k, &mut ctx, timer, -30 * 10_000, 0, 0), 0, "was not armed");
    assert_eq!(k.mem().read_u8(timer + INSERTED), 1);
    assert_eq!(wait(k, &mut ctx, timer, mem + 0x100, 0), status::TIMEOUT, "not due yet");
    assert_eq!(wait(k, &mut ctx, timer, mem + 0x100, 5000), status::SUCCESS);
    assert!(start.elapsed() >= Duration::from_millis(25), "fired after {:?}", start.elapsed());
    assert_eq!(k.mem().read_u8(timer + INSERTED), 0, "one-shot timers leave the queue");
    assert_eq!(k.mem().read_i32(timer + 4), 1);
    // A notification timer stays signaled for every later wait...
    assert_eq!(wait(k, &mut ctx, timer, mem + 0x100, 0), status::SUCCESS);
    // ...until it is set again, which clears it.
    assert_eq!(set_timer(k, &mut ctx, timer, -10_000_000, 0, 0), 0);
    assert_eq!(k.mem().read_i32(timer + 4), 0);
    assert_eq!(wait(k, &mut ctx, timer, mem + 0x100, 0), status::TIMEOUT);
    assert_eq!(k.call_export("KeCancelTimer", &mut ctx, &[timer]), 1, "was armed");
    assert_eq!(k.call_export("KeCancelTimer", &mut ctx, &[timer]), 0);
    assert_eq!(k.mem().read_u8(timer + INSERTED), 0);
}

#[test]
fn periodic_synchronization_timer_releases_one_wait_per_expiry_and_runs_its_dpc() {
    const TAG: u32 = 0x71AE_0001;
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let (timer, dpc) = (mem, mem + 0x40);
    k.call_export("KeInitializeTimerEx", &mut ctx, &[timer, SYNCHRONIZATION, 1]);
    k.call_export("KeInitializeDpc", &mut ctx, &[dpc, RECORDER, TAG]);
    assert_eq!(set_timer(k, &mut ctx, timer, -10 * 10_000, 10, dpc), 0);
    for _ in 0..3 {
        assert_eq!(wait(k, &mut ctx, timer, mem + 0x100, 5000), status::SUCCESS);
    }
    assert_eq!(k.mem().read_u8(timer + INSERTED), 1, "periodic timers stay queued");
    assert_eq!(k.call_export("KeCancelTimer", &mut ctx, &[timer]), 1);
    // A DPC already running finishes; nothing fires after that.
    thread::sleep(Duration::from_millis(30));
    let calls = recorded_with_arg1(TAG);
    assert!(calls.len() >= 3, "{} DPC runs", calls.len());
    for c in &calls {
        assert_eq!(c.args[0], dpc);
        assert_eq!((c.irql, c.dpc_active, c.process_type_in_dpc), (2, 1, 1));
    }
    // The system arguments are the expiry's system time (low, high).
    let time = (calls[0].args[3] as u64) << 32 | calls[0].args[2] as u64;
    assert!(time > 0x01D0_0000_0000_0000, "system time {time:#X}");
    thread::sleep(Duration::from_millis(50));
    assert_eq!(recorded_with_arg1(TAG).len(), calls.len(), "cancelled timer fired");
}

#[test]
fn setting_an_armed_timer_replaces_its_expiry() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let timer = mem;
    k.call_export("KeInitializeTimerEx", &mut ctx, &[timer, NOTIFICATION, 1]);
    assert_eq!(set_timer(k, &mut ctx, timer, -100_000_000, 0, 0), 0);
    // Re-armed with an absolute due time in the past: due at once.
    assert_eq!(set_timer(k, &mut ctx, timer, 1, 0, 0), 1, "was armed");
    assert_eq!(wait(k, &mut ctx, timer, mem + 0x100, 5000), status::SUCCESS);
    assert_eq!(k.call_export("KeCancelTimer", &mut ctx, &[timer]), 0);
}

#[test]
fn a_timer_wakes_a_thread_already_waiting_on_it() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let timer = mem;
    k.call_export("KeInitializeTimerEx", &mut ctx, &[timer, SYNCHRONIZATION, 1]);
    let waiter = thread::spawn(move || {
        let mut ctx = guest(k);
        wait(k, &mut ctx, timer, mem + 0x200, 5000)
    });
    thread::sleep(Duration::from_millis(20));
    set_timer(k, &mut ctx, timer, -10 * 10_000, 0, 0);
    assert_eq!(waiter.join().unwrap(), status::SUCCESS);
    // The waiter consumed the synchronization timer's signal.
    assert_eq!(k.mem().read_i32(timer + 4), 0);
}
