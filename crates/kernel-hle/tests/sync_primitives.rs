mod common;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use common::{guest, kernel, scratch, timeout_ms};
use kernel_hle::status;

const NOTIFICATION: u32 = 0;
const SYNCHRONIZATION: u32 = 1;

#[test]
fn auto_reset_event_wakes_one_waiter_and_resets() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let ev = mem;
    k.call_export("KeInitializeEvent", &mut ctx, &[ev, SYNCHRONIZATION, 0]);
    assert_eq!(k.mem().read_u8(ev), 1, "header type");

    let waiter = thread::spawn(move || {
        let mut ctx = guest(k);
        let t = timeout_ms(k, mem + 0x100, 5000);
        k.call_export("KeWaitForSingleObject", &mut ctx, &[ev, 0, 0, 0, t]) as u32
    });
    thread::sleep(Duration::from_millis(50));
    assert_eq!(k.call_export("KeSetEvent", &mut ctx, &[ev, 1, 0]), 0, "previous state");
    assert_eq!(waiter.join().unwrap(), status::SUCCESS);
    // The handoff consumed the signal; the mirrored header shows it.
    assert_eq!(k.mem().read_i32(ev + 4), 0);

    let t = timeout_ms(k, mem + 0x100, 0);
    assert_eq!(k.call_export("KeWaitForSingleObject", &mut ctx, &[ev, 0, 0, 0, t]) as u32, status::TIMEOUT);
}

#[test]
fn manual_reset_event_releases_every_waiter_and_stays_signaled() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let ev = mem;
    k.call_export("KeInitializeEvent", &mut ctx, &[ev, NOTIFICATION, 0]);
    let waiters: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let mut ctx = guest(k);
                let t = timeout_ms(k, mem + 0x100 + i * 8, 5000);
                k.call_export("KeWaitForSingleObject", &mut ctx, &[ev, 0, 0, 0, t]) as u32
            })
        })
        .collect();
    thread::sleep(Duration::from_millis(50));
    k.call_export("KeSetEvent", &mut ctx, &[ev, 0, 0]);
    for w in waiters {
        assert_eq!(w.join().unwrap(), status::SUCCESS);
    }
    assert_eq!(k.mem().read_i32(ev + 4), 1);
    assert_eq!(k.call_export("KeResetEvent", &mut ctx, &[ev]), 1);
    assert_eq!(k.mem().read_i32(ev + 4), 0);
}

#[test]
fn reset_waits_for_released_threads_to_move_on() {
    // A thread released by a notification event gets past its wait (to its
    // next kernel call) before the event can be reset for the next round, as
    // it would on hardware, however late the host schedules it.
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let (ev, marker) = (mem, mem + 0x200);
    k.call_export("KeInitializeEvent", &mut ctx, &[ev, NOTIFICATION, 0]);
    let waiter = thread::spawn(move || {
        let mut ctx = guest(k);
        let t = timeout_ms(k, mem + 0x100, 5000);
        let r = k.call_export("KeWaitForSingleObject", &mut ctx, &[ev, 0, 0, 0, t]) as u32;
        // A slow stretch of guest code before the next kernel call.
        thread::sleep(Duration::from_millis(20));
        k.mem().write_u32(marker, 1);
        k.call_export("KeQuerySystemTime", &mut ctx, &[0]);
        r
    });
    thread::sleep(Duration::from_millis(50));
    k.call_export("KeSetEvent", &mut ctx, &[ev, 0, 0]);
    assert_eq!(k.call_export("KeResetEvent", &mut ctx, &[ev]), 1);
    assert_eq!(k.mem().read_u32(marker), 1, "reset overtook the released thread");
    assert_eq!(waiter.join().unwrap(), status::SUCCESS);
}

#[test]
fn pulse_releases_current_waiters_without_staying_signaled() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let ev = mem;
    k.call_export("KeInitializeEvent", &mut ctx, &[ev, NOTIFICATION, 0]);
    let waiter = thread::spawn(move || {
        let mut ctx = guest(k);
        let t = timeout_ms(k, mem + 0x100, 5000);
        k.call_export("KeWaitForSingleObject", &mut ctx, &[ev, 0, 0, 0, t]) as u32
    });
    thread::sleep(Duration::from_millis(50));
    k.call_export("KePulseEvent", &mut ctx, &[ev, 0, 0]);
    assert_eq!(waiter.join().unwrap(), status::SUCCESS);
    assert_eq!(k.mem().read_i32(ev + 4), 0);
    let t = timeout_ms(k, mem + 0x100, 0);
    assert_eq!(k.call_export("KeWaitForSingleObject", &mut ctx, &[ev, 0, 0, 0, t]) as u32, status::TIMEOUT);
}

#[test]
fn relative_timeouts_expire() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    k.call_export("KeInitializeEvent", &mut ctx, &[mem, SYNCHRONIZATION, 0]);
    let t = timeout_ms(k, mem + 0x100, 30);
    let start = Instant::now();
    assert_eq!(k.call_export("KeWaitForSingleObject", &mut ctx, &[mem, 0, 0, 0, t]) as u32, status::TIMEOUT);
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(29), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");

    let start = Instant::now();
    let interval = timeout_ms(k, mem + 0x108, 20);
    assert_eq!(k.call_export("KeDelayExecutionThread", &mut ctx, &[0, 0, interval]) as u32, status::SUCCESS);
    assert!(start.elapsed() >= Duration::from_millis(19));
}

#[test]
fn semaphore_counts_and_limits() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let handle_ptr = mem;
    assert_eq!(k.call_export("NtCreateSemaphore", &mut ctx, &[handle_ptr, 0, 1, 2]) as u32, status::SUCCESS);
    let h = k.mem().read_u32(handle_ptr);
    let zero = timeout_ms(k, mem + 0x100, 0);

    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, zero]) as u32, status::SUCCESS);
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, zero]) as u32, status::TIMEOUT);

    let prev = mem + 0x10;
    assert_eq!(k.call_export("NtReleaseSemaphore", &mut ctx, &[h, 2, prev]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_i32(prev), 0);
    assert_eq!(k.call_export("NtReleaseSemaphore", &mut ctx, &[h, 1, prev]) as u32, status::SEMAPHORE_LIMIT_EXCEEDED);
    assert_eq!(k.mem().read_i32(prev), 2);

    assert_eq!(k.call_export("NtCreateSemaphore", &mut ctx, &[handle_ptr, 0, 3, 2]) as u32, status::INVALID_PARAMETER);

    // Kernel-level semaphore in title memory.
    let sem = mem + 0x200;
    k.call_export("KeInitializeSemaphore", &mut ctx, &[sem, 0, 10]);
    assert_eq!(k.mem().read_i32(sem + 0x10), 10);
    assert_eq!(k.call_export("KeReleaseSemaphore", &mut ctx, &[sem, 1, 3, 0]), 0);
    assert_eq!(k.mem().read_i32(sem + 4), 3);
}

#[test]
fn semaphore_wakes_exactly_as_many_waiters_as_released() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let sem = mem;
    k.call_export("KeInitializeSemaphore", &mut ctx, &[sem, 0, 100]);
    let waiters: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                let mut ctx = guest(k);
                let t = timeout_ms(k, mem + 0x100 + i * 8, 400);
                k.call_export("KeWaitForSingleObject", &mut ctx, &[sem, 0, 0, 0, t]) as u32
            })
        })
        .collect();
    thread::sleep(Duration::from_millis(50));
    k.call_export("KeReleaseSemaphore", &mut ctx, &[sem, 0, 3, 0]);
    let results: Vec<u32> = waiters.into_iter().map(|w| w.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|&&r| r == status::SUCCESS).count(), 3, "{results:?}");
    assert_eq!(results.iter().filter(|&&r| r == status::TIMEOUT).count(), 2, "{results:?}");
    assert_eq!(k.mem().read_i32(sem + 4), 0);
}

#[test]
fn mutant_is_recursive_and_owned() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    assert_eq!(k.call_export("NtCreateMutant", &mut ctx, &[mem, 0, 1]) as u32, status::SUCCESS);
    let h = k.mem().read_u32(mem);
    let zero = timeout_ms(k, mem + 0x100, 0);

    // Owner re-acquires recursively.
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, zero]) as u32, status::SUCCESS);

    let other = thread::spawn(move || {
        let mut ctx = guest(k);
        let zero = timeout_ms(k, mem + 0x108, 0);
        let busy = k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, zero]) as u32;
        let not_owner = k.call_export("NtReleaseMutant", &mut ctx, &[h, 0]) as u32;
        let t = timeout_ms(k, mem + 0x110, 5000);
        let acquired = k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h, 1, 0, t]) as u32;
        let released = k.call_export("NtReleaseMutant", &mut ctx, &[h, 0]) as u32;
        (busy, not_owner, acquired, released)
    });
    thread::sleep(Duration::from_millis(50));
    assert_eq!(k.call_export("NtReleaseMutant", &mut ctx, &[h, 0]) as u32, status::SUCCESS);
    thread::sleep(Duration::from_millis(20));
    // Still held once (recursion 2 -> 1); the other thread must keep waiting.
    assert!(!other.is_finished());
    assert_eq!(k.call_export("NtReleaseMutant", &mut ctx, &[h, 0]) as u32, status::SUCCESS);
    let (busy, not_owner, acquired, released) = other.join().unwrap();
    assert_eq!(busy, status::TIMEOUT);
    assert_eq!(not_owner, status::MUTANT_NOT_OWNED);
    assert_eq!(acquired, status::SUCCESS);
    assert_eq!(released, status::SUCCESS);
}

#[test]
fn wait_any_reports_index_and_wait_all_needs_every_object() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let (a, b, list) = (mem, mem + 0x20, mem + 0x40);
    k.call_export("KeInitializeEvent", &mut ctx, &[a, NOTIFICATION, 0]);
    k.call_export("KeInitializeEvent", &mut ctx, &[b, NOTIFICATION, 0]);
    k.mem().write_u32(list, a);
    k.mem().write_u32(list + 4, b);
    let zero = timeout_ms(k, mem + 0x100, 0);

    k.call_export("KeSetEvent", &mut ctx, &[b, 0, 0]);
    // WaitAny (1) -> STATUS_WAIT_0 + 1
    assert_eq!(k.call_export("KeWaitForMultipleObjects", &mut ctx, &[2, list, 1, 0, 0, 0, zero, 0]) as u32, 1);
    // WaitAll (0) fails while `a` is clear.
    assert_eq!(
        k.call_export("KeWaitForMultipleObjects", &mut ctx, &[2, list, 0, 0, 0, 0, zero, 0]) as u32,
        status::TIMEOUT
    );

    let setter = thread::spawn(move || {
        let mut ctx = guest(k);
        thread::sleep(Duration::from_millis(50));
        k.call_export("KeSetEvent", &mut ctx, &[a, 0, 0]);
    });
    let t = timeout_ms(k, mem + 0x108, 5000);
    assert_eq!(
        k.call_export("KeWaitForMultipleObjects", &mut ctx, &[2, list, 0, 0, 0, 0, t, 0]) as u32,
        status::SUCCESS
    );
    setter.join().unwrap();
}

#[test]
fn critical_section_provides_mutual_exclusion() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let (cs, counter) = (mem, mem + 0x40);
    k.call_export("RtlInitializeCriticalSectionAndSpinCount", &mut ctx, &[cs, 1000]);
    assert_eq!(k.mem().read_i32(cs + 0x10), -1);
    assert_eq!(k.mem().read_u8(cs + 1), 4, "spin count / 256, rounded up");

    const THREADS: usize = 6;
    const ITERS: u32 = 2_000;
    let start = Arc::new(Barrier::new(THREADS));
    let workers: Vec<_> = (0..THREADS)
        .map(|_| {
            let start = start.clone();
            thread::spawn(move || {
                let mut ctx = guest(k);
                start.wait();
                for _ in 0..ITERS {
                    k.call_export("RtlEnterCriticalSection", &mut ctx, &[cs]);
                    // Deliberately non-atomic read-modify-write.
                    let v = k.mem().read_u32(counter);
                    std::hint::spin_loop();
                    k.mem().write_u32(counter, v + 1);
                    k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(k.mem().read_u32(counter), THREADS as u32 * ITERS);
    assert_eq!(k.mem().read_i32(cs + 0x10), -1, "lock count back to free");
    assert_eq!(k.mem().read_u32(cs + 0x18), 0, "no owner");
}

#[test]
fn critical_section_recursion_and_try_enter() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let cs = mem;
    k.call_export("RtlInitializeCriticalSection", &mut ctx, &[cs]);
    k.call_export("RtlEnterCriticalSection", &mut ctx, &[cs]);
    assert_eq!(k.call_export("RtlTryEnterCriticalSection", &mut ctx, &[cs]), 1);
    assert_eq!(k.mem().read_i32(cs + 0x14), 2, "recursion count");
    let me = k.mem().read_u32(cs + 0x18);
    assert_ne!(me, 0);

    let other = thread::spawn(move || {
        let mut ctx = guest(k);
        k.call_export("RtlTryEnterCriticalSection", &mut ctx, &[cs])
    });
    assert_eq!(other.join().unwrap(), 0, "held by another thread");

    k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
    assert_eq!(k.mem().read_u32(cs + 0x18), me, "still owned after one leave");
    k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
    assert_eq!(k.mem().read_u32(cs + 0x18), 0);

    let other = thread::spawn(move || {
        let mut ctx = guest(k);
        let r = k.call_export("RtlTryEnterCriticalSection", &mut ctx, &[cs]);
        k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
        r
    });
    assert_eq!(other.join().unwrap(), 1);
}

#[test]
fn preinitialized_critical_section_in_data_works() {
    // Titles ship critical sections pre-initialized in .data and never call
    // RtlInitializeCriticalSection on them.
    let k = kernel();
    let mut ctx = guest(k);
    let cs = scratch(k, 0x1000);
    k.mem().write_u8(cs, 1);
    k.mem().write_i32(cs + 0x10, -1);
    k.call_export("RtlEnterCriticalSection", &mut ctx, &[cs]);
    let waiter = thread::spawn(move || {
        let mut ctx = guest(k);
        k.call_export("RtlEnterCriticalSection", &mut ctx, &[cs]);
        k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
    });
    thread::sleep(Duration::from_millis(50));
    assert!(!waiter.is_finished());
    k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
    waiter.join().unwrap();
}

#[test]
fn first_contention_on_fresh_critical_sections_never_loses_a_wakeup() {
    // Pre-initialized (never Rtl-initialized) critical sections whose first use is contended:
    // every thread looks up the embedded event for the first time at once. All of them must end
    // up with the same kernel object, or a waiter sleeps on one nobody signals.
    let k = kernel();
    const SECTIONS: u32 = 150;
    const THREADS: usize = 6;
    let base = scratch(k, SECTIONS * 0x20);
    for i in 0..SECTIONS {
        let cs = base + i * 0x20;
        k.mem().write_u8(cs, 1);
        k.mem().write_i32(cs + 0x10, -1);
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let start = Arc::new(Barrier::new(THREADS));
    for _ in 0..THREADS {
        let (start, done_tx) = (start.clone(), done_tx.clone());
        thread::spawn(move || {
            let mut ctx = guest(k);
            for i in 0..SECTIONS {
                let cs = base + i * 0x20;
                start.wait();
                for _ in 0..3 {
                    k.call_export("RtlEnterCriticalSection", &mut ctx, &[cs]);
                    std::hint::spin_loop();
                    k.call_export("RtlLeaveCriticalSection", &mut ctx, &[cs]);
                }
            }
            done_tx.send(()).unwrap();
        });
    }
    for _ in 0..THREADS {
        done_rx.recv_timeout(Duration::from_secs(30)).expect("a thread hung on a critical section");
    }
}

#[test]
fn named_events_are_shared() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    let (name_buf, ansi, attrs) = (mem + 0x100, mem + 0x200, mem + 0x300);
    k.mem().write_bytes(name_buf, b"SharedEvent");
    k.mem().write_u16(ansi, 11);
    k.mem().write_u16(ansi + 2, 12);
    k.mem().write_u32(ansi + 4, name_buf);
    k.mem().write_u32(attrs + 4, ansi);

    assert_eq!(k.call_export("NtCreateEvent", &mut ctx, &[mem, attrs, NOTIFICATION, 0]) as u32, status::SUCCESS);
    assert_eq!(
        k.call_export("NtCreateEvent", &mut ctx, &[mem + 4, attrs, NOTIFICATION, 0]) as u32,
        status::OBJECT_NAME_EXISTS
    );
    let (h1, h2) = (k.mem().read_u32(mem), k.mem().read_u32(mem + 4));
    assert_ne!(h1, h2);
    k.call_export("NtSetEvent", &mut ctx, &[h1, 0]);
    let zero = timeout_ms(k, mem + 0x400, 0);
    assert_eq!(k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[h2, 1, 0, zero]) as u32, status::SUCCESS);

    let q = mem + 0x500;
    assert_eq!(k.call_export("NtQueryEvent", &mut ctx, &[h2, q]) as u32, status::SUCCESS);
    assert_eq!((k.mem().read_u32(q), k.mem().read_u32(q + 4)), (0, 1));

    assert_eq!(k.call_export("NtClose", &mut ctx, &[h1]) as u32, status::SUCCESS);
    assert_eq!(k.call_export("NtClose", &mut ctx, &[h1]) as u32, status::INVALID_HANDLE);
    assert_eq!(k.call_export("NtSetEvent", &mut ctx, &[h2, 0]) as u32, status::SUCCESS, "second handle keeps it alive");
}

#[test]
fn signal_and_wait_is_one_call() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = scratch(k, 0x1000);
    k.call_export("NtCreateEvent", &mut ctx, &[mem, 0, SYNCHRONIZATION, 0]);
    k.call_export("NtCreateEvent", &mut ctx, &[mem + 4, 0, SYNCHRONIZATION, 0]);
    let (ping, pong) = (k.mem().read_u32(mem), k.mem().read_u32(mem + 4));
    let responder = thread::spawn(move || {
        let mut ctx = guest(k);
        let t = timeout_ms(k, mem + 0x100, 5000);
        let r = k.call_export("NtWaitForSingleObjectEx", &mut ctx, &[ping, 1, 0, t]) as u32;
        k.call_export("NtSetEvent", &mut ctx, &[pong, 0]);
        r
    });
    let t = timeout_ms(k, mem + 0x108, 5000);
    assert_eq!(
        k.call_export("NtSignalAndWaitForSingleObjectEx", &mut ctx, &[ping, pong, 1, 0, t]) as u32,
        status::SUCCESS
    );
    assert_eq!(responder.join().unwrap(), status::SUCCESS);
}

#[test]
fn tls_slots_are_per_thread() {
    let k = kernel();
    let mut ctx = guest(k);
    let slot = k.call_export("KeTlsAlloc", &mut ctx, &[]) as u32;
    assert_ne!(slot, u32::MAX);
    assert_eq!(k.call_export("KeTlsSetValue", &mut ctx, &[slot, 0xAABB_CCDD]), 1);
    assert_eq!(k.call_export("KeTlsGetValue", &mut ctx, &[slot]) as u32, 0xAABB_CCDD);
    let other = thread::spawn(move || {
        let mut ctx = guest(k);
        let before = k.call_export("KeTlsGetValue", &mut ctx, &[slot]) as u32;
        k.call_export("KeTlsSetValue", &mut ctx, &[slot, 7]);
        (before, k.call_export("KeTlsGetValue", &mut ctx, &[slot]) as u32)
    });
    assert_eq!(other.join().unwrap(), (0, 7));
    assert_eq!(k.call_export("KeTlsGetValue", &mut ctx, &[slot]) as u32, 0xAABB_CCDD);
    let slot2 = k.call_export("KeTlsAlloc", &mut ctx, &[]) as u32;
    assert_ne!(slot, slot2);
    assert_eq!(k.call_export("KeTlsFree", &mut ctx, &[slot]), 1);
    assert_eq!(k.call_export("KeTlsFree", &mut ctx, &[slot]), 0);
    assert_eq!(k.call_export("KeTlsAlloc", &mut ctx, &[]) as u32, slot, "freed slot is reused");
}

#[test]
fn spinlocks_record_owner_pcr_and_irql() {
    let k = kernel();
    let mut ctx = guest(k);
    let lock = scratch(k, 0x1000);
    let pcr = ctx.gpr32(13);
    let old = k.call_export("KfAcquireSpinLock", &mut ctx, &[lock]) as u32;
    assert_eq!(k.mem().read_u32(lock), pcr);
    assert_eq!(k.mem().read_u8(pcr + 0x18), 2, "IRQL raised to DISPATCH_LEVEL");
    assert_eq!(k.call_export("KeTryToAcquireSpinLockAtRaisedIrql", &mut ctx, &[lock]), 0);
    k.call_export("KfReleaseSpinLock", &mut ctx, &[lock, old]);
    assert_eq!(k.mem().read_u32(lock), 0);
    assert_eq!(k.mem().read_u8(pcr + 0x18), old as u8);
}
