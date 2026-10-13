//! Guest clocks.
//!
//! The PPC timebase (mftb) and KeQueryPerformanceFrequency share one 50 MHz
//! counter so QueryPerformanceCounter/Frequency pairs in guest code agree.
//! System time is Windows FILETIME (100 ns since 1601-01-01 UTC).

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const TIMEBASE_FREQUENCY: u64 = 50_000_000;

/// 100 ns intervals between 1601-01-01 and 1970-01-01.
const FILETIME_UNIX_EPOCH: u64 = 116_444_736_000_000_000;

fn boot() -> Instant {
    static BOOT: OnceLock<Instant> = OnceLock::new();
    *BOOT.get_or_init(Instant::now)
}

/// Starts the clocks; later calls are no-ops.
pub fn init() {
    boot();
}

/// 50 MHz ticks since boot.
pub fn timebase() -> u64 {
    let ns = boot().elapsed().as_nanos();
    (ns * TIMEBASE_FREQUENCY as u128 / 1_000_000_000) as u64
}

/// 100 ns units since boot (KeQueryInterruptTime, KeTimeStampBundle).
pub fn interrupt_time() -> u64 {
    (boot().elapsed().as_nanos() / 100) as u64
}

/// Milliseconds since boot (KeTimeStampBundle tick count).
pub fn tick_count() -> u32 {
    boot().elapsed().as_millis() as u32
}

/// Current wall-clock time as FILETIME.
pub fn system_time() -> u64 {
    let since_unix = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    FILETIME_UNIX_EPOCH + (since_unix.as_nanos() / 100) as u64
}

pub fn filetime_from_system_time(t: SystemTime) -> u64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => FILETIME_UNIX_EPOCH + (d.as_nanos() / 100) as u64,
        Err(e) => FILETIME_UNIX_EPOCH.saturating_sub((e.duration().as_nanos() / 100) as u64),
    }
}

/// Read by recompiled `mftb` (see cpp/ppc_context.h).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_guest_timebase() -> u64 {
    timebase()
}
