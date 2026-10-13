//! Fatal-signal reporting for debugging runs.
//!
//! When recompiled code faults, report the faulting address translated into the
//! guest address space, a backtrace (recompiled functions are named
//! `sub_<guest address>`, so frames read as guest functions), and the kernel
//! import summary. None of this is async-signal-safe; the process is about to
//! die anyway, and the report is worth more than the risk of a hang.

use std::sync::atomic::{AtomicUsize, Ordering};

static GUEST_BASE: AtomicUsize = AtomicUsize::new(0);

extern "C" fn on_fatal_signal(sig: libc::c_int, info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    let name = match sig {
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGBUS => "SIGBUS",
        libc::SIGILL => "SIGILL",
        libc::SIGFPE => "SIGFPE",
        libc::SIGTRAP => "SIGTRAP (guest trap instruction or debug break)",
        _ => "signal",
    };
    // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
    let addr = unsafe { (*info).si_addr() } as usize;
    let base = GUEST_BASE.load(Ordering::Relaxed);
    let thread = std::thread::current();
    eprintln!("\n=== {name} on thread {} ===", thread.name().unwrap_or("?"));
    if base != 0 && addr >= base && addr - base < (1usize << 32) {
        eprintln!("fault address {addr:#x} = guest {:#010X}", addr - base);
    } else {
        eprintln!("fault address {addr:#x} (outside guest memory)");
    }
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    for line in bt.lines().filter(|l| !l.trim_start().starts_with("at ")).take(60) {
        eprintln!("{line}");
    }
    if let Some(k) = KERNEL.get() {
        k.log_import_summary();
    }
    // SAFETY: terminate immediately without running destructors.
    unsafe { libc::_exit(128 + sig) };
}

static KERNEL: std::sync::OnceLock<&'static kernel_hle::Kernel> = std::sync::OnceLock::new();

/// Installs handlers for synchronous fault signals.
pub fn install() {
    for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE, libc::SIGTRAP] {
        // SAFETY: plain sigaction setup; the alternate stack below lets the
        // handler run after a guest stack overflow on the host stack.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_fatal_signal as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
    }
    install_alt_stack();
}

/// Per-thread alternate signal stack so stack overflows still get reported.
pub fn install_alt_stack() {
    const SIZE: usize = 256 * 1024;
    let stack = Box::leak(vec![0u8; SIZE].into_boxed_slice());
    let ss = libc::stack_t { ss_sp: stack.as_mut_ptr() as *mut libc::c_void, ss_flags: 0, ss_size: SIZE };
    // SAFETY: the leaked buffer outlives the thread.
    unsafe { libc::sigaltstack(&ss, std::ptr::null_mut()) };
}

pub fn set_kernel(k: &'static kernel_hle::Kernel) {
    GUEST_BASE.store(k.rt.base() as usize, Ordering::Relaxed);
    let _ = KERNEL.set(k);
}
