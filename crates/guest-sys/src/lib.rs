//! The recompiled title, linked in from XENON_RECOMP_OUT (see build.rs).

use runtime_core::{GuestModule, PPCContext};

include!(concat!(env!("OUT_DIR"), "/imports.rs"));

unsafe extern "C" {
    fn skyrt_init_context(ctx: *mut PPCContext);
    fn skyrt_func_addrs(count: *mut usize) -> *const u32;
    fn skyrt_ppc_config(out: *mut u64);
}

unsafe extern "C-unwind" {
    /// The generated dispatch (build.rs): calls the function starting at `addr`.
    fn skyrt_call_func(addr: u32, ctx: *mut PPCContext, base: *mut u8);
}

// The generated C++ calls these; referencing them here keeps the linker from
// discarding the objects that define them.
#[used]
static KEEP_KERNEL_CALL: unsafe extern "C-unwind" fn(u32, u32, *mut PPCContext, *mut u8) =
    kernel_hle::dispatch::skyrt_kernel_call;
#[used]
static KEEP_TIMEBASE: extern "C" fn() -> u64 = runtime_core::timebase::skyrt_guest_timebase;
#[used]
static KEEP_MISSING_FUNCTION: extern "C" fn(u32) -> ! = runtime_core::guest::skyrt_missing_function;

/// Descriptor of the recompiled code for runtime-core.
pub fn module() -> GuestModule {
    let mut config = [0u64; 4];
    let mut count = 0usize;
    // SAFETY: the shim fills exactly four values; the dispatch returns its static array of
    // function addresses and their count.
    let functions = unsafe {
        skyrt_ppc_config(config.as_mut_ptr());
        let ptr = skyrt_func_addrs(&mut count);
        std::slice::from_raw_parts(ptr, count)
    };
    GuestModule {
        image_base: config[0] as u32,
        image_size: config[1] as u32,
        code_base: config[2] as u32,
        code_size: config[3] as u32,
        functions,
        call: skyrt_call_func,
        init_context: skyrt_init_context,
    }
}
