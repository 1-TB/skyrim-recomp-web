//! Browser-side threading for the renderer.
//!
//! WebGPU objects live in the JS realm of the worker that created them, and both promise
//! callbacks (adapter/device requests, `mapAsync`, error scopes) and canvas presentation only
//! happen when that worker returns to its event loop. A renderer pthread therefore runs as a
//! frame callback ([`run_frame_loop`]) instead of a blocking loop.

use std::ffi::{CString, c_void};

use crate::ext::emscripten::*;
use crate::{Error, Result};

/// Starts a detached pthread that owns the page's canvas `selector` (an element id selector
/// such as `"#canvas"`) as an OffscreenCanvas, and runs `f` on it.
///
/// Works from any thread, including a PROXY_TO_PTHREAD `main`: Emscripten proxies the creation
/// to the browser thread, which calls `transferControlToOffscreen()` and hands the canvas over.
/// Needs `-sOFFSCREENCANVAS_SUPPORT=1` at link time. `f` typically creates the WebGPU objects
/// and calls [`run_frame_loop`]; the thread stays alive while that loop runs.
pub fn spawn_canvas_thread<F: FnOnce() + Send + 'static>(selector: &str, stack_size: usize, f: F) -> Result<()> {
    extern "C" fn start(arg: *mut c_void) -> *mut c_void {
        let f = unsafe { Box::from_raw(arg.cast::<Box<dyn FnOnce() + Send>>()) };
        f();
        std::ptr::null_mut()
    }
    let canvases = CString::new(selector).map_err(|_| Error::Invalid("selector contains NUL".into()))?;
    let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(f));
    let arg = Box::into_raw(boxed).cast::<c_void>();
    let mut attr = PthreadAttr([0; 128]);
    let mut thread: PthreadT = 0;
    let rc = unsafe {
        pthread_attr_init(&mut attr);
        pthread_attr_setstacksize(&mut attr, stack_size);
        emscripten_pthread_attr_settransferredcanvases(&mut attr, canvases.as_ptr());
        let rc = pthread_create(&mut thread, &attr, start, arg);
        pthread_attr_destroy(&mut attr);
        rc
    };
    if rc != 0 {
        drop(unsafe { Box::from_raw(arg.cast::<Box<dyn FnOnce() + Send>>()) });
        return Err(Error::Surface(format!("pthread_create with canvas {selector} failed: {rc}")));
    }
    unsafe { pthread_detach(thread) };
    Ok(())
}

/// Calls `frame` once per display frame (requestAnimationFrame of the calling worker) until it
/// returns `false`. Returns immediately; the calling pthread is kept alive by the loop, so its
/// start routine should return afterwards. One loop per thread.
pub fn run_frame_loop<F: FnMut() -> bool + 'static>(frame: F) {
    extern "C" fn tick(arg: *mut c_void) {
        let frame = unsafe { &mut *arg.cast::<Box<dyn FnMut() -> bool>>() };
        if !frame() {
            unsafe {
                emscripten_cancel_main_loop();
                drop(Box::from_raw(arg.cast::<Box<dyn FnMut() -> bool>>()));
            }
        }
    }
    let boxed: Box<Box<dyn FnMut() -> bool>> = Box::new(Box::new(frame));
    unsafe { emscripten_set_main_loop_arg(tick, Box::into_raw(boxed).cast(), 0, false) };
}

/// What [`run_event_loop`] does after a tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// Run again as soon as the event loop has handled pending events (GPU callbacks).
    Now,
    /// Run again after this many milliseconds.
    After(u32),
    /// Run again once the word at `addr` changes from `value` (another thread changed it and
    /// woke waiters on it with a futex wake), or after `timeout_ms`.
    OnChange {
        addr: *const std::sync::atomic::AtomicU32,
        value: u32,
        timeout_ms: u32,
    },
    Stop,
}

/// Calls `tick` repeatedly from the calling worker's event loop, returning to it between calls
/// so promise callbacks run, but without waiting for the display like [`run_frame_loop`]: a tick
/// that needs a GPU result (a mapped buffer, an error scope) can continue within a millisecond.
/// Returns immediately; the thread stays alive while the loop runs. Needs `web/event_loop.js`
/// linked with `--js-library`. A canvas presents what was rendered when the tick returns.
pub fn run_event_loop<F: FnMut() -> Next + 'static>(tick: F) {
    extern "C" fn run(arg: *mut c_void) {
        let tick = unsafe { &mut *arg.cast::<Box<dyn FnMut() -> Next>>() };
        match tick() {
            Next::Now => unsafe { gpu_webgpu_schedule(run, arg, 0) },
            Next::After(ms) => unsafe { gpu_webgpu_schedule(run, arg, ms.clamp(1, i32::MAX as u32) as i32) },
            Next::OnChange { addr, value, timeout_ms } => unsafe {
                gpu_webgpu_schedule_on_change(run, arg, addr.cast(), value, timeout_ms.min(i32::MAX as u32) as i32)
            },
            Next::Stop => unsafe {
                drop(Box::from_raw(arg.cast::<Box<dyn FnMut() -> Next>>()));
                gpu_webgpu_loop_keepalive(false);
            },
        }
    }
    let boxed: Box<Box<dyn FnMut() -> Next>> = Box::new(Box::new(tick));
    unsafe {
        gpu_webgpu_loop_keepalive(true);
        gpu_webgpu_schedule(run, Box::into_raw(boxed).cast(), 0);
    }
}

/// Milliseconds since the page started (`performance.now()`).
pub fn now_ms() -> f64 {
    unsafe { emscripten_get_now() }
}
