//! Declarations outside the common webgpu.h subset: wgpu-native's `wgpu.h` and native surface
//! sources natively, emdawnwebgpu's canvas source and Emscripten runtime calls on the web.

use crate::ffi::WGPUChainedStruct;
#[cfg(target_os = "emscripten")]
use crate::ffi::WGPUStringView;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUSurfaceSourceXlibWindow {
    pub chain: WGPUChainedStruct,
    pub display: *mut core::ffi::c_void,
    pub window: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUSurfaceSourceWaylandSurface {
    pub chain: WGPUChainedStruct,
    pub display: *mut core::ffi::c_void,
    pub surface: *mut core::ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUSurfaceSourceXCBWindow {
    pub chain: WGPUChainedStruct,
    pub connection: *mut core::ffi::c_void,
    pub window: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUSurfaceSourceWindowsHWND {
    pub chain: WGPUChainedStruct,
    pub hinstance: *mut core::ffi::c_void,
    pub hwnd: *mut core::ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUSurfaceSourceMetalLayer {
    pub chain: WGPUChainedStruct,
    pub layer: *mut core::ffi::c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUSurfaceSourceAndroidNativeWindow {
    pub chain: WGPUChainedStruct,
    pub window: *mut core::ffi::c_void,
}

/// `WGPUSType` values of the native surface sources (webgpu.h, absent from emdawnwebgpu).
pub mod stype {
    use crate::ffi::WGPUSType;
    pub const SURFACE_SOURCE_METAL_LAYER: WGPUSType = WGPUSType(0x4);
    pub const SURFACE_SOURCE_WINDOWS_HWND: WGPUSType = WGPUSType(0x5);
    pub const SURFACE_SOURCE_XLIB_WINDOW: WGPUSType = WGPUSType(0x6);
    pub const SURFACE_SOURCE_WAYLAND_SURFACE: WGPUSType = WGPUSType(0x7);
    pub const SURFACE_SOURCE_ANDROID_NATIVE_WINDOW: WGPUSType = WGPUSType(0x8);
    pub const SURFACE_SOURCE_XCB_WINDOW: WGPUSType = WGPUSType(0x9);
    /// emdawnwebgpu's `WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector`.
    #[cfg(target_os = "emscripten")]
    pub const EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR: WGPUSType = WGPUSType(0x0004_0000);
}

#[cfg(not(target_os = "emscripten"))]
pub mod native {
    use crate::ffi::{WGPUBool, WGPUDevice, WGPUStringView};

    /// `WGPULogLevel` from wgpu.h.
    pub const LOG_LEVEL_OFF: u32 = 0;
    pub const LOG_LEVEL_ERROR: u32 = 1;
    pub const LOG_LEVEL_WARN: u32 = 2;
    pub const LOG_LEVEL_INFO: u32 = 3;
    pub const LOG_LEVEL_DEBUG: u32 = 4;
    pub const LOG_LEVEL_TRACE: u32 = 5;

    pub type WGPULogCallback =
        Option<unsafe extern "C" fn(level: u32, message: WGPUStringView, userdata: *mut core::ffi::c_void)>;

    unsafe extern "C" {
        pub fn wgpuDevicePoll(device: WGPUDevice, wait: WGPUBool, submission_index: *const u64) -> WGPUBool;
        pub fn wgpuSetLogCallback(callback: WGPULogCallback, userdata: *mut core::ffi::c_void);
        pub fn wgpuSetLogLevel(level: u32);
        pub fn wgpuGetVersion() -> u32;
    }
}

#[cfg(target_os = "emscripten")]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct WGPUEmscriptenSurfaceSourceCanvasHTMLSelector {
    pub chain: WGPUChainedStruct,
    pub selector: WGPUStringView,
}

#[cfg(target_os = "emscripten")]
pub mod emscripten {
    use core::ffi::{c_char, c_int, c_void};

    /// Storage for a `pthread_attr_t` (musl's is 36 bytes plus Emscripten's canvas list pointer
    /// on wasm32); oversized and aligned so its exact layout does not matter here.
    #[repr(C, align(8))]
    pub struct PthreadAttr(pub [u8; 128]);

    pub type PthreadT = usize;

    unsafe extern "C" {
        pub fn pthread_attr_init(attr: *mut PthreadAttr) -> c_int;
        pub fn pthread_attr_destroy(attr: *mut PthreadAttr) -> c_int;
        pub fn pthread_attr_setstacksize(attr: *mut PthreadAttr, size: usize) -> c_int;
        pub fn pthread_create(
            thread: *mut PthreadT,
            attr: *const PthreadAttr,
            start: extern "C" fn(*mut c_void) -> *mut c_void,
            arg: *mut c_void,
        ) -> c_int;
        pub fn pthread_detach(thread: PthreadT) -> c_int;
        pub fn emscripten_pthread_attr_settransferredcanvases(attr: *mut PthreadAttr, canvases: *const c_char)
        -> c_int;
        pub fn emscripten_set_main_loop_arg(
            func: extern "C" fn(*mut c_void),
            arg: *mut c_void,
            fps: c_int,
            simulate_infinite_loop: bool,
        );
        pub fn emscripten_cancel_main_loop();
        /// web/event_loop.js
        pub fn gpu_webgpu_schedule(func: extern "C" fn(*mut c_void), arg: *mut c_void, delay_ms: c_int);
        /// Like `gpu_webgpu_schedule(func, arg, 0)` once the 32-bit word at `addr` no longer
        /// holds `value` (`Atomics.waitAsync`), or after `timeout_ms`.
        pub fn gpu_webgpu_schedule_on_change(
            func: extern "C" fn(*mut c_void),
            arg: *mut c_void,
            addr: *const u32,
            value: u32,
            timeout_ms: c_int,
        );
        pub fn gpu_webgpu_loop_keepalive(start: bool);
        pub fn emscripten_get_now() -> f64;
    }
}
