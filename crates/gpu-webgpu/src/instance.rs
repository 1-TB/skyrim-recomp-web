use std::ffi::c_void;
use std::ptr;

use crate::device::{self, Device, DeviceDescriptor};
use crate::handle::{handle, string, wbool};
use crate::pending::{Pending, complete};
use crate::{Error, FeatureName, Limits, PowerPreference, Result, Surface, ffi};

handle!(
    /// Entry point: adapters, surfaces and event processing.
    Instance,
    WGPUInstance,
    wgpuInstanceAddRef,
    wgpuInstanceRelease
);

handle!(
    /// A physical GPU (or the browser's choice of one).
    Adapter,
    WGPUAdapter,
    wgpuAdapterAddRef,
    wgpuAdapterRelease
);

#[derive(Clone, Copy, Debug, Default)]
pub struct RequestAdapterOptions<'a> {
    pub power_preference: PowerPreference,
    pub force_fallback_adapter: bool,
    pub compatible_surface: Option<&'a Surface>,
}

#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub vendor: String,
    pub architecture: String,
    pub device: String,
    pub description: String,
    pub backend: ffi::WGPUBackendType,
    pub adapter_type: ffi::WGPUAdapterType,
    pub vendor_id: u32,
    pub device_id: u32,
}

impl Instance {
    pub fn new() -> Result<Self> {
        #[cfg(not(target_os = "emscripten"))]
        native_log::install();
        Self::from_created(unsafe { ffi::wgpuCreateInstance(ptr::null()) }, "instance").map_err(|_| Error::NoInstance)
    }

    /// Runs the callbacks of operations that have completed (and, natively, polls all devices).
    pub fn process_events(&self) {
        unsafe { ffi::wgpuInstanceProcessEvents(self.raw()) };
    }

    pub fn request_adapter(&self, options: &RequestAdapterOptions<'_>) -> Pending<Result<Adapter>> {
        unsafe extern "C" fn done(
            status: ffi::WGPURequestAdapterStatus,
            adapter: ffi::WGPUAdapter,
            message: ffi::WGPUStringView,
            userdata: *mut c_void,
            _: *mut c_void,
        ) {
            let result = if status == ffi::WGPURequestAdapterStatus::Success && !adapter.is_null() {
                Ok(unsafe { Adapter::from_raw(adapter) })
            } else {
                Err(Error::RequestAdapter(format!("status {}: {}", status.0, string(message))))
            };
            unsafe { complete(userdata, result) };
        }
        let opts = ffi::WGPURequestAdapterOptions {
            powerPreference: options.power_preference,
            forceFallbackAdapter: wbool(options.force_fallback_adapter),
            compatibleSurface: options.compatible_surface.map_or(ptr::null_mut(), |s| s.raw()),
            ..Default::default()
        };
        let (pending, userdata) = Pending::new();
        let info = ffi::WGPURequestAdapterCallbackInfo {
            mode: ffi::WGPUCallbackMode::AllowProcessEvents,
            callback: Some(done),
            userdata1: userdata,
            ..Default::default()
        };
        unsafe { ffi::wgpuInstanceRequestAdapter(self.raw(), &opts, info) };
        pending
    }

    /// Waits for `pending` by processing events in a loop. Native only: on the web the
    /// callbacks run from the browser event loop, which a blocked thread never reaches.
    #[cfg(not(target_os = "emscripten"))]
    pub fn block_on<T>(&self, pending: Pending<T>) -> T {
        loop {
            if let Some(v) = pending.try_take() {
                return v;
            }
            self.process_events();
            std::thread::yield_now();
        }
    }
}

impl Adapter {
    pub fn info(&self) -> AdapterInfo {
        let mut info = ffi::WGPUAdapterInfo::default();
        unsafe { ffi::wgpuAdapterGetInfo(self.raw(), &mut info) };
        let out = AdapterInfo {
            vendor: string(info.vendor),
            architecture: string(info.architecture),
            device: string(info.device),
            description: string(info.description),
            backend: info.backendType,
            adapter_type: info.adapterType,
            vendor_id: info.vendorID,
            device_id: info.deviceID,
        };
        unsafe { ffi::wgpuAdapterInfoFreeMembers(info) };
        out
    }

    pub fn limits(&self) -> Limits {
        let mut limits = Limits::default();
        unsafe { ffi::wgpuAdapterGetLimits(self.raw(), &mut limits) };
        limits
    }

    pub fn features(&self) -> Vec<FeatureName> {
        let mut f = ffi::WGPUSupportedFeatures { featureCount: 0, features: ptr::null() };
        unsafe { ffi::wgpuAdapterGetFeatures(self.raw(), &mut f) };
        let out = slice_or_empty(f.features, f.featureCount).to_vec();
        unsafe { ffi::wgpuSupportedFeaturesFreeMembers(f) };
        out
    }

    pub fn has_feature(&self, feature: FeatureName) -> bool {
        unsafe { ffi::wgpuAdapterHasFeature(self.raw(), feature) != 0 }
    }

    /// Requests a device whose uncaptured errors are logged and collected (see
    /// [`Device::take_errors`]).
    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Pending<Result<Device>> {
        device::request(self, desc)
    }
}

pub(crate) fn slice_or_empty<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
}

/// Routes wgpu-native's internal log into the `log` crate.
#[cfg(not(target_os = "emscripten"))]
mod native_log {
    use crate::ext::native::*;
    use crate::ffi::WGPUStringView;
    use crate::handle::string;

    unsafe extern "C" fn forward(level: u32, message: WGPUStringView, _: *mut core::ffi::c_void) {
        let level = match level {
            LOG_LEVEL_ERROR => log::Level::Error,
            LOG_LEVEL_WARN => log::Level::Warn,
            LOG_LEVEL_INFO => log::Level::Info,
            LOG_LEVEL_DEBUG => log::Level::Debug,
            _ => log::Level::Trace,
        };
        log::log!(target: "wgpu-native", level, "{}", string(message));
    }

    pub fn install() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let level = match log::max_level() {
                log::LevelFilter::Off => LOG_LEVEL_OFF,
                log::LevelFilter::Error => LOG_LEVEL_ERROR,
                log::LevelFilter::Warn => LOG_LEVEL_WARN,
                log::LevelFilter::Info => LOG_LEVEL_INFO,
                log::LevelFilter::Debug => LOG_LEVEL_DEBUG,
                log::LevelFilter::Trace => LOG_LEVEL_TRACE,
            };
            unsafe {
                wgpuSetLogCallback(Some(forward), core::ptr::null_mut());
                wgpuSetLogLevel(level);
            }
        });
    }
}
