use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use crate::command::Queue;
use crate::handle::{label, string};
use crate::instance::slice_or_empty;
use crate::pending::{Pending, complete};
use crate::{Adapter, Buffer, Error, ErrorFilter, FeatureName, GpuError, Limits, Result, Sampler, TextureView, ffi};

/// Uncaptured errors kept for [`Device::take_errors`]; older ones are only logged.
const MAX_KEPT_ERRORS: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct DeviceDescriptor<'a> {
    pub label: &'a str,
    pub required_features: &'a [FeatureName],
    /// `None` takes the WebGPU defaults; use [`Adapter::limits`] to ask for everything.
    pub required_limits: Option<Limits>,
}

/// State reachable from the device's callbacks. It is leaked (one small allocation per device)
/// because wgpu-native may report errors for objects outliving the last `Device` handle and
/// does not promise a final device-lost callback that could free it.
struct Shared {
    errors: RefCell<Vec<GpuError>>,
    error_count: Cell<u64>,
    lost: RefCell<Option<String>>,
    scope_depth: Cell<u32>,
}

/// A logical device and its queue.
pub struct Device {
    raw: ffi::WGPUDevice,
    queue: Queue,
    shared: &'static Shared,
    _thread: PhantomData<*const ()>,
}

impl Clone for Device {
    fn clone(&self) -> Self {
        unsafe { ffi::wgpuDeviceAddRef(self.raw) };
        Self { raw: self.raw, queue: self.queue.clone(), shared: self.shared, _thread: PhantomData }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { ffi::wgpuDeviceRelease(self.raw) };
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({:p})", self.raw)
    }
}

unsafe extern "C" fn uncaptured_error(
    _device: *const ffi::WGPUDevice,
    ty: ffi::WGPUErrorType,
    message: ffi::WGPUStringView,
    userdata: *mut c_void,
    _: *mut c_void,
) {
    let shared = unsafe { &*userdata.cast::<Shared>() };
    let err = GpuError { ty, message: string(message) };
    log::error!("uncaptured WebGPU {err}");
    shared.error_count.set(shared.error_count.get() + 1);
    let mut errors = shared.errors.borrow_mut();
    if errors.len() < MAX_KEPT_ERRORS {
        errors.push(err);
    }
}

unsafe extern "C" fn device_lost(
    _device: *const ffi::WGPUDevice,
    reason: ffi::WGPUDeviceLostReason,
    message: ffi::WGPUStringView,
    userdata: *mut c_void,
    _: *mut c_void,
) {
    let shared = unsafe { &*userdata.cast::<Shared>() };
    let message = string(message);
    match reason {
        ffi::WGPUDeviceLostReason::Destroyed
        | ffi::WGPUDeviceLostReason::CallbackCancelled
        | ffi::WGPUDeviceLostReason::FailedCreation => log::debug!("WebGPU device lost ({}): {message}", reason.0),
        _ => log::error!("WebGPU device lost ({}): {message}", reason.0),
    }
    *shared.lost.borrow_mut() = Some(message);
}

pub(crate) fn request(adapter: &Adapter, desc: &DeviceDescriptor<'_>) -> Pending<Result<Device>> {
    struct Request {
        shared: &'static Shared,
        userdata: *mut c_void,
    }
    unsafe extern "C" fn done(
        status: ffi::WGPURequestDeviceStatus,
        device: ffi::WGPUDevice,
        message: ffi::WGPUStringView,
        userdata: *mut c_void,
        _: *mut c_void,
    ) {
        let req = unsafe { Box::from_raw(userdata.cast::<Request>()) };
        let result = if status == ffi::WGPURequestDeviceStatus::Success && !device.is_null() {
            let queue = unsafe { Queue::from_raw(ffi::wgpuDeviceGetQueue(device)) };
            Ok(Device { raw: device, queue, shared: req.shared, _thread: PhantomData })
        } else {
            Err(Error::RequestDevice(format!("status {}: {}", status.0, string(message))))
        };
        unsafe { complete(req.userdata, result) };
    }

    let shared: &'static Shared = Box::leak(Box::new(Shared {
        errors: RefCell::new(Vec::new()),
        error_count: Cell::new(0),
        lost: RefCell::new(None),
        scope_depth: Cell::new(0),
    }));
    let shared_ptr = ptr::from_ref(shared).cast_mut().cast::<c_void>();
    let limits = desc.required_limits.map(|l| Limits { nextInChain: ptr::null_mut(), ..l });
    let raw_desc = ffi::WGPUDeviceDescriptor {
        label: label(desc.label),
        requiredFeatureCount: desc.required_features.len(),
        requiredFeatures: desc.required_features.as_ptr(),
        requiredLimits: limits.as_ref().map_or(ptr::null(), ptr::from_ref),
        deviceLostCallbackInfo: ffi::WGPUDeviceLostCallbackInfo {
            mode: ffi::WGPUCallbackMode::AllowSpontaneous,
            callback: Some(device_lost),
            userdata1: shared_ptr,
            ..Default::default()
        },
        uncapturedErrorCallbackInfo: ffi::WGPUUncapturedErrorCallbackInfo {
            callback: Some(uncaptured_error),
            userdata1: shared_ptr,
            ..Default::default()
        },
        ..Default::default()
    };
    let (pending, userdata) = Pending::new();
    let req = Box::into_raw(Box::new(Request { shared, userdata })).cast::<c_void>();
    let info = ffi::WGPURequestDeviceCallbackInfo {
        mode: ffi::WGPUCallbackMode::AllowProcessEvents,
        callback: Some(done),
        userdata1: req,
        ..Default::default()
    };
    unsafe { ffi::wgpuAdapterRequestDevice(adapter.raw(), &raw_desc, info) };
    pending
}

/// What a bind group entry points at.
#[derive(Clone, Copy, Debug)]
pub enum BindingResource<'a> {
    /// `size: None` binds the rest of the buffer.
    Buffer {
        buffer: &'a Buffer,
        offset: u64,
        size: Option<u64>,
    },
    Sampler(&'a Sampler),
    TextureView(&'a TextureView),
}

#[derive(Clone, Copy, Debug)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

impl BindGroupEntry<'_> {
    pub(crate) fn to_ffi(self) -> ffi::WGPUBindGroupEntry {
        let mut e = ffi::WGPUBindGroupEntry { binding: self.binding, ..Default::default() };
        match self.resource {
            BindingResource::Buffer { buffer, offset, size } => {
                e.buffer = buffer.raw();
                e.offset = offset;
                e.size = size.unwrap_or(ffi::WGPU_WHOLE_SIZE);
            }
            BindingResource::Sampler(s) => e.sampler = s.raw(),
            BindingResource::TextureView(v) => e.textureView = v.raw(),
        }
        e
    }
}

impl Device {
    pub fn raw(&self) -> ffi::WGPUDevice {
        self.raw
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn limits(&self) -> Limits {
        let mut limits = Limits::default();
        unsafe { ffi::wgpuDeviceGetLimits(self.raw, &mut limits) };
        limits
    }

    pub fn features(&self) -> Vec<FeatureName> {
        let mut f = ffi::WGPUSupportedFeatures { featureCount: 0, features: ptr::null() };
        unsafe { ffi::wgpuDeviceGetFeatures(self.raw, &mut f) };
        let out = slice_or_empty(f.features, f.featureCount).to_vec();
        unsafe { ffi::wgpuSupportedFeaturesFreeMembers(f) };
        out
    }

    pub fn has_feature(&self, feature: FeatureName) -> bool {
        unsafe { ffi::wgpuDeviceHasFeature(self.raw, feature) != 0 }
    }

    pub fn push_error_scope(&self, filter: ErrorFilter) {
        self.shared.scope_depth.set(self.shared.scope_depth.get() + 1);
        unsafe { ffi::wgpuDevicePushErrorScope(self.raw, filter) };
    }

    /// Resolves to the first error captured by the innermost scope, if any.
    pub fn pop_error_scope(&self) -> Pending<Result<Option<GpuError>>> {
        unsafe extern "C" fn done(
            status: ffi::WGPUPopErrorScopeStatus,
            ty: ffi::WGPUErrorType,
            message: ffi::WGPUStringView,
            userdata: *mut c_void,
            _: *mut c_void,
        ) {
            let result = if status != ffi::WGPUPopErrorScopeStatus::Success {
                Err(Error::ErrorScope(format!("status {}: {}", status.0, string(message))))
            } else if ty == ffi::WGPUErrorType::NoError {
                Ok(None)
            } else {
                Ok(Some(GpuError { ty, message: string(message) }))
            };
            unsafe { complete(userdata, result) };
        }
        // wgpu-native aborts on popping an empty stack; report it like the web does.
        let depth = self.shared.scope_depth.get();
        if depth == 0 {
            return Pending::ready(Err(Error::ErrorScope("error scope stack is empty".into())));
        }
        self.shared.scope_depth.set(depth - 1);
        let (pending, userdata) = Pending::new();
        let info = ffi::WGPUPopErrorScopeCallbackInfo {
            mode: ffi::WGPUCallbackMode::AllowProcessEvents,
            callback: Some(done),
            userdata1: userdata,
            ..Default::default()
        };
        unsafe { ffi::wgpuDevicePopErrorScope(self.raw, info) };
        pending
    }

    /// Uncaptured errors reported since the last call (at most the first 64 are kept).
    pub fn take_errors(&self) -> Vec<GpuError> {
        std::mem::take(&mut *self.shared.errors.borrow_mut())
    }

    /// Total uncaptured errors reported over the device's lifetime.
    pub fn error_count(&self) -> u64 {
        self.shared.error_count.get()
    }

    /// The device-lost message once the device has been lost.
    pub fn lost_reason(&self) -> Option<String> {
        self.shared.lost.borrow().clone()
    }

    pub fn destroy(&self) {
        unsafe { ffi::wgpuDeviceDestroy(self.raw) };
    }

    /// Non-blocking progress natively (maps and work-done callbacks then run on the next
    /// [`crate::Instance::process_events`]); a no-op on the web, where the browser drives it.
    pub fn poll(&self) {
        #[cfg(not(target_os = "emscripten"))]
        unsafe {
            crate::ext::native::wgpuDevicePoll(self.raw, ffi::WGPU_FALSE, ptr::null());
        }
    }

    /// Waits for `pending` by blocking on queue progress. Native only (see
    /// [`crate::Instance::block_on`]).
    #[cfg(not(target_os = "emscripten"))]
    pub fn block_on<T>(&self, pending: Pending<T>) -> T {
        loop {
            if let Some(v) = pending.try_take() {
                return v;
            }
            unsafe { crate::ext::native::wgpuDevicePoll(self.raw, ffi::WGPU_TRUE, ptr::null()) };
            if pending.is_ready() {
                continue;
            }
            std::thread::yield_now();
        }
    }
}
