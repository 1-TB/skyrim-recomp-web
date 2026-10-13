use crate::ffi;

/// Defines a reference-counted wrapper around a webgpu.h object: `Clone` adds a reference,
/// `Drop` releases one. The marker keeps handles on their creating thread.
macro_rules! handle {
    ($(#[$m:meta])* $name:ident, $raw:ident, $add_ref:ident, $release:ident) => {
        $(#[$m])*
        pub struct $name {
            raw: crate::ffi::$raw,
            _thread: core::marker::PhantomData<*const ()>,
        }

        impl $name {
            /// Takes ownership of one reference. `raw` must be a live, non-null object.
            pub(crate) unsafe fn from_raw(raw: crate::ffi::$raw) -> Self {
                debug_assert!(!raw.is_null());
                Self { raw, _thread: core::marker::PhantomData }
            }

            /// Wraps a pointer returned by a `Create*` call, mapping null to an error.
            #[allow(dead_code)]
            pub(crate) fn from_created(raw: crate::ffi::$raw, what: &str) -> crate::Result<Self> {
                if raw.is_null() {
                    Err(crate::Error::Invalid(format!("{what} creation returned null")))
                } else {
                    Ok(unsafe { Self::from_raw(raw) })
                }
            }

            /// The underlying webgpu.h handle (still owned by `self`).
            pub fn raw(&self) -> crate::ffi::$raw {
                self.raw
            }
        }

        impl Clone for $name {
            fn clone(&self) -> Self {
                unsafe { crate::ffi::$add_ref(self.raw) };
                Self { raw: self.raw, _thread: core::marker::PhantomData }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { crate::ffi::$release(self.raw) };
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({:p})", stringify!($name), self.raw)
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.raw == other.raw
            }
        }

        impl Eq for $name {}
    };
}
pub(crate) use handle;

/// Borrows a Rust string as a webgpu.h string view (no terminator needed).
pub(crate) fn sv(s: &str) -> ffi::WGPUStringView {
    ffi::WGPUStringView { data: s.as_ptr().cast(), length: s.len() }
}

/// A label view: empty labels become the "null" view.
pub(crate) fn label(s: &str) -> ffi::WGPUStringView {
    if s.is_empty() { ffi::WGPUStringView::default() } else { sv(s) }
}

/// Copies a string view handed to a callback.
pub(crate) fn string(v: ffi::WGPUStringView) -> String {
    if v.data.is_null() {
        return String::new();
    }
    let len = if v.length == ffi::WGPU_STRLEN {
        unsafe { core::ffi::CStr::from_ptr(v.data) }.to_bytes().len()
    } else {
        v.length
    };
    String::from_utf8_lossy(unsafe { core::slice::from_raw_parts(v.data.cast::<u8>(), len) }).into_owned()
}

pub(crate) fn wbool(b: bool) -> ffi::WGPUBool {
    if b { ffi::WGPU_TRUE } else { ffi::WGPU_FALSE }
}
