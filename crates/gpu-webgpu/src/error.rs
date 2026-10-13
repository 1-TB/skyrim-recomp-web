use std::fmt;

use crate::ffi;

pub type Result<T> = std::result::Result<T, Error>;

/// A GPU error reported by the implementation (error scope or uncaptured-error callback).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuError {
    pub ty: ffi::WGPUErrorType,
    pub message: String,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.ty {
            ffi::WGPUErrorType::Validation => "validation",
            ffi::WGPUErrorType::OutOfMemory => "out of memory",
            ffi::WGPUErrorType::Internal => "internal",
            _ => "unknown",
        };
        write!(f, "{kind} error: {}", self.message)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No WebGPU instance could be created (e.g. `navigator.gpu` missing).
    NoInstance,
    RequestAdapter(String),
    RequestDevice(String),
    Map(String),
    ErrorScope(String),
    WorkDone(String),
    Surface(String),
    /// `wgpuSurfaceGetCurrentTexture` did not produce a texture (reconfigure on Outdated/Lost).
    SurfaceStatus(ffi::WGPUSurfaceGetCurrentTextureStatus),
    /// The operation does not exist on this platform (e.g. an X11 surface in the browser).
    Unsupported(&'static str),
    /// Argument validation done on the Rust side before reaching the implementation.
    Invalid(String),
    Gpu(GpuError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoInstance => write!(f, "WebGPU is not available"),
            Error::RequestAdapter(m) => write!(f, "requesting an adapter failed: {m}"),
            Error::RequestDevice(m) => write!(f, "requesting a device failed: {m}"),
            Error::Map(m) => write!(f, "mapping a buffer failed: {m}"),
            Error::ErrorScope(m) => write!(f, "popping an error scope failed: {m}"),
            Error::WorkDone(m) => write!(f, "waiting for queue work failed: {m}"),
            Error::Surface(m) => write!(f, "surface: {m}"),
            Error::SurfaceStatus(s) => write!(f, "surface texture unavailable (status {})", s.0),
            Error::Unsupported(m) => write!(f, "unsupported on this platform: {m}"),
            Error::Invalid(m) => write!(f, "invalid argument: {m}"),
            Error::Gpu(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<GpuError> for Error {
    fn from(e: GpuError) -> Self {
        Error::Gpu(e)
    }
}
