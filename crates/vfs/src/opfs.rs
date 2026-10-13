//! Browser storage for the VFS via Emscripten WasmFS.
//!
//! WasmFS proxies the asynchronous OPFS API to a dedicated thread and exposes
//! it as ordinary synchronous POSIX files, so once a backend is mounted the VFS
//! uses `std::fs` exactly as on native. Requires linking with `-sWASMFS` and
//! must not run on the browser's main thread (the web host uses
//! `-sPROXY_TO_PTHREAD`).

use std::ffi::{CString, c_char};
use std::io;
use std::path::PathBuf;

#[repr(C)]
struct Backend {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn wasmfs_create_opfs_backend() -> *mut Backend;
    fn wasmfs_create_node_backend(root: *const c_char) -> *mut Backend;
    fn wasmfs_create_directory(path: *const c_char, mode: u32, backend: *mut Backend) -> i32;
}

fn mount(mount_point: &str, backend: *mut Backend) -> io::Result<PathBuf> {
    if backend.is_null() {
        return Err(io::Error::other("WasmFS backend creation failed"));
    }
    let c = CString::new(mount_point).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: valid NUL-terminated path and a live backend handle.
    let rc = unsafe { wasmfs_create_directory(c.as_ptr(), 0o777, backend) };
    if rc != 0 {
        return Err(io::Error::from_raw_os_error(-rc));
    }
    Ok(PathBuf::from(mount_point))
}

/// Mounts the origin-private file system root at `mount_point` (e.g. `/opfs`).
pub fn mount_opfs(mount_point: &str) -> io::Result<PathBuf> {
    // SAFETY: plain constructor call; WasmFS owns the backend.
    mount(mount_point, unsafe { wasmfs_create_opfs_backend() })
}

/// Mounts a host directory when running under Node.js (headless testing of
/// the web build).
pub fn mount_node_dir(mount_point: &str, host_root: &str) -> io::Result<PathBuf> {
    let root = CString::new(host_root).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: valid NUL-terminated path; WasmFS owns the backend.
    mount(mount_point, unsafe { wasmfs_create_node_backend(root.as_ptr()) })
}
