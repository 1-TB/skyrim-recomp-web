//! The game files streamed from the server the page came from (`cargo xtask serve --game-dir`)
//! instead of copied into OPFS: requests go through lib/http_files.js.

use std::ffi::{CStr, CString, c_char, c_void};
use std::io;
use std::sync::Arc;

use vfs::remote::{DEFAULT_CACHE_CHUNKS, RangeReader, RemoteTree};

unsafe extern "C" {
    fn skyrt_http_get_text(url: *const c_char) -> *mut c_char;
    fn skyrt_http_read(base: *const c_char, path: *const c_char, offset: f64, dst: *mut u8, len: u32) -> i32;
    fn free(ptr: *mut c_void);
}

/// Range reads of `<base>files/<path>`.
struct HttpRanges {
    files: CString,
}

impl RangeReader for HttpRanges {
    fn read(&self, path: &str, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let path = CString::new(path).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // Each request is capped; the tree asks again for the rest. Offsets fit an f64 exactly.
        let len = buf.len().min(MAX_REQUEST) as u32;
        // SAFETY: valid C strings; `buf` has at least `len` writable bytes.
        let n = unsafe { skyrt_http_read(self.files.as_ptr(), path.as_ptr(), offset as f64, buf.as_mut_ptr(), len) };
        if n < 0 {
            return Err(io::Error::other(format!("streaming {}: request failed", path.to_string_lossy())));
        }
        Ok(n as usize)
    }
}

/// Largest single range request.
const MAX_REQUEST: usize = 16 << 20;

/// Fetches `<base>manifest` and returns the tree it describes.
pub fn open(base: &str) -> Result<Arc<RemoteTree>, String> {
    let base = if base.ends_with('/') { base.to_string() } else { format!("{base}/") };
    let url = CString::new(format!("{base}manifest")).map_err(|e| e.to_string())?;
    // SAFETY: a valid C string; the result is a malloc'd C string (or null) that we free.
    let manifest = unsafe {
        let text = skyrt_http_get_text(url.as_ptr());
        if text.is_null() {
            return Err(format!("no game manifest at {base}manifest (is the server running with --game-dir?)"));
        }
        let s = CStr::from_ptr(text).to_string_lossy().into_owned();
        free(text.cast());
        s
    };
    let files = CString::new(format!("{base}files/")).map_err(|e| e.to_string())?;
    let tree = RemoteTree::from_manifest(&manifest, Arc::new(HttpRanges { files }), DEFAULT_CACHE_CHUNKS);
    if tree.lookup("default.xex").is_none() {
        return Err(format!("the streamed game at {base} has no default.xex"));
    }
    Ok(Arc::new(tree))
}
