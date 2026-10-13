//! Text entry for the title's on-screen keyboard (XamShowKeyboardUI): a browser prompt on the
//! page's main thread (lib/text_input.js), answered while the XAM dispatch thread waits.

use std::ffi::{CStr, CString, c_char, c_void};

use kernel_hle::xam::KeyboardPrompt;

unsafe extern "C" {
    /// Returns a malloc'd UTF-8 string, or null if the user cancelled.
    fn skyrt_prompt(title: *const c_char, description: *const c_char, default_text: *const c_char, max: u32)
    -> *mut c_char;
    fn free(ptr: *mut c_void);
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("NULs removed")
}

fn prompt(p: &KeyboardPrompt) -> Option<String> {
    let (title, description, default_text) = (c_string(&p.title), c_string(&p.description), c_string(&p.default_text));
    // The buffer's last character is the terminator.
    let max = p.max_chars.saturating_sub(1);
    // SAFETY: valid C strings; the result is a malloc'd C string (or null) that we free.
    unsafe {
        let text = skyrt_prompt(title.as_ptr(), description.as_ptr(), default_text.as_ptr(), max);
        if text.is_null() {
            return None;
        }
        let s = CStr::from_ptr(text).to_string_lossy().into_owned();
        free(text.cast());
        Some(s)
    }
}

pub fn install() {
    kernel_hle::xam::set_text_input(prompt);
}
