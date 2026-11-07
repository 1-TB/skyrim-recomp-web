//! A line of text over the game (waiting for files, errors, the title exiting), set from any
//! thread and shown by the main thread, which owns UIKit.

use std::sync::Mutex;
use std::time::Duration;

/// The message to show (None: nothing) and a counter bumped on every change.
static STATUS: Mutex<(Option<String>, u64)> = Mutex::new((None, 0));

pub fn set(text: String) {
    let mut s = STATUS.lock().unwrap();
    *s = (Some(text), s.1 + 1);
}

pub fn clear() {
    let mut s = STATUS.lock().unwrap();
    *s = (None, s.1 + 1);
}

/// The message if it changed since `seen` (updated to the current version).
pub fn take_change(seen: &mut u64) -> Option<Option<String>> {
    let s = STATUS.lock().unwrap();
    if s.1 == *seen {
        return None;
    }
    *seen = s.1;
    Some(s.0.clone())
}

/// Removes the "Starting…" message once the title shows its first frame.
pub fn clear_when_presenting(kernel: &'static kernel_hle::Kernel) {
    let _ = std::thread::Builder::new().name("status".into()).spawn(move || {
        while kernel.gpu.swap_count() == 0 {
            std::thread::sleep(Duration::from_millis(250));
        }
        clear();
    });
}
