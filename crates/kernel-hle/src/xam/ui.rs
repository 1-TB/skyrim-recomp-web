//! System UI. There is no dashboard overlay: each dialog is answered
//! immediately the way Xenia's headless mode answers it, bracketed by the
//! "system UI shown/hidden" notifications titles pause on. The on-screen
//! keyboard asks the host for text (see [`set_text_input`]) if it can.
#![allow(non_snake_case)]

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::dispatch::{Call, Export, ModuleId};
use crate::input::MAX_USERS;
use crate::status::win32;
use crate::xam::content::DEVICE_HDD;
use crate::xam::notify::id;

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] =
    &[export!(X, 0x2CB, XamShowDeviceSelectorUI), export!(X, 0x2C1, XamShowKeyboardUI)];

/// What the title's on-screen keyboard asks for.
#[derive(Clone, Debug)]
pub struct KeyboardPrompt {
    pub title: String,
    pub description: String,
    pub default_text: String,
    /// Characters the title's buffer holds (including the terminator).
    pub max_chars: u32,
}

/// Asks the user for text; `None` means cancelled. Called on a dispatch thread (may block).
type TextInput = Arc<dyn Fn(&KeyboardPrompt) -> Option<String> + Send + Sync>;

static TEXT_INPUT: OnceLock<TextInput> = OnceLock::new();

/// Lets the host answer XamShowKeyboardUI. Without it the default text is returned (Xenia's
/// headless behaviour).
pub fn set_text_input(input: impl Fn(&KeyboardPrompt) -> Option<String> + Send + Sync + 'static) {
    let _ = TEXT_INPUT.set(Arc::new(input));
}

/// A null-terminated big-endian UTF-16 string (at most `max` units), or empty for null.
fn read_wstr(mem: &runtime_core::GuestMemory, ptr: u32, max: u32) -> String {
    if ptr == 0 {
        return String::new();
    }
    let units: Vec<u16> = (0..max).map(|i| mem.read_u16(ptr + i * 2)).take_while(|&u| u != 0).collect();
    String::from_utf16_lossy(&units)
}

/// Longest string read from the keyboard UI's title / description / default text.
const MAX_PROMPT_CHARS: u32 = 1024;

/// XUSER_INDEX_ANY.
const USER_INDEX_ANY: u32 = 0xFF;
/// Content flags the device selector rejects (Xenia).
const INVALID_SELECTOR_FLAGS: u32 = 0x83F0_0008;
/// How long the system UI stays "open" after answering.
const UI_CLOSE_DELAY: Duration = Duration::from_millis(200);
/// Lead time before the answer, as Xenia's headless dispatch.
const UI_OPEN_DELAY: Duration = Duration::from_millis(25);

/// (u32 UserIndex, u32 ContentType, u32 ContentFlags, u64 TotalRequested,
///  u32* DeviceId, XOVERLAPPED* Overlapped): picks the hard drive.
fn XamShowDeviceSelectorUI(c: &mut Call) -> u64 {
    let (user, flags, device_ptr, overlapped) = (c.arg(0), c.arg(2), c.arg(4), c.arg(5));
    if overlapped == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    if (user as usize >= MAX_USERS && user != USER_INDEX_ANY) || flags & INVALID_SELECTOR_FLAGS != 0 || device_ptr == 0
    {
        c.mem().write_u32(overlapped + 0x18, win32::INVALID_PARAMETER);
        return win32::INVALID_PARAMETER as u64;
    }
    let k = c.kernel;
    k.complete_overlapped_deferred(
        overlapped,
        move || {
            k.broadcast_notification(id::SYSTEM_UI, 1);
            std::thread::sleep(UI_OPEN_DELAY);
        },
        move || {
            log::info!("device selector: hard drive");
            k.mem().write_u32(device_ptr, DEVICE_HDD);
            (win32::SUCCESS, win32::SUCCESS, 0)
        },
        move || {
            std::thread::sleep(UI_CLOSE_DELAY);
            k.broadcast_notification(id::SYSTEM_UI, 0);
        },
    );
    win32::IO_PENDING as u64
}

/// (u32 UserIndex, u32 Flags, wchar* DefaultText, wchar* Title, wchar* Description,
///  wchar* Buffer, u32 BufferLength (chars), XOVERLAPPED* Overlapped)
fn XamShowKeyboardUI(c: &mut Call) -> u64 {
    let (default_ptr, title_ptr, desc_ptr) = (c.arg(2), c.arg(3), c.arg(4));
    let (buffer, buffer_chars, overlapped) = (c.arg(5), c.arg(6), c.arg(7));
    if buffer == 0 || overlapped == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    let mem = c.mem();
    let prompt = KeyboardPrompt {
        title: read_wstr(mem, title_ptr, MAX_PROMPT_CHARS),
        description: read_wstr(mem, desc_ptr, MAX_PROMPT_CHARS),
        default_text: read_wstr(mem, default_ptr, MAX_PROMPT_CHARS),
        max_chars: buffer_chars,
    };
    let k = c.kernel;
    k.complete_overlapped_deferred(
        overlapped,
        move || {
            k.broadcast_notification(id::SYSTEM_UI, 1);
            std::thread::sleep(UI_OPEN_DELAY);
        },
        move || {
            let text = match TEXT_INPUT.get() {
                Some(input) => input(&prompt),
                None => Some(prompt.default_text.clone()),
            };
            let Some(text) = text else {
                log::info!("keyboard UI \"{}\": cancelled", prompt.title);
                return (win32::SUCCESS, win32::CANCELLED, 0);
            };
            log::info!("keyboard UI \"{}\": {text:?}", prompt.title);
            // Truncated to the buffer, always terminated (Xenia's copy_and_swap_truncating).
            let mut units: Vec<u16> = text.encode_utf16().take(buffer_chars.saturating_sub(1) as usize).collect();
            units.push(0);
            k.mem().write_utf16(buffer, &units);
            (win32::SUCCESS, win32::SUCCESS, 0)
        },
        move || {
            std::thread::sleep(UI_CLOSE_DELAY);
            k.broadcast_notification(id::SYSTEM_UI, 0);
        },
    );
    win32::IO_PENDING as u64
}
