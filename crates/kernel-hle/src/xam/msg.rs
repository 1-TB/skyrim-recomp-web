//! XMsg*: messages to XAM's built-in "apps" (music player, content, Live).
//! Every request completes before the call returns. Behavior follows Xenia's
//! xam_msg.cc and apps/.
#![allow(non_snake_case)]

use std::sync::Mutex;

use runtime_core::layout::kthread;

use crate::Kernel;
use crate::dispatch::{Call, Export, ModuleId};
use crate::status::{hresult, win32};

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x1F4, XMsgInProcessCall),
    export!(X, 0x1F5, XMsgCompleteIORequest),
    export!(X, 0x1F7, XMsgStartIORequest),
    export!(X, 0x1F8, XMsgCancelIORequest),
    export!(X, 0x1FB, XamGetOverlappedResult),
    export!(X, 0x1FC, XMsgStartIORequestEx),
];

/// App ids.
mod app {
    pub const MESSENGER: u32 = 0xF7;
    pub const XMP: u32 = 0xFA;
    pub const XGI: u32 = 0xFB;
    pub const XLIVEBASE: u32 = 0xFC;
    pub const XAM: u32 = 0xFE;
}

/// XMP (music player) client and playback-controller ids.
const XMP_CLIENT_GAME: u32 = 2;
const PLAYBACK_CONTROLLER_GAME: u32 = 0;
const PLAYBACK_CONTROLLER_RESTORE: u32 = 4;
/// kXNotificationXmpPlaybackControllerChanged.
const XMP_PLAYBACK_CONTROLLER_CHANGED: u32 = 0x0A00_0003;

/// Content device types (XContentQueryVolumeDeviceType).
const DEVICE_TYPE_HDD: u32 = 1;

/// Background-music ownership: the title or the user's own soundtrack.
pub struct MediaPlayer {
    state: Mutex<(u32, u32, bool)>,
}

impl Default for MediaPlayer {
    fn default() -> Self {
        MediaPlayer { state: Mutex::new((XMP_CLIENT_GAME, PLAYBACK_CONTROLLER_GAME, false)) }
    }
}

impl MediaPlayer {
    fn title_in_control(&self) -> bool {
        self.state.lock().unwrap().1 == PLAYBACK_CONTROLLER_GAME
    }
}

impl Kernel {
    /// Delivers one app message. `None` when no such app exists.
    fn dispatch_app_message(&self, app_id: u32, message: u32, buffer: u32, length: u32) -> Option<u32> {
        let mem = self.mem();
        let result = match (app_id, message) {
            // XMPSetPlaybackController(client, request, locked)
            (app::XMP, 0x0007_001A) => {
                let (client, request, locked) =
                    (mem.read_u32(buffer), mem.read_u32(buffer + 4), mem.read_u32(buffer + 8));
                let in_control = {
                    let mut s = self.media.state.lock().unwrap();
                    *s = if request == PLAYBACK_CONTROLLER_RESTORE {
                        (XMP_CLIENT_GAME, PLAYBACK_CONTROLLER_GAME, locked != 0)
                    } else {
                        (client, request, locked != 0)
                    };
                    s.1 == PLAYBACK_CONTROLLER_GAME
                };
                self.broadcast_notification(XMP_PLAYBACK_CONTROLLER_CHANGED, in_control as u32);
                hresult::S_OK
            }
            // XMPGetPlaybackController(client, &controller, &locked)
            (app::XMP, 0x0007_001B) => {
                let (controller_ptr, locked_ptr) = (mem.read_u32(buffer + 4), mem.read_u32(buffer + 8));
                let controller = self.media.state.lock().unwrap().1;
                if controller_ptr != 0 {
                    mem.write_u32(controller_ptr, controller);
                }
                if locked_ptr != 0 {
                    mem.write_u32(locked_ptr, !self.media.title_in_control() as u32);
                }
                hresult::S_OK
            }
            // XContentQueryVolumeDeviceType(root_name[64], is_title, &type, overlapped)
            (app::XAM, 0x0002_0021) => {
                let name = String::from_utf8_lossy(&mem.read_cstr(buffer, 64)).into_owned();
                match self.vfs.symlink_target(&format!("{name}:")) {
                    // Content packages mounted by XamContentCreate live on the
                    // hard drive; anything else (the disc) is not a package.
                    Some(t) if t.starts_with("\\Device\\Content_") => {
                        let type_ptr = mem.read_u32(buffer + 0x44);
                        if type_ptr != 0 {
                            mem.write_u32(type_ptr, DEVICE_TYPE_HDD);
                        }
                        hresult::S_OK
                    }
                    _ => hresult::E_INVALIDARG,
                }
            }
            (app::XMP | app::XAM | app::XGI | app::XLIVEBASE | app::MESSENGER, _) => {
                log::warn!(
                    "unimplemented XAM app message app={app_id:#04X} msg={message:#010X} buffer={buffer:#010X} len={length:#X}"
                );
                hresult::E_FAIL
            }
            _ => return None,
        };
        Some(result)
    }
}

fn set_last_error(c: &Call, error: u32) {
    let k = c.thread().block.kthread;
    c.mem().write_u32(k + kthread::LAST_ERROR, error);
}

/// (u32 App, u32 Message, u32 Arg1, u32 Arg2): synchronous message.
fn XMsgInProcessCall(c: &mut Call) -> u64 {
    let (app_id, message, arg1, arg2) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3));
    match c.kernel.dispatch_app_message(app_id, message, arg1, arg2) {
        Some(r) => r as u64,
        None => {
            log::error!("XMsgInProcessCall: app {app_id:#04X} undefined");
            hresult::E_NOTFOUND as u64
        }
    }
}

fn start_io_request(c: &Call, app_id: u32, message: u32, overlapped: u32, buffer: u32, length: u32) -> u64 {
    let mut result = match c.kernel.dispatch_app_message(app_id, message, buffer, length) {
        Some(r) => r,
        None => {
            log::error!("XMsgStartIORequest: app {app_id:#04X} undefined");
            set_last_error(c, win32::NOT_FOUND);
            hresult::E_INVALIDARG
        }
    };
    if overlapped != 0 {
        c.kernel.complete_overlapped(overlapped, result);
        result = win32::IO_PENDING;
    }
    if result == hresult::S_OK || result == win32::IO_PENDING {
        set_last_error(c, 0);
    }
    result as u64
}

/// (u32 App, u32 Message, XOVERLAPPED* Overlapped, void* Buffer, u32 Length)
fn XMsgStartIORequest(c: &mut Call) -> u64 {
    start_io_request(c, c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4))
}

/// As XMsgStartIORequest plus task arguments, which only matter for
/// asynchronous delivery.
fn XMsgStartIORequestEx(c: &mut Call) -> u64 {
    start_io_request(c, c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4))
}

/// (XOVERLAPPED* Overlapped, BOOL Wait): requests are never left pending.
fn XMsgCancelIORequest(_c: &mut Call) -> u64 {
    0
}

/// (XOVERLAPPED* Overlapped, u32 Result, u32 ExtendedError, u32 Length)
fn XMsgCompleteIORequest(c: &mut Call) -> u64 {
    c.kernel.complete_overlapped_ex(c.arg(0), c.arg(1), c.arg(2), c.arg(3));
    win32::SUCCESS as u64
}

/// (XOVERLAPPED* Overlapped, u32* Length, BOOL Wait)
fn XamGetOverlappedResult(c: &mut Call) -> u64 {
    let (overlapped, length_ptr) = (c.arg(0), c.arg(1));
    let result = c.kernel.overlapped_result(overlapped);
    if result == win32::IO_PENDING {
        return win32::IO_INCOMPLETE as u64;
    }
    if length_ptr != 0 {
        c.mem().write_u32(length_ptr, c.mem().read_u32(overlapped + 4));
    }
    result as u64
}
