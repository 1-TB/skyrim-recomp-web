//! xam.xex exports.
#![allow(non_snake_case)]

pub mod content;
mod input;
pub mod msg;
pub mod notify;
pub mod profile;
mod ui;

pub use ui::{KeyboardPrompt, set_text_input};
pub mod user;

use std::sync::LazyLock;

use runtime_core::heap::protect;

use crate::dispatch::{Call, Export, ModuleId};
use crate::status::win32;
use crate::xconfig::{LANGUAGE_ENGLISH, category};

const X: ModuleId = ModuleId::XAM;

const INFO: &[Export] = &[
    export!(X, 0x3CB, XGetAVPack),
    export!(X, 0x3CC, XGetGameRegion),
    export!(X, 0x3CD, XGetLanguage),
    export!(X, 0x3D1, XGetVideoMode),
    export!(X, 0x282, XamGetSystemVersion),
    export!(X, 0x1EA, XamAlloc),
    export!(X, 0x1EC, XamFree),
    export!(X, 0x1A7, XamLoaderGetLaunchDataSize),
    export!(X, 0x1A8, XamLoaderGetLaunchData),
    export!(X, 0x001, NetDll_WSAStartup),
    export!(X, 0x002, NetDll_WSACleanup),
];

pub static EXPORTS: LazyLock<Vec<Export>> = LazyLock::new(|| {
    [
        INFO,
        content::EXPORTS,
        input::EXPORTS,
        msg::EXPORTS,
        notify::EXPORTS,
        profile::EXPORTS,
        ui::EXPORTS,
        user::EXPORTS,
    ]
    .concat()
});

/// AV pack: HDMI.
const AV_PACK_HDMI: u32 = 8;

fn XGetAVPack(_c: &mut Call) -> u64 {
    AV_PACK_HDMI as u64
}

fn XGetGameRegion(c: &mut Call) -> u64 {
    c.kernel.xconfig.read(category::SECURED, 0x03).map_or(0xFFFF, |b| u16::from_be_bytes([b[0], b[1]]) as u64)
}

fn XGetLanguage(c: &mut Call) -> u64 {
    c.kernel
        .xconfig
        .read(category::USER, 0x09)
        .map_or(LANGUAGE_ENGLISH, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])) as u64
}

/// Display resolution from the AV settings (default 1280x720).
pub(crate) fn display_size(kernel: &crate::Kernel) -> (u32, u32) {
    let packed = kernel
        .xconfig
        .read(category::USER, 0x13)
        .map_or((1280 << 16) | 720, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    (packed >> 16, packed & 0xFFFF)
}

/// Fills X_VIDEO_MODE (48 bytes) for the configured HDMI resolution.
pub(crate) fn write_video_mode(kernel: &crate::Kernel, out: u32) {
    let mem = kernel.mem();
    let (width, height) = display_size(kernel);
    let widescreen = width * 9 >= height * 16;
    mem.fill(out, 48, 0);
    mem.write_u32(out, width);
    mem.write_u32(out + 4, height);
    mem.write_u32(out + 8, 0);
    mem.write_u32(out + 12, widescreen as u32);
    mem.write_u32(out + 16, (width >= 0x500) as u32);
    mem.write_f32(out + 20, 60.0);
    mem.write_u32(out + 24, 1); // NTSC
    mem.write_u32(out + 28, 0x8A);
    mem.write_u32(out + 32, if widescreen { 0x01 } else { 0x03 });
}

fn XGetVideoMode(c: &mut Call) -> u64 {
    write_video_mode(c.kernel, c.arg(0));
    0
}

/// Reported as 0 (pre-versioned), matching Xenia: titles use it to gate newer
/// system APIs, which keeps them on the paths this kernel implements.
fn XamGetSystemVersion(_c: &mut Call) -> u64 {
    0
}

/// (u32 Flags, u32 Size, void** Out)
fn XamAlloc(c: &mut Call) -> u64 {
    let (flags, size, out) = (c.arg(0), c.arg(1), c.arg(2));
    let space = &c.kernel.rt.space;
    let addr = if size > 0xFD8 {
        space.alloc_virtual(size, 0x1000, true, protect::READWRITE, false)
    } else {
        space.system_alloc(size, 16)
    };
    match addr {
        Ok(a) => {
            c.mem().write_u32(out, a);
            win32::SUCCESS as u64
        }
        Err(e) => {
            log::warn!("XamAlloc({flags:#X}, {size:#X}): {e}");
            c.mem().write_u32(out, 0);
            win32::NOT_ENOUGH_MEMORY as u64
        }
    }
}

fn XamFree(c: &mut Call) -> u64 {
    let ptr = c.arg(0);
    let space = &c.kernel.rt.space;
    let r = if ptr.is_multiple_of(0x1000)
        && runtime_core::address_space::region_of(ptr) == runtime_core::address_space::Region::Virtual
    {
        space.release_virtual(ptr).map(|_| ())
    } else {
        space.system_free(ptr).map(|_| ())
    };
    if let Err(e) = r {
        log::warn!("XamFree({ptr:#010X}): {e}");
    }
    win32::SUCCESS as u64
}

/// (u32* Size): the title was started directly, with no launch data.
fn XamLoaderGetLaunchDataSize(c: &mut Call) -> u64 {
    let size_ptr = c.arg(0);
    if size_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    c.mem().write_u32(size_ptr, 0);
    win32::NOT_FOUND as u64
}

/// (void* Buffer, u32 Size)
fn XamLoaderGetLaunchData(_c: &mut Call) -> u64 {
    win32::NOT_FOUND as u64
}

/// sizeof(XWSADATA).
const WSADATA_SIZE: u32 = 0x190;

/// (u32 Caller, u16 Version, XWSADATA* Data). There is no network stack; the
/// call succeeds so titles reach their offline paths, as in Xenia.
fn NetDll_WSAStartup(c: &mut Call) -> u64 {
    let (version, data) = (c.arg(1) as u16, c.arg(2));
    if data != 0 {
        let mem = c.mem();
        mem.fill(data, WSADATA_SIZE, 0);
        mem.write_u16(data, version);
        mem.write_u16(data + 2, 0x0202);
    }
    0
}

fn NetDll_WSACleanup(_c: &mut Call) -> u64 {
    0
}
