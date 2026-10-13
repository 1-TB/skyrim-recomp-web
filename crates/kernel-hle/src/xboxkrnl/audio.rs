//! XAudio render-driver exports, backed by [`crate::audio`].
//! Behavior follows Xenia's xboxkrnl_audio.cc.
#![allow(non_snake_case)]

use crate::audio::{client_index, driver_handle};
use crate::dispatch::{Call, Export, ModuleId};
use crate::status::{hresult, win32};
use crate::xconfig::category;

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x1F3, XAudioRegisterRenderDriverClient),
    export!(K, 0x1F4, XAudioUnregisterRenderDriverClient),
    export!(K, 0x1F5, XAudioSubmitRenderDriverFrame),
    export!(K, 0x1F7, XAudioGetVoiceCategoryVolumeChangeMask),
    export!(K, 0x1F8, XAudioGetVoiceCategoryVolume),
    export!(K, 0x1FF, XAudioGetSpeakerConfig),
    export!(K, 0x34D, XAudioEnableDucker),
];

/// XCONFIG user setting: audio flags.
const USER_AUDIO_FLAGS: u16 = 0x0B;

/// (u32* Config): the dashboard's audio flags.
fn XAudioGetSpeakerConfig(c: &mut Call) -> u64 {
    let out = c.arg(0);
    let flags = c
        .kernel
        .xconfig
        .read(category::USER, USER_AUDIO_FLAGS)
        .map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    if out != 0 {
        c.mem().write_u32(out, flags);
    }
    win32::SUCCESS as u64
}

/// (void* Driver, u32* Mask): voice-chat volumes never change here.
fn XAudioGetVoiceCategoryVolumeChangeMask(c: &mut Call) -> u64 {
    let out = c.arg(1);
    if out != 0 {
        c.mem().write_u32(out, 0);
    }
    win32::SUCCESS as u64
}

/// (u32 Category, float* Volume)
fn XAudioGetVoiceCategoryVolume(c: &mut Call) -> u64 {
    let out = c.arg(1);
    if out != 0 {
        c.mem().write_f32(out, 1.0);
    }
    win32::SUCCESS as u64
}

fn XAudioEnableDucker(_c: &mut Call) -> u64 {
    win32::SUCCESS as u64
}

/// (u32 Callback[2] = { routine, argument }, u32* Driver)
fn XAudioRegisterRenderDriverClient(c: &mut Call) -> u64 {
    let (callback_ptr, driver_ptr) = (c.arg(0), c.arg(1));
    if callback_ptr == 0 {
        return hresult::E_INVALIDARG as u64;
    }
    let (callback, arg) = (c.mem().read_u32(callback_ptr), c.mem().read_u32(callback_ptr + 4));
    if callback == 0 {
        return hresult::E_INVALIDARG as u64;
    }
    match c.kernel.register_audio_client(callback, arg) {
        Some(index) => {
            if driver_ptr != 0 {
                c.mem().write_u32(driver_ptr, driver_handle(index));
            }
            win32::SUCCESS as u64
        }
        None => {
            log::error!("XAudioRegisterRenderDriverClient: no free client slot");
            hresult::E_FAIL as u64
        }
    }
}

/// (void* Driver)
fn XAudioUnregisterRenderDriverClient(c: &mut Call) -> u64 {
    if let Some(i) = client_index(c.arg(0)) {
        c.kernel.unregister_audio_client(i);
    }
    win32::SUCCESS as u64
}

/// (void* Driver, float* Samples): 6 x 256 big-endian floats.
fn XAudioSubmitRenderDriverFrame(c: &mut Call) -> u64 {
    let (driver, samples) = (c.arg(0), c.arg(1));
    match client_index(driver) {
        Some(i) if samples != 0 => {
            c.kernel.submit_audio_frame(i, samples);
            win32::SUCCESS as u64
        }
        _ => hresult::E_INVALIDARG as u64,
    }
}
