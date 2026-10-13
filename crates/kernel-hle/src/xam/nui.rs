//! Kinect (NUI) and its microphone array. No Kinect is ever connected: the
//! status queries report it absent the way Xenia does when its NUI emulation
//! is off, so titles with optional Kinect features (voice commands) skip them.
#![allow(non_snake_case)]

use crate::dispatch::{Call, Export, ModuleId};
use crate::status::hresult;

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x885, XamNuiGetDeviceStatus),
    export!(X, 0x870, XamNuiCameraTiltGetStatus),
    export!(X, 0x318, XamVoiceGetMicArrayStatus),
];

/// XamNuiGetDeviceStatus's result when the sensor isn't initialized.
const NUI_NOT_CONNECTED: u32 = 0xC005_0006;
/// sizeof(X_NUI_DEVICE_STATUS): camera, tilt and mic status, status bits.
const DEVICE_STATUS_SIZE: u32 = 0x10;
/// sizeof(X_NUI_TILT_STATUS) and its signature ('XtSs').
const TILT_STATUS_SIZE: u32 = 0x50;
const TILT_STATUS_SIGNATURE: u32 = 0x5874_5373;

/// (X_NUI_DEVICE_STATUS* Status): zeroed (no status bits) and "not connected".
fn XamNuiGetDeviceStatus(c: &mut Call) -> u64 {
    let out = c.arg(0);
    if out != 0 {
        c.mem().fill(out, DEVICE_STATUS_SIZE, 0);
    }
    NUI_NOT_CONNECTED as u64
}

/// (X_NUI_TILT_STATUS* Status): the tilt motor's accelerometer state. A caller's
/// structure must declare its size and signature; with no sensor there is never
/// a reading (E_FAIL either way, as Xenia).
fn XamNuiCameraTiltGetStatus(c: &mut Call) -> u64 {
    let status = c.arg(0);
    if status != 0 {
        let mem = c.mem();
        if mem.read_u32(status) < TILT_STATUS_SIZE || mem.read_u32(status + 4) != TILT_STATUS_SIGNATURE {
            log::debug!("XamNuiCameraTiltGetStatus({status:#010X}): bad size or signature");
        }
    }
    hresult::E_FAIL as u64
}

/// () -> mic array status; 0 means no microphone array is connected.
fn XamVoiceGetMicArrayStatus(_c: &mut Call) -> u64 {
    0
}
