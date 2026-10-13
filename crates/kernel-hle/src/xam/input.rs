//! XamInput* (XInput) exports, backed by [`crate::input::InputSystem`].
//! Behavior follows Xenia's xam_input.cc.
#![allow(non_snake_case)]

use crate::dispatch::{Call, Export, ModuleId};
use crate::input::{Gamepad, MAX_USERS};
use crate::status::win32;

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x190, XamInputGetCapabilities),
    export!(X, 0x191, XamInputGetState),
    export!(X, 0x192, XamInputSetState),
    export!(X, 0x2AD, XamInputGetCapabilitiesEx),
];

/// XUSER_INDEX_ANY.
const USER_INDEX_ANY: u32 = 0xFF;
/// XINPUT_FLAG_ANYUSER.
const FLAG_ANY_USER: u32 = 0x4000_0000;
/// XINPUT_DEVTYPE_GAMEPAD / XINPUT_DEVSUBTYPE_GAMEPAD.
const DEVTYPE_GAMEPAD: u8 = 1;
const DEVSUBTYPE_GAMEPAD: u8 = 1;

/// The user a call addresses; "any user" means the first port, as in Xenia.
fn resolve_user(user: u32, flags: u32) -> u32 {
    if user & USER_INDEX_ANY == USER_INDEX_ANY || flags & FLAG_ANY_USER != 0 { 0 } else { user }
}

fn write_gamepad(c: &Call, at: u32, g: &Gamepad) {
    let mem = c.mem();
    mem.write_u16(at, g.buttons);
    mem.write_u8(at + 2, g.left_trigger);
    mem.write_u8(at + 3, g.right_trigger);
    mem.write_u16(at + 4, g.thumb_lx as u16);
    mem.write_u16(at + 6, g.thumb_ly as u16);
    mem.write_u16(at + 8, g.thumb_rx as u16);
    mem.write_u16(at + 10, g.thumb_ry as u16);
}

/// Fills XINPUT_CAPABILITIES (0x14 bytes) as Xenia's SDL driver reports a
/// standard gamepad: every control but Guide, full-range axes, both motors.
fn write_capabilities(c: &Call, caps: u32) {
    let mem = c.mem();
    mem.fill(caps, 0x14, 0);
    mem.write_u8(caps, DEVTYPE_GAMEPAD);
    mem.write_u8(caps + 1, DEVSUBTYPE_GAMEPAD);
    let all = Gamepad {
        buttons: 0xF3FF,
        left_trigger: 0xFF,
        right_trigger: 0xFF,
        thumb_lx: -1,
        thumb_ly: -1,
        thumb_rx: -1,
        thumb_ry: -1,
    };
    write_gamepad(c, caps + 4, &all);
    mem.write_u16(caps + 0x10, 0xFFFF);
    mem.write_u16(caps + 0x12, 0xFFFF);
}

fn get_capabilities(c: &Call, user: u32, flags: u32, caps: u32) -> u64 {
    if caps == 0 {
        return win32::BAD_ARGUMENTS as u64;
    }
    c.mem().fill(caps, 0x14, 0);
    let user = resolve_user(user, flags);
    if user as usize >= MAX_USERS || !c.kernel.input.is_connected(user as usize) {
        return win32::DEVICE_NOT_CONNECTED as u64;
    }
    write_capabilities(c, caps);
    win32::SUCCESS as u64
}

/// (u32 UserIndex, u32 Flags, XINPUT_CAPABILITIES* Caps)
fn XamInputGetCapabilities(c: &mut Call) -> u64 {
    get_capabilities(c, c.arg(0), c.arg(1), c.arg(2))
}

/// (u32 Version, u32 UserIndex, u32 Flags, XINPUT_CAPABILITIES_EX* Caps)
fn XamInputGetCapabilitiesEx(c: &mut Call) -> u64 {
    if c.arg(0) > 1 {
        return win32::NOT_SUPPORTED as u64;
    }
    get_capabilities(c, c.arg(1), c.arg(2), c.arg(3))
}

/// (u32 UserIndex, u32 Flags, XINPUT_STATE* State)
fn XamInputGetState(c: &mut Call) -> u64 {
    let (user, flags, out) = (c.arg(0), c.arg(1), c.arg(2));
    if out != 0 {
        c.mem().fill(out, 0x10, 0);
    }
    let user = resolve_user(user, flags);
    if user as usize >= MAX_USERS {
        return win32::DEVICE_NOT_CONNECTED as u64;
    }
    match c.kernel.input.state(user as usize) {
        Some((packet, state)) => {
            if out != 0 {
                c.mem().write_u32(out, packet);
                write_gamepad(c, out + 4, &state);
            }
            win32::SUCCESS as u64
        }
        None => win32::DEVICE_NOT_CONNECTED as u64,
    }
}

/// (u32 UserIndex, u32 Flags, XINPUT_VIBRATION* Vibration)
fn XamInputSetState(c: &mut Call) -> u64 {
    let (user, vibration) = (c.arg(0), c.arg(2));
    if user as usize >= MAX_USERS {
        return win32::DEVICE_NOT_CONNECTED as u64;
    }
    if vibration == 0 {
        return win32::BAD_ARGUMENTS as u64;
    }
    let (left, right) = (c.mem().read_u16(vibration), c.mem().read_u16(vibration + 2));
    if c.kernel.input.set_vibration(user as usize, left, right) {
        win32::SUCCESS as u64
    } else {
        win32::DEVICE_NOT_CONNECTED as u64
    }
}
