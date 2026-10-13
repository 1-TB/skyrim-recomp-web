//! Controller state the host feeds in and XamInputGetState reads.
//!
//! Hosts own the devices (keyboard, browser Gamepad API, ...) and push a
//! complete XINPUT_GAMEPAD snapshot whenever something changes; the title polls.

use std::sync::Mutex;

/// Users (controller ports).
pub const MAX_USERS: usize = 4;

/// XINPUT_GAMEPAD button bits.
pub mod buttons {
    pub const DPAD_UP: u16 = 0x0001;
    pub const DPAD_DOWN: u16 = 0x0002;
    pub const DPAD_LEFT: u16 = 0x0004;
    pub const DPAD_RIGHT: u16 = 0x0008;
    pub const START: u16 = 0x0010;
    pub const BACK: u16 = 0x0020;
    pub const LEFT_THUMB: u16 = 0x0040;
    pub const RIGHT_THUMB: u16 = 0x0080;
    pub const LEFT_SHOULDER: u16 = 0x0100;
    pub const RIGHT_SHOULDER: u16 = 0x0200;
    pub const GUIDE: u16 = 0x0400;
    pub const A: u16 = 0x1000;
    pub const B: u16 = 0x2000;
    pub const X: u16 = 0x4000;
    pub const Y: u16 = 0x8000;
}

/// XINPUT_GAMEPAD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gamepad {
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub thumb_lx: i16,
    pub thumb_ly: i16,
    pub thumb_rx: i16,
    pub thumb_ry: i16,
}

#[derive(Debug, Clone, Copy, Default)]
struct Pad {
    state: Gamepad,
    /// XINPUT_STATE.dwPacketNumber: changes whenever the state does.
    packet: u32,
    /// Last XINPUT_VIBRATION motor speeds the title set.
    vibration: (u16, u16),
}

#[derive(Default)]
pub struct InputSystem {
    pads: [Mutex<Option<Pad>>; MAX_USERS],
}

impl InputSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Plugs in a controller for `user` with everything released.
    pub fn connect(&self, user: usize) {
        if let Some(p) = self.pads.get(user) {
            p.lock().unwrap().get_or_insert_with(Pad::default);
        }
    }

    pub fn disconnect(&self, user: usize) {
        if let Some(p) = self.pads.get(user) {
            *p.lock().unwrap() = None;
        }
    }

    pub fn is_connected(&self, user: usize) -> bool {
        self.pads.get(user).is_some_and(|p| p.lock().unwrap().is_some())
    }

    /// Replaces the state of `user`'s controller, connecting it if needed.
    pub fn update(&self, user: usize, state: Gamepad) {
        let Some(p) = self.pads.get(user) else { return };
        let mut p = p.lock().unwrap();
        let pad = p.get_or_insert_with(Pad::default);
        if pad.state != state {
            pad.state = state;
            pad.packet = pad.packet.wrapping_add(1);
        }
    }

    /// (packet number, state) if a controller is connected.
    pub fn state(&self, user: usize) -> Option<(u32, Gamepad)> {
        let p = self.pads.get(user)?.lock().unwrap();
        p.map(|p| (p.packet, p.state))
    }

    /// Records rumble; returns false when no controller is connected.
    pub fn set_vibration(&self, user: usize, left: u16, right: u16) -> bool {
        let Some(p) = self.pads.get(user) else { return false };
        match p.lock().unwrap().as_mut() {
            Some(pad) => {
                pad.vibration = (left, right);
                true
            }
            None => false,
        }
    }

    pub fn vibration(&self, user: usize) -> Option<(u16, u16)> {
        self.pads.get(user)?.lock().unwrap().map(|p| p.vibration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_number_changes_only_with_state() {
        let input = InputSystem::new();
        assert!(input.state(0).is_none());
        input.connect(0);
        let (p0, s0) = input.state(0).unwrap();
        assert_eq!(s0, Gamepad::default());
        input.update(0, Gamepad { buttons: buttons::A, ..Default::default() });
        let (p1, s1) = input.state(0).unwrap();
        assert_eq!(s1.buttons, buttons::A);
        assert_ne!(p0, p1);
        input.update(0, s1);
        assert_eq!(input.state(0).unwrap().0, p1);
        assert!(input.set_vibration(0, 1, 2));
        assert!(!input.set_vibration(1, 1, 2));
        input.disconnect(0);
        assert!(!input.is_connected(0));
    }
}
