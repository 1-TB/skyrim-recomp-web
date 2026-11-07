//! Controllers through Apple's GameController framework (gilrs has no iOS backend): the first
//! extended gamepad (Xbox, PlayStation, MFi), and a hardware keyboard mapped like the desktop
//! host's window.rs and the web page's input.js. Polled on the main thread.

use objc2_game_controller::*;

/// XINPUT button bits (as in host-native's window.rs).
pub mod bits {
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
    pub const A: u16 = 0x1000;
    pub const B: u16 = 0x2000;
    pub const X: u16 = 0x4000;
    pub const Y: u16 = 0x8000;
}
use bits::*;

/// One source's controller state, before merging.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Reading {
    pub buttons: u16,
    /// 0 to 1.
    pub left_trigger: f64,
    pub right_trigger: f64,
    /// -1 to 1, y up.
    pub left: (f64, f64),
    pub right: (f64, f64),
}

impl Reading {
    /// Combines two sources: buttons add up, triggers and stick axes take the larger deflection.
    pub fn merge(self, other: Reading) -> Reading {
        let pick = |a: f64, b: f64| if b.abs() > a.abs() { b } else { a };
        Reading {
            buttons: self.buttons | other.buttons,
            left_trigger: self.left_trigger.max(other.left_trigger),
            right_trigger: self.right_trigger.max(other.right_trigger),
            left: (pick(self.left.0, other.left.0), pick(self.left.1, other.left.1)),
            right: (pick(self.right.0, other.right.0), pick(self.right.1, other.right.1)),
        }
    }

    pub fn gamepad(&self) -> kernel_hle::input::Gamepad {
        let axis = |v: f64| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let trigger = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        kernel_hle::input::Gamepad {
            buttons: self.buttons,
            left_trigger: trigger(self.left_trigger),
            right_trigger: trigger(self.right_trigger),
            thumb_lx: axis(self.left.0),
            thumb_ly: axis(self.left.1),
            thumb_rx: axis(self.right.0),
            thumb_ry: axis(self.right.1),
        }
    }
}

/// The first connected extended gamepad, if any.
pub fn gamepad() -> Option<Reading> {
    // SAFETY: GameController getters, called on the main thread.
    unsafe {
        let controllers = GCController::controllers();
        let pad = controllers.iter().find_map(|c| c.extendedGamepad())?;
        let pressed = |b: &GCControllerButtonInput| b.isPressed();
        let mut buttons = 0;
        let mut set = |on: bool, bit: u16| {
            if on {
                buttons |= bit;
            }
        };
        set(pressed(&pad.buttonA()), A);
        set(pressed(&pad.buttonB()), B);
        set(pressed(&pad.buttonX()), X);
        set(pressed(&pad.buttonY()), Y);
        set(pressed(&pad.leftShoulder()), LEFT_SHOULDER);
        set(pressed(&pad.rightShoulder()), RIGHT_SHOULDER);
        set(pressed(&pad.buttonMenu()), START);
        set(pad.buttonOptions().is_some_and(|b| pressed(&b)), BACK);
        set(pad.leftThumbstickButton().is_some_and(|b| pressed(&b)), LEFT_THUMB);
        set(pad.rightThumbstickButton().is_some_and(|b| pressed(&b)), RIGHT_THUMB);
        let dpad = pad.dpad();
        set(pressed(&dpad.up()), DPAD_UP);
        set(pressed(&dpad.down()), DPAD_DOWN);
        set(pressed(&dpad.left()), DPAD_LEFT);
        set(pressed(&dpad.right()), DPAD_RIGHT);
        let stick = |s: &GCControllerDirectionPad| (f64::from(s.xAxis().value()), f64::from(s.yAxis().value()));
        Some(Reading {
            buttons,
            left_trigger: f64::from(pad.leftTrigger().value()),
            right_trigger: f64::from(pad.rightTrigger().value()),
            left: stick(&pad.leftThumbstick()),
            right: stick(&pad.rightThumbstick()),
        })
    }
}

/// A hardware keyboard (iPad Magic Keyboard, Bluetooth), with host-native's key map: WASD left
/// stick, arrows right stick, Space A, Escape/Tab B, R X, F Y, Q/E bumpers, Z/C triggers,
/// Enter Start, Backspace Back, X/V stick clicks, 1-4 d-pad.
pub fn keyboard() -> Option<Reading> {
    // SAFETY: GameController getters and its key code constants, on the main thread.
    unsafe {
        let input = GCKeyboard::coalescedKeyboard()?.keyboardInput()?;
        let held = |code: GCKeyCode| input.buttonForKeyCode(code).is_some_and(|b| b.isPressed());
        let keys: [(GCKeyCode, u16); 15] = [
            (GCKeyCodeSpacebar, A),
            (GCKeyCodeEscape, B),
            (GCKeyCodeTab, B),
            (GCKeyCodeKeyR, X),
            (GCKeyCodeKeyF, Y),
            (GCKeyCodeKeyQ, LEFT_SHOULDER),
            (GCKeyCodeKeyE, RIGHT_SHOULDER),
            (GCKeyCodeReturnOrEnter, START),
            (GCKeyCodeDeleteOrBackspace, BACK),
            (GCKeyCodeKeyX, LEFT_THUMB),
            (GCKeyCodeKeyV, RIGHT_THUMB),
            (GCKeyCodeOne, DPAD_UP),
            (GCKeyCodeTwo, DPAD_DOWN),
            (GCKeyCodeThree, DPAD_LEFT),
            (GCKeyCodeFour, DPAD_RIGHT),
        ];
        let buttons = keys.iter().filter(|(k, _)| held(*k)).fold(0, |b, (_, bit)| b | bit);
        let dir = |neg: GCKeyCode, pos: GCKeyCode| f64::from(u8::from(held(pos))) - f64::from(u8::from(held(neg)));
        let reading = Reading {
            buttons,
            left_trigger: f64::from(u8::from(held(GCKeyCodeKeyZ))),
            right_trigger: f64::from(u8::from(held(GCKeyCodeKeyC))),
            left: (dir(GCKeyCodeKeyA, GCKeyCodeKeyD), dir(GCKeyCodeKeyS, GCKeyCodeKeyW)),
            right: (dir(GCKeyCodeLeftArrow, GCKeyCodeRightArrow), dir(GCKeyCodeDownArrow, GCKeyCodeUpArrow)),
        };
        Some(reading)
    }
}
