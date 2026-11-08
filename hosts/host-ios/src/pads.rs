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

/// The title's rumble on the controller's haptics (CoreHaptics engines on its left and right
/// handles, like the XInput motors: the left one heavy, the right one light). Updated on the
/// main thread with the motor speeds the title last set.
#[derive(Default)]
pub struct Rumble {
    /// The controller the engines are for (its address) and the engines, left then right.
    engines: Option<(usize, [Option<objc2::rc::Retained<objc2_core_haptics::CHHapticEngine>>; 2])>,
    /// Per motor: the speed playing and its player.
    playing: [(u16, Option<Player>); 2],
    /// When the events playing started (they're restarted before they run out).
    started: Option<std::time::Instant>,
    /// After a failure, engines aren't made again before this.
    retry_at: Option<std::time::Instant>,
}

type Player = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_core_haptics::CHHapticPatternPlayer>>;

/// A continuous haptic event lasts at most this long (seconds); a longer rumble is restarted
/// before it ends.
const RUMBLE_EVENT_SECONDS: f64 = 30.0;

impl Rumble {
    /// Plays motor speeds `(left, right)` (0-65535) on the first controller with haptics.
    pub fn update(&mut self, speeds: (u16, u16), now: std::time::Instant) {
        use objc2::AnyThread;
        use objc2_core_haptics::*;
        // SAFETY: GameController and CoreHaptics calls on the main thread with valid arguments.
        unsafe {
            let controllers = GCController::controllers();
            let Some(controller) = controllers.iter().find(|c| c.extendedGamepad().is_some()) else {
                self.stop_all();
                self.engines = None;
                return;
            };
            let id = objc2::rc::Retained::as_ptr(&controller) as usize;
            if self.engines.as_ref().is_none_or(|(c, _)| *c != id) && self.retry_at.is_none_or(|t| now >= t) {
                self.stop_all();
                let haptics = controller.haptics();
                let engine = |locality: &GCHapticsLocality| {
                    let engine = haptics.as_ref()?.createEngineWithLocality(locality)?;
                    engine.startAndReturnError().ok()?;
                    Some(engine)
                };
                let mut engines = [engine(GCHapticsLocalityLeftHandle), engine(GCHapticsLocalityRightHandle)];
                if engines.iter().all(Option::is_none) {
                    // One actuator (or none per handle): both motors on the default locality.
                    let one = engine(GCHapticsLocalityDefault);
                    engines = [one.clone(), one];
                }
                if engines.iter().any(Option::is_some) {
                    log::info!("controller rumble: on");
                }
                self.engines = Some((id, engines));
                self.started = None;
            }
            let Some((_, engines)) = self.engines.clone() else { return };
            let mut failed = false;
            let restart =
                self.started.is_some_and(|t| now.duration_since(t).as_secs_f64() > RUMBLE_EVENT_SECONDS - 5.0);
            for (motor, speed) in [speeds.0, speeds.1].into_iter().enumerate() {
                let (playing, player) = &mut self.playing[motor];
                if *playing == speed && !(restart && speed != 0) {
                    continue;
                }
                if let Some(p) = player.take() {
                    let _ = p.stopAtTime_error(CHHapticTimeImmediate);
                }
                *playing = speed;
                let Some(engine) = &engines[motor] else { continue };
                if speed == 0 {
                    continue;
                }
                let param = |id: &CHHapticEventParameterID, v: f32| {
                    CHHapticEventParameter::initWithParameterID_value(CHHapticEventParameter::alloc(), id, v)
                };
                // The left motor is the heavy, low-frequency one.
                let sharpness = if motor == 0 { 0.2 } else { 0.7 };
                let params = objc2_foundation::NSArray::from_retained_slice(&[
                    param(CHHapticEventParameterIDHapticIntensity, f32::from(speed) / 65535.0),
                    param(CHHapticEventParameterIDHapticSharpness, sharpness),
                ]);
                let event = CHHapticEvent::initWithEventType_parameters_relativeTime_duration(
                    CHHapticEvent::alloc(),
                    CHHapticEventTypeHapticContinuous,
                    &params,
                    0.0,
                    RUMBLE_EVENT_SECONDS,
                );
                let events = objc2_foundation::NSArray::from_retained_slice(&[event]);
                let none = objc2_foundation::NSArray::new();
                let Ok(pattern) =
                    CHHapticPattern::initWithEvents_parameters_error(CHHapticPattern::alloc(), &events, &none)
                else {
                    continue;
                };
                match engine.createPlayerWithPattern_error(&pattern) {
                    Ok(p) if p.startAtTime_error(CHHapticTimeImmediate).is_ok() => *player = Some(p),
                    _ => failed = true,
                }
            }
            if failed {
                // The system stops engines (in the background, say): new ones next time.
                self.engines = None;
                self.playing = Default::default();
                self.started = None;
                self.retry_at = Some(now + std::time::Duration::from_secs(2));
                return;
            }
            if speeds != (0, 0) && (restart || self.started.is_none()) {
                self.started = Some(now);
            } else if speeds == (0, 0) {
                self.started = None;
            }
        }
    }

    fn stop_all(&mut self) {
        for (speed, player) in &mut self.playing {
            if let Some(p) = player.take() {
                // SAFETY: stopping a player on the main thread.
                let _ = unsafe {
                    objc2_core_haptics::CHHapticPatternPlayer::stopAtTime_error(
                        &*p,
                        objc2_core_haptics::CHHapticTimeImmediate,
                    )
                };
            }
            *speed = 0;
        }
    }
}
