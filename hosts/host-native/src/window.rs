//! The game window: winit owns the main thread (the title boots on another one), the renderer
//! presents into the window from its own thread, and controller state comes from the first
//! gamepad (gilrs) together with the keyboard and mouse, mapped like the web page's input.js.
//!
//! Keys: WASD left stick, arrows right stick, Space A, Escape/Tab B, R X, F Y, Q/E bumpers,
//! Z/C triggers, Enter Start, Backspace Back, X/V stick clicks, 1-4 d-pad. Clicking the window
//! captures the mouse: moving it turns the camera (right stick), the left/right buttons are the
//! right/left triggers, and Escape releases it. F11 toggles fullscreen.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use kernel_hle::Kernel;
use kernel_hle::input::Gamepad;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};
use xenos_render::native::{Window as RenderWindow, WindowHandles};

const A: u16 = 0x1000;
const B: u16 = 0x2000;
const X: u16 = 0x4000;
const Y: u16 = 0x8000;
const START: u16 = 0x0010;
const BACK: u16 = 0x0020;
const LEFT_THUMB: u16 = 0x0040;
const RIGHT_THUMB: u16 = 0x0080;
const LEFT_SHOULDER: u16 = 0x0100;
const RIGHT_SHOULDER: u16 = 0x0200;
const DPAD_UP: u16 = 0x0001;
const DPAD_DOWN: u16 = 0x0002;
const DPAD_LEFT: u16 = 0x0004;
const DPAD_RIGHT: u16 = 0x0008;

const KEY_BUTTONS: [(KeyCode, u16); 16] = [
    (KeyCode::Space, A),
    (KeyCode::Escape, B),
    (KeyCode::Tab, B),
    (KeyCode::KeyR, X),
    (KeyCode::KeyF, Y),
    (KeyCode::KeyQ, LEFT_SHOULDER),
    (KeyCode::KeyE, RIGHT_SHOULDER),
    (KeyCode::Enter, START),
    (KeyCode::Backspace, BACK),
    (KeyCode::KeyX, LEFT_THUMB),
    (KeyCode::KeyV, RIGHT_THUMB),
    (KeyCode::Digit1, DPAD_UP),
    (KeyCode::Digit2, DPAD_DOWN),
    (KeyCode::Digit3, DPAD_LEFT),
    (KeyCode::Digit4, DPAD_RIGHT),
    (KeyCode::NumpadEnter, START),
];

const PAD_BUTTONS: [(gilrs::Button, u16); 14] = [
    (gilrs::Button::South, A),
    (gilrs::Button::East, B),
    (gilrs::Button::West, X),
    (gilrs::Button::North, Y),
    (gilrs::Button::LeftTrigger, LEFT_SHOULDER),
    (gilrs::Button::RightTrigger, RIGHT_SHOULDER),
    (gilrs::Button::Select, BACK),
    (gilrs::Button::Start, START),
    (gilrs::Button::LeftThumb, LEFT_THUMB),
    (gilrs::Button::RightThumb, RIGHT_THUMB),
    (gilrs::Button::DPadUp, DPAD_UP),
    (gilrs::Button::DPadDown, DPAD_DOWN),
    (gilrs::Button::DPadLeft, DPAD_LEFT),
    (gilrs::Button::DPadRight, DPAD_RIGHT),
];

/// How often controller state is sent to the title (it polls once a frame).
const INPUT_PERIOD: Duration = Duration::from_millis(8);
/// Mouse movement (pixels per 60 Hz frame) that deflects the right stick fully, as on the page.
const MOUSE_FULL_DEFLECTION: f64 = 12.0;
const STICK: f64 = 32767.0;

/// The kernel, once the title has booted (input goes nowhere before that).
static KERNEL: OnceLock<&'static Kernel> = OnceLock::new();

/// Opens the window and runs its event loop on this (the main) thread. `boot` gets the window
/// for the renderer once it exists and must start the title on another thread.
pub fn run(boot: impl FnOnce(RenderWindow) + 'static) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    let gilrs = match gilrs::Gilrs::new() {
        Ok(g) => Some(g),
        Err(e) => {
            log::warn!("gamepads unavailable: {e}");
            None
        }
    };
    let mut app = App {
        boot: Some(Box::new(boot)),
        window: None,
        size: Arc::new(AtomicU64::new(0)),
        gilrs,
        keys: HashSet::new(),
        mouse: (0.0, 0.0),
        mouse_buttons: (false, false),
        grabbed: false,
        last_poll: Instant::now(),
        sent: None,
    };
    event_loop.run_app(&mut app).map_err(|e| e.to_string())
}

/// Lets the title see the controllers (called from the kernel hook).
pub fn attach(kernel: &'static Kernel) {
    let _ = KERNEL.set(kernel);
}

struct App {
    boot: Option<Box<dyn FnOnce(RenderWindow)>>,
    window: Option<Arc<Window>>,
    size: Arc<AtomicU64>,
    gilrs: Option<gilrs::Gilrs>,
    keys: HashSet<KeyCode>,
    /// Mouse movement since the last poll, while captured.
    mouse: (f64, f64),
    /// Left and right buttons, while captured.
    mouse_buttons: (bool, bool),
    grabbed: bool,
    last_poll: Instant,
    sent: Option<Gamepad>,
}

fn handles(window: &Window) -> Result<WindowHandles, String> {
    let w = window.window_handle().map_err(|e| e.to_string())?.as_raw();
    let d = window.display_handle().map_err(|e| e.to_string())?.as_raw();
    match (w, d) {
        (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
            Ok(WindowHandles::Wayland { display: d.display.as_ptr() as usize, surface: w.surface.as_ptr() as usize })
        }
        (RawWindowHandle::Xlib(w), RawDisplayHandle::Xlib(d)) => Ok(WindowHandles::Xlib {
            display: d.display.ok_or("no Xlib display")?.as_ptr() as usize,
            window: w.window,
        }),
        (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => Ok(WindowHandles::Xcb {
            connection: d.connection.ok_or("no XCB connection")?.as_ptr() as usize,
            window: w.window.get(),
        }),
        (w, d) => Err(format!("unsupported window system ({w:?}, {d:?})")),
    }
}

impl App {
    fn set_grab(&mut self, grab: bool) {
        let Some(window) = &self.window else { return };
        if grab {
            let ok = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            if !ok {
                return;
            }
            window.set_cursor_visible(false);
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.mouse_buttons = (false, false);
        }
        self.grabbed = grab;
        self.mouse = (0.0, 0.0);
    }

    /// Combines the gamepad, keyboard and mouse and sends the result when it changed.
    fn poll_input(&mut self) {
        let now = Instant::now();
        let frames = now.duration_since(self.last_poll).as_secs_f64() * 60.0;
        self.last_poll = now;
        let axis = |v: f64| (v.clamp(-1.0, 1.0) * STICK).round() as i32;
        let held = |k: KeyCode| self.keys.contains(&k);
        let dir = |neg: KeyCode, pos: KeyCode| f64::from(u8::from(held(pos))) - f64::from(u8::from(held(neg)));

        let mut buttons = 0u16;
        for (key, bit) in KEY_BUTTONS {
            if held(key) {
                buttons |= bit;
            }
        }
        let mut lt = if held(KeyCode::KeyZ) || self.mouse_buttons.1 { 255 } else { 0 };
        let mut rt = if held(KeyCode::KeyC) || self.mouse_buttons.0 { 255 } else { 0 };
        let mut lx = axis(dir(KeyCode::KeyA, KeyCode::KeyD));
        let mut ly = axis(dir(KeyCode::KeyS, KeyCode::KeyW));
        let mut rx = axis(dir(KeyCode::ArrowLeft, KeyCode::ArrowRight));
        let mut ry = axis(dir(KeyCode::ArrowDown, KeyCode::ArrowUp));
        if self.grabbed && (self.mouse.0 != 0.0 || self.mouse.1 != 0.0) {
            let full = MOUSE_FULL_DEFLECTION * frames.max(0.25);
            rx = axis(self.mouse.0 / full);
            ry = axis(-self.mouse.1 / full);
        }
        self.mouse = (0.0, 0.0);

        if let Some(gilrs) = &mut self.gilrs {
            while gilrs.next_event().is_some() {}
            if let Some((_, pad)) = gilrs.gamepads().find(|(_, p)| p.is_connected()) {
                for (button, bit) in PAD_BUTTONS {
                    if pad.is_pressed(button) {
                        buttons |= bit;
                    }
                }
                let trigger = |b| pad.button_data(b).map_or(0, |d| (d.value() * 255.0).round() as i32);
                lt = lt.max(trigger(gilrs::Button::LeftTrigger2));
                rt = rt.max(trigger(gilrs::Button::RightTrigger2));
                // The larger deflection of keyboard/mouse and pad wins (gilrs: up is positive).
                let pick = |a: i32, v: f32| {
                    let b = axis(f64::from(v));
                    if b.abs() > a.abs() { b } else { a }
                };
                lx = pick(lx, pad.value(gilrs::Axis::LeftStickX));
                ly = pick(ly, pad.value(gilrs::Axis::LeftStickY));
                rx = pick(rx, pad.value(gilrs::Axis::RightStickX));
                ry = pick(ry, pad.value(gilrs::Axis::RightStickY));
            }
        }

        let state = Gamepad {
            buttons,
            left_trigger: lt.clamp(0, 255) as u8,
            right_trigger: rt.clamp(0, 255) as u8,
            thumb_lx: lx as i16,
            thumb_ly: ly as i16,
            thumb_rx: rx as i16,
            thumb_ry: ry as i16,
        };
        let Some(kernel) = KERNEL.get() else { return };
        if self.sent.as_ref().is_none_or(|s| !same(s, &state)) {
            kernel.input.update(0, state);
            self.sent = Some(state);
        }
    }
}

fn same(a: &Gamepad, b: &Gamepad) -> bool {
    (a.buttons, a.left_trigger, a.right_trigger, a.thumb_lx, a.thumb_ly, a.thumb_rx, a.thumb_ry)
        == (b.buttons, b.left_trigger, b.right_trigger, b.thumb_lx, b.thumb_ly, b.thumb_rx, b.thumb_ry)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Skyrim (recompiled)")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        // The Wayland app id window managers match rules against; SKYRIM_RECOMP_APP_ID overrides
        // it (test runs use their own so a rule can keep them off the user's workspace).
        let app_id = std::env::var("SKYRIM_RECOMP_APP_ID").unwrap_or_else(|_| "skyrim-recomp".into());
        let attributes =
            winit::platform::wayland::WindowAttributesExtWayland::with_name(attributes, app_id.clone(), app_id);
        let window = match event_loop.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("creating the window: {e}");
                event_loop.exit();
                return;
            }
        };
        let PhysicalSize { width, height } = window.inner_size();
        self.size.store(RenderWindow::pack_size(width, height), Ordering::Relaxed);
        match handles(&window) {
            Ok(handles) => {
                if let Some(boot) = self.boot.take() {
                    boot(RenderWindow { handles, size: self.size.clone() });
                }
            }
            Err(e) => {
                log::error!("window: {e}");
                event_loop.exit();
            }
        }
        self.window = Some(window);
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + INPUT_PERIOD));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.size.store(RenderWindow::pack_size(size.width, size.height), Ordering::Relaxed);
            }
            WindowEvent::Focused(false) => {
                self.keys.clear();
                self.set_grab(false);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let pressed = event.state == ElementState::Pressed;
                if code == KeyCode::F11 && pressed && !event.repeat {
                    if let Some(w) = &self.window {
                        let full = w.fullscreen().is_some();
                        w.set_fullscreen(if full { None } else { Some(Fullscreen::Borderless(None)) });
                    }
                    return;
                }
                // Escape first releases a captured mouse, as in the browser.
                if code == KeyCode::Escape && pressed && self.grabbed {
                    self.set_grab(false);
                    return;
                }
                if pressed {
                    self.keys.insert(code);
                } else {
                    self.keys.remove(&code);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if !self.grabbed {
                    if pressed && button == MouseButton::Left {
                        self.set_grab(true);
                    }
                    return;
                }
                match button {
                    MouseButton::Left => self.mouse_buttons.0 = pressed,
                    MouseButton::Right => self.mouse_buttons.1 = pressed,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.grabbed
        {
            self.mouse.0 += delta.0;
            self.mouse.1 += delta.1;
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.last_poll.elapsed() >= INPUT_PERIOD {
            self.poll_input();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.last_poll + INPUT_PERIOD));
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        crate::exit(0);
    }
}
