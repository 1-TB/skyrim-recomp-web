//! On-screen controller for the touch screen, laid out and behaving like the web page's
//! `touch.js`: floating left and right sticks (touch anywhere on that half, away from the
//! buttons), and the face, shoulder, trigger, stick-click, D-pad, Start and Back buttons.
//!
//! [`Touchpad`] is the state (fed winit's touches, in points); [`Overlay`] draws it with UIKit
//! views that don't take touches, so every touch still reaches winit's view.

use std::collections::HashMap;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::NSString;
use objc2_ui_kit::{
    NSTextAlignment, UIColor, UIFont, UIImpactFeedbackGenerator, UIImpactFeedbackStyle, UILabel, UIView,
};

use crate::pads::bits::*;

/// What a button presses: a button bit or a trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    Bit(u16),
    LeftTrigger,
    RightTrigger,
}

/// Distance from one horizontal and one vertical edge, as fractions of the width and height.
#[derive(Debug, Clone, Copy)]
enum Edge {
    Left(f64),
    Right(f64),
    Top(f64),
    Bottom(f64),
}

struct Button {
    label: &'static str,
    control: Control,
    x: Edge,
    y: Edge,
}

const fn button(label: &'static str, control: Control, x: Edge, y: Edge) -> Button {
    Button { label, control, x, y }
}

use Control::*;
use Edge::*;

/// Same positions as touch.js.
const BUTTONS: [Button; 16] = [
    button("A", Bit(A), Right(0.085), Bottom(0.05)),
    button("B", Bit(B), Right(0.03), Bottom(0.155)),
    button("X", Bit(X), Right(0.14), Bottom(0.155)),
    button("Y", Bit(Y), Right(0.085), Bottom(0.26)),
    button("LB", Bit(LEFT_SHOULDER), Left(0.03), Top(0.14)),
    button("LT", LeftTrigger, Left(0.03), Top(0.03)),
    button("RB", Bit(RIGHT_SHOULDER), Right(0.03), Top(0.14)),
    button("RT", RightTrigger, Right(0.03), Top(0.03)),
    button("LS", Bit(LEFT_THUMB), Left(0.10), Top(0.14)),
    button("RS", Bit(RIGHT_THUMB), Right(0.10), Top(0.14)),
    button("Back", Bit(BACK), Left(0.41), Top(0.03)),
    button("Start", Bit(START), Right(0.41), Top(0.03)),
    button("▲", Bit(DPAD_UP), Left(0.075), Top(0.33)),
    button("▼", Bit(DPAD_DOWN), Left(0.075), Top(0.51)),
    button("◀", Bit(DPAD_LEFT), Left(0.03), Top(0.42)),
    button("▶", Bit(DPAD_RIGHT), Left(0.12), Top(0.42)),
];

/// Face buttons are 7% of the width across, the others 5% (touch.js's min-width).
fn is_face(b: &Button) -> bool {
    matches!(b.control, Bit(A | B | X | Y))
}

/// Fraction of the height a stick deflects fully over.
const STICK_RADIUS: f64 = 0.12;
/// Stick ring diameter as a fraction of the width, and the knob's as a fraction of the ring.
const RING: f64 = 0.16;
const KNOB: f64 = 0.42;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl Rect {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// The buttons' rectangles in an area of `w`×`h` points.
fn layout(w: f64, h: f64) -> Vec<Rect> {
    BUTTONS
        .iter()
        .map(|b| {
            let size = w * if is_face(b) { 0.07 } else { 0.05 };
            let x = match b.x {
                Left(f) => f * w,
                Right(f) => w - f * w - size,
                _ => unreachable!("x is Left or Right"),
            };
            let y = match b.y {
                Top(f) => f * h,
                Bottom(f) => h - f * h - size,
                _ => unreachable!("y is Top or Bottom"),
            };
            Rect { x, y, w: size, h: size }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
enum Pointer {
    /// On a button (None once it slid off every button).
    Button(Option<usize>),
    /// Driving a stick from where it went down.
    Stick { side: Side, origin: (f64, f64) },
}

/// Controller state from the touch screen.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct State {
    pub buttons: u16,
    pub left_trigger: bool,
    pub right_trigger: bool,
    /// Stick deflections, -1 to 1, y up.
    pub left: (f64, f64),
    pub right: (f64, f64),
}

#[derive(Default)]
pub struct Touchpad {
    size: (f64, f64),
    rects: Vec<Rect>,
    pointers: HashMap<u64, Pointer>,
    /// Touches holding each button.
    pressed: [u32; BUTTONS.len()],
    /// The touch driving each stick (one per side: a second touch takes the stick over), and
    /// the stick's centre and knob offset for drawing.
    sticks: [Option<(u64, (f64, f64), (f64, f64))>; 2],
    pub state: State,
    /// A button was newly pressed (for haptic feedback).
    pub pressed_new: bool,
}

impl Touchpad {
    /// The area the controls cover, in points.
    pub fn resize(&mut self, w: f64, h: f64) {
        if self.size != (w, h) {
            self.size = (w, h);
            self.rects = layout(w, h);
        }
    }

    fn hit(&self, x: f64, y: f64) -> Option<usize> {
        self.rects.iter().position(|r| r.contains(x, y))
    }

    fn press(&mut self, button: usize, delta: i32) {
        let n = &mut self.pressed[button];
        if delta > 0 {
            self.pressed_new |= *n == 0;
            *n += 1;
        } else {
            *n = n.saturating_sub(1);
        }
    }

    fn stick_index(side: Side) -> usize {
        if side == Side::Left { 0 } else { 1 }
    }

    pub fn down(&mut self, id: u64, x: f64, y: f64) {
        if let Some(b) = self.hit(x, y) {
            self.pointers.insert(id, Pointer::Button(Some(b)));
            self.press(b, 1);
        } else {
            let side = if x < self.size.0 / 2.0 { Side::Left } else { Side::Right };
            let i = Self::stick_index(side);
            if let Some((old, ..)) = self.sticks[i] {
                self.pointers.remove(&old);
            }
            self.sticks[i] = Some((id, (x, y), (0.0, 0.0)));
            self.pointers.insert(id, Pointer::Stick { side, origin: (x, y) });
        }
        self.recompute();
    }

    pub fn moved(&mut self, id: u64, x: f64, y: f64) {
        let Some(p) = self.pointers.get(&id).copied() else { return };
        match p {
            Pointer::Stick { side, origin } => {
                let radius = (self.size.1 * STICK_RADIUS).max(1.0);
                let (mut dx, mut dy) = ((x - origin.0) / radius, (y - origin.1) / radius);
                let len = dx.hypot(dy);
                if len > 1.0 {
                    (dx, dy) = (dx / len, dy / len);
                }
                if let Some(s) = &mut self.sticks[Self::stick_index(side)] {
                    s.2 = (dx, dy);
                }
            }
            Pointer::Button(on) => {
                // Sliding off a button releases it; sliding onto another presses that one.
                let now = self.hit(x, y);
                if now != on {
                    if let Some(b) = on {
                        self.press(b, -1);
                    }
                    if let Some(b) = now {
                        self.press(b, 1);
                    }
                    self.pointers.insert(id, Pointer::Button(now));
                }
            }
        }
        self.recompute();
    }

    pub fn up(&mut self, id: u64) {
        match self.pointers.remove(&id) {
            Some(Pointer::Button(Some(b))) => self.press(b, -1),
            Some(Pointer::Stick { side, .. }) => self.sticks[Self::stick_index(side)] = None,
            _ => {}
        }
        self.recompute();
    }

    /// Lets go of everything (the controls were hidden, or the app left the foreground).
    pub fn reset(&mut self) {
        self.pointers.clear();
        self.pressed = [0; BUTTONS.len()];
        self.sticks = [None; 2];
        self.recompute();
    }

    fn recompute(&mut self) {
        let mut s = State::default();
        for (b, &n) in BUTTONS.iter().zip(&self.pressed) {
            if n == 0 {
                continue;
            }
            match b.control {
                Bit(bit) => s.buttons |= bit,
                LeftTrigger => s.left_trigger = true,
                RightTrigger => s.right_trigger = true,
            }
        }
        let deflection = |i: usize| self.sticks[i].map_or((0.0, 0.0), |(_, _, (dx, dy))| (dx, -dy));
        s.left = deflection(0);
        s.right = deflection(1);
        self.state = s;
    }
}

const fn color(white: f64, alpha: f64) -> (f64, f64) {
    (white, alpha)
}
const BORDER: (f64, f64) = color(1.0, 0.45);
const FILL: (f64, f64) = color(0.0, 0.25);
const FILL_ON: (f64, f64) = color(1.0, 0.35);
const TEXT: (f64, f64) = color(1.0, 0.85);
const RING_BORDER: (f64, f64) = color(1.0, 0.35);
const KNOB_FILL: (f64, f64) = color(1.0, 0.4);

fn ui_color((white, alpha): (f64, f64)) -> Retained<UIColor> {
    UIColor::colorWithWhite_alpha(white, alpha)
}

fn rect(x: f64, y: f64, w: f64, h: f64) -> CGRect {
    CGRect { origin: CGPoint { x, y }, size: CGSize { width: w, height: h } }
}

/// A view that draws only: touches pass through to the view behind it.
fn passive_view(mtm: MainThreadMarker) -> Retained<UIView> {
    let view = UIView::new(mtm);
    view.setUserInteractionEnabled(false);
    view
}

fn circle(view: &UIView, border: Option<(f64, f64)>, fill: (f64, f64), radius: f64) {
    view.setBackgroundColor(Some(&ui_color(fill)));
    let layer = view.layer();
    layer.setCornerRadius(radius);
    if let Some(border) = border {
        layer.setBorderWidth(2.0);
        // SAFETY: the color is retained for the call; the layer copies it.
        let cg = unsafe { ui_color(border).CGColor() };
        layer.setBorderColor(Some(&cg));
    }
}

/// The controls' views, inside winit's view.
pub struct Overlay {
    root: Retained<UIView>,
    buttons: Vec<(Retained<UIView>, Retained<UILabel>)>,
    rings: [(Retained<UIView>, Retained<UIView>); 2],
    haptics: Retained<UIImpactFeedbackGenerator>,
    drawn: Option<DrawnState>,
    /// The frame the controls cover (the view's safe area), in points.
    pub area: CGRect,
}

#[derive(Clone, PartialEq)]
struct DrawnState {
    size: (f64, f64),
    pressed: Vec<bool>,
    sticks: [Option<((f64, f64), (f64, f64))>; 2],
    visible: bool,
}

impl Overlay {
    pub fn new(mtm: MainThreadMarker, parent: &UIView) -> Self {
        let root = passive_view(mtm);
        root.setHidden(true);
        parent.addSubview(&root);
        let buttons = BUTTONS
            .iter()
            .map(|b| {
                let view = passive_view(mtm);
                let label = UILabel::new(mtm);
                label.setText(Some(&NSString::from_str(b.label)));
                // SAFETY: on the main thread (the label is main-thread only).
                unsafe { label.setTextColor(Some(&ui_color(TEXT))) };
                label.setTextAlignment(NSTextAlignment::Center);
                label.setUserInteractionEnabled(false);
                view.addSubview(&label);
                root.addSubview(&view);
                (view, label)
            })
            .collect();
        let rings = [(); 2].map(|()| {
            let ring = passive_view(mtm);
            let knob = passive_view(mtm);
            ring.addSubview(&knob);
            ring.setHidden(true);
            root.addSubview(&ring);
            (ring, knob)
        });
        let haptics =
            UIImpactFeedbackGenerator::feedbackGeneratorWithStyle_forView(UIImpactFeedbackStyle::Light, parent);
        Overlay { root, buttons, rings, haptics, drawn: None, area: rect(0.0, 0.0, 0.0, 0.0) }
    }

    /// Places the controls in `area` (points, in the parent view's coordinates).
    pub fn set_area(&mut self, area: CGRect) {
        self.area = area;
        self.root.setFrame(area);
    }

    /// Redraws what changed since the last call.
    pub fn draw(&mut self, pad: &mut Touchpad, visible: bool) {
        if std::mem::take(&mut pad.pressed_new) && visible {
            self.haptics.impactOccurred();
        }
        let state = DrawnState {
            size: pad.size,
            pressed: pad.pressed.iter().map(|&n| n > 0).collect(),
            sticks: pad.sticks.map(|s| s.map(|(_, at, knob)| (at, knob))),
            visible,
        };
        let previous = self.drawn.replace(state.clone());
        if previous.as_ref() == Some(&state) {
            return;
        }
        self.root.setHidden(!visible);
        let (w, _) = state.size;
        let resized = previous.as_ref().is_none_or(|p| p.size != state.size);
        for (i, ((view, label), r)) in self.buttons.iter().zip(&pad.rects).enumerate() {
            if resized {
                view.setFrame(rect(r.x, r.y, r.w, r.h));
                label.setFrame(rect(0.0, 0.0, r.w, r.h));
                let face = is_face(&BUTTONS[i]);
                // touch.js: clamp(9px, 1.8vw, 16px) for face buttons, (8px, 1.5vw, 14px) else.
                let pt = if face { (w * 0.018).clamp(9.0, 16.0) } else { (w * 0.015).clamp(8.0, 14.0) };
                // SAFETY: a system font, on the main thread.
                unsafe { label.setFont(Some(&UIFont::boldSystemFontOfSize(pt))) };
                circle(view, Some(BORDER), FILL, if face { r.w / 2.0 } else { r.w * 0.3 });
            }
            view.setBackgroundColor(Some(&ui_color(if state.pressed[i] { FILL_ON } else { FILL })));
        }
        let ring_size = w * RING;
        let knob_size = ring_size * KNOB;
        for ((ring, knob), stick) in self.rings.iter().zip(&state.sticks) {
            let Some(((x, y), (dx, dy))) = *stick else {
                ring.setHidden(true);
                continue;
            };
            if resized || ring.isHidden() {
                circle(ring, Some(RING_BORDER), (0.0, 0.0), ring_size / 2.0);
                circle(knob, None, KNOB_FILL, knob_size / 2.0);
            }
            ring.setFrame(rect(x - ring_size / 2.0, y - ring_size / 2.0, ring_size, ring_size));
            let centre = ring_size / 2.0;
            let (kx, ky) = (centre + dx * centre, centre + dy * centre);
            knob.setFrame(rect(kx - knob_size / 2.0, ky - knob_size / 2.0, knob_size, knob_size));
            ring.setHidden(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad() -> Touchpad {
        let mut p = Touchpad::default();
        p.resize(900.0, 400.0);
        p
    }

    fn centre(p: &Touchpad, label: &str) -> (f64, f64) {
        let i = BUTTONS.iter().position(|b| b.label == label).unwrap();
        let r = p.rects[i];
        (r.x + r.w / 2.0, r.y + r.h / 2.0)
    }

    #[test]
    fn buttons_press_and_slide() {
        let mut p = pad();
        let a = centre(&p, "A");
        p.down(1, a.0, a.1);
        assert_eq!(p.state.buttons, A);
        let b = centre(&p, "B");
        p.moved(1, b.0, b.1);
        assert_eq!(p.state.buttons, B);
        p.moved(1, 450.0, 200.0);
        assert_eq!(p.state.buttons, 0);
        p.up(1);
        let rt = centre(&p, "RT");
        p.down(2, rt.0, rt.1);
        assert!(p.state.right_trigger && !p.state.left_trigger);
        p.up(2);
        assert_eq!(p.state, State::default());
    }

    #[test]
    fn sticks_float_and_clamp() {
        let mut p = pad();
        p.down(1, 300.0, 300.0);
        // Half the full radius (12% of the height) to the right and up.
        p.moved(1, 324.0, 276.0);
        assert!((p.state.left.0 - 0.5).abs() < 1e-9 && (p.state.left.1 - 0.5).abs() < 1e-9);
        p.down(2, 600.0, 300.0);
        p.moved(2, 600.0, 900.0);
        assert!((p.state.right.1 + 1.0).abs() < 1e-9, "clamped to full deflection down");
        // A second touch on the left takes the left stick over.
        p.down(3, 100.0, 350.0);
        assert_eq!(p.state.left, (0.0, 0.0));
        p.up(1);
        p.moved(3, 100.0, 302.0);
        assert!((p.state.left.1 - 1.0).abs() < 1e-9);
        p.reset();
        assert_eq!(p.state, State::default());
    }
}
