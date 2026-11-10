//! The cheats panel, opened from the in-game settings: kernel-hle's cheats (cheats.rs there)
//! as buttons. The game is paused while it shows, so what is asked for takes effect on Resume.

use kernel_hle::cheats::{self, Cheat};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, sel};
use objc2_core_foundation::CGRect;
use objc2_foundation::NSString;
use objc2_ui_kit::{
    UIButton, UIButtonType, UIColor, UIControlEvents, UIControlState, UIFont, UILabel, UISegmentedControl, UISwitch,
    UIView,
};

use crate::app::rect;
use crate::settings::{label, segments};

/// The buttons: each asks for its cheat once per tap.
const ACTIONS: [(&str, Cheat); 6] = [
    ("+1,000 gold", Cheat::Gold(1000)),
    ("+10 lockpicks", Cheat::Lockpicks(10)),
    ("Restore health", Cheat::Heal),
    ("Carry weight +500", Cheat::CarryWeight(500.0)),
    ("+1 perk point", Cheat::PerkPoints(1)),
    ("All skills +1", Cheat::SkillLevels(1)),
];

/// Run speeds offered, in percent.
const SPEEDS: [u32; 3] = [100, 150, 200];

pub struct Panel {
    root: Retained<UIView>,
    title: Retained<UILabel>,
    god_label: Retained<UILabel>,
    god: Retained<UISwitch>,
    speed_label: Retained<UILabel>,
    speed: Retained<UISegmentedControl>,
    buttons: Vec<Retained<UIButton>>,
    note: Retained<UILabel>,
    done: Retained<UIButton>,
    /// How often each button was tapped since the panel opened (shown in the note).
    taps: [u32; ACTIONS.len()],
}

fn button(mtm: MainThreadMarker, title: &str, gold: bool) -> Retained<UIButton> {
    let button = UIButton::buttonWithType(UIButtonType::System, mtm);
    button.setTitle_forState(Some(&NSString::from_str(title)), UIControlState::Normal);
    // SAFETY: colors and fonts on the main thread.
    unsafe {
        if gold {
            button.setBackgroundColor(Some(&UIColor::colorWithRed_green_blue_alpha(0.85, 0.65, 0.25, 1.0)));
            button.setTitleColor_forState(Some(&UIColor::blackColor()), UIControlState::Normal);
        } else {
            button.setBackgroundColor(Some(&UIColor::colorWithWhite_alpha(1.0, 0.14)));
            button.setTitleColor_forState(Some(&UIColor::whiteColor()), UIControlState::Normal);
        }
        if let Some(l) = button.titleLabel() {
            let font = if gold { UIFont::boldSystemFontOfSize(20.0) } else { UIFont::systemFontOfSize(17.0) };
            l.setFont(Some(&font));
        }
        button.layer().setCornerRadius(10.0);
    }
    button
}

impl Panel {
    /// Builds the panel in `parent`, over everything in it. Its controls send `cheatGod:`,
    /// `cheatSpeed:`, `cheatTapped:` (the sender's tag is the button's index) and
    /// `cheatsDone:` to `target`.
    pub fn new(mtm: MainThreadMarker, parent: &UIView, target: &AnyObject) -> Panel {
        let root = UIView::new(mtm);
        root.setBackgroundColor(Some(&UIColor::colorWithWhite_alpha(0.08, 1.0)));
        let title = label(mtm, "Cheats", 28.0, 1.0);
        let god_label = label(mtm, "God mode (no loss of health, magicka or stamina)", 17.0, 0.9);
        let god = UISwitch::new(mtm);
        god.setOn(cheats::god_mode());
        let speed_label = label(mtm, "Run speed", 17.0, 0.9);
        let speed = segments(
            mtm,
            &["Normal", "1.5×", "2×"],
            SPEEDS.iter().position(|&s| s == cheats::run_speed()).unwrap_or(0),
        );
        let buttons: Vec<Retained<UIButton>> = ACTIONS
            .iter()
            .enumerate()
            .map(|(i, (name, _))| {
                let b = button(mtm, name, false);
                b.setTag(i as isize);
                // SAFETY: the target (the game view) lives as long as the process and
                // implements the action.
                unsafe {
                    b.addTarget_action_forControlEvents(
                        Some(target),
                        sel!(cheatTapped:),
                        UIControlEvents::TouchUpInside,
                    )
                };
                b
            })
            .collect();
        let note = label(mtm, "", 13.0, 0.6);
        note.setNumberOfLines(0);
        let done = button(mtm, "Done", true);
        // SAFETY: as for the buttons above.
        unsafe {
            god.addTarget_action_forControlEvents(Some(target), sel!(cheatGod:), UIControlEvents::ValueChanged);
            speed.addTarget_action_forControlEvents(Some(target), sel!(cheatSpeed:), UIControlEvents::ValueChanged);
            done.addTarget_action_forControlEvents(Some(target), sel!(cheatsDone:), UIControlEvents::TouchUpInside);
        }
        root.addSubview(&title);
        root.addSubview(&god_label);
        root.addSubview(&god);
        root.addSubview(&speed_label);
        root.addSubview(&speed);
        for b in &buttons {
            root.addSubview(b);
        }
        root.addSubview(&note);
        root.addSubview(&done);
        parent.addSubview(&root);
        let panel =
            Panel { root, title, god_label, god, speed_label, speed, buttons, note, done, taps: [0; ACTIONS.len()] };
        panel.show_note();
        panel
    }

    /// Covers the whole view, with the content laid out in `area` (the safe area).
    pub fn layout(&self, bounds: CGRect, area: CGRect) {
        self.root.setFrame(bounds);
        let width = area.size.width.min(720.0);
        let x = area.origin.x + (area.size.width - width) / 2.0;
        // Title, two control rows, three rows of two buttons, the note and Done: rows shrink
        // to fit a phone held sideways (about 400 points high).
        let rows = 5.0;
        let row = ((area.size.height - 40.0 - 44.0 - 52.0 - 4.0 * (rows + 4.0)) / rows).clamp(28.0, 46.0);
        let gap = ((area.size.height - 40.0 - row * rows - 44.0 - 52.0) / (rows + 4.0)).clamp(2.0, 16.0);
        let mut y = area.origin.y + gap;
        self.title.setFrame(rect(x, y, width, 40.0));
        y += 40.0 + gap;
        self.god_label.setFrame(rect(x, y, width - 60.0, row));
        self.god.setFrame(rect(x + width - 51.0, y + (row - 31.0) / 2.0, 51.0, 31.0));
        y += row + gap;
        let label_w = width * 0.32;
        self.speed_label.setFrame(rect(x, y, label_w, row));
        let control_h = (row - 4.0).min(34.0);
        self.speed.setFrame(rect(x + label_w, y + (row - control_h) / 2.0, width - label_w, control_h));
        y += row + gap;
        let button_w = (width - 12.0) / 2.0;
        for (i, b) in self.buttons.iter().enumerate() {
            let column = (i % 2) as f64;
            b.setFrame(rect(x + column * (button_w + 12.0), y, button_w, row));
            if i % 2 == 1 || i + 1 == self.buttons.len() {
                y += row + gap;
            }
        }
        self.note.setFrame(rect(x, y, width, 44.0));
        y += 44.0 + gap;
        self.done.setFrame(rect(x + width * 0.3, y, width * 0.4, 52.0));
    }

    /// The God mode switch changed: on also restores what was lost.
    pub fn god_changed(&self) {
        let on = self.god.isOn();
        cheats::set_god_mode(on);
        if on {
            cheats::request(Cheat::Heal);
        }
    }

    pub fn speed_changed(&self) {
        let index = self.speed.selectedSegmentIndex().max(0) as usize;
        cheats::request(Cheat::RunSpeed(SPEEDS[index.min(SPEEDS.len() - 1)] as f32));
    }

    /// Button `index` was tapped.
    pub fn tapped(&mut self, index: usize) {
        let Some((_, cheat)) = ACTIONS.get(index) else { return };
        cheats::request(*cheat);
        self.taps[index] += 1;
        self.show_note();
    }

    fn show_note(&self) {
        let asked: Vec<String> = ACTIONS
            .iter()
            .zip(self.taps)
            .filter(|(_, n)| *n > 0)
            .map(|((name, _), n)| if n > 1 { format!("{name} ×{n}") } else { name.to_string() })
            .collect();
        let text = if asked.is_empty() {
            "Cheats change your character like the PC console's commands do, and are saved with the game. \
             They take effect when you resume."
                .to_string()
        } else {
            format!("On resume: {}.", asked.join(", "))
        };
        self.note.setText(Some(&NSString::from_str(&text)));
    }

    pub fn remove(&self) {
        self.root.removeFromSuperview();
    }
}
