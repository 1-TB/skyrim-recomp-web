//! The settings panel shown at launch, before the title boots (resolution scale and the other
//! renderer options only take effect when the renderer starts). The choices are kept in
//! `settings.txt` in Documents and become command-line options ahead of `args.txt`'s, so
//! `args.txt` can still override them.

use std::path::Path;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, MainThreadOnly, sel};
use objc2_core_foundation::CGRect;
use objc2_foundation::{NSArray, NSString};
use objc2_ui_kit::{
    NSTextAlignment, UIButton, UIButtonType, UIColor, UIControlEvents, UIControlState, UIFont, UILabel,
    UISegmentedControl, UISwitch, UIView,
};

use crate::app::rect;

/// What the panel offers, and the options each choice becomes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Draw at this many times the console's 1280×720 along each axis (1-3).
    pub resolution_scale: u32,
    /// Least anisotropic filtering (1 = the title's own).
    pub anisotropy: u16,
    /// Frame rate cap (30, 60 or 120).
    pub fps_cap: u32,
    pub show_fps: bool,
    /// Scale frames up to the screen with MetalFX rather than bicubic filtering.
    pub metalfx: bool,
    /// The title's 4x multisample anti-aliasing (off: faster).
    pub msaa: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { resolution_scale: 1, anisotropy: 1, fps_cap: 120, show_fps: false, metalfx: true, msaa: true }
    }
}

const SCALES: [u32; 3] = [1, 2, 3];
const ANISOTROPY: [u16; 4] = [1, 4, 8, 16];
const FPS_CAPS: [u32; 3] = [30, 60, 120];

impl Settings {
    /// Reads `path` (`key=value` lines); missing or unknown values keep their defaults.
    pub fn load(path: &Path) -> Settings {
        let mut s = Settings::default();
        let Ok(text) = std::fs::read_to_string(path) else { return s };
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else { continue };
            let value = value.trim();
            match key.trim() {
                "resolution_scale" => {
                    if let Some(v) = value.parse().ok().filter(|v| SCALES.contains(v)) {
                        s.resolution_scale = v;
                    }
                }
                "anisotropy" => {
                    if let Some(v) = value.parse().ok().filter(|v| ANISOTROPY.contains(v)) {
                        s.anisotropy = v;
                    }
                }
                "fps_cap" => {
                    if let Some(v) = value.parse().ok().filter(|v| FPS_CAPS.contains(v)) {
                        s.fps_cap = v;
                    }
                }
                "show_fps" => s.show_fps = value == "true",
                "metalfx" => s.metalfx = value == "true",
                "msaa" => s.msaa = value != "false",
                _ => {}
            }
        }
        s
    }

    pub fn save(&self, path: &Path) {
        let text = format!(
            "resolution_scale={}\nanisotropy={}\nfps_cap={}\nshow_fps={}\nmetalfx={}\nmsaa={}\n",
            self.resolution_scale, self.anisotropy, self.fps_cap, self.show_fps, self.metalfx, self.msaa
        );
        if let Err(e) = std::fs::write(path, text) {
            log::warn!("saving {}: {e}", path.display());
        }
    }

    /// The command-line options for these settings.
    pub fn args(&self) -> Vec<String> {
        // The title presents every second vblank.
        let vblank = self.fps_cap * 2;
        let mut args: Vec<String> = [
            ("--resolution-scale", self.resolution_scale.to_string()),
            ("--anisotropy", self.anisotropy.to_string()),
            ("--vblank-hz", vblank.to_string()),
        ]
        .into_iter()
        .flat_map(|(flag, value)| [flag.to_string(), value])
        .collect();
        if self.metalfx {
            args.push("--metalfx".to_string());
        }
        if !self.msaa {
            args.push("--no-msaa".to_string());
        }
        args
    }
}

/// The panel's views. Its controls are read when Play is tapped.
pub struct Panel {
    root: Retained<UIView>,
    title: Retained<UILabel>,
    /// Each row's label and control, and whether the control is a switch (sized as such).
    rows: Vec<(Retained<UILabel>, Retained<UIView>, bool)>,
    note: Retained<UILabel>,
    play: Retained<UIButton>,
    scale: Retained<UISegmentedControl>,
    anisotropy: Retained<UISegmentedControl>,
    fps_cap: Retained<UISegmentedControl>,
    upscaling: Retained<UISegmentedControl>,
    msaa: Retained<UISegmentedControl>,
    show_fps: Retained<UISwitch>,
}

fn label(mtm: MainThreadMarker, text: &str, size: f64, alpha: f64) -> Retained<UILabel> {
    let label = UILabel::new(mtm);
    label.setText(Some(&NSString::from_str(text)));
    // SAFETY: a color and a system font, on the main thread.
    unsafe {
        label.setTextColor(Some(&UIColor::colorWithWhite_alpha(1.0, alpha)));
        label.setFont(Some(&UIFont::systemFontOfSize(size)));
    }
    label
}

fn segments(mtm: MainThreadMarker, titles: &[&str], selected: usize) -> Retained<UISegmentedControl> {
    let items: Vec<Retained<AnyObject>> =
        titles.iter().map(|t| Retained::into_super(Retained::into_super(NSString::from_str(t)))).collect();
    let array = NSArray::from_retained_slice(&items);
    // SAFETY: the items are NSStrings, as the initializer expects.
    let control = unsafe { UISegmentedControl::initWithItems(UISegmentedControl::alloc(mtm), Some(&array)) };
    control.setSelectedSegmentIndex(selected as isize);
    control
}

impl Panel {
    /// Builds the panel in `parent` showing `settings`; Play sends `settingsPlay:` to `target`.
    pub fn new(mtm: MainThreadMarker, parent: &UIView, settings: &Settings, target: &AnyObject) -> Panel {
        let root = UIView::new(mtm);
        root.setBackgroundColor(Some(&UIColor::colorWithWhite_alpha(0.08, 0.96)));
        let index = |list: &[u32], v: u32| list.iter().position(|&x| x == v).unwrap_or(0);
        let scale = segments(mtm, &["720p", "1440p (2×)", "2160p (3×)"], index(&SCALES, settings.resolution_scale));
        let anisotropy = segments(
            mtm,
            &["Game default", "4×", "8×", "16×"],
            ANISOTROPY.iter().position(|&x| x == settings.anisotropy).unwrap_or(0),
        );
        let fps_cap = segments(mtm, &["30", "60", "120"], index(&FPS_CAPS, settings.fps_cap));
        let upscaling = segments(mtm, &["Bicubic", "MetalFX"], usize::from(settings.metalfx));
        let msaa = segments(mtm, &["Off (faster)", "4× MSAA"], usize::from(settings.msaa));
        let show_fps = UISwitch::new(mtm);
        show_fps.setOn(settings.show_fps);
        let control = |c: Retained<UISegmentedControl>| Retained::into_super(Retained::into_super(c));
        let rows: Vec<(Retained<UILabel>, Retained<UIView>, bool)> = vec![
            (label(mtm, "Resolution", 17.0, 0.9), control(scale.clone()), false),
            (label(mtm, "Texture filtering", 17.0, 0.9), control(anisotropy.clone()), false),
            (label(mtm, "Frame rate limit", 17.0, 0.9), control(fps_cap.clone()), false),
            (label(mtm, "Upscaling", 17.0, 0.9), control(upscaling.clone()), false),
            (label(mtm, "Anti-aliasing", 17.0, 0.9), control(msaa.clone()), false),
            (label(mtm, "Show FPS", 17.0, 0.9), Retained::into_super(Retained::into_super(show_fps.clone())), true),
        ];
        let title = label(mtm, "Settings", 28.0, 1.0);
        let note = label(
            mtm,
            "Higher resolutions are smoother and sharper but cost frame rate. Close the app to change \
             these again; args.txt in Documents still overrides them.",
            13.0,
            0.6,
        );
        note.setNumberOfLines(0);
        let play = UIButton::buttonWithType(UIButtonType::System, mtm);
        play.setTitle_forState(Some(&NSString::from_str("Play")), UIControlState::Normal);
        // SAFETY: colors and fonts on the main thread; the target outlives the button (it is the
        // game view, which lives as long as the process) and implements `settingsPlay:`.
        unsafe {
            play.setBackgroundColor(Some(&UIColor::colorWithRed_green_blue_alpha(0.85, 0.65, 0.25, 1.0)));
            play.setTitleColor_forState(Some(&UIColor::blackColor()), UIControlState::Normal);
            if let Some(l) = play.titleLabel() {
                l.setFont(Some(&UIFont::boldSystemFontOfSize(22.0)));
            }
            play.layer().setCornerRadius(12.0);
            play.addTarget_action_forControlEvents(Some(target), sel!(settingsPlay:), UIControlEvents::TouchUpInside);
        }
        root.addSubview(&title);
        for (l, c, _) in &rows {
            root.addSubview(l);
            root.addSubview(c);
        }
        root.addSubview(&note);
        root.addSubview(&play);
        title.setTextAlignment(NSTextAlignment::Left);
        parent.addSubview(&root);
        Panel { root, title, rows, note, play, scale, anisotropy, fps_cap, upscaling, msaa, show_fps }
    }

    /// Covers the whole view, with the content laid out in `area` (the safe area).
    pub fn layout(&self, bounds: CGRect, area: CGRect) {
        self.root.setFrame(bounds);
        let width = area.size.width.min(720.0);
        let x = area.origin.x + (area.size.width - width) / 2.0;
        let rows = self.rows.len() as f64;
        // Rows shrink to fit a phone held sideways (about 400 points high).
        let row = ((area.size.height - 40.0 - 40.0 - 56.0 - 4.0 * (rows + 3.0)) / rows).clamp(32.0, 44.0);
        let gap = ((area.size.height - 40.0 - row * rows - 40.0 - 56.0) / (rows + 3.0)).clamp(2.0, 18.0);
        let mut y = area.origin.y + gap;
        self.title.setFrame(rect(x, y, width, 40.0));
        y += 40.0 + gap;
        let label_w = width * 0.32;
        for (label, control, switch) in &self.rows {
            label.setFrame(rect(x, y, label_w, row));
            let (control_w, control_h) = if *switch { (51.0, 31.0) } else { (width - label_w, (row - 4.0).min(34.0)) };
            control.setFrame(rect(x + label_w, y + (row - control_h) / 2.0, control_w, control_h));
            y += row + gap;
        }
        self.note.setFrame(rect(x, y, width, 40.0));
        y += 40.0 + gap;
        self.play.setFrame(rect(x + width * 0.3, y, width * 0.4, 56.0));
    }

    /// The settings the controls show.
    pub fn settings(&self) -> Settings {
        let pick = |c: &UISegmentedControl| c.selectedSegmentIndex().max(0) as usize;
        Settings {
            resolution_scale: SCALES[pick(&self.scale).min(SCALES.len() - 1)],
            anisotropy: ANISOTROPY[pick(&self.anisotropy).min(ANISOTROPY.len() - 1)],
            fps_cap: FPS_CAPS[pick(&self.fps_cap).min(FPS_CAPS.len() - 1)],
            show_fps: self.show_fps.isOn(),
            metalfx: self.upscaling.selectedSegmentIndex() == 1,
            msaa: self.msaa.selectedSegmentIndex() == 1,
        }
    }

    pub fn remove(&self) {
        self.root.removeFromSuperview();
    }
}

/// The frame-rate counter in the top corner (frames the title presented per second).
pub struct FpsLabel {
    label: Retained<UILabel>,
    last: Option<(std::time::Instant, u64)>,
}

impl FpsLabel {
    pub fn new(mtm: MainThreadMarker, parent: &UIView) -> FpsLabel {
        let label = label(mtm, "", 15.0, 0.95);
        // SAFETY: colors and fonts on the main thread.
        unsafe {
            label.setBackgroundColor(Some(&UIColor::colorWithWhite_alpha(0.0, 0.45)));
            label.setFont(Some(&UIFont::monospacedDigitSystemFontOfSize_weight(15.0, 0.3)));
        }
        label.setTextAlignment(NSTextAlignment::Center);
        label.setUserInteractionEnabled(false);
        parent.addSubview(&label);
        FpsLabel { label, last: None }
    }

    pub fn layout(&self, area: CGRect) {
        self.label.setFrame(rect(area.origin.x + 8.0, area.origin.y + 8.0, 84.0, 26.0));
    }

    /// Updates the text about twice a second from the title's swap count.
    pub fn update(&mut self, swaps: u64) {
        let now = std::time::Instant::now();
        match self.last {
            Some((t, n)) if now.duration_since(t).as_secs_f64() >= 0.5 => {
                let fps = (swaps.saturating_sub(n)) as f64 / now.duration_since(t).as_secs_f64();
                self.label.setText(Some(&NSString::from_str(&format!("{fps:.0} fps"))));
                self.last = Some((now, swaps));
            }
            Some(_) => {}
            None => self.last = Some((now, swaps)),
        }
    }
}
