//! The app's UIKit side, on the main thread. UIKit in the iOS 27 SDK requires the scene
//! lifecycle (an app without it is stopped at launch), which winit doesn't implement, so the few
//! classes involved are defined here:
//!
//! * `SkyrtAppDelegate`, the application delegate; Info.plist's scene manifest names
//! * `SkyrtSceneDelegate`, which makes the window when the scene connects and pauses the
//!   renderer while the scene is inactive (iOS aborts GPU work submitted from the background);
//! * `SkyrtViewController`: landscape, no status bar, home indicator hidden, edge swipes go to
//!   the game first;
//! * `SkyrtView`, whose layer is the `CAMetalLayer` the renderer presents into, and which gets
//!   the touches. A display link calls its `tick:` every frame: controller state (touch
//!   screen, gamepad, hardware keyboard) is merged and sent to the title, and the touch controls
//!   and status line are redrawn.
//!
//! The settings panel (settings.rs) covers the view at launch; its Play button boots the title.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use kernel_hle::Kernel;
use kernel_hle::input::Gamepad;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, NSObjectProtocol};
use objc2::{ClassType, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::{NSRunLoop, NSRunLoopCommonModes, NSSet, NSString};
use objc2_quartz_core::{CADisplayLink, CAFrameRateRange, CAMetalLayer};
use objc2_ui_kit::{
    NSTextAlignment, UIApplication, UIApplicationDelegate, UIColor, UIEvent, UIFont, UIInterfaceOrientationMask, UILabel,
    UIRectEdge, UIResponder, UIScene, UISceneConnectionOptions, UISceneDelegate, UISceneSession, UITouch, UIView,
    UIViewController, UIWindow, UIWindowScene, UIWindowSceneDelegate,
};
use xenos_render::native::{Window as RenderWindow, WindowHandles};

use crate::pads::{self, Reading};
use crate::settings::{FpsLabel, Panel, Settings};
use crate::touch::{Overlay, Touchpad};

/// The kernel, once the title has booted (input goes nowhere before that).
static KERNEL: OnceLock<&'static Kernel> = OnceLock::new();

/// Lets the title see the controllers (called from the kernel hook).
pub fn attach(kernel: &'static Kernel) {
    let _ = KERNEL.set(kernel);
}

type Boot = Box<dyn FnOnce(RenderWindow, Settings)>;

/// Main-thread state: everything UIKit's callbacks share.
struct State {
    boot: Option<Boot>,
    /// Where the settings are kept.
    settings_path: std::path::PathBuf,
    /// The settings panel, until Play is tapped.
    panel: Option<Panel>,
    /// Start with the saved settings, without showing the panel (`--autoplay` in args.txt).
    autoplay: bool,
    /// The renderer's window, handed to `boot` on Play.
    render_window: Option<RenderWindow>,
    fps: Option<FpsLabel>,
    /// Drawable size in pixels, read by the renderer thread.
    size: Arc<AtomicU64>,
    /// Kept alive: the scene only holds its window weakly.
    window: Option<Retained<UIWindow>>,
    overlay: Option<Overlay>,
    status: Option<Retained<UILabel>>,
    touch: Touchpad,
    sent: Option<Gamepad>,
    status_seen: u64,
    /// A gamepad was connected at the last poll (the touch controls hide while one is).
    pad_connected: bool,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State {
        boot: None,
        settings_path: std::path::PathBuf::new(),
        panel: None,
        autoplay: false,
        render_window: None,
        fps: None,
        size: Arc::new(AtomicU64::new(0)),
        window: None,
        overlay: None,
        status: None,
        touch: Touchpad::default(),
        sent: None,
        status_seen: 0,
        pad_connected: false,
    });
}

/// Runs `f` on the main-thread state, or not at all if a caller further up the stack holds it
/// (UIKit can call back into us from inside a call we make, e.g. a layout).
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| s.try_borrow_mut().ok().map(|mut s| f(&mut s)))
}

/// Runs UIKit on this (the main) thread; never returns. `boot` gets the window for the renderer
/// and the settings chosen (kept in `settings_path`) once Play is tapped (at once with
/// `autoplay`), and must start the title on another thread.
pub fn run(
    settings_path: std::path::PathBuf,
    autoplay: bool,
    boot: impl FnOnce(RenderWindow, Settings) + 'static,
) -> Result<(), String> {
    let mtm = MainThreadMarker::new().ok_or("not on the main thread")?;
    with_state(|s| {
        s.boot = Some(Box::new(boot));
        s.settings_path = settings_path;
        s.autoplay = autoplay;
    });
    // Registered before UIKit looks them up by name (the scene delegate through Info.plist).
    let _ = (SkyrtSceneDelegate::class(), SkyrtViewController::class(), SkyrtView::class());
    let delegate = NSString::from_str(&SkyrtAppDelegate::class().name().to_string_lossy());
    UIApplication::main(None, Some(&delegate), mtm)
}

pub(crate) fn rect(x: f64, y: f64, w: f64, h: f64) -> CGRect {
    CGRect { origin: CGPoint { x, y }, size: CGSize { width: w, height: h } }
}

define_class!(
    #[unsafe(super(UIResponder))]
    #[thread_kind = MainThreadOnly]
    #[name = "SkyrtAppDelegate"]
    struct SkyrtAppDelegate;

    unsafe impl NSObjectProtocol for SkyrtAppDelegate {}

    unsafe impl UIApplicationDelegate for SkyrtAppDelegate {}
);

define_class!(
    #[unsafe(super(UIResponder))]
    #[thread_kind = MainThreadOnly]
    #[name = "SkyrtSceneDelegate"]
    struct SkyrtSceneDelegate;

    unsafe impl NSObjectProtocol for SkyrtSceneDelegate {}

    unsafe impl UISceneDelegate for SkyrtSceneDelegate {
        #[unsafe(method(scene:willConnectToSession:options:))]
        fn scene_will_connect(&self, scene: &UIScene, _session: &UISceneSession, _options: &UISceneConnectionOptions) {
            match scene.downcast_ref::<UIWindowScene>() {
                Some(scene) => connect(self.mtm(), scene),
                None => log::error!("the scene is not a window scene"),
            }
        }

        #[unsafe(method(sceneDidBecomeActive:))]
        fn scene_did_become_active(&self, _scene: &UIScene) {
            log::info!("app active; renderer running");
            xenos_render::native::set_paused(false);
        }

        #[unsafe(method(sceneWillResignActive:))]
        fn scene_will_resign_active(&self, _scene: &UIScene) {
            log::info!("app inactive; pausing the renderer");
            xenos_render::native::set_paused(true);
            with_state(|s| s.touch.reset());
        }
    }

    unsafe impl UIWindowSceneDelegate for SkyrtSceneDelegate {}
);

define_class!(
    #[unsafe(super(UIViewController))]
    #[thread_kind = MainThreadOnly]
    #[name = "SkyrtViewController"]
    struct SkyrtViewController;

    impl SkyrtViewController {
        #[unsafe(method(loadView))]
        fn load_view(&self) {
            self.setView(Some(&SkyrtView::new(self.mtm())));
        }

        #[unsafe(method(prefersStatusBarHidden))]
        fn prefers_status_bar_hidden(&self) -> bool {
            true
        }

        #[unsafe(method(prefersHomeIndicatorAutoHidden))]
        fn prefers_home_indicator_auto_hidden(&self) -> bool {
            true
        }

        /// Swipes from the edges reach the game first; a second swipe still opens the system UI.
        #[unsafe(method(preferredScreenEdgesDeferringSystemGestures))]
        fn preferred_screen_edges_deferring_system_gestures(&self) -> UIRectEdge {
            UIRectEdge::All
        }

        #[unsafe(method(supportedInterfaceOrientations))]
        fn supported_interface_orientations(&self) -> UIInterfaceOrientationMask {
            UIInterfaceOrientationMask::Landscape
        }
    }
);

define_class!(
    #[unsafe(super(UIView))]
    #[thread_kind = MainThreadOnly]
    #[name = "SkyrtView"]
    struct SkyrtView;

    impl SkyrtView {
        /// The view's own layer is the CAMetalLayer, so UIKit keeps it sized and scaled.
        #[unsafe(method(layerClass))]
        fn layer_class() -> &'static AnyClass {
            CAMetalLayer::class()
        }

        #[unsafe(method(layoutSubviews))]
        fn layout_subviews(&self) {
            // SAFETY: the superclass's implementation of this very method.
            let _: () = unsafe { msg_send![super(self), layoutSubviews] };
            layout(self);
        }

        #[unsafe(method(touchesBegan:withEvent:))]
        fn touches_began(&self, touches: &NSSet<UITouch>, _event: Option<&UIEvent>) {
            touches_changed(self, touches, Phase::Down);
        }

        #[unsafe(method(touchesMoved:withEvent:))]
        fn touches_moved(&self, touches: &NSSet<UITouch>, _event: Option<&UIEvent>) {
            touches_changed(self, touches, Phase::Moved);
        }

        #[unsafe(method(touchesEnded:withEvent:))]
        fn touches_ended(&self, touches: &NSSet<UITouch>, _event: Option<&UIEvent>) {
            touches_changed(self, touches, Phase::Up);
        }

        #[unsafe(method(touchesCancelled:withEvent:))]
        fn touches_cancelled(&self, touches: &NSSet<UITouch>, _event: Option<&UIEvent>) {
            touches_changed(self, touches, Phase::Up);
        }

        /// The settings panel's Play button: saves the settings and boots the title.
        #[unsafe(method(settingsPlay:))]
        fn settings_play(&self, _sender: &AnyObject) {
            let settings = with_state(|s| {
                let panel = s.panel.take()?;
                let settings = panel.settings();
                settings.save(&s.settings_path);
                panel.remove();
                // Touches on the panel reached the controls under it too.
                s.touch.reset();
                Some(settings)
            })
            .flatten();
            if let Some(settings) = settings {
                play(self, settings);
            }
        }

        #[unsafe(method(tick:))]
        fn tick(&self, _link: &CADisplayLink) {
            with_state(|s| {
                poll_input(s);
                update_status(s);
                if let (Some(fps), Some(kernel)) = (&mut s.fps, KERNEL.get()) {
                    fps.update(kernel.gpu.swap_count());
                }
                if let Some(overlay) = &mut s.overlay {
                    overlay.draw(&mut s.touch, !s.pad_connected);
                }
            });
        }
    }
);

impl SkyrtView {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        // SAFETY: plain init of a UIView subclass without ivars.
        let view: Retained<Self> = unsafe { msg_send![Self::alloc(mtm), init] };
        view.setMultipleTouchEnabled(true);
        view.setBackgroundColor(Some(&UIColor::blackColor()));
        view
    }
}

/// Builds the window when the scene connects: the view, its controls and status line, then
/// hands the layer to `boot`.
fn connect(mtm: MainThreadMarker, scene: &UIWindowScene) {
    let window = UIWindow::initWithWindowScene(UIWindow::alloc(mtm), scene);
    // SAFETY: plain init of a UIViewController subclass without ivars.
    let controller: Retained<SkyrtViewController> = unsafe { msg_send![SkyrtViewController::alloc(mtm), init] };
    window.setRootViewController(Some(&controller));
    window.makeKeyAndVisible();
    let Some(view) = controller.view().and_then(|v| v.downcast::<SkyrtView>().ok()) else {
        log::error!("the view controller has no game view");
        return;
    };
    // A view backed by a CAMetalLayer keeps a scale factor of 1 unless told otherwise, which
    // made the drawable the screen's size in points (a third of its pixels on a 3x iPhone).
    let native = scene.screen().nativeScale();
    view.setContentScaleFactor(native);
    log::info!("screen scale {native}");
    let overlay = Overlay::new(mtm, &view);
    let status = status_label(mtm, &view);
    UIApplication::sharedApplication(mtm).setIdleTimerDisabled(true);
    start_audio_session();
    // SAFETY: the display link retains its target (the view) and calls `tick:` with itself on
    // the main run loop.
    unsafe {
        let link = CADisplayLink::displayLinkWithTarget_selector(&view, sel!(tick:));
        link.setPreferredFrameRateRange(CAFrameRateRange { minimum: 60.0, maximum: 120.0, preferred: 120.0 });
        link.addToRunLoop_forMode(&NSRunLoop::mainRunLoop(), NSRunLoopCommonModes);
    }
    // The layer lives as long as the view, which the window keeps for the life of the process.
    let layer = Retained::as_ptr(&view.layer()).cast::<AnyObject>() as usize;
    with_state(|s| {
        s.overlay = Some(overlay);
        s.status = Some(status);
        s.window = Some(window);
        s.render_window = Some(RenderWindow { handles: WindowHandles::Metal { layer }, size: s.size.clone() });
    });
    let (settings, autoplay) = with_state(|s| (Settings::load(&s.settings_path), s.autoplay)).unwrap_or_default();
    if autoplay {
        layout(&view);
        play(&view, settings);
        return;
    }
    with_state(|s| s.panel = Some(Panel::new(mtm, &view, &settings, &view)));
    layout(&view);
}

/// Boots the title with `settings` (from the panel's Play button, or at launch with autoplay).
fn play(view: &SkyrtView, settings: Settings) {
    let mtm = view.mtm();
    let boot = with_state(|s| {
        if settings.show_fps {
            s.fps = Some(FpsLabel::new(mtm, view));
        }
        Some((s.boot.take()?, s.render_window.take()?))
    })
    .flatten();
    layout(view);
    if let Some((boot, window)) = boot {
        log::info!("settings: {settings:?}");
        boot(window, settings);
    }
}

/// Fits the drawable to the view, the controls to its safe area (clear of the Dynamic Island
/// and the home indicator) and the status line to the middle.
fn layout(view: &SkyrtView) {
    let bounds = view.bounds();
    let scale = view.contentScaleFactor();
    let (width, height) = ((bounds.size.width * scale).round() as u32, (bounds.size.height * scale).round() as u32);
    let inset = view.safeAreaInsets();
    let area = rect(
        bounds.origin.x + inset.left,
        bounds.origin.y + inset.top,
        (bounds.size.width - inset.left - inset.right).max(1.0),
        (bounds.size.height - inset.top - inset.bottom).max(1.0),
    );
    with_state(|s| {
        s.size.store(RenderWindow::pack_size(width, height), Ordering::Relaxed);
        if let Some(overlay) = &mut s.overlay
            && area != overlay.area
        {
            overlay.set_area(area);
            s.touch.resize(area.size.width, area.size.height);
        }
        if let Some(panel) = &s.panel {
            panel.layout(bounds, area);
        }
        if let Some(fps) = &s.fps {
            fps.layout(area);
        }
        if let Some(status) = &s.status {
            let margin = area.size.width * 0.1;
            status.setFrame(rect(area.origin.x + margin, area.origin.y, area.size.width - 2.0 * margin, area.size.height));
        }
    });
}

#[derive(Clone, Copy)]
enum Phase {
    Down,
    Moved,
    Up,
}

fn touches_changed(view: &SkyrtView, touches: &NSSet<UITouch>, phase: Phase) {
    with_state(|s| {
        if s.pad_connected {
            return;
        }
        let origin = s.overlay.as_ref().map_or(CGPoint { x: 0.0, y: 0.0 }, |o| o.area.origin);
        for touch in touches.iter() {
            // A UITouch is the same object for the whole touch.
            let id = Retained::as_ptr(&touch) as usize as u64;
            let p = touch.locationInView(Some(view));
            let (x, y) = (p.x - origin.x, p.y - origin.y);
            match phase {
                Phase::Down => s.touch.down(id, x, y),
                Phase::Moved => s.touch.moved(id, x, y),
                Phase::Up => s.touch.up(id),
            }
        }
        // Presses go out at once rather than at the next frame.
        poll_input(s);
    });
}

/// Combines the touch controls, gamepad and keyboard and sends the result when it changed.
fn poll_input(s: &mut State) {
    let pad = pads::gamepad();
    if s.pad_connected != pad.is_some() {
        s.pad_connected = pad.is_some();
        log::info!("controller {}", if s.pad_connected { "connected" } else { "disconnected" });
        if s.pad_connected {
            s.touch.reset();
        }
    }
    let t = s.touch.state;
    let touch = Reading {
        buttons: t.buttons,
        left_trigger: f64::from(u8::from(t.left_trigger)),
        right_trigger: f64::from(u8::from(t.right_trigger)),
        left: t.left,
        right: t.right,
    };
    let state = [pad, pads::keyboard()].into_iter().flatten().fold(touch, Reading::merge).gamepad();
    let Some(kernel) = KERNEL.get() else { return };
    if s.sent.as_ref().is_none_or(|old| !same(old, &state)) {
        kernel.input.update(0, state);
        s.sent = Some(state);
    }
}

fn same(a: &Gamepad, b: &Gamepad) -> bool {
    (a.buttons, a.left_trigger, a.right_trigger, a.thumb_lx, a.thumb_ly, a.thumb_rx, a.thumb_ry)
        == (b.buttons, b.left_trigger, b.right_trigger, b.thumb_lx, b.thumb_ly, b.thumb_rx, b.thumb_ry)
}

fn update_status(s: &mut State) {
    let Some(label) = &s.status else { return };
    if let Some(text) = crate::status::take_change(&mut s.status_seen) {
        label.setHidden(text.is_none());
        label.setText(text.map(|t| NSString::from_str(&t)).as_deref());
    }
}

fn status_label(mtm: MainThreadMarker, parent: &UIView) -> Retained<UILabel> {
    let label = UILabel::new(mtm);
    label.setNumberOfLines(0);
    label.setTextAlignment(NSTextAlignment::Center);
    // SAFETY: a color and a system font, on the main thread.
    unsafe {
        label.setTextColor(Some(&UIColor::colorWithWhite_alpha(1.0, 0.9)));
        label.setFont(Some(&UIFont::systemFontOfSize(17.0)));
    }
    label.setUserInteractionEnabled(false);
    label.setHidden(true);
    parent.addSubview(&label);
    label
}

/// The Playback category keeps sound on with the ring/silent switch set to silent, as games
/// are expected to.
fn start_audio_session() {
    use objc2_avf_audio::{AVAudioSession, AVAudioSessionCategoryPlayback};
    // SAFETY: AVAudioSession calls with a category constant from the framework.
    unsafe {
        let session = AVAudioSession::sharedInstance();
        if let Some(category) = AVAudioSessionCategoryPlayback
            && let Err(e) = session.setCategory_error(category)
        {
            log::warn!("audio session category: {}", e.localizedDescription());
        }
        if let Err(e) = session.setActive_error(true) {
            log::warn!("audio session: {}", e.localizedDescription());
        }
    }
}
