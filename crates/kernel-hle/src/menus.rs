//! Skyrim's menus (Scaleform movies) on a frame wider than 16:9.
//!
//! The title loads each menu's movie with a scale mode (`BSScaleformManager::LoadMovie`, hooked
//! in guest-sys): ShowAll keeps the whole 16:9 movie in view, NoBorder fills the frame and
//! crops the overhang. At 16:9 they are the same. With `--game-aspect` NoBorder crops the top and
//! bottom: fine for the main menu and message boxes (nothing near those edges, and they fill the
//! width), but it pushed the skills menu's bars (name and level; magicka, health and stamina)
//! off the screen, so on a wider frame that menu loads with ShowAll.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::Kernel;

/// Movies whose content reaches the top and bottom edges.
const EDGE_TO_EDGE: [&str; 1] = ["StatsMenu"];

/// GFxMovieView::ScaleModeType.
const SHOW_ALL: u32 = 1;
const NO_BORDER: u32 = 3;

static WIDE: AtomicBool = AtomicBool::new(false);

/// The title draws a frame wider than 16:9 (`--game-aspect`).
pub fn set_wide_frame(wide: bool) {
    WIDE.store(wide, Ordering::Relaxed);
}

/// The scale mode to load the movie `name` (a guest C string) with, given the title's `mode`.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_movie_scale_mode(name: u32, mode: u32) -> u32 {
    let wide = WIDE.load(Ordering::Relaxed);
    let name = match Kernel::try_global() {
        Some(k) if name != 0 => String::from_utf8_lossy(&k.mem().read_cstr(name, 64)).into_owned(),
        _ => String::new(),
    };
    let new = if wide && mode == NO_BORDER && EDGE_TO_EDGE.contains(&name.as_str()) { SHOW_ALL } else { mode };
    log::debug!("menu movie {name:?}: scale mode {mode} -> {new}");
    new
}
