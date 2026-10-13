//! xboxkrnl.exe exports.

mod audio;
mod dpc;
mod io;
mod ldi;
mod memory;
mod misc;
mod ob;
mod rtl;
mod sync;
mod threading;
mod timer;
mod video;
mod xma;

use std::sync::LazyLock;

use crate::dispatch::Export;

pub static EXPORTS: LazyLock<Vec<Export>> = LazyLock::new(|| {
    [
        audio::EXPORTS,
        dpc::EXPORTS,
        io::EXPORTS,
        ldi::EXPORTS,
        memory::EXPORTS,
        misc::EXPORTS,
        ob::EXPORTS,
        rtl::EXPORTS,
        sync::EXPORTS,
        threading::EXPORTS,
        timer::EXPORTS,
        video::EXPORTS,
        xma::EXPORTS,
    ]
    .concat()
});

pub use dpc::{begin_dpc, call_as_dpc, end_dpc};
pub use ldi::LdiContexts;
pub use sync::{enter_critical_section, leave_critical_section};
pub(crate) use sync::{init_critical_section_at, spin_count_field};
pub use timer::Timers;
