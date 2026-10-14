//! Write tracking for GPU-visible memory: one byte per 4 KiB page of the 512 MiB physical space,
//! set (to 1) by every write that can land there — recompiled guest stores (ppc_context.h marks
//! `skyrt_dirty_pages` in its store macros) and the runtime's own writes into guest memory.
//!
//! The renderer clears the bytes as it consumes them (see xenos-render's page generations), so
//! it re-uploads only pages written since, instead of comparing their contents. Wasm can't
//! write-protect pages, which is how Xenia tracks this natively.
//!
//! A physical page is reachable at two guest addresses: its physical view (`PHYSICAL_BASE + p`)
//! and, when that page isn't allocated, the identity-mapped XEX region (`XEX_BASE + p`, see
//! `AddressSpace::translate_physical`); writes through either mark it, as do writes through the
//! 16 MiB and 4 KiB views, which `fold_view` maps onto the first.

use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use crate::memory::{FRAMES, FRAME_SHIFT, fold_view};
use crate::memory::map::{PHYSICAL_BASE, PHYSICAL_SIZE, XEX_BASE};

const PAGE_SHIFT: u32 = 12;
/// Physical pages tracked.
pub const PAGES: usize = (PHYSICAL_SIZE >> PAGE_SHIFT) as usize;

/// The dirty bytes, indexed by physical page. All start dirty: nothing was seen yet.
#[unsafe(no_mangle)]
pub static skyrt_dirty_pages: [AtomicU8; PAGES] = [const { AtomicU8::new(1) }; PAGES];

/// Physical address of the first byte of each 1 MiB guest frame ([`physical`] of the frame's
/// start), or `u32::MAX` for frames that aren't physical memory: recompiled stores find the
/// pages they mark through this (`skyrt_mark_written` in cpp/ppc_context.h), since their own
/// address translation yields a linear address, not a physical one.
#[unsafe(no_mangle)]
pub static skyrt_dirty_frame: [u32; FRAMES] = {
    let mut t = [u32::MAX; FRAMES];
    let mut i = 0;
    while i < FRAMES {
        let folded = fold_view((i as u32) << FRAME_SHIFT);
        if folded >= PHYSICAL_BASE && folded - PHYSICAL_BASE < PHYSICAL_SIZE {
            t[i] = folded - PHYSICAL_BASE;
        } else if folded >= XEX_BASE && folded - XEX_BASE < PHYSICAL_SIZE {
            t[i] = folded - XEX_BASE;
        }
        i += 1;
    }
    t
};

/// The physical address `addr` (a guest address, as written to) aliases, if any.
#[inline]
fn physical(addr: u32) -> Option<u32> {
    let addr = fold_view(addr);
    if let Some(p) = addr.checked_sub(PHYSICAL_BASE).filter(|&p| p < PHYSICAL_SIZE) {
        return Some(p);
    }
    addr.checked_sub(XEX_BASE).filter(|&p| p < PHYSICAL_SIZE)
}

/// Marks the physical pages the guest range `[addr, addr + len)` covers as written.
#[inline]
pub fn mark(addr: u32, len: u32) {
    if len == 0 {
        return;
    }
    let Some(start) = physical(addr) else { return };
    let end = (u64::from(start) + u64::from(len) - 1).min(u64::from(PHYSICAL_SIZE - 1)) as u32;
    for page in start >> PAGE_SHIFT..=end >> PAGE_SHIFT {
        skyrt_dirty_pages[page as usize].store(1, Ordering::Release);
    }
}

/// Marks physical pages `[addr, addr + len)` (physical addresses) as written.
pub fn mark_physical(addr: u32, len: u32) {
    if len == 0 || addr >= PHYSICAL_SIZE {
        return;
    }
    let end = (u64::from(addr) + u64::from(len) - 1).min(u64::from(PHYSICAL_SIZE - 1)) as u32;
    for page in addr >> PAGE_SHIFT..=end >> PAGE_SHIFT {
        skyrt_dirty_pages[page as usize].store(1, Ordering::Release);
    }
}

/// Per page: how many times [`version`] found it written.
static GENERATIONS: [AtomicU32; PAGES] = [const { AtomicU32::new(0) }; PAGES];

/// A counter for physical page `page` that has changed whenever the page was written since it was
/// last read. Consumes the page's mark, so everything that needs to know about writes must ask
/// this (not [`take`]).
#[inline]
pub fn version(page: u32) -> u32 {
    let Some(generation) = GENERATIONS.get(page as usize) else { return 0 };
    if take(page) { generation.fetch_add(1, Ordering::AcqRel) + 1 } else { generation.load(Ordering::Acquire) }
}

/// A number that has changed whenever a byte of physical `[addr, addr + len)` was written since it
/// was last asked for (the sum of the pages' [`version`]s, which only grow).
pub fn range_version(addr: u32, len: u32) -> u64 {
    if len == 0 || addr >= PHYSICAL_SIZE {
        return 0;
    }
    let end = (u64::from(addr) + u64::from(len) - 1).min(u64::from(PHYSICAL_SIZE - 1)) as u32;
    (addr >> PAGE_SHIFT..=end >> PAGE_SHIFT).map(|p| u64::from(version(p))).sum()
}

/// Whether physical page `page` was written since the last `take`, clearing the mark.
#[inline]
pub fn take(page: u32) -> bool {
    skyrt_dirty_pages.get(page as usize).is_some_and(|b| b.swap(0, Ordering::AcqRel) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_both_aliases_and_page_spans() {
        for p in 0..4 {
            take(0x100 + p);
        }
        mark(PHYSICAL_BASE + 0x10_0FFE, 4);
        assert!(take(0x100) && take(0x101), "a write across a page boundary marks both");
        assert!(!take(0x100), "taking clears");
        mark(XEX_BASE + 0x10_2000, 1);
        assert!(take(0x102));
        mark(0xE010_3FFF, 1);
        assert!(take(0x104), "the 4 KiB view is a page ahead");
        mark(0x4000_0000, 16);
        assert!(!take(0x103), "virtual memory isn't tracked");
        mark_physical(0x10_3000, 0x1000);
        assert!(take(0x103) && !take(0x104) && !take(0x102));
    }

    #[test]
    fn versions_change_only_after_writes() {
        let v = range_version(0x20_0000, 0x2000);
        assert_eq!(range_version(0x20_0000, 0x2000), v);
        mark(PHYSICAL_BASE + 0x20_1800, 4);
        let w = range_version(0x20_0000, 0x2000);
        assert_ne!(w, v);
        assert_eq!(range_version(0x20_0000, 0x2000), w);
        assert_eq!(range_version(0x20_0000, 0x1000), u64::from(version(0x200)));
    }
}

#[cfg(test)]
mod frame_table_tests {
    use super::*;

    #[test]
    fn frame_table_matches_physical() {
        for ea in (0..=u32::MAX).step_by(0x1000) {
            let from_table = match skyrt_dirty_frame[(ea >> FRAME_SHIFT) as usize] {
                u32::MAX => None,
                p => Some(p + (ea & ((1 << FRAME_SHIFT) - 1))),
            };
            // The top page of the 4 KiB view folds past physical memory.
            if ea >= 0xFFF0_0000 {
                continue;
            }
            assert_eq!(from_table, physical(ea), "{ea:#010X}");
        }
    }
}
