//! Write tracking for GPU-visible memory: the renderer asks, per 4 KiB page of the 512 MiB
//! physical space, whether it was written since it last asked. Every write that can land there
//! marks it — recompiled guest stores (ppc_context.h marks `skyrt_written` in its store macros)
//! and the runtime's own writes into guest memory.
//!
//! The renderer consumes the marks as it reads (see xenos-render's page generations), so it
//! re-uploads only pages written since, instead of comparing their contents. Wasm can't
//! write-protect pages, which is how Xenia tracks this natively.
//!
//! Marks are kept per guest page, not per physical page, so a store marks the page of the address
//! it wrote with a shift and a byte store, and no translation (that is most of the code a store
//! compiles to, and Safari's memory use follows code size). A physical page is reachable at four
//! guest addresses, so asking about it checks all of them ([`aliases`]): its physical view
//! (`PHYSICAL_BASE + p`), the 16 MiB and 4 KiB views that `fold_view` maps onto it, and the
//! identity-mapped XEX region (`XEX_BASE + p`, see `AddressSpace::translate_physical`).

use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use crate::memory::map::{PHYS_VIEW_4K, PHYS_VIEW_16M, PHYSICAL_BASE, PHYSICAL_SIZE, XEX_BASE};

const PAGE_SHIFT: u32 = 12;
/// Physical pages tracked.
pub const PAGES: usize = (PHYSICAL_SIZE >> PAGE_SHIFT) as usize;
/// Pages of the guest address space.
const GUEST_PAGES: usize = 1 << (32 - PAGE_SHIFT);

/// Per guest page: 0 when a write may have changed it since [`take`] last consumed the mark, 1
/// otherwise. Zero means written so the 1 MiB table is zero-initialized (no data in the module)
/// and every page starts out written: nothing was seen yet. Only pages that alias physical memory
/// are ever consumed; the marks stores elsewhere leave are never read.
#[unsafe(no_mangle)]
pub static skyrt_written: [AtomicU8; GUEST_PAGES] = [const { AtomicU8::new(0) }; GUEST_PAGES];

/// The guest pages physical page `page` is visible at. The 4 KiB view is a page ahead of
/// physical memory (`0xE0000000 + x` is physical `x + 0x1000`), so physical page 0 has no address
/// there.
#[inline]
fn aliases(page: u32) -> impl Iterator<Item = u32> {
    let views = [XEX_BASE, PHYSICAL_BASE, PHYS_VIEW_16M].map(|base| Some((base >> PAGE_SHIFT) + page));
    let view_4k = page.checked_sub(1).map(|p| (PHYS_VIEW_4K >> PAGE_SHIFT) + p);
    views.into_iter().chain([view_4k]).flatten()
}

/// Marks the guest pages `[addr, addr + len)` covers as written.
#[inline]
pub fn mark(addr: u32, len: u32) {
    if len == 0 {
        return;
    }
    let last = addr.saturating_add(len - 1);
    for page in addr >> PAGE_SHIFT..=last >> PAGE_SHIFT {
        skyrt_written[page as usize].store(0, Ordering::Release);
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

/// Whether physical page `page` was written, through any of its addresses, since the last
/// `take`, consuming the marks.
#[inline]
pub fn take(page: u32) -> bool {
    if page as usize >= PAGES {
        return false;
    }
    let mut written = false;
    for alias in aliases(page) {
        let mark = &skyrt_written[alias as usize];
        // Most pages are unwritten: read before swapping.
        if mark.load(Ordering::Acquire) == 0 {
            written |= mark.swap(1, Ordering::AcqRel) == 0;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::fold_view;

    #[test]
    fn marks_every_alias_and_page_spans() {
        for p in 0..5 {
            take(0x100 + p);
        }
        mark(PHYSICAL_BASE + 0x10_0FFE, 4);
        assert!(take(0x100) && take(0x101), "a write across a page boundary marks both");
        assert!(!take(0x100), "taking clears");
        mark(XEX_BASE + 0x10_2000, 1);
        assert!(take(0x102));
        mark(0xE010_3FFF, 1);
        assert!(take(0x104), "the 4 KiB view is a page ahead");
        mark(PHYS_VIEW_16M + 0x10_3000, 2);
        assert!(take(0x103));
        mark(0x4000_0000, 16);
        assert!(!take(0x103), "virtual memory isn't tracked");
        assert!(!take(0x100) && !take(0x101) && !take(0x102) && !take(0x104));
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

    /// The physical address a guest address writes, as the runtime translates it.
    fn physical(addr: u32) -> Option<u32> {
        let addr = fold_view(addr);
        if let Some(p) = addr.checked_sub(PHYSICAL_BASE).filter(|&p| p < PHYSICAL_SIZE) {
            return Some(p);
        }
        addr.checked_sub(XEX_BASE).filter(|&p| p < PHYSICAL_SIZE)
    }

    #[test]
    fn aliases_are_exactly_the_addresses_of_a_physical_page() {
        let mut expected = vec![Vec::new(); PAGES];
        for page in 0..GUEST_PAGES as u32 {
            if let Some(p) = physical(page << PAGE_SHIFT) {
                expected[(p >> PAGE_SHIFT) as usize].push(page);
            }
        }
        for (p, pages) in expected.iter().enumerate() {
            let mut got: Vec<u32> = aliases(p as u32).collect();
            got.sort_unstable();
            assert_eq!(&got, pages, "physical page {p:#x}");
        }
    }
}
