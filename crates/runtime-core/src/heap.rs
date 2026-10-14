//! Page-granular reserve/commit bookkeeping for one guest address range.
//!
//! A heap only tracks page state; committing host memory is the caller's job
//! (see [`crate::address_space::AddressSpace`]). Granularity is always 4 KiB;
//! larger page sizes are expressed through allocation alignment and size.

use std::collections::BTreeMap;
use std::fmt;

use crate::memory::PAGE_SIZE;

/// Xbox protection flags (`X_PAGE_*`), stored verbatim.
pub mod protect {
    pub const NOACCESS: u32 = 0x01;
    pub const READONLY: u32 = 0x02;
    pub const READWRITE: u32 = 0x04;
    pub const WRITECOPY: u32 = 0x08;
    pub const EXECUTE: u32 = 0x10;
    pub const EXECUTE_READ: u32 = 0x20;
    pub const EXECUTE_READWRITE: u32 = 0x40;
    pub const EXECUTE_WRITECOPY: u32 = 0x80;
    pub const GUARD: u32 = 0x100;
    pub const NOCACHE: u32 = 0x200;
    pub const WRITECOMBINE: u32 = 0x400;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    Free,
    Reserved,
    Committed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapError {
    OutOfRange,
    NoSpace,
    Conflict,
    NotAllocationBase,
    NotAllocated,
}

impl fmt::Display for HeapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            HeapError::OutOfRange => "address range outside heap",
            HeapError::NoSpace => "no free range large enough",
            HeapError::Conflict => "range overlaps an incompatible allocation",
            HeapError::NotAllocationBase => "address is not the base of an allocation",
            HeapError::NotAllocated => "range is not allocated",
        })
    }
}

impl std::error::Error for HeapError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionInfo {
    pub base_address: u32,
    pub allocation_base: u32,
    pub allocation_protect: u32,
    pub region_size: u32,
    pub state: PageState,
    pub protect: u32,
}

#[derive(Clone, Copy)]
struct Page {
    state: PageState,
    protect: u32,
    /// Page index of the allocation this page belongs to (meaningless when free).
    alloc_base: u32,
}

#[derive(Clone, Copy)]
struct Allocation {
    pages: u32,
    protect: u32,
}

/// Result of an allocation: the base address and the byte ranges that went from
/// uncommitted to committed (callers commit and zero exactly these).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocated {
    pub address: u32,
    pub newly_committed: Vec<(u32, u32)>,
}

pub struct PageHeap {
    name: &'static str,
    base: u32,
    pages: Vec<Page>,
    allocations: BTreeMap<u32, Allocation>,
    /// Free page runs (first page -> length), so allocation doesn't scan the page array (with
    /// tens of thousands of pages in use that took milliseconds per call, and the title
    /// allocates thousands of times while streaming).
    free: BTreeMap<u32, u32>,
    /// Pages reserved or committed, and committed (MmQueryStatistics asks often).
    used_pages: u32,
    committed_pages: u32,
}

impl PageHeap {
    pub fn new(name: &'static str, base: u32, size: u32) -> Self {
        assert!(base.is_multiple_of(PAGE_SIZE) && size.is_multiple_of(PAGE_SIZE));
        let free = Page { state: PageState::Free, protect: 0, alloc_base: 0 };
        let count = size / PAGE_SIZE;
        PageHeap {
            name,
            base,
            pages: vec![free; count as usize],
            allocations: BTreeMap::new(),
            free: if count > 0 { BTreeMap::from([(0, count)]) } else { BTreeMap::new() },
            used_pages: 0,
            committed_pages: 0,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn base(&self) -> u32 {
        self.base
    }

    pub fn size(&self) -> u32 {
        self.pages.len() as u32 * PAGE_SIZE
    }

    pub fn contains(&self, addr: u32) -> bool {
        addr >= self.base && ((addr - self.base) as u64) < self.size() as u64
    }

    fn page_index(&self, addr: u32) -> Result<u32, HeapError> {
        if !self.contains(addr) {
            return Err(HeapError::OutOfRange);
        }
        Ok((addr - self.base) / PAGE_SIZE)
    }

    fn page_count(size: u32) -> u32 {
        size.div_ceil(PAGE_SIZE)
    }

    fn range_free(&self, first: u32, count: u32) -> bool {
        self.pages[first as usize..(first + count) as usize].iter().all(|p| p.state == PageState::Free)
    }

    /// Removes `[first, first + count)` (wholly free) from the free runs.
    fn take_free(&mut self, first: u32, count: u32) {
        let (&start, &len) = self.free.range(..=first).next_back().expect("range is free");
        debug_assert!(first + count <= start + len);
        self.free.remove(&start);
        if first > start {
            self.free.insert(start, first - start);
        }
        if first + count < start + len {
            self.free.insert(first + count, start + len - (first + count));
        }
    }

    /// Returns `[first, first + count)` to the free runs, merging with its neighbours.
    fn add_free(&mut self, first: u32, count: u32) {
        let (mut start, mut len) = (first, count);
        if let Some((&s, &l)) = self.free.range(..first).next_back()
            && s + l == first
        {
            self.free.remove(&s);
            start = s;
            len += l;
        }
        if let Some(&l) = self.free.get(&(first + count)) {
            self.free.remove(&(first + count));
            len += l;
        }
        self.free.insert(start, len);
    }

    fn mark(&mut self, first: u32, count: u32, commit: bool, protect: u32) -> Vec<(u32, u32)> {
        self.take_free(first, count);
        self.used_pages += count;
        if commit {
            self.committed_pages += count;
        }
        let state = if commit { PageState::Committed } else { PageState::Reserved };
        for p in &mut self.pages[first as usize..(first + count) as usize] {
            *p = Page { state, protect, alloc_base: first };
        }
        self.allocations.insert(first, Allocation { pages: count, protect });
        if commit { vec![(self.base + first * PAGE_SIZE, count * PAGE_SIZE)] } else { Vec::new() }
    }

    /// Reserves (and optionally commits) `size` bytes aligned to `align`.
    pub fn alloc(
        &mut self,
        size: u32,
        align: u32,
        commit: bool,
        protect: u32,
        top_down: bool,
    ) -> Result<Allocated, HeapError> {
        let end = self.base as u64 + self.size() as u64;
        self.alloc_range(self.base, (end - 1) as u32, size, align, commit, protect, top_down)
    }

    /// Like [`PageHeap::alloc`] but the allocation must lie within `[min, max]`.
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_range(
        &mut self,
        min: u32,
        max: u32,
        size: u32,
        align: u32,
        commit: bool,
        protect: u32,
        top_down: bool,
    ) -> Result<Allocated, HeapError> {
        if size == 0 {
            return Err(HeapError::NoSpace);
        }
        let align_pages = (align.max(PAGE_SIZE) / PAGE_SIZE).max(1);
        let count = Self::page_count(size);
        let total = self.pages.len() as u32;
        let lo = ((min.max(self.base) - self.base) / PAGE_SIZE).div_ceil(align_pages) * align_pages;
        let hi_addr = (max as u64).min(self.base as u64 + self.size() as u64 - 1);
        if hi_addr < self.base as u64 {
            return Err(HeapError::NoSpace);
        }
        let hi_end = ((hi_addr - self.base as u64) / PAGE_SIZE as u64 + 1) as u32;
        if hi_end > total || count > hi_end || lo > hi_end - count {
            return Err(HeapError::NoSpace);
        }
        let last_start = (hi_end - count) / align_pages * align_pages;

        // First fit over the free runs, from the bottom or the top of [lo, hi_end).
        let fit = |run_start: u32, run_len: u32| -> Option<u32> {
            let (from, to) = (run_start.max(lo), (run_start + run_len).min(hi_end));
            if to < from + count {
                return None;
            }
            let c = if top_down {
                (to - count) / align_pages * align_pages
            } else {
                from.div_ceil(align_pages) * align_pages
            };
            (c >= from && c + count <= to && c <= last_start).then_some(c)
        };
        let found = if top_down {
            self.free.range(..hi_end).rev().find_map(|(&s, &l)| fit(s, l))
        } else {
            let first_run = self.free.range(..=lo).next_back().map_or(lo, |(&s, _)| s);
            self.free.range(first_run..).find_map(|(&s, &l)| fit(s, l))
        };
        let candidate = found.ok_or(HeapError::NoSpace)?;
        let newly = self.mark(candidate, count, commit, protect);
        Ok(Allocated { address: self.base + candidate * PAGE_SIZE, newly_committed: newly })
    }

    /// Allocation at a caller-chosen address. Free ranges are reserved (and
    /// optionally committed) as a new allocation; a range wholly inside one
    /// existing allocation can be committed in place.
    pub fn alloc_fixed(&mut self, addr: u32, size: u32, commit: bool, protect: u32) -> Result<Allocated, HeapError> {
        if size == 0 {
            return Err(HeapError::OutOfRange);
        }
        let first = self.page_index(addr)?;
        let count = Self::page_count(size + (addr % PAGE_SIZE));
        if (first + count) as usize > self.pages.len() {
            return Err(HeapError::OutOfRange);
        }
        if self.range_free(first, count) {
            let newly = self.mark(first, count, commit, protect);
            return Ok(Allocated { address: self.base + first * PAGE_SIZE, newly_committed: newly });
        }
        let owner = self.pages[first as usize];
        let same_alloc = self.pages[first as usize..(first + count) as usize]
            .iter()
            .all(|p| p.state != PageState::Free && p.alloc_base == owner.alloc_base);
        if !same_alloc {
            return Err(HeapError::Conflict);
        }
        let mut newly = Vec::new();
        if commit {
            let mut run: Option<(u32, u32)> = None;
            for i in first..first + count {
                let page = &mut self.pages[i as usize];
                if page.state != PageState::Committed {
                    page.state = PageState::Committed;
                    page.protect = protect;
                    self.committed_pages += 1;
                    run = Some(match run {
                        Some((s, n)) if s + n == i => (s, n + 1),
                        Some(prev) => {
                            newly.push((self.base + prev.0 * PAGE_SIZE, prev.1 * PAGE_SIZE));
                            (i, 1)
                        }
                        None => (i, 1),
                    });
                } else {
                    page.protect = protect;
                }
            }
            if let Some((s, n)) = run {
                newly.push((self.base + s * PAGE_SIZE, n * PAGE_SIZE));
            }
        }
        Ok(Allocated { address: self.base + first * PAGE_SIZE, newly_committed: newly })
    }

    /// Returns committed pages in the range to the reserved state. Returns the
    /// byte ranges that were committed (callers decommit exactly these).
    pub fn decommit(&mut self, addr: u32, size: u32) -> Result<Vec<(u32, u32)>, HeapError> {
        let first = self.page_index(addr)?;
        let count = Self::page_count(size + (addr % PAGE_SIZE));
        if (first + count) as usize > self.pages.len() {
            return Err(HeapError::OutOfRange);
        }
        if self.pages[first as usize..(first + count) as usize].iter().any(|p| p.state == PageState::Free) {
            return Err(HeapError::NotAllocated);
        }
        let mut ranges = Vec::new();
        for i in first..first + count {
            let page = &mut self.pages[i as usize];
            if page.state == PageState::Committed {
                page.state = PageState::Reserved;
                self.committed_pages -= 1;
                match ranges.last_mut() {
                    Some((s, n)) if *s + *n == self.base + i * PAGE_SIZE => *n += PAGE_SIZE,
                    _ => ranges.push((self.base + i * PAGE_SIZE, PAGE_SIZE)),
                }
            }
        }
        Ok(ranges)
    }

    /// Releases the whole allocation starting at `addr`. Returns its size and the
    /// byte ranges that were committed.
    pub fn release(&mut self, addr: u32) -> Result<(u32, Vec<(u32, u32)>), HeapError> {
        let first = self.page_index(addr)?;
        if !addr.is_multiple_of(PAGE_SIZE) {
            return Err(HeapError::NotAllocationBase);
        }
        let alloc = self.allocations.remove(&first).ok_or(HeapError::NotAllocationBase)?;
        let mut committed = Vec::new();
        for i in first..first + alloc.pages {
            let page = &mut self.pages[i as usize];
            if page.state == PageState::Committed {
                self.committed_pages -= 1;
                match committed.last_mut() {
                    Some((s, n)) if *s + *n == self.base + i * PAGE_SIZE => *n += PAGE_SIZE,
                    _ => committed.push((self.base + i * PAGE_SIZE, PAGE_SIZE)),
                }
            }
            *page = Page { state: PageState::Free, protect: 0, alloc_base: 0 };
        }
        self.used_pages -= alloc.pages;
        self.add_free(first, alloc.pages);
        Ok((alloc.pages * PAGE_SIZE, committed))
    }

    /// Changes the protection of committed pages; returns the previous protection
    /// of the first page.
    pub fn protect(&mut self, addr: u32, size: u32, protect: u32) -> Result<u32, HeapError> {
        let first = self.page_index(addr)?;
        let count = Self::page_count(size + (addr % PAGE_SIZE));
        if (first + count) as usize > self.pages.len() {
            return Err(HeapError::OutOfRange);
        }
        if self.pages[first as usize..(first + count) as usize].iter().any(|p| p.state != PageState::Committed) {
            return Err(HeapError::NotAllocated);
        }
        let old = self.pages[first as usize].protect;
        for p in &mut self.pages[first as usize..(first + count) as usize] {
            p.protect = protect;
        }
        Ok(old)
    }

    /// Describes the run of pages starting at `addr` that share state and
    /// protection (NtQueryVirtualMemory semantics).
    pub fn query(&self, addr: u32) -> Option<RegionInfo> {
        let first = self.page_index(addr).ok()?;
        let page = self.pages[first as usize];
        let mut end = first + 1;
        while (end as usize) < self.pages.len() {
            let p = self.pages[end as usize];
            let same_alloc = page.state == PageState::Free || p.alloc_base == page.alloc_base;
            if p.state != page.state || p.protect != page.protect || !same_alloc {
                break;
            }
            end += 1;
        }
        let (allocation_base, allocation_protect) = if page.state == PageState::Free {
            (0, 0)
        } else {
            let a = self.allocations[&page.alloc_base];
            (self.base + page.alloc_base * PAGE_SIZE, a.protect)
        };
        Some(RegionInfo {
            base_address: self.base + first * PAGE_SIZE,
            allocation_base,
            allocation_protect,
            region_size: (end - first) * PAGE_SIZE,
            state: page.state,
            protect: page.protect,
        })
    }

    /// Size of the allocation containing `addr`.
    pub fn allocation_size(&self, addr: u32) -> Option<u32> {
        let first = self.page_index(addr).ok()?;
        let page = self.pages[first as usize];
        if page.state == PageState::Free {
            return None;
        }
        Some(self.allocations[&page.alloc_base].pages * PAGE_SIZE)
    }

    pub fn state_of(&self, addr: u32) -> Option<PageState> {
        Some(self.pages[self.page_index(addr).ok()? as usize].state)
    }

    /// Counts of (reserved-or-committed pages, committed pages).
    pub fn usage(&self) -> (u32, u32) {
        (self.used_pages, self.committed_pages)
    }
}

#[cfg(test)]
mod free_run_tests {
    use super::*;

    /// The free runs and counters stay consistent with the page array through random
    /// allocations (aligned, top-down, ranged, fixed), commits and releases.
    #[test]
    fn free_runs_track_the_page_array() {
        let mut h = PageHeap::new("test", 0x4000_0000, 0x400_0000);
        let mut live = Vec::new();
        let mut seed = 0x1234_5678u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        for _ in 0..4000 {
            match rand() % 5 {
                0 | 1 => {
                    let size = (rand() % 64 + 1) * PAGE_SIZE;
                    let align = [PAGE_SIZE, 0x1_0000, 0x10_0000][(rand() % 3) as usize];
                    if let Ok(a) = h.alloc(size, align, rand() % 2 == 0, protect::READWRITE, rand() % 2 == 0) {
                        assert_eq!(a.address % align, 0);
                        live.push(a.address);
                    }
                }
                2 => {
                    let lo = 0x4000_0000 + (rand() % 0x300) * 0x1_0000;
                    if let Ok(a) = h.alloc_range(lo, lo + 0x40_0000, 0x3000, PAGE_SIZE, true, protect::READWRITE, false) {
                        assert!(a.address >= lo && a.address + 0x3000 <= lo + 0x40_0001);
                        live.push(a.address);
                    }
                }
                3 => {
                    let addr = 0x4000_0000 + (rand() % 0x4000) * PAGE_SIZE;
                    if let Ok(a) = h.alloc_fixed(addr, 0x2000, true, protect::READWRITE)
                        && h.allocations.contains_key(&((a.address - h.base) / PAGE_SIZE))
                        && !live.contains(&a.address)
                    {
                        live.push(a.address);
                    }
                }
                _ if !live.is_empty() => {
                    let i = (rand() as usize) % live.len();
                    h.release(live.swap_remove(i)).unwrap();
                }
                _ => {}
            }
            let free_pages: u32 = h.free.values().sum();
            assert_eq!(free_pages + h.used_pages, h.pages.len() as u32);
            for (&start, &len) in &h.free {
                assert!(h.pages[start as usize..(start + len) as usize].iter().all(|p| p.state == PageState::Free));
            }
            let committed = h.pages.iter().filter(|p| p.state == PageState::Committed).count() as u32;
            assert_eq!(committed, h.committed_pages);
        }
    }
}
