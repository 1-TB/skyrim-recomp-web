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
}

impl PageHeap {
    pub fn new(name: &'static str, base: u32, size: u32) -> Self {
        assert!(base.is_multiple_of(PAGE_SIZE) && size.is_multiple_of(PAGE_SIZE));
        let free = Page { state: PageState::Free, protect: 0, alloc_base: 0 };
        PageHeap { name, base, pages: vec![free; (size / PAGE_SIZE) as usize], allocations: BTreeMap::new() }
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

    fn mark(&mut self, first: u32, count: u32, commit: bool, protect: u32) -> Vec<(u32, u32)> {
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

        let mut candidate = if top_down { last_start } else { lo };
        loop {
            if candidate < lo || candidate > last_start {
                return Err(HeapError::NoSpace);
            }
            // Find the first non-free page in the window to skip past it quickly.
            let window = &self.pages[candidate as usize..(candidate + count) as usize];
            match if top_down {
                window.iter().rposition(|p| p.state != PageState::Free)
            } else {
                window.iter().position(|p| p.state != PageState::Free)
            } {
                None => {
                    let newly = self.mark(candidate, count, commit, protect);
                    return Ok(Allocated { address: self.base + candidate * PAGE_SIZE, newly_committed: newly });
                }
                Some(busy) => {
                    let busy = candidate + busy as u32;
                    if top_down {
                        // The next window must end at or before the busy page.
                        if busy < count {
                            return Err(HeapError::NoSpace);
                        }
                        candidate = (busy - count) / align_pages * align_pages;
                    } else {
                        candidate = (busy + 1).div_ceil(align_pages) * align_pages;
                    }
                }
            }
        }
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
                match committed.last_mut() {
                    Some((s, n)) if *s + *n == self.base + i * PAGE_SIZE => *n += PAGE_SIZE,
                    _ => committed.push((self.base + i * PAGE_SIZE, PAGE_SIZE)),
                }
            }
            *page = Page { state: PageState::Free, protect: 0, alloc_base: 0 };
        }
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
        let mut used = 0;
        let mut committed = 0;
        for p in &self.pages {
            match p.state {
                PageState::Free => {}
                PageState::Reserved => used += 1,
                PageState::Committed => {
                    used += 1;
                    committed += 1;
                }
            }
        }
        (used, committed)
    }
}
