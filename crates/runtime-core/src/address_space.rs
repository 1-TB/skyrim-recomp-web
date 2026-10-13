//! Guest memory plus the allocators for each region of [`crate::memory::map`].

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Mutex;

use crate::heap::{Allocated, HeapError, PageHeap, PageState, RegionInfo, protect};
use crate::memory::{GuestMemory, MemoryError, PAGE_SIZE, map};

pub const PAGE_4K: u32 = 0x1000;
pub const PAGE_64K: u32 = 0x1_0000;
pub const PAGE_16M: u32 = 0x100_0000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    Heap(HeapError),
    Memory(MemoryError),
    /// The address is not in a region this operation applies to.
    WrongRegion(u32),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::Heap(e) => write!(f, "{e}"),
            AllocError::Memory(e) => write!(f, "{e}"),
            AllocError::WrongRegion(a) => write!(f, "address {a:#010X} is not in the expected region"),
        }
    }
}

impl std::error::Error for AllocError {}

impl From<HeapError> for AllocError {
    fn from(e: HeapError) -> Self {
        AllocError::Heap(e)
    }
}

impl From<MemoryError> for AllocError {
    fn from(e: MemoryError) -> Self {
        AllocError::Memory(e)
    }
}

pub type AllocResult<T> = Result<T, AllocError>;

/// Which allocator owns an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    HostReserved,
    Virtual,
    Stack,
    System,
    Mmio,
    Xex,
    /// A physical view; the payload is the view base.
    Physical(u32),
}

pub fn region_of(addr: u32) -> Region {
    match addr {
        a if a < map::HOST_RESERVED_END => Region::HostReserved,
        a if a < map::VIRTUAL_END => Region::Virtual,
        a if a < map::STACK_END => Region::Stack,
        a if a < map::SYSTEM_END => Region::System,
        a if a < map::MMIO_END => Region::Mmio,
        a if a < map::XEX_END => Region::Xex,
        a if a < map::VIRTUAL_4K_END => Region::Virtual,
        a if a < map::PHYS_VIEW_16M => Region::Physical(map::PHYS_VIEW_64K),
        a if a < map::PHYS_VIEW_4K => Region::Physical(map::PHYS_VIEW_16M),
        _ => Region::Physical(map::PHYS_VIEW_4K),
    }
}

/// Offset between a view address and the physical address it shows. The 4 KiB
/// view is shifted by one page on hardware: 0xE0000000 + x is physical
/// x + 0x1000. Titles compute GPU addresses with this rule themselves.
pub fn physical_view_offset(view: u32) -> u32 {
    if view == map::PHYS_VIEW_4K { PAGE_4K } else { 0 }
}

/// Physical view used for a given page size (MmAllocatePhysicalMemoryEx).
pub fn physical_view_for_page_size(page_size: u32) -> u32 {
    if page_size <= PAGE_4K {
        map::PHYS_VIEW_4K
    } else if page_size <= PAGE_64K {
        map::PHYS_VIEW_64K
    } else {
        map::PHYS_VIEW_16M
    }
}

/// First-fit allocator for small kernel-owned objects. The whole system region
/// is committed up front.
struct SystemHeap {
    free: BTreeMap<u32, u32>,
    used: HashMap<u32, u32>,
}

impl SystemHeap {
    const GRANULE: u32 = 16;

    fn new() -> Self {
        let mut free = BTreeMap::new();
        free.insert(map::SYSTEM_BASE, map::SYSTEM_END - map::SYSTEM_BASE);
        SystemHeap { free, used: HashMap::new() }
    }

    fn alloc(&mut self, size: u32, align: u32) -> Option<u32> {
        let size = size.max(1).div_ceil(Self::GRANULE) * Self::GRANULE;
        let align = align.max(Self::GRANULE);
        let (block, block_size, start) = self.free.iter().find_map(|(&b, &s)| {
            let start = b.div_ceil(align) * align;
            (start as u64 + size as u64 <= b as u64 + s as u64).then_some((b, s, start))
        })?;
        self.free.remove(&block);
        if start > block {
            self.free.insert(block, start - block);
        }
        let tail = block + block_size - (start + size);
        if tail > 0 {
            self.free.insert(start + size, tail);
        }
        self.used.insert(start, size);
        Some(start)
    }

    fn free(&mut self, addr: u32) -> Option<u32> {
        let size = self.used.remove(&addr)?;
        let mut start = addr;
        let mut len = size;
        if let Some((&prev, &prev_len)) = self.free.range(..addr).next_back()
            && prev + prev_len == addr
        {
            self.free.remove(&prev);
            start = prev;
            len += prev_len;
        }
        if let Some(&next_len) = self.free.get(&(addr + size)) {
            self.free.remove(&(addr + size));
            len += next_len;
        }
        self.free.insert(start, len);
        Some(size)
    }

    fn size_of(&self, addr: u32) -> Option<u32> {
        self.used.get(&addr).copied()
    }
}

/// Stack allocation with guard pages on both ends (layout matches Xenia).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackAllocation {
    pub alloc_base: u32,
    /// Lowest usable address.
    pub limit: u32,
    /// One past the highest usable address (initial r1 is derived from this).
    pub base: u32,
}

pub struct AddressSpace {
    pub mem: GuestMemory,
    virtual_heap: Mutex<PageHeap>,
    virtual_4k_heap: Mutex<PageHeap>,
    stack_heap: Mutex<PageHeap>,
    xex_heap: Mutex<PageHeap>,
    /// Indexed by physical address (0..512 MiB).
    physical: Mutex<PageHeap>,
    system: Mutex<SystemHeap>,
}

impl AddressSpace {
    pub fn new() -> AllocResult<Self> {
        let mem = GuestMemory::reserve()?;
        mem.commit(map::SYSTEM_BASE, map::SYSTEM_END - map::SYSTEM_BASE)?;
        mem.commit(map::MMIO_BASE, map::MMIO_END - map::MMIO_BASE)?;
        Ok(AddressSpace {
            mem,
            virtual_heap: Mutex::new(PageHeap::new("virtual", map::VIRTUAL_BASE, map::VIRTUAL_END - map::VIRTUAL_BASE)),
            virtual_4k_heap: Mutex::new(PageHeap::new(
                "virtual-4k",
                map::VIRTUAL_4K_BASE,
                map::VIRTUAL_4K_END - map::VIRTUAL_4K_BASE,
            )),
            stack_heap: Mutex::new(PageHeap::new("stack", map::STACK_BASE, map::STACK_END - map::STACK_BASE)),
            xex_heap: Mutex::new(PageHeap::new("xex", map::XEX_BASE, map::XEX_END - map::XEX_BASE)),
            physical: Mutex::new(PageHeap::new("physical", 0, map::PHYSICAL_SIZE)),
            system: Mutex::new(SystemHeap::new()),
        })
    }

    fn commit_ranges(&self, offset: u32, ranges: &[(u32, u32)]) -> AllocResult<()> {
        for &(addr, len) in ranges {
            self.mem.commit(offset + addr, len)?;
        }
        Ok(())
    }

    fn decommit_ranges(&self, offset: u32, ranges: &[(u32, u32)]) -> AllocResult<()> {
        for &(addr, len) in ranges {
            self.mem.decommit(offset + addr, len)?;
        }
        Ok(())
    }

    /// Frees the committed ranges of a released allocation (they stay
    /// accessible; see [`GuestMemory::release`]).
    fn release_ranges(&self, offset: u32, ranges: &[(u32, u32)]) -> AllocResult<()> {
        for &(addr, len) in ranges {
            self.mem.release(offset + addr, len)?;
        }
        Ok(())
    }

    fn finish(&self, offset: u32, a: Allocated) -> AllocResult<u32> {
        self.commit_ranges(offset, &a.newly_committed)?;
        Ok(a.address + offset)
    }

    // --- NtAllocateVirtualMemory heap --------------------------------------------------

    pub fn alloc_virtual(
        &self,
        size: u32,
        page_size: u32,
        commit: bool,
        protect: u32,
        top_down: bool,
    ) -> AllocResult<u32> {
        // 4 KiB-page requests prefer their own range, 64 KiB ones the main
        // heap; either spills into the other when its range is exhausted.
        let (first, second) = if page_size <= PAGE_4K {
            (&self.virtual_4k_heap, &self.virtual_heap)
        } else {
            (&self.virtual_heap, &self.virtual_4k_heap)
        };
        let r = first.lock().unwrap().alloc(size, page_size, commit, protect, top_down);
        let a = match r {
            Err(HeapError::NoSpace) => second.lock().unwrap().alloc(size, page_size, commit, protect, top_down)?,
            r => r?,
        };
        self.finish(0, a)
    }

    /// Fixed-address virtual allocation (or commit inside an existing reservation).
    /// Returns the page-aligned base and whether every page was already committed.
    pub fn alloc_virtual_fixed(&self, addr: u32, size: u32, commit: bool, protect: u32) -> AllocResult<(u32, bool)> {
        if region_of(addr) != Region::Virtual {
            return Err(AllocError::WrongRegion(addr));
        }
        let mut heap = self.virtual_heap_at(addr).lock().unwrap();
        let was_committed =
            (0..size.div_ceil(PAGE_SIZE)).all(|i| heap.state_of(addr + i * PAGE_SIZE) == Some(PageState::Committed));
        let a = heap.alloc_fixed(addr, size, commit, protect)?;
        drop(heap);
        Ok((self.finish(0, a)?, was_committed))
    }

    pub fn decommit_virtual(&self, addr: u32, size: u32) -> AllocResult<()> {
        let ranges = self.with_page_heap(addr, |h| h.decommit(addr, size))??;
        self.decommit_ranges(0, &ranges)
    }

    /// Releases the allocation based at `addr`; returns its size.
    pub fn release_virtual(&self, addr: u32) -> AllocResult<u32> {
        let (size, ranges) = self.with_page_heap(addr, |h| h.release(addr))??;
        self.release_ranges(0, &ranges)?;
        Ok(size)
    }

    fn virtual_heap_at(&self, addr: u32) -> &Mutex<PageHeap> {
        if addr >= map::VIRTUAL_4K_BASE { &self.virtual_4k_heap } else { &self.virtual_heap }
    }

    fn with_page_heap<R>(&self, addr: u32, f: impl FnOnce(&mut PageHeap) -> R) -> AllocResult<R> {
        let heap = match region_of(addr) {
            Region::Virtual => self.virtual_heap_at(addr),
            Region::Stack => &self.stack_heap,
            Region::Xex => &self.xex_heap,
            _ => return Err(AllocError::WrongRegion(addr)),
        };
        Ok(f(&mut heap.lock().unwrap()))
    }

    // --- Physical memory ----------------------------------------------------------------

    /// Allocates physical memory (one copy, at [`map::PHYSICAL_BASE`] plus the
    /// physical address) and returns its address in the view for `page_size`.
    /// `min_phys`/`max_phys` bound the physical address and `align` aligns it,
    /// as in Xenia; every view shows the allocation (see
    /// [`crate::memory::fold_view`]).
    pub fn alloc_physical(
        &self,
        size: u32,
        page_size: u32,
        min_phys: u32,
        max_phys: u32,
        align: u32,
        protect: u32,
    ) -> AllocResult<u32> {
        let page_size = if page_size <= PAGE_4K {
            PAGE_4K
        } else if page_size <= PAGE_64K {
            PAGE_64K
        } else {
            PAGE_16M
        };
        let view = physical_view_for_page_size(page_size);
        let size = size.div_ceil(page_size) * page_size;
        let align = align.div_ceil(page_size).max(1) * page_size;
        // Physical addresses the view can show.
        let shift = physical_view_offset(view);
        let view_size = if view == map::PHYS_VIEW_4K { map::PHYS_VIEW_4K_SIZE } else { map::PHYSICAL_SIZE };
        let min = min_phys.max(shift);
        let max = max_phys.min(map::PHYSICAL_SIZE - 1).min(view_size - 1 + shift);
        let a = self.physical.lock().unwrap().alloc_range(min, max, size, align, true, protect, true)?;
        self.commit_ranges(map::PHYSICAL_BASE, &a.newly_committed)?;
        Ok(view + a.address - shift)
    }

    pub fn free_physical(&self, addr: u32) -> AllocResult<u32> {
        if !matches!(region_of(addr), Region::Physical(_)) {
            return Err(AllocError::WrongRegion(addr));
        }
        let phys = self.physical_address(addr).ok_or(AllocError::WrongRegion(addr))?;
        let (size, ranges) = self.physical.lock().unwrap().release(phys)?;
        self.release_ranges(map::PHYSICAL_BASE, &ranges)?;
        Ok(size)
    }

    /// The guest address a physical (GPU) address is visible at: physical
    /// memory if that page is allocated, or else the identity-mapped XEX region
    /// (see [`AddressSpace::physical_address`]). Only the low 29 bits of `phys`
    /// count, as on hardware.
    pub fn translate_physical(&self, phys: u32) -> Option<u32> {
        let p = phys & (map::PHYSICAL_SIZE - 1);
        let canonical = map::PHYSICAL_BASE + p;
        if self.is_committed(canonical) {
            return Some(canonical);
        }
        let xex = map::XEX_BASE + p;
        self.is_committed(xex).then_some(xex)
    }

    /// MmGetPhysicalAddress (see [`physical_view_offset`]). XEX-region
    /// addresses are treated as identity-mapped onto physical memory like on
    /// hardware.
    pub fn physical_address(&self, addr: u32) -> Option<u32> {
        match region_of(addr) {
            Region::Physical(view) => {
                let phys = addr - view + physical_view_offset(view);
                (phys < map::PHYSICAL_SIZE).then_some(phys)
            }
            Region::Xex => Some(addr & (map::PHYSICAL_SIZE - 1)),
            _ => None,
        }
    }

    // --- Stacks -------------------------------------------------------------------------

    pub fn alloc_stack(&self, size: u32) -> AllocResult<StackAllocation> {
        let size = size.div_ceil(PAGE_4K) * PAGE_4K;
        let guard = PAGE_4K;
        let a = self.stack_heap.lock().unwrap().alloc(size + 2 * guard, PAGE_4K, true, protect::READWRITE, false)?;
        let alloc_base = self.finish(0, a)?;
        let mut heap = self.stack_heap.lock().unwrap();
        heap.protect(alloc_base, guard, protect::NOACCESS)?;
        heap.protect(alloc_base + guard + size, guard, protect::NOACCESS)?;
        drop(heap);
        // Make the guards fault on the host too, so an overflow stops at once
        // instead of silently corrupting the neighbouring stack.
        self.mem.decommit(alloc_base, guard)?;
        self.mem.decommit(alloc_base + guard + size, guard)?;
        Ok(StackAllocation { alloc_base, limit: alloc_base + guard, base: alloc_base + guard + size })
    }

    pub fn free_stack(&self, alloc_base: u32) -> AllocResult<()> {
        let (_, ranges) = self.stack_heap.lock().unwrap().release(alloc_base)?;
        self.decommit_ranges(0, &ranges)
    }

    // --- XEX region ---------------------------------------------------------------------

    /// Reserves and commits a fixed range in the XEX region (image, header copy,
    /// indirect-call table).
    pub fn alloc_xex(&self, addr: u32, size: u32) -> AllocResult<u32> {
        if region_of(addr) != Region::Xex {
            return Err(AllocError::WrongRegion(addr));
        }
        let a = self.xex_heap.lock().unwrap().alloc_fixed(addr, size, true, protect::READWRITE)?;
        self.finish(0, a)
    }

    /// Any free range in the XEX region.
    pub fn alloc_xex_anywhere(&self, size: u32) -> AllocResult<u32> {
        let a = self.xex_heap.lock().unwrap().alloc(size, PAGE_64K, true, protect::READWRITE, true)?;
        self.finish(0, a)
    }

    // --- System heap --------------------------------------------------------------------

    /// Zeroed kernel-owned memory.
    pub fn system_alloc(&self, size: u32, align: u32) -> AllocResult<u32> {
        let addr = self.system.lock().unwrap().alloc(size, align).ok_or(AllocError::Heap(HeapError::NoSpace))?;
        self.mem.fill(addr, size, 0);
        Ok(addr)
    }

    pub fn system_free(&self, addr: u32) -> AllocResult<u32> {
        self.system.lock().unwrap().free(addr).ok_or(AllocError::Heap(HeapError::NotAllocationBase))
    }

    pub fn system_alloc_size(&self, addr: u32) -> Option<u32> {
        self.system.lock().unwrap().size_of(addr)
    }

    // --- Queries ------------------------------------------------------------------------

    pub fn query(&self, addr: u32) -> Option<RegionInfo> {
        match region_of(addr) {
            Region::Virtual | Region::Stack | Region::Xex => {
                self.with_page_heap(addr, |h| h.query(addr)).ok().flatten()
            }
            Region::Physical(view) => {
                let phys = self.physical_address(addr)?;
                let mut info = self.physical.lock().unwrap().query(phys)?;
                // Back from physical addresses to this view's, without letting
                // a free run spill outside the view.
                let shift = physical_view_offset(view);
                let to_view = |p: u32| (view as u64 + p as u64).saturating_sub(shift as u64).max(view as u64);
                let start = to_view(info.base_address);
                let end = to_view(info.base_address + info.region_size).min(view as u64 + map::PHYSICAL_SIZE as u64);
                info.base_address = start as u32;
                info.region_size = (end - start) as u32;
                if info.state != PageState::Free {
                    info.allocation_base = to_view(info.allocation_base) as u32;
                }
                Some(info)
            }
            Region::System | Region::Mmio => Some(RegionInfo {
                base_address: addr & !(PAGE_SIZE - 1),
                allocation_base: if region_of(addr) == Region::System { map::SYSTEM_BASE } else { map::MMIO_BASE },
                allocation_protect: protect::READWRITE,
                region_size: PAGE_SIZE,
                state: PageState::Committed,
                protect: protect::READWRITE,
            }),
            Region::HostReserved => None,
        }
    }

    pub fn protect(&self, addr: u32, size: u32, new: u32) -> AllocResult<u32> {
        match region_of(addr) {
            Region::Physical(_) => {
                let phys = self.physical_address(addr).ok_or(AllocError::WrongRegion(addr))?;
                Ok(self.physical.lock().unwrap().protect(phys, size, new)?)
            }
            _ => self.with_page_heap(addr, |h| h.protect(addr, size, new))?.map_err(Into::into),
        }
    }

    pub fn allocation_size(&self, addr: u32) -> Option<u32> {
        match region_of(addr) {
            Region::Physical(_) => self.physical.lock().unwrap().allocation_size(self.physical_address(addr)?),
            Region::System => self.system_alloc_size(addr),
            _ => self.with_page_heap(addr, |h| h.allocation_size(addr)).ok().flatten(),
        }
    }

    pub fn is_committed(&self, addr: u32) -> bool {
        self.query(addr).is_some_and(|i| i.state == PageState::Committed)
    }

    /// Committed physical pages (for MmQueryStatistics).
    pub fn physical_pages_used(&self) -> u32 {
        self.physical.lock().unwrap().usage().1
    }

    /// Reserved bytes in the virtual heap.
    pub fn virtual_bytes_reserved(&self) -> u32 {
        (self.virtual_heap.lock().unwrap().usage().0 + self.virtual_4k_heap.lock().unwrap().usage().0) * PAGE_SIZE
    }
}
