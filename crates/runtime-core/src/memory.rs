//! The 4 GiB guest address space.
//!
//! Recompiled code addresses guest memory as `base + ea` with no translation hook
//! for vector loads/stores, atomics and dcbz, so the whole 32-bit space must be
//! one flat range starting at `base`.
//!
//! * Native: a 4 GiB reservation whose first 3 GiB (everything below the folded
//!   physical views) is readable and writable from the start, like wasm linear
//!   memory: pages cost nothing until touched, and the title's stray accesses
//!   (reads past a commit, through null pointers) behave as in the browser
//!   instead of faulting.
//! * Web (wasm32): `base` is 0 and guest addresses *are* linear-memory
//!   addresses. Emscripten's static data, stacks and heap live below
//!   [`map::HOST_RESERVED_END`], a range the guest allocators never hand out.
//!   Committing grows linear memory far enough to cover the range.
//!
//! Page protections are tracked by the heaps for queries but not enforced on
//! either target, so both builds behave the same.

use std::fmt;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};

pub const PAGE_SIZE: u32 = 0x1000;

/// Bytes of guest memory committed and not decommitted or released (what the title holds; on
/// the web, roughly how much of linear memory it can have touched).
static COMMITTED: AtomicU64 = AtomicU64::new(0);

pub fn committed_bytes() -> u64 {
    COMMITTED.load(Ordering::Relaxed)
}

fn uncount(len: u32) {
    let _ = COMMITTED.try_update(Ordering::Relaxed, Ordering::Relaxed, |c| Some(c.saturating_sub(u64::from(len))));
}
pub const GUEST_SPACE_SIZE: u64 = 1 << 32;

/// Guest address-space layout shared by every allocator.
pub mod map {
    /// Everything below this is host-owned on wasm and never given to the guest
    /// (on native it stays unmapped so the two builds hand out the same addresses).
    pub const HOST_RESERVED_END: u32 = 0x4000_0000;
    /// NtAllocateVirtualMemory heap.
    pub const VIRTUAL_BASE: u32 = 0x4000_0000;
    pub const VIRTUAL_END: u32 = 0x7000_0000;
    /// Guest thread stacks (same range Xenia uses).
    pub const STACK_BASE: u32 = 0x7000_0000;
    pub const STACK_END: u32 = 0x7E00_0000;
    /// Kernel objects, PCRs, TLS blocks, pool allocations, exported variables.
    pub const SYSTEM_BASE: u32 = 0x7E00_0000;
    pub const SYSTEM_END: u32 = 0x7FC0_0000;
    /// GPU register aperture. Committed as plain memory so register writes from
    /// guest code land somewhere harmless until a GPU backend claims them.
    pub const MMIO_BASE: u32 = 0x7FC0_0000;
    pub const MMIO_END: u32 = 0x8000_0000;
    /// XEX image, its copied header, and XenonRecomp's indirect-call table.
    pub const XEX_BASE: u32 = 0x8000_0000;
    pub const XEX_END: u32 = 0x9000_0000;
    /// NtAllocateVirtualMemory heap for 4 KiB-page requests. On hardware (and
    /// in Xenia) these come from 0x00000000-0x3FFFFFFF, which the host owns
    /// here; titles reserve address space in both page sizes, so one shared
    /// range is too small (Skyrim reserves ~600 MiB of 64 KiB pages plus
    /// ~200 MiB of 4 KiB pages while loading).
    pub const VIRTUAL_4K_BASE: u32 = 0x9000_0000;
    pub const VIRTUAL_4K_END: u32 = 0xA000_0000;
    /// Physical memory is 512 MiB, exposed through three views that differ in
    /// page size and all show the same memory. It is stored once, at
    /// [`PHYSICAL_BASE`] + physical address (the 64 KiB view); addresses in the
    /// other two views are folded onto it on every access ([`super::fold_view`]),
    /// so nothing at or above [`PHYS_VIEW_16M`] is ever committed.
    pub const PHYSICAL_SIZE: u32 = 0x2000_0000;
    pub const PHYSICAL_BASE: u32 = PHYS_VIEW_64K;
    pub const PHYS_VIEW_64K: u32 = 0xA000_0000;
    pub const PHYS_VIEW_16M: u32 = 0xC000_0000;
    pub const PHYS_VIEW_4K: u32 = 0xE000_0000;
    /// The 4 KiB view stops at 0xFFD00000, as on hardware.
    pub const PHYS_VIEW_4K_SIZE: u32 = 0x1FD0_0000;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    OutOfRange {
        addr: u32,
        len: u64,
    },
    /// Commit inside a folded view (see [`fold_view`]).
    FoldedView {
        addr: u32,
    },
    HostReserved {
        addr: u32,
    },
    Os(String),
    GrowFailed {
        needed_end: u64,
    },
    HostHeapCollision {
        host_break: u64,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::OutOfRange { addr, len } => write!(f, "range {addr:#010X}+{len:#X} exceeds 4 GiB"),
            MemoryError::FoldedView { addr } => {
                write!(f, "address {addr:#010X} is in a physical view folded onto {:#010X}", map::PHYSICAL_BASE)
            }
            MemoryError::HostReserved { addr } => {
                write!(f, "address {addr:#010X} is inside the host-reserved range")
            }
            MemoryError::Os(e) => write!(f, "OS error: {e}"),
            MemoryError::GrowFailed { needed_end } => {
                write!(f, "could not grow linear memory to {needed_end:#X}")
            }
            MemoryError::HostHeapCollision { host_break } => {
                write!(f, "host heap break {host_break:#X} reached the guest range at {:#X}", map::HOST_RESERVED_END)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

pub struct GuestMemory {
    /// Address of guest address 0 in the host address space (0 on wasm).
    base: usize,
}

// The memory is shared by every guest thread; synchronization is the guest's job
// exactly as on hardware.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

/// Where guest address `addr` is stored: addresses in the 16 MiB view
/// (0xC0000000) and the 4 KiB view (0xE0000000, one page ahead of its physical
/// address) are folded onto the single copy of physical memory at
/// [`map::PHYSICAL_BASE`]; everything else is stored at its own address.
/// [`skyrt_ea_offset`] starts out as this mapping; recompiled code and
/// [`GuestMemory`] translate through the table.
#[inline]
pub const fn fold_view(addr: u32) -> u32 {
    if addr < map::PHYS_VIEW_16M {
        addr
    } else if addr < map::PHYS_VIEW_4K {
        map::PHYSICAL_BASE + (addr & (map::PHYSICAL_SIZE - 1))
    } else {
        map::PHYSICAL_BASE + PAGE_SIZE + (addr & (map::PHYSICAL_SIZE - 1))
    }
}

impl GuestMemory {
    /// Reserves the address space. Nothing is accessible until committed.
    pub fn reserve() -> Result<Self, MemoryError> {
        #[cfg(not(target_os = "emscripten"))]
        {
            // SAFETY: anonymous PROT_NONE reservation; no existing mapping is touched.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    GUEST_SPACE_SIZE as usize,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(MemoryError::Os(std::io::Error::last_os_error().to_string()));
            }
            // SAFETY: the start of the reservation made above.
            let rc = unsafe { libc::mprotect(ptr, map::PHYS_VIEW_16M as usize, libc::PROT_READ | libc::PROT_WRITE) };
            if rc != 0 {
                return Err(MemoryError::Os(std::io::Error::last_os_error().to_string()));
            }
            Ok(GuestMemory { base: (ptr as *mut u8).expose_provenance() })
        }
        #[cfg(target_os = "emscripten")]
        {
            Ok(GuestMemory { base: 0 })
        }
    }

    /// Host pointer that recompiled code receives as `base`.
    pub fn base_ptr(&self) -> *mut u8 {
        std::ptr::with_exposed_provenance_mut(self.base)
    }

    #[inline]
    fn check(addr: u32, len: u64) -> Result<(), MemoryError> {
        if addr as u64 + len > GUEST_SPACE_SIZE {
            return Err(MemoryError::OutOfRange { addr, len });
        }
        Ok(())
    }

    /// Host pointer for a guest address (see [`linear`]). Callers must ensure the range is
    /// committed, and that it doesn't leave the address's frame unless they know the layout
    /// keeps it contiguous: use [`GuestMemory::runs`] for ranges.
    #[inline]
    pub fn ptr(&self, addr: u32) -> *mut u8 {
        self.linear_ptr(linear(addr))
    }

    #[inline]
    fn linear_ptr(&self, lin: u32) -> *mut u8 {
        std::ptr::with_exposed_provenance_mut(self.base + lin as usize)
    }

    /// Calls `f(offset, host pointer, len)` for each piece of `[addr, addr + len)` that is
    /// contiguous in host memory (usually one), in order.
    pub fn runs(&self, addr: u32, len: u32, mut f: impl FnMut(usize, *mut u8, usize)) {
        linear_runs(addr, u64::from(len), |off, lin, n| f(off as usize, self.linear_ptr(lin), n as usize));
    }

    /// Makes `[addr, addr+len)` readable and writable. Newly committed pages are
    /// zero; already committed pages keep their contents.
    pub fn commit(&self, addr: u32, len: u32) -> Result<(), MemoryError> {
        Self::check(addr, len as u64)?;
        // The 16 MiB and 4 KiB views are folded onto PHYSICAL_BASE; their own
        // addresses never hold memory.
        if addr as u64 + len as u64 > map::PHYS_VIEW_16M as u64 {
            return Err(MemoryError::FoldedView { addr });
        }
        if addr < map::HOST_RESERVED_END {
            return Err(MemoryError::HostReserved { addr });
        }
        if len == 0 {
            return Ok(());
        }
        let start = addr & !(PAGE_SIZE - 1);
        let end = (addr as u64 + len as u64 + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
        self.commit_pages(start, end)?;
        COMMITTED.fetch_add(end - u64::from(start), Ordering::Relaxed);
        Ok(())
    }

    #[cfg(not(target_os = "emscripten"))]
    fn commit_pages(&self, start: u32, end: u64) -> Result<(), MemoryError> {
        // SAFETY: the range lies inside our own (already accessible) reservation and is
        // only being committed now, so nothing live is in it. MADV_DONTNEED makes it read
        // back as zero even if the guest wrote to it while uncommitted or after releasing
        // it (see `release`).
        crate::dirty::mark(start, (end - start as u64) as u32);
        let mut result = Ok(());
        linear_runs(start, end - u64::from(start), |_, lin, n| {
            let rc = unsafe { libc::madvise(self.linear_ptr(lin) as *mut libc::c_void, n as usize, libc::MADV_DONTNEED) };
            if rc != 0 && result.is_ok() {
                result = Err(MemoryError::Os(std::io::Error::last_os_error().to_string()));
            }
        });
        result
    }

    #[cfg(target_os = "emscripten")]
    fn commit_pages(&self, start: u32, end: u64) -> Result<(), MemoryError> {
        host_heap_guard()?;
        const WASM_PAGE: u64 = 0x10000;
        let mut end_linear = 0u64;
        linear_runs(start, end - u64::from(start), |_, lin, n| end_linear = end_linear.max(u64::from(lin) + n));
        let end = end_linear;
        let needed_pages = end.div_ceil(WASM_PAGE) as usize;
        loop {
            let current = core::arch::wasm32::memory_size(0);
            if current >= needed_pages {
                return Ok(());
            }
            if core::arch::wasm32::memory_grow(0, needed_pages - current) == usize::MAX {
                // Another thread may have grown it concurrently; re-check before failing.
                if core::arch::wasm32::memory_size(0) >= needed_pages {
                    return Ok(());
                }
                return Err(MemoryError::GrowFailed { needed_end: end });
            }
        }
    }

    /// Frees `[addr, addr+len)` after the guest released it: the contents are
    /// dropped (they read back as zero) but the pages stay accessible. Titles
    /// touch memory after freeing it (Skyrim's loader reads a vertex buffer
    /// whose physical memory was just freed); Xenia likewise leaves released
    /// memory mapped (`protect_on_release` is off by default).
    pub fn release(&self, addr: u32, len: u32) -> Result<(), MemoryError> {
        Self::check(addr, len as u64)?;
        if len == 0 {
            return Ok(());
        }
        uncount(len);
        #[cfg(not(target_os = "emscripten"))]
        {
            let start = addr & !(PAGE_SIZE - 1);
            let end = (addr as u64 + len as u64 + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
            // SAFETY: inside our reservation; MADV_DONTNEED returns the memory to
            // the host and leaves the mapping (and its protection) in place.
            linear_runs(start, end - u64::from(start), |_, lin, n| unsafe {
                libc::madvise(self.linear_ptr(lin) as *mut libc::c_void, n as usize, libc::MADV_DONTNEED);
            });
            crate::dirty::mark(start, (end - start as u64) as u32);
        }
        #[cfg(target_os = "emscripten")]
        self.fill(addr, len, 0);
        Ok(())
    }

    /// Discards the contents of `[addr, addr+len)`: it reads back as zero and
    /// stays accessible, natively as on wasm (which cannot release linear memory).
    pub fn decommit(&self, addr: u32, len: u32) -> Result<(), MemoryError> {
        Self::check(addr, len as u64)?;
        if len == 0 {
            return Ok(());
        }
        uncount(len);
        #[cfg(not(target_os = "emscripten"))]
        {
            let start = addr & !(PAGE_SIZE - 1);
            let end = (addr as u64 + len as u64 + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
            let size = (end - start as u64) as usize;
            // SAFETY: inside our reservation; MADV_DONTNEED drops the pages, which then
            // read back as zero.
            linear_runs(start, end - u64::from(start), |_, lin, n| unsafe {
                libc::madvise(self.linear_ptr(lin) as *mut libc::c_void, n as usize, libc::MADV_DONTNEED);
            });
            crate::dirty::mark(start, size as u32);
        }
        #[cfg(target_os = "emscripten")]
        self.fill(addr, len, 0);
        Ok(())
    }

    // --- Big-endian scalar accessors -------------------------------------------------

    #[inline]
    pub fn read_u8(&self, addr: u32) -> u8 {
        // SAFETY: caller-committed guest memory; unaligned reads are fine for bytes.
        unsafe { self.ptr(addr).read_volatile() }
    }

    #[inline]
    pub fn write_u8(&self, addr: u32, v: u8) {
        // SAFETY: as above.
        unsafe { self.ptr(addr).write_volatile(v) }
        crate::dirty::mark(addr, 1);
    }

    #[inline]
    pub fn read_u16(&self, addr: u32) -> u16 {
        let mut b = [0u8; 2];
        if within_frame(addr, 2) {
            // SAFETY: committed guest memory inside one frame.
            b = unsafe { self.ptr(addr).cast::<[u8; 2]>().read_unaligned() };
        } else {
            self.read_bytes(addr, &mut b);
        }
        u16::from_be_bytes(b)
    }

    #[inline]
    pub fn write_u16(&self, addr: u32, v: u16) {
        if within_frame(addr, 2) {
            // SAFETY: as above.
            unsafe { self.ptr(addr).cast::<[u8; 2]>().write_unaligned(v.to_be_bytes()) };
            crate::dirty::mark(addr, 2);
        } else {
            self.write_bytes(addr, &v.to_be_bytes());
        }
    }

    #[inline]
    pub fn read_u32(&self, addr: u32) -> u32 {
        let mut b = [0u8; 4];
        if within_frame(addr, 4) {
            // SAFETY: committed guest memory inside one frame.
            b = unsafe { self.ptr(addr).cast::<[u8; 4]>().read_unaligned() };
        } else {
            self.read_bytes(addr, &mut b);
        }
        u32::from_be_bytes(b)
    }

    #[inline]
    pub fn write_u32(&self, addr: u32, v: u32) {
        if within_frame(addr, 4) {
            // SAFETY: as above.
            unsafe { self.ptr(addr).cast::<[u8; 4]>().write_unaligned(v.to_be_bytes()) };
            crate::dirty::mark(addr, 4);
        } else {
            self.write_bytes(addr, &v.to_be_bytes());
        }
    }

    #[inline]
    pub fn read_u64(&self, addr: u32) -> u64 {
        let mut b = [0u8; 8];
        if within_frame(addr, 8) {
            // SAFETY: committed guest memory inside one frame.
            b = unsafe { self.ptr(addr).cast::<[u8; 8]>().read_unaligned() };
        } else {
            self.read_bytes(addr, &mut b);
        }
        u64::from_be_bytes(b)
    }

    #[inline]
    pub fn write_u64(&self, addr: u32, v: u64) {
        if within_frame(addr, 8) {
            // SAFETY: as above.
            unsafe { self.ptr(addr).cast::<[u8; 8]>().write_unaligned(v.to_be_bytes()) };
            crate::dirty::mark(addr, 8);
        } else {
            self.write_bytes(addr, &v.to_be_bytes());
        }
    }

    #[inline]
    pub fn read_i32(&self, addr: u32) -> i32 {
        self.read_u32(addr) as i32
    }

    #[inline]
    pub fn write_i32(&self, addr: u32, v: i32) {
        self.write_u32(addr, v as u32);
    }

    #[inline]
    pub fn read_i64(&self, addr: u32) -> i64 {
        self.read_u64(addr) as i64
    }

    #[inline]
    pub fn write_i64(&self, addr: u32, v: i64) {
        self.write_u64(addr, v as u64);
    }

    #[inline]
    pub fn read_f32(&self, addr: u32) -> f32 {
        f32::from_bits(self.read_u32(addr))
    }

    #[inline]
    pub fn write_f32(&self, addr: u32, v: f32) {
        self.write_u32(addr, v.to_bits());
    }

    #[inline]
    pub fn read_f64(&self, addr: u32) -> f64 {
        f64::from_bits(self.read_u64(addr))
    }

    #[inline]
    pub fn write_f64(&self, addr: u32, v: f64) {
        self.write_u64(addr, v.to_bits());
    }

    // --- Bulk access -----------------------------------------------------------------

    pub fn read_bytes(&self, addr: u32, out: &mut [u8]) {
        assert!(addr as u64 + out.len() as u64 <= GUEST_SPACE_SIZE, "guest read out of range");
        let len = out.len() as u32;
        if within_frame(addr, len) {
            // SAFETY: bounds checked against the 4 GiB space; the range must be committed.
            unsafe { std::ptr::copy_nonoverlapping(self.ptr(addr), out.as_mut_ptr(), out.len()) };
            return;
        }
        // SAFETY: as above.
        self.runs(addr, len, |off, p, n| unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr().add(off), n) });
    }

    pub fn write_bytes(&self, addr: u32, data: &[u8]) {
        assert!(addr as u64 + data.len() as u64 <= GUEST_SPACE_SIZE, "guest write out of range");
        if within_frame(addr, data.len() as u32) {
            // SAFETY: as above.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr(addr), data.len()) };
        } else {
            // SAFETY: as above.
            self.runs(addr, data.len() as u32, |off, p, n| unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(off), p, n) });
        }
        crate::dirty::mark(addr, data.len() as u32);
    }

    pub fn fill(&self, addr: u32, len: u32, byte: u8) {
        assert!(addr as u64 + len as u64 <= GUEST_SPACE_SIZE, "guest fill out of range");
        // SAFETY: as above.
        self.runs(addr, len, |_, p, n| unsafe { std::ptr::write_bytes(p, byte, n) });
        crate::dirty::mark(addr, len);
    }

    /// memmove within guest memory.
    pub fn copy_within(&self, dst: u32, src: u32, len: u32) {
        assert!(dst as u64 + len as u64 <= GUEST_SPACE_SIZE && src as u64 + len as u64 <= GUEST_SPACE_SIZE);
        let contiguous = |a: u32| {
            let mut pieces = 0;
            linear_runs(a, u64::from(len), |_, _, _| pieces += 1);
            pieces <= 1
        };
        if contiguous(src) && contiguous(dst) {
            // SAFETY: as above; `copy` handles overlap.
            unsafe { std::ptr::copy(self.ptr(src), self.ptr(dst), len as usize) }
        } else {
            let tmp = self.read_vec(src, len);
            self.write_bytes(dst, &tmp);
        }
        crate::dirty::mark(dst, len);
    }

    pub fn read_vec(&self, addr: u32, len: u32) -> Vec<u8> {
        let mut v = vec![0u8; len as usize];
        self.read_bytes(addr, &mut v);
        v
    }

    /// NUL-terminated byte string, at most `max` bytes.
    pub fn read_cstr(&self, addr: u32, max: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for i in 0..max {
            let b = self.read_u8(addr.wrapping_add(i));
            if b == 0 {
                break;
            }
            out.push(b);
        }
        out
    }

    /// `count` big-endian UTF-16 code units.
    pub fn read_utf16(&self, addr: u32, count: u32) -> Vec<u16> {
        (0..count).map(|i| self.read_u16(addr + i * 2)).collect()
    }

    pub fn write_utf16(&self, addr: u32, units: &[u16]) {
        for (i, &u) in units.iter().enumerate() {
            self.write_u16(addr + i as u32 * 2, u);
        }
    }

    // --- Atomics on big-endian words ------------------------------------------------
    //
    // Guest lwarx/stwcx. compile to host CAS on the raw (byte-swapped) word, so these
    // interoperate with recompiled code touching the same location.

    fn atomic_u32(&self, addr: u32) -> &AtomicU32 {
        assert!(addr.is_multiple_of(4), "unaligned atomic access at {addr:#010X}");
        // SAFETY: aligned, committed guest word; AtomicU32 has the layout of u32.
        unsafe { &*(self.ptr(addr) as *const AtomicU32) }
    }

    /// Big-endian compare-and-swap; returns whether the swap happened.
    pub fn compare_exchange_u32(&self, addr: u32, current: u32, new: u32) -> bool {
        let swapped = self
            .atomic_u32(addr)
            .compare_exchange(current.swap_bytes_be(), new.swap_bytes_be(), Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if swapped {
            crate::dirty::mark(addr, 4);
        }
        swapped
    }

    /// Atomically adds `delta` to a big-endian word, returning the new value.
    pub fn atomic_add_i32(&self, addr: u32, delta: i32) -> i32 {
        let cell = self.atomic_u32(addr);
        let mut raw = cell.load(Ordering::SeqCst);
        loop {
            let new = (u32::from_be_raw(raw) as i32).wrapping_add(delta);
            match cell.compare_exchange_weak(raw, (new as u32).swap_bytes_be(), Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    crate::dirty::mark(addr, 4);
                    return new;
                }
                Err(actual) => raw = actual,
            }
        }
    }

    pub fn atomic_load_u32(&self, addr: u32) -> u32 {
        u32::from_be_raw(self.atomic_u32(addr).load(Ordering::SeqCst))
    }

    pub fn atomic_store_u32(&self, addr: u32, v: u32) {
        self.atomic_u32(addr).store(v.swap_bytes_be(), Ordering::SeqCst);
        crate::dirty::mark(addr, 4);
    }

    pub fn atomic_exchange_u32(&self, addr: u32, v: u32) -> u32 {
        let old = u32::from_be_raw(self.atomic_u32(addr).swap(v.swap_bytes_be(), Ordering::SeqCst));
        crate::dirty::mark(addr, 4);
        old
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        #[cfg(not(target_os = "emscripten"))]
        // SAFETY: unmaps exactly the reservation created in `reserve`.
        unsafe {
            libc::munmap(self.base_ptr() as *mut libc::c_void, GUEST_SPACE_SIZE as usize);
        }
    }
}

/// Conversions between a value and its in-memory big-endian representation read
/// as a native integer.
trait BeRaw {
    fn swap_bytes_be(self) -> u32;
    fn from_be_raw(raw: u32) -> u32;
}

impl BeRaw for u32 {
    #[inline]
    fn swap_bytes_be(self) -> u32 {
        u32::from_ne_bytes(self.to_be_bytes())
    }

    #[inline]
    fn from_be_raw(raw: u32) -> u32 {
        u32::from_be_bytes(raw.to_ne_bytes())
    }
}

/// Top of the host heap (the wasm sbrk break; 0 natively, where the host heap is not in the
/// guest address space).
pub fn host_heap_top() -> u64 {
    #[cfg(target_os = "emscripten")]
    {
        unsafe extern "C" {
            fn sbrk(increment: isize) -> *mut libc::c_void;
        }
        // SAFETY: sbrk(0) only queries the current break.
        unsafe { sbrk(0) as usize as u64 }
    }
    #[cfg(not(target_os = "emscripten"))]
    0
}

/// On wasm the host heap shares linear memory with the guest. Emscripten's sbrk
/// only checks the current memory size, which guest commits push far above the
/// host range, so the break must be checked against the guest boundary directly.
#[cfg(target_os = "emscripten")]
pub fn host_heap_guard() -> Result<(), MemoryError> {
    unsafe extern "C" {
        fn sbrk(increment: isize) -> *mut libc::c_void;
    }
    // SAFETY: sbrk(0) only queries the current break.
    let brk = unsafe { sbrk(0) } as usize as u64;
    if brk >= map::HOST_RESERVED_END as u64 {
        return Err(MemoryError::HostHeapCollision { host_break: brk });
    }
    Ok(())
}

#[cfg(not(target_os = "emscripten"))]
pub fn host_heap_guard() -> Result<(), MemoryError> {
    Ok(())
}

// --- Guest-to-linear translation ----------------------------------------------------------
//
// Every guest address is stored at `linear(addr) = addr + skyrt_ea_offset[addr >> FRAME_SHIFT]`
// (relative to `base`): recompiled code does the same lookup on every access (`skyrt_ea` in
// cpp/ppc_context.h), and so does everything here. One entry per 1 MiB frame. The table holds
// the physical-view folding ([`fold_view`]) and is the hook for laying guest memory out
// compactly in linear memory: frames that are adjacent in the guest needn't be adjacent in
// linear memory, so ranges are processed in runs of linearly contiguous frames
// ([`GuestMemory::runs`]).

pub const FRAME_SHIFT: u32 = 20;
pub const FRAME_SIZE: u32 = 1 << FRAME_SHIFT;
pub const FRAMES: usize = 1 << (32 - FRAME_SHIFT);

/// Linear offset of each guest frame. A load and an add per access measured ~25% fewer cycles
/// in guest worker threads than the old compare-and-select fold (natively, at the main menu);
/// 1 MiB entries cost the same as 16 MiB ones there.
#[unsafe(no_mangle)]
pub static skyrt_ea_offset: [AtomicI32; FRAMES] = {
    let mut t = [const { AtomicI32::new(0) }; FRAMES];
    let mut i = 0;
    while i < FRAMES {
        let ea = (i as u32) << FRAME_SHIFT;
        t[i] = AtomicI32::new(fold_view(ea).wrapping_sub(ea) as i32);
        i += 1;
    }
    t
};

/// Whether `[addr, addr + len)` lies in one frame (and so is contiguous in linear memory).
#[inline]
fn within_frame(addr: u32, len: u32) -> bool {
    (addr & (FRAME_SIZE - 1)) as u64 + u64::from(len) <= u64::from(FRAME_SIZE)
}

/// Where guest address `addr` is stored, relative to `base`.
#[inline]
pub fn linear(addr: u32) -> u32 {
    addr.wrapping_add(skyrt_ea_offset[(addr >> FRAME_SHIFT) as usize].load(Ordering::Relaxed) as u32)
}

/// Calls `f(offset, linear_start, len)` for each maximal piece of the guest range
/// `[addr, addr + len)` whose frames are contiguous in linear memory, in order.
pub fn linear_runs(addr: u32, len: u64, mut f: impl FnMut(u64, u32, u64)) {
    let mut done = 0u64;
    while done < len {
        let start = addr.wrapping_add(done as u32);
        let lin = linear(start);
        let mut n = (u64::from(FRAME_SIZE - (start & (FRAME_SIZE - 1)))).min(len - done);
        while done + n < len {
            let next = addr.wrapping_add((done + n) as u32);
            if next == 0 || linear(next) != lin.wrapping_add(n as u32) {
                break;
            }
            n = (n + u64::from(FRAME_SIZE)).min(len - done);
        }
        f(done, lin, n);
        done += n;
    }
}

#[cfg(test)]
mod translation_tests {
    use super::*;

    #[test]
    fn identity_layout_matches_fold_view() {
        for ea in (0..=u32::MAX).step_by(0x1000).chain([0xBFFF_FFFF, 0xC000_0000, 0xDFFF_FFFF, 0xE000_0000, 0xFFFF_FFFF]) {
            assert_eq!(linear(ea), fold_view(ea), "{ea:#010X}");
        }
    }

    #[test]
    fn runs_split_where_linear_memory_is_not_contiguous() {
        let mut runs = Vec::new();
        // 0xBFF00000 (identity) is followed by 0xC0000000, which folds onto 0xA0000000.
        linear_runs(0xBFE0_0000, 0x40_0000, |off, lin, n| runs.push((off, lin, n)));
        assert_eq!(runs, [(0, 0xBFE0_0000, 0x20_0000), (0x20_0000, 0xA000_0000, 0x20_0000)]);
        runs.clear();
        linear_runs(0x4000_0800, 0x30_0000, |off, lin, n| runs.push((off, lin, n)));
        assert_eq!(runs, [(0, 0x4000_0800, 0x30_0000)]);
    }
}
