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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

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
/// Recompiled code applies the same function to every access (`skyrt_ea` in
/// cpp/ppc_context.h); the two must stay identical.
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

    /// Host pointer for a guest address (after [`fold_view`]). Callers must
    /// ensure the range is committed.
    #[inline]
    pub fn ptr(&self, addr: u32) -> *mut u8 {
        std::ptr::with_exposed_provenance_mut(self.base + fold_view(addr) as usize)
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
        let rc = unsafe {
            crate::dirty::mark(start, (end - start as u64) as u32);
            libc::madvise(self.ptr(start) as *mut libc::c_void, (end - start as u64) as usize, libc::MADV_DONTNEED)
        };
        if rc != 0 {
            return Err(MemoryError::Os(std::io::Error::last_os_error().to_string()));
        }
        Ok(())
    }

    #[cfg(target_os = "emscripten")]
    fn commit_pages(&self, _start: u32, end: u64) -> Result<(), MemoryError> {
        host_heap_guard()?;
        const WASM_PAGE: u64 = 0x10000;
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
            unsafe {
                libc::madvise(self.ptr(start) as *mut libc::c_void, (end - start as u64) as usize, libc::MADV_DONTNEED);
            }
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
            unsafe {
                libc::madvise(self.ptr(start) as *mut libc::c_void, size, libc::MADV_DONTNEED);
            }
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
        self.read_bytes(addr, &mut b);
        u16::from_be_bytes(b)
    }

    #[inline]
    pub fn write_u16(&self, addr: u32, v: u16) {
        self.write_bytes(addr, &v.to_be_bytes());
    }

    #[inline]
    pub fn read_u32(&self, addr: u32) -> u32 {
        let mut b = [0u8; 4];
        self.read_bytes(addr, &mut b);
        u32::from_be_bytes(b)
    }

    #[inline]
    pub fn write_u32(&self, addr: u32, v: u32) {
        self.write_bytes(addr, &v.to_be_bytes());
    }

    #[inline]
    pub fn read_u64(&self, addr: u32) -> u64 {
        let mut b = [0u8; 8];
        self.read_bytes(addr, &mut b);
        u64::from_be_bytes(b)
    }

    #[inline]
    pub fn write_u64(&self, addr: u32, v: u64) {
        self.write_bytes(addr, &v.to_be_bytes());
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
        // SAFETY: bounds checked against the 4 GiB space; the range must be committed.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr(addr), out.as_mut_ptr(), out.len()) }
    }

    pub fn write_bytes(&self, addr: u32, data: &[u8]) {
        assert!(addr as u64 + data.len() as u64 <= GUEST_SPACE_SIZE, "guest write out of range");
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr(addr), data.len()) }
        crate::dirty::mark(addr, data.len() as u32);
    }

    pub fn fill(&self, addr: u32, len: u32, byte: u8) {
        assert!(addr as u64 + len as u64 <= GUEST_SPACE_SIZE, "guest fill out of range");
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(self.ptr(addr), byte, len as usize) }
        crate::dirty::mark(addr, len);
    }

    /// memmove within guest memory.
    pub fn copy_within(&self, dst: u32, src: u32, len: u32) {
        assert!(dst as u64 + len as u64 <= GUEST_SPACE_SIZE && src as u64 + len as u64 <= GUEST_SPACE_SIZE);
        // SAFETY: as above; `copy` handles overlap.
        unsafe { std::ptr::copy(self.ptr(src), self.ptr(dst), len as usize) }
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

/// [`fold_view`] as a table for recompiled code: `skyrt_ea(ea)` in cpp/ppc_context.h is
/// `ea + skyrt_ea_offset[ea >> 24]`. A load and an add per access measured ~25% fewer cycles in
/// guest worker threads than the compare-and-select form (natively, at the main menu), and the
/// views are 16 MiB-aligned, so one entry per 16 MiB covers them.
#[unsafe(no_mangle)]
pub static skyrt_ea_offset: [i32; 256] = {
    let mut t = [0i32; 256];
    let mut i = 0;
    while i < 256 {
        let ea = (i as u32) << 24;
        t[i] = fold_view(ea).wrapping_sub(ea) as i32;
        i += 1;
    }
    t
};

#[cfg(test)]
mod ea_table_tests {
    use super::*;

    #[test]
    fn table_matches_fold_view() {
        for ea in (0..=u32::MAX).step_by(0x1000).chain([0xBFFF_FFFF, 0xC000_0000, 0xDFFF_FFFF, 0xE000_0000, 0xFFFF_FFFF]) {
            let folded = ea.wrapping_add(skyrt_ea_offset[(ea >> 24) as usize] as u32);
            assert_eq!(folded, fold_view(ea), "{ea:#010X}");
        }
    }
}
