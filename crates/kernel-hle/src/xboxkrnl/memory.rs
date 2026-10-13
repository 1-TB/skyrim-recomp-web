//! Virtual/physical memory and pool exports.
#![allow(non_snake_case)]

use runtime_core::address_space::{AllocError, PAGE_4K, PAGE_16M, PAGE_64K, Region, region_of};
use runtime_core::heap::{HeapError, PageState, protect};
use runtime_core::memory::map;

use crate::dispatch::{Call, Export, ModuleId};
use crate::status;

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x0C2, MmMapIoSpace),
    export!(K, 0x0CC, NtAllocateVirtualMemory),
    export!(K, 0x0DC, NtFreeVirtualMemory),
    export!(K, 0x0EE, NtQueryVirtualMemory),
    export!(K, 0x0E1, NtProtectVirtualMemory),
    export!(K, 0x0BA, MmAllocatePhysicalMemoryEx),
    export!(K, 0x0B9, MmAllocatePhysicalMemory),
    export!(K, 0x0BD, MmFreePhysicalMemory),
    export!(K, 0x0BE, MmGetPhysicalAddress),
    export!(K, 0x0C4, MmQueryAddressProtect),
    export!(K, 0x0C7, MmSetAddressProtect),
    export!(K, 0x0C5, MmQueryAllocationSize),
    export!(K, 0x0C6, MmQueryStatistics),
    export!(K, 0x009, ExAllocatePool),
    export!(K, 0x00A, ExAllocatePoolWithTag),
    export!(K, 0x00B, ExAllocatePoolTypeWithTag),
    export!(K, 0x00F, ExFreePool),
];

pub const MEM_COMMIT: u32 = 0x0000_1000;
pub const MEM_RESERVE: u32 = 0x0000_2000;
pub const MEM_DECOMMIT: u32 = 0x0000_4000;
pub const MEM_RELEASE: u32 = 0x0000_8000;
pub const MEM_FREE: u32 = 0x0001_0000;
pub const MEM_PRIVATE: u32 = 0x0002_0000;
pub const MEM_RESET: u32 = 0x0008_0000;
pub const MEM_TOP_DOWN: u32 = 0x0010_0000;
pub const MEM_LARGE_PAGES: u32 = 0x2000_0000;
pub const MEM_16MB_PAGES: u32 = 0x8000_0000;

const EXECUTE_BITS: u32 =
    protect::EXECUTE | protect::EXECUTE_READ | protect::EXECUTE_READWRITE | protect::EXECUTE_WRITECOPY;
/// Protection bits stored in page tables (strips the allocation-type bits that
/// MmAllocatePhysicalMemoryEx passes in the same word).
const PROTECT_MASK: u32 = 0x7FF;

fn NtAllocateVirtualMemory(c: &mut Call) -> u64 {
    let (base_ptr, size_ptr, alloc_type, protect_bits, debug_memory) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let mem = c.mem();
    if base_ptr == 0 || size_ptr == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let requested_base = mem.read_u32(base_ptr);
    let requested_size = mem.read_i32(size_ptr);
    if requested_size == 0 || alloc_type & (MEM_COMMIT | MEM_RESERVE | MEM_RESET) == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    if alloc_type & MEM_RESET != 0 && alloc_type & !MEM_RESET != 0 {
        return status::INVALID_PARAMETER as u64;
    }
    if debug_memory != 0 {
        log::warn!("NtAllocateVirtualMemory: devkit debug memory requested; using title memory");
    }
    if protect_bits & EXECUTE_BITS != 0 {
        log::warn!(
            "NtAllocateVirtualMemory: execute protection {protect_bits:#X} requested (code is never generated at runtime)"
        );
    }

    // Some titles pass negated sizes.
    let size = requested_size.unsigned_abs();
    let page_size = if requested_base != 0 || alloc_type & MEM_LARGE_PAGES == 0 { PAGE_4K } else { PAGE_64K };
    let base = requested_base - requested_base % page_size;
    let granule = if base != 0 { page_size } else { PAGE_64K };
    // Every page touched by [requested_base, requested_base + size).
    let size = ((requested_base - base) as u64 + size as u64).div_ceil(granule as u64) * granule as u64;
    let Ok(size) = u32::try_from(size) else { return status::INVALID_PARAMETER as u64 };
    let prot = protect_bits & PROTECT_MASK;

    if alloc_type == MEM_RESET {
        // MEM_RESET only declares the contents disposable; keeping them is valid.
        mem.write_u32(base_ptr, base);
        mem.write_u32(size_ptr, size);
        return status::SUCCESS as u64;
    }

    let commit = alloc_type & MEM_COMMIT != 0;
    let space = &c.kernel.rt.space;
    let result = if base != 0 {
        if region_of(base) != Region::Virtual {
            log::warn!(
                "NtAllocateVirtualMemory: fixed address {base:#010X} is outside the virtual heaps {:#010X}-{:#010X} and {:#010X}-{:#010X}",
                map::VIRTUAL_BASE,
                map::VIRTUAL_END,
                map::VIRTUAL_4K_BASE,
                map::VIRTUAL_4K_END
            );
            return status::INVALID_PARAMETER as u64;
        }
        space.alloc_virtual_fixed(base, size, commit, prot).map(|(a, _)| a)
    } else {
        space.alloc_virtual(size, page_size, commit, prot, alloc_type & MEM_TOP_DOWN != 0)
    };
    match result {
        Ok(addr) => {
            log::debug!("NtAllocateVirtualMemory({requested_base:#010X}, {size:#X}, {alloc_type:#X}) -> {addr:#010X}");
            mem.write_u32(base_ptr, addr);
            mem.write_u32(size_ptr, size);
            status::SUCCESS as u64
        }
        Err(e) => {
            log::warn!("NtAllocateVirtualMemory({requested_base:#010X}, {size:#X}, {alloc_type:#X}) failed: {e}");
            status::NO_MEMORY as u64
        }
    }
}

fn NtFreeVirtualMemory(c: &mut Call) -> u64 {
    let (base_ptr, size_ptr, free_type) = (c.arg(0), c.arg(1), c.arg(2));
    let mem = c.mem();
    if base_ptr == 0 || size_ptr == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let base = mem.read_u32(base_ptr);
    let size = mem.read_u32(size_ptr);
    if base == 0 {
        return status::MEMORY_NOT_ALLOCATED as u64;
    }
    if region_of(base) != Region::Virtual {
        return status::INVALID_PARAMETER as u64;
    }
    let space = &c.kernel.rt.space;
    let (base, size, r) = match free_type {
        MEM_DECOMMIT => {
            let start = base & !(PAGE_4K - 1);
            let end = (base as u64 + size as u64).div_ceil(PAGE_4K as u64) * PAGE_4K as u64;
            let len = (end - start as u64) as u32;
            (start, len, space.decommit_virtual(start, len))
        }
        MEM_RELEASE => match space.release_virtual(base) {
            Ok(released) => (base, released, Ok(())),
            Err(e) => (base, size, Err(e)),
        },
        _ => return status::INVALID_PARAMETER as u64,
    };
    log::debug!("NtFreeVirtualMemory({base:#010X}, {size:#X}, {free_type:#X}) -> {r:?}");
    match r {
        Ok(()) => {
            mem.write_u32(base_ptr, base);
            mem.write_u32(size_ptr, size);
            status::SUCCESS as u64
        }
        Err(AllocError::Heap(HeapError::NotAllocationBase)) | Err(AllocError::Heap(HeapError::NotAllocated)) => {
            status::MEMORY_NOT_ALLOCATED as u64
        }
        Err(_) => status::UNSUCCESSFUL as u64,
    }
}

fn state_value(s: PageState) -> u32 {
    match s {
        PageState::Free => MEM_FREE,
        PageState::Reserved => MEM_RESERVE,
        PageState::Committed => MEM_COMMIT,
    }
}

fn NtQueryVirtualMemory(c: &mut Call) -> u64 {
    let (addr, info, region_type) = (c.arg(0), c.arg(1), c.arg(2));
    if region_type > 2 || info == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let Some(r) = c.kernel.rt.space.query(addr) else {
        return status::INVALID_PARAMETER as u64;
    };
    let mem = c.mem();
    let or_noaccess = |p: u32| if p == 0 { protect::NOACCESS } else { p };
    mem.write_u32(info, r.base_address);
    mem.write_u32(info + 4, r.allocation_base);
    mem.write_u32(info + 8, if r.state == PageState::Free { 0 } else { or_noaccess(r.allocation_protect) });
    mem.write_u32(info + 12, r.region_size);
    mem.write_u32(info + 16, state_value(r.state));
    mem.write_u32(info + 20, or_noaccess(r.protect));
    mem.write_u32(info + 24, if r.state == PageState::Free { 0 } else { MEM_PRIVATE });
    status::SUCCESS as u64
}

fn NtProtectVirtualMemory(c: &mut Call) -> u64 {
    let (base_ptr, size_ptr, new_protect, old_ptr) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3));
    let mem = c.mem();
    if base_ptr == 0 || size_ptr == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    if new_protect & EXECUTE_BITS != 0 {
        return status::INVALID_PAGE_PROTECTION as u64;
    }
    let base = mem.read_u32(base_ptr);
    let size = mem.read_u32(size_ptr);
    if size == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let start = base & !(PAGE_4K - 1);
    let len = ((base as u64 + size as u64).div_ceil(PAGE_4K as u64) * PAGE_4K as u64 - start as u64) as u32;
    match c.kernel.rt.space.protect(start, len, new_protect & PROTECT_MASK) {
        Ok(old) => {
            mem.write_u32(base_ptr, start);
            mem.write_u32(size_ptr, len);
            if old_ptr != 0 {
                mem.write_u32(old_ptr, old);
            }
            status::SUCCESS as u64
        }
        Err(_) => status::ACCESS_DENIED as u64,
    }
}

fn physical_alloc(c: &Call, size: u32, protect_bits: u32, min: u32, max: u32, align: u32) -> u32 {
    if protect_bits & (protect::READONLY | protect::READWRITE) == 0 {
        log::warn!("MmAllocatePhysicalMemoryEx: protection {protect_bits:#X} has neither READONLY nor READWRITE");
        return 0;
    }
    let page_size = if protect_bits & MEM_LARGE_PAGES != 0 {
        PAGE_64K
    } else if protect_bits & MEM_16MB_PAGES != 0 {
        PAGE_16M
    } else {
        PAGE_4K
    };
    match c.kernel.rt.space.alloc_physical(size, page_size, min, max, align, protect_bits & PROTECT_MASK) {
        Ok(a) => a,
        Err(e) => {
            log::warn!("MmAllocatePhysicalMemoryEx({size:#X}, page {page_size:#X}, {min:#X}-{max:#X}) failed: {e}");
            0
        }
    }
}

fn MmAllocatePhysicalMemoryEx(c: &mut Call) -> u64 {
    physical_alloc(c, c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5)) as u64
}

fn MmAllocatePhysicalMemory(c: &mut Call) -> u64 {
    physical_alloc(c, c.arg(1), c.arg(2), 0, u32::MAX, 0) as u64
}

fn MmFreePhysicalMemory(c: &mut Call) -> u64 {
    let addr = c.arg(1);
    if let Err(e) = c.kernel.rt.space.free_physical(addr) {
        log::warn!("MmFreePhysicalMemory({addr:#010X}): {e}");
    }
    0
}

fn MmGetPhysicalAddress(c: &mut Call) -> u64 {
    let addr = c.arg(0);
    c.kernel.rt.space.physical_address(addr).unwrap_or_else(|| {
        log::warn!("MmGetPhysicalAddress({addr:#010X}): not physical memory");
        0
    }) as u64
}

fn MmQueryAddressProtect(c: &mut Call) -> u64 {
    c.kernel.rt.space.query(c.arg(0)).filter(|r| r.state == PageState::Committed).map_or(0, |r| r.protect) as u64
}

fn MmSetAddressProtect(c: &mut Call) -> u64 {
    let (addr, size, prot) = (c.arg(0), c.arg(1), c.arg(2));
    let required =
        protect::NOACCESS | protect::READONLY | protect::READWRITE | protect::EXECUTE_READ | protect::EXECUTE_READWRITE;
    if (prot & required).count_ones() != 1 {
        return 0;
    }
    if region_of(addr) != Region::Xex
        && let Err(e) = c.kernel.rt.space.protect(addr, size, prot & PROTECT_MASK)
    {
        log::warn!("MmSetAddressProtect({addr:#010X}, {size:#X}): {e}");
    }
    0
}

fn MmQueryAllocationSize(c: &mut Call) -> u64 {
    c.kernel.rt.space.allocation_size(c.arg(0)).unwrap_or(0) as u64
}

const STATS_SIZE: u32 = 104;

fn MmQueryStatistics(c: &mut Call) -> u64 {
    let p = c.arg(0);
    if p == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let mem = c.mem();
    if mem.read_u32(p) != STATS_SIZE {
        return status::BUFFER_TOO_SMALL as u64;
    }
    mem.fill(p, STATS_SIZE, 0);
    const TOTAL_PAGES: u32 = 0x2_0000; // 512 MiB / 4 KiB
    const KERNEL_PAGES: u32 = 0x100;
    let used = c.kernel.rt.space.physical_pages_used();
    mem.write_u32(p, STATS_SIZE);
    mem.write_u32(p + 4, TOTAL_PAGES);
    mem.write_u32(p + 8, KERNEL_PAGES);
    let title = p + 12;
    mem.write_u32(title, TOTAL_PAGES.saturating_sub(KERNEL_PAGES + used));
    mem.write_u32(title + 4, 0x2FFE_0000);
    mem.write_u32(title + 8, c.kernel.rt.space.virtual_bytes_reserved());
    mem.write_u32(title + 12, used);
    mem.write_u32(p + 100, TOTAL_PAGES - 1);
    status::SUCCESS as u64
}

/// Pool header for small allocations (Xenia X_POOL_ALLOC_HEADER).
const POOL_HEADER: u32 = 8;
const POOL_SMALL_MAX: u32 = 0xFD8;

fn pool_alloc(c: &Call, size: u32, tag: u32) -> u32 {
    let space = &c.kernel.rt.space;
    if size <= POOL_SMALL_MAX {
        match space.system_alloc(size + POOL_HEADER, 64) {
            Ok(a) => {
                c.mem().write_u8(a + 2, 170);
                c.mem().write_u32(a + 4, tag);
                a + POOL_HEADER
            }
            Err(_) => 0,
        }
    } else {
        // Large pool blocks are page aligned; ExFreePool tells them apart by that.
        space.alloc_virtual(size, PAGE_4K, true, protect::READWRITE, false).unwrap_or(0)
    }
}

fn ExAllocatePoolTypeWithTag(c: &mut Call) -> u64 {
    pool_alloc(c, c.arg(0), c.arg(1)) as u64
}

fn ExAllocatePoolWithTag(c: &mut Call) -> u64 {
    pool_alloc(c, c.arg(0), c.arg(1)) as u64
}

fn ExAllocatePool(c: &mut Call) -> u64 {
    pool_alloc(c, c.arg(0), u32::from_be_bytes(*b"None")) as u64
}

fn ExFreePool(c: &mut Call) -> u64 {
    let addr = c.arg(0);
    let space = &c.kernel.rt.space;
    let r = if addr.is_multiple_of(PAGE_4K) && region_of(addr) == Region::Virtual {
        space.release_virtual(addr).map(|_| 0)
    } else {
        space.system_free(addr - POOL_HEADER)
    };
    if let Err(e) = r {
        log::warn!("ExFreePool({addr:#010X}): {e}");
    }
    0
}

/// (u32 Unknown, u32 PhysicalAddress, u32 Size, u32 Protect). Titles map XMA
/// contexts by physical address. Xenia hands the address back unchanged; here
/// it comes back as the 4 KiB-view address of that physical memory (the same
/// pointer XMACreateContext returned), which recompiled code can dereference.
fn MmMapIoSpace(c: &mut Call) -> u64 {
    let phys = c.arg(1) & (map::PHYSICAL_SIZE - 1);
    if phys >= PAGE_4K { (map::PHYS_VIEW_4K + phys - PAGE_4K) as u64 } else { (map::PHYS_VIEW_64K + phys) as u64 }
}
