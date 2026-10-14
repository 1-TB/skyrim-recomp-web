use std::sync::Arc;

use runtime_core::address_space::{AddressSpace, PAGE_4K, PAGE_16M, PAGE_64K};
use runtime_core::heap::{PageState, protect};
use runtime_core::memory::{MemoryError, map};

fn space() -> AddressSpace {
    AddressSpace::new().expect("reserve guest address space")
}

#[test]
fn scalars_are_stored_big_endian() {
    let s = space();
    let a = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    let m = &s.mem;

    m.write_u32(a, 0x1122_3344);
    assert_eq!(m.read_vec(a, 4), [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(m.read_u32(a), 0x1122_3344);
    assert_eq!(m.read_u16(a), 0x1122);
    assert_eq!(m.read_u16(a + 2), 0x3344);
    assert_eq!(m.read_u8(a + 3), 0x44);

    m.write_u64(a + 8, 0x0102_0304_0506_0708);
    assert_eq!(m.read_vec(a + 8, 8), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(m.read_u32(a + 8), 0x0102_0304);

    m.write_u16(a + 0x11, 0xBEEF); // unaligned
    assert_eq!(m.read_vec(a + 0x11, 2), [0xBE, 0xEF]);

    m.write_f32(a + 0x20, 1.0);
    assert_eq!(m.read_vec(a + 0x20, 4), [0x3F, 0x80, 0, 0]);
    m.write_f64(a + 0x28, -2.5);
    assert_eq!(m.read_vec(a + 0x28, 8), [0xC0, 0x04, 0, 0, 0, 0, 0, 0]);
    assert_eq!(m.read_f64(a + 0x28), -2.5);

    m.write_i32(a + 0x30, -2);
    assert_eq!(m.read_vec(a + 0x30, 4), [0xFF, 0xFF, 0xFF, 0xFE]);
    assert_eq!(m.read_i32(a + 0x30), -2);
}

#[test]
fn raw_pointer_view_matches_recompiled_code() {
    // Recompiled code does `__builtin_bswap32(*(uint32_t*)(base + skyrt_ea(ea)))`.
    let s = space();
    let a = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    s.mem.write_u32(a, 0xCAFE_F00D);
    let raw = unsafe { std::ptr::read_unaligned(s.mem.base_ptr().add(runtime_core::memory::linear(a) as usize) as *const u32) };
    assert_eq!(raw.swap_bytes(), 0xCAFE_F00D);
}

#[test]
fn atomics_operate_on_big_endian_words() {
    let s = space();
    let a = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    let m = &s.mem;
    m.write_i32(a, -1);
    assert!(m.compare_exchange_u32(a, 0xFFFF_FFFF, 0));
    assert_eq!(m.read_u32(a), 0);
    assert!(!m.compare_exchange_u32(a, 5, 6));
    assert_eq!(m.atomic_add_i32(a, 0x101), 0x101);
    assert_eq!(m.read_vec(a, 4), [0, 0, 1, 1]);
    assert_eq!(m.atomic_add_i32(a, -0x102), -1);
    assert_eq!(m.atomic_exchange_u32(a, 0x0A0B_0C0D), 0xFFFF_FFFF);
    assert_eq!(m.read_vec(a, 4), [0x0A, 0x0B, 0x0C, 0x0D]);
}

#[test]
fn concurrent_atomic_adds_are_not_lost() {
    let s = Arc::new(space());
    let a = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let s = s.clone();
            std::thread::spawn(move || {
                for _ in 0..10_000 {
                    s.mem.atomic_add_i32(a, 1);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(s.mem.read_u32(a), 80_000);
}

#[test]
fn virtual_allocations_respect_alignment_and_reuse_space() {
    let s = space();
    let a = s.alloc_virtual(0x1800, PAGE_4K, true, protect::READWRITE, false).unwrap();
    // 4 KiB-page requests have their own range; 64 KiB ones use the main heap.
    assert!((map::VIRTUAL_4K_BASE..map::VIRTUAL_4K_END).contains(&a));
    assert_eq!(a % PAGE_4K, 0);
    let b = s.alloc_virtual(0x10, PAGE_64K, false, protect::READWRITE, false).unwrap();
    assert!((map::VIRTUAL_BASE..map::VIRTUAL_END).contains(&b));
    assert_eq!(b % PAGE_64K, 0);
    let top = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, true).unwrap();
    assert!(top > a, "top-down allocation should come from the end of the heap");

    let info = s.query(a).unwrap();
    assert_eq!(info.state, PageState::Committed);
    assert_eq!(info.allocation_base, a);
    assert_eq!(info.region_size, 0x2000);
    assert_eq!(s.query(b).unwrap().state, PageState::Reserved);

    // Commit inside an existing reservation.
    let (base, was_committed) = s.alloc_virtual_fixed(b + 0x3000, 0x1000, true, protect::READWRITE).unwrap();
    assert_eq!(base, b + 0x3000);
    assert!(!was_committed);
    s.mem.write_u32(b + 0x3000, 7);

    assert_eq!(s.release_virtual(a).unwrap(), 0x2000);
    let again = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    assert_eq!(again, a, "released space is reused");
    assert_eq!(s.mem.read_u32(again), 0, "recommitted memory is zero");
}

#[test]
fn released_memory_stays_accessible_and_recommits_zeroed() {
    let s = space();
    let v = s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    let p = s.alloc_physical(0x1000, PAGE_4K, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    for a in [v, p] {
        s.mem.write_u32(a, 0x1234_5678);
    }
    s.release_virtual(v).unwrap();
    s.free_physical(p).unwrap();
    for a in [v, p] {
        // Use after free reads zero instead of faulting, and a stray write
        // does not leak into the next allocation of the page.
        assert_eq!(s.mem.read_u32(a), 0);
        s.mem.write_u32(a, 0xDEAD_BEEF);
    }
    assert_eq!(s.alloc_virtual(0x1000, PAGE_4K, true, protect::READWRITE, false).unwrap(), v);
    assert_eq!(s.alloc_physical(0x1000, PAGE_4K, 0, u32::MAX, 0, protect::READWRITE).unwrap(), p);
    for a in [v, p] {
        assert_eq!(s.mem.read_u32(a), 0);
    }
}

#[test]
fn decommit_zeroes_contents() {
    let s = space();
    let a = s.alloc_virtual(0x2000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    s.mem.write_u32(a + 0x1000, 0xDEAD_BEEF);
    s.decommit_virtual(a + 0x1000, 0x1000).unwrap();
    assert_eq!(s.query(a + 0x1000).unwrap().state, PageState::Reserved);
    s.alloc_virtual_fixed(a + 0x1000, 0x1000, true, protect::READWRITE).unwrap();
    assert_eq!(s.mem.read_u32(a + 0x1000), 0);
}

#[test]
fn fixed_allocation_rejects_overlap_with_other_allocation() {
    let s = space();
    let a = s.alloc_virtual(0x2000, PAGE_4K, true, protect::READWRITE, false).unwrap();
    let b = s.alloc_virtual_fixed(a + 0x2000, 0x1000, true, protect::READWRITE).unwrap().0;
    assert_eq!(b, a + 0x2000);
    assert!(s.alloc_virtual_fixed(a + 0x1000, 0x2000, true, protect::READWRITE).is_err());
}

#[test]
fn physical_views_show_one_memory() {
    let s = space();
    let small = s.alloc_physical(0x3000, PAGE_4K, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    let mid = s.alloc_physical(0x10000, PAGE_64K, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    let big = s.alloc_physical(0x10, PAGE_16M, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    assert!(small >= map::PHYS_VIEW_4K);
    assert!(small + 0x3000 <= map::PHYS_VIEW_4K + map::PHYS_VIEW_4K_SIZE, "4 KiB view ends at 0xFFD00000");
    assert!((map::PHYS_VIEW_64K..map::PHYS_VIEW_16M).contains(&mid));
    assert!((map::PHYS_VIEW_16M..map::PHYS_VIEW_4K).contains(&big));
    assert_eq!(mid % PAGE_64K, 0);
    assert_eq!(big % PAGE_16M, 0);

    let ps = s.physical_address(small).unwrap();
    let pm = s.physical_address(mid).unwrap();
    let pb = s.physical_address(big).unwrap();
    // The 4 KiB view is one page ahead of its physical addresses, as on hardware.
    assert_eq!(ps, small - map::PHYS_VIEW_4K + PAGE_4K);
    assert_eq!(pm, mid - map::PHYS_VIEW_64K);
    let ranges = [(ps, 0x3000), (pm, 0x10000), (pb, PAGE_16M)];
    for (i, &(a, al)) in ranges.iter().enumerate() {
        for &(b, bl) in &ranges[i + 1..] {
            assert!(a + al <= b || b + bl <= a, "physical ranges overlap");
        }
    }

    // Every view of a physical page reaches the same memory.
    // (The 4 KiB view starts one page into physical memory.)
    let views = |p: u32| {
        [Some(map::PHYS_VIEW_64K + p), Some(map::PHYS_VIEW_16M + p), p.checked_sub(PAGE_4K).map(|q| map::PHYS_VIEW_4K + q)]
            .into_iter()
            .flatten()
    };
    s.mem.write_u32(mid + 8, 0x1234_5678);
    for v in views(pm + 8) {
        assert_eq!(s.mem.read_u32(v), 0x1234_5678, "view address {v:#010X}");
        assert!(s.is_committed(v));
        assert_eq!(s.physical_address(v), Some(pm + 8));
    }
    s.mem.write_u32(map::PHYS_VIEW_16M + ps + 4, 0xABCD);
    assert_eq!(s.mem.read_u32(small + 4), 0xABCD);
    assert_eq!(s.allocation_size(mid), Some(0x10000));
    assert_eq!(s.allocation_size(map::PHYS_VIEW_16M + pm), Some(0x10000));
    assert_eq!(s.query(map::PHYS_VIEW_16M + pm).unwrap().allocation_base, map::PHYS_VIEW_16M + pm);

    let limited = s.alloc_physical(0x1000, PAGE_4K, 0x0200_0000, 0x02FF_FFFF, 0, protect::READWRITE).unwrap();
    let pl = s.physical_address(limited).unwrap();
    assert!((0x0200_0000..0x0300_0000).contains(&pl));
    // Alignment applies to the physical address.
    let aligned = s.alloc_physical(0x1000, PAGE_4K, 0, u32::MAX, 0x10000, protect::READWRITE).unwrap();
    assert_eq!(s.physical_address(aligned).unwrap() % 0x10000, 0);

    // GPU addresses resolve to the one copy of physical memory.
    assert_eq!(s.translate_physical(ps), Some(map::PHYSICAL_BASE + ps));
    assert_eq!(s.translate_physical(map::PHYS_VIEW_4K + pm + 8), Some(mid + 8));
    assert_eq!(s.translate_physical(pb), Some(big - map::PHYS_VIEW_16M + map::PHYSICAL_BASE));
    assert_eq!(s.translate_physical(ps - 4), None);

    // Only the canonical copy is ever committed.
    assert_eq!(s.mem.commit(map::PHYS_VIEW_16M, 0x1000), Err(MemoryError::FoldedView { addr: map::PHYS_VIEW_16M }));

    assert_eq!(s.free_physical(map::PHYS_VIEW_16M + pm).unwrap(), 0x10000, "freed through another view");
    assert!(s.free_physical(mid).is_err());
    assert_eq!(s.translate_physical(pm), None);
    assert!(!s.is_committed(mid));
}

#[test]
fn stacks_have_guard_pages() {
    let s = space();
    let st = s.alloc_stack(0x10000).unwrap();
    assert_eq!(st.base - st.limit, 0x10000);
    assert_eq!(st.limit - st.alloc_base, 0x1000);
    assert_eq!(s.query(st.alloc_base).unwrap().protect, protect::NOACCESS);
    assert_eq!(s.query(st.limit).unwrap().protect, protect::READWRITE);
    assert_eq!(s.query(st.base).unwrap().protect, protect::NOACCESS);
    s.mem.write_u32(st.base - 4, 1);
}

#[test]
fn system_heap_reuses_freed_blocks() {
    let s = space();
    let a = s.system_alloc(0x30, 16).unwrap();
    let b = s.system_alloc(0x30, 16).unwrap();
    assert!((map::SYSTEM_BASE..map::SYSTEM_END).contains(&a));
    assert_ne!(a, b);
    s.mem.write_u32(a, 5);
    s.system_free(a).unwrap();
    let c = s.system_alloc(0x20, 16).unwrap();
    assert_eq!(c, a);
    assert_eq!(s.mem.read_u32(c), 0, "system allocations are zeroed");
    let aligned = s.system_alloc(0x10, 0x1000).unwrap();
    assert_eq!(aligned % 0x1000, 0);
}

#[test]
fn host_reserved_range_is_never_committed() {
    let s = space();
    assert!(matches!(s.mem.commit(0x1000, 0x1000), Err(MemoryError::HostReserved { .. })));
    assert!(s.query(0x3000_0000).is_none());
}
