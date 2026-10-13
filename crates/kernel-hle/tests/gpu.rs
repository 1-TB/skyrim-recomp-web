mod common;

use std::time::{Duration, Instant};

use common::{RECORDER, guest, kernel, recorded_with_arg1};
use kernel_hle::Kernel;
use kernel_hle::gpu::{REGISTER_BASE, SWAP_SIGNATURE, packet_type0, packet_type2, packet_type3, pm4, reg};
use runtime_core::address_space::PAGE_4K;
use runtime_core::heap::protect;
use runtime_core::layout::kpcr;

fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A zeroed physical page: (CPU address, physical address).
fn physical_page(k: &Kernel) -> (u32, u32) {
    let a = k.rt.space.alloc_physical(0x1000, PAGE_4K, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    (a, k.rt.space.physical_address(a).unwrap())
}

/// Writes `words` into the ring at dword index `at` (wrapping) and returns the
/// new write index, as the D3D runtime would before storing CP_RB_WPTR.
fn ring_write(k: &Kernel, ring: u32, ring_dwords: u32, at: u32, words: &[u32]) -> u32 {
    let mut i = at;
    for &w in words {
        k.mem().write_u32(ring + i * 4, w);
        i = (i + 1) % ring_dwords;
    }
    i
}

fn kick(k: &Kernel, write_index: u32) {
    k.mem().write_u32(REGISTER_BASE + reg::CP_RB_WPTR * 4, write_index);
}

#[test]
fn video_variables_are_initialized() {
    let k = kernel();
    let mem = k.mem();
    let v = &k.vars;
    assert_eq!(mem.read_u32(v.vd_gpu_clock_in_mhz), 500);
    assert_eq!(mem.read_u32(v.vd_global_xam_device), 0);
    let device = mem.read_u32(v.vd_global_device);
    assert!(k.rt.space.physical_address(device).is_some(), "device block is physical memory");
    // An initialized, unowned critical section (lock count -1), not a zeroed one.
    let cs = v.vd_hsio_calibration_lock;
    assert_eq!(mem.read_u8(cs), 1, "synchronization event header");
    assert_eq!(mem.read_i32(cs + 0x10), -1);
    assert_eq!(mem.read_u32(cs + 0x18), 0);
    // Registers read back during bring-up.
    assert_eq!(mem.read_u32(REGISTER_BASE + 0x0F00 * 4), 0x0810_0748);
    assert_eq!(mem.read_u32(REGISTER_BASE + 0x1961 * 4), 0x0500_02D0);
}

#[test]
fn dpc_runs_inline_at_dispatch_level() {
    let k = kernel();
    let mut ctx = guest(k);
    let pcr = ctx.gpr32(13);
    let (dpc, _) = physical_page(k);
    const CONTEXT: u32 = 0xC0FF_EE01;
    k.call_export("KeInitializeDpc", &mut ctx, &[dpc, RECORDER, CONTEXT]);
    assert_eq!(k.mem().read_u16(dpc), 19);
    let sp = ctx.gpr(1);
    assert_eq!(k.call_export("KeInsertQueueDpc", &mut ctx, &[dpc, 0x11, 0x22]), 1);
    let calls = recorded_with_arg1(CONTEXT);
    assert_eq!(calls.len(), 1);
    let c = calls[0];
    assert_eq!(c.args, [dpc, CONTEXT, 0x11, 0x22]);
    assert_eq!((c.irql, c.dpc_active), (2, 1));
    // The caller's state is back as it was.
    assert_eq!(k.mem().read_u8(pcr + kpcr::CURRENT_IRQL), 0);
    assert_eq!(k.mem().read_u32(pcr + kpcr::PRCB_DPC_ACTIVE), 0);
    assert_eq!(ctx.gpr(1), sp);
    assert_eq!(k.call_export("KeRemoveQueueDpc", &mut ctx, &[dpc]), 0);
}

#[test]
fn vblank_interrupts_run_as_dpcs() {
    let k = kernel();
    let mut ctx = guest(k);
    const USER_DATA: u32 = 0x7AB1_0001;
    k.call_export("VdSetGraphicsInterruptCallback", &mut ctx, &[RECORDER, USER_DATA]);
    wait_until("three vblanks", || recorded_with_arg1(USER_DATA).len() >= 3);
    for c in recorded_with_arg1(USER_DATA) {
        assert_eq!(c.args[0], 0, "source: vblank");
        assert_eq!((c.irql, c.dpc_active), (2, 1));
        assert_eq!(c.process_type_in_dpc, 1, "title process while in the callback");
        assert_eq!(c.cpu, 2);
    }
    assert!(k.gpu.read_register(k.mem(), reg::D1MODE_V_COUNTER) >= 3);
    assert!(k.gpu.vblank_count() >= 3);
}

#[test]
fn command_processor_executes_ring_packets() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = k.mem();
    let (ring, ring_phys) = physical_page(k);
    let (writeback, writeback_phys) = physical_page(k);
    let (data, data_phys) = physical_page(k);
    let (ib, ib_phys) = physical_page(k);
    const USER_DATA: u32 = 0x7AB1_0002;

    // 2^(9+3) bytes = 1024 dwords.
    k.call_export("VdInitializeRingBuffer", &mut ctx, &[ring_phys, 9]);
    k.call_export("VdEnableRingBufferRPtrWriteBack", &mut ctx, &[writeback_phys, 6]);
    k.call_export("VdSetGraphicsInterruptCallback", &mut ctx, &[RECORDER, USER_DATA]);

    // Indirect buffer: one big-endian memory write.
    for (i, w) in [packet_type3(pm4::MEM_WRITE, 2), (data_phys + 0x10) | 2, 0xABCD].into_iter().enumerate() {
        mem.write_u32(ib + i as u32 * 4, w);
    }
    // Stored by MEM_WRITE below; COND_WRITE polls it.
    let first = 0x1122_3344;
    let words = [
        // Scratch register 0 written back to memory.
        packet_type0(reg::SCRATCH_UMSK, 1),
        1,
        packet_type0(reg::SCRATCH_ADDR, 1),
        data_phys + 0x100,
        packet_type0(reg::SCRATCH_REG0, 1),
        0xCAFE,
        // 8-in-32 swap: what the CPU sees as a big-endian word.
        packet_type3(pm4::MEM_WRITE, 3),
        data_phys | 2,
        first,
        0x5566_7788,
        packet_type3(pm4::INDIRECT_BUFFER, 2),
        ib_phys,
        3,
        // REG_RMW: (0xF0 & 0x30) | 0x1.
        packet_type0(reg::SCRATCH_REG0 + 1, 1),
        0xF0,
        packet_type3(pm4::REG_RMW, 3),
        reg::SCRATCH_REG0 + 1,
        0x30,
        0x1,
        // COND_WRITE: memory poll (0x10), ==, write to memory (0x100).
        packet_type3(pm4::COND_WRITE, 6),
        0x113,
        data_phys | 2,
        first,
        0xFFFF_FFFF,
        (data_phys + 0x20) | 2,
        0x77,
        // A fence the CPU waits on.
        packet_type3(pm4::EVENT_WRITE_SHD, 3),
        0,
        (data_phys + 0x30) | 2,
        0x1234,
        // WAIT_REG_MEM on a value already in memory.
        packet_type3(pm4::WAIT_REG_MEM, 5),
        0x13,
        (data_phys + 0x30) | 2,
        0x1234,
        0xFFFF_FFFF,
        0,
        // Interrupt to CPU 1.
        packet_type3(pm4::INTERRUPT, 1),
        0b10,
        packet_type3(pm4::XE_SWAP, 4),
        SWAP_SIGNATURE,
        0,
        1280,
        720,
        packet_type2(),
    ];
    let wptr = ring_write(k, ring, 1024, 0, &words);
    kick(k, wptr);
    wait_until("read pointer writeback", || mem.read_u32(writeback) == wptr);

    assert_eq!(mem.read_u32(data + 0x100), 0xCAFE);
    assert_eq!(mem.read_u32(data), first);
    assert_eq!(mem.read_u32(data + 4), 0x5566_7788);
    assert_eq!(mem.read_u32(data + 0x10), 0xABCD);
    assert_eq!(k.gpu.read_register(mem, reg::SCRATCH_REG0 + 1), 0x31);
    assert_eq!(mem.read_u32(data + 0x20), 0x77);
    assert_eq!(mem.read_u32(data + 0x30), 0x1234);
    assert_eq!(k.gpu.swap_count(), 1);
    let interrupts: Vec<_> = recorded_with_arg1(USER_DATA).into_iter().filter(|c| c.args[0] == 1).collect();
    assert_eq!(interrupts.len(), 1, "one command-stream interrupt");
    assert_eq!(interrupts[0].cpu, 1);
    assert_eq!((interrupts[0].irql, interrupts[0].dpc_active), (2, 1));
}

#[test]
fn ring_buffer_wraps_around() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = k.mem();
    let (ring, ring_phys) = physical_page(k);
    let (writeback, writeback_phys) = physical_page(k);
    let (data, data_phys) = physical_page(k);
    // 2^(3+3) bytes = 16 dwords.
    k.call_export("VdInitializeRingBuffer", &mut ctx, &[ring_phys, 3]);
    k.call_export("VdEnableRingBufferRPtrWriteBack", &mut ctx, &[writeback_phys, 2]);
    let write = |slot: u32, value: u32| [packet_type3(pm4::MEM_WRITE, 2), (data_phys + slot * 4) | 2, value];

    let mut wptr = 0;
    for slot in 0..4 {
        wptr = ring_write(k, ring, 16, wptr, &write(slot, 0x100 + slot));
    }
    kick(k, wptr);
    wait_until("first batch", || mem.read_u32(writeback) == 12);
    // The second packet straddles the end of the ring.
    for slot in 4..6 {
        wptr = ring_write(k, ring, 16, wptr, &write(slot, 0x100 + slot));
    }
    assert_eq!(wptr, 2);
    kick(k, wptr);
    wait_until("second batch", || mem.read_u32(writeback) == 2);
    for slot in 0..6 {
        assert_eq!(mem.read_u32(data + slot * 4), 0x100 + slot);
    }
}

#[test]
fn vd_swap_emits_a_swap_packet_the_command_processor_counts() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = k.mem();
    let (ring, ring_phys) = physical_page(k);
    let (writeback, writeback_phys) = physical_page(k);
    let (scratch, _) = physical_page(k);
    let frontbuffer = k.rt.space.alloc_physical(1280 * 720 * 4, PAGE_4K, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    let fb_phys = k.rt.space.physical_address(frontbuffer).unwrap();

    // Texture fetch constant: dword 1 holds the base address (bits 12+) and
    // format; dword 2 the size minus one.
    let fetch = scratch;
    let fetch_words = [0x0000_0002, frontbuffer | 0x86, (719 << 13) | 1279, 0, 0, 0];
    for (i, w) in fetch_words.into_iter().enumerate() {
        mem.write_u32(fetch + i as u32 * 4, w);
    }
    let (fb_ptr, format, color_space, width, height) =
        (scratch + 0x100, scratch + 0x104, scratch + 0x108, scratch + 0x10C, scratch + 0x110);
    mem.write_u32(fb_ptr, frontbuffer);
    mem.write_u32(format, 6);
    mem.write_u32(width, 1280);
    mem.write_u32(height, 720);

    k.call_export("VdInitializeRingBuffer", &mut ctx, &[ring_phys, 9]);
    k.call_export("VdEnableRingBufferRPtrWriteBack", &mut ctx, &[writeback_phys, 6]);
    let (sys_buf, sys_id) = (scratch + 0x200, scratch + 0x300);
    k.call_export("VdGetSystemCommandBuffer", &mut ctx, &[sys_buf, sys_id]);
    k.call_export("VdSwap", &mut ctx, &[ring, fetch, 0, sys_buf, sys_id, fb_ptr, format, color_space, width, height]);

    let word = |i: u32| mem.read_u32(ring + i * 4);
    assert_eq!(word(0), packet_type0(reg::SHADER_CONSTANT_FETCH_00_0, 6));
    assert_eq!(word(2), fb_phys | 0x86, "fetch constant rebased to the physical address");
    assert_eq!(word(7), packet_type3(pm4::XE_SWAP, 4));
    assert_eq!([word(8), word(9), word(10), word(11)], [SWAP_SIGNATURE, fb_phys, 1280, 720]);
    assert!((12..64).all(|i| word(i) == packet_type2()));

    kick(k, 64);
    wait_until("swap", || mem.read_u32(writeback) == 64);
    assert_eq!(k.gpu.swap_count(), 1);
    assert_eq!(k.gpu.read_register(mem, reg::SHADER_CONSTANT_FETCH_00_0 + 1), fb_phys | 0x86);
}

#[test]
fn video_queries_report_the_configured_mode() {
    let k = kernel();
    let mut ctx = guest(k);
    let (out, _) = physical_page(k);
    let mem = k.mem();
    k.call_export("VdQueryVideoMode", &mut ctx, &[out]);
    assert_eq!((mem.read_u32(out), mem.read_u32(out + 4)), (1280, 720));
    assert_eq!(k.call_export("VdQueryVideoFlags", &mut ctx, &[]), 0b011);
    k.call_export("VdGetCurrentDisplayGamma", &mut ctx, &[out + 0x100, out + 0x104]);
    assert_eq!(mem.read_u32(out + 0x100), 2);
    assert!((mem.read_f32(out + 0x104) - 2.2222).abs() < 1e-3);
    k.call_export("VdGetCurrentDisplayInformation", &mut ctx, &[out + 0x200]);
    assert_eq!((mem.read_u16(out + 0x200), mem.read_u16(out + 0x202)), (1280, 720));
    assert_eq!(mem.read_f32(out + 0x200 + 0x4C), 60.0);
    assert_eq!(k.call_export("VdInitializeEngines", &mut ctx, &[0, 0, 0, 0, 0]), 1);
}
