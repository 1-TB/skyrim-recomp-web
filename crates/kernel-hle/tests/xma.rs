mod common;

use common::{guest, kernel};
use kernel_hle::xma::{ContextData, field};
use runtime_core::address_space::PAGE_4K;
use runtime_core::heap::protect;

#[test]
fn contexts_are_set_up_kicked_and_released_through_the_exports() {
    let k = kernel();
    let mut ctx = guest(k);
    let mem = k.mem();
    let phys = |size| k.rt.space.alloc_physical(size, PAGE_4K, 0, u32::MAX, 0, protect::READWRITE).unwrap();
    // The output ring is followed by guard bytes the decoder must not touch.
    let (input0, input1, output, init, out_ptr) = (phys(2 * 2048), phys(2048), phys(9 * 256), phys(0x100), phys(0x10));

    assert_eq!(k.call_export("XMACreateContext", &mut ctx, &[out_ptr]), 0);
    let c = mem.read_u32(out_ptr);
    assert_ne!(c, 0);

    // XMA_CONTEXT_INIT: 2 packets in buffer 0, 1 in buffer 1, mono, an
    // 8-block output ring, 1 subframe per decode pass.
    mem.fill(init, 56, 0);
    mem.write_u32(init, input0);
    mem.write_u32(init + 4, 2);
    mem.write_u32(init + 8, input1);
    mem.write_u32(init + 0xC, 1);
    mem.write_u32(init + 0x14, output);
    mem.write_u32(init + 0x18, 8);
    mem.write_u32(init + 0x20, 1);
    assert_eq!(k.call_export("XMAInitializeContext", &mut ctx, &[c, init]), 0);
    let d = ContextData::load(k, c);
    assert_eq!(d.get(field::INPUT_BUFFER_0_PTR), k.rt.space.physical_address(input0).unwrap());
    assert_eq!(d.get(field::INPUT_BUFFER_1_PTR), k.rt.space.physical_address(input1).unwrap());
    assert_eq!(d.get(field::OUTPUT_BUFFER_PTR), k.rt.space.physical_address(output).unwrap());
    assert_eq!(d.get(field::OUTPUT_BUFFER_BLOCK_COUNT), 8);
    assert_eq!(d.get(field::INPUT_BUFFER_READ_OFFSET), 32, "cleared to the first packet");

    // Packets of garbage (not valid XMA): a kick must return, leave the
    // context consistent and never write past the ring.
    mem.fill(input0, 2 * 2048, 0x5A);
    mem.fill(input1, 2048, 0xA5);
    mem.fill(output, 9 * 256, 0xAA);
    k.call_export("XMASetInputBuffer0Valid", &mut ctx, &[c]);
    k.call_export("XMASetInputBuffer1Valid", &mut ctx, &[c]);
    k.call_export("XMASetOutputBufferValid", &mut ctx, &[c]);
    for _ in 0..4 {
        k.call_export("XMAEnableContext", &mut ctx, &[c]);
    }
    assert!(k.call_export("XMAGetOutputBufferWriteOffset", &mut ctx, &[c]) < 8);
    assert!(mem.read_vec(output + 8 * 256, 256).iter().all(|&b| b == 0xAA), "guard bytes untouched");

    // Titles map contexts by physical address and get the same pointer back.
    let phys = k.rt.space.physical_address(c).unwrap();
    assert_eq!(k.call_export("MmMapIoSpace", &mut ctx, &[2, phys, 0x40, 0x404]) as u32, c);

    assert_eq!(k.call_export("XMADisableContext", &mut ctx, &[c, 1]), 0);
    k.call_export("XMAReleaseContext", &mut ctx, &[c]);
    assert_eq!(ContextData::load(k, c), ContextData::default());
}
