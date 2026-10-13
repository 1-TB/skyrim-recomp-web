//! XMA* exports, backed by the decoder stand-in in [`crate::xma`].
//! Behavior follows Xenia's xboxkrnl_audio_xma.cc.
#![allow(non_snake_case)]

use crate::dispatch::{Call, Export, ModuleId};
use crate::status;
use crate::xma::{ContextData, field};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x224, XMACreateContext),
    export!(K, 0x225, XMAInitializeContext),
    export!(K, 0x226, XMAReleaseContext),
    export!(K, 0x227, XMAEnableContext),
    export!(K, 0x228, XMADisableContext),
    export!(K, 0x229, XMAGetOutputBufferWriteOffset),
    export!(K, 0x22A, XMASetOutputBufferReadOffset),
    export!(K, 0x22B, XMAGetOutputBufferReadOffset),
    export!(K, 0x22C, XMASetOutputBufferValid),
    export!(K, 0x22D, XMAIsOutputBufferValid),
    export!(K, 0x22E, XMASetInputBuffer0Valid),
    export!(K, 0x22F, XMAIsInputBuffer0Valid),
    export!(K, 0x230, XMASetInputBuffer1Valid),
    export!(K, 0x231, XMAIsInputBuffer1Valid),
    export!(K, 0x232, XMASetInputBuffer0),
    export!(K, 0x233, XMASetInputBuffer1),
    export!(K, 0x234, XMAGetPacketMetadata),
    export!(K, 0x235, XMABlockWhileInUse),
    export!(K, 0x236, XMASetLoopData),
    export!(K, 0x237, XMASetInputBufferReadOffset),
    export!(K, 0x238, XMAGetInputBufferReadOffset),
];

/// X_E_FALSE: returned for buffers outside physical memory.
const E_FALSE: u32 = 1;

fn get(c: &Call, f: (usize, u32, u32)) -> u64 {
    ContextData::load(c.kernel, c.arg(0)).get(f) as u64
}

fn set(c: &Call, f: (usize, u32, u32), value: u32) -> u64 {
    let ptr = c.arg(0);
    let mut d = ContextData::load(c.kernel, ptr);
    d.set(f, value);
    d.store(c.kernel, ptr);
    0
}

/// (XMA_CONTEXT_DATA** Out)
fn XMACreateContext(c: &mut Call) -> u64 {
    let out = c.arg(0);
    let ptr = c.kernel.xma_allocate();
    if out != 0 {
        c.mem().write_u32(out, ptr);
    }
    if ptr == 0 { status::NO_MEMORY as u64 } else { status::SUCCESS as u64 }
}

fn XMAReleaseContext(c: &mut Call) -> u64 {
    c.kernel.xma_release(c.arg(0));
    0
}

/// (XMA_CONTEXT_DATA* Context, XMA_CONTEXT_INIT* Init)
fn XMAInitializeContext(c: &mut Call) -> u64 {
    let (ptr, init) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    let space = &c.kernel.rt.space;
    let phys = |guest: u32| if guest == 0 { Some(0) } else { space.physical_address(guest) };
    let (Some(in0), Some(in1), Some(out)) =
        (phys(mem.read_u32(init)), phys(mem.read_u32(init + 8)), phys(mem.read_u32(init + 0x14)))
    else {
        log::error!("XMAInitializeContext: a buffer is not in physical memory");
        return E_FALSE as u64;
    };
    let mut d = ContextData::default();
    d.set(field::INPUT_BUFFER_0_PTR, in0);
    d.set(field::INPUT_BUFFER_0_PACKET_COUNT, mem.read_u32(init + 4));
    d.set(field::INPUT_BUFFER_1_PTR, in1);
    d.set(field::INPUT_BUFFER_1_PACKET_COUNT, mem.read_u32(init + 0xC));
    d.set(field::INPUT_BUFFER_READ_OFFSET, mem.read_u32(init + 0x10));
    d.set(field::OUTPUT_BUFFER_PTR, out);
    d.set(field::OUTPUT_BUFFER_BLOCK_COUNT, mem.read_u32(init + 0x18));
    d.set(field::SUBFRAME_DECODE_COUNT, mem.read_u32(init + 0x20));
    // As in Xenia: a channel count of 1 or more means stereo.
    d.set(field::IS_STEREO, (mem.read_u32(init + 0x24) >= 1) as u32);
    d.set(field::SAMPLE_RATE, mem.read_u32(init + 0x28));
    // XMA_LOOP_DATA at 0x2C: start, end, count, subframe end, subframe skip.
    d.set(field::LOOP_START, mem.read_u32(init + 0x2C));
    d.set(field::LOOP_END, mem.read_u32(init + 0x30));
    d.set(field::LOOP_COUNT, mem.read_u8(init + 0x34) as u32);
    d.set(field::LOOP_SUBFRAME_END, mem.read_u8(init + 0x35) as u32);
    d.set(field::LOOP_SUBFRAME_SKIP, mem.read_u8(init + 0x36) as u32);
    d.store(c.kernel, ptr);
    c.kernel.xma_clear(ptr);
    0
}

/// (XMA_CONTEXT_DATA* Context, XMA_LOOP_DATA* Loop)
fn XMASetLoopData(c: &mut Call) -> u64 {
    let (ptr, l) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    let mut d = ContextData::load(c.kernel, ptr);
    d.set(field::LOOP_START, mem.read_u32(l));
    d.set(field::LOOP_END, mem.read_u32(l + 4));
    d.set(field::LOOP_COUNT, mem.read_u8(l + 8) as u32);
    d.set(field::LOOP_SUBFRAME_END, mem.read_u8(l + 9) as u32);
    d.set(field::LOOP_SUBFRAME_SKIP, mem.read_u8(l + 10) as u32);
    d.store(c.kernel, ptr);
    0
}

fn set_input_buffer(c: &Call, ptr_field: (usize, u32, u32), count_field: (usize, u32, u32)) -> u64 {
    let (buffer, packets) = (c.arg(1), c.arg(2));
    let Some(phys) = c.kernel.rt.space.physical_address(buffer) else {
        log::error!("XMASetInputBuffer: {buffer:#010X} is not in physical memory");
        return E_FALSE as u64;
    };
    let ptr = c.arg(0);
    let mut d = ContextData::load(c.kernel, ptr);
    d.set(ptr_field, phys);
    d.set(count_field, packets);
    d.store(c.kernel, ptr);
    0
}

/// (XMA_CONTEXT_DATA* Context, void* Buffer, u32 PacketCount)
fn XMASetInputBuffer0(c: &mut Call) -> u64 {
    set_input_buffer(c, field::INPUT_BUFFER_0_PTR, field::INPUT_BUFFER_0_PACKET_COUNT)
}

fn XMASetInputBuffer1(c: &mut Call) -> u64 {
    set_input_buffer(c, field::INPUT_BUFFER_1_PTR, field::INPUT_BUFFER_1_PACKET_COUNT)
}

fn XMAIsInputBuffer0Valid(c: &mut Call) -> u64 {
    get(c, field::INPUT_BUFFER_0_VALID)
}

fn XMASetInputBuffer0Valid(c: &mut Call) -> u64 {
    set(c, field::INPUT_BUFFER_0_VALID, 1)
}

fn XMAIsInputBuffer1Valid(c: &mut Call) -> u64 {
    get(c, field::INPUT_BUFFER_1_VALID)
}

fn XMASetInputBuffer1Valid(c: &mut Call) -> u64 {
    set(c, field::INPUT_BUFFER_1_VALID, 1)
}

fn XMAIsOutputBufferValid(c: &mut Call) -> u64 {
    get(c, field::OUTPUT_BUFFER_VALID)
}

fn XMASetOutputBufferValid(c: &mut Call) -> u64 {
    set(c, field::OUTPUT_BUFFER_VALID, 1)
}

fn XMAGetInputBufferReadOffset(c: &mut Call) -> u64 {
    get(c, field::INPUT_BUFFER_READ_OFFSET)
}

fn XMASetInputBufferReadOffset(c: &mut Call) -> u64 {
    set(c, field::INPUT_BUFFER_READ_OFFSET, c.arg(1))
}

fn XMAGetOutputBufferReadOffset(c: &mut Call) -> u64 {
    get(c, field::OUTPUT_BUFFER_READ_OFFSET)
}

fn XMASetOutputBufferReadOffset(c: &mut Call) -> u64 {
    set(c, field::OUTPUT_BUFFER_READ_OFFSET, c.arg(1))
}

fn XMAGetOutputBufferWriteOffset(c: &mut Call) -> u64 {
    get(c, field::OUTPUT_BUFFER_WRITE_OFFSET)
}

fn XMAGetPacketMetadata(c: &mut Call) -> u64 {
    get(c, field::PACKET_METADATA)
}

/// Kicks the context; the stand-in decodes before returning.
fn XMAEnableContext(c: &mut Call) -> u64 {
    c.kernel.xma_work(c.arg(0));
    0
}

/// (XMA_CONTEXT_DATA* Context, BOOL Wait): decoding never runs in the
/// background, so the context is already idle.
fn XMADisableContext(_c: &mut Call) -> u64 {
    0
}

/// Waits until the decoder no longer needs the context's buffers.
fn XMABlockWhileInUse(c: &mut Call) -> u64 {
    let ptr = c.arg(0);
    loop {
        let d = ContextData::load(c.kernel, ptr);
        let in_use = d.get(field::INPUT_BUFFER_0_VALID) != 0 || d.get(field::INPUT_BUFFER_1_VALID) != 0;
        if !in_use || d.get(field::WORK_BUFFER_PTR) == 0 {
            return 0;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
