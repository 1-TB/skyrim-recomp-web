//! LZX decompression (LDI*), Microsoft's LZX "decompression interface" as the
//! cabinet SDK defines it: a context is created for one window size, then
//! each LDIDecompress call expands one compressed block of at most 32 KB of
//! output, carrying the window and repeat offsets over to the next block.
//!
//! Xenia doesn't implement these exports. The decoder state lives on the host
//! (the `lzxd` crate) behind an opaque handle, so the title's allocator
//! callbacks are never needed; the file callbacks are unused by LDI itself.
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use lzxd::{Lzxd, WindowSize};

use crate::dispatch::{Call, Export, ModuleId};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x0B6, LDICreateDecompression),
    export!(K, 0x0B7, LDIDecompress),
    export!(K, 0x0B8, LDIDestroyDecompression),
    export!(K, 0x2E2, LDIResetDecompression),
];

/// MDI_ERROR_* results.
pub mod mdi {
    pub const NO_ERROR: u32 = 0;
    pub const BAD_PARAMETERS: u32 = 2;
    pub const FAILED: u32 = 4;
    pub const CONFIGURATION: u32 = 5;
}

/// Largest block one LDIDecompress call expands (an LZX frame).
pub const CHUNK_SIZE: u32 = 0x8000;
/// Worst-case growth of an LZX block over its data, which the source buffer
/// must allow for (the SDK's MAX_GROWTH).
const MAX_GROWTH: u32 = 6144;
/// Handles are tagged so a stray value is recognisably not one ("LD").
const HANDLE_TAG: u32 = 0x4C44_0000;

struct Context {
    decoder: Lzxd,
    block_max: u32,
}

#[derive(Default)]
pub struct LdiContexts {
    contexts: Mutex<HashMap<u32, Context>>,
    next: AtomicU32,
}

fn window_size(bytes: u32) -> Option<WindowSize> {
    Some(match bytes {
        0x0000_8000 => WindowSize::KB32,
        0x0001_0000 => WindowSize::KB64,
        0x0002_0000 => WindowSize::KB128,
        0x0004_0000 => WindowSize::KB256,
        0x0008_0000 => WindowSize::KB512,
        0x0010_0000 => WindowSize::MB1,
        0x0020_0000 => WindowSize::MB2,
        _ => return None,
    })
}

/// (u32* DataBlockMax, LZXDECOMPRESS* Configuration { s32 WindowSize; s32 CpuType; },
///  PFNALLOC, PFNFREE, u32* SrcBufferMin, LDI_CONTEXT_HANDLE* Handle,
///  PFNOPEN, PFNREAD, PFNWRITE, PFNCLOSE, PFNSEEK) -> MDI_ERROR.
/// A null Handle only reports the buffer sizes.
fn LDICreateDecompression(c: &mut Call) -> u64 {
    let (block_max_ptr, config, src_min_ptr, handle_ptr) = (c.arg(0), c.arg(1), c.arg(4), c.arg(5));
    let mem = c.mem();
    if block_max_ptr == 0 || config == 0 {
        return mdi::BAD_PARAMETERS as u64;
    }
    let block_max = match mem.read_u32(block_max_ptr) {
        0 => CHUNK_SIZE,
        n if n <= CHUNK_SIZE => n,
        n => {
            log::warn!("LDICreateDecompression: block size {n:#X} is larger than an LZX frame");
            return mdi::BAD_PARAMETERS as u64;
        }
    };
    let window = mem.read_u32(config);
    let Some(size) = window_size(window) else {
        log::warn!("LDICreateDecompression: unsupported window size {window:#X}");
        return mdi::CONFIGURATION as u64;
    };
    mem.write_u32(block_max_ptr, block_max);
    if src_min_ptr != 0 {
        mem.write_u32(src_min_ptr, block_max + MAX_GROWTH);
    }
    if handle_ptr == 0 {
        return mdi::NO_ERROR as u64;
    }
    let ldi = &c.kernel.ldi;
    let handle = HANDLE_TAG | (ldi.next.fetch_add(1, Ordering::Relaxed) & 0xFFFF);
    ldi.contexts.lock().unwrap().insert(handle, Context { decoder: Lzxd::new(size), block_max });
    mem.write_u32(handle_ptr, handle);
    log::debug!("LDICreateDecompression: window {window:#X}, blocks up to {block_max:#X} -> {handle:#010X}");
    mdi::NO_ERROR as u64
}

/// (Handle, void* Src, u32 SrcSize, void* Dst, u32* DstSize) -> MDI_ERROR.
/// *DstSize is the block's uncompressed size on entry and the bytes written on return.
fn LDIDecompress(c: &mut Call) -> u64 {
    let (handle, src, src_len, dst, dst_len_ptr) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let mem = c.mem();
    let mut contexts = c.kernel.ldi.contexts.lock().unwrap();
    let Some(ctx) = contexts.get_mut(&handle) else { return mdi::BAD_PARAMETERS as u64 };
    if src == 0 || dst == 0 || dst_len_ptr == 0 || src_len == 0 {
        return mdi::BAD_PARAMETERS as u64;
    }
    let want = mem.read_u32(dst_len_ptr);
    if want == 0 || want > ctx.block_max || src_len > ctx.block_max + MAX_GROWTH {
        return mdi::BAD_PARAMETERS as u64;
    }
    let input = mem.read_vec(src, src_len);
    // The decoder asserts on some malformed streams; a bad block must fail the call, not the
    // title's thread.
    let result =
        catch_unwind(AssertUnwindSafe(|| ctx.decoder.decompress_next(&input, want as usize).map(<[u8]>::to_vec)));
    match result {
        Ok(Ok(out)) => {
            mem.write_bytes(dst, &out);
            mem.write_u32(dst_len_ptr, out.len() as u32);
            mdi::NO_ERROR as u64
        }
        failure => {
            let why = match failure {
                Ok(Err(e)) => e.to_string(),
                _ => "decoder panicked".into(),
            };
            log::warn!("LDIDecompress({handle:#010X}, {src_len:#X} -> {want:#X} bytes): {why}");
            // The window no longer matches the stream; later blocks can't decode either.
            ctx.decoder.reset();
            mem.write_u32(dst_len_ptr, 0);
            mdi::FAILED as u64
        }
    }
}

/// (Handle) -> MDI_ERROR: starts a new stream (empty window, initial repeat offsets).
fn LDIResetDecompression(c: &mut Call) -> u64 {
    match c.kernel.ldi.contexts.lock().unwrap().get_mut(&c.arg(0)) {
        Some(ctx) => {
            ctx.decoder.reset();
            mdi::NO_ERROR as u64
        }
        None => mdi::BAD_PARAMETERS as u64,
    }
}

/// (Handle) -> MDI_ERROR.
fn LDIDestroyDecompression(c: &mut Call) -> u64 {
    match c.kernel.ldi.contexts.lock().unwrap().remove(&c.arg(0)) {
        Some(_) => mdi::NO_ERROR as u64,
        None => mdi::BAD_PARAMETERS as u64,
    }
}
