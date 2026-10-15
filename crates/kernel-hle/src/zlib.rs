//! Host implementation of the title's zlib: deflate (`deflateInit2_`, `deflate`, `deflateEnd`)
//! and inflate (`inflateInit_`, `inflate`, `inflateReset`, `inflateEnd`, below).
//!
//! Skyrim compresses buffers it keeps in memory with zlib level 6 on a job thread while the
//! frame waits for it: recompiled, one call took ~330 ms natively and ~560 ms in the browser
//! (a visible hitch). guest-sys routes those three functions here (generated overrides for the
//! disc XEX, see its build script); miniz_oxide produces a different but equally valid stream,
//! which the title's own inflate reads.
//!
//! The guest `z_stream` (zlib 1.2, 32-bit, big-endian) stays the interface: next_in @0,
//! avail_in @4, total_in @8, next_out @12, avail_out @16, total_out @20, msg @24, state @28,
//! data_type @44, adler @48. The compressor lives here, keyed by the stream's address; `state`
//! gets a non-null value so the title's own checks pass.

use std::collections::HashMap;
use std::sync::Mutex;

use miniz_oxide::deflate::core::{CompressorOxide, create_comp_flags_from_zip_params};
use miniz_oxide::deflate::stream::deflate as mz_deflate;
use miniz_oxide::{MZError, MZFlush, MZStatus};

use crate::Kernel;

const Z_OK: i32 = 0;
const Z_STREAM_END: i32 = 1;
const Z_NEED_DICT: i32 = 2;
const Z_STREAM_ERROR: i32 = -2;
const Z_DATA_ERROR: i32 = -3;
const Z_MEM_ERROR: i32 = -4;
const Z_BUF_ERROR: i32 = -5;
/// Returned to the generated override when the call should go to the recompiled zlib instead.
const FALLBACK: i32 = i32::MIN;

const NEXT_IN: u32 = 0;
const AVAIL_IN: u32 = 4;
const TOTAL_IN: u32 = 8;
const NEXT_OUT: u32 = 12;
const AVAIL_OUT: u32 = 16;
const TOTAL_OUT: u32 = 20;
const MSG: u32 = 24;
const STATE: u32 = 28;
const DATA_TYPE: u32 = 44;
const ADLER: u32 = 48;
/// Z_UNKNOWN
const DATA_TYPE_UNKNOWN: u32 = 2;

static STREAMS: Mutex<Option<HashMap<u32, Box<CompressorOxide>>>> = Mutex::new(None);

/// deflateInit2_(strm, level, method, windowBits, memLevel, strategy, version, stream_size).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_deflate_init(strm: u32, level: i32, method: i32, window_bits: i32, strategy: i32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    // Only zlib and raw streams of Z_DEFLATED; anything else (gzip) stays recompiled.
    if strm == 0 || method != 8 || !(-15..=15).contains(&window_bits) || (0..8).contains(&window_bits.abs()) {
        return FALLBACK;
    }
    let level = if level == -1 { 6 } else { level };
    if !(0..=9).contains(&level) {
        return Z_STREAM_ERROR;
    }
    let compressor = Box::new(CompressorOxide::new(create_comp_flags_from_zip_params(level, window_bits, strategy)));
    let mem = k.mem();
    for (field, v) in [(TOTAL_IN, 0), (TOTAL_OUT, 0), (MSG, 0), (DATA_TYPE, DATA_TYPE_UNKNOWN), (ADLER, 1)] {
        mem.write_u32(strm + field, v);
    }
    // A marker the title can test for non-null; it never reads the internal state itself.
    mem.write_u32(strm + STATE, strm);
    STREAMS.lock().unwrap().get_or_insert_with(HashMap::new).insert(strm, compressor);
    Z_OK
}

/// deflate(strm, flush).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_deflate(strm: u32, flush: i32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    let mut streams = STREAMS.lock().unwrap();
    let Some(compressor) = streams.as_mut().and_then(|s| s.get_mut(&strm)) else { return FALLBACK };
    let flush = match flush {
        0 => MZFlush::None,
        1 => MZFlush::Partial,
        2 | 5 => MZFlush::Sync,
        3 => MZFlush::Full,
        4 => MZFlush::Finish,
        _ => return Z_STREAM_ERROR,
    };
    let mem = k.mem();
    let (next_in, avail_in) = (mem.read_u32(strm + NEXT_IN), mem.read_u32(strm + AVAIL_IN));
    let (next_out, avail_out) = (mem.read_u32(strm + NEXT_OUT), mem.read_u32(strm + AVAIL_OUT));
    if next_out == 0 || (next_in == 0 && avail_in != 0) {
        return Z_STREAM_ERROR;
    }
    let input = mem.read_vec(next_in, avail_in);
    let mut output = vec![0u8; avail_out as usize];
    let r = mz_deflate(compressor, &input, &mut output, flush);
    let (consumed, written) = (r.bytes_consumed as u32, r.bytes_written as u32);
    mem.write_bytes(next_out, &output[..written as usize]);
    mem.write_u32(strm + NEXT_IN, next_in.wrapping_add(consumed));
    mem.write_u32(strm + AVAIL_IN, avail_in - consumed);
    mem.write_u32(strm + TOTAL_IN, mem.read_u32(strm + TOTAL_IN).wrapping_add(consumed));
    mem.write_u32(strm + NEXT_OUT, next_out.wrapping_add(written));
    mem.write_u32(strm + AVAIL_OUT, avail_out - written);
    mem.write_u32(strm + TOTAL_OUT, mem.read_u32(strm + TOTAL_OUT).wrapping_add(written));
    mem.write_u32(strm + ADLER, compressor.adler32());
    match r.status {
        Ok(MZStatus::StreamEnd) => Z_STREAM_END,
        Ok(_) => Z_OK,
        Err(MZError::Buf) => Z_BUF_ERROR,
        Err(_) => Z_STREAM_ERROR,
    }
}

/// deflateEnd(strm).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_deflate_end(strm: u32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    if STREAMS.lock().unwrap().as_mut().and_then(|s| s.remove(&strm)).is_none() {
        return FALLBACK;
    }
    k.mem().write_u32(strm + STATE, 0);
    Z_OK
}

// --- Inflate ------------------------------------------------------------------------------------
//
// The loader inflates zlib streams out of the BSAs on the loading screen's critical path, with
// the title's zlib (inflateInit_, inflate, inflateReset, inflateEnd); recompiled, inflate_fast was
// the loader thread's hottest function. zlib-rs is a port of zlib's inflate: it consumes input
// and produces output call by call exactly as the title's zlib does (the loader depends on that:
// miniz_oxide reads ahead and broke it), so the guest's stream bookkeeping stays the same.
// `SKYRT_ZLIB_VERIFY` set: the title's inflate runs and a host shadow of every stream checks each
// call against it (bytes produced, bytes consumed, return code).

struct HostInflate {
    stream: zlib_rs::Inflate,
    /// zlib's strm->adler for a zlib stream: adler-32 of the output so far.
    adler: u32,
}

impl HostInflate {
    fn new() -> Self {
        // inflateInit_ is inflateInit2_ with windowBits 15: a zlib header, 32 KiB window.
        HostInflate { stream: zlib_rs::Inflate::new(true, 15), adler: 1 }
    }

    /// One inflate call: (bytes consumed, bytes written, zlib return code).
    fn step(&mut self, input: &[u8], output: &mut [u8], flush: zlib_rs::InflateFlush) -> (u32, u32, i32) {
        let (in0, out0) = (self.stream.total_in(), self.stream.total_out());
        let r = self.stream.decompress(input, output, flush);
        let consumed = (self.stream.total_in() - in0) as u32;
        let written = (self.stream.total_out() - out0) as u32;
        self.adler = zlib_rs::adler32::adler32(self.adler, &output[..written as usize]);
        let ret = match r {
            Ok(zlib_rs::Status::Ok) => Z_OK,
            Ok(zlib_rs::Status::StreamEnd) => Z_STREAM_END,
            Ok(zlib_rs::Status::BufError) => Z_BUF_ERROR,
            Err(zlib_rs::InflateError::NeedDict { .. }) => Z_NEED_DICT,
            Err(zlib_rs::InflateError::StreamError) => Z_STREAM_ERROR,
            Err(zlib_rs::InflateError::DataError) => Z_DATA_ERROR,
            Err(zlib_rs::InflateError::MemError) => Z_MEM_ERROR,
        };
        (consumed, written, ret)
    }
}

type SharedInflate = std::sync::Arc<Mutex<HostInflate>>;

/// Host inflate streams by guest z_stream address.
static INFLATES: Mutex<Option<HashMap<u32, SharedInflate>>> = Mutex::new(None);

fn verify_inflate() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("SKYRT_ZLIB_VERIFY").is_some();
        if on {
            log::info!("zlib: verifying the host inflate against the title's");
        }
        on
    })
}

fn inflate_stream(strm: u32) -> Option<SharedInflate> {
    INFLATES.lock().unwrap().as_ref()?.get(&strm).cloned()
}

fn inflate_flush(flush: i32) -> Option<zlib_rs::InflateFlush> {
    use zlib_rs::InflateFlush::*;
    // Z_PARTIAL_FLUSH and Z_FULL_FLUSH mean nothing special to inflate.
    Some(match flush {
        0 | 1 | 3 => NoFlush,
        2 => SyncFlush,
        4 => Finish,
        5 => Block,
        6 => Trees,
        _ => return None,
    })
}

/// The host address of `[addr, addr + len)` if it is contiguous in host memory (it nearly always
/// is: only a range crossing into another 1 MiB frame of a different region isn't).
fn contiguous(mem: &runtime_core::memory::GuestMemory, addr: u32, len: u32) -> Option<*mut u8> {
    if len == 0 {
        return Some(std::ptr::NonNull::dangling().as_ptr());
    }
    let (mut first, mut pieces) = (std::ptr::null_mut(), 0);
    mem.runs(addr, len, |_, p, _| {
        if pieces == 0 {
            first = p;
        }
        pieces += 1;
    });
    (pieces == 1).then_some(first)
}

const INFLATE_STATE_SIZE: u32 = 64;

/// inflateInit_(strm, version, stream_size).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_inflate_init(strm: u32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    if strm == 0 {
        return FALLBACK;
    }
    let host = std::sync::Arc::new(Mutex::new(HostInflate::new()));
    if verify_inflate() {
        INFLATES.lock().unwrap().get_or_insert_with(HashMap::new).insert(strm, host);
        return FALLBACK;
    }
    // The title only tests `state` for null; a block of its own keeps it a valid allocation.
    let Ok(state) = k.rt.space.system_alloc(INFLATE_STATE_SIZE, 16) else { return Z_MEM_ERROR };
    INFLATES.lock().unwrap().get_or_insert_with(HashMap::new).insert(strm, host);
    let mem = k.mem();
    for (field, v) in [(TOTAL_IN, 0), (TOTAL_OUT, 0), (MSG, 0), (STATE, state), (ADLER, 1)] {
        mem.write_u32(strm + field, v);
    }
    Z_OK
}

/// inflate(strm, flush).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_inflate(strm: u32, flush: i32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    let Some(host) = inflate_stream(strm) else { return FALLBACK };
    let mem = k.mem();
    let (next_in, avail_in) = (mem.read_u32(strm + NEXT_IN), mem.read_u32(strm + AVAIL_IN));
    let (next_out, avail_out) = (mem.read_u32(strm + NEXT_OUT), mem.read_u32(strm + AVAIL_OUT));
    if verify_inflate() {
        let input = mem.read_vec(next_in, avail_in);
        VERIFY_CALL.with(|c| *c.borrow_mut() = Some(VerifyCall { strm, input, next_out, avail_in, avail_out, flush }));
        return FALLBACK;
    }
    let Some(flush) = inflate_flush(flush) else { return Z_STREAM_ERROR };
    if next_out == 0 || (next_in == 0 && avail_in != 0) {
        return Z_STREAM_ERROR;
    }
    let mut host = host.lock().unwrap();
    let (consumed, written, ret) = match (contiguous(mem, next_in, avail_in), contiguous(mem, next_out, avail_out)) {
        (Some(src), Some(dst)) => {
            // SAFETY: committed guest ranges the title handed to inflate; input and output don't
            // overlap (zlib requires it), and the stream's owner waits for this call.
            let (input, output) = unsafe {
                (std::slice::from_raw_parts(src, avail_in as usize), std::slice::from_raw_parts_mut(dst, avail_out as usize))
            };
            host.step(input, output, flush)
        }
        _ => {
            let input = mem.read_vec(next_in, avail_in);
            let mut output = vec![0u8; avail_out as usize];
            let r = host.step(&input, &mut output, flush);
            mem.write_bytes(next_out, &output[..r.1 as usize]);
            r
        }
    };
    runtime_core::dirty::mark(next_out, written);
    mem.write_u32(strm + NEXT_IN, next_in.wrapping_add(consumed));
    mem.write_u32(strm + AVAIL_IN, avail_in - consumed);
    mem.write_u32(strm + TOTAL_IN, mem.read_u32(strm + TOTAL_IN).wrapping_add(consumed));
    mem.write_u32(strm + NEXT_OUT, next_out.wrapping_add(written));
    mem.write_u32(strm + AVAIL_OUT, avail_out - written);
    mem.write_u32(strm + TOTAL_OUT, mem.read_u32(strm + TOTAL_OUT).wrapping_add(written));
    mem.write_u32(strm + ADLER, host.adler);
    ret
}

/// An inflate call as the title made it, kept while its own inflate runs (verification).
struct VerifyCall {
    strm: u32,
    input: Vec<u8>,
    next_out: u32,
    avail_in: u32,
    avail_out: u32,
    flush: i32,
}

thread_local! {
    static VERIFY_CALL: std::cell::RefCell<Option<VerifyCall>> = const { std::cell::RefCell::new(None) };
}

/// After the title's inflate returned `title_ret` (verification): the same call on the host
/// shadow must consume and produce the same bytes and return the same code.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_inflate_verify(strm: u32, title_ret: i32) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);
    static DIFFERED: AtomicU64 = AtomicU64::new(0);
    let Some(k) = Kernel::try_global() else { return };
    let Some(call) = VERIFY_CALL.with(|c| c.borrow_mut().take()).filter(|c| c.strm == strm) else { return };
    let Some(host) = inflate_stream(strm) else { return };
    let Some(flush) = inflate_flush(call.flush) else { return };
    let mem = k.mem();
    let title_consumed = call.avail_in.wrapping_sub(mem.read_u32(strm + AVAIL_IN));
    let title_written = call.avail_out.wrapping_sub(mem.read_u32(strm + AVAIL_OUT));
    let mut output = vec![0u8; call.avail_out as usize];
    let (consumed, written, ret) = host.lock().unwrap().step(&call.input, &mut output, flush);
    let same_bytes = written == title_written && mem.read_vec(call.next_out, written) == output[..written as usize];
    let calls = CALLS.fetch_add(1, Relaxed) + 1;
    let bytes = BYTES.fetch_add(u64::from(written), Relaxed) + u64::from(written);
    if consumed != title_consumed || ret != title_ret || !same_bytes {
        let n = DIFFERED.fetch_add(1, Relaxed) + 1;
        log::warn!(
            "zlib verify: inflate call differs ({n} so far) on stream {strm:#010X}: title consumed {title_consumed}, \
             wrote {title_written}, returned {title_ret}; host {consumed}, {written}, {ret}; output equal: {same_bytes}"
        );
    } else if calls.is_power_of_two() || calls % 10_000 == 0 {
        log::info!("zlib verify: {calls} inflate calls ({bytes} bytes out), {} differed", DIFFERED.load(Relaxed));
    }
}

/// inflateReset(strm).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_inflate_reset(strm: u32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    let Some(host) = inflate_stream(strm) else { return FALLBACK };
    *host.lock().unwrap() = HostInflate::new();
    if verify_inflate() {
        return FALLBACK;
    }
    let mem = k.mem();
    for (field, v) in [(TOTAL_IN, 0), (TOTAL_OUT, 0), (MSG, 0), (ADLER, 1)] {
        mem.write_u32(strm + field, v);
    }
    Z_OK
}

/// inflateEnd(strm). The title shares this function between its two zlib copies; streams the
/// host doesn't own go to the recompiled one.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_inflate_end(strm: u32) -> i32 {
    let Some(k) = Kernel::try_global() else { return FALLBACK };
    if INFLATES.lock().unwrap().as_mut().and_then(|s| s.remove(&strm)).is_none() || verify_inflate() {
        return FALLBACK;
    }
    let mem = k.mem();
    let state = mem.read_u32(strm + STATE);
    if state != 0 {
        let _ = k.rt.space.system_free(state);
    }
    mem.write_u32(strm + STATE, 0);
    Z_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The configuration the title gets (level 6, zlib wrapper) yields a standard zlib stream,
    /// also when the output arrives in small pieces as a caller with a short buffer loops.
    #[test]
    fn zlib_stream_round_trips_in_pieces() {
        let input: Vec<u8> = (0..200_000u32).map(|i| (i * 7 % 251) as u8 ^ (i / 1000) as u8).collect();
        let mut c = CompressorOxide::new(create_comp_flags_from_zip_params(6, 15, 0));
        let mut out = Vec::new();
        let mut consumed = 0;
        loop {
            let mut piece = [0u8; 4096];
            let r = mz_deflate(&mut c, &input[consumed..], &mut piece, MZFlush::Finish);
            consumed += r.bytes_consumed;
            out.extend_from_slice(&piece[..r.bytes_written]);
            if r.status == Ok(MZStatus::StreamEnd) {
                break;
            }
            assert!(r.status.is_ok(), "{:?}", r.status);
        }
        assert_eq!(consumed, input.len());
        assert_eq!(out[0], 0x78, "zlib header");
        assert_eq!(miniz_oxide::inflate::decompress_to_vec_zlib(&out).unwrap(), input);
        let adler = u32::from_be_bytes(out[out.len() - 4..].try_into().unwrap());
        assert_eq!(c.adler32(), adler);
    }

    /// Fed in small pieces of input and output like the loader's loop, the host inflate
    /// reproduces the data, ends with Z_STREAM_END and keeps the zlib adler-32.
    #[test]
    fn inflate_in_pieces() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 13 % 241) as u8 ^ (i / 777) as u8).collect();
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&data, 6);
        let mut h = HostInflate::new();
        let (mut fed, mut out) = (0, Vec::new());
        loop {
            let input = &packed[fed..(fed + 1000).min(packed.len())];
            let mut piece = [0u8; 3000];
            let (consumed, written, ret) = h.step(input, &mut piece, zlib_rs::InflateFlush::NoFlush);
            fed += consumed as usize;
            out.extend_from_slice(&piece[..written as usize]);
            if ret == Z_STREAM_END {
                break;
            }
            assert_eq!(ret, Z_OK);
        }
        assert_eq!((fed, out == data), (packed.len(), true));
        assert_eq!(h.adler, zlib_rs::adler32::adler32(1, &data));
    }
}

