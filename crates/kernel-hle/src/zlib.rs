//! Host implementation of the title's zlib deflate (`deflateInit2_`, `deflate`, `deflateEnd`).
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
//! gets a non-null marker so the title's own checks pass.

use std::collections::HashMap;
use std::sync::Mutex;

use miniz_oxide::deflate::core::{CompressorOxide, create_comp_flags_from_zip_params};
use miniz_oxide::deflate::stream::deflate as mz_deflate;
use miniz_oxide::{MZError, MZFlush, MZStatus};

use crate::Kernel;

const Z_OK: i32 = 0;
const Z_STREAM_END: i32 = 1;
const Z_STREAM_ERROR: i32 = -2;
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
}
