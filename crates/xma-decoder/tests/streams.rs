// SPDX-License-Identifier: LGPL-2.1-or-later

//! Checks against real XMA2 streams and FFmpeg reference output; these need
//! files that cannot be committed, so they are ignored by default.
//!
//! `XMA_TEST_DIR` must hold `NAME.xma` (an XMA2 RIFF file with one stream)
//! and `NAME.f32`, the output of FFmpeg's (Xenia fork) "xmaframes" decoder for
//! the same frames: per frame a little-endian i32 status (1: decoded) followed,
//! when 1, by 512 little-endian f32 samples per channel, planar.
//! Run with `cargo test -p xma-decoder --release -- --ignored --nocapture`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use xma_decoder::context::BYTES_PER_PACKET;
use xma_decoder::{Context, FrameDecoder, SAMPLES_PER_FRAME, XmaMemory, field, float_to_s16};

struct Stream {
    name: String,
    channels: usize,
    sample_rate: u32,
    packets: Vec<u8>,
}

fn le16(b: &[u8], at: usize) -> u32 {
    u16::from_le_bytes([b[at], b[at + 1]]) as u32
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn load_stream(path: &Path) -> Stream {
    let d = std::fs::read(path).unwrap();
    assert_eq!(&d[..4], b"RIFF");
    let (mut channels, mut sample_rate, mut packets) = (0, 0, Vec::new());
    let mut i = 12;
    while i + 8 <= d.len() {
        let len = le32(&d, i + 4) as usize;
        match &d[i..i + 4] {
            b"fmt " => {
                assert_eq!(le16(&d, i + 8), 0x166, "not XMA2");
                channels = le16(&d, i + 10) as usize;
                sample_rate = le32(&d, i + 12);
            }
            b"data" => packets = d[i + 8..(i + 8 + len).min(d.len())].to_vec(),
            _ => {}
        }
        i += 8 + len + (len & 1);
    }
    packets.truncate(packets.len() / 2048 * 2048);
    Stream { name: path.file_stem().unwrap().to_string_lossy().into(), channels, sample_rate, packets }
}

/// Splits a single-stream packet sequence into frames: (payload bit offset,
/// length) over the packets' concatenated payloads, using the XMA2 headers.
fn frames(packets: &[u8]) -> (Vec<u8>, Vec<usize>) {
    let mut payload = Vec::new();
    for p in packets.as_chunks::<2048>().0 {
        payload.extend_from_slice(&p[4..]);
    }
    let bit = |pos: usize| -> u32 { payload.get(pos >> 3).map_or(0, |b| (b >> (7 - (pos & 7)) & 1) as u32) };
    let mut starts = Vec::new();
    for (k, p) in packets.as_chunks::<2048>().0.iter().enumerate() {
        let count = p[0] >> 2;
        let offset = (((p[0] & 3) as usize) << 13) | ((p[1] as usize) << 5) | ((p[2] >> 3) as usize);
        assert!(p[3] == 0 || p[3] == 0xFF, "multi-stream file");
        let mut pos = k * 2044 * 8 + offset;
        for _ in 0..count {
            let len = (0..15).fold(0, |v, i| (v << 1) | bit(pos + i)) as usize;
            if len == 0 || len == 0x7FFF {
                break;
            }
            starts.push(pos);
            pos += len;
        }
    }
    (payload, starts)
}

/// Reference frames: None where FFmpeg produced no output.
fn load_reference(path: &Path, channels: usize) -> Vec<Option<Vec<Vec<f32>>>> {
    let d = std::fs::read(path).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    while i < d.len() {
        let status = le32(&d, i);
        i += 4;
        if status == 0 {
            out.push(None);
            continue;
        }
        let mut chans = Vec::new();
        for _ in 0..channels {
            chans.push(
                d[i..i + SAMPLES_PER_FRAME * 4].as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
            );
            i += SAMPLES_PER_FRAME * 4;
        }
        out.push(Some(chans));
    }
    out
}

fn test_files() -> Vec<PathBuf> {
    let dir = std::env::var("XMA_TEST_DIR").expect("XMA_TEST_DIR not set");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "xma") && p.with_extension("f32").exists())
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no NAME.xma + NAME.f32 pairs in XMA_TEST_DIR");
    files
}

#[test]
#[ignore]
fn frames_match_ffmpeg() {
    let mut overall = 0.0f32;
    for path in test_files() {
        let s = load_stream(&path);
        let reference = load_reference(&path.with_extension("f32"), s.channels);
        let (payload, starts) = frames(&s.packets);
        assert_eq!(starts.len(), reference.len(), "{}: frame count", s.name);
        let rate = xma_decoder::context::SAMPLE_RATES.into_iter().find(|&r| r >= s.sample_rate).unwrap_or(48000);
        let mut dec = FrameDecoder::new(rate, s.channels);
        let (mut max_err, mut peak) = (0.0f32, 0.0f32);
        let t = Instant::now();
        for (n, (&start, want)) in starts.iter().zip(&reference).enumerate() {
            let got = dec.decode_frame(&payload, start);
            let Some(want) = want else {
                assert!(got.is_err(), "{}: frame {n} decoded but FFmpeg failed", s.name);
                continue;
            };
            let info = got.unwrap_or_else(|e| panic!("{}: frame {n}: {e}", s.name));
            assert!(!info.length_mismatch, "{}: frame {n} length mismatch", s.name);
            for (c, want) in want.iter().enumerate() {
                for (a, b) in dec.pcm(c).iter().zip(want) {
                    max_err = max_err.max((a - b).abs());
                    peak = peak.max(b.abs());
                }
            }
        }
        let elapsed = t.elapsed().as_secs_f64();
        let samples = starts.len() * SAMPLES_PER_FRAME;
        println!(
            "{}: {} ch {} Hz, {} frames, max |err| {:e} (peak {:.3}), {:.1} Msamples/s per channel",
            s.name,
            s.channels,
            s.sample_rate,
            starts.len(),
            max_err,
            peak,
            samples as f64 / elapsed / 1e6
        );
        assert!(max_err < 1e-5, "{}: max error {max_err}", s.name);
        overall = overall.max(max_err);
    }
    println!("overall max |err| vs FFmpeg: {overall:e}");
}

/// Flat guest physical memory.
struct Memory(Vec<u8>);

impl XmaMemory for Memory {
    fn read(&self, addr: u32, buf: &mut [u8]) -> bool {
        let a = addr as usize;
        match self.0.get(a..a + buf.len()) {
            Some(src) => {
                buf.copy_from_slice(src);
                true
            }
            None => false,
        }
    }

    fn write(&mut self, addr: u32, data: &[u8]) -> bool {
        let a = addr as usize;
        match self.0.get_mut(a..a + data.len()) {
            Some(dst) => {
                dst.copy_from_slice(data);
                true
            }
            None => false,
        }
    }
}

const BUFFER_BASE: [u32; 2] = [0x1_0000, 0x40_0000];
const OUTPUT_BASE: u32 = 0x80_0000;

/// Plays a stream through a context the way a title streams: two input
/// buffers of `packets_per_buffer` packets refilled when the decoder hands
/// them back, and the output ring drained after every kick.
fn play(s: &Stream, packets_per_buffer: usize, ring_blocks: u32, decode_count: u32) -> Vec<u8> {
    let mut mem = Memory(vec![0; OUTPUT_BASE as usize + 32 * 256]);
    let mut ctx: Context = Context::new();
    let mut d = [0u32; 16];
    let rate_id = xma_decoder::context::SAMPLE_RATES.iter().position(|&r| r >= s.sample_rate).unwrap_or(3);
    field::set(&mut d, field::INPUT_BUFFER_0_PTR, BUFFER_BASE[0]);
    field::set(&mut d, field::INPUT_BUFFER_1_PTR, BUFFER_BASE[1]);
    field::set(&mut d, field::OUTPUT_BUFFER_PTR, OUTPUT_BASE);
    field::set(&mut d, field::OUTPUT_BUFFER_BLOCK_COUNT, ring_blocks);
    field::set(&mut d, field::SUBFRAME_DECODE_COUNT, decode_count);
    field::set(&mut d, field::IS_STEREO, (s.channels == 2) as u32);
    field::set(&mut d, field::SAMPLE_RATE, rate_id as u32);
    ctx.clear(&mut d);

    let chunks: Vec<&[u8]> = s.packets.chunks(packets_per_buffer * BYTES_PER_PACKET as usize).collect();
    let mut next_chunk = 0;
    let mut out = Vec::new();
    let mut idle_kicks = 0;
    loop {
        for b in 0..2 {
            let (valid, ptr, count) = if b == 0 {
                (field::INPUT_BUFFER_0_VALID, field::INPUT_BUFFER_0_PTR, field::INPUT_BUFFER_0_PACKET_COUNT)
            } else {
                (field::INPUT_BUFFER_1_VALID, field::INPUT_BUFFER_1_PTR, field::INPUT_BUFFER_1_PACKET_COUNT)
            };
            if field::get(&d, valid) == 0 && next_chunk < chunks.len() {
                let chunk = chunks[next_chunk];
                next_chunk += 1;
                assert!(mem.write(field::get(&d, ptr), chunk));
                field::set(&mut d, count, (chunk.len() / BYTES_PER_PACKET as usize) as u32);
                field::set(&mut d, valid, 1);
            }
        }
        field::set(&mut d, field::OUTPUT_BUFFER_VALID, 1);
        let input_valid_before =
            field::get(&d, field::INPUT_BUFFER_0_VALID) != 0 || field::get(&d, field::INPUT_BUFFER_1_VALID) != 0;
        ctx.work(&mut d, &mut mem);
        // Drain everything written.
        let mut read = field::get(&d, field::OUTPUT_BUFFER_READ_OFFSET);
        let write = field::get(&d, field::OUTPUT_BUFFER_WRITE_OFFSET);
        // A completely filled ring has write == read and is marked invalid.
        let full = input_valid_before && field::get(&d, field::OUTPUT_BUFFER_VALID) == 0 && read == write;
        let mut produced = false;
        while read != write || (full && !produced) {
            let mut block = [0u8; 256];
            assert!(mem.read(OUTPUT_BASE + read * 256, &mut block));
            out.extend_from_slice(&block);
            read = (read + 1) % ring_blocks;
            produced = true;
        }
        field::set(&mut d, field::OUTPUT_BUFFER_READ_OFFSET, read);
        let input_left = next_chunk < chunks.len()
            || field::get(&d, field::INPUT_BUFFER_0_VALID) != 0
            || field::get(&d, field::INPUT_BUFFER_1_VALID) != 0;
        if !produced {
            idle_kicks += 1;
            if !input_left || idle_kicks > 4 {
                break;
            }
        } else {
            idle_kicks = 0;
        }
    }
    assert_eq!(field::get(&d, field::ERROR_STATUS), 0, "{}: decoder error", s.name);
    out
}

#[test]
#[ignore]
fn context_streams_match_ffmpeg() {
    for path in test_files() {
        let s = load_stream(&path);
        let reference = load_reference(&path.with_extension("f32"), s.channels);
        let mut want = Vec::new();
        for frame in reference.iter().flatten() {
            for i in 0..SAMPLES_PER_FRAME {
                for ch in frame {
                    want.extend_from_slice(&float_to_s16(ch[i]).to_be_bytes());
                }
            }
        }
        let total_packets = s.packets.len() / 2048;
        for (per_buffer, ring, count) in [(1, 31, 1), (4, 16, 8), (7, 31, 4), (total_packets, 8, 2)] {
            let got = play(&s, per_buffer, ring, count);
            assert_eq!(got.len(), want.len(), "{}: output length ({per_buffer} packets/buffer)", s.name);
            let mut max_diff = 0;
            for (&a, &b) in got.as_chunks::<2>().0.iter().zip(want.as_chunks::<2>().0) {
                let (a, b) = (i16::from_be_bytes(a) as i32, i16::from_be_bytes(b) as i32);
                max_diff = max_diff.max((a - b).abs());
            }
            println!(
                "{}: {per_buffer} packets/buffer, {ring}-block ring, decode count {count}: {} samples, max diff {max_diff} LSB",
                s.name,
                got.len() / 2 / s.channels
            );
            assert!(max_diff <= 1, "{}: max diff {max_diff}", s.name);
        }
    }
}
