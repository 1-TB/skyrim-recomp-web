// SPDX-License-Identifier: LGPL-2.1-or-later AND BSD-3-Clause
//
// XMA hardware context processor: a port of Xenia's XmaContextNew
// (src/xenia/apu/xma_context_new.cc / .h), the XMA_CONTEXT_DATA layout and
// ConvertFrame from xma_context.h / .cc, and xma_helpers.h. The ported logic
// remains under Xenia's BSD license, reproduced below; this crate as a whole
// is distributed under LGPL-2.1-or-later.
//
// Copyright (c) 2015, Ben Vanik. Copyright 2021-2024 Ben Vanik, Xenia Canary.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the project nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL BEN VANIK BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! The XMA context processor.
//!
//! A title describes each decoder voice with a 64-byte XMA_CONTEXT_DATA in
//! physical memory: two input buffers of 2 KiB packets, a read position in
//! bits, loop points, and an output ring of 256-byte blocks. Each kick
//! ([`Context::work`]) decodes frames from the current input buffer into the
//! ring until the ring is full or input runs out, then updates the dwords.
//!
//! Packet layout (XMA2): byte 0 bits 7..2 = frames starting in the packet;
//! bits 1..0 + byte 1 + byte 2 bits 7..3 = bit offset of the first frame
//! after the 32-bit header (0x7FFF: none); byte 2 bits 2..0 = metadata
//! (1 for XMA2); byte 3 = packets to skip to the stream's next packet.
//! Frames are bit-packed, may cross packet (and buffer) boundaries, start
//! with a 15-bit length and end with a "more frames in this packet" bit.

use crate::wmapro::{DecodeError, FrameDecoder, FrameInfo, SAMPLES_PER_FRAME};

/// Bit fields of XMA_CONTEXT_DATA as `(dword, first bit, width)`, with bits
/// counted from the least significant bit of the host-order dword.
pub mod field {
    pub type Field = (usize, u32, u32);
    pub const INPUT_BUFFER_0_PACKET_COUNT: Field = (0, 0, 12);
    pub const LOOP_COUNT: Field = (0, 12, 8);
    pub const INPUT_BUFFER_0_VALID: Field = (0, 20, 1);
    pub const INPUT_BUFFER_1_VALID: Field = (0, 21, 1);
    pub const OUTPUT_BUFFER_BLOCK_COUNT: Field = (0, 22, 5);
    pub const OUTPUT_BUFFER_WRITE_OFFSET: Field = (0, 27, 5);
    pub const INPUT_BUFFER_1_PACKET_COUNT: Field = (1, 0, 12);
    pub const LOOP_SUBFRAME_START: Field = (1, 12, 2);
    pub const LOOP_SUBFRAME_END: Field = (1, 14, 3);
    pub const LOOP_SUBFRAME_SKIP: Field = (1, 17, 3);
    pub const SUBFRAME_DECODE_COUNT: Field = (1, 20, 4);
    /// Extra output blocks reserved per decoded frame (Xenia's guess).
    pub const OUTPUT_BUFFER_PADDING: Field = (1, 24, 3);
    pub const SAMPLE_RATE: Field = (1, 27, 2);
    pub const IS_STEREO: Field = (1, 29, 1);
    pub const OUTPUT_BUFFER_VALID: Field = (1, 31, 1);
    pub const INPUT_BUFFER_READ_OFFSET: Field = (2, 0, 26);
    pub const ERROR_STATUS: Field = (2, 26, 5);
    pub const ERROR_SET: Field = (2, 31, 1);
    pub const LOOP_START: Field = (3, 0, 26);
    pub const PARSER_ERROR_STATUS: Field = (3, 26, 5);
    pub const PARSER_ERROR_SET: Field = (3, 31, 1);
    pub const LOOP_END: Field = (4, 0, 26);
    pub const PACKET_METADATA: Field = (4, 26, 5);
    pub const CURRENT_BUFFER: Field = (4, 31, 1);
    pub const INPUT_BUFFER_0_PTR: Field = (5, 0, 32);
    pub const INPUT_BUFFER_1_PTR: Field = (6, 0, 32);
    pub const OUTPUT_BUFFER_PTR: Field = (7, 0, 32);
    pub const WORK_BUFFER_PTR: Field = (8, 0, 32);
    pub const OUTPUT_BUFFER_READ_OFFSET: Field = (9, 0, 5);
    pub const STOP_WHEN_DONE: Field = (9, 30, 1);
    pub const INTERRUPT_WHEN_DONE: Field = (9, 31, 1);

    pub fn get(d: &[u32; 16], (dword, shift, width): Field) -> u32 {
        let mask = if width == 32 { u32::MAX } else { (1 << width) - 1 };
        (d[dword] >> shift) & mask
    }

    pub fn set(d: &mut [u32; 16], (dword, shift, width): Field, value: u32) {
        let mask = if width == 32 { u32::MAX } else { (1 << width) - 1 };
        d[dword] = (d[dword] & !(mask << shift)) | ((value & mask) << shift);
    }
}

use field::{get, set};

pub const BYTES_PER_PACKET: u32 = 2048;
const BYTES_PER_PACKET_HEADER: usize = 4;
const BYTES_PER_PACKET_DATA: usize = BYTES_PER_PACKET as usize - BYTES_PER_PACKET_HEADER;
const BITS_PER_PACKET: u32 = BYTES_PER_PACKET * 8;
const BITS_PER_PACKET_HEADER: u32 = 32;
const BITS_PER_FRAME_HEADER: u32 = 15;
/// Largest first-frame offset that points inside a packet.
const MAX_FRAME_SIZE_IN_BITS: u32 = 0x4000 - BITS_PER_PACKET_HEADER;
/// Frame length value meaning "no frame".
const MAX_FRAME_LENGTH: u32 = 0x7FFF;
pub const OUTPUT_BYTES_PER_BLOCK: u32 = 256;
/// Output bytes per channel per frame (512 16-bit samples).
const BYTES_PER_FRAME_CHANNEL: u32 = SAMPLES_PER_FRAME as u32 * 2;
/// Output blocks per channel per frame.
const BLOCKS_PER_FRAME_CHANNEL: u32 = BYTES_PER_FRAME_CHANNEL / OUTPUT_BYTES_PER_BLOCK;
/// Decoder error status at which kicks stop (Xenia's guess for any error).
const ERROR_STATUS_STOP: u32 = 4;
/// The SAMPLE_RATE field's values.
pub const SAMPLE_RATES: [u32; 4] = [24000, 32000, 44100, 48000];
/// Bound on consecutive decode passes that neither produce nor drain a frame
/// within one kick (Xenia has none; a looping stream whose frames all fail to
/// decode would otherwise spin forever).
const MAX_IDLE_PASSES: u32 = 1 << 14;

/// Guest physical memory as the XMA unit sees it.
pub trait XmaMemory {
    /// Fills `buf` from physical address `addr`; false if unmapped.
    fn read(&self, addr: u32, buf: &mut [u8]) -> bool;
    /// Stores `data` at physical address `addr`; false if unmapped.
    fn write(&mut self, addr: u32, data: &[u8]) -> bool;
}

/// A frame decoder the context drives; [`FrameDecoder`] in production.
pub trait FrameDecoding {
    /// A decoder for a stream of `channels` (1 or 2) at `sample_rate` Hz.
    fn new(sample_rate: u32, channels: usize) -> Self;
    /// Decodes the frame at bit `bit_offset` of `data` (see
    /// [`FrameDecoder::decode_frame`]).
    fn decode_frame(&mut self, data: &[u8], bit_offset: usize) -> Result<FrameInfo, DecodeError>;
    /// The last frame's samples for `channel`.
    fn pcm(&self, channel: usize) -> &[f32; SAMPLES_PER_FRAME];
    /// Drops the overlap carried between frames.
    fn reset(&mut self);
}

impl FrameDecoding for FrameDecoder {
    fn new(sample_rate: u32, channels: usize) -> Self {
        FrameDecoder::new(sample_rate, channels)
    }

    fn decode_frame(&mut self, data: &[u8], bit_offset: usize) -> Result<FrameInfo, DecodeError> {
        FrameDecoder::decode_frame(self, data, bit_offset)
    }

    fn pcm(&self, channel: usize) -> &[f32; SAMPLES_PER_FRAME] {
        FrameDecoder::pcm(self, channel)
    }

    fn reset(&mut self) {
        FrameDecoder::reset(self)
    }
}

/// Xenia's float to output sample conversion: clamp to [-1, 1], scale by
/// 32767, round to nearest-even (lrintf).
#[inline]
pub fn float_to_s16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16
}

/// XmaContext::ConvertFrame: one frame as big-endian signed 16-bit samples,
/// channels interleaved, into `out` (1024 bytes per channel).
pub fn convert_to_be_s16(decoder: &impl FrameDecoding, channels: usize, out: &mut [u8]) {
    let out = &mut out[..SAMPLES_PER_FRAME * channels * 2];
    for (i, frame) in out.chunks_exact_mut(channels * 2).enumerate() {
        for (c, sample) in frame.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            *sample = float_to_s16(decoder.pcm(c)[i]).to_be_bytes();
        }
    }
}

/// XmaContextNew::StoreContextMerged, for callers that decode while the
/// title may modify the context: applies the decoder-owned fields of
/// `decoded` (started from `initial`) to `fresh`, a re-read of guest memory.
pub fn merge_decoder_fields(fresh: &mut [u32; 16], initial: &[u32; 16], decoded: &[u32; 16]) {
    for f in [field::LOOP_COUNT, field::OUTPUT_BUFFER_WRITE_OFFSET] {
        set(fresh, f, get(decoded, f));
    }
    // Valid flags are only ever cleared by the decoder, and only if it saw them set.
    for f in [field::INPUT_BUFFER_0_VALID, field::INPUT_BUFFER_1_VALID, field::OUTPUT_BUFFER_VALID] {
        if get(initial, f) != 0 && get(decoded, f) == 0 {
            set(fresh, f, 0);
        }
    }
    for f in
        [field::INPUT_BUFFER_READ_OFFSET, field::ERROR_STATUS, field::CURRENT_BUFFER, field::OUTPUT_BUFFER_READ_OFFSET]
    {
        set(fresh, f, get(decoded, f));
    }
}

fn any_input_valid(d: &[u32; 16]) -> bool {
    get(d, field::INPUT_BUFFER_0_VALID) != 0 || get(d, field::INPUT_BUFFER_1_VALID) != 0
}

fn input_valid(d: &[u32; 16], buffer: u32) -> bool {
    get(d, if buffer == 0 { field::INPUT_BUFFER_0_VALID } else { field::INPUT_BUFFER_1_VALID }) != 0
}

fn input_ptr(d: &[u32; 16], buffer: u32) -> u32 {
    get(d, if buffer == 0 { field::INPUT_BUFFER_0_PTR } else { field::INPUT_BUFFER_1_PTR })
}

fn input_packet_count(d: &[u32; 16], buffer: u32) -> u32 {
    get(d, if buffer == 0 { field::INPUT_BUFFER_0_PACKET_COUNT } else { field::INPUT_BUFFER_1_PACKET_COUNT })
}

fn current_buffer(d: &[u32; 16]) -> u32 {
    get(d, field::CURRENT_BUFFER)
}

fn read_offset(d: &[u32; 16]) -> u32 {
    get(d, field::INPUT_BUFFER_READ_OFFSET)
}

fn set_read_offset(d: &mut [u32; 16], v: u32) {
    set(d, field::INPUT_BUFFER_READ_OFFSET, v)
}

/// The current buffer is used up: invalidate it and move to the other one.
fn swap_input_buffer(d: &mut [u32; 16]) {
    let cur = current_buffer(d);
    set(d, if cur == 0 { field::INPUT_BUFFER_0_VALID } else { field::INPUT_BUFFER_1_VALID }, 0);
    set(d, field::CURRENT_BUFFER, cur ^ 1);
    set_read_offset(d, BITS_PER_PACKET_HEADER);
}

/// Bit offset of the first frame in a packet, from the packet start.
fn packet_frame_offset(packet: &[u8]) -> u32 {
    ((((packet[0] & 3) as u32) << 13) | ((packet[1] as u32) << 5) | ((packet[2] >> 3) as u32)) + 32
}

fn packet_is_xma2(packet: &[u8]) -> bool {
    packet[2] & 7 == 1
}

fn packet_frame_count(packet: &[u8]) -> u32 {
    (packet[0] >> 2) as u32
}

fn packet_skip_count(packet: &[u8]) -> u32 {
    packet[3] as u32
}

/// `n` (<= 32) bits at bit `pos` of `data`; bits past the end read as zero.
fn peek_bits(data: &[u8], pos: u32, n: u32) -> u32 {
    let byte = (pos >> 3) as usize;
    let mut b = [0u8; 8];
    if byte < data.len() {
        let len = (data.len() - byte).min(8);
        b[..len].copy_from_slice(&data[byte..byte + len]);
    }
    ((u64::from_be_bytes(b) << (pos & 7)) >> (64 - n)) as u32
}

struct PacketInfo {
    /// Frames that begin in the packet.
    frame_count: u32,
    /// Index of the frame at the read offset.
    current_frame: u32,
    /// Its size in bits; 0 when its header is split across packets.
    current_frame_size: u32,
}

impl PacketInfo {
    fn is_last_frame_in_packet(&self) -> bool {
        self.frame_count == 0 || self.current_frame == self.frame_count - 1
    }
}

/// XmaContextNew::GetPacketInfo: walks the frames that start in `packet` to
/// find the one at `frame_offset` and whether it is the packet's last.
fn packet_info(packet: &[u8], frame_offset: u32) -> PacketInfo {
    let mut info = PacketInfo { frame_count: 0, current_frame: 0, current_frame_size: 0 };
    let first = packet_frame_offset(packet);
    let mut pos = first.min(BITS_PER_PACKET);
    if frame_offset < first {
        info.current_frame_size = first - frame_offset;
    }
    loop {
        let remaining = BITS_PER_PACKET - pos;
        if remaining < BITS_PER_FRAME_HEADER {
            // The header of this frame continues in the next packet: count
            // it, with size 0 so the caller resolves it across packets.
            if remaining > 0 {
                if pos == frame_offset {
                    info.current_frame = info.frame_count;
                    info.current_frame_size = 0;
                }
                info.frame_count += 1;
            }
            break;
        }
        let frame_size = peek_bits(packet, pos, BITS_PER_FRAME_HEADER);
        if frame_size == 0 || frame_size == MAX_FRAME_LENGTH {
            break;
        }
        if pos == frame_offset {
            info.current_frame = info.frame_count;
            info.current_frame_size = frame_size;
        }
        info.frame_count += 1;
        if frame_size > remaining {
            break;
        }
        pos = (pos + frame_size - 1).min(BITS_PER_PACKET);
        let more = peek_bits(packet, pos, 1);
        pos = (pos + 1).min(BITS_PER_PACKET);
        if more == 0 {
            break;
        }
    }
    if packet_is_xma2(packet) {
        let header_count = packet_frame_count(packet);
        if header_count > info.frame_count {
            // A split frame header the scan could not read: trust the header.
            if info.current_frame_size == 0 {
                info.current_frame = info.frame_count;
            }
            info.frame_count = header_count;
        }
    }
    info
}

/// The output ring, offsets in bytes.
struct OutputRing {
    base: u32,
    capacity: u32,
    read: u32,
    write: u32,
    /// Bytes written during this kick.
    written: u32,
}

impl OutputRing {
    fn new(d: &[u32; 16]) -> OutputRing {
        OutputRing {
            base: get(d, field::OUTPUT_BUFFER_PTR),
            capacity: get(d, field::OUTPUT_BUFFER_BLOCK_COUNT) * OUTPUT_BYTES_PER_BLOCK,
            read: get(d, field::OUTPUT_BUFFER_READ_OFFSET) * OUTPUT_BYTES_PER_BLOCK,
            write: get(d, field::OUTPUT_BUFFER_WRITE_OFFSET) * OUTPUT_BYTES_PER_BLOCK,
            written: 0,
        }
    }

    /// Equal offsets read as empty (so a completely filled ring looks empty,
    /// as in Xenia's RingBuffer).
    fn read_count(&self) -> u32 {
        if self.write >= self.read { self.write - self.read } else { self.capacity - self.read + self.write }
    }

    fn write_count(&self) -> u32 {
        self.capacity - self.read_count()
    }

    fn is_empty(&self) -> bool {
        self.read == self.write
    }

    fn write(&mut self, mem: &mut impl XmaMemory, data: &[u8]) {
        let count = (data.len() as u32).min(self.capacity);
        if count == 0 {
            return;
        }
        let first = count.min(self.capacity - self.write);
        mem.write(self.base.wrapping_add(self.write), &data[..first as usize]);
        if first < count {
            mem.write(self.base, &data[first as usize..count as usize]);
        }
        self.write = (self.write + count) % self.capacity;
        self.written += count;
    }
}

/// Host-side state of one hardware XMA context, kept between kicks.
pub struct Context<D: FrameDecoding = FrameDecoder> {
    /// Created on first use and recreated when the rate or channel count changes.
    decoder: Option<Box<D>>,
    decoder_config: (u32, usize),
    /// The last decoded frame, as output blocks.
    raw_frame: [u8; 2 * BYTES_PER_FRAME_CHANNEL as usize],
    /// Output blocks of `raw_frame` not yet written to the ring.
    current_frame_remaining_subframes: u32,
    /// Blocks of the current frame to output at most (loop end), 0: all.
    loop_frame_output_limit: u32,
    /// Skip leading blocks of the next frame (loop start).
    loop_start_skip_pending: bool,
    /// Free ring blocks during a kick.
    remaining_subframe_blocks_in_output_buffer: i32,
    frames_decoded: u64,
    frames_failed: u64,
}

impl<D: FrameDecoding> Default for Context<D> {
    fn default() -> Self {
        Context {
            decoder: None,
            decoder_config: (0, 0),
            raw_frame: [0; 2 * BYTES_PER_FRAME_CHANNEL as usize],
            current_frame_remaining_subframes: 0,
            loop_frame_output_limit: 0,
            loop_start_skip_pending: false,
            remaining_subframe_blocks_in_output_buffer: 0,
            frames_decoded: 0,
            frames_failed: 0,
        }
    }
}

impl<D: FrameDecoding> Context<D> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Back to the freshly allocated state (XMACreateContext / XMAReleaseContext).
    pub fn reset(&mut self) {
        // Keep the decoder allocation; only its state goes.
        let (decoder, config) = (self.decoder.take(), self.decoder_config);
        *self = Self::default();
        if let Some(mut decoder) = decoder {
            decoder.reset();
            self.decoder = Some(decoder);
            self.decoder_config = config;
        }
    }

    /// Frames decoded and frames that failed to decode since creation.
    pub fn frame_stats(&self) -> (u64, u64) {
        (self.frames_decoded, self.frames_failed)
    }

    /// XmaContextNew::ClearLocked (the reset done by XMAInitializeContext):
    /// inputs and output invalid, offsets rewound, pending output dropped.
    /// Unlike Xenia, the decoder's overlap from the previous stream is
    /// dropped too, so a new stream does not start with the old one's tail.
    pub fn clear(&mut self, d: &mut [u32; 16]) {
        set(d, field::INPUT_BUFFER_0_VALID, 0);
        set(d, field::INPUT_BUFFER_1_VALID, 0);
        set(d, field::OUTPUT_BUFFER_VALID, 0);
        set_read_offset(d, BITS_PER_PACKET_HEADER);
        set(d, field::OUTPUT_BUFFER_READ_OFFSET, 0);
        set(d, field::OUTPUT_BUFFER_WRITE_OFFSET, 0);
        self.current_frame_remaining_subframes = 0;
        self.loop_frame_output_limit = 0;
        self.loop_start_skip_pending = false;
        if let Some(decoder) = &mut self.decoder {
            decoder.reset();
        }
    }

    /// A kick (XmaContextNew::Work): decodes from the input buffers into the
    /// output ring while it has room, updating the context dwords `d` (host
    /// order, as loaded from guest memory) in place.
    pub fn work(&mut self, d: &mut [u32; 16], mem: &mut impl XmaMemory) {
        let initial = *d;
        if get(d, field::OUTPUT_BUFFER_VALID) == 0 {
            return;
        }
        let mut ring = OutputRing::new(d);
        self.remaining_subframe_blocks_in_output_buffer = (ring.write_count() / OUTPUT_BYTES_PER_BLOCK) as i32;

        if get(d, field::INPUT_BUFFER_0_PACKET_COUNT) | get(d, field::INPUT_BUFFER_1_PACKET_COUNT) == 0 {
            // Consume-only context: only drain a pending frame, and leave the
            // offsets alone otherwise or titles re-read stale PCM.
            if self.current_frame_remaining_subframes == 0 {
                return;
            }
            self.consume(&mut ring, d, mem);
            set(d, field::OUTPUT_BUFFER_WRITE_OFFSET, ring.write / OUTPUT_BYTES_PER_BLOCK);
            if ring.is_empty() {
                set(d, field::OUTPUT_BUFFER_VALID, 0);
            }
            return;
        }

        // Room for one pass of Consume() plus the padding headroom.
        let minimum = get(d, field::SUBFRAME_DECODE_COUNT).max(1) as i32 + get(d, field::OUTPUT_BUFFER_PADDING) as i32;
        if minimum > self.remaining_subframe_blocks_in_output_buffer {
            return;
        }

        let mut idle_passes = 0;
        while self.remaining_subframe_blocks_in_output_buffer >= minimum {
            let pre_decode_offset = read_offset(d);
            let pre_remaining = self.current_frame_remaining_subframes;
            self.decode(d, mem);
            self.consume(&mut ring, d, mem);
            // Keep draining a partly written frame even once input is gone:
            // nothing else would deliver its remainder.
            if (!any_input_valid(d) || get(d, field::ERROR_STATUS) == ERROR_STATUS_STOP)
                && self.current_frame_remaining_subframes == 0
            {
                break;
            }
            if pre_remaining == 0 && self.current_frame_remaining_subframes == 0 {
                if read_offset(d) == pre_decode_offset {
                    break; // No progress: wait for the next buffer.
                }
                idle_passes += 1;
                if idle_passes > MAX_IDLE_PASSES {
                    break;
                }
            } else {
                idle_passes = 0;
            }
        }

        if any_input_valid(&initial) || ring.written > 0 {
            set(d, field::OUTPUT_BUFFER_WRITE_OFFSET, ring.write / OUTPUT_BYTES_PER_BLOCK);
        } else if get(d, field::OUTPUT_BUFFER_WRITE_OFFSET) != get(d, field::OUTPUT_BUFFER_READ_OFFSET) {
            // Starved of input: some titles (NFS) detect stalls by write ==
            // read. Xenia applies this even when the kick just drained the
            // rest of the last frame, discarding it; here only when nothing
            // was written.
            set(d, field::OUTPUT_BUFFER_WRITE_OFFSET, get(d, field::OUTPUT_BUFFER_READ_OFFSET));
            set(d, field::OUTPUT_BUFFER_VALID, 0);
        }
        if self.remaining_subframe_blocks_in_output_buffer == 0 && ring.is_empty() {
            // The ring is full.
            set(d, field::OUTPUT_BUFFER_VALID, 0);
        }
    }

    /// XmaContextNew::Consume: writes up to SUBFRAME_DECODE_COUNT blocks of
    /// the pending frame to the ring.
    fn consume(&mut self, ring: &mut OutputRing, d: &[u32; 16], mem: &mut impl XmaMemory) {
        if self.current_frame_remaining_subframes == 0 {
            return;
        }
        let total = BLOCKS_PER_FRAME_CHANNEL << get(d, field::IS_STEREO);
        let consumed = total.saturating_sub(self.current_frame_remaining_subframes);
        let padding = get(d, field::OUTPUT_BUFFER_PADDING) as i32;
        if self.loop_frame_output_limit > 0 && consumed >= self.loop_frame_output_limit {
            // Loop end reached inside this frame: drop the rest, charging the
            // headroom as if it completed.
            self.remaining_subframe_blocks_in_output_buffer -= padding;
            self.current_frame_remaining_subframes = 0;
            self.loop_frame_output_limit = 0;
            return;
        }
        let mut to_write = self.current_frame_remaining_subframes.min(get(d, field::SUBFRAME_DECODE_COUNT).max(1));
        if self.loop_frame_output_limit > 0 {
            to_write = to_write.min(self.loop_frame_output_limit - consumed);
        }
        let start = (consumed * OUTPUT_BYTES_PER_BLOCK) as usize;
        let end = start + (to_write * OUTPUT_BYTES_PER_BLOCK) as usize;
        if let Some(blocks) = self.raw_frame.get(start..end) {
            ring.write(mem, blocks);
        }
        let headroom = if self.current_frame_remaining_subframes == to_write { padding } else { 0 };
        self.remaining_subframe_blocks_in_output_buffer -= to_write as i32 + headroom;
        self.current_frame_remaining_subframes -= to_write;
    }

    /// XmaContextNew::UpdateLoopStatus: at the loop end, jump to the start.
    fn update_loop_status(&mut self, d: &mut [u32; 16]) {
        let count = get(d, field::LOOP_COUNT);
        if count == 0 {
            return;
        }
        let start = get(d, field::LOOP_START).max(BITS_PER_PACKET_HEADER);
        let end = get(d, field::LOOP_END).max(BITS_PER_PACKET_HEADER);
        if read_offset(d) != end {
            return;
        }
        set_read_offset(d, start);
        self.loop_start_skip_pending = true;
        if count != 255 {
            set(d, field::LOOP_COUNT, count - 1);
        }
    }

    /// XmaContextNew::Decode: decodes the frame at the read offset (if no
    /// frame is pending) and advances the read offset to the next frame.
    fn decode(&mut self, d: &mut [u32; 16], mem: &mut impl XmaMemory) {
        if !any_input_valid(d) || self.current_frame_remaining_subframes > 0 {
            return;
        }
        if !input_valid(d, current_buffer(d)) {
            swap_input_buffer(d);
            if !input_valid(d, current_buffer(d)) {
                return;
            }
        }

        // Detect the loop end frame before UpdateLoopStatus moves the offset.
        let is_loop_end_frame =
            get(d, field::LOOP_COUNT) > 0 && read_offset(d) == get(d, field::LOOP_END).max(BITS_PER_PACKET_HEADER);
        self.update_loop_status(d);

        if get(d, field::OUTPUT_BUFFER_BLOCK_COUNT) == 0 {
            return;
        }
        // Some titles kick with the offset inside the packet header.
        if read_offset(d) < BITS_PER_PACKET_HEADER {
            set_read_offset(d, BITS_PER_PACKET_HEADER);
        }

        let buffer = current_buffer(d);
        let packet_count = input_packet_count(d, buffer);
        let offset = read_offset(d);
        if offset >= packet_count * BITS_PER_PACKET {
            return; // Invalid packet index; the kick stalls until the title fixes it.
        }
        let packet_index = offset / BITS_PER_PACKET;
        let mut packet = [0u8; BYTES_PER_PACKET as usize];
        if !mem.read(input_ptr(d, buffer).wrapping_add(packet_index * BYTES_PER_PACKET), &mut packet) {
            swap_input_buffer(d);
            return;
        }
        let first_frame_offset = packet_frame_offset(&packet);
        let mut relative_offset = offset % BITS_PER_PACKET;
        // Before the first frame: the tail of a frame whose start we lack.
        if relative_offset < first_frame_offset {
            set_read_offset(d, packet_index * BITS_PER_PACKET + first_frame_offset);
            relative_offset = first_frame_offset;
        }

        let skip_count = packet_skip_count(&packet);
        if skip_count == 0xFF {
            // No frame starts here; go to the next packet.
            let next_index = packet_index + 1;
            let next_offset = self.next_packet_read_offset(d, mem, next_index, packet_count);
            if next_index >= packet_count || next_offset == BITS_PER_PACKET_HEADER {
                swap_input_buffer(d);
            }
            set_read_offset(d, next_offset);
            return;
        }

        let mut info = packet_info(&packet, relative_offset);
        let next_packet_index = packet_index + skip_count + 1;
        // Packet data of this packet and (for frames crossing it) the next
        // packet of the stream, plus zero padding for the bit reader.
        let mut input = [0u8; 2 * BYTES_PER_PACKET_DATA + 16];

        if info.current_frame_size == 0 {
            // Frame header split across packets.
            let Some(next) = self.next_packet(d, mem, next_packet_index, packet_count) else {
                if !Self::next_buffer_pending(d, next_packet_index, packet_count) {
                    swap_input_buffer(d);
                }
                return;
            };
            input[..BYTES_PER_PACKET_DATA].copy_from_slice(&packet[BYTES_PER_PACKET_HEADER..]);
            input[BYTES_PER_PACKET_DATA..2 * BYTES_PER_PACKET_DATA].copy_from_slice(&next[BYTES_PER_PACKET_HEADER..]);
            let frame_size = peek_bits(&input, relative_offset - BITS_PER_PACKET_HEADER, BITS_PER_FRAME_HEADER);
            if frame_size == MAX_FRAME_LENGTH {
                set(d, field::ERROR_STATUS, ERROR_STATUS_STOP);
                return;
            }
            info.current_frame_size = frame_size;
        }

        let offset = read_offset(d);
        let bits_remaining = ((packet_index + 1) * BITS_PER_PACKET).saturating_sub(offset);
        let bits_to_copy = bits_remaining.min(info.current_frame_size);
        if bits_to_copy == 0 {
            swap_input_buffer(d);
            return;
        }
        if info.is_last_frame_in_packet() && bits_remaining < info.current_frame_size {
            // The frame continues in the stream's next packet.
            let Some(next) = self.next_packet(d, mem, next_packet_index, packet_count) else {
                if !Self::next_buffer_pending(d, next_packet_index, packet_count) {
                    set(d, field::ERROR_STATUS, ERROR_STATUS_STOP);
                }
                return;
            };
            input[BYTES_PER_PACKET_DATA..2 * BYTES_PER_PACKET_DATA].copy_from_slice(&next[BYTES_PER_PACKET_HEADER..]);
        }
        input[..BYTES_PER_PACKET_DATA].copy_from_slice(&packet[BYTES_PER_PACKET_HEADER..]);

        self.raw_frame.fill(0);
        let stereo = get(d, field::IS_STEREO);
        let channels = 1 + stereo as usize;
        self.prepare_decoder(get(d, field::SAMPLE_RATE), channels);
        let decoder = self.decoder.as_deref_mut().expect("decoder prepared");
        match decoder.decode_frame(&input, (relative_offset - BITS_PER_PACKET_HEADER) as usize) {
            Ok(_) => {
                convert_to_be_s16(decoder, channels, &mut self.raw_frame);
                self.frames_decoded += 1;
                self.current_frame_remaining_subframes = BLOCKS_PER_FRAME_CHANNEL << stereo;
                // Loop end: output only subframes 0..=LOOP_SUBFRAME_END.
                self.loop_frame_output_limit =
                    if is_loop_end_frame { (get(d, field::LOOP_SUBFRAME_END) + 1) << stereo } else { 0 };
                // Loop start: skip LOOP_SUBFRAME_SKIP leading subframes.
                if self.loop_start_skip_pending {
                    let skip = get(d, field::LOOP_SUBFRAME_SKIP) << stereo;
                    if skip < self.current_frame_remaining_subframes {
                        self.current_frame_remaining_subframes -= skip;
                    }
                    self.loop_start_skip_pending = false;
                }
            }
            Err(_) => self.frames_failed += 1,
        }

        // Where the next frame starts.
        if !info.is_last_frame_in_packet() {
            let next_frame_offset = (offset + bits_to_copy) % BITS_PER_PACKET;
            set_read_offset(d, packet_index * BITS_PER_PACKET + next_frame_offset);
            return;
        }
        let mut next_offset = self.next_packet_read_offset(d, mem, next_packet_index, packet_count);
        if next_packet_index >= packet_count || next_offset == BITS_PER_PACKET_HEADER {
            swap_input_buffer(d);
        }
        if next_offset == BITS_PER_PACKET_HEADER && any_input_valid(d) {
            // At the start of the next buffer: go to its first frame, or skip
            // the buffer if no frame starts in its first packet.
            let mut header = [0u8; BYTES_PER_PACKET_HEADER];
            if !mem.read(input_ptr(d, current_buffer(d)), &mut header) {
                swap_input_buffer(d);
                return;
            }
            next_offset = packet_frame_offset(&header);
            if next_offset > MAX_FRAME_SIZE_IN_BITS {
                swap_input_buffer(d);
                return;
            }
        }
        set_read_offset(d, next_offset);
    }

    /// PrepareDecoder: the decoder for this rate and channel count.
    fn prepare_decoder(&mut self, rate_id: u32, channels: usize) {
        let rate = SAMPLE_RATES[rate_id.min(3) as usize];
        if self.decoder.is_none() || self.decoder_config != (rate, channels) {
            self.decoder = Some(Box::new(D::new(rate, channels)));
            self.decoder_config = (rate, channels);
        }
    }

    /// GetPacketHandle: (buffer, packet) of packet `index` counted from the
    /// current buffer's start, continuing into the other buffer.
    fn packet_handle(d: &[u32; 16], index: u32, current_count: u32) -> Option<(u32, u32)> {
        let (mut buffer, mut index) = (current_buffer(d), index);
        if index >= current_count {
            buffer ^= 1;
            index -= current_count;
        }
        let valid = input_valid(d, buffer) && input_ptr(d, buffer) != 0 && index < input_packet_count(d, buffer);
        valid.then_some((buffer, index))
    }

    /// Packet `index` lies in the other input buffer, which the title has not
    /// handed over yet. Xenia gives up on the frame here (dropping the buffer
    /// or flagging an error); the hardware waits for the buffer, and so does
    /// this port: the kick stalls with the offset unchanged.
    fn next_buffer_pending(d: &[u32; 16], index: u32, current_count: u32) -> bool {
        index >= current_count && !input_valid(d, current_buffer(d) ^ 1)
    }

    /// GetNextPacket.
    fn next_packet(
        &self,
        d: &[u32; 16],
        mem: &impl XmaMemory,
        index: u32,
        current_count: u32,
    ) -> Option<[u8; BYTES_PER_PACKET as usize]> {
        let (buffer, index) = Self::packet_handle(d, index, current_count)?;
        let mut packet = [0u8; BYTES_PER_PACKET as usize];
        mem.read(input_ptr(d, buffer).wrapping_add(index * BYTES_PER_PACKET), &mut packet).then_some(packet)
    }

    /// GetNextPacketReadOffset: the read offset of the first frame starting
    /// at or after packet `index` (following skip counts), in the
    /// coordinates of the buffer holding it; 32 if there is none.
    fn next_packet_read_offset(&self, d: &[u32; 16], mem: &impl XmaMemory, index: u32, current_count: u32) -> u32 {
        let Some((buffer, mut index)) = Self::packet_handle(d, index, current_count) else {
            return BITS_PER_PACKET_HEADER;
        };
        let (base, count) = (input_ptr(d, buffer), input_packet_count(d, buffer));
        while index < count {
            let mut header = [0u8; BYTES_PER_PACKET_HEADER];
            if !mem.read(base.wrapping_add(index * BYTES_PER_PACKET), &mut header) {
                break;
            }
            let frame_offset = packet_frame_offset(&header);
            if frame_offset <= MAX_FRAME_SIZE_IN_BITS {
                return index * BITS_PER_PACKET + frame_offset;
            }
            // Entirely the continuation of a frame: follow the skip count.
            let skip = packet_skip_count(&header);
            if skip == 0xFF {
                break;
            }
            index += skip + 1;
        }
        BITS_PER_PACKET_HEADER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes synthetic frames `[len:15][id:16][filler][more:1]` into a
    /// sample ramp that identifies the frame.
    struct FakeDecoder {
        channels: usize,
        pcm: [[f32; SAMPLES_PER_FRAME]; 2],
    }

    fn sample(id: u32, channel: usize, i: usize) -> i16 {
        ((id as usize * 613 + channel * 7 + i) % 30000) as i16
    }

    impl FrameDecoding for FakeDecoder {
        fn new(_: u32, channels: usize) -> Self {
            FakeDecoder { channels, pcm: [[0.0; SAMPLES_PER_FRAME]; 2] }
        }

        fn decode_frame(&mut self, data: &[u8], bit_offset: usize) -> Result<FrameInfo, DecodeError> {
            let len = peek_bits(data, bit_offset as u32, 15) as usize;
            let id = peek_bits(data, bit_offset as u32 + 15, 16);
            if id == 0xDEAD {
                return Err(DecodeError::TileHeader);
            }
            for (c, pcm) in self.pcm[..self.channels].iter_mut().enumerate() {
                for (i, v) in pcm.iter_mut().enumerate() {
                    *v = sample(id, c, i) as f32 / 32767.0;
                }
            }
            Ok(FrameInfo { bits: len, ..FrameInfo::default() })
        }

        fn pcm(&self, channel: usize) -> &[f32; SAMPLES_PER_FRAME] {
            &self.pcm[channel]
        }

        fn reset(&mut self) {}
    }

    /// Builds XMA2 packets holding frames of the given (bit length, id).
    /// Returns the packets and each frame's read offset (bit offset in the
    /// packet sequence).
    fn build_packets(frames: &[(usize, u32)]) -> (Vec<u8>, Vec<u32>) {
        let payload_bits = BYTES_PER_PACKET_DATA * 8;
        let total: usize = frames.iter().map(|f| f.0).sum();
        let packet_count = total.div_ceil(payload_bits);
        let mut payload = vec![0xFFu8; packet_count * BYTES_PER_PACKET_DATA];
        let mut put = |pos: usize, value: u32, n: usize| {
            for i in 0..n {
                let bit = (value >> (n - 1 - i)) & 1;
                let p = pos + i;
                payload[p >> 3] = (payload[p >> 3] & !(0x80 >> (p & 7))) | ((bit as u8) << (7 - (p & 7)));
            }
        };
        let mut starts = Vec::new();
        let mut pos = 0;
        for (k, &(len, id)) in frames.iter().enumerate() {
            starts.push(pos);
            put(pos, len as u32, 15);
            put(pos + 15, id, 16);
            for b in pos + 31..pos + len - 1 {
                put(b, 0, 1);
            }
            let next_in_same_packet = k + 1 < frames.len() && (pos + len) / payload_bits == pos / payload_bits;
            put(pos + len - 1, next_in_same_packet as u32, 1);
            pos += len;
        }
        let mut packets = Vec::new();
        let mut offsets = Vec::new();
        for p in 0..packet_count {
            let in_packet: Vec<usize> = starts.iter().copied().filter(|&s| s / payload_bits == p).collect();
            let first = in_packet.first().map_or(0x7FFF, |&s| (s % payload_bits) as u32);
            packets.push(((in_packet.len() as u8) << 2) | (first >> 13) as u8);
            packets.push((first >> 5) as u8);
            packets.push(((first as u8) << 3) | 1);
            packets.push(0);
            packets.extend_from_slice(&payload[p * BYTES_PER_PACKET_DATA..(p + 1) * BYTES_PER_PACKET_DATA]);
        }
        for s in starts {
            let p = s / payload_bits;
            offsets.push((p * BITS_PER_PACKET as usize + s % payload_bits + 32) as u32);
        }
        (packets, offsets)
    }

    struct Memory(Vec<u8>);

    impl XmaMemory for Memory {
        fn read(&self, addr: u32, buf: &mut [u8]) -> bool {
            let a = addr as usize;
            self.0.get(a..a + buf.len()).map(|s| buf.copy_from_slice(s)).is_some()
        }

        fn write(&mut self, addr: u32, data: &[u8]) -> bool {
            let a = addr as usize;
            self.0.get_mut(a..a + data.len()).map(|s| s.copy_from_slice(data)).is_some()
        }
    }

    const IN: [u32; 2] = [0x1000, 0x10_0000];
    const OUT: u32 = 0x20_0000;

    struct Harness {
        ctx: Context<FakeDecoder>,
        d: [u32; 16],
        mem: Memory,
        ring_blocks: u32,
        out: Vec<i16>,
    }

    impl Harness {
        fn new(stereo: bool, ring_blocks: u32, decode_count: u32) -> Harness {
            let mut d = [0u32; 16];
            set(&mut d, field::INPUT_BUFFER_0_PTR, IN[0]);
            set(&mut d, field::INPUT_BUFFER_1_PTR, IN[1]);
            set(&mut d, field::OUTPUT_BUFFER_PTR, OUT);
            set(&mut d, field::OUTPUT_BUFFER_BLOCK_COUNT, ring_blocks);
            set(&mut d, field::SUBFRAME_DECODE_COUNT, decode_count);
            set(&mut d, field::IS_STEREO, stereo as u32);
            let mut ctx = Context::new();
            ctx.clear(&mut d);
            Harness { ctx, d, mem: Memory(vec![0; OUT as usize + 32 * 256]), ring_blocks, out: Vec::new() }
        }

        fn submit(&mut self, buffer: u32, packets: &[u8]) {
            let (valid, count) = if buffer == 0 {
                (field::INPUT_BUFFER_0_VALID, field::INPUT_BUFFER_0_PACKET_COUNT)
            } else {
                (field::INPUT_BUFFER_1_VALID, field::INPUT_BUFFER_1_PACKET_COUNT)
            };
            assert!(self.mem.write(IN[buffer as usize], packets));
            set(&mut self.d, count, (packets.len() / BYTES_PER_PACKET as usize) as u32);
            set(&mut self.d, valid, 1);
        }

        /// One kick, then drains the ring. Returns the blocks produced.
        fn kick(&mut self) -> usize {
            set(&mut self.d, field::OUTPUT_BUFFER_VALID, 1);
            let input_before = any_input_valid(&self.d);
            self.ctx.work(&mut self.d, &mut self.mem);
            let mut read = get(&self.d, field::OUTPUT_BUFFER_READ_OFFSET);
            let write = get(&self.d, field::OUTPUT_BUFFER_WRITE_OFFSET);
            let full = input_before && get(&self.d, field::OUTPUT_BUFFER_VALID) == 0 && read == write;
            let mut blocks = 0;
            while read != write || (full && blocks == 0) {
                let mut block = [0u8; 256];
                assert!(self.mem.read(OUT + read * 256, &mut block));
                self.out.extend(block.as_chunks::<2>().0.iter().map(|&b| i16::from_be_bytes(b)));
                read = (read + 1) % self.ring_blocks;
                blocks += 1;
            }
            set(&mut self.d, field::OUTPUT_BUFFER_READ_OFFSET, read);
            blocks
        }

        /// Streams `packets` through both buffers, `per_buffer` at a time.
        fn stream(&mut self, packets: &[u8], per_buffer: usize) {
            let mut chunks = packets.chunks(per_buffer * BYTES_PER_PACKET as usize);
            let mut idle = 0;
            loop {
                for b in 0..2 {
                    if !input_valid(&self.d, b)
                        && let Some(c) = chunks.next()
                    {
                        self.submit(b, c);
                    }
                }
                if self.kick() == 0 {
                    idle += 1;
                    if idle > 3 {
                        break;
                    }
                } else {
                    idle = 0;
                }
            }
            assert!(!any_input_valid(&self.d), "input left over");
        }
    }

    fn expected(ids: &[u32], channels: usize) -> Vec<i16> {
        let mut v = Vec::new();
        for &id in ids {
            for i in 0..SAMPLES_PER_FRAME {
                for c in 0..channels {
                    v.push(sample(id, c, i));
                }
            }
        }
        v
    }

    fn varied_frames(n: u32) -> Vec<(usize, u32)> {
        (0..n).map(|i| (40 + (i as usize * 7919) % 6000, i + 1)).collect()
    }

    #[test]
    fn frames_cross_packets_and_buffers() {
        let frames = varied_frames(60);
        let (packets, _) = build_packets(&frames);
        let ids: Vec<u32> = frames.iter().map(|f| f.1).collect();
        for stereo in [false, true] {
            for (per_buffer, ring, count) in [(1, 31, 1), (2, 8, 2), (3, 31, 8), (64, 5, 3)] {
                let mut h = Harness::new(stereo, ring, count);
                h.stream(&packets, per_buffer);
                let want = expected(&ids, 1 + stereo as usize);
                let first_diff = h.out.iter().zip(&want).position(|(a, b)| a != b);
                assert!(
                    h.out == want,
                    "stereo {stereo}, {per_buffer} per buffer: {} vs {} samples, first difference at {first_diff:?}",
                    h.out.len(),
                    want.len()
                );
                assert_eq!(get(&h.d, field::ERROR_STATUS), 0);
            }
        }
    }

    #[test]
    fn frame_header_split_across_packets() {
        // The second frame starts 10 bits before the end of the first packet.
        let first = BYTES_PER_PACKET_DATA * 8 - 10;
        let frames = [(first, 1), (900, 2), (1200, 3)];
        let (packets, offsets) = build_packets(&frames);
        assert_eq!(offsets[1] % BITS_PER_PACKET, BITS_PER_PACKET - 10);
        let mut h = Harness::new(false, 31, 4);
        h.stream(&packets, 1);
        assert!(h.out == expected(&[1, 2, 3], 1));
    }

    #[test]
    fn failed_frames_are_skipped() {
        let frames = [(300, 1), (300, 0xDEAD), (300, 3)];
        let (packets, _) = build_packets(&frames);
        let mut h = Harness::new(true, 16, 8);
        h.stream(&packets, 4);
        assert!(h.out == expected(&[1, 3], 2));
        assert_eq!(h.ctx.frame_stats(), (2, 1));
    }

    #[test]
    fn decode_count_limits_blocks_per_pass_and_padding_reserves_room() {
        let frames = varied_frames(6);
        let (packets, _) = build_packets(&frames);
        let mut h = Harness::new(false, 6, 1);
        set(&mut h.d, field::OUTPUT_BUFFER_PADDING, 2);
        h.submit(0, &packets);
        // 6 free blocks, 2 reserved per frame: one frame (4 blocks) fits.
        assert_eq!(h.kick(), 4);
        h.stream(&[], 1);
        assert!(h.out == expected(&frames.iter().map(|f| f.1).collect::<Vec<_>>(), 1));
    }

    #[test]
    fn full_ring_invalidates_output() {
        let frames = varied_frames(20);
        let (packets, _) = build_packets(&frames);
        let mut h = Harness::new(true, 16, 8);
        h.submit(0, &packets);
        set(&mut h.d, field::OUTPUT_BUFFER_VALID, 1);
        h.ctx.work(&mut h.d, &mut h.mem);
        // Two stereo frames fill the ring exactly; it reads as full.
        assert_eq!(get(&h.d, field::OUTPUT_BUFFER_VALID), 0);
        assert_eq!(get(&h.d, field::OUTPUT_BUFFER_WRITE_OFFSET), 0);
        // Nothing more is written until the title consumes.
        let offset = read_offset(&h.d);
        h.ctx.work(&mut h.d, &mut h.mem);
        assert_eq!(read_offset(&h.d), offset);
    }

    #[test]
    fn loop_repeats_region_with_subframe_trim() {
        let frames: Vec<(usize, u32)> = (1..=8).map(|i| (1500, i)).collect();
        let (packets, offsets) = build_packets(&frames);
        let mut h = Harness::new(false, 31, 4);
        // Loop back from frame 6 to frame 3 once; the frame decoded after the
        // jump skips its first subframe and stops after subframe 2.
        set(&mut h.d, field::LOOP_START, offsets[2]);
        set(&mut h.d, field::LOOP_END, offsets[5]);
        set(&mut h.d, field::LOOP_COUNT, 1);
        set(&mut h.d, field::LOOP_SUBFRAME_SKIP, 1);
        set(&mut h.d, field::LOOP_SUBFRAME_END, 2);
        h.stream(&packets, 16);
        let mut want = expected(&[1, 2, 3, 4, 5], 1);
        want.extend_from_slice(&expected(&[3], 1)[128..384]);
        want.extend(expected(&[4, 5, 6, 7, 8], 1));
        assert!(h.out == want);
        assert_eq!(get(&h.d, field::LOOP_COUNT), 0);
    }

    #[test]
    fn invalid_read_offset_stalls_without_spinning() {
        let (packets, _) = build_packets(&varied_frames(3));
        let mut h = Harness::new(false, 8, 2);
        h.submit(0, &packets);
        set_read_offset(&mut h.d, 63 * BITS_PER_PACKET);
        assert_eq!(h.kick(), 0);
        assert_eq!(read_offset(&h.d), 63 * BITS_PER_PACKET);
        assert!(input_valid(&h.d, 0));
    }

    #[test]
    fn consume_only_context_drains_pending_frame() {
        let (packets, _) = build_packets(&[(400, 9), (400, 10)]);
        let mut h = Harness::new(true, 8, 2);
        h.submit(0, &packets);
        // 8 blocks, decode count 2: the first frame's 8 blocks fill the ring.
        assert_eq!(h.kick(), 8);
        // The title detaches its input; the second frame is decoded but pending.
        set(&mut h.d, field::INPUT_BUFFER_0_PACKET_COUNT, 0);
        assert_eq!(h.kick(), 0);
        set(&mut h.d, field::INPUT_BUFFER_0_PACKET_COUNT, 1);
        h.stream(&[], 1);
        assert!(h.out == expected(&[9, 10], 2));
    }

    #[test]
    fn clear_rewinds_and_drops_pending_output() {
        let (packets, _) = build_packets(&varied_frames(4));
        let mut h = Harness::new(true, 4, 2);
        h.submit(0, &packets);
        h.kick();
        h.ctx.clear(&mut h.d);
        assert_eq!(read_offset(&h.d), BITS_PER_PACKET_HEADER);
        assert!(!any_input_valid(&h.d));
        assert_eq!(get(&h.d, field::OUTPUT_BUFFER_WRITE_OFFSET), 0);
        assert_eq!(h.ctx.current_frame_remaining_subframes, 0);
    }

    #[test]
    fn merge_keeps_title_writes() {
        let mut initial = [0u32; 16];
        set(&mut initial, field::INPUT_BUFFER_0_VALID, 1);
        set(&mut initial, field::OUTPUT_BUFFER_VALID, 1);
        let mut decoded = initial;
        set(&mut decoded, field::INPUT_BUFFER_0_VALID, 0);
        set(&mut decoded, field::INPUT_BUFFER_READ_OFFSET, 1234);
        set(&mut decoded, field::OUTPUT_BUFFER_WRITE_OFFSET, 5);
        let mut fresh = initial;
        // Meanwhile the title submitted buffer 1 and moved the read offset.
        set(&mut fresh, field::INPUT_BUFFER_1_VALID, 1);
        set(&mut fresh, field::OUTPUT_BUFFER_READ_OFFSET, 3);
        merge_decoder_fields(&mut fresh, &initial, &decoded);
        assert_eq!(get(&fresh, field::INPUT_BUFFER_0_VALID), 0);
        assert_eq!(get(&fresh, field::INPUT_BUFFER_1_VALID), 1);
        assert_eq!(get(&fresh, field::OUTPUT_BUFFER_VALID), 1);
        assert_eq!(get(&fresh, field::INPUT_BUFFER_READ_OFFSET), 1234);
        assert_eq!(get(&fresh, field::OUTPUT_BUFFER_WRITE_OFFSET), 5);
    }

    #[test]
    fn conversion_saturates_and_rounds_to_even() {
        assert_eq!(float_to_s16(1.5), 32767);
        assert_eq!(float_to_s16(-1.5), -32767);
        assert_eq!(float_to_s16(0.5 / 32767.0), 0);
        assert_eq!(float_to_s16(1.5 / 32767.0), 2);
        assert_eq!(float_to_s16(f32::NAN), 0);
    }
}
