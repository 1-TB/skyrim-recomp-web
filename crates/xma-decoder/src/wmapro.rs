// SPDX-License-Identifier: LGPL-2.1-or-later
//
// WMA Pro decoder in XMA mode: a Rust port of FFmpeg's libavcodec/wmaprodec.c
// (decode_init, decode_tilehdr, decode_channel_transform, decode_coeffs,
// decode_scale_factors, inverse_channel_transform, wmapro_window,
// decode_subframe, decode_frame) and of ff_wma_run_level_decode /
// ff_wma_get_large_val from libavcodec/wma.c, specialised to XMA streams
// (decode flags 0x10D6, 512-sample frames, one or two channels) and to the
// single-frame entry point of the "xmaframes" decoder that Xenia's FFmpeg fork
// (github.com/has207/FFmpeg, branch xmaframes) adds to wmaprodec.c.
//
// Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
// Copyright (c) 2008 - 2011 Sascha Sommer, Benjamin Larsson
// Copyright (c) 2002-2007 The FFmpeg Project (wma.c)
// Rust port is part of the xma-decoder crate.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; either version 2.1 of the License, or (at your
// option) any later version. It is distributed WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License for more details.

use std::fmt;
use std::sync::OnceLock;

use crate::bits::{BitReader, copy_bits};
use crate::mdct::{Imdct, fmul_window_in_place, sine_window};
use crate::tables::*;
use crate::vlc::Vlc;

/// PCM samples per channel in one XMA frame.
pub const SAMPLES_PER_FRAME: usize = 512;
/// The longest representable frame, in bits (the 15-bit length field; the
/// all-ones value marks "no frame").
pub const MAX_FRAME_BITS: usize = 0x7FFE;
/// XMA streams carry one or two channels.
pub const MAX_CHANNELS: usize = 2;

const MAX_SUBFRAMES: usize = 32;
const MAX_BANDS: usize = 29;
const LOG2_FRAME_SIZE: u32 = 15;
/// Block sizes 512, 256, 128 (XMA's max_num_subframes of 4).
const BLOCK_SIZES: usize = 3;
const MIN_SAMPLES_PER_SUBFRAME: usize = SAMPLES_PER_FRAME / 4;
/// Integer sample size the IMDCT output is scaled from.
const BITS_PER_SAMPLE: i32 = 16;
/// Length of the per-channel output/overlap buffer.
const OUT_LEN: usize = SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2;
const FRAME_BUF_LEN: usize = MAX_FRAME_BITS.div_ceil(8) + 16;
const VEC1_ESCAPE: i32 = VEC1_TABLE.len() as i32 - 1;

/// Why a frame produced no samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The frame's length field is 0 or too small to hold a frame.
    BadLength(usize),
    /// The tile header (subframe layout) is invalid.
    TileHeader,
    /// A subframe is malformed.
    Subframe(&'static str),
    /// A WMA Pro feature that never occurs in XMA streams.
    Unsupported(&'static str),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadLength(n) => write!(f, "invalid XMA frame length {n}"),
            DecodeError::TileHeader => write!(f, "invalid tile header"),
            DecodeError::Subframe(why) => write!(f, "invalid subframe: {why}"),
            DecodeError::Unsupported(what) => write!(f, "unsupported bitstream feature: {what}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// What a successfully decoded frame said about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameInfo {
    /// Frame length in bits, including the length field and trailer bit.
    pub bits: usize,
    /// Encoder delay / padding samples signalled in the frame header (callers
    /// emulating the hardware ignore these, as Xenia does).
    pub trim_start: u16,
    pub trim_end: u16,
    /// The frame continues in the same packet (the trailer bit).
    pub more_frames: bool,
    /// The parsed bits did not add up to the length field. FFmpeg still
    /// returns the samples of such a frame, and so does this decoder.
    pub length_mismatch: bool,
}

/// Process-wide tables shared by every decoder.
struct Statics {
    sf: Vlc,
    sf_rl: Vlc,
    coef: [Vlc; 2],
    vec4: Vlc,
    vec2: Vlc,
    vec1: Vlc,
    /// Indexed by log2(block) - 7: 128, 256, 512 samples.
    imdct: [Imdct; BLOCK_SIZES],
    windows: [Box<[f32]>; BLOCK_SIZES],
}

fn statics() -> &'static Statics {
    static STATICS: OnceLock<Statics> = OnceLock::new();
    STATICS.get_or_init(|| {
        let pairs = |t: &'static [(u8, u8)]| t.iter().map(|&(sym, len)| (len, sym as i32));
        let imdct = |n: usize| {
            // decode_init: 1 / (1 << (log2(n) - 1)) / (1 << (bits_per_sample - 1)).
            Imdct::new(n, 1.0 / (n / 2) as f64 / (1u64 << (BITS_PER_SAMPLE - 1)) as f64)
        };
        Statics {
            sf: Vlc::from_lengths(pairs(&SCALE_TABLE), -60),
            sf_rl: Vlc::from_lengths(pairs(&SCALE_RL_TABLE), 0),
            coef: [
                Vlc::from_lengths(COEF0_LENS.iter().zip(&COEF0_SYMS).map(|(&l, &s)| (l, s as i32)), 0),
                Vlc::from_lengths(pairs(&COEF1_TABLE), 0),
            ],
            vec4: Vlc::from_lengths(VEC4_LENS.iter().zip(&VEC4_SYMS).map(|(&l, &s)| (l, s as i32)), -1),
            vec2: Vlc::from_lengths(pairs(&VEC2_TABLE), -1),
            vec1: Vlc::from_lengths(pairs(&VEC1_TABLE), 0),
            imdct: [imdct(128), imdct(256), imdct(512)],
            windows: [sine_window(128), sine_window(256), sine_window(512)],
        }
    })
}

#[cfg(test)]
pub(crate) fn all_vlcs() -> Vec<(&'static str, &'static Vlc)> {
    let s = statics();
    vec![
        ("scale", &s.sf),
        ("scale_rl", &s.sf_rl),
        ("coef0", &s.coef[0]),
        ("coef1", &s.coef[1]),
        ("vec4", &s.vec4),
        ("vec2", &s.vec2),
        ("vec1", &s.vec1),
    ]
}

/// av_log2: floor(log2(v)), 0 for 0.
fn log2(v: usize) -> u32 {
    if v == 0 { 0 } else { usize::BITS - 1 - v.leading_zeros() }
}

/// Per-channel state (WMAProChannelCtx).
#[derive(Clone)]
struct Channel {
    prev_block_len: usize,
    transmit_coefs: bool,
    num_subframes: usize,
    subframe_len: [u16; MAX_SUBFRAMES],
    cur_subframe: usize,
    decoded_samples: usize,
    grouped: bool,
    quant_step: i32,
    reuse_sf: bool,
    scale_factor_step: i32,
    max_scale_factor: i32,
    saved_scale_factors: [[i32; MAX_BANDS]; 2],
    scale_factor_idx: usize,
    /// Which `saved_scale_factors` buffer holds the current scale factors.
    scale_factors: usize,
    table_idx: usize,
    /// Offset of the current subframe's coefficients in `out`.
    coeffs: usize,
    num_vec_coeffs: usize,
    /// Output samples of the current frame followed by the overlap half of
    /// the last block.
    out: [f32; OUT_LEN],
}

impl Default for Channel {
    fn default() -> Self {
        Channel {
            prev_block_len: SAMPLES_PER_FRAME,
            transmit_coefs: false,
            num_subframes: 0,
            subframe_len: [0; MAX_SUBFRAMES],
            cur_subframe: 0,
            decoded_samples: 0,
            grouped: false,
            quant_step: 0,
            reuse_sf: false,
            scale_factor_step: 0,
            max_scale_factor: 0,
            saved_scale_factors: [[0; MAX_BANDS]; 2],
            scale_factor_idx: 0,
            scale_factors: 0,
            table_idx: 0,
            coeffs: 0,
            num_vec_coeffs: 0,
            out: [0.0; OUT_LEN],
        }
    }
}

/// Channel group for joint stereo (WMAProChannelGrp, at most two channels).
#[derive(Clone, Copy, Default)]
struct ChannelGroup {
    num_channels: usize,
    transform: bool,
    transform_band: [bool; MAX_BANDS],
    decorrelation_matrix: [f32; 4],
    channels: [usize; MAX_CHANNELS],
}

/// Decodes individual XMA frames of one stream (one or two channels) into
/// 512 float samples per channel, carrying the MDCT overlap between frames.
/// Equivalent to an FFmpeg "xmaframes" decoder context.
pub struct FrameDecoder {
    nb_channels: usize,
    sample_rate: u32,
    num_sfb: [usize; BLOCK_SIZES],
    sfb_offsets: [[u16; MAX_BANDS]; BLOCK_SIZES],
    sf_offsets: [[[u8; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES],

    // Frame/subframe decode state.
    num_saved_bits: usize,
    parsed_all_subframes: bool,
    subframe_len: usize,
    channels_for_cur_subframe: usize,
    channel_indexes_for_cur_subframe: [usize; MAX_CHANNELS],
    num_bands: usize,
    transmit_num_vec_coeffs: bool,
    table_idx: usize,
    esc_len: u32,
    num_chgroups: usize,
    chgroup: [ChannelGroup; MAX_CHANNELS],
    channel: [Channel; MAX_CHANNELS],

    tmp: [f32; SAMPLES_PER_FRAME],
    /// The frame being decoded, realigned to bit 0 and zero padded.
    frame: Vec<u8>,
    pcm: [[f32; SAMPLES_PER_FRAME]; MAX_CHANNELS],
}

impl FrameDecoder {
    /// `sample_rate` in Hz (only used to lay out scale factor bands, after
    /// rounding to the XMA rates 24000/32000/44100/48000); `channels` 1 or 2.
    pub fn new(sample_rate: u32, channels: usize) -> FrameDecoder {
        assert!((1..=MAX_CHANNELS).contains(&channels), "XMA streams have 1 or 2 channels");
        let mut d = FrameDecoder {
            nb_channels: channels,
            sample_rate,
            num_sfb: [0; BLOCK_SIZES],
            sfb_offsets: [[0; MAX_BANDS]; BLOCK_SIZES],
            sf_offsets: [[[0; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES],
            num_saved_bits: 0,
            parsed_all_subframes: false,
            subframe_len: 0,
            channels_for_cur_subframe: 0,
            channel_indexes_for_cur_subframe: [0; MAX_CHANNELS],
            num_bands: 0,
            transmit_num_vec_coeffs: false,
            table_idx: 0,
            esc_len: 0,
            num_chgroups: 0,
            chgroup: [ChannelGroup::default(); MAX_CHANNELS],
            channel: [Channel::default(), Channel::default()],
            tmp: [0.0; SAMPLES_PER_FRAME],
            frame: vec![0; FRAME_BUF_LEN],
            pcm: [[0.0; SAMPLES_PER_FRAME]; MAX_CHANNELS],
        };
        d.init_bands();
        d
    }

    pub fn channels(&self) -> usize {
        self.nb_channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Forgets the overlap carried from previous frames (a new stream).
    pub fn reset(&mut self) {
        for c in &mut self.channel {
            *c = Channel::default();
        }
    }

    /// Samples of the last decoded frame for `channel`.
    pub fn pcm(&self, channel: usize) -> &[f32; SAMPLES_PER_FRAME] {
        &self.pcm[channel]
    }

    /// decode_init: scale factor band layout per block size, and the matrix
    /// that maps bands between block sizes.
    fn init_bands(&mut self) {
        // get_rate() for XMA.
        let rate = match self.sample_rate {
            r if r > 44100 => 48000,
            r if r > 32000 => 44100,
            r if r > 24000 => 32000,
            _ => 24000,
        };
        for i in 0..BLOCK_SIZES {
            let subframe_len = (SAMPLES_PER_FRAME >> i) as i64;
            let sfb = &mut self.sfb_offsets[i];
            let mut band = 1;
            sfb[0] = 0;
            let mut x = 0;
            while x < MAX_BANDS - 1 && (sfb[band - 1] as i64) < subframe_len {
                let offset = ((subframe_len * 2 * CRITICAL_FREQ[x] as i64) / rate + 2) & !3;
                if offset > sfb[band - 1] as i64 {
                    sfb[band] = offset as u16;
                    band += 1;
                }
                if offset >= subframe_len {
                    break;
                }
                x += 1;
            }
            sfb[band - 1] = subframe_len as u16;
            self.num_sfb[i] = band - 1;
        }
        for i in 0..BLOCK_SIZES {
            for b in 0..self.num_sfb[i] {
                let offset = ((self.sfb_offsets[i][b] as i32 + self.sfb_offsets[i][b + 1] as i32 - 1) << i) >> 1;
                for x in 0..BLOCK_SIZES {
                    let mut v = 0;
                    while ((self.sfb_offsets[x][v + 1] as i32) << x) < offset {
                        v += 1;
                    }
                    self.sf_offsets[i][x][b] = v as u8;
                }
            }
        }
    }

    /// Decodes the frame starting at bit `bit_offset` of `data`; its length
    /// comes from its own 15-bit header. Bits past the end of `data` read as
    /// zero. On success the samples are in [`FrameDecoder::pcm`]. On error no
    /// samples are produced, but (as in FFmpeg) the decoder state may have
    /// been partially updated.
    pub fn decode_frame(&mut self, data: &[u8], bit_offset: usize) -> Result<FrameInfo, DecodeError> {
        let mut len_buf = [0u8; 2];
        copy_bits(&mut len_buf, data, bit_offset, LOG2_FRAME_SIZE as usize);
        let len = (u16::from_be_bytes(len_buf) >> 1) as usize;
        // xmaframes_decode_packet rejects packets under 3 bytes (header byte
        // + bit padding + frame); save_bits rejects empty frames.
        if len == 0 || len > MAX_FRAME_BITS || (bit_offset & 7) + len <= 8 {
            return Err(DecodeError::BadLength(len));
        }
        let mut frame = std::mem::take(&mut self.frame);
        frame.fill(0);
        copy_bits(&mut frame, data, bit_offset, len);
        self.num_saved_bits = len;
        let result = self.decode(&frame, len);
        self.frame = frame;
        result
    }

    /// decode_frame() on a frame copied to the start of `buf`.
    fn decode(&mut self, buf: &[u8], len: usize) -> Result<FrameInfo, DecodeError> {
        let s = statics();
        let mut gb = BitReader::new(buf, len);
        let frame_len = gb.bits(LOG2_FRAME_SIZE) as usize;

        self.decode_tilehdr(&mut gb)?;

        // Postproc transform: parsed and ignored.
        if self.nb_channels > 1 && gb.bit() && gb.bit() {
            gb.skip(4 * self.nb_channels * self.nb_channels);
        }
        // Dynamic range compression gain: parsed and ignored.
        gb.bits(8);
        let mut info = FrameInfo { bits: len, ..FrameInfo::default() };
        if gb.bit() {
            if gb.bit() {
                info.trim_start = gb.bits(log2(SAMPLES_PER_FRAME * 2)) as u16;
            }
            if gb.bit() {
                info.trim_end = gb.bits(log2(SAMPLES_PER_FRAME * 2)) as u16;
            }
        }

        self.parsed_all_subframes = false;
        for c in &mut self.channel[..self.nb_channels] {
            c.decoded_samples = 0;
            c.cur_subframe = 0;
            c.reuse_sf = false;
        }
        while !self.parsed_all_subframes {
            self.decode_subframe(&mut gb, s)?;
        }

        for (c, ch) in self.channel[..self.nb_channels].iter_mut().enumerate() {
            self.pcm[c].copy_from_slice(&ch.out[..SAMPLES_PER_FRAME]);
            // Keep the second half of the last IMDCT for the next frame.
            ch.out.copy_within(SAMPLES_PER_FRAME..SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2, 0);
        }

        if frame_len != gb.pos() + 2 {
            info.length_mismatch = true;
        } else {
            gb.skip(frame_len - gb.pos() - 1);
            info.more_frames = gb.bit();
        }
        Ok(info)
    }

    /// decode_subframe_length.
    fn decode_subframe_length(&self, gb: &mut BitReader, offset: usize) -> Result<usize, DecodeError> {
        if offset == SAMPLES_PER_FRAME - MIN_SAMPLES_PER_SUBFRAME {
            return Ok(MIN_SAMPLES_PER_SUBFRAME);
        }
        if gb.left() < 1 {
            return Err(DecodeError::TileHeader);
        }
        // max_subframe_len_bit with subframe_len_bits of 2.
        let shift = if gb.bit() { 1 + gb.bits(1) } else { 0 };
        Ok(SAMPLES_PER_FRAME >> shift)
    }

    /// decode_tilehdr: how the frame is split into subframes per channel.
    fn decode_tilehdr(&mut self, gb: &mut BitReader) -> Result<(), DecodeError> {
        let nb = self.nb_channels;
        let mut num_samples = [0usize; MAX_CHANNELS];
        let mut contains_subframe = [false; MAX_CHANNELS];
        let mut channels_for_cur_subframe = nb;
        let mut min_channel_len = 0;
        for c in &mut self.channel[..nb] {
            c.num_subframes = 0;
        }
        let fixed_channel_layout = gb.bit();
        loop {
            for c in 0..nb {
                contains_subframe[c] = if num_samples[c] == min_channel_len {
                    if fixed_channel_layout
                        || channels_for_cur_subframe == 1
                        || min_channel_len == SAMPLES_PER_FRAME - MIN_SAMPLES_PER_SUBFRAME
                    {
                        true
                    } else {
                        gb.bit()
                    }
                } else {
                    false
                };
            }
            let subframe_len = self.decode_subframe_length(gb, min_channel_len)?;
            min_channel_len += subframe_len;
            for c in 0..nb {
                let chan = &mut self.channel[c];
                if contains_subframe[c] {
                    if chan.num_subframes >= MAX_SUBFRAMES {
                        return Err(DecodeError::TileHeader);
                    }
                    chan.subframe_len[chan.num_subframes] = subframe_len as u16;
                    num_samples[c] += subframe_len;
                    chan.num_subframes += 1;
                    if num_samples[c] > SAMPLES_PER_FRAME {
                        return Err(DecodeError::TileHeader);
                    }
                } else if num_samples[c] <= min_channel_len {
                    if num_samples[c] < min_channel_len {
                        channels_for_cur_subframe = 0;
                        min_channel_len = num_samples[c];
                    }
                    channels_for_cur_subframe += 1;
                }
            }
            if min_channel_len >= SAMPLES_PER_FRAME {
                return Ok(());
            }
        }
    }

    /// decode_channel_transform, for at most two channels.
    fn decode_channel_transform(&mut self, gb: &mut BitReader) -> Result<(), DecodeError> {
        self.num_chgroups = 0;
        if self.nb_channels == 1 {
            return Ok(());
        }
        if gb.bit() {
            return Err(DecodeError::Unsupported("channel transform bit"));
        }
        let mut remaining = self.channels_for_cur_subframe;
        while remaining > 0 && self.num_chgroups < self.channels_for_cur_subframe {
            let mut g = ChannelGroup { num_channels: remaining, ..ChannelGroup::default() };
            // With two channels at most, "remaining > 2" never happens: the
            // group takes every channel not grouped yet.
            let mut n = 0;
            for i in 0..self.channels_for_cur_subframe {
                let idx = self.channel_indexes_for_cur_subframe[i];
                if !self.channel[idx].grouped {
                    g.channels[n] = idx;
                    n += 1;
                }
                self.channel[idx].grouped = true;
            }
            if g.num_channels == 2 {
                if gb.bit() {
                    if gb.bit() {
                        return Err(DecodeError::Unsupported("channel transform type"));
                    }
                } else {
                    g.transform = true;
                    // nb_channels == 2: unnormalized M/S.
                    g.decorrelation_matrix = [1.0, -1.0, 1.0, 1.0];
                }
            }
            if g.transform {
                if !gb.bit() {
                    for t in &mut g.transform_band[..self.num_bands] {
                        *t = gb.bit();
                    }
                } else {
                    g.transform_band[..self.num_bands].fill(true);
                }
            }
            remaining -= g.num_channels;
            self.chgroup[self.num_chgroups] = g;
            self.num_chgroups += 1;
        }
        Ok(())
    }

    /// decode_coeffs: vector-coded coefficients, then run/level coding.
    fn decode_coeffs(&mut self, gb: &mut BitReader, s: &Statics, c: usize) {
        let vlctable = gb.bit() as usize;
        let (run, level): (&[u16], &[f32]) =
            if vlctable == 1 { (&COEF1_RUN, &COEF1_LEVEL) } else { (&COEF0_RUN, &COEF0_LEVEL) };
        let subframe_len = self.subframe_len;
        let transmit_num_vec_coeffs = self.transmit_num_vec_coeffs;
        let ci = &mut self.channel[c];
        let coeffs = &mut ci.out[ci.coeffs..ci.coeffs + subframe_len];
        let mut rl_mode = false;
        let mut cur_coeff = 0;
        let mut num_zeros = 0;

        while (transmit_num_vec_coeffs || !rl_mode) && cur_coeff + 3 < ci.num_vec_coeffs {
            let mut vals = [0u32; 4];
            let idx = s.vec4.read(gb);
            if idx < 0 {
                for i in (0..4).step_by(2) {
                    let idx = s.vec2.read(gb);
                    if idx < 0 {
                        let mut v0 = s.vec1.read(gb) as u32;
                        if v0 as i32 == VEC1_ESCAPE {
                            v0 = v0.wrapping_add(get_large_val(gb));
                        }
                        let mut v1 = s.vec1.read(gb) as u32;
                        if v1 as i32 == VEC1_ESCAPE {
                            v1 = v1.wrapping_add(get_large_val(gb));
                        }
                        vals[i] = (v0 as f32).to_bits();
                        vals[i + 1] = (v1 as f32).to_bits();
                    } else {
                        vals[i] = ((idx >> 4) as f32).to_bits();
                        vals[i + 1] = ((idx & 0xF) as f32).to_bits();
                    }
                }
            } else {
                vals = [idx >> 12, (idx >> 8) & 0xF, (idx >> 4) & 0xF, idx & 0xF].map(|v| (v as f32).to_bits());
            }
            for v in vals {
                if v != 0 {
                    // A set sign bit means positive.
                    let negative = !gb.bit();
                    coeffs[cur_coeff] = f32::from_bits(v ^ ((negative as u32) << 31));
                    num_zeros = 0;
                } else {
                    coeffs[cur_coeff] = 0.0;
                    num_zeros += 1;
                    rl_mode |= num_zeros > subframe_len >> 8;
                }
                cur_coeff += 1;
            }
        }

        if cur_coeff < subframe_len {
            coeffs[cur_coeff..].fill(0.0);
            // ff_wma_run_level_decode (version 1); its error return is
            // ignored by the caller.
            let vlc = &s.coef[vlctable];
            let mask = subframe_len - 1;
            let mut offset = cur_coeff;
            while offset < subframe_len {
                let code = vlc.read(gb);
                if code > 1 {
                    offset += run[code as usize] as usize;
                    let negative = !gb.bit();
                    coeffs[offset & mask] = f32::from_bits(level[code as usize].to_bits() ^ ((negative as u32) << 31));
                } else if code == 1 {
                    break;
                } else {
                    let level = get_large_val(gb) as i32;
                    if gb.bit() {
                        if gb.bit() {
                            if gb.bit() {
                                break; // Broken escape sequence.
                            }
                            offset += gb.bits(self.esc_len) as usize + 4;
                        } else {
                            offset += gb.bits(2) as usize + 1;
                        }
                    }
                    let sign = gb.bit() as i32 - 1;
                    coeffs[offset & mask] = ((level ^ sign).wrapping_sub(sign)) as f32;
                }
                offset += 1;
            }
        }
    }

    /// decode_scale_factors.
    fn decode_scale_factors(&mut self, gb: &mut BitReader, s: &Statics) -> Result<(), DecodeError> {
        let num_bands = self.num_bands;
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let sf_offsets = self.sf_offsets[self.table_idx][self.channel[c].table_idx];
            let ch = &mut self.channel[c];
            let cur = 1 - ch.scale_factor_idx;
            ch.scale_factors = cur;
            if ch.reuse_sf {
                let saved = ch.saved_scale_factors[ch.scale_factor_idx];
                for b in 0..num_bands {
                    ch.saved_scale_factors[cur][b] = saved[sf_offsets[b] as usize];
                }
            }
            if ch.cur_subframe == 0 || gb.bit() {
                let sf = &mut ch.saved_scale_factors[cur];
                if !ch.reuse_sf {
                    // DPCM coded.
                    ch.scale_factor_step = gb.bits(2) as i32 + 1;
                    let mut val = 45 / ch.scale_factor_step;
                    for v in &mut sf[..num_bands] {
                        val += s.sf.read(gb);
                        *v = val;
                    }
                } else {
                    // Run/level coded differences to the resampled factors.
                    let mut i = 0;
                    while i < num_bands {
                        let idx = s.sf_rl.read(gb);
                        let (skip, val, sign);
                        if idx == 0 {
                            let code = gb.bits(14);
                            val = (code >> 6) as i32;
                            sign = (code & 1) as i32 - 1;
                            skip = ((code & 0x3F) >> 1) as usize;
                        } else if idx == 1 {
                            break;
                        } else {
                            let idx = idx as usize;
                            skip = SCALE_RL_RUN[idx] as usize;
                            val = SCALE_RL_LEVEL[idx] as i32;
                            sign = gb.bit() as i32 - 1;
                        }
                        i += skip;
                        if i >= num_bands {
                            return Err(DecodeError::Subframe("scale factor run"));
                        }
                        sf[i] += (val ^ sign) - sign;
                        i += 1;
                    }
                }
                ch.scale_factor_idx = cur;
                ch.table_idx = self.table_idx;
                ch.reuse_sf = true;
            }
            let sf = &ch.saved_scale_factors[ch.scale_factors][..num_bands];
            ch.max_scale_factor = sf.iter().copied().max().unwrap_or(0);
        }
        Ok(())
    }

    /// inverse_channel_transform.
    fn inverse_channel_transform(&mut self) {
        let sfb = self.sfb_offsets[self.table_idx];
        for g in &self.chgroup[..self.num_chgroups] {
            if !g.transform {
                continue;
            }
            // Transforms only exist for two-channel groups here.
            let (c0, c1) = (g.channels[0], g.channels[1]);
            let (a, b) = self.channel.split_at_mut(1);
            let (ch0, ch1) = if c0 == 0 { (&mut a[0], &mut b[0]) } else { (&mut b[0], &mut a[0]) };
            debug_assert!(c0 != c1);
            let (o0, o1) = (ch0.coeffs, ch1.coeffs);
            let m = g.decorrelation_matrix;
            for band in 0..self.num_bands {
                let start = sfb[band] as usize;
                let end = (sfb[band + 1] as usize).min(self.subframe_len);
                let (d0, d1) = (&mut ch0.out[o0 + start..o0 + end], &mut ch1.out[o1 + start..o1 + end]);
                if g.transform_band[band] {
                    for (x, y) in d0.iter_mut().zip(d1.iter_mut()) {
                        let (v0, v1) = (*x, *y);
                        *x = 0.0 + v0 * m[0] + v1 * m[1];
                        *y = 0.0 + v0 * m[2] + v1 * m[3];
                    }
                } else {
                    // nb_channels == 2: untransformed bands get sqrt(2).
                    const SCALE: f32 = 181.0 / 128.0;
                    d0.iter_mut().for_each(|v| *v *= SCALE);
                    d1.iter_mut().for_each(|v| *v *= SCALE);
                }
            }
        }
    }

    /// wmapro_window: overlap-add with the previous block.
    fn window(&mut self, s: &Statics) {
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let ch = &mut self.channel[c];
            let mut winlen = ch.prev_block_len;
            let mut start = ch.coeffs - (winlen >> 1);
            if self.subframe_len < winlen {
                start += (winlen - self.subframe_len) >> 1;
                winlen = self.subframe_len;
            }
            let window = &s.windows[log2(winlen) as usize - 7];
            fmul_window_in_place(&mut ch.out[start..start + winlen], window, winlen >> 1);
            ch.prev_block_len = self.subframe_len;
        }
    }

    /// decode_subframe: one block of one or both channels.
    fn decode_subframe(&mut self, gb: &mut BitReader, s: &Statics) -> Result<(), DecodeError> {
        let nb = self.nb_channels;
        let mut offset = SAMPLES_PER_FRAME;
        let mut subframe_len = SAMPLES_PER_FRAME;
        let mut total_samples = SAMPLES_PER_FRAME * nb;

        // The next block is that of the channel with the fewest decoded samples.
        for ch in &mut self.channel[..nb] {
            ch.grouped = false;
            if offset > ch.decoded_samples {
                offset = ch.decoded_samples;
                subframe_len = ch.subframe_len[ch.cur_subframe] as usize;
            }
        }
        self.channels_for_cur_subframe = 0;
        for (i, ch) in self.channel[..nb].iter_mut().enumerate() {
            total_samples -= ch.decoded_samples;
            let len = ch.subframe_len[ch.cur_subframe] as usize;
            if offset == ch.decoded_samples && subframe_len == len {
                total_samples -= len;
                ch.decoded_samples += len;
                self.channel_indexes_for_cur_subframe[self.channels_for_cur_subframe] = i;
                self.channels_for_cur_subframe += 1;
            }
        }
        if total_samples == 0 {
            self.parsed_all_subframes = true;
        }

        self.table_idx = log2(SAMPLES_PER_FRAME / subframe_len) as usize;
        self.num_bands = self.num_sfb[self.table_idx];
        let offset = offset + SAMPLES_PER_FRAME / 2;
        for i in 0..self.channels_for_cur_subframe {
            self.channel[self.channel_indexes_for_cur_subframe[i]].coeffs = offset;
        }
        self.subframe_len = subframe_len;
        self.esc_len = log2(subframe_len - 1) + 1;

        // Extended header (fill bits).
        if gb.bit() {
            let mut num_fill_bits = gb.bits(2) as usize;
            if num_fill_bits == 0 {
                let len = gb.bits(4);
                num_fill_bits = gb.bits(len) as usize + 1;
            }
            if gb.pos() + num_fill_bits > self.num_saved_bits {
                return Err(DecodeError::Subframe("fill bits"));
            }
            gb.skip(num_fill_bits);
        }
        if gb.bit() {
            return Err(DecodeError::Unsupported("reserved bit"));
        }

        self.decode_channel_transform(gb)?;

        let mut transmit_coeffs = false;
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            self.channel[c].transmit_coefs = gb.bit();
            transmit_coeffs |= self.channel[c].transmit_coefs;
        }

        if transmit_coeffs {
            self.transmit_num_vec_coeffs = gb.bit();
            if self.transmit_num_vec_coeffs {
                let num_bits = log2(subframe_len.div_ceil(4)) + 1;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    let num_vec_coeffs = (gb.bits(num_bits) as usize) << 2;
                    if num_vec_coeffs > subframe_len {
                        return Err(DecodeError::Subframe("num_vec_coeffs"));
                    }
                    self.channel[c].num_vec_coeffs = num_vec_coeffs;
                }
            } else {
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].num_vec_coeffs = subframe_len;
                }
            }

            let mut quant_step = (90 * BITS_PER_SAMPLE) >> 4;
            let mut step = gb.sbits(6);
            quant_step += step;
            if step == -32 || step == 31 {
                let sign = (step == 31) as i32 - 1;
                let mut quant = 0;
                while gb.pos() + 5 < self.num_saved_bits {
                    step = gb.bits(5) as i32;
                    if step != 31 {
                        break;
                    }
                    quant += 31;
                }
                quant_step += ((quant + step) ^ sign) - sign;
            }

            if self.channels_for_cur_subframe == 1 {
                self.channel[self.channel_indexes_for_cur_subframe[0]].quant_step = quant_step;
            } else {
                let modifier_len = gb.bits(3);
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].quant_step = quant_step;
                    if gb.bit() {
                        self.channel[c].quant_step +=
                            if modifier_len != 0 { gb.bits(modifier_len) as i32 + 1 } else { 1 };
                    }
                }
            }
            self.decode_scale_factors(gb, s)?;
        }

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].transmit_coefs && gb.pos() < self.num_saved_bits {
                self.decode_coeffs(gb, s, c);
            } else {
                let ch = &mut self.channel[c];
                ch.out[ch.coeffs..ch.coeffs + subframe_len].fill(0.0);
            }
        }

        if transmit_coeffs {
            let imdct = &s.imdct[log2(subframe_len) as usize - 7];
            self.inverse_channel_transform();
            let sfb = self.sfb_offsets[self.table_idx];
            for i in 0..self.channels_for_cur_subframe {
                let c = self.channel_indexes_for_cur_subframe[i];
                let ch = &mut self.channel[c];
                let sf = &ch.saved_scale_factors[ch.scale_factors];
                for b in 0..self.num_bands {
                    let start = sfb[b] as usize;
                    let end = (sfb[b + 1] as usize).min(subframe_len);
                    let exp = ch.quant_step - (ch.max_scale_factor - sf[b]) * ch.scale_factor_step;
                    let quant = exp10(exp as f64 / 20.0) as f32;
                    let src = &ch.out[ch.coeffs + start..ch.coeffs + end];
                    for (t, v) in self.tmp[start..end].iter_mut().zip(src) {
                        *t = v * quant;
                    }
                }
                imdct.transform(&self.tmp[..subframe_len], &mut ch.out[ch.coeffs..ch.coeffs + subframe_len]);
            }
        }

        self.window(s);

        for i in 0..self.channels_for_cur_subframe {
            let ch = &mut self.channel[self.channel_indexes_for_cur_subframe[i]];
            if ch.cur_subframe >= ch.num_subframes {
                return Err(DecodeError::Subframe("subframe count"));
            }
            ch.cur_subframe += 1;
        }
        Ok(())
    }
}

/// ff_exp10.
fn exp10(x: f64) -> f64 {
    (std::f64::consts::LOG2_10 * x).exp2()
}

/// ff_wma_get_large_val: an 8, 16, 24 or 31-bit value.
fn get_large_val(gb: &mut BitReader) -> u32 {
    let mut n_bits = 8;
    if gb.bit() {
        n_bits += 8;
        if gb.bit() {
            n_bits += 8;
            if gb.bit() {
                n_bits += 7;
            }
        }
    }
    gb.bits(n_bits)
}
