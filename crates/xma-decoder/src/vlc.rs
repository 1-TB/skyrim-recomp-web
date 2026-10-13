// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Variable length code tables built the way FFmpeg's
// ff_vlc_init_from_lengths() (libavcodec/vlc.c) assigns codes: canonically,
// in table order, each code following the previous one.
//
// Copyright (c) 2000, 2001 Fabrice Bellard
// Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>
// Copyright (c) 2010 Loren Merritt
// Rust implementation is part of the xma-decoder crate.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; either version 2.1 of the License, or (at your
// option) any later version. It is distributed WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License for more details.

use crate::bits::BitReader;

/// Bits resolved by the first table lookup.
const PRIMARY_BITS: u32 = 10;

#[derive(Clone, Copy)]
struct Primary {
    /// Code length when the prefix resolves a whole code, else 0.
    len: u8,
    sym: i16,
    /// Otherwise, the candidate codes to search (`lo > hi`: no valid code).
    lo: u16,
    hi: u16,
}

/// Because codes are assigned in increasing order, the codes in table order
/// tile the 32-bit code space: a code word is found by locating the last code
/// start at or below the next 32 input bits.
pub(crate) struct Vlc {
    primary: Box<[Primary]>,
    /// Left-aligned code values, ascending.
    codes: Box<[u32]>,
    lens: Box<[u8]>,
    syms: Box<[i16]>,
}

impl Vlc {
    /// Builds a table from `(code length, symbol)` pairs in code order; the
    /// decoded value is `symbol + offset`.
    pub(crate) fn from_lengths(entries: impl IntoIterator<Item = (u8, i32)>, offset: i32) -> Vlc {
        let mut code: u64 = 0;
        let (mut codes, mut lens, mut syms) = (Vec::new(), Vec::new(), Vec::new());
        for (len, sym) in entries {
            assert!((1..=32).contains(&len), "VLC code length {len}");
            assert!(code & ((1u64 << (32 - len)) - 1) == 0, "VLC code misaligned");
            codes.push(code as u32);
            lens.push(len);
            syms.push((sym + offset) as i16);
            code += 1u64 << (32 - len);
            assert!(code <= 1u64 << 32, "overdetermined VLC");
        }
        let mut vlc = Vlc {
            primary: Box::new([]),
            codes: codes.into_boxed_slice(),
            lens: lens.into_boxed_slice(),
            syms: syms.into_boxed_slice(),
        };
        let shift = 32 - PRIMARY_BITS;
        vlc.primary = (0..1u32 << PRIMARY_BITS)
            .map(|p| {
                let first = p << shift;
                let last = first | ((1 << shift) - 1);
                match (vlc.find(first), vlc.find(last)) {
                    (Some(a), _) if vlc.lens[a] as u32 <= PRIMARY_BITS => {
                        Primary { len: vlc.lens[a], sym: vlc.syms[a], lo: 1, hi: 0 }
                    }
                    (a, b) => {
                        // Codes inside the prefix: from the one covering its start
                        // (or the first starting inside it) to the one covering its end.
                        let lo = a.unwrap_or_else(|| vlc.codes.partition_point(|&c| c < first));
                        let hi = b.unwrap_or_else(|| vlc.codes.partition_point(|&c| c <= last).wrapping_sub(1));
                        Primary { len: 0, sym: -1, lo: lo as u16, hi: hi.min(u16::MAX as usize) as u16 }
                    }
                }
            })
            .collect();
        vlc
    }

    /// Index of the code that `word` (32 left-aligned input bits) starts with.
    fn find(&self, word: u32) -> Option<usize> {
        let i = self.codes.partition_point(|&c| c <= word).checked_sub(1)?;
        let span = 1u64 << (32 - self.lens[i]);
        (((word - self.codes[i]) as u64) < span).then_some(i)
    }

    /// Decodes one symbol; an invalid code yields -1 without consuming input,
    /// as FFmpeg's get_vlc2() does for a missing table entry.
    #[inline]
    pub(crate) fn read(&self, r: &mut BitReader) -> i32 {
        let word = r.peek32();
        let p = self.primary[(word >> (32 - PRIMARY_BITS)) as usize];
        if p.len != 0 {
            r.skip(p.len as usize);
            return p.sym as i32;
        }
        if p.lo > p.hi {
            return -1;
        }
        let (lo, hi) = (p.lo as usize, p.hi as usize);
        let i = lo + self.codes[lo..=hi].partition_point(|&c| c <= word);
        let Some(i) = i.checked_sub(1).filter(|&i| i >= lo) else { return -1 };
        if ((word - self.codes[i]) as u64) >= 1u64 << (32 - self.lens[i]) {
            return -1;
        }
        r.skip(self.lens[i] as usize);
        self.syms[i] as i32
    }

    #[cfg(test)]
    pub(crate) fn codes(&self) -> impl Iterator<Item = (u32, u8, i16)> + '_ {
        (0..self.codes.len()).map(|i| (self.codes[i] >> (32 - self.lens[i]), self.lens[i], self.syms[i]))
    }

    #[cfg(test)]
    pub(crate) fn kraft_complete(&self) -> bool {
        let total: u64 = self.lens.iter().map(|&l| 1u64 << (32 - l)).sum();
        total == 1u64 << 32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(bits: &mut Vec<bool>, code: u32, len: u8) {
        for i in (0..len).rev() {
            bits.push((code >> i) & 1 != 0);
        }
    }

    fn pack(bits: &[bool]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8) + 8];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                out[i >> 3] |= 0x80 >> (i & 7);
            }
        }
        out
    }

    /// Every table round-trips every symbol, including codes longer than the
    /// primary lookup, and the tables are complete prefix codes.
    #[test]
    fn all_tables_round_trip() {
        for (name, vlc) in crate::wmapro::all_vlcs() {
            let mut bits = Vec::new();
            let codes: Vec<_> = vlc.codes().collect();
            for &(code, len, _) in &codes {
                encode(&mut bits, code, len);
            }
            let data = pack(&bits);
            let mut r = BitReader::new(&data, bits.len());
            for &(_, len, sym) in &codes {
                let before = r.pos();
                assert_eq!(vlc.read(&mut r), sym as i32, "{name}");
                assert_eq!(r.pos() - before, len as usize, "{name}");
            }
            assert!(vlc.kraft_complete(), "{name} is not a complete code");
        }
    }

    #[test]
    fn incomplete_code_rejects_gaps() {
        // Codes: 0 (len 1), 10 (len 2); "11" is unassigned.
        let vlc = Vlc::from_lengths([(1, 5), (2, 7)], 0);
        let data = [0b0101_1000, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut r = BitReader::new(&data, 8);
        assert_eq!(vlc.read(&mut r), 5);
        assert_eq!(vlc.read(&mut r), 7);
        assert_eq!(vlc.read(&mut r), -1);
        assert_eq!(r.pos(), 3);
    }
}
