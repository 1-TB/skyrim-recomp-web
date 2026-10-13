// SPDX-License-Identifier: LGPL-2.1-or-later
//
// MSB-first bit reader with the overread behavior of FFmpeg's checked
// GetBitContext (libavcodec/get_bits.h, Copyright (c) 2004 Michael
// Niedermayer): the position saturates 8 bits past the end. Bits past the end
// read as zero.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; either version 2.1 of the License, or (at your
// option) any later version. It is distributed WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License for more details.

/// Reads bits from `buf`, which should extend at least 8 bytes past
/// `size_bits` for the fast path; shorter buffers are zero-extended.
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    size_bits: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(buf: &'a [u8], size_bits: usize) -> Self {
        BitReader { buf, pos: 0, size_bits }
    }

    /// Bits consumed so far.
    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// Bits left before the end (negative after an overread).
    #[inline]
    pub(crate) fn left(&self) -> isize {
        self.size_bits as isize - self.pos as isize
    }

    /// The next 32 bits, MSB first, without consuming them.
    #[inline]
    pub(crate) fn peek32(&self) -> u32 {
        let byte = self.pos >> 3;
        let word = match self.buf.get(byte..byte + 8) {
            Some(b) => u64::from_be_bytes(b.try_into().unwrap()),
            None => {
                let mut b = [0u8; 8];
                if byte < self.buf.len() {
                    let n = (self.buf.len() - byte).min(8);
                    b[..n].copy_from_slice(&self.buf[byte..byte + n]);
                }
                u64::from_be_bytes(b)
            }
        };
        ((word << (self.pos & 7)) >> 32) as u32
    }

    #[inline]
    pub(crate) fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.size_bits + 8);
    }

    /// Reads `n` (0..=32) bits as an unsigned value.
    #[inline]
    pub(crate) fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = self.peek32() >> (32 - n);
        self.skip(n as usize);
        v
    }

    #[inline]
    pub(crate) fn bit(&mut self) -> bool {
        self.bits(1) != 0
    }

    /// Reads `n` (1..=32) bits as a two's complement value.
    #[inline]
    pub(crate) fn sbits(&mut self, n: u32) -> i32 {
        let v = self.peek32() as i32 >> (32 - n);
        self.skip(n as usize);
        v
    }
}

/// Copies `len` bits starting at bit `offset` of `src` to the start of `dst`,
/// zero-filling bits that lie beyond `src`. `dst` must hold `len` bits.
pub(crate) fn copy_bits(dst: &mut [u8], src: &[u8], offset: usize, len: usize) {
    let bytes = len.div_ceil(8);
    let first = offset >> 3;
    let shift = (offset & 7) as u32;
    let byte_at = |i: usize| src.get(i).copied().unwrap_or(0);
    for (i, d) in dst[..bytes].iter_mut().enumerate() {
        let hi = byte_at(first + i);
        *d = if shift == 0 { hi } else { (hi << shift) | (byte_at(first + i + 1) >> (8 - shift)) };
    }
    if len & 7 != 0 {
        dst[bytes - 1] &= 0xFF << (8 - (len & 7));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first_across_bytes() {
        let data = [0b1010_1100, 0b0101_0011, 0xFF, 0x00];
        let mut r = BitReader::new(&data, 32);
        assert_eq!(r.bits(1), 1);
        assert_eq!(r.bits(3), 0b010);
        assert_eq!(r.bits(8), 0b1100_0101);
        assert_eq!(r.sbits(4), 0b0011);
        assert_eq!(r.sbits(4), -1);
        assert_eq!(r.pos(), 20);
        assert_eq!(r.bits(12), 0xF00);
        assert_eq!(r.left(), 0);
    }

    #[test]
    fn overread_saturates_and_reads_zero() {
        let data = [0xFF];
        let mut r = BitReader::new(&data, 8);
        assert_eq!(r.bits(8), 0xFF);
        assert_eq!(r.bits(32), 0);
        assert_eq!(r.pos(), 16);
        assert_eq!(r.left(), -8);
    }

    #[test]
    fn long_reads() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A];
        let mut r = BitReader::new(&data, 40);
        r.skip(4);
        assert_eq!(r.bits(32), 0x2345_6789);
        assert_eq!(r.bits(4), 0xA);
    }

    #[test]
    fn copy_bits_realigns_and_masks() {
        let src = [0b0001_1111, 0b1010_1010, 0b1100_0000];
        let mut dst = [0xEEu8; 3];
        copy_bits(&mut dst, &src, 3, 13);
        assert_eq!(dst[0], 0b1111_1101);
        assert_eq!(dst[1], 0b0101_0000);
        copy_bits(&mut dst, &src, 20, 12);
        assert_eq!(&dst[..2], &[0, 0]);
    }
}
