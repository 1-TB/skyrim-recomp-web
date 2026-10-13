//! Final step of a resolve on the CPU: linear converted pixels (as produced by the resolve shader)
//! to the tiled guest destination with its endian swap.

use xenos_texture::Endian128;
use xenos_texture::address::{tiled_2d, tiled_3d};

use crate::GuestMemoryAccess;

/// Where and how converted pixels go in guest memory.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DestLayout {
    pub base: u32,
    pub pitch_aligned: u32,
    pub height_aligned: u32,
    pub is_3d: bool,
    pub slice: u32,
    /// Offset of the copy origin from `base`, in pixels.
    pub offset_x: u32,
    pub offset_y: u32,
    /// Bytes per pixel (per 4x4 block for `block_1111`).
    pub bytes_per_element: u32,
    pub endian: Endian128,
    pub extent_start: u32,
    pub extent_len: u32,
    /// k_DXT3A_AS_1_1_1_1: one 4-bit word per pixel, assembled into 8-byte 4x4 blocks.
    pub block_1111: bool,
}

/// Applies the destination endian swap to one element as the Xenos resolve stores it: 8bpp is
/// never swapped, 16bpp only by 8in16, 64/128bpp also by 8in64/8in128.
pub(crate) fn swap_element(bytes: &mut [u8], endian: Endian128) {
    let swap32 = |b: &mut [u8]| match endian {
        Endian128::K8in16 | Endian128::K8in32 | Endian128::K16in32 => endian.swap_in_place(b),
        _ => {}
    };
    match bytes.len() {
        2 => {
            if endian == Endian128::K8in16 {
                bytes.swap(0, 1);
            }
        }
        4 => swap32(bytes),
        8 | 16 => match endian {
            Endian128::K8in64 => Endian128::K8in64.swap_in_place(bytes),
            Endian128::K8in128 if bytes.len() == 16 => bytes.reverse(),
            Endian128::K8in128 => {}
            _ => swap32(bytes),
        },
        _ => {}
    }
}

impl DestLayout {
    fn element_offset(&self, x: u32, y: u32) -> Option<usize> {
        let bpp_log2 = self.bytes_per_element.ilog2();
        let rel = if self.is_3d {
            tiled_3d(x as i32, y as i32, self.slice as i32, self.pitch_aligned, self.height_aligned, bpp_log2)
        } else {
            tiled_2d(x as i32, y as i32, self.pitch_aligned, bpp_log2) as i64
        };
        let address = (self.base as i64).checked_add(rel)?;
        let offset = address.checked_sub(self.extent_start as i64)?;
        (offset >= 0 && offset + self.bytes_per_element as i64 <= self.extent_len as i64).then_some(offset as usize)
    }

    /// Writes `width` x `height` pixels of `words` (`words_per_pixel` u32 each, row-major) into guest
    /// memory: reads the extent, patches it and writes it back. Returns the number of elements that
    /// fell outside the extent (0 normally).
    pub fn write(
        &self,
        words: &[u32],
        words_per_pixel: usize,
        width: u32,
        height: u32,
        memory: &dyn GuestMemoryAccess,
    ) -> usize {
        let mut extent = vec![0u8; self.extent_len as usize];
        memory.read(self.extent_start, &mut extent);
        let mut dropped = 0;
        let bpe = self.bytes_per_element as usize;
        let mut element = [0u8; 16];
        let mut put = |x: u32, y: u32, element: &mut [u8]| {
            swap_element(element, self.endian);
            match self.element_offset(x, y) {
                Some(o) => extent[o..o + element.len()].copy_from_slice(element),
                None => dropped += 1,
            }
        };
        if self.block_1111 {
            let word = |x: u32, y: u32| words[(y * width + x) as usize] & 0xF;
            for by in 0..height / 4 {
                for bx in 0..width / 4 {
                    let row = |r: u32| (0..4).fold(0u32, |acc, i| acc | (word(bx * 4 + i, by * 4 + r) << (4 * i)));
                    let lo = row(0) | (row(1) << 16);
                    let hi = row(2) | (row(3) << 16);
                    element[..4].copy_from_slice(&lo.to_le_bytes());
                    element[4..8].copy_from_slice(&hi.to_le_bytes());
                    put(bx + self.offset_x / 4, by + self.offset_y / 4, &mut element[..8]);
                }
            }
        } else {
            for y in 0..height {
                for x in 0..width {
                    let i = (y * width + x) as usize * words_per_pixel;
                    for (w, chunk) in words[i..i + words_per_pixel].iter().zip(element.as_chunks_mut::<4>().0) {
                        chunk.copy_from_slice(&w.to_le_bytes());
                    }
                    put(x + self.offset_x, y + self.offset_y, &mut element[..bpe]);
                }
            }
        }
        memory.write(self.extent_start, &extent);
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_swaps() {
        let mut b = [1u8, 2];
        swap_element(&mut b, Endian128::K8in32);
        assert_eq!(b, [1, 2]);
        swap_element(&mut b, Endian128::K8in16);
        assert_eq!(b, [2, 1]);
        let mut b = [1u8, 2, 3, 4];
        swap_element(&mut b, Endian128::K8in64);
        assert_eq!(b, [1, 2, 3, 4]);
        swap_element(&mut b, Endian128::K16in32);
        assert_eq!(b, [3, 4, 1, 2]);
        let mut b: Vec<u8> = (0..8).collect();
        swap_element(&mut b, Endian128::K8in64);
        assert_eq!(b, [7, 6, 5, 4, 3, 2, 1, 0]);
        let mut b: Vec<u8> = (0..8).collect();
        swap_element(&mut b, Endian128::K8in32);
        assert_eq!(b, [3, 2, 1, 0, 7, 6, 5, 4]);
        let mut b: Vec<u8> = (0..16).collect();
        swap_element(&mut b, Endian128::K8in128);
        assert_eq!(b, (0..16).rev().collect::<Vec<u8>>());
        let mut b = [9u8];
        swap_element(&mut b, Endian128::K8in32);
        assert_eq!(b, [9]);
    }
}
