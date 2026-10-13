//! Xenos endianness swaps as applied to texture and vertex data in guest memory.
//!
//! Guest memory is consumed as raw bytes in guest order. Swaps operate on aligned 32-bit
//! groups (or 64/128-bit groups for [`Endian128`]); after swapping, multi-byte values are
//! read as little-endian, matching how the GPU interprets the data.

/// 2-bit endianness used by texture and vertex fetch constants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Endian {
    #[default]
    None = 0,
    K8in16 = 1,
    K8in32 = 2,
    K16in32 = 3,
}

impl Endian {
    pub fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Endian::None,
            1 => Endian::K8in16,
            2 => Endian::K8in32,
            _ => Endian::K16in32,
        }
    }

    /// Swaps a 32-bit value loaded little-endian from guest-order bytes.
    pub fn swap_u32(self, value: u32) -> u32 {
        match self {
            Endian::None => value,
            Endian::K8in16 => ((value << 8) & 0xFF00_FF00) | ((value >> 8) & 0x00FF_00FF),
            Endian::K8in32 => value.swap_bytes(),
            Endian::K16in32 => value.rotate_left(16),
        }
    }

    /// Swaps a 16-bit value; only `K8in16` affects a lone half word.
    pub fn swap_u16(self, value: u16) -> u16 {
        match self {
            Endian::K8in16 => value.swap_bytes(),
            _ => value,
        }
    }

    /// Swaps guest-order bytes in place in aligned 4-byte groups. A trailing partial group is
    /// swapped as far as the mode allows (pairs for `K8in16`) and left alone otherwise.
    pub fn swap_in_place(self, data: &mut [u8]) {
        match self {
            Endian::None => {}
            Endian::K8in16 => {
                for pair in data.as_chunks_mut::<2>().0 {
                    pair.swap(0, 1);
                }
            }
            Endian::K8in32 => {
                for word in data.as_chunks_mut::<4>().0 {
                    word.reverse();
                }
            }
            Endian::K16in32 => {
                for word in data.as_chunks_mut::<4>().0 {
                    word.swap(0, 2);
                    word.swap(1, 3);
                }
            }
        }
    }
}

/// 3-bit endianness used by memexport / resolve destinations (adds 64- and 128-bit swaps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Endian128 {
    #[default]
    None = 0,
    K8in16 = 1,
    K8in32 = 2,
    K16in32 = 3,
    K8in64 = 4,
    K8in128 = 5,
}

impl Endian128 {
    /// Values 6 and 7 are undefined and treated as no swap.
    pub fn from_bits(bits: u32) -> Self {
        match bits & 7 {
            1 => Endian128::K8in16,
            2 => Endian128::K8in32,
            3 => Endian128::K16in32,
            4 => Endian128::K8in64,
            5 => Endian128::K8in128,
            _ => Endian128::None,
        }
    }

    pub fn swap_in_place(self, data: &mut [u8]) {
        match self {
            Endian128::None => {}
            Endian128::K8in16 => Endian::K8in16.swap_in_place(data),
            Endian128::K8in32 => Endian::K8in32.swap_in_place(data),
            Endian128::K16in32 => Endian::K16in32.swap_in_place(data),
            Endian128::K8in64 => {
                for group in data.as_chunks_mut::<8>().0 {
                    group.reverse();
                }
            }
            Endian128::K8in128 => {
                for group in data.as_chunks_mut::<16>().0 {
                    group.reverse();
                }
            }
        }
    }
}

impl From<Endian> for Endian128 {
    fn from(e: Endian) -> Self {
        Endian128::from_bits(e as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_u32_modes() {
        assert_eq!(Endian::None.swap_u32(0x1122_3344), 0x1122_3344);
        assert_eq!(Endian::K8in16.swap_u32(0x1122_3344), 0x2211_4433);
        assert_eq!(Endian::K8in32.swap_u32(0x1122_3344), 0x4433_2211);
        assert_eq!(Endian::K16in32.swap_u32(0x1122_3344), 0x3344_1122);
        assert_eq!(Endian::K8in16.swap_u16(0x1122), 0x2211);
        assert_eq!(Endian::K8in32.swap_u16(0x1122), 0x1122);
    }

    #[test]
    fn swap_in_place_matches_word_swap() {
        let bytes: Vec<u8> = (0u8..16).collect();
        for e in [Endian::None, Endian::K8in16, Endian::K8in32, Endian::K16in32] {
            let mut swapped = bytes.clone();
            e.swap_in_place(&mut swapped);
            for i in 0..4 {
                let word = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                let got = u32::from_le_bytes(swapped[i * 4..i * 4 + 4].try_into().unwrap());
                assert_eq!(got, e.swap_u32(word), "{e:?}");
            }
        }
        let mut b = [0u8, 1, 2, 3];
        Endian::K8in32.swap_in_place(&mut b);
        assert_eq!(b, [3, 2, 1, 0]);
        let mut b = [0u8, 1, 2, 3];
        Endian::K16in32.swap_in_place(&mut b);
        assert_eq!(b, [2, 3, 0, 1]);
        let mut b = [0u8, 1, 2, 3, 4, 5];
        Endian::K8in16.swap_in_place(&mut b);
        assert_eq!(b, [1, 0, 3, 2, 5, 4]);
    }

    #[test]
    fn swap_128_modes() {
        let mut b: Vec<u8> = (0u8..16).collect();
        Endian128::K8in64.swap_in_place(&mut b);
        assert_eq!(b, [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]);
        let mut b: Vec<u8> = (0u8..16).collect();
        Endian128::K8in128.swap_in_place(&mut b);
        assert_eq!(b, (0u8..16).rev().collect::<Vec<_>>());
        assert_eq!(Endian128::from(Endian::K16in32), Endian128::K16in32);
        assert_eq!(Endian128::from_bits(7), Endian128::None);
    }
}
