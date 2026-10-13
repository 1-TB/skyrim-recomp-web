//! Xenos texture formats (`a2xx_sq_surfaceformat`) and their storage properties.

macro_rules! texture_formats {
    ($($name:ident = $value:literal, $kind:ident, $bw:literal, $bh:literal, $bpp:literal, [$c0:literal, $c1:literal, $c2:literal, $c3:literal];)*) => {
        /// The 6-bit texture format field of a texture fetch constant.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum TextureFormat {
            $($name = $value,)*
        }

        impl TextureFormat {
            pub const ALL: [TextureFormat; 64] = [$(TextureFormat::$name,)*];

            pub fn from_bits(bits: u32) -> Self {
                Self::ALL[(bits & 63) as usize]
            }

            pub fn name(self) -> &'static str {
                match self {
                    $(TextureFormat::$name => stringify!($name),)*
                }
            }

            pub fn info(self) -> FormatInfo {
                match self {
                    $(TextureFormat::$name => FormatInfo {
                        kind: FormatKind::$kind,
                        block_width: $bw,
                        block_height: $bh,
                        bits_per_pixel: $bpp,
                        component_bits: [$c0, $c1, $c2, $c3],
                    },)*
                }
            }
        }
    };
}

texture_formats! {
    k_1_REVERSE = 0, Uncompressed, 1, 1, 1, [0, 0, 0, 0];
    k_1 = 1, Uncompressed, 1, 1, 1, [0, 0, 0, 0];
    k_8 = 2, Resolvable, 1, 1, 8, [8, 0, 0, 0];
    k_1_5_5_5 = 3, Resolvable, 1, 1, 16, [5, 5, 5, 1];
    k_5_6_5 = 4, Resolvable, 1, 1, 16, [5, 6, 5, 0];
    k_6_5_5 = 5, Resolvable, 1, 1, 16, [5, 5, 6, 0];
    k_8_8_8_8 = 6, Resolvable, 1, 1, 32, [8, 8, 8, 8];
    k_2_10_10_10 = 7, Resolvable, 1, 1, 32, [10, 10, 10, 2];
    k_8_A = 8, Resolvable, 1, 1, 8, [8, 0, 0, 0];
    k_8_B = 9, Resolvable, 1, 1, 8, [8, 0, 0, 0];
    k_8_8 = 10, Resolvable, 1, 1, 16, [8, 8, 0, 0];
    k_Cr_Y1_Cb_Y0_REP = 11, Compressed, 2, 1, 16, [0, 0, 0, 0];
    k_Y1_Cr_Y0_Cb_REP = 12, Compressed, 2, 1, 16, [0, 0, 0, 0];
    k_16_16_EDRAM = 13, Uncompressed, 1, 1, 32, [16, 16, 0, 0];
    k_8_8_8_8_A = 14, Resolvable, 1, 1, 32, [8, 8, 8, 8];
    k_4_4_4_4 = 15, Resolvable, 1, 1, 16, [4, 4, 4, 4];
    k_10_11_11 = 16, Resolvable, 1, 1, 32, [11, 11, 10, 0];
    k_11_11_10 = 17, Resolvable, 1, 1, 32, [10, 11, 11, 0];
    k_DXT1 = 18, Compressed, 4, 4, 4, [0, 0, 0, 0];
    k_DXT2_3 = 19, Compressed, 4, 4, 8, [0, 0, 0, 0];
    k_DXT4_5 = 20, Compressed, 4, 4, 8, [0, 0, 0, 0];
    k_16_16_16_16_EDRAM = 21, Uncompressed, 1, 1, 64, [16, 16, 16, 16];
    k_24_8 = 22, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
    k_24_8_FLOAT = 23, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
    k_16 = 24, Resolvable, 1, 1, 16, [16, 0, 0, 0];
    k_16_16 = 25, Resolvable, 1, 1, 32, [16, 16, 0, 0];
    k_16_16_16_16 = 26, Resolvable, 1, 1, 64, [16, 16, 16, 16];
    k_16_EXPAND = 27, Uncompressed, 1, 1, 16, [0, 0, 0, 0];
    k_16_16_EXPAND = 28, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
    k_16_16_16_16_EXPAND = 29, Uncompressed, 1, 1, 64, [0, 0, 0, 0];
    k_16_FLOAT = 30, Resolvable, 1, 1, 16, [0, 0, 0, 0];
    k_16_16_FLOAT = 31, Resolvable, 1, 1, 32, [0, 0, 0, 0];
    k_16_16_16_16_FLOAT = 32, Resolvable, 1, 1, 64, [0, 0, 0, 0];
    k_32 = 33, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
    k_32_32 = 34, Uncompressed, 1, 1, 64, [0, 0, 0, 0];
    k_32_32_32_32 = 35, Uncompressed, 1, 1, 128, [0, 0, 0, 0];
    k_32_FLOAT = 36, Resolvable, 1, 1, 32, [0, 0, 0, 0];
    k_32_32_FLOAT = 37, Resolvable, 1, 1, 64, [0, 0, 0, 0];
    k_32_32_32_32_FLOAT = 38, Resolvable, 1, 1, 128, [0, 0, 0, 0];
    k_32_AS_8 = 39, Compressed, 4, 1, 8, [0, 0, 0, 0];
    k_32_AS_8_8 = 40, Compressed, 2, 1, 16, [0, 0, 0, 0];
    k_16_MPEG = 41, Uncompressed, 1, 1, 16, [0, 0, 0, 0];
    k_16_16_MPEG = 42, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
    k_8_INTERLACED = 43, Uncompressed, 1, 1, 8, [0, 0, 0, 0];
    k_32_AS_8_INTERLACED = 44, Compressed, 4, 1, 8, [0, 0, 0, 0];
    k_32_AS_8_8_INTERLACED = 45, Compressed, 1, 1, 16, [0, 0, 0, 0];
    k_16_INTERLACED = 46, Uncompressed, 1, 1, 16, [0, 0, 0, 0];
    k_16_MPEG_INTERLACED = 47, Uncompressed, 1, 1, 16, [0, 0, 0, 0];
    k_16_16_MPEG_INTERLACED = 48, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
    k_DXN = 49, Compressed, 4, 4, 8, [8, 8, 0, 0];
    k_8_8_8_8_AS_16_16_16_16 = 50, Resolvable, 1, 1, 32, [8, 8, 8, 8];
    k_DXT1_AS_16_16_16_16 = 51, Compressed, 4, 4, 4, [0, 0, 0, 0];
    k_DXT2_3_AS_16_16_16_16 = 52, Compressed, 4, 4, 8, [0, 0, 0, 0];
    k_DXT4_5_AS_16_16_16_16 = 53, Compressed, 4, 4, 8, [0, 0, 0, 0];
    k_2_10_10_10_AS_16_16_16_16 = 54, Resolvable, 1, 1, 32, [10, 10, 10, 2];
    k_10_11_11_AS_16_16_16_16 = 55, Resolvable, 1, 1, 32, [11, 11, 10, 0];
    k_11_11_10_AS_16_16_16_16 = 56, Resolvable, 1, 1, 32, [10, 11, 11, 0];
    k_32_32_32_FLOAT = 57, Uncompressed, 1, 1, 96, [0, 0, 0, 0];
    k_DXT3A = 58, Compressed, 4, 4, 4, [8, 0, 0, 0];
    k_DXT5A = 59, Compressed, 4, 4, 4, [8, 0, 0, 0];
    k_CTX1 = 60, Compressed, 4, 4, 4, [8, 8, 0, 0];
    k_DXT3A_AS_1_1_1_1 = 61, Compressed, 4, 4, 4, [0, 0, 0, 0];
    k_8_8_8_8_GAMMA_EDRAM = 62, Uncompressed, 1, 1, 32, [8, 8, 8, 8];
    k_2_10_10_10_FLOAT_EDRAM = 63, Uncompressed, 1, 1, 32, [0, 0, 0, 0];
}

/// Storage class of a format (Xenia's `FormatType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatKind {
    /// Uncompressed and also a render-target resolve destination format.
    Resolvable,
    Uncompressed,
    /// Stored in multi-texel blocks.
    Compressed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    pub kind: FormatKind,
    pub block_width: u32,
    pub block_height: u32,
    pub bits_per_pixel: u32,
    /// Bits of the x/y/z/w components for fixed-point formats, 0 where not applicable.
    pub component_bits: [u8; 4],
}

impl FormatInfo {
    pub fn bits_per_block(&self) -> u32 {
        self.block_width * self.block_height * self.bits_per_pixel
    }

    /// Zero for the 1bpp formats, whose blocks are smaller than a byte.
    pub fn bytes_per_block(&self) -> u32 {
        self.bits_per_block() / 8
    }

    /// Floor of log2, as Direct3D 9 / the tiler compute it (12-byte blocks use 3).
    pub fn bytes_per_block_log2(&self) -> u32 {
        let b = self.bytes_per_block();
        if b == 0 { 0 } else { 31 - b.leading_zeros() }
    }

    pub fn width_shift(&self) -> u32 {
        self.block_width.trailing_zeros()
    }

    pub fn height_shift(&self) -> u32 {
        self.block_height.trailing_zeros()
    }

    pub fn is_compressed(&self) -> bool {
        self.kind == FormatKind::Compressed
    }
}

impl TextureFormat {
    /// Formats that only differ in filtering precision or resolve behavior share storage with
    /// a base format (Xenia's `GetBaseFormat`).
    pub fn base_format(self) -> TextureFormat {
        use TextureFormat::*;
        match self {
            k_16_EXPAND => k_16_FLOAT,
            k_16_16_EXPAND => k_16_16_FLOAT,
            k_16_16_16_16_EXPAND => k_16_16_16_16_FLOAT,
            k_8_8_8_8_AS_16_16_16_16 => k_8_8_8_8,
            k_DXT1_AS_16_16_16_16 => k_DXT1,
            k_DXT2_3_AS_16_16_16_16 => k_DXT2_3,
            k_DXT4_5_AS_16_16_16_16 => k_DXT4_5,
            k_2_10_10_10_AS_16_16_16_16 => k_2_10_10_10,
            k_10_11_11_AS_16_16_16_16 => k_10_11_11,
            k_11_11_10_AS_16_16_16_16 => k_11_11_10,
            k_8_8_8_8_GAMMA_EDRAM => k_8_8_8_8,
            other => other,
        }
    }

    pub fn block_width(self) -> u32 {
        self.info().block_width
    }

    pub fn block_height(self) -> u32 {
        self.info().block_height
    }

    pub fn bytes_per_block(self) -> u32 {
        self.info().bytes_per_block()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        for (i, f) in TextureFormat::ALL.iter().enumerate() {
            assert_eq!(*f as usize, i);
            assert_eq!(TextureFormat::from_bits(i as u32), *f);
        }
        assert_eq!(TextureFormat::from_bits(64 + 18), TextureFormat::k_DXT1);
    }

    #[test]
    fn block_sizes() {
        use TextureFormat::*;
        let cases = [
            (k_8, 1, 1, 1),
            (k_5_6_5, 1, 1, 2),
            (k_8_8_8_8, 1, 1, 4),
            (k_16_16_16_16_FLOAT, 1, 1, 8),
            (k_32_32_32_32_FLOAT, 1, 1, 16),
            (k_32_32_32_FLOAT, 1, 1, 12),
            (k_DXT1, 4, 4, 8),
            (k_DXT2_3, 4, 4, 16),
            (k_DXT4_5, 4, 4, 16),
            (k_DXN, 4, 4, 16),
            (k_DXT3A, 4, 4, 8),
            (k_DXT5A, 4, 4, 8),
            (k_CTX1, 4, 4, 8),
            (k_Y1_Cr_Y0_Cb_REP, 2, 1, 4),
            (k_32_AS_8, 4, 1, 4),
            (k_1, 1, 1, 0),
        ];
        for (f, bw, bh, bytes) in cases {
            let info = f.info();
            assert_eq!((info.block_width, info.block_height, info.bytes_per_block()), (bw, bh, bytes), "{f:?}");
        }
        assert_eq!(k_32_32_32_FLOAT.info().bytes_per_block_log2(), 3);
        assert_eq!(k_DXT2_3.info().bytes_per_block_log2(), 4);
        assert_eq!(k_DXT1.info().width_shift(), 2);
        assert_eq!(k_DXT1_AS_16_16_16_16.base_format(), k_DXT1);
        assert_eq!(k_DXT1.name(), "k_DXT1");
    }
}
