//! CPU decoders for Xenos block-compressed formats (after endian swapping the data is laid
//! out like the PC BC formats). Interpolation follows Xenia's load shaders (truncating
//! integer division, 5:6:5 endpoints expanded by bit replication).

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn expand_565(c: u16) -> [u32; 3] {
    let r = u32::from(c >> 11) & 31;
    let g = u32::from(c >> 5) & 63;
    let b = u32::from(c) & 31;
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// Decodes the 8-byte color part of a BC1/2/3 block into 16 RGBA8 texels (row-major).
/// `allow_punchthrough` selects BC1 semantics (3-color + transparent when c0 <= c1).
fn decode_color(block: &[u8], allow_punchthrough: bool, out: &mut [[u8; 4]; 16]) {
    let c0 = u16_at(block, 0);
    let c1 = u16_at(block, 2);
    let e0 = expand_565(c0);
    let e1 = expand_565(c1);
    let mut palette = [[0u8; 4]; 4];
    for ch in 0..3 {
        palette[0][ch] = e0[ch] as u8;
        palette[1][ch] = e1[ch] as u8;
    }
    palette[0][3] = 255;
    palette[1][3] = 255;
    if !allow_punchthrough || c0 > c1 {
        for ch in 0..3 {
            palette[2][ch] = ((2 * e0[ch] + e1[ch]) / 3) as u8;
            palette[3][ch] = ((e0[ch] + 2 * e1[ch]) / 3) as u8;
        }
        palette[2][3] = 255;
        palette[3][3] = 255;
    } else {
        for ch in 0..3 {
            palette[2][ch] = ((e0[ch] + e1[ch]) >> 1) as u8;
        }
        palette[2][3] = 255;
        palette[3] = [0, 0, 0, 0];
    }
    let indices = u32_at(block, 4);
    for (i, texel) in out.iter_mut().enumerate() {
        *texel = palette[((indices >> (2 * i)) & 3) as usize];
    }
}

/// BC1 / DXT1: 8 bytes -> 16 RGBA8 texels.
pub fn decode_bc1(block: &[u8], out: &mut [[u8; 4]; 16]) {
    decode_color(block, true, out);
}

/// 16 4-bit values (BC2 alpha half / DXT3A), row-major, low nibble first.
fn nibbles(block: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, n) in out.iter_mut().enumerate() {
        *n = (block[i / 2] >> (4 * (i % 2))) & 0xF;
    }
    out
}

/// BC2 / DXT2_3: 16 bytes -> 16 RGBA8 texels.
pub fn decode_bc2(block: &[u8], out: &mut [[u8; 4]; 16]) {
    decode_color(&block[8..16], false, out);
    for (texel, a) in out.iter_mut().zip(nibbles(&block[0..8])) {
        texel[3] = a * 17;
    }
}

/// BC3 / DXT4_5: 16 bytes -> 16 RGBA8 texels.
pub fn decode_bc3(block: &[u8], out: &mut [[u8; 4]; 16]) {
    decode_color(&block[8..16], false, out);
    let mut alpha = [0u8; 16];
    decode_bc4_unorm(&block[0..8], &mut alpha);
    for (texel, a) in out.iter_mut().zip(alpha) {
        texel[3] = a;
    }
}

fn alpha_codes(block: &[u8]) -> u64 {
    let mut v = 0u64;
    for i in 0..6 {
        v |= u64::from(block[2 + i]) << (8 * i);
    }
    v
}

/// BC4 unsigned / DXT5A / BC3 alpha: 8 bytes -> 16 values.
pub fn decode_bc4_unorm(block: &[u8], out: &mut [u8; 16]) {
    let a0 = u32::from(block[0]);
    let a1 = u32::from(block[1]);
    let codes = alpha_codes(block);
    for (i, v) in out.iter_mut().enumerate() {
        let code = ((codes >> (3 * i)) & 7) as u32;
        *v = match code {
            0 => a0,
            1 => a1,
            k if a0 > a1 => ((8 - k) * a0 + (k - 1) * a1) / 7,
            6 => 0,
            7 => 255,
            k => ((6 - k) * a0 + (k - 1) * a1) / 5,
        } as u8;
    }
}

/// BC4 signed (signed DXT5A / DXN): 8 bytes -> 16 two's complement values in [-127, 127].
pub fn decode_bc4_snorm(block: &[u8], out: &mut [i8; 16]) {
    let a0 = i32::from((block[0] as i8).max(-127));
    let a1 = i32::from((block[1] as i8).max(-127));
    let codes = alpha_codes(block);
    let div_round = |n: i32, d: i32| if n >= 0 { (n + d / 2) / d } else { (n - d / 2) / d };
    for (i, v) in out.iter_mut().enumerate() {
        let code = ((codes >> (3 * i)) & 7) as i32;
        *v = match code {
            0 => a0,
            1 => a1,
            k if a0 > a1 => div_round((8 - k) * a0 + (k - 1) * a1, 7),
            6 => -127,
            7 => 127,
            k => div_round((6 - k) * a0 + (k - 1) * a1, 5),
        } as i8;
    }
}

/// BC5 / DXN: 16 bytes -> 16 RG8 texels.
pub fn decode_bc5_unorm(block: &[u8], out: &mut [[u8; 2]; 16]) {
    let mut r = [0u8; 16];
    let mut g = [0u8; 16];
    decode_bc4_unorm(&block[0..8], &mut r);
    decode_bc4_unorm(&block[8..16], &mut g);
    for i in 0..16 {
        out[i] = [r[i], g[i]];
    }
}

/// Signed BC5 / DXN: 16 bytes -> 16 RG8 snorm texels.
pub fn decode_bc5_snorm(block: &[u8], out: &mut [[i8; 2]; 16]) {
    let mut r = [0i8; 16];
    let mut g = [0i8; 16];
    decode_bc4_snorm(&block[0..8], &mut r);
    decode_bc4_snorm(&block[8..16], &mut g);
    for i in 0..16 {
        out[i] = [r[i], g[i]];
    }
}

/// DXT3A: 8 bytes of 4-bit values -> 16 R8 values.
pub fn decode_dxt3a(block: &[u8], out: &mut [u8; 16]) {
    for (o, n) in out.iter_mut().zip(nibbles(block)) {
        *o = n * 17;
    }
}

/// DXT3A_AS_1_1_1_1: bits 0..3 of each 4-bit value are R, G, B, A (each 0 or 255).
pub fn decode_dxt3a_as_1111(block: &[u8], out: &mut [[u8; 4]; 16]) {
    for (texel, n) in out.iter_mut().zip(nibbles(block)) {
        for (bit, c) in texel.iter_mut().enumerate() {
            *c = if n >> bit & 1 != 0 { 255 } else { 0 };
        }
    }
}

/// CTX1: 8 bytes -> 16 RG8 texels. Endpoints are 8:8 (red in the high byte of each 16-bit
/// endpoint), always interpolated in 4-color mode.
pub fn decode_ctx1_unorm(block: &[u8], out: &mut [[u8; 2]; 16]) {
    let d = u32_at(block, 0);
    let e0 = [(d >> 8) & 0xFF, d & 0xFF];
    let e1 = [d >> 24, (d >> 16) & 0xFF];
    let indices = u32_at(block, 4);
    for (i, texel) in out.iter_mut().enumerate() {
        let code = (indices >> (2 * i)) & 3;
        for ch in 0..2 {
            texel[ch] = match code {
                0 => e0[ch],
                1 => e1[ch],
                2 => (2 * e0[ch] + e1[ch]) / 3,
                _ => (e0[ch] + 2 * e1[ch]) / 3,
            } as u8;
        }
    }
}

/// Signed CTX1: endpoint bytes are two's complement.
pub fn decode_ctx1_snorm(block: &[u8], out: &mut [[i8; 2]; 16]) {
    let d = u32_at(block, 0);
    let s = |v: u32| i32::from((v as u8 as i8).max(-127));
    let e0 = [s(d >> 8), s(d)];
    let e1 = [s(d >> 24), s(d >> 16)];
    let indices = u32_at(block, 4);
    for (i, texel) in out.iter_mut().enumerate() {
        let code = (indices >> (2 * i)) & 3;
        for ch in 0..2 {
            texel[ch] = match code {
                0 => e0[ch],
                1 => e1[ch],
                2 => (2 * e0[ch] + e1[ch]) / 3,
                _ => (e0[ch] + 2 * e1[ch]) / 3,
            } as i8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_four_color() {
        // c0 = pure red (0xF800), c1 = pure blue (0x001F), indices 0,1,2,3 in row 0.
        let block = [0x00, 0xF8, 0x1F, 0x00, 0b1110_0100, 0, 0, 0];
        let mut out = [[0u8; 4]; 16];
        decode_bc1(&block, &mut out);
        assert_eq!(out[0], [255, 0, 0, 255]);
        assert_eq!(out[1], [0, 0, 255, 255]);
        assert_eq!(out[2], [170, 0, 85, 255]);
        assert_eq!(out[3], [85, 0, 170, 255]);
        assert_eq!(out[4], [255, 0, 0, 255]);
    }

    #[test]
    fn bc1_punchthrough() {
        // c0 = blue < c1 = red -> 3-color mode with transparent index 3.
        let block = [0x1F, 0x00, 0x00, 0xF8, 0b1110_0100, 0, 0, 0];
        let mut out = [[0u8; 4]; 16];
        decode_bc1(&block, &mut out);
        assert_eq!(out[0], [0, 0, 255, 255]);
        assert_eq!(out[1], [255, 0, 0, 255]);
        assert_eq!(out[2], [127, 0, 127, 255]);
        assert_eq!(out[3], [0, 0, 0, 0]);
        // The same endpoints in BC2 stay in 4-color mode.
        let mut bc2 = [0u8; 16];
        bc2[8..].copy_from_slice(&block);
        bc2[0] = 0xF0;
        decode_bc2(&bc2, &mut out);
        assert_eq!(out[3], [170, 0, 85, 0]);
        assert_eq!(out[0][3], 0);
        assert_eq!(out[1][3], 255);
    }

    #[test]
    fn bc1_endpoint_expansion() {
        // Mid green 0x07E0 -> (0, 255, 0); 0x0841 -> r=1,g=2,b=1 -> (8, 8, 8).
        let block = [0xE0, 0x07, 0x41, 0x08, 0b0100, 0, 0, 0];
        let mut out = [[0u8; 4]; 16];
        decode_bc1(&block, &mut out);
        assert_eq!(out[0], [0, 255, 0, 255]);
        assert_eq!(out[1], [8, 8, 8, 255]);
    }

    #[test]
    fn bc4_modes() {
        // a0=255 > a1=0: codes 0..7 for texels 0..7.
        let codes: u64 = (0..8u64).fold(0, |acc, i| acc | (i << (3 * i)));
        let mut block = [255u8, 0, 0, 0, 0, 0, 0, 0];
        for i in 0..6 {
            block[2 + i] = (codes >> (8 * i)) as u8;
        }
        let mut out = [0u8; 16];
        decode_bc4_unorm(&block, &mut out);
        assert_eq!(&out[..8], &[255, 0, 218, 182, 145, 109, 72, 36]);
        // a0=0 <= a1=255: 6-step with constants.
        block[0] = 0;
        block[1] = 255;
        decode_bc4_unorm(&block, &mut out);
        assert_eq!(&out[..8], &[0, 255, 51, 102, 153, 204, 0, 255]);
        let mut s = [0i8; 16];
        block[0] = 0x81; // -127
        block[1] = 0x7F;
        decode_bc4_snorm(&block, &mut s);
        assert_eq!(&s[..8], &[-127, 127, -76, -25, 25, 76, -127, 127]);
        block[0] = 0x7F;
        block[1] = 0x80; // -128 clamps to -127
        decode_bc4_snorm(&block, &mut s);
        assert_eq!(&s[..3], &[127, -127, 91]);
    }

    #[test]
    fn bc3_and_bc5() {
        let mut block = [0u8; 16];
        block[0] = 200;
        block[1] = 100;
        block[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes());
        let mut out = [[0u8; 4]; 16];
        decode_bc3(&block, &mut out);
        assert_eq!(out[5], [255, 255, 255, 200]);
        let mut rg = [[0u8; 2]; 16];
        block[8] = 7;
        block[9] = 9;
        decode_bc5_unorm(&block, &mut rg);
        assert_eq!(rg[0], [200, 7]);
        let mut srg = [[0i8; 2]; 16];
        decode_bc5_snorm(&block, &mut srg);
        assert_eq!(srg[0], [-56, 7]);
    }

    #[test]
    fn dxt3a_variants() {
        let block = [0x10, 0xF2, 0, 0, 0, 0, 0, 0x8F];
        let mut out = [0u8; 16];
        decode_dxt3a(&block, &mut out);
        assert_eq!(&out[..4], &[0, 17, 34, 255]);
        assert_eq!(&out[14..], &[255, 136]);
        let mut texels = [[0u8; 4]; 16];
        decode_dxt3a_as_1111(&block, &mut texels);
        assert_eq!(texels[1], [255, 0, 0, 0]);
        assert_eq!(texels[2], [0, 255, 0, 0]);
        assert_eq!(texels[15], [0, 0, 0, 255]);
        assert_eq!(texels[14], [255; 4]);
    }

    #[test]
    fn ctx1_endpoints() {
        // e0 = (R 0x10, G 0x20), e1 = (R 0xF0, G 0x80); indices 0,1,2,3.
        let d: u32 = 0x20 | (0x10 << 8) | (0x80 << 16) | (0xF0 << 24);
        let mut block = [0u8; 8];
        block[..4].copy_from_slice(&d.to_le_bytes());
        block[4] = 0b1110_0100;
        let mut out = [[0u8; 2]; 16];
        decode_ctx1_unorm(&block, &mut out);
        assert_eq!(out[0], [0x10, 0x20]);
        assert_eq!(out[1], [0xF0, 0x80]);
        assert_eq!(out[2], [90, 64]);
        assert_eq!(out[3], [165, 96]);
        let mut s = [[0i8; 2]; 16];
        decode_ctx1_snorm(&block, &mut s);
        assert_eq!(s[1], [-16, -127]);
        assert_eq!(s[0], [16, 32]);
    }
}
