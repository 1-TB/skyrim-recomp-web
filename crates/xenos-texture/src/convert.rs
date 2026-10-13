//! Per-block conversion of endian-swapped guest data into host texel data.

use crate::bc;
use crate::host::Conversion;

/// IEEE 754 binary32 to binary16 with round-to-nearest-even.
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// IEEE 754 binary16 to binary32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from(h >> 10) & 31;
    let man = u32::from(h) & 0x3FF;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            let lz = man.leading_zeros() - 21;
            let man = (man << lz) & 0x3FF;
            sign | ((113 - lz) << 23) | (man << 13)
        }
        31 => sign | 0x7F80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Xenos 7e3 float (EDRAM 2_10_10_10_FLOAT color) to f32.
pub fn f7e3_to_f32(v: u32) -> f32 {
    let v = v & 0x3FF;
    if v == 0 {
        return 0.0;
    }
    let mut mantissa = v & 0x7F;
    let mut exponent = v >> 7;
    if exponent == 0 {
        let lz = 7 - (31 - mantissa.leading_zeros());
        exponent = 1u32.wrapping_sub(lz);
        mantissa = (mantissa << lz) & 0x7F;
    }
    f32::from_bits((exponent.wrapping_add(124) << 23) | (mantissa << 16))
}

/// Xenos 20e4 float (k_24_8_FLOAT depth, 24 bits) to f32.
pub fn f20e4_to_f32(v: u32) -> f32 {
    let v = v & 0xFF_FFFF;
    if v == 0 {
        return 0.0;
    }
    let mut mantissa = v & 0xF_FFFF;
    let mut exponent = v >> 20;
    if exponent == 0 {
        let lz = 20 - (31 - mantissa.leading_zeros());
        exponent = 1u32.wrapping_sub(lz);
        mantissa = (mantissa << lz) & 0xF_FFFF;
    }
    f32::from_bits((exponent.wrapping_add(112) << 23) | (mantissa << 3))
}

/// 24-bit unorm depth to f32, exact for the common 0xC00000-style values.
pub fn unorm24_to_f32(n: u32) -> f32 {
    let n = n & 0xFF_FFFF;
    (n + (n >> 23)) as f32 * (1.0 / 16_777_216.0)
}

fn sign_extend(raw: u32, bits: u32) -> i32 {
    ((raw << (32 - bits)) as i32) >> (32 - bits)
}

/// Signed normalized value in [-1, 1] (most negative code clamps to -1).
fn snorm_value(raw: u32, bits: u32) -> f32 {
    let s = sign_extend(raw, bits);
    let max = (1i64 << (bits - 1)) - 1;
    if max == 0 {
        return if s < 0 { -1.0 } else { 0.0 };
    }
    (s as f64 / max as f64).max(-1.0) as f32
}

fn unorm_value(raw: u32, bits: u32) -> f32 {
    let max = ((1u64 << bits) - 1) as f64;
    (raw as f64 / max) as f32
}

fn unorm_rescale(raw: u32, bits: u32, to_bits: u32) -> u32 {
    let max = (1u64 << bits) - 1;
    let to_max = (1u64 << to_bits) - 1;
    ((u64::from(raw) * to_max + max / 2) / max) as u32
}

fn snorm_rescale(raw: u32, bits: u32, to_max: f32) -> i32 {
    (snorm_value(raw, bits) * to_max).round() as i32
}

fn le_u16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn le_u32(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn packed_value(src: &[u8]) -> u32 {
    match src.len() {
        2 => u32::from(le_u16(src, 0)),
        _ => le_u32(src, 0),
    }
}

/// Converts one guest block (`src`, `bytes_per_block` long) into host data. For `Copy` the
/// output is the block itself; otherwise it is the block's texels in row-major order.
pub fn convert_block(conversion: Conversion, src: &[u8], out: &mut [u8]) {
    use Conversion as C;
    match conversion {
        C::Copy => out[..src.len()].copy_from_slice(src),
        C::PackedToRgba8 { fields, count, signed } => {
            let v = packed_value(src);
            for c in 0..4 {
                out[c] = if c < count as usize {
                    let (shift, bits) = (u32::from(fields[c].0), u32::from(fields[c].1));
                    let raw = (v >> shift) & ((1u32 << bits) - 1);
                    if signed { snorm_rescale(raw, bits, 127.0) as i8 as u8 } else { unorm_rescale(raw, bits, 8) as u8 }
                } else if signed {
                    127
                } else {
                    255
                };
            }
        }
        C::PackedToRgba16 { fields, count, signed, float } => {
            let v = packed_value(src);
            for c in 0..4 {
                let word: u16 = if c < count as usize {
                    let (shift, bits) = (u32::from(fields[c].0), u32::from(fields[c].1));
                    let raw = (v >> shift) & ((1u32 << bits) - 1);
                    match (signed, float) {
                        (false, false) => unorm_rescale(raw, bits, 16) as u16,
                        (true, false) => snorm_rescale(raw, bits, 32767.0) as i16 as u16,
                        (false, true) => f32_to_f16(unorm_value(raw, bits)),
                        (true, true) => f32_to_f16(snorm_value(raw, bits)),
                    }
                } else {
                    match (signed, float) {
                        (false, false) => 0xFFFF,
                        (true, false) => 0x7FFF,
                        (_, true) => 0x3C00,
                    }
                };
                out[c * 2..c * 2 + 2].copy_from_slice(&word.to_le_bytes());
            }
        }
        C::Norm16ToFloat16 { components, signed } => {
            for c in 0..components as usize {
                let raw = u32::from(le_u16(src, c * 2));
                let f = if signed { snorm_value(raw, 16) } else { unorm_value(raw, 16) };
                out[c * 2..c * 2 + 2].copy_from_slice(&f32_to_f16(f).to_le_bytes());
            }
        }
        C::Norm32ToFloat32 { components, signed } => {
            for c in 0..components as usize {
                let raw = le_u32(src, c * 4);
                let f = if signed {
                    ((raw as i32) as f64 / i32::MAX as f64).max(-1.0) as f32
                } else {
                    (raw as f64 / u32::MAX as f64) as f32
                };
                out[c * 4..c * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
        C::Edram16ToFloat16 { components } => {
            for c in 0..components as usize {
                let s = le_u16(src, c * 2) as i16;
                let f = (f32::from(s) * (32.0 / 32767.0)).max(-32.0);
                out[c * 2..c * 2 + 2].copy_from_slice(&f32_to_f16(f).to_le_bytes());
            }
        }
        C::Float7e3ToRgba16Float => {
            let v = le_u32(src, 0);
            for c in 0..3 {
                let h = f32_to_f16(f7e3_to_f32(v >> (10 * c)));
                out[c * 2..c * 2 + 2].copy_from_slice(&h.to_le_bytes());
            }
            out[6..8].copy_from_slice(&f32_to_f16((v >> 30) as f32 / 3.0).to_le_bytes());
        }
        C::Rgb32FloatToRgba32Float => {
            out[..12].copy_from_slice(&src[..12]);
            out[12..16].copy_from_slice(&1.0f32.to_le_bytes());
        }
        C::Depth24UnormToFloat32 => {
            out[..4].copy_from_slice(&unorm24_to_f32(le_u32(src, 0) >> 8).to_le_bytes());
        }
        C::Depth20e4ToFloat32 => {
            out[..4].copy_from_slice(&f20e4_to_f32(le_u32(src, 0) >> 8).to_le_bytes());
        }
        C::YuvGbgrToRgba8 => {
            let b = [src[0], src[1], src[2], src[3]];
            out[..8].copy_from_slice(&[b[3], b[0], b[1], b[1], b[3], b[2], b[1], b[1]]);
        }
        C::YuvBgrgToRgba8 => {
            let b = [src[0], src[1], src[2], src[3]];
            out[..8].copy_from_slice(&[b[2], b[1], b[0], b[0], b[2], b[3], b[0], b[0]]);
        }
        C::Bc1ToRgba8 | C::Bc2ToRgba8 | C::Bc3ToRgba8 | C::Dxt3aAs1111ToRgba8 => {
            let mut texels = [[0u8; 4]; 16];
            match conversion {
                C::Bc1ToRgba8 => bc::decode_bc1(src, &mut texels),
                C::Bc2ToRgba8 => bc::decode_bc2(src, &mut texels),
                C::Bc3ToRgba8 => bc::decode_bc3(src, &mut texels),
                _ => bc::decode_dxt3a_as_1111(src, &mut texels),
            }
            for (i, t) in texels.iter().enumerate() {
                out[i * 4..i * 4 + 4].copy_from_slice(t);
            }
        }
        C::Bc4ToR8 { signed } => {
            if signed {
                let mut v = [0i8; 16];
                bc::decode_bc4_snorm(src, &mut v);
                for (o, s) in out.iter_mut().zip(v) {
                    *o = s as u8;
                }
            } else {
                let mut v = [0u8; 16];
                bc::decode_bc4_unorm(src, &mut v);
                out[..16].copy_from_slice(&v);
            }
        }
        C::Bc5ToRg8 { signed } | C::Ctx1ToRg8 { signed } => {
            let is_bc5 = matches!(conversion, C::Bc5ToRg8 { .. });
            if signed {
                let mut v = [[0i8; 2]; 16];
                if is_bc5 {
                    bc::decode_bc5_snorm(src, &mut v)
                } else {
                    bc::decode_ctx1_snorm(src, &mut v)
                }
                for (i, t) in v.iter().enumerate() {
                    out[i * 2] = t[0] as u8;
                    out[i * 2 + 1] = t[1] as u8;
                }
            } else {
                let mut v = [[0u8; 2]; 16];
                if is_bc5 {
                    bc::decode_bc5_unorm(src, &mut v)
                } else {
                    bc::decode_ctx1_unorm(src, &mut v)
                }
                for (i, t) in v.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(t);
                }
            }
        }
        C::Dxt3aToR8 => {
            let mut v = [0u8; 16];
            bc::decode_dxt3a(src, &mut v);
            out[..16].copy_from_slice(&v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::TextureFormat;
    use crate::host::{HostCaps, host_mapping};

    fn convert(format: TextureFormat, signed: bool, caps: HostCaps, src: &[u8]) -> Vec<u8> {
        let m = host_mapping(format, signed, caps, 4, 4).unwrap();
        let mut out = vec![0u8; 256];
        convert_block(m.conversion, src, &mut out);
        out
    }

    #[test]
    fn half_float_conversion() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(65520.0), 0x7C00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NAN) & 0x7E00, 0x7E00);
        assert_eq!(f32_to_f16(2f32.powi(-24)), 1);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0);
        assert_eq!(f32_to_f16(3.0 * 2f32.powi(-25)), 2);
        assert_eq!(f32_to_f16(0.333_333_34), 0x3555);
        for h in 0u16..=0xFFFF {
            if (h >> 10) & 31 == 31 {
                continue;
            }
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#x}");
        }
    }

    #[test]
    fn special_floats() {
        assert_eq!(f7e3_to_f32(0), 0.0);
        // Exponent 3, mantissa 0 -> 2^(3-3) = 1.
        assert_eq!(f7e3_to_f32(3 << 7), 1.0);
        assert_eq!(f7e3_to_f32((7 << 7) | 0x7F), 31.875);
        // Denormal: mantissa 64 at exponent 0 -> 0.5 * 2^-2.
        assert_eq!(f7e3_to_f32(64), 0.125);
        assert_eq!(f20e4_to_f32(0), 0.0);
        assert_eq!(f20e4_to_f32(15 << 20), 1.0);
        assert_eq!(f20e4_to_f32(0x80000), 0.5 * 2f32.powi(-14));
        assert_eq!(unorm24_to_f32(0xFF_FFFF), 1.0);
        // Matches n / 16777215 rounding (0xC00000 is slightly above 0.75).
        assert_eq!(unorm24_to_f32(0xC0_0000), 12_582_913.0 / 16_777_216.0);
        assert_eq!(unorm24_to_f32(0x40_0000), 0.25);
        assert_eq!(unorm24_to_f32(0), 0.0);
    }

    #[test]
    fn packed_formats() {
        use TextureFormat::*;
        let caps = HostCaps::default();
        // 5_6_5: R in the low bits.
        let v: u16 = 31 | (32 << 5);
        assert_eq!(&convert(k_5_6_5, false, caps, &v.to_le_bytes())[..4], &[255, 130, 0, 255]);
        // 6_5_5: blue is 6 bits at the top.
        let v: u16 = (63 << 10) | 16;
        assert_eq!(&convert(k_6_5_5, false, caps, &v.to_le_bytes())[..4], &[132, 0, 255, 255]);
        let v: u16 = 0x8000 | (31 << 5);
        assert_eq!(&convert(k_1_5_5_5, false, caps, &v.to_le_bytes())[..4], &[0, 255, 0, 255]);
        let v: u16 = 0xF0A1;
        assert_eq!(&convert(k_4_4_4_4, false, caps, &v.to_le_bytes())[..4], &[17, 170, 0, 255]);
        // Signed 4_4_4_4: 0x8 -> -1, 0x7 -> 1.
        let v: u16 = 0x0087;
        assert_eq!(&convert(k_4_4_4_4, true, caps, &v.to_le_bytes())[..4], &[127, 0x81, 0, 0]);
    }

    #[test]
    fn wide_fixed_formats() {
        use TextureFormat::*;
        // 10_11_11: R 11 bits low, G 11, B 10 high.
        let v: u32 = 2047 | (1023 << 22);
        let out = convert(k_10_11_11, false, HostCaps::default(), &v.to_le_bytes());
        let halves: Vec<f32> = out[..8].chunks(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        assert_eq!(halves, [1.0, 0.0, 1.0, 1.0]);
        let norm = HostCaps { norm16: true, ..Default::default() };
        let v: u32 = 1023 | (1024 << 10);
        let out = convert(k_11_11_10, false, norm, &v.to_le_bytes());
        assert_eq!(&out[..8], &[0xFF, 0xFF, 0x10, 0x80, 0, 0, 0xFF, 0xFF]);
        let v: u32 = 0x200 | (0x3FF << 10);
        let out = convert(k_2_10_10_10, true, norm, &v.to_le_bytes());
        // -512 clamps to -1 (0x8001), 0x3FF = -1/511.
        assert_eq!(i16::from_le_bytes([out[0], out[1]]), -32767);
        assert_eq!(i16::from_le_bytes([out[2], out[3]]), -64);
    }

    #[test]
    fn norm_and_float_formats() {
        use TextureFormat::*;
        let caps = HostCaps::default();
        let out = convert(k_16_16, false, caps, &[0xFF, 0xFF, 0x00, 0x80]);
        assert_eq!(f16_to_f32(u16::from_le_bytes([out[0], out[1]])), 1.0);
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), f32_to_f16(32768.0 / 65535.0));
        let out = convert(k_16, true, caps, &[0x00, 0x80]);
        assert_eq!(f16_to_f32(u16::from_le_bytes([out[0], out[1]])), -1.0);
        let out = convert(k_16_16_EDRAM, false, caps, &[0xFF, 0x7F, 0x01, 0x80]);
        assert_eq!(f16_to_f32(u16::from_le_bytes([out[0], out[1]])), 32.0);
        assert_eq!(f16_to_f32(u16::from_le_bytes([out[2], out[3]])), -32.0);
        let out = convert(k_32, false, caps, &u32::MAX.to_le_bytes());
        assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 1.0);
        let mut rgb = Vec::new();
        for f in [1.5f32, -2.0, 0.25] {
            rgb.extend_from_slice(&f.to_le_bytes());
        }
        let out = convert(k_32_32_32_FLOAT, false, caps, &rgb);
        assert_eq!(&out[..12], &rgb[..]);
        assert_eq!(f32::from_le_bytes(out[12..16].try_into().unwrap()), 1.0);
        let out = convert(k_24_8, false, caps, &(0x7F_FFFFu32 << 8 | 0x55).to_le_bytes());
        assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 8_388_607.0 / 16_777_216.0);
        let out = convert(k_24_8_FLOAT, false, caps, &((15u32 << 20) << 8).to_le_bytes());
        assert_eq!(f32::from_le_bytes(out[..4].try_into().unwrap()), 1.0);
        let v: u32 = (3 << 7) | ((4 << 7) << 10) | (3 << 30);
        let out = convert(k_2_10_10_10_FLOAT_EDRAM, false, caps, &v.to_le_bytes());
        let h: Vec<f32> = out[..8].chunks(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        assert_eq!(h, [1.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn yuv_formats() {
        use TextureFormat::*;
        let caps = HostCaps::default();
        // Cr_Y1_Cb_Y0 from MSB: byte3 = Cr, byte2 = Y1, byte1 = Cb, byte0 = Y0.
        let out = convert(k_Cr_Y1_Cb_Y0_REP, false, caps, &[10, 20, 30, 40]);
        assert_eq!(&out[..8], &[40, 10, 20, 20, 40, 30, 20, 20]);
        // Y1_Cr_Y0_Cb: byte3 = Y1, byte2 = Cr, byte1 = Y0, byte0 = Cb.
        let out = convert(k_Y1_Cr_Y0_Cb_REP, false, caps, &[10, 20, 30, 40]);
        assert_eq!(&out[..8], &[30, 20, 10, 10, 30, 40, 10, 10]);
    }
}
