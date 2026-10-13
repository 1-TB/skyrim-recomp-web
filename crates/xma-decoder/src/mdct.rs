// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Half-length inverse MDCT and sine windows with the conventions of FFmpeg's
// libavutil/tx (AV_TX_FLOAT_MDCT, inverse: N coefficients in, the N middle
// samples of the 2N-sample IMDCT out), libavcodec/sinewin_tablegen.h and
// libavutil/float_dsp.c (vector_fmul_window), as used by wmaprodec.c.
//
// Copyright (c) Lynne; (c) 2008 Loren Merritt; (c) 2002 Fabrice Bellard (libavutil/tx)
// Copyright (c) 2009 Reimar Döffinger (sinewin_tablegen.h)
// Copyright 2005 Balatoni Denes, 2006 Loren Merritt (float_dsp.c)
// Rust implementation is part of the xma-decoder crate.
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation; either version 2.1 of the License, or (at your
// option) any later version. It is distributed WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU Lesser General Public License for more details.

use std::f64::consts::PI;

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn from_angle(angle: f64, scale: f64) -> Complex {
        Complex { re: (angle.cos() * scale) as f32, im: (angle.sin() * scale) as f32 }
    }

    #[inline(always)]
    fn mul(self, o: Complex) -> Complex {
        Complex { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

/// Largest transform: XMA frames are 512 samples, so subframes are 128..=512.
pub(crate) const MAX_LEN: usize = 512;

/// Inverse MDCT of one size, computed as a reversed DCT-IV through an
/// N/2-point complex FFT.
pub(crate) struct Imdct {
    n: usize,
    /// Pre-twiddle exp(-iπ(4k+1)/4N), pre-multiplied by the output scale.
    pre: Box<[Complex]>,
    /// Post-twiddle exp(-iπk/N).
    post: Box<[Complex]>,
    /// FFT roots exp(-2πij/(N/2)), j < N/4.
    roots: Box<[Complex]>,
    /// Bit reversal permutation of 0..N/2.
    bitrev: Box<[u16]>,
}

impl Imdct {
    /// `scale` multiplies every output sample.
    pub(crate) fn new(n: usize, scale: f64) -> Imdct {
        assert!(n.is_power_of_two() && (8..=MAX_LEN).contains(&n));
        let m = n / 2;
        let bits = m.trailing_zeros();
        Imdct {
            n,
            pre: (0..m).map(|k| Complex::from_angle(-PI * (4 * k + 1) as f64 / (4 * n) as f64, scale)).collect(),
            post: (0..m).map(|k| Complex::from_angle(-PI * k as f64 / n as f64, 1.0)).collect(),
            roots: (0..m / 2).map(|j| Complex::from_angle(-2.0 * PI * j as f64 / m as f64, 1.0)).collect(),
            bitrev: (0..m).map(|i| ((i as u32).reverse_bits() >> (32 - bits)) as u16).collect(),
        }
    }

    /// `output[i] = scale * Σ_k input[k]·cos(π/N·(N-1-i+½)(k+½))`, i.e. the
    /// DCT-IV of `input` reversed, which equals the middle half of the IMDCT.
    pub(crate) fn transform(&self, input: &[f32], output: &mut [f32]) {
        let n = self.n;
        let m = n / 2;
        let (input, output) = (&input[..n], &mut output[..n]);
        let mut buf = [Complex::default(); MAX_LEN / 2];
        let buf = &mut buf[..m];
        for k in 0..m {
            let v = Complex { re: input[2 * k], im: input[n - 1 - 2 * k] };
            buf[self.bitrev[k] as usize] = v.mul(self.pre[k]);
        }
        fft_in_place(buf, &self.roots);
        for k in 0..m {
            let u = buf[k].mul(self.post[k]);
            // DCT-IV c[2k] = Re u, c[N-1-2k] = -Im u; output is c reversed.
            output[n - 1 - 2 * k] = u.re;
            output[2 * k] = -u.im;
        }
    }
}

/// Radix-2 decimation-in-time FFT over bit-reversed input.
fn fft_in_place(buf: &mut [Complex], roots: &[Complex]) {
    let m = buf.len();
    let mut half = 1;
    while half < m {
        let step = m / (half * 2);
        for start in (0..m).step_by(half * 2) {
            let (lo, hi) = buf[start..start + 2 * half].split_at_mut(half);
            for (j, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let t = b.mul(roots[j * step]);
                *b = Complex { re: a.re - t.re, im: a.im - t.im };
                *a = Complex { re: a.re + t.re, im: a.im + t.im };
            }
        }
        half *= 2;
    }
}

/// FFmpeg's ff_sine_window_init: the rising half of a sine window, `n`
/// samples long, for overlapping two blocks over `n` samples.
pub(crate) fn sine_window(n: usize) -> Box<[f32]> {
    (0..n).map(|i| ((i as f64 + 0.5) * (PI / (2.0 * n as f64))) as f32).map(f32::sin).collect()
}

/// FFmpeg's vector_fmul_window, in place over `buf` (2·`len` samples):
/// overlaps the first half (previous block tail) with the second half
/// (current block head) using `win` (2·`len` samples).
pub(crate) fn fmul_window_in_place(buf: &mut [f32], win: &[f32], len: usize) {
    let (buf, win) = (&mut buf[..2 * len], &win[..2 * len]);
    for i in 0..len {
        let j = 2 * len - 1 - i;
        let (s0, s1) = (buf[i], buf[j]);
        let (wi, wj) = (win[i], win[j]);
        buf[i] = s0 * wj - s1 * wi;
        buf[j] = s0 * wi + s1 * wj;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FFmpeg's ff_tx_mdct_naive_inv, in double precision.
    fn naive(input: &[f32], scale: f64) -> Vec<f64> {
        let len2 = input.len();
        let len = len2 / 2;
        let phase = PI / (4.0 * len2 as f64);
        let mut out = vec![0.0; len2];
        for i in 0..len {
            let i_d = phase * (4 * len - 2 * i - 1) as f64;
            let i_u = phase * (3 * len2 + 2 * i + 1) as f64;
            let (mut sum_d, mut sum_u) = (0.0, 0.0);
            for (j, &x) in input.iter().enumerate() {
                let a = (2 * j + 1) as f64;
                sum_d += (a * i_d).cos() * x as f64;
                sum_u += (a * i_u).cos() * x as f64;
            }
            out[i] = sum_d * scale;
            out[i + len] = -sum_u * scale;
        }
        out
    }

    #[test]
    fn matches_naive_imdct() {
        let mut seed = 0x1234_5678u32;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for n in [8, 16, 128, 256, 512] {
            let scale = 1.0 / (n / 2) as f64 / 32768.0;
            let t = Imdct::new(n, scale);
            let input: Vec<f32> = (0..n).map(|_| rand() * 3000.0).collect();
            let mut out = vec![0.0f32; n];
            t.transform(&input, &mut out);
            let want = naive(&input, scale);
            let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (a, b) in out.iter().zip(&want) {
                assert!((*a as f64 - b).abs() <= peak * 2e-6, "n={n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn single_coefficient_is_a_cosine() {
        let n = 128;
        let t = Imdct::new(n, 1.0);
        let mut input = vec![0.0f32; n];
        input[3] = 1.0;
        let mut out = vec![0.0f32; n];
        t.transform(&input, &mut out);
        for (i, v) in out.iter().enumerate() {
            let want = (PI / n as f64 * ((n - 1 - i) as f64 + 0.5) * 3.5).cos();
            assert!((*v as f64 - want).abs() < 1e-5);
        }
    }

    #[test]
    fn window_is_power_complementary() {
        let w = sine_window(256);
        for i in 0..256 {
            let s = w[i] * w[i] + w[255 - i] * w[255 - i];
            assert!((s - 1.0).abs() < 1e-6);
        }
        // Overlapping a constant tail with a constant head under a power
        // complementary window preserves energy pairwise.
        let mut buf = vec![1.0f32; 256];
        fmul_window_in_place(&mut buf, &w, 128);
        for i in 0..128 {
            assert!((buf[i] - (w[255 - i] - w[i])).abs() < 1e-6);
            assert!((buf[255 - i] - (w[i] + w[255 - i])).abs() < 1e-6);
        }
    }
}
