//! Fast non-cryptographic hashing of guest memory for revalidation.

const P1: u64 = 0x9E37_79B9_7F4A_7C15;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;

fn fmix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ (h >> 33)
}

fn lane(acc: u64, v: u64) -> u64 {
    (acc ^ v.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

/// Hashes `data` with four independent 64-bit lanes (32 bytes per step).
pub(crate) fn hash(seed: u64, data: &[u8]) -> u64 {
    let mut acc = [seed ^ P1, seed ^ P2, seed.rotate_left(23) ^ P1, seed.rotate_left(41) ^ P2];
    let (chunks, rest) = data.as_chunks::<32>();
    for chunk in chunks {
        let (words, _) = chunk.as_chunks::<8>();
        for (a, word) in acc.iter_mut().zip(words) {
            *a = lane(*a, u64::from_le_bytes(*word));
        }
    }
    let mut h = acc[0].rotate_left(1) ^ acc[1].rotate_left(7) ^ acc[2].rotate_left(12) ^ acc[3].rotate_left(18);
    let (words, rest) = rest.as_chunks::<8>();
    for word in words {
        h = lane(h, u64::from_le_bytes(*word));
    }
    if !rest.is_empty() {
        let mut last = [0u8; 8];
        last[..rest.len()].copy_from_slice(rest);
        h = lane(h, u64::from_le_bytes(last));
    }
    fmix(h ^ data.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::hash;

    #[test]
    fn sensitive_to_every_byte_and_length() {
        let base: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let h = hash(0, &base);
        assert_eq!(h, hash(0, &base));
        for i in [0, 7, 31, 32, 500, 991, 999] {
            let mut changed = base.clone();
            changed[i] ^= 1;
            assert_ne!(hash(0, &changed), h, "byte {i}");
        }
        assert_ne!(hash(0, &base[..999]), h);
        assert_ne!(hash(1, &base), h);
        assert_ne!(hash(0, &[0u8; 8]), hash(0, &[0u8; 9]));
    }
}
