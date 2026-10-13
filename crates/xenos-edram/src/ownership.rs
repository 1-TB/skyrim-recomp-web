//! Which host render target holds the current contents of each EDRAM tile (Xenia's ownership
//! ranges). Ranges are kept sorted, non-overlapping and merged within one 2048-tile period.

use xenos_state::edram::{EDRAM_TILE_COUNT, RenderTargetKey};

/// A run of tiles `[start, end)` and its owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedRange {
    pub start: u32,
    pub end: u32,
    pub owner: RenderTargetKey,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OwnershipMap {
    ranges: Vec<OwnedRange>,
}

/// Splits a wrapping `(start, length)` span into at most two non-wrapping `[start, end)` extents.
pub(crate) fn split_span(start: u32, length: u32) -> impl Iterator<Item = (u32, u32)> {
    let start = start & (EDRAM_TILE_COUNT - 1);
    let length = length.min(EDRAM_TILE_COUNT);
    let end = start + length;
    let first = (start, end.min(EDRAM_TILE_COUNT));
    let second = (end > EDRAM_TILE_COUNT).then(|| (0, end - EDRAM_TILE_COUNT));
    std::iter::once(first).filter(|r| r.1 > r.0).chain(second)
}

impl OwnershipMap {
    pub fn ranges(&self) -> &[OwnedRange] {
        &self.ranges
    }

    /// Ranges overlapping the wrapping span, clipped to it.
    pub fn owners_in(&self, start: u32, length: u32) -> Vec<OwnedRange> {
        let mut out = Vec::new();
        for (a, b) in split_span(start, length) {
            for r in &self.ranges {
                let (s, e) = (r.start.max(a), r.end.min(b));
                if s < e {
                    out.push(OwnedRange { start: s, end: e, owner: r.owner });
                }
            }
        }
        out
    }

    /// Makes `owner` own the wrapping span; returns the clipped ranges previously owned by others
    /// (the contents to transfer).
    pub fn claim(&mut self, start: u32, length: u32, owner: RenderTargetKey) -> Vec<OwnedRange> {
        let mut previous = Vec::new();
        for (a, b) in split_span(start, length) {
            let mut next = Vec::with_capacity(self.ranges.len() + 2);
            for r in &self.ranges {
                if r.end <= a || r.start >= b {
                    next.push(*r);
                    continue;
                }
                if r.start < a {
                    next.push(OwnedRange { end: a, ..*r });
                }
                if r.end > b {
                    next.push(OwnedRange { start: b, ..*r });
                }
                if r.owner != owner {
                    previous.push(OwnedRange { start: r.start.max(a), end: r.end.min(b), owner: r.owner });
                }
            }
            next.push(OwnedRange { start: a, end: b, owner });
            next.sort_by_key(|r| r.start);
            self.ranges = merge(next);
        }
        previous
    }
}

fn merge(sorted: Vec<OwnedRange>) -> Vec<OwnedRange> {
    let mut out: Vec<OwnedRange> = Vec::with_capacity(sorted.len());
    for r in sorted {
        match out.last_mut() {
            Some(last) if last.end == r.start && last.owner == r.owner => last.end = r.end,
            _ => out.push(r),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use xenos_state::xenos::MsaaSamples;

    fn key(base: u32, format: u32) -> RenderTargetKey {
        RenderTargetKey {
            base_tiles: base,
            pitch_tiles_at_32bpp: 16,
            msaa_samples: MsaaSamples::X1,
            is_depth: false,
            format,
        }
    }

    #[test]
    fn claims_split_and_merge() {
        let mut map = OwnershipMap::default();
        let a = key(0, 0);
        let b = key(100, 7);
        assert!(map.claim(0, 300, a).is_empty());
        let prev = map.claim(100, 50, b);
        assert_eq!(prev, vec![OwnedRange { start: 100, end: 150, owner: a }]);
        assert_eq!(map.ranges().len(), 3);
        // Reclaiming by the same owner transfers nothing back except the middle.
        let prev = map.claim(0, 300, a);
        assert_eq!(prev, vec![OwnedRange { start: 100, end: 150, owner: b }]);
        assert_eq!(map.ranges(), &[OwnedRange { start: 0, end: 300, owner: a }]);
        assert_eq!(map.owners_in(250, 100), vec![OwnedRange { start: 250, end: 300, owner: a }]);
    }

    #[test]
    fn wrapping_spans() {
        let mut map = OwnershipMap::default();
        let a = key(2000, 0);
        map.claim(2000, 100, a);
        assert_eq!(
            map.ranges(),
            &[OwnedRange { start: 0, end: 52, owner: a }, OwnedRange { start: 2000, end: 2048, owner: a }]
        );
        let b = key(10, 0);
        let prev = map.claim(10, 2048, b);
        assert_eq!(prev.len(), 3);
        assert_eq!(map.ranges(), &[OwnedRange { start: 0, end: 2048, owner: b }]);
        assert_eq!(split_span(5, 0).count(), 0);
    }
}
