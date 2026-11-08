//! Full-resolution copies of resolves, for resolution scales above 1.
//!
//! A resolve stores the guest's pixels into (the GPU copy of) guest memory, averaging each
//! guest pixel's host pixels: textures made from that memory have the guest's resolution. So
//! that rendering at a higher resolution shows, each resolve into a 2D 32 or 64 bpp destination
//! also writes its host pixels, converted the same way but not averaged, into a buffer of its
//! own: the destination at `scale` times its size, rows of `pitch * scale` pixels, untiled.
//! A texture of that memory (same base, pitch, element size and byte order, one level, within
//! what has been resolved there) is loaded from the buffer instead — the same conversion, at
//! `scale` times the size — as long as the GPU copy of its memory still holds resolve output.
//! Shaders address textures in guest texels from the fetch constant, so nothing else changes.

use gpu_webgpu::{Buffer, BufferDescriptor, BufferUsage, Device};

/// Bytes of full-resolution copies kept (least recently used ones go first).
const BUDGET_BYTES: u64 = 512 << 20;

/// Frames a copy is kept without being resolved to or loaded from.
const MAX_IDLE_FRAMES: u64 = 600;

/// What a destination is, as far as its copy's layout goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DestKey {
    /// Physical address.
    pub base: u32,
    /// Row pitch and height of the (tiled) destination in pixels.
    pub pitch: u32,
    pub height: u32,
    pub bytes_per_element: u32,
    /// The destination's `Endian128` byte swap.
    pub endian: u32,
}

impl DestKey {
    /// Guest bytes the destination spans (whole 32x32 tiles).
    fn span(&self) -> (u32, u32) {
        (self.base, self.pitch * self.height.next_multiple_of(32) * self.bytes_per_element)
    }
}

struct Copy {
    key: DestKey,
    buffer: Buffer,
    /// Rectangles resolved into it (guest pixels: x, y, width, height).
    rects: Vec<[u32; 4]>,
    last_used: u64,
}

/// A copy a texture can be loaded from.
pub struct ScaledResolve {
    pub buffer: Buffer,
    /// Host pixels per guest pixel along each axis.
    pub scale: u32,
    /// Row pitch of the copy in host pixels (`pitch * scale`).
    pub pitch: u32,
}

#[derive(Default)]
pub(crate) struct ScaledCopies {
    copies: Vec<Copy>,
    frame: u64,
}

fn overlaps(a: (u32, u32), b: (u32, u32)) -> bool {
    u64::from(a.0) < u64::from(b.0) + u64::from(b.1) && u64::from(b.0) < u64::from(a.0) + u64::from(a.1)
}

impl ScaledCopies {
    /// A resolve of `rect` (guest pixels) is stored into `key`'s destination: the copy to write
    /// its full-resolution pixels into (made if needed) and where in it (guest pixels; further
    /// down than `rect` for rows of a destination with a copy, see `canonical`). Copies of
    /// other destinations it writes over are dropped into `retired`.
    pub fn resolving(
        &mut self,
        device: &Device,
        key: DestKey,
        rect: [u32; 4],
        scale: u32,
        retired: &mut Vec<Buffer>,
    ) -> (Buffer, [u32; 4]) {
        let (key, rect) = canonical(self.copies.iter().map(|c| c.key), key, rect);
        self.forget(key.span(), Some(key), retired);
        let frame = self.frame;
        if let Some(c) = self.copies.iter_mut().find(|c| c.key == key) {
            c.last_used = frame;
            // A resolve over an earlier one's rectangle replaces it.
            c.rects.retain(|r| !contains(rect, *r));
            c.rects.push(rect);
            return (c.buffer.clone(), rect);
        }
        let size = u64::from(key.pitch * scale) * u64::from(key.height * scale) * u64::from(key.bytes_per_element);
        while !self.copies.is_empty() && self.bytes() + size > BUDGET_BYTES {
            let oldest = (0..self.copies.len()).min_by_key(|&i| self.copies[i].last_used).expect("not empty");
            retired.push(self.copies.swap_remove(oldest).buffer);
        }
        let buffer = device.create_buffer(&BufferDescriptor {
            label: "edram scaled resolve",
            size: size.max(4),
            usage: BufferUsage::Storage,
            mapped_at_creation: false,
        });
        self.copies.push(Copy { key, buffer: buffer.clone(), rects: vec![rect], last_used: frame });
        (buffer, rect)
    }

    /// Guest memory `span` was written other than by a resolve into `except`: copies of what
    /// it overlaps are out of date.
    pub fn forget(&mut self, span: (u32, u32), except: Option<DestKey>, retired: &mut Vec<Buffer>) {
        let mut i = 0;
        while i < self.copies.len() {
            let c = &self.copies[i];
            if Some(c.key) != except && overlaps(c.key.span(), span) {
                retired.push(self.copies.swap_remove(i).buffer);
            } else {
                i += 1;
            }
        }
    }

    /// The copy of a destination `matches` accepts that holds `width` x `height` pixels from
    /// its origin.
    pub fn find_any_height(
        &mut self,
        matches: impl Fn(&DestKey) -> bool,
        width: u32,
        height: u32,
        scale: u32,
    ) -> Option<ScaledResolve> {
        let frame = self.frame;
        let c = self.copies.iter_mut().find(|c| matches(&c.key) && covered(&c.rects, width, height))?;
        c.last_used = frame;
        Some(ScaledResolve { buffer: c.buffer.clone(), scale, pitch: c.key.pitch * scale })
    }

    /// Frame boundary: drops copies unused for a while.
    pub fn end_frame(&mut self, retired: &mut Vec<Buffer>) {
        self.frame += 1;
        let frame = self.frame;
        let mut i = 0;
        while i < self.copies.len() {
            if frame - self.copies[i].last_used > MAX_IDLE_FRAMES {
                retired.push(self.copies.swap_remove(i).buffer);
            } else {
                i += 1;
            }
        }
    }

    fn bytes(&self) -> u64 {
        self.copies.iter().map(|c| c.buffer.size()).sum()
    }
}

/// A resolve into rows further down a destination that has a copy (as with predicated
/// tiling, where each screen tile is resolved to the address of its first row) as a resolve
/// into that destination: its key, and the rectangle moved down by those rows.
fn canonical(copies: impl Iterator<Item = DestKey>, key: DestKey, rect: [u32; 4]) -> (DestKey, [u32; 4]) {
    let row_bytes = key.pitch * key.bytes_per_element;
    for k in copies {
        if (k.pitch, k.bytes_per_element, k.endian) != (key.pitch, key.bytes_per_element, key.endian)
            || key.base < k.base
        {
            continue;
        }
        // Tiled rows are contiguous in bands of 32: a band boundary is `32 * row_bytes` in.
        let offset = key.base - k.base;
        if offset % (32 * row_bytes) != 0 {
            continue;
        }
        let rows = offset / row_bytes;
        if rows + rect[1] + rect[3] <= k.height {
            return (k, [rect[0], rect[1] + rows, rect[2], rect[3]]);
        }
    }
    (key, rect)
}


/// Whether rectangle `outer` contains `inner` (x, y, width, height).
fn contains(outer: [u32; 4], inner: [u32; 4]) -> bool {
    inner[0] >= outer[0]
        && inner[1] >= outer[1]
        && inner[0] + inner[2] <= outer[0] + outer[2]
        && inner[1] + inner[3] <= outer[1] + outer[3]
}

/// Whether `rects` together cover `[0, width) x [0, height)`.
fn covered(rects: &[[u32; 4]], width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    // Split at every rectangle edge; each cell must lie in some rectangle.
    let mut xs: Vec<u32> = vec![0, width];
    let mut ys: Vec<u32> = vec![0, height];
    for r in rects {
        xs.extend([r[0], r[0] + r[2]].into_iter().filter(|&x| x < width));
        ys.extend([r[1], r[1] + r[3]].into_iter().filter(|&y| y < height));
    }
    xs.sort_unstable();
    xs.dedup();
    ys.sort_unstable();
    ys.dedup();
    ys.windows(2).all(|y| {
        xs.windows(2).all(|x| rects.iter().any(|r| contains(*r, [x[0], y[0], x[1] - x[0], y[1] - y[0]])))
    })
}

#[cfg(test)]
mod tests {
    use super::{DestKey, canonical, covered};

    #[test]
    fn tiles_resolved_to_their_rows_join_the_destination() {
        let key = |base| DestKey { base, pitch: 1280, height: 736, bytes_per_element: 4, endian: 2 };
        let screen = key(0x0199_8000);
        let other = key(0x01D3_0000);
        // Rows 256-511 resolved to the address of row 256 (256 rows of 5120 bytes in).
        let (k, rect) = canonical([other, screen].into_iter(), key(0x01AD_8000), [0, 0, 1280, 256]);
        assert_eq!((k, rect), (screen, [0, 256, 1280, 256]));
        let (k, rect) = canonical([other, screen].into_iter(), key(0x01C1_8000), [0, 0, 1280, 208]);
        assert_eq!((k, rect), (screen, [0, 512, 1280, 208]));
        // Past the destination's rows, or not at a 32-row band: a destination of its own.
        assert_eq!(canonical([screen].into_iter(), other, [0, 0, 1280, 256]).0, other);
        let odd = key(0x0199_8000 + 16 * 5120);
        assert_eq!(canonical([screen].into_iter(), odd, [0, 0, 64, 64]).0, odd);
    }

    #[test]
    fn strips_cover_the_screen() {
        let strips = [[0, 0, 1280, 208], [0, 208, 1280, 256], [0, 464, 1280, 256]];
        assert!(covered(&strips, 1280, 720));
        assert!(!covered(&strips[..2], 1280, 720));
        assert!(covered(&strips[..2], 1280, 464));
        assert!(!covered(&[[0, 0, 640, 720]], 1280, 720));
        assert!(covered(&[[0, 0, 640, 720], [640, 0, 640, 720]], 1280, 720));
    }
}
