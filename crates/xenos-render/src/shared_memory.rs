//! Guest physical memory mirrored in GPU storage buffers (`xe_shared_memory_N`, 128 MiB each
//! since that is WebGPU's default storage binding limit), which the translated shaders read
//! vertex data from.
//!
//! Before each draw the ranges it reads are compared page by page against what was last uploaded
//! and changed pages are re-sent: by write version when the memory tracks writes
//! ([`PhysicalMemory::version`]), else by content hash.
//! Uploads (`Queue::write_buffer`) take effect before the next submission as a whole, so a page
//! that changes after a draw of the same not-yet-submitted batch read it is a conflict: the
//! executor must submit first.

use gpu_webgpu::{Buffer, BufferDescriptor, BufferUsage, Device};
use xenos_shader::bindings::{SHARED_MEMORY_BUFFER_BYTES, SHARED_MEMORY_BUFFERS};

use crate::PhysicalMemory;

/// Xenos physical memory size.
pub const PHYSICAL_SIZE: u64 = 512 << 20;
const PAGE: u32 = 4096;

/// A page changed after an earlier draw of the current batch read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conflict;

pub struct SharedMemory {
    /// One buffer per [`SHARED_MEMORY_BUFFER_BYTES`] of physical memory, created when a range in
    /// it is first synced; until then the slot holds a small placeholder (shader reads past a
    /// buffer's end return zero). Titles rarely use all 512 MiB, and on phones GPU memory comes
    /// out of the tab's budget.
    pub buffers: Vec<Buffer>,
    /// Bumped whenever a placeholder in `buffers` is replaced: bind groups holding the old
    /// buffers must be recreated.
    pub generation: u64,
    mirrored: Vec<bool>,
    /// Bytes mirrored per buffer: [`SHARED_MEMORY_BUFFER_BYTES`], or less when the device can't
    /// bind that much (shader reads past the end return zero).
    part: u64,
    /// Write version (or content hash) of each uploaded page, by page number.
    pages: Vec<Option<u64>>,
    /// Pages read by draws of the batch being recorded.
    batch: PageSet,
    /// Pages the GPU wrote (resolves stored into the buffers) after their CPU content, identified
    /// in `pages`, was uploaded: the buffers hold their current data and guest memory is stale.
    /// A page leaves the set when the CPU rewrites it (its identity changes).
    gpu_pages: PageSet,
    scratch: Vec<u8>,
    changed: Vec<(u32, u64)>,
    /// Bytes uploaded so far.
    pub uploaded: u64,
}

/// A set of page numbers: a bitmap plus a list of what was inserted, so clearing costs only
/// what was inserted.
struct PageSet {
    bits: Vec<u64>,
    members: Vec<u32>,
    count: usize,
}

impl PageSet {
    fn new() -> Self {
        PageSet { bits: vec![0; (PAGES as usize).div_ceil(64)], members: Vec::new(), count: 0 }
    }

    fn contains(&self, page: u32) -> bool {
        self.bits[page as usize / 64] >> (page % 64) & 1 != 0
    }

    fn insert(&mut self, page: u32) {
        let word = &mut self.bits[page as usize / 64];
        if *word >> (page % 64) & 1 == 0 {
            *word |= 1 << (page % 64);
            self.count += 1;
            self.members.push(page);
            if self.members.len() > PAGES as usize {
                // Removed pages leave stale entries behind; drop them.
                let bits = &self.bits;
                self.members.retain(|&p| bits[p as usize / 64] >> (p % 64) & 1 != 0);
                self.members.sort_unstable();
                self.members.dedup();
            }
        }
    }

    fn remove(&mut self, page: u32) {
        let word = &mut self.bits[page as usize / 64];
        if *word >> (page % 64) & 1 != 0 {
            *word &= !(1 << (page % 64));
            self.count -= 1;
        }
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn clear(&mut self) {
        for p in self.members.drain(..) {
            self.bits[p as usize / 64] = 0;
        }
        self.count = 0;
    }
}

/// Number of pages of physical memory.
const PAGES: u32 = (PHYSICAL_SIZE / PAGE as u64) as u32;

/// Fast non-cryptographic page hash (multiply-xorshift over 64-bit words, 4 lanes).
fn hash_page(data: &[u8]) -> u64 {
    const K: [u64; 4] = [0x9E37_79B9_7F4A_7C15, 0xC2B2_AE3D_27D4_EB4F, 0x1656_67B1_9E37_79F9, 0x85EB_CA77_C2B2_AE63];
    let mut lanes = K;
    for chunk in data.as_chunks::<32>().0 {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let w = u64::from_le_bytes(chunk[i * 8..i * 8 + 8].try_into().unwrap());
            *lane = (*lane ^ w).wrapping_mul(K[i]).rotate_left(29);
        }
    }
    let mut h = lanes[0] ^ lanes[1].rotate_left(17) ^ lanes[2].rotate_left(31) ^ lanes[3].rotate_left(47);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^ (h >> 33)
}

impl SharedMemory {
    /// Identifies page `p`'s current contents (`scratch` is clobbered).
    fn identify(memory: &dyn PhysicalMemory, p: u32, scratch: &mut [u8]) -> u64 {
        if let Some(version) = memory.version(p, PAGE) {
            return version;
        }
        memory.read(p, scratch);
        hash_page(scratch)
    }

    pub fn new(device: &Device) -> Self {
        debug_assert_eq!(SHARED_MEMORY_BUFFER_BYTES * u64::from(SHARED_MEMORY_BUFFERS), PHYSICAL_SIZE);
        let limits = device.limits();
        let part = SHARED_MEMORY_BUFFER_BYTES.min(limits.maxStorageBufferBindingSize).min(limits.maxBufferSize)
            & !u64::from(PAGE - 1);
        if part < SHARED_MEMORY_BUFFER_BYTES {
            log::warn!(
                "shared memory: device binds at most {part:#X} bytes per buffer; some vertex data reads as zero"
            );
        }
        let placeholder = device.create_buffer(&BufferDescriptor {
            label: "xenos shared memory placeholder",
            size: 256,
            usage: Self::usage(),
            mapped_at_creation: false,
        });
        SharedMemory {
            buffers: vec![placeholder; SHARED_MEMORY_BUFFERS as usize],
            generation: 0,
            mirrored: vec![false; SHARED_MEMORY_BUFFERS as usize],
            part,
            pages: vec![None; PAGES as usize],
            batch: PageSet::new(),
            gpu_pages: PageSet::new(),
            scratch: vec![0; PAGE as usize],
            changed: Vec::new(),
            uploaded: 0,
        }
    }

    fn usage() -> BufferUsage {
        // Index: draws read guest index buffers from here directly.
        BufferUsage::Storage | BufferUsage::Index | BufferUsage::CopyDst
    }

    /// Creates the buffers mirroring `[addr, end)` that are still placeholders. Nothing recorded
    /// can read a placeholder's range (reading it means syncing it first), so replacing one
    /// mid-batch only needs new bind groups.
    fn mirror(&mut self, device: &Device, addr: u32, end: u64) {
        let first = u64::from(addr) / SHARED_MEMORY_BUFFER_BYTES;
        let last = (end - 1) / SHARED_MEMORY_BUFFER_BYTES;
        for n in first..=last {
            let n = n as usize;
            if self.mirrored[n] {
                continue;
            }
            log::info!("shared memory: mirroring buffer {n} (first used at {addr:#010X})");
            self.buffers[n] = device.create_buffer(&BufferDescriptor {
                label: &format!("xenos shared memory {n}"),
                size: self.part,
                usage: Self::usage(),
                mapped_at_creation: false,
            });
            self.mirrored[n] = true;
            self.generation += 1;
        }
    }

    /// Makes `[addr, addr + len)` current for the batch being recorded. On a conflict nothing is
    /// uploaded; submit the batch, call [`SharedMemory::end_batch`] and retry.
    pub fn sync(&mut self, device: &Device, memory: &dyn PhysicalMemory, addr: u32, len: u32) -> Result<(), Conflict> {
        if len == 0 || u64::from(addr) >= PHYSICAL_SIZE {
            return Ok(());
        }
        let end = (u64::from(addr) + u64::from(len)).min(PHYSICAL_SIZE);
        self.mirror(device, addr, end);
        self.changed.clear();
        let mut page = u64::from(addr & !(PAGE - 1));
        while page < end {
            let p = page as u32;
            if page % SHARED_MEMORY_BUFFER_BYTES >= self.part {
                page += u64::from(PAGE);
                continue;
            }
            let hash = Self::identify(memory, p, &mut self.scratch);
            if self.pages[(p / PAGE) as usize] != Some(hash) {
                if self.batch.contains(p / PAGE) {
                    return Err(Conflict);
                }
                self.changed.push((p, hash));
            }
            page += u64::from(PAGE);
        }
        // Runs of consecutive pages in one buffer go up in one write (each WebGPU call has a
        // fixed cost in browsers).
        let mut i = 0;
        while i < self.changed.len() {
            let first = u64::from(self.changed[i].0);
            let mut n = 1;
            while let Some(&(p, _)) = self.changed.get(i + n)
                && u64::from(p) == first + n as u64 * u64::from(PAGE)
                && (u64::from(p) % SHARED_MEMORY_BUFFER_BYTES) < self.part
                && u64::from(p) / SHARED_MEMORY_BUFFER_BYTES == first / SHARED_MEMORY_BUFFER_BYTES
            {
                n += 1;
            }
            self.scratch.resize(n * PAGE as usize, 0);
            memory.read(first as u32, &mut self.scratch);
            let buffer = &self.buffers[(first / SHARED_MEMORY_BUFFER_BYTES) as usize];
            device.queue().write_buffer(buffer, first % SHARED_MEMORY_BUFFER_BYTES, &self.scratch);
            self.uploaded += self.scratch.len() as u64;
            for &(p, hash) in &self.changed[i..i + n] {
                self.pages[(p / PAGE) as usize] = Some(hash);
                self.gpu_pages.remove(p / PAGE);
            }
            i += n;
        }
        self.scratch.truncate(PAGE as usize);
        let mut page = u64::from(addr & !(PAGE - 1));
        while page < end {
            self.batch.insert(page as u32 / PAGE);
            page += u64::from(PAGE);
        }
        Ok(())
    }

    /// `[addr, addr + len)` was written on the GPU (a resolve). Its pages must have been
    /// [`SharedMemory::sync`]ed right before, so the parts the GPU didn't write are current too.
    pub fn mark_gpu_written(&mut self, addr: u32, len: u32) {
        let end = (u64::from(addr) + u64::from(len)).min(PHYSICAL_SIZE);
        let mut page = u64::from(addr & !(PAGE - 1));
        while page < end {
            let p = page as u32 / PAGE;
            if self.pages[p as usize].is_some() {
                self.gpu_pages.insert(p);
            }
            page += u64::from(PAGE);
        }
    }

    /// Whether the buffers hold the only current copy of every page of `ranges` (all written by
    /// the GPU and not rewritten by the CPU since).
    pub fn gpu_covers(&mut self, memory: &dyn PhysicalMemory, ranges: &[(u32, u32)]) -> bool {
        if self.gpu_pages.is_empty() {
            return false;
        }
        for &(addr, len) in ranges {
            let end = u64::from(addr) + u64::from(len);
            let mut page = u64::from(addr & !(PAGE - 1));
            while page < end {
                if page >= PHYSICAL_SIZE || !self.gpu_pages.contains(page as u32 / PAGE) {
                    return false;
                }
                let p = page as u32;
                if self.pages[(p / PAGE) as usize] != Some(Self::identify(memory, p, &mut self.scratch)) {
                    // The CPU rewrote it: guest memory is current again.
                    self.gpu_pages.remove(p / PAGE);
                    return false;
                }
                page += u64::from(PAGE);
            }
        }
        true
    }

    /// Bytes mirrored per buffer.
    pub fn part(&self) -> u64 {
        self.part
    }

    /// The recorded batch was submitted.
    pub fn end_batch(&mut self) {
        self.batch.clear();
    }

    /// Forgets what was uploaded for `[addr, addr + len)` (it changed behind our back, e.g. a
    /// resolve wrote it), so the next `sync` re-sends it.
    pub fn invalidate(&mut self, addr: u32, len: u32) {
        let end = (u64::from(addr) + u64::from(len)).min(PHYSICAL_SIZE);
        let mut page = u64::from(addr & !(PAGE - 1));
        while page < end {
            let p = page as u32 / PAGE;
            self.pages[p as usize] = None;
            self.gpu_pages.remove(p);
            page += u64::from(PAGE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PageSet, hash_page};

    #[test]
    fn page_set_inserts_removes_and_clears() {
        let mut set = PageSet::new();
        assert!(set.is_empty());
        set.insert(5);
        set.insert(5);
        set.insert(70_000);
        assert!(set.contains(5) && set.contains(70_000) && !set.contains(6));
        set.remove(5);
        assert!(!set.contains(5) && !set.is_empty());
        set.insert(5);
        set.clear();
        assert!(set.is_empty() && !set.contains(5) && !set.contains(70_000));
    }

    #[test]
    fn hash_sees_every_word() {
        let base = vec![0u8; 4096];
        let h = hash_page(&base);
        for i in (0..4096).step_by(8) {
            let mut d = base.clone();
            d[i] = 1;
            assert_ne!(hash_page(&d), h, "byte {i}");
        }
    }
}
