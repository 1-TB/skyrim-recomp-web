//! Guest physical memory mirrored on the GPU, where the translated shaders read vertex and index
//! data, textures are loaded from and resolves store their output.
//!
//! Only the [`BLOCK`]s the GPU uses are mirrored, each in a slot of up to
//! [`SHARED_MEMORY_CHUNKS`] buffers of [`SHARED_MEMORY_CHUNK_BYTES`] (created as slots run out)
//! and found through a table of slot numbers per block (`xe_shared_table`). Skyrim's GPU-visible
//! physical memory spans ~200 MiB, of which the GPU reads ~20 MiB until the world loads and
//! ~50-70 MiB at a time in it; on phones GPU memory comes out of the tab's budget. When the slots
//! are full, the least recently used blocks are dropped (and uploaded again when used), except
//! ones the batch being recorded reads and ones holding resolve output only the GPU has.
//!
//! Before each draw the ranges it reads are compared page by page against what was last uploaded
//! and changed pages are re-sent: by write version when the memory tracks writes
//! ([`PhysicalMemory::version`]), else by content hash.
//! Uploads (`Queue::write_buffer`) take effect before the next submission as a whole, so a page
//! that changes after a draw of the same not-yet-submitted batch read it is a conflict: the
//! executor must submit first. The same holds for slots: a block the batch reads keeps its slot.

use gpu_webgpu::{Buffer, BufferDescriptor, BufferUsage, CommandEncoder, Device};
use xenos_shader::bindings::{
    SHARED_MEMORY_BLOCK_BYTES, SHARED_MEMORY_BLOCKS, SHARED_MEMORY_CHUNK_BYTES, SHARED_MEMORY_CHUNKS,
};

use crate::PhysicalMemory;

/// Xenos physical memory size.
pub const PHYSICAL_SIZE: u64 = 512 << 20;
const PAGE: u32 = 4096;
/// Unit of mirroring.
pub const BLOCK: u32 = SHARED_MEMORY_BLOCK_BYTES;
const PAGES_PER_BLOCK: u32 = BLOCK / PAGE;
const SLOTS_PER_CHUNK: u32 = SHARED_MEMORY_CHUNK_BYTES / BLOCK;
const SLOTS: u32 = SLOTS_PER_CHUNK * SHARED_MEMORY_CHUNKS;

/// Counters for the stats log; `slot_bytes` and `blocks` are current, the rest totals.
#[derive(Clone, Copy, Debug, Default)]
pub struct MirrorStats {
    /// GPU memory of the slot buffers.
    pub slot_bytes: u64,
    /// Blocks mirrored now.
    pub blocks: u32,
    /// Blocks dropped from their slots for others.
    pub evictions: u64,
    /// Syncs that had to wait for the batch to be submitted to get a slot.
    pub slot_conflicts: u64,
}

/// A page changed after an earlier draw of the current batch read it (or a slot is needed and
/// only blocks the batch reads could give theirs up).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conflict;

pub struct SharedMemory {
    /// The slot buffers, [`SHARED_MEMORY_CHUNK_BYTES`] each once created; small placeholders
    /// (one per buffer, as they're bound writable side by side) until slots in them are needed.
    pub chunks: Vec<Buffer>,
    /// The GPU's copy of `slot_of`.
    pub table: Buffer,
    /// Bumped whenever a buffer in `chunks` is replaced: bind groups holding the old ones must
    /// be recreated.
    pub generation: u64,
    chunks_created: u32,
    /// Per block: its slot + 1, or 0 when it isn't mirrored.
    slot_of: Vec<u32>,
    /// Per slot: its block + 1, or 0 when it's free.
    owner: Vec<u32>,
    /// Per slot: the frame its block was last used in.
    last_used: Vec<u32>,
    frame: u32,
    warned_pinned: bool,
    stats: MirrorStats,
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

    fn usage() -> BufferUsage {
        // Index: draws read guest index buffers from here directly. CopySrc: texture loads
        // gather their data from here.
        BufferUsage::Storage | BufferUsage::Index | BufferUsage::CopyDst | BufferUsage::CopySrc
    }

    pub fn new(device: &Device) -> Self {
        let chunks = (0..SHARED_MEMORY_CHUNKS)
            .map(|_| {
                device.create_buffer(&BufferDescriptor {
                    label: "xenos shared memory placeholder",
                    size: 256,
                    usage: Self::usage(),
                    mapped_at_creation: false,
                })
            })
            .collect();
        let table = device.create_buffer(&BufferDescriptor {
            label: "xenos shared memory table",
            size: u64::from(SHARED_MEMORY_BLOCKS) * 4,
            usage: BufferUsage::Storage | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        SharedMemory {
            chunks,
            table,
            generation: 0,
            chunks_created: 0,
            slot_of: vec![0; SHARED_MEMORY_BLOCKS as usize],
            owner: vec![0; SLOTS as usize],
            last_used: vec![0; SLOTS as usize],
            frame: 1,
            warned_pinned: false,
            stats: MirrorStats::default(),
            pages: vec![None; PAGES as usize],
            batch: PageSet::new(),
            gpu_pages: PageSet::new(),
            scratch: vec![0; PAGE as usize],
            changed: Vec::new(),
            uploaded: 0,
        }
    }

    pub fn stats(&self) -> MirrorStats {
        MirrorStats {
            slot_bytes: u64::from(self.chunks_created) * u64::from(SHARED_MEMORY_CHUNK_BYTES),
            blocks: self.owner.iter().filter(|&&o| o != 0).count() as u32,
            ..self.stats
        }
    }

    /// Frame boundary (for choosing the least recently used blocks to drop).
    pub fn end_frame(&mut self) {
        self.frame += 1;
    }

    /// Where slot `slot` is: its buffer and the byte offset in it.
    fn slot_location(slot: u32) -> (usize, u64) {
        ((slot / SLOTS_PER_CHUNK) as usize, u64::from(slot % SLOTS_PER_CHUNK) * u64::from(BLOCK))
    }

    /// Where physical byte `addr` is mirrored, if its block is.
    fn location(&self, addr: u32) -> Option<(usize, u64)> {
        let slot = self.slot_of.get((addr / BLOCK) as usize).copied().filter(|&s| s != 0)? - 1;
        let (chunk, offset) = Self::slot_location(slot);
        Some((chunk, offset + u64::from(addr % BLOCK)))
    }

    /// Whether a page of `block` is read by the batch being recorded, or holds GPU-only data.
    fn block_busy(&self, block: u32) -> (bool, bool) {
        let pages = block * PAGES_PER_BLOCK..(block + 1) * PAGES_PER_BLOCK;
        (pages.clone().any(|p| self.batch.contains(p)), pages.into_iter().any(|p| self.gpu_pages.contains(p)))
    }

    /// Creates the next slot buffer, if there is one left; returns its first slot.
    fn add_chunk(&mut self, device: &Device) -> Option<u32> {
        if self.chunks_created == SHARED_MEMORY_CHUNKS {
            return None;
        }
        let n = self.chunks_created;
        log::info!(
            "shared memory: {} MiB of slots for guest memory blocks",
            (n + 1) * (SHARED_MEMORY_CHUNK_BYTES >> 20)
        );
        self.chunks[n as usize] = device.create_buffer(&BufferDescriptor {
            label: "xenos shared memory",
            size: u64::from(SHARED_MEMORY_CHUNK_BYTES),
            usage: Self::usage(),
            mapped_at_creation: false,
        });
        self.chunks_created += 1;
        self.generation += 1;
        Some(n * SLOTS_PER_CHUNK)
    }

    /// Whether slots `[start, start + n)` are free and in one buffer.
    fn run_free(&self, start: u32, n: u32) -> bool {
        start / SLOTS_PER_CHUNK == (start + n - 1) / SLOTS_PER_CHUNK
            && start + n <= self.chunks_created * SLOTS_PER_CHUNK
            && (start..start + n).all(|s| self.owner[s as usize] == 0)
    }

    /// `n` free consecutive slots in one buffer for blocks `[block, block + n)`: right after
    /// the previous block's or right before the next one's if free (so ranges mapped piecemeal
    /// still end up consecutive), else the first free run (in a new buffer if need be).
    fn free_run(&mut self, device: &Device, block: u32, n: u32) -> Option<u32> {
        if n > SLOTS_PER_CHUNK {
            return None;
        }
        let after = block.checked_sub(1).and_then(|b| self.slot_of[b as usize].checked_sub(1)).map(|s| s + 1);
        let before =
            self.slot_of.get((block + n) as usize).and_then(|s| s.checked_sub(1)).and_then(|s| s.checked_sub(n));
        if let Some(start) = after.into_iter().chain(before).find(|&start| self.run_free(start, n)) {
            return Some(start);
        }
        for chunk in 0..self.chunks_created {
            let mut run = 0;
            for slot in chunk * SLOTS_PER_CHUNK..(chunk + 1) * SLOTS_PER_CHUNK {
                run = if self.owner[slot as usize] == 0 { run + 1 } else { 0 };
                if run == n {
                    return Some(slot + 1 - n);
                }
            }
        }
        self.add_chunk(device)
    }

    /// Drops block `block` from its slot.
    fn evict(&mut self, device: &Device, slot: u32) {
        self.stats.evictions += 1;
        let block = self.owner[slot as usize] - 1;
        self.owner[slot as usize] = 0;
        self.slot_of[block as usize] = 0;
        device.queue().write_buffer(&self.table, u64::from(block) * 4, &0u32.to_le_bytes());
        for p in block * PAGES_PER_BLOCK..(block + 1) * PAGES_PER_BLOCK {
            self.pages[p as usize] = None;
            self.gpu_pages.remove(p);
        }
    }

    /// A free slot: an unused one, a new buffer's, or the least recently used block's that
    /// neither the batch nor `keep` (blocks being mapped) needs.
    fn free_slot(&mut self, device: &Device, keep: std::ops::RangeInclusive<u32>) -> Result<Option<u32>, Conflict> {
        let used = self.chunks_created * SLOTS_PER_CHUNK;
        if let Some(slot) = (0..used).find(|&s| self.owner[s as usize] == 0) {
            return Ok(Some(slot));
        }
        if let Some(slot) = self.add_chunk(device) {
            return Ok(Some(slot));
        }
        let mut best = None;
        let mut best_pinned = None;
        let mut batch_held = false;
        for slot in 0..used {
            let block = self.owner[slot as usize] - 1;
            if keep.contains(&block) {
                continue;
            }
            let (in_batch, pinned) = self.block_busy(block);
            let age = self.last_used[slot as usize];
            if in_batch {
                batch_held = true;
            } else if pinned {
                if best_pinned.is_none_or(|(_, a)| age < a) {
                    best_pinned = Some((slot, age));
                }
            } else if best.is_none_or(|(_, a)| age < a) {
                best = Some((slot, age));
            }
        }
        if best.is_none() && batch_held {
            // Submitting the batch frees its blocks.
            self.stats.slot_conflicts += 1;
            return Err(Conflict);
        }
        let Some((slot, _)) = best.or_else(|| {
            // Only resolve output is left: dropping it loses data the GPU produced.
            if !self.warned_pinned {
                self.warned_pinned = true;
                log::warn!("shared memory: slots full of resolve output; dropping some");
            }
            best_pinned
        }) else {
            return Ok(None);
        };
        self.evict(device, slot);
        Ok(Some(slot))
    }

    /// Gives every block of `[addr, end)` a slot.
    fn map(&mut self, device: &Device, addr: u32, end: u64) -> Result<(), Conflict> {
        let first = addr / BLOCK;
        let last = ((end - 1) / u64::from(BLOCK)) as u32;
        let mut changed = false;
        // The table is updated for whatever was mapped, also when a conflict stops this early.
        let result = self.assign(device, first, last, &mut changed);
        if changed {
            let entries: Vec<u8> =
                self.slot_of[first as usize..=last as usize].iter().flat_map(|s| s.to_le_bytes()).collect();
            device.queue().write_buffer(&self.table, u64::from(first) * 4, &entries);
        }
        for b in first..=last {
            if let Some(slot) = self.slot_of[b as usize].checked_sub(1) {
                self.last_used[slot as usize] = self.frame;
            }
        }
        result
    }

    fn assign(&mut self, device: &Device, first: u32, last: u32, changed: &mut bool) -> Result<(), Conflict> {
        let mut b = first;
        while b <= last {
            if self.slot_of[b as usize] != 0 {
                b += 1;
                continue;
            }
            let mut n = 1;
            while b + n <= last && self.slot_of[(b + n) as usize] == 0 {
                n += 1;
            }
            // Consecutive blocks in consecutive slots when possible: an index buffer is then
            // bound straight from the slots, and pages go up in fewer writes.
            if let Some(start) = self.free_run(device, b, n) {
                for k in 0..n {
                    self.slot_of[(b + k) as usize] = start + k + 1;
                    self.owner[(start + k) as usize] = b + k + 1;
                }
                *changed = true;
            } else {
                for k in 0..n {
                    match self.free_slot(device, first..=last)? {
                        Some(slot) => {
                            self.slot_of[(b + k) as usize] = slot + 1;
                            self.owner[slot as usize] = b + k + 1;
                            *changed = true;
                        }
                        None => log::warn!("shared memory: no slot for block {:#010X}", (b + k) * BLOCK),
                    }
                }
            }
            b += n;
        }
        Ok(())
    }

    /// Makes `[addr, addr + len)` current for the batch being recorded. On a conflict nothing is
    /// uploaded; submit the batch, call [`SharedMemory::end_batch`] and retry.
    pub fn sync(&mut self, device: &Device, memory: &dyn PhysicalMemory, addr: u32, len: u32) -> Result<(), Conflict> {
        if len == 0 || u64::from(addr) >= PHYSICAL_SIZE {
            return Ok(());
        }
        let end = (u64::from(addr) + u64::from(len)).min(PHYSICAL_SIZE);
        self.map(device, addr, end)?;
        self.changed.clear();
        let mut page = u64::from(addr & !(PAGE - 1));
        while page < end {
            let p = page as u32;
            if self.slot_of[(p / BLOCK) as usize] == 0 {
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
        // Runs of pages consecutive in one slot buffer go up in one write (each WebGPU call has
        // a fixed cost in browsers).
        let mut i = 0;
        while i < self.changed.len() {
            let first = self.changed[i].0;
            let (chunk, offset) = self.location(first).expect("mapped above");
            let mut n = 1;
            while let Some(&(p, _)) = self.changed.get(i + n)
                && p == first + n as u32 * PAGE
                && self.location(p) == Some((chunk, offset + u64::from(n as u32 * PAGE)))
            {
                n += 1;
            }
            self.scratch.resize(n * PAGE as usize, 0);
            memory.read(first, &mut self.scratch);
            device.queue().write_buffer(&self.chunks[chunk], offset, &self.scratch);
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

    /// Whether the GPU copy holds the only current copy of every page of `ranges` (all written
    /// by the GPU and not rewritten by the CPU since).
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

    /// The slot buffer and offset `[addr, addr + len)` lies at, if it's mirrored in consecutive
    /// slots of one buffer (an index buffer can then be bound from there).
    pub fn contiguous(&self, addr: u32, len: u32) -> Option<(usize, u64)> {
        let (chunk, offset) = self.location(addr)?;
        let last = addr.checked_add(len.max(1) - 1)?;
        let mut block = addr / BLOCK + 1;
        while block <= last / BLOCK {
            let at = block * BLOCK;
            if self.location(at) != Some((chunk, offset + u64::from(at - addr))) {
                return None;
            }
            block += 1;
        }
        Some((chunk, offset))
    }

    /// Encodes copies of physical `[addr, addr + len)` from the slots into `dst` at
    /// `dst_offset`. `false` (nothing encoded) if a block of it isn't mirrored.
    pub fn gather(&self, encoder: &mut CommandEncoder, addr: u32, len: u32, dst: &Buffer, dst_offset: u64) -> bool {
        let end = u64::from(addr) + u64::from(len);
        if end > PHYSICAL_SIZE
            || (addr / BLOCK..=((end.max(1) - 1) / u64::from(BLOCK)) as u32).any(|b| self.slot_of[b as usize] == 0)
        {
            return false;
        }
        let mut pos = u64::from(addr);
        while pos < end {
            let (chunk, offset) = self.location(pos as u32).expect("checked above");
            // As far as the following blocks continue in the same buffer.
            let mut stop = ((pos / u64::from(BLOCK)) + 1) * u64::from(BLOCK);
            while stop < end && self.location(stop as u32) == Some((chunk, offset + (stop - pos))) {
                stop += u64::from(BLOCK);
            }
            let n = stop.min(end) - pos;
            encoder.copy_buffer_to_buffer(&self.chunks[chunk], offset, dst, dst_offset + (pos - u64::from(addr)), n);
            pos += n;
        }
        true
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
