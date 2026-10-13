//! The XMA hardware decoder.
//!
//! Titles drive it through XMA_CONTEXT_DATA blocks (64 bytes each, in
//! physical memory): they hand it input packet buffers and an output ring,
//! then kick the context. Decoding (packets, frames, loops, the output ring
//! of big-endian 16-bit PCM) is done by the xma-decoder crate (LGPL: a port
//! of FFmpeg's WMA Pro decoder with Xenia's context logic); this module keeps
//! the context array and per-context decoder state.

use std::sync::Mutex;

use runtime_core::address_space::PAGE_4K;
use runtime_core::heap::protect;

use crate::Kernel;

pub const CONTEXT_COUNT: u32 = 320;
pub const CONTEXT_SIZE: u32 = 64;

/// A bit field of XMA_CONTEXT_DATA: (dword, first bit, width), bits counted
/// from the least significant bit of the big-endian dword.
type Field = (usize, u32, u32);

pub mod field {
    use super::Field;
    pub const INPUT_BUFFER_0_PACKET_COUNT: Field = (0, 0, 12);
    pub const LOOP_COUNT: Field = (0, 12, 8);
    pub const INPUT_BUFFER_0_VALID: Field = (0, 20, 1);
    pub const INPUT_BUFFER_1_VALID: Field = (0, 21, 1);
    pub const OUTPUT_BUFFER_BLOCK_COUNT: Field = (0, 22, 5);
    pub const OUTPUT_BUFFER_WRITE_OFFSET: Field = (0, 27, 5);
    pub const INPUT_BUFFER_1_PACKET_COUNT: Field = (1, 0, 12);
    pub const LOOP_SUBFRAME_END: Field = (1, 14, 3);
    pub const LOOP_SUBFRAME_SKIP: Field = (1, 17, 3);
    pub const SUBFRAME_DECODE_COUNT: Field = (1, 20, 4);
    pub const SAMPLE_RATE: Field = (1, 27, 2);
    pub const IS_STEREO: Field = (1, 29, 1);
    pub const OUTPUT_BUFFER_VALID: Field = (1, 31, 1);
    pub const INPUT_BUFFER_READ_OFFSET: Field = (2, 0, 26);
    pub const ERROR_STATUS: Field = (2, 26, 5);
    pub const LOOP_START: Field = (3, 0, 26);
    pub const LOOP_END: Field = (4, 0, 26);
    pub const PACKET_METADATA: Field = (4, 26, 5);
    pub const CURRENT_BUFFER: Field = (4, 31, 1);
    pub const INPUT_BUFFER_0_PTR: Field = (5, 0, 32);
    pub const INPUT_BUFFER_1_PTR: Field = (6, 0, 32);
    pub const OUTPUT_BUFFER_PTR: Field = (7, 0, 32);
    pub const WORK_BUFFER_PTR: Field = (8, 0, 32);
    pub const OUTPUT_BUFFER_READ_OFFSET: Field = (9, 0, 5);
}

/// XMA_CONTEXT_DATA as host dwords.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ContextData([u32; 16]);

impl ContextData {
    pub fn load(k: &Kernel, ptr: u32) -> Self {
        ContextData(std::array::from_fn(|i| k.mem().read_u32(ptr + i as u32 * 4)))
    }

    pub fn store(&self, k: &Kernel, ptr: u32) {
        for (i, d) in self.0.iter().enumerate() {
            k.mem().write_u32(ptr + i as u32 * 4, *d);
        }
    }

    pub fn get(&self, (dword, shift, width): Field) -> u32 {
        let mask = if width == 32 { u32::MAX } else { (1 << width) - 1 };
        (self.0[dword] >> shift) & mask
    }

    pub fn set(&mut self, (dword, shift, width): Field, value: u32) {
        let mask = if width == 32 { u32::MAX } else { (1 << width) - 1 };
        self.0[dword] = (self.0[dword] & !(mask << shift)) | ((value & mask) << shift);
    }
}

#[derive(Default)]
struct Slot {
    allocated: bool,
    /// Decoder state carried between kicks.
    ctx: xma_decoder::Context,
}

pub struct XmaDecoder {
    /// Guest address of the 320-entry context array (0 until first use).
    array: Mutex<u32>,
    slots: Mutex<Vec<Slot>>,
}

impl Default for XmaDecoder {
    fn default() -> Self {
        XmaDecoder { array: Mutex::new(0), slots: Mutex::new((0..CONTEXT_COUNT).map(|_| Slot::default()).collect()) }
    }
}

/// Guest physical memory for the decoder, mapped page by page (a range may cross from physical
/// allocations into the identity-mapped XEX region).
struct PhysicalMemory<'a>(&'a Kernel);

impl PhysicalMemory<'_> {
    fn each_page(&self, addr: u32, len: usize, mut f: impl FnMut(u32, usize, usize)) -> bool {
        let mut done = 0usize;
        while done < len {
            let phys = addr.wrapping_add(done as u32);
            let n = ((PAGE_4K - phys % PAGE_4K) as usize).min(len - done);
            let Some(guest) = self.0.rt.space.translate_physical(phys) else { return false };
            f(guest, done, n);
            done += n;
        }
        true
    }
}

impl xma_decoder::XmaMemory for PhysicalMemory<'_> {
    fn read(&self, addr: u32, buf: &mut [u8]) -> bool {
        let mem = self.0.mem();
        let len = buf.len();
        self.each_page(addr, len, |guest, offset, n| mem.read_bytes(guest, &mut buf[offset..offset + n]))
    }

    fn write(&mut self, addr: u32, data: &[u8]) -> bool {
        let mem = self.0.mem();
        self.each_page(addr, data.len(), |guest, offset, n| mem.write_bytes(guest, &data[offset..offset + n]))
    }
}

impl Kernel {
    fn xma_array(&self) -> Option<u32> {
        let mut array = self.xma.array.lock().unwrap();
        if *array == 0 {
            match self.rt.space.alloc_physical(
                CONTEXT_COUNT * CONTEXT_SIZE,
                PAGE_4K,
                0,
                u32::MAX,
                256,
                protect::READWRITE,
            ) {
                Ok(a) => *array = a,
                Err(e) => {
                    log::error!("XMA: allocating the context array: {e}");
                    return None;
                }
            }
        }
        Some(*array)
    }

    fn xma_index(&self, ptr: u32) -> Option<usize> {
        let base = *self.xma.array.lock().unwrap();
        (base != 0
            && ptr >= base
            && ptr < base + CONTEXT_COUNT * CONTEXT_SIZE
            && (ptr - base).is_multiple_of(CONTEXT_SIZE))
        .then(|| ((ptr - base) / CONTEXT_SIZE) as usize)
    }

    /// XMACreateContext: a zeroed context, or 0 when all are in use.
    pub fn xma_allocate(&self) -> u32 {
        let Some(base) = self.xma_array() else { return 0 };
        let mut slots = self.xma.slots.lock().unwrap();
        match slots.iter().position(|s| !s.allocated) {
            Some(i) => {
                slots[i].allocated = true;
                slots[i].ctx.reset();
                let ptr = base + i as u32 * CONTEXT_SIZE;
                ContextData::default().store(self, ptr);
                ptr
            }
            None => 0,
        }
    }

    pub fn xma_release(&self, ptr: u32) {
        if let Some(i) = self.xma_index(ptr) {
            let mut slots = self.xma.slots.lock().unwrap();
            slots[i].allocated = false;
            slots[i].ctx.reset();
            ContextData::default().store(self, ptr);
        }
    }

    /// The decoder's reset: inputs and output invalid, offsets rewound.
    pub fn xma_clear(&self, ptr: u32) {
        let Some(i) = self.xma_index(ptr) else { return };
        let mut slots = self.xma.slots.lock().unwrap();
        let mut d = ContextData::load(self, ptr);
        slots[i].ctx.clear(&mut d.0);
        d.store(self, ptr);
    }

    /// A kick: decodes until the output ring is full or input runs out.
    pub fn xma_work(&self, ptr: u32) {
        let Some(i) = self.xma_index(ptr) else { return };
        let mut slots = self.xma.slots.lock().unwrap();
        let slot = &mut slots[i];
        if !slot.allocated {
            return;
        }
        let mut d = ContextData::load(self, ptr);
        slot.ctx.work(&mut d.0, &mut PhysicalMemory(self));
        d.store(self, ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_fields_round_trip() {
        let mut d = ContextData::default();
        d.set(field::OUTPUT_BUFFER_WRITE_OFFSET, 31);
        d.set(field::INPUT_BUFFER_0_VALID, 1);
        d.set(field::INPUT_BUFFER_0_PACKET_COUNT, 0xFFF);
        assert_eq!(d.0[0], (31 << 27) | (1 << 20) | 0xFFF);
        d.set(field::OUTPUT_BUFFER_VALID, 1);
        assert_eq!(d.0[1], 0x8000_0000);
        d.set(field::INPUT_BUFFER_0_PACKET_COUNT, 3);
        assert_eq!(d.get(field::INPUT_BUFFER_0_PACKET_COUNT), 3);
        assert_eq!(d.get(field::OUTPUT_BUFFER_WRITE_OFFSET), 31);
        d.set(field::OUTPUT_BUFFER_PTR, 0x1234_5678);
        assert_eq!(d.get(field::OUTPUT_BUFFER_PTR), 0x1234_5678);
    }
}
