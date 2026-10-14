//! Per-draw data (constants, indices) suballocated from large buffers. Every draw gets its own
//! region, so writes for later draws never clobber data an earlier draw in the same submission
//! still reads; the ring rewinds after each submission.
//!
//! Pushed data is staged on the host and sent with one `Queue::write_buffer` per submission
//! ([`Ring::upload`]): browsers pay a fixed cost per WebGPU call, and a frame pushes thousands
//! of small blocks.

use gpu_webgpu::{Buffer, BufferDescriptor, BufferUsage, Device};

pub struct Ring {
    pub buffer: Buffer,
    size: u64,
    alignment: u64,
    cursor: u64,
    /// Bytes `[0, cursor)` not yet uploaded (alignment gaps zeroed).
    staged: Vec<u8>,
}

impl Ring {
    pub fn new(device: &Device, label: &str, size: u64, usage: BufferUsage, alignment: u64) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: usage | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        Ring { buffer, size, alignment: alignment.max(4), cursor: 0, staged: Vec::new() }
    }

    /// Stages `data` in the ring; returns its offset, or `None` if the ring is full (submit
    /// and `rewind` first). It reaches the GPU at the next [`Ring::upload`].
    pub fn push(&mut self, data: &[u8]) -> Option<u64> {
        let offset = self.cursor.next_multiple_of(self.alignment);
        let len = (data.len() as u64).next_multiple_of(4);
        if offset + len > self.size {
            return None;
        }
        self.staged.resize(offset as usize, 0);
        self.staged.extend_from_slice(data);
        self.staged.resize((offset + len) as usize, 0);
        self.cursor = offset + len;
        Some(offset)
    }

    /// Room for `len` bytes that commands of the submission write, in a ring nothing is pushed
    /// to (pushed data would be uploaded over it); returns its offset, or `None` if the ring is
    /// full.
    pub fn reserve(&mut self, len: u64) -> Option<u64> {
        debug_assert!(self.staged.is_empty(), "reserve in a ring with pushed data");
        let offset = self.cursor.next_multiple_of(self.alignment);
        let end = offset + len.next_multiple_of(4);
        if end > self.size {
            return None;
        }
        self.cursor = end;
        Some(offset)
    }

    /// Like [`Ring::push`], for data bound with a fixed `window` size (≥ the data): the window
    /// must fit in the buffer too. What follows the data in the window is whatever later pushes
    /// store there.
    pub fn push_window(&mut self, data: &[u8], window: u64) -> Option<u64> {
        let offset = self.cursor.next_multiple_of(self.alignment);
        if offset + window.max(data.len() as u64) > self.size {
            return None;
        }
        self.push(data)
    }

    /// Sends everything pushed since the last rewind; call before submitting work that reads it.
    /// Returns the bytes sent.
    pub fn upload(&mut self, device: &Device) -> u64 {
        if !self.staged.is_empty() {
            device.queue().write_buffer(&self.buffer, 0, &self.staged);
        }
        self.staged.len() as u64
    }

    /// Bytes still free (ignoring alignment).
    pub fn remaining(&self) -> u64 {
        self.size - self.cursor
    }

    pub fn rewind(&mut self) {
        self.cursor = 0;
        self.staged.clear();
    }
}
