//! Uploads through a small ring of mapped staging buffers instead of `Queue::write_buffer`
//! (opt-in, see [`enable`]). Each upload is copied into a buffer mapped for writing and a copy
//! into its destination is recorded; [`Queue::submit`] sends the copies first, so uploads still
//! take effect before the submission's commands, like `write_buffer`. Browsers copy every
//! `write_buffer` out of the page's memory into a message for the GPU process, while a mapped
//! buffer's memory is set up once and reused: on phones, bursts of uploads otherwise grow the
//! page's process.

use std::cell::RefCell;

use crate::{Buffer, BufferDescriptor, BufferUsage, CommandBuffer, CommandEncoder, Device, MapMode, Pending, Queue};

enum State {
    /// Mapped for writing, `used` bytes taken.
    Mapped,
    /// Unmapped: its copies are submitted.
    InFlight,
    /// Being mapped again (resolves once the GPU is done with it).
    Mapping(Pending<crate::Result<()>>),
}

struct Slot {
    buffer: Buffer,
    used: u64,
    state: State,
}

struct Ring {
    device: Device,
    slots: Vec<Slot>,
    size: u64,
    current: usize,
    /// Copies recorded since the last submission.
    encoder: Option<CommandEncoder>,
}

thread_local! {
    static RING: RefCell<Option<Ring>> = const { RefCell::new(None) };
}

/// Makes this thread's uploads go through `slots` staging buffers of `size` bytes (uploads
/// larger than that, or when none has room, still use `write_buffer`).
pub fn enable(device: &Device, slots: usize, size: u64) {
    let size = size.next_multiple_of(4);
    let slots = (0..slots.max(1))
        .map(|_| Slot {
            buffer: device.create_buffer(&BufferDescriptor {
                label: "upload staging",
                size,
                usage: BufferUsage::MapWrite | BufferUsage::CopySrc,
                mapped_at_creation: true,
            }),
            used: 0,
            state: State::Mapped,
        })
        .collect();
    RING.with(|r| *r.borrow_mut() = Some(Ring { device: device.clone(), slots, size, current: 0, encoder: None }));
}

/// Stages an upload of `data` to `dst` at `offset`; `false` when it must go another way.
pub(crate) fn stage(queue: &Queue, dst: &Buffer, offset: u64, data: &[u8]) -> bool {
    RING.with(|r| {
        let mut r = r.borrow_mut();
        let Some(ring) = r.as_mut() else { return false };
        let len = data.len() as u64;
        let found = if len <= ring.size {
            ring.poll();
            let n = ring.slots.len();
            (0..n)
                .map(|i| (ring.current + i) % n)
                .find(|&i| matches!(ring.slots[i].state, State::Mapped) && ring.slots[i].used + len <= ring.size)
        } else {
            None
        };
        let Some(i) = found else {
            // A direct write would land before the copies staged so far: send them first.
            if let Some(encoder) = ring.encoder.take() {
                let commands = encoder.finish("upload staging");
                ring.unmap_used();
                queue.submit_raw([commands]);
                ring.remap();
            }
            return false;
        };
        let slot = &mut ring.slots[i];
        if slot.buffer.write_mapped(slot.used, data).is_err() {
            return false;
        }
        let device = &ring.device;
        let encoder = ring.encoder.get_or_insert_with(|| device.create_command_encoder("upload staging"));
        encoder.copy_buffer_to_buffer(&slot.buffer, slot.used, dst, offset, len);
        slot.used += len;
        ring.current = i;
        true
    })
}

/// The staged copies for the submission being made (they go before its commands).
pub(crate) fn take_copies() -> Option<CommandBuffer> {
    RING.with(|r| {
        let mut r = r.borrow_mut();
        let ring = r.as_mut()?;
        let commands = ring.encoder.take()?.finish("upload staging");
        ring.unmap_used();
        Some(commands)
    })
}

/// After a submission: the staging buffers it read get mapped again.
pub(crate) fn submitted() {
    RING.with(|r| {
        if let Some(ring) = r.borrow_mut().as_mut() {
            ring.remap();
        }
    });
}

impl Ring {
    fn unmap_used(&mut self) {
        for slot in &mut self.slots {
            if matches!(slot.state, State::Mapped) && slot.used > 0 {
                slot.buffer.unmap();
                slot.state = State::InFlight;
            }
        }
    }

    fn remap(&mut self) {
        for slot in &mut self.slots {
            if matches!(slot.state, State::InFlight) {
                slot.state = State::Mapping(slot.buffer.map_async(MapMode::Write, 0, None));
            }
        }
    }

    fn poll(&mut self) {
        for slot in &mut self.slots {
            if let State::Mapping(pending) = &slot.state
                && let Some(result) = pending.try_take()
            {
                match result {
                    Ok(()) => {
                        slot.state = State::Mapped;
                        slot.used = 0;
                    }
                    // Mapping failed (device lost): map again, it stays out of use meanwhile.
                    Err(_) => slot.state = State::InFlight,
                }
            }
        }
    }
}
