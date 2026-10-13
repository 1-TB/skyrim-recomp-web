//! Audio output for the page: played XAudio frames (5.1, 48 kHz) downmixed to stereo into a ring
//! buffer in wasm memory, which `web/audio-worklet.js` reads on the audio thread.
//!
//! Ring layout (`skyrt_audio_ring` returns the header address): u32 write position, u32 read
//! position (both in stereo frames, wrapping), u32 capacity in frames, u32 address of the
//! interleaved f32 L/R data. The producer adds each client's samples at its own cursor, so
//! several clients mix; the reader zeroes what it consumed.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use kernel_hle::Kernel;
use kernel_hle::audio::{CHANNEL_SAMPLES, FRAME_SAMPLES, MAX_CLIENTS};

/// Stereo frames in the ring (1.4 s); a power of two so wrapping positions index it consistently.
const CAPACITY: u32 = 1 << 16;
/// When the producer runs this far ahead of the reader (its clock drifted, or the page was in
/// the background), the oldest audio is dropped.
const MAX_LATENCY: u32 = 12_000;

#[repr(C)]
struct Header {
    write: AtomicU32,
    read: AtomicU32,
    capacity: u32,
    data: u32,
}

struct Ring {
    header: Box<Header>,
    /// f32 bits (the page reads them as a Float32Array).
    data: Box<[AtomicU32]>,
    /// Per client: the absolute frame position its next frame goes to.
    cursors: Mutex<[Option<u32>; MAX_CLIENTS]>,
}

static RING: OnceLock<Ring> = OnceLock::new();

fn ring() -> &'static Ring {
    RING.get_or_init(|| {
        let data: Box<[AtomicU32]> = (0..CAPACITY * 2).map(|_| AtomicU32::new(0)).collect();
        let header = Box::new(Header {
            write: AtomicU32::new(0),
            read: AtomicU32::new(0),
            capacity: CAPACITY,
            data: data.as_ptr() as usize as u32,
        });
        Ring { header, data, cursors: Mutex::new([None; MAX_CLIENTS]) }
    })
}

/// 5.1 (front left, front right, center, LFE, rear left, rear right) to stereo, ITU-style.
fn downmix(frame: &[f32; FRAME_SAMPLES], i: usize) -> (f32, f32) {
    let c = |ch: usize| frame[ch * CHANNEL_SAMPLES + i];
    const K: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let center = c(2) * K + c(3) * 0.5;
    ((c(0) + center + c(4) * K).clamp(-1.0, 1.0), (c(1) + center + c(5) * K).clamp(-1.0, 1.0))
}

fn play(client: usize, frame: &[f32; FRAME_SAMPLES]) {
    let ring = ring();
    let read = ring.header.read.load(Ordering::Acquire);
    let mut cursors = ring.cursors.lock().unwrap();
    let write = ring.header.write.load(Ordering::Relaxed);
    let Some(slot) = cursors.get_mut(client) else { return };
    // A new client, or one the reader overtook (underrun), continues at the reader. One too far
    // ahead (the page's clock runs slower, or it is in the background) drops the frame.
    let mut at = slot.unwrap_or(read);
    let ahead = at.wrapping_sub(read) as i32;
    if ahead < 0 {
        at = read;
    } else if ahead >= MAX_LATENCY as i32 {
        return;
    }
    // The reader only touches frames before `write`; these are at or past the read position.
    let add = |index: usize, v: f32| {
        let cell = &ring.data[index];
        cell.store((f32::from_bits(cell.load(Ordering::Relaxed)) + v).to_bits(), Ordering::Relaxed);
    };
    for i in 0..CHANNEL_SAMPLES {
        let (l, r) = downmix(frame, i);
        let index = (at.wrapping_add(i as u32) % CAPACITY) as usize * 2;
        add(index, l);
        add(index + 1, r);
    }
    at = at.wrapping_add(CHANNEL_SAMPLES as u32);
    *slot = Some(at);
    if at.wrapping_sub(write) as i32 > 0 {
        ring.header.write.store(at, Ordering::Release);
    }
}

/// Sends the kernel's audio to the page.
pub fn install(kernel: &'static Kernel) {
    ring();
    kernel.audio.set_output(Arc::new(play));
}

/// Address of the ring header, for `web/audio-worklet.js`.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_audio_ring() -> u32 {
    &*ring().header as *const Header as usize as u32
}
