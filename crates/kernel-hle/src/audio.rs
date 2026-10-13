//! The XAudio render driver: titles register a client callback, and the
//! system asks each client for one frame (256 samples x 6 channels, 48 kHz)
//! every 5.33 ms while it has room in its queue. Frames drain at real-time
//! rate into an optional host output. Behavior follows Xenia's AudioSystem.

use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

use runtime_core::PPCContext;

use crate::Kernel;

pub const MAX_CLIENTS: usize = 8;
pub const CHANNELS: usize = 6;
pub const CHANNEL_SAMPLES: usize = 256;
pub const FRAME_SAMPLES: usize = CHANNELS * CHANNEL_SAMPLES;
pub const SAMPLE_RATE: u32 = 48_000;
/// Frames a client may have queued ahead of playback.
const MAX_QUEUED_FRAMES: u32 = 64;
/// One frame of audio: 256 samples at 48 kHz.
const FRAME_PERIOD: Duration = Duration::from_nanos(1_000_000_000 * CHANNEL_SAMPLES as u64 / SAMPLE_RATE as u64);

/// Receives every played frame: channel-major (6 blocks of 256 samples:
/// front left, front right, center, LFE, rear left, rear right).
pub type AudioOutput = Arc<dyn Fn(usize, &[f32; FRAME_SAMPLES]) + Send + Sync>;

/// Driver handle given to the title (Xenia's encoding).
const DRIVER_TAG: u32 = 0x4155_0000;

struct Client {
    callback: u32,
    /// Guest word holding the title's callback argument; its address is
    /// what the callback receives.
    wrapped_arg: u32,
    /// Frames submitted and not yet played.
    queued: u32,
    next_pump: Instant,
    pending: std::collections::VecDeque<Box<[f32; FRAME_SAMPLES]>>,
}

#[derive(Default)]
pub struct AudioSystem {
    clients: Mutex<[Option<Client>; MAX_CLIENTS]>,
    worker: Once,
    output: OnceLock<AudioOutput>,
}

impl AudioSystem {
    /// Installs the host's output (once; later calls are ignored).
    pub fn set_output(&self, output: AudioOutput) {
        let _ = self.output.set(output);
    }

    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().iter().filter(|c| c.is_some()).count()
    }
}

pub fn driver_handle(index: usize) -> u32 {
    DRIVER_TAG | index as u32
}

pub fn client_index(driver: u32) -> Option<usize> {
    (driver & 0xFFFF_0000 == DRIVER_TAG).then_some((driver & 0xFFFF) as usize).filter(|&i| i < MAX_CLIENTS)
}

impl Kernel {
    /// XAudioRegisterRenderDriverClient. Returns the client index.
    pub fn register_audio_client(&'static self, callback: u32, callback_arg: u32) -> Option<usize> {
        let wrapped_arg = self.rt.space.system_alloc(4, 4).ok()?;
        self.mem().write_u32(wrapped_arg, callback_arg);
        let index = {
            let mut clients = self.audio.clients.lock().unwrap();
            let index = clients.iter().position(Option::is_none)?;
            clients[index] = Some(Client {
                callback,
                wrapped_arg,
                queued: 0,
                next_pump: Instant::now(),
                pending: Default::default(),
            });
            index
        };
        log::info!("audio: client {index} registered (callback {callback:#010X}, arg {callback_arg:#010X})");
        self.audio.worker.call_once(|| self.spawn_kernel_thread("audio", audio_worker_main));
        Some(index)
    }

    pub fn unregister_audio_client(&self, index: usize) {
        let removed = self.audio.clients.lock().unwrap().get_mut(index).and_then(Option::take);
        if let Some(c) = removed {
            let _ = self.rt.space.system_free(c.wrapped_arg);
        }
    }

    /// XAudioSubmitRenderDriverFrame: copies the title's big-endian samples.
    pub fn submit_audio_frame(&self, index: usize, samples: u32) {
        let mem = self.mem();
        let mut frame = Box::new([0f32; FRAME_SAMPLES]);
        for (i, s) in frame.iter_mut().enumerate() {
            *s = mem.read_f32(samples + i as u32 * 4);
        }
        let mut clients = self.audio.clients.lock().unwrap();
        if let Some(Some(c)) = clients.get_mut(index) {
            c.queued += 1;
            c.pending.push_back(frame);
        }
    }
}

fn audio_worker_main(k: &'static Kernel, ctx: &mut PPCContext) {
    let start = Instant::now();
    let mut played_until = start;
    loop {
        // Play frames that are due, one per period, freeing queue slots.
        let now = Instant::now();
        let mut finished: Vec<(usize, Box<[f32; FRAME_SAMPLES]>)> = Vec::new();
        while played_until + FRAME_PERIOD <= now {
            played_until += FRAME_PERIOD;
            let mut clients = k.audio.clients.lock().unwrap();
            for (i, c) in clients.iter_mut().enumerate() {
                if let Some(c) = c
                    && let Some(f) = c.pending.pop_front()
                {
                    c.queued -= 1;
                    finished.push((i, f));
                }
            }
        }
        if let Some(out) = k.audio.output.get() {
            for (i, f) in &finished {
                out(*i, f);
            }
        }

        // Ask the most overdue client with room for another frame.
        let due = {
            let mut clients = k.audio.clients.lock().unwrap();
            let pick = clients
                .iter_mut()
                .enumerate()
                .filter_map(|(i, c)| c.as_mut().map(|c| (i, c)))
                .filter(|(_, c)| c.queued < MAX_QUEUED_FRAMES)
                .min_by_key(|(_, c)| c.next_pump);
            pick.and_then(|(_, c)| {
                (c.next_pump <= now).then(|| {
                    c.next_pump = c.next_pump.max(now.checked_sub(FRAME_PERIOD).unwrap_or(now)) + FRAME_PERIOD;
                    (c.callback, c.wrapped_arg)
                })
            })
        };
        match due {
            Some((callback, arg)) => {
                let saved = ctx.clone();
                ctx.set_gpr(3, arg as u64);
                if let Err(e) = k.rt.call(ctx, callback) {
                    log::error!("audio client callback {callback:#010X}: {e}");
                }
                *ctx = saved;
            }
            None => std::thread::sleep(Duration::from_millis(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_handles_round_trip() {
        assert_eq!(driver_handle(3), 0x4155_0003);
        assert_eq!(client_index(0x4155_0003), Some(3));
        assert_eq!(client_index(0x1234_0003), None);
        assert_eq!(client_index(0x4155_0009), None);
        assert_eq!(FRAME_PERIOD.as_micros(), 5333);
    }
}
