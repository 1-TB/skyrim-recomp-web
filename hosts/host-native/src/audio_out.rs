//! Plays the title's audio on the default output device (cpal: PipeWire/PulseAudio/ALSA).
//!
//! Played XAudio frames (5.1, 48 kHz) are downmixed to stereo into a ring that the device's
//! callback drains, like the web page's ring (hosts/common/audio.rs): each client's samples are
//! added at its own cursor so several clients mix, and the reader zeroes what it consumed.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use kernel_hle::Kernel;
use kernel_hle::audio::{CHANNEL_SAMPLES, FRAME_SAMPLES, MAX_CLIENTS, SAMPLE_RATE};

/// Stereo frames in the ring (1.4 s).
const CAPACITY: usize = 1 << 16;
/// When the producer runs this far ahead of the device (the clocks drift), its frames are
/// dropped rather than adding latency.
const MAX_LATENCY: u64 = 12_000;

struct Ring {
    data: Vec<[f32; 2]>,
    /// Absolute frame positions.
    read: u64,
    write: u64,
    cursors: [Option<u64>; MAX_CLIENTS],
}

impl Ring {
    fn add(&mut self, client: usize, frame: &[f32; FRAME_SAMPLES]) {
        let Some(&cursor) = self.cursors.get(client) else { return };
        // A new client, or one the device overtook (underrun), continues at the reader.
        let at = cursor.unwrap_or(self.read).max(self.read);
        if at - self.read >= MAX_LATENCY {
            return;
        }
        for i in 0..CHANNEL_SAMPLES {
            let (l, r) = downmix(frame, i);
            let cell = &mut self.data[(at as usize + i) % CAPACITY];
            cell[0] += l;
            cell[1] += r;
        }
        let end = at + CHANNEL_SAMPLES as u64;
        self.cursors[client] = Some(end);
        self.write = self.write.max(end);
    }

    /// The next stereo frame, or silence when the producer is behind.
    fn take(&mut self) -> [f32; 2] {
        if self.read >= self.write {
            return [0.0; 2];
        }
        let cell = &mut self.data[self.read as usize % CAPACITY];
        let v = *cell;
        *cell = [0.0; 2];
        self.read += 1;
        v
    }
}

/// 5.1 (front left, front right, center, LFE, rear left, rear right) to stereo, ITU-style.
fn downmix(frame: &[f32; FRAME_SAMPLES], i: usize) -> (f32, f32) {
    let c = |ch: usize| frame[ch * CHANNEL_SAMPLES + i];
    const K: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let center = c(2) * K + c(3) * 0.5;
    ((c(0) + center + c(4) * K).clamp(-1.0, 1.0), (c(1) + center + c(5) * K).clamp(-1.0, 1.0))
}

/// Sends the kernel's audio to the default output device. The stream lives on its own thread
/// (cpal streams may not move between threads); failures are logged and leave the game silent.
pub fn install(kernel: &'static Kernel) {
    let ring =
        Arc::new(Mutex::new(Ring { data: vec![[0.0; 2]; CAPACITY], read: 0, write: 0, cursors: [None; MAX_CLIENTS] }));
    let producer = ring.clone();
    kernel.audio.set_output(Arc::new(move |client, frame| producer.lock().unwrap().add(client, frame)));
    let spawned = std::thread::Builder::new().name("audio-out".into()).spawn(move || match open(ring) {
        Ok(_stream) => loop {
            std::thread::park();
        },
        Err(e) => log::error!("audio output: {e}"),
    });
    if let Err(e) = spawned {
        log::error!("audio output thread: {e}");
    }
}

fn open(ring: Arc<Mutex<Ring>>) -> Result<cpal::Stream, String> {
    let device = cpal::default_host().default_output_device().ok_or("no output device")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!("the device wants {:?} samples; only f32 is supported", supported.sample_format()));
    }
    let config: cpal::StreamConfig = supported.into();
    let channels = usize::from(config.channels);
    let rate = config.sample_rate;
    log::info!("audio output: {channels} channels at {rate} Hz");
    // Linear resampling from the title's 48 kHz to the device rate.
    let step = f64::from(SAMPLE_RATE) / f64::from(rate);
    let (mut prev, mut next, mut phase) = ([0.0f32; 2], [0.0f32; 2], 1.0f64);
    let stream = device
        .build_output_stream(
            config,
            move |out: &mut [f32], _| {
                let mut ring = ring.lock().unwrap();
                for frame in out.chunks_mut(channels) {
                    while phase >= 1.0 {
                        prev = next;
                        next = ring.take();
                        phase -= 1.0;
                    }
                    let t = phase as f32;
                    let l = prev[0] + (next[0] - prev[0]) * t;
                    let r = prev[1] + (next[1] - prev[1]) * t;
                    phase += step;
                    for (ch, sample) in frame.iter_mut().enumerate() {
                        *sample = match ch {
                            0 => l,
                            1 => r,
                            _ => 0.0,
                        };
                    }
                }
            },
            |e| log::warn!("audio output: {e}"),
            None,
        )
        .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}
