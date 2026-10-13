// Plays the runtime's audio ring (hosts/common/audio.rs) on the audio rendering thread.
// The main thread posts { memory, header } once the runtime is up: `memory` is the wasm
// memory's SharedArrayBuffer and `header` the address returned by skyrt_audio_ring().
class SkyrtAudio extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = null;
    this.port.onmessage = (e) => {
      const { memory, header } = e.data;
      const h = new Uint32Array(memory, header, 4);
      this.ring = {
        header: h,
        positions: new Int32Array(memory, header, 2),
        capacity: h[2],
        data: new Float32Array(memory, h[3], h[2] * 2),
      };
    };
  }

  process(_inputs, outputs) {
    const [left, right] = outputs[0];
    const ring = this.ring;
    if (!ring) return true;
    const write = Atomics.load(ring.positions, 0);
    let read = Atomics.load(ring.positions, 1);
    const available = (write - read) | 0;
    if (available < left.length) {
      // Underrun: keep the buffered audio for the next quantum.
      left.fill(0);
      right?.fill(0);
      return true;
    }
    for (let i = 0; i < left.length; i++) {
      const index = ((read >>> 0) % ring.capacity) * 2;
      left[i] = ring.data[index];
      if (right) right[i] = ring.data[index + 1];
      ring.data[index] = 0;
      ring.data[index + 1] = 0;
      read = (read + 1) | 0;
    }
    Atomics.store(ring.positions, 1, read);
    return true;
  }
}

registerProcessor('skyrt-audio', SkyrtAudio);
