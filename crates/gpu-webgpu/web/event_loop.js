// Scheduling for gpu_webgpu::web::run_event_loop (link with --js-library). A MessageChannel
// task runs the next tick as soon as the worker's event loop is free (setTimeout(0) is clamped
// to 4 ms and Emscripten's setImmediate polyfill posts to the parent thread from workers).
addToLibrary({
  gpu_webgpu_loop_keepalive__deps: ['$runtimeKeepalivePush', '$runtimeKeepalivePop'],
  gpu_webgpu_loop_keepalive: (start) => {
    if (start) runtimeKeepalivePush();
    else runtimeKeepalivePop();
  },
  gpu_webgpu_schedule__deps: ['$getWasmTableEntry', '$callUserCallback'],
  gpu_webgpu_schedule: (func, arg, delayMs) => {
    const run = () => callUserCallback(() => getWasmTableEntry(func)(arg));
    if (delayMs > 0) {
      setTimeout(run, delayMs);
      return;
    }
    let channel = globalThis.__gpuWebgpuChannel;
    if (!channel) {
      channel = globalThis.__gpuWebgpuChannel = new MessageChannel();
      channel.queue = [];
      channel.port1.onmessage = () => channel.queue.shift()?.();
    }
    channel.queue.push(run);
    channel.port2.postMessage(0);
  },
  gpu_webgpu_schedule_on_change__deps: ['gpu_webgpu_schedule'],
  gpu_webgpu_schedule_on_change: (func, arg, addr, value, timeoutMs) => {
    // Wakes on Atomics.notify (an Emscripten futex wake) of the word, without blocking the
    // worker's event loop.
    const result = Atomics.waitAsync(HEAP32, addr >> 2, value | 0, timeoutMs);
    if (result.async) {
      result.value.then(() => _gpu_webgpu_schedule(func, arg, 0));
    } else {
      _gpu_webgpu_schedule(func, arg, 0);
    }
  },
});
