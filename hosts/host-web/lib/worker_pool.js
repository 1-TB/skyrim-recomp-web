// Emscripten keeps the worker of every thread that exits for a later thread. Each one holds a
// JavaScript engine and an instance of the (huge) module: ~8 MiB in Safari's engine, and Skyrim
// ends ~8 threads while loading. All but a few spares are terminated instead.
Module['preRun'] = [].concat(Module['preRun'] || [], () => {
  if (typeof ENVIRONMENT_IS_PTHREAD !== 'undefined' && ENVIRONMENT_IS_PTHREAD) return;
  const SPARE_WORKERS = 2;
  const returnWorkerToPool = PThread.returnWorkerToPool;
  PThread.returnWorkerToPool = (worker) => {
    returnWorkerToPool(worker);
    while (PThread.unusedWorkers.length > SPARE_WORKERS) {
      const spare = PThread.unusedWorkers.shift();
      spare.onmessage = () => {};
      spare.terminate();
    }
  };
});
