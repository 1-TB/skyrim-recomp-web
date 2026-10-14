// Runs the web build in Chromium with WebGPU and a local game folder, for automated testing.
//
// Serves the build with COOP/COEP plus the game folder under /__game/ (127.0.0.1 only), copies
// the game into the page's OPFS once (the browser profile persists; files already present with
// the same size are skipped), boots host-web.js with the canvas, and for `--seconds` writes the
// runtime log and periodic canvas screenshots to `--out`.
//
// usage: PW_DIR=<dir with node_modules/playwright> xvfb-run -a node browser-run.mjs
//          --dist web-dist --game <game dir> --profile <browser profile dir> --out <output dir>
//          [--seconds 120] [--shot-every 10] [--args "--log info"] [--port 8091] [--reload-after S]
//          [--stacks-at S[,S...]] [--audio] [--profile-at S:SECONDS] [--stream] [--saves DIR]
//          [--step "CONDITION => ACTIONS" ...] [--step-timeout S] [--allocators]
//
// `--shot-every 0` turns the periodic screenshots off. `--allocators` adds Chromium's memory-infra
// allocator sizes per process to each measurement (slow; only for finding where memory goes).
//
// `--stream` skips the OPFS import and has the runtime stream the game from this server instead
// (`--stream-game`), through the same /game/manifest and ranged /game/files/ endpoints as
// `cargo xtask serve --game-dir`.
//
// `--saves DIR` replaces the page's title content (OPFS content/, where saves live) with DIR
// before booting. DIR has the layout of `cargo xtask serve`'s target/web-saves/ (what the
// page's Export saves button writes), so an exported save can be loaded with taps on Continue.
//
// `--audio` plays the runtime's audio through web/audio-worklet.js like the page does and logs
// the ring positions (both advancing means audio flows). `--profile-at S:D` runs the sampling
// profiler in every worker for D seconds starting S seconds after booting and logs each
// thread's hottest functions (self time).
//
// `--reload-after S` reloads the page S seconds after booting and boots again (as a user
// refreshing mid-game would). `--stacks-at S` pauses every worker (through the DevTools
// protocol) S seconds after booting and logs their JS/wasm call stacks, for finding hangs.
//
// Without a GPU, Chromium's WebGPU runs on SwiftShader (see CHROME_ARGS). Headless Chromium
// never composites a worker-owned OffscreenCanvas, so run headed (on Xvfb) for screenshots.
//
// Scripted runs: each `--step "CONDITION => ACTION; ACTION..."` waits for its condition, then
// runs its actions; steps run in order and the run ends after the last one (with exit status 1
// if a step's condition doesn't hold within `--step-timeout` seconds, default 300, or the run
// exceeds `--seconds`). Conditions, joined with `&` (empty: at once):
//   log:REGEX          a log line since the step became active matches (JS regex)
//   fps>=N, draws>=N   the newest `perf:` report since the step became active (also <=, >, <)
//   swaps>=N           frames presented since boot
//   after:S            S seconds since the step became active
// Actions: `tap BUTTON` (port 1, held for 200 ms and two presented frames), `tap BUTTON every S`
// (repeats until the next step's condition holds), `shot NAME`, `measure S LABEL` (collects the
// perf reports and memory use for S seconds into the summary), `wait S`, `gc` (collects garbage in
// the page and every worker), `exit`. E.g.
//   --step "log:XamContentCreateEnumerator => measure 15 title" --step "=> exit"
// The summary of every `measure` is printed at the end and written to OUT/summary.json.
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
import { appendFileSync, createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { extname, join, relative, resolve, sep } from 'node:path';

function option(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  if (fallback === undefined) {
    console.error(`missing --${name}`);
    process.exit(2);
  }
  return fallback;
}

function options(name) {
  const values = [];
  process.argv.forEach((a, i) => {
    if (a === `--${name}` && i + 1 < process.argv.length) values.push(process.argv[i + 1]);
  });
  return values;
}

const XINPUT_BUTTONS = {
  DPAD_UP: 0x0001, DPAD_DOWN: 0x0002, DPAD_LEFT: 0x0004, DPAD_RIGHT: 0x0008, UP: 0x0001, DOWN: 0x0002,
  LEFT: 0x0004, RIGHT: 0x0008, START: 0x0010, BACK: 0x0020, LEFT_THUMB: 0x0040, RIGHT_THUMB: 0x0080,
  LB: 0x0100, RB: 0x0200, A: 0x1000, B: 0x2000, X: 0x4000, Y: 0x8000,
};

/// Parses `CONDITION => ACTION; ACTION`.
function parseStep(text) {
  const [condText, actionText = ''] = text.split('=>').map((p) => p.trim());
  const conditions = condText.split('&').map((c) => c.trim()).filter(Boolean).map((c) => {
    if (c.startsWith('log:')) return { kind: 'log', re: new RegExp(c.slice(4)) };
    if (c.startsWith('after:')) return { kind: 'after', seconds: Number(c.slice(6)) };
    const m = /^(fps|draws|swaps)\s*(>=|<=|>|<)\s*([\d.]+)$/.exec(c);
    if (!m) throw new Error(`--step ${text}: unknown condition ${c}`);
    return { kind: m[1], op: m[2], value: Number(m[3]) };
  });
  const actions = actionText.split(';').map((a) => a.trim()).filter(Boolean).map((a) => {
    const w = a.split(/\s+/);
    switch (w[0]) {
      case 'tap': {
        const mask = XINPUT_BUTTONS[w[1]?.toUpperCase()];
        if (mask === undefined) throw new Error(`--step ${text}: unknown button ${w[1]}`);
        return { kind: 'tap', name: w[1].toUpperCase(), mask, every: w[2] === 'every' ? Number(w[3]) : 0 };
      }
      case 'shot': return { kind: 'shot', name: w[1] ?? 'shot' };
      case 'measure': return { kind: 'measure', seconds: Number(w[1]), label: w[2] ?? `measure${w[1]}` };
      case 'wait': return { kind: 'wait', seconds: Number(w[1]) };
      case 'gc': return { kind: 'gc' };
      case 'exit': return { kind: 'exit' };
      default: throw new Error(`--step ${text}: unknown action ${a}`);
    }
  });
  return { text, conditions, actions };
}

const dist = resolve(option('dist'));
const game = resolve(option('game'));
const profile = resolve(option('profile'));
const out = resolve(option('out'));
const seconds = Number(option('seconds', '120'));
const shotEvery = Number(option('shot-every', '10'));
const runtimeArgs = option('args', '--log info').split(/\s+/).filter(Boolean);
if (process.argv.includes('--stream')) runtimeArgs.push('--stream-game', `http://127.0.0.1:${option('port', '8091')}/game/`);
// Fixed so the origin (and with it the imported game in OPFS) is the same on every run.
const port = Number(option('port', '8091'));
const reloadAfter = Number(option('reload-after', '0'));
const audio = process.argv.includes('--audio');
const stream = process.argv.includes('--stream');
const [profileAt, profileFor] = option('profile-at', '0:0').split(':').map(Number);
const stacksAt = option('stacks-at', '').split(',').filter(Boolean).map(Number);
const saves = process.argv.includes('--saves') ? resolve(option('saves')) : null;
const steps = options('step').map(parseStep);
const stepTimeout = Number(option('step-timeout', '300'));
const allocatorDumps = process.argv.includes('--allocators');
mkdirSync(out, { recursive: true });
const logPath = join(out, 'log.txt');
writeFileSync(logPath, '');

const require = createRequire(join(process.env.PW_DIR ?? '.', 'noop.js'));
const { chromium } = require('playwright');

function listFiles(dir, base = dir) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...listFiles(path, base));
    else if (entry.isFile()) files.push({ path: relative(base, path).split(sep).join('/'), size: statSync(path).size });
  }
  return files;
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'Cache-Control': 'no-cache',
};
const server = createServer((req, res) => {
  const path = decodeURIComponent(req.url.split('?')[0]);
  if (path === '/__game/manifest.json') {
    res.writeHead(200, { 'Content-Type': 'application/json', ...isolation }).end(JSON.stringify(listFiles(game)));
    return;
  }
  if (path === '/__saves/manifest.json' && saves) {
    res.writeHead(200, { 'Content-Type': 'application/json', ...isolation }).end(JSON.stringify(listFiles(saves)));
    return;
  }
  if (path === '/game/manifest') {
    const lines = listFiles(game).map((f) => `${f.size}\t${f.path}\n`).join('');
    res.writeHead(200, { 'Content-Type': 'text/plain', ...isolation }).end(lines);
    return;
  }
  if (path.startsWith('/game/files/')) {
    const file = resolve(game, path.slice('/game/files/'.length));
    if (!file.startsWith(game + sep) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404, isolation).end();
      return;
    }
    const size = statSync(file).size;
    const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
    const start = range ? Number(range[1]) : 0;
    const end = range && range[2] ? Math.min(Number(range[2]) + 1, size) : size;
    if (start >= end) {
      res.writeHead(416, { 'Content-Range': `bytes */${size}`, ...isolation }).end();
      return;
    }
    res.writeHead(range ? 206 : 200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': end - start,
      ...(range ? { 'Content-Range': `bytes ${start}-${end - 1}/${size}` } : {}),
      ...isolation,
    });
    createReadStream(file, { start, end: end - 1 }).pipe(res);
    return;
  }
  const [root, name] = path.startsWith('/__game/')
    ? [game, path.slice(8)]
    : saves && path.startsWith('/__saves/')
      ? [saves, path.slice(9)]
      : [dist, path.slice(1) || 'index.html'];
  const file = resolve(root, name);
  if (!file.startsWith(root + sep) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, {
    'Content-Type': types[extname(file)] ?? 'application/octet-stream',
    'Content-Length': statSync(file).size,
    ...isolation,
  });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(port, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;

const args = (process.env.CHROME_ARGS ??
  '--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=swiftshader --use-webgpu-adapter=swiftshader')
  .split(/\s+/).filter(Boolean);
if (audio) args.push('--autoplay-policy=no-user-gesture-required');
if (allocatorDumps) args.push('--remote-debugging-port=0');
const context = await chromium.launchPersistentContext(profile, {
  channel: 'chromium',
  headless: !!process.env.HEADLESS,
  args,
  viewport: { width: 1400, height: 1100 },
});
// Worker errors reach the page only as opaque ErrorEvents: log their details from the Worker.
await context.addInitScript(() => {
  const Base = globalThis.Worker;
  globalThis.Worker = class extends Base {
    constructor(...args) {
      super(...args);
      this.addEventListener('error', (e) => console.error(`worker error: ${e.message} (${e.filename}:${e.lineno}:${e.colno})\n${e.error?.stack ?? ''}`));
    }
  };
});
const page = context.pages()[0] ?? (await context.newPage());
/// The runtime's log so far, for step conditions and measurements.
const lines = [];
const note = (line) => {
  console.log(line);
  appendFileSync(logPath, line + '\n');
};
page.on('pageerror', (e) => note(`PAGEERROR ${e.message}${e.stack ? `\n${e.stack}` : ''}`));
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') note(`console.${m.type()}: ${m.text()}`);
});
await page.goto(url);
note(`crossOriginIsolated=${await page.evaluate(() => crossOriginIsolated)} webgpu=${await page.evaluate(() => !!navigator.gpu)}`);

// Import (incremental).
const imported = stream ? { files: 0, copied: 0 } : await page.evaluate(async () => {
  const manifest = await (await fetch('/__game/manifest.json')).json();
  const root = await navigator.storage.getDirectory();
  const gameDir = await root.getDirectoryHandle('game', { create: true });
  let copied = 0;
  for (const { path, size } of manifest) {
    const parts = path.split('/');
    let dir = gameDir;
    for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create: true });
    const handle = await dir.getFileHandle(parts.at(-1), { create: true });
    if ((await handle.getFile()).size === size) continue;
    const response = await fetch('/__game/' + parts.map(encodeURIComponent).join('/'));
    await response.body.pipeTo(await handle.createWritable());
    copied += size;
  }
  return { files: manifest.length, copied };
});
note(`game: ${imported.files} files, ${(imported.copied / 2 ** 20).toFixed(1)} MiB copied into OPFS`);

if (saves) {
  const count = await page.evaluate(async () => {
    const manifest = await (await fetch('/__saves/manifest.json')).json();
    const root = await navigator.storage.getDirectory();
    await root.removeEntry('content', { recursive: true }).catch(() => {});
    const content = await root.getDirectoryHandle('content', { create: true });
    for (const { path } of manifest) {
      const parts = path.split('/');
      let dir = content;
      for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create: true });
      const response = await fetch('/__saves/' + parts.map(encodeURIComponent).join('/'));
      await response.body.pipeTo(await (await dir.getFileHandle(parts.at(-1), { create: true })).createWritable());
    }
    return manifest.length;
  });
  note(`saves: ${count} files from ${saves} copied into OPFS content/`);
}

// Boot.
const boot = () => page.evaluate(async ([argv, withAudio]) => {
  window.__lines = [];
  let audioNode = null;
  if (withAudio) {
    const context = new AudioContext({ sampleRate: 48000 });
    await context.audioWorklet.addModule('audio-worklet.js');
    audioNode = new AudioWorkletNode(context, 'skyrt-audio', { outputChannelCount: [2] });
    audioNode.connect(context.destination);
    await context.resume();
  }
  window.Module = {
    canvas: document.getElementById('canvas'),
    arguments: argv,
    print: (l) => window.__lines.push(l),
    printErr: (l) => window.__lines.push(l),
    onRuntimeInitialized() {
      if (!audioNode) return;
      window.__audioHeader = window.Module._skyrt_audio_ring();
      audioNode.port.postMessage({ memory: window.Module.HEAPU8.buffer, header: window.__audioHeader });
    },
  };
  const s = document.createElement('script');
  s.src = 'host-web.js';
  document.body.appendChild(s);
}, [runtimeArgs, audio]);
await boot();
note(`booted with ${runtimeArgs.join(' ')}`);

const canvas = page.locator('#canvas');
await canvas.scrollIntoViewIfNeeded();
const start = Date.now();
const elapsed = () => (Date.now() - start) / 1000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let nextShot = shotEvery;
let reloaded = false;
let stacksDumped = 0;
let profiled = false;
let lastAudio = 0;

/// Moves the runtime's log into `lines` and runs the time-based options; called every 250 ms.
async function pump() {
  for (const l of await page.evaluate(() => window.__lines.splice(0))) {
    lines.push(l);
    note(l);
  }
  const t = elapsed();
  if (audio && t - lastAudio >= 5) {
    lastAudio = t;
    const positions = await page.evaluate(() => {
      if (!window.__audioHeader) return null;
      const view = new Uint32Array(window.Module.HEAPU8.buffer, window.__audioHeader, 2);
      return [view[0], view[1]];
    });
    if (positions) note(`audio ring: written ${positions[0]} frames, played ${positions[1]}`);
  }
  if (reloadAfter > 0 && !reloaded && t >= reloadAfter) {
    reloaded = true;
    note('reloading the page');
    await page.reload();
    await boot();
    note('booted again after reload');
  }
  if (profileAt > 0 && !profiled && t >= profileAt) {
    profiled = true;
    await profileWorkers(profileFor);
  }
  if (stacksDumped < stacksAt.length && t >= stacksAt[stacksDumped]) {
    stacksDumped += 1;
    await dumpWorkerStacks();
  }
  if (shotEvery > 0 && t >= nextShot) {
    await screenshot(`canvas_${String(Math.round(t)).padStart(4, '0')}s.png`);
    nextShot += shotEvery;
  }
}

async function pumpFor(seconds) {
  const until = Date.now() + seconds * 1000;
  while (Date.now() < until) {
    await sleep(Math.min(250, Math.max(0, until - Date.now())));
    await pump();
  }
}

async function screenshot(name) {
  const shot = join(out, name.endsWith('.png') ? name : `${name}.png`);
  await canvas.screenshot({ path: shot, timeout: 30000 }).then(
    () => note(`screenshot ${shot}`),
    (e) => note(`screenshot failed: ${e.message.split('\n')[0]}`),
  );
}

const swapCount = () => page.evaluate(() => window.Module?._skyrt_swap_count?.() ?? 0);
const setPad = (mask) => page.evaluate((m) => window.Module._skyrt_set_pad(0, m, 0, 0, 0, 0, 0, 0), mask);

/// Presses `mask` on port 1 for 200 ms and at least two presented frames (titles poll input once
/// a frame), giving up on the frames after 5 s, like `--tap`.
async function tap(name, mask) {
  note(`[${elapsed().toFixed(1)} s] step: tap ${name}`);
  const frames = await swapCount();
  await setPad(mask);
  const pressed = Date.now();
  while (Date.now() - pressed < 200 || ((await swapCount()) < frames + 2 && Date.now() - pressed < 5000)) {
    await sleep(20);
  }
  await setPad(0);
}

const PERF = /perf: ([\d.]+) fps, (\d+) draws\/frame; renderer thread: (\d+)% executing, (\d+)% waiting for the GPU, (\d+)% waiting for the game/;
const PER_FRAME = /perf: per frame: .*?(\d+) submits, (\d+) write_buffer \((\d+) KiB\), (\d+) write_texture/;
const GPU_MEMORY = /perf: GPU memory ([\d.]+) MiB in (\d+) objects; texture cache ([\d.]+) MiB in (\d+) textures; (.*)/;

/// This browser's processes: the one started with the profile and its descendants, with each
/// one's `--type` (the page's renderer, the GPU process, ...; none for the browser itself).
function browserProcesses() {
  const procs = [];
  for (const pid of readdirSync('/proc').filter((d) => /^\d+$/.test(d))) {
    try {
      const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
      const ppid = stat.slice(stat.lastIndexOf(')') + 2).split(' ')[1];
      // Chromium rewrites its children's command lines into one space-separated string.
      procs.push({ pid, ppid, cmd: readFileSync(`/proc/${pid}/cmdline`, 'utf8').split(/[\0 ]+/) });
    } catch {
      // The process exited meanwhile.
    }
  }
  const root = (p) => p.pid !== String(process.pid) && p.cmd.some((a) => a.includes(profile)) && !p.cmd.some((a) => a.startsWith('--type='));
  const ours = new Set(procs.filter(root).map((p) => p.pid));
  for (let grew = true; grew; ) {
    grew = false;
    for (const p of procs) {
      if (!ours.has(p.pid) && ours.has(p.ppid)) {
        ours.add(p.pid);
        grew = true;
      }
    }
  }
  return procs
    .filter((p) => ours.has(p.pid))
    .map((p) => ({ pid: p.pid, type: (p.cmd.find((a) => a.startsWith('--type=')) ?? '--type=browser').slice(7) }));
}

/// Proportional set size (KiB) of this browser's processes by type, from /proc.
function processMemory() {
  const byType = {};
  for (const { pid, type } of browserProcesses()) {
    try {
      const pss = /^Pss:\s+(\d+) kB/m.exec(readFileSync(`/proc/${pid}/smaps_rollup`, 'utf8'));
      byType[type] = (byType[type] ?? 0) + Number(pss?.[1] ?? 0);
    } catch {
      // Gone.
    }
  }
  return byType;
}

/// The largest kinds of mappings (by PSS) of this browser's renderer and GPU processes: which of the wasm
/// memory, compiled code and driver allocations hold the memory.
function largestMappings() {
  const result = [];
  for (const { pid, type } of browserProcesses().filter((p) => p.type === 'renderer' || p.type === 'gpu-process')) {
    try {
      let current = null;
      for (const line of readFileSync(`/proc/${pid}/smaps`, 'utf8').split('\n')) {
        const head = /^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)$/.exec(line);
        if (head) {
          current = { pid, type, sizeMiB: (parseInt(head[2], 16) - parseInt(head[1], 16)) / 2 ** 20, perms: head[3], name: head[4], pssMiB: 0 };
          result.push(current);
        } else if (current && line.startsWith('Pss:')) {
          current.pssMiB = Number(/(\d+)/.exec(line)[1]) / 1024;
        }
      }
    } catch {
      // Gone.
    }
  }
  // Grouped by process type and mapping name (anonymous memory comes in many mappings).
  const groups = new Map();
  for (const m of result) {
    const key = `${m.type} ${m.perms} ${m.name}`;
    const g = groups.get(key) ?? { type: m.type, perms: m.perms, name: m.name, mappings: 0, sizeMiB: 0, pssMiB: 0 };
    g.mappings += 1;
    g.sizeMiB += m.sizeMiB;
    g.pssMiB += m.pssMiB;
    groups.set(key, g);
  }
  return [...groups.values()]
    .sort((a, b) => b.pssMiB - a.pssMiB)
    .slice(0, 12)
    .map((g) => ({ ...g, sizeMiB: Math.round(g.sizeMiB), pssMiB: Math.round(g.pssMiB) }));
}

/// Chromium's own accounting of its processes' memory (a memory-infra dump through DevTools
/// tracing): the largest allocators by size in each process, e.g. `partition_alloc/partitions/
/// array_buffer`, `gpu/dawn`, `v8/main/heap`.
async function allocatorDump() {
  // Memory dumps need the browser target, which Playwright's persistent contexts don't expose:
  // talk to it over the remote debugging port (written to DevToolsActivePort in the profile).
  const [port, path] = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').trim().split('\n');
  const ws = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = rej;
  });
  const events = [];
  const replies = new Map();
  let complete;
  const completed = new Promise((r) => (complete = r));
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id) replies.get(msg.id)?.(msg);
    else if (msg.method === 'Tracing.dataCollected') events.push(...msg.params.value);
    else if (msg.method === 'Tracing.tracingComplete') complete();
  };
  let nextId = 1;
  const send = (method, params = {}) =>
    new Promise((res) => {
      const id = nextId++;
      replies.set(id, res);
      ws.send(JSON.stringify({ id, method, params }));
    });
  await send('Tracing.start', {
    traceConfig: { includedCategories: ['disabled-by-default-memory-infra'], memoryDumpConfig: { triggers: [] } },
  });
  // Every process must have started tracing, or the dump fails.
  await sleep(2000);
  const dump = await send('Tracing.requestMemoryDump', { deterministic: true, levelOfDetail: 'detailed' });
  await send('Tracing.end');
  await completed;
  ws.close();
  // `success` is false when any process failed to dump; the others' dumps are still there.
  const types = Object.fromEntries(browserProcesses().map((p) => [String(p.pid), p.type]));
  const byProcess = {};
  for (const e of events) {
    const allocators = e.ph === 'v' && e.args?.dumps?.allocators;
    if (!allocators) continue;
    const rows = Object.entries(allocators)
      .map(([name, a]) => [name, parseInt(a.attrs?.size?.value ?? '0', 16)])
      // Shallow names only: deeper ones repeat their parents' bytes.
      .filter(([name, size]) => size >= 8 << 20 && name.split('/').length <= 3)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([name, size]) => `${name} ${(size / 2 ** 20).toFixed(0)} MiB`);
    byProcess[`${types[String(e.pid)] ?? 'other'} ${e.pid}`] = rows;
  }
  if (!dump.result?.success) byProcess.incomplete = true;
  return byProcess;
}

/// GPU memory the driver attributes to this browser's processes (NVIDIA only), in MiB.
function driverGpuMemory() {
  try {
    const pids = new Set(browserProcesses().map((p) => p.pid));
    const out = execFileSync('nvidia-smi', ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], { encoding: 'utf8' });
    let total = 0;
    for (const line of out.split('\n')) {
      const [pid, mib] = line.split(',').map((v) => v.trim());
      if (pids.has(pid)) total += Number(mib);
    }
    return total || null;
  } catch {
    return null;
  }
}

const measurements = [];

/// Averages the perf reports logged while measuring and samples the memory use at the end.
async function measure(seconds, label) {
  note(`[${elapsed().toFixed(1)} s] step: measuring "${label}" for ${seconds} s`);
  const from = lines.length;
  // Resolves after a garbage collection (Chromium forces one within about 20 s), so it is
  // started now and collected at the end if done by then.
  await page.evaluate(() => {
    window.__uaMemory = null;
    performance.measureUserAgentSpecificMemory?.().then((m) => { window.__uaMemory = m; }, () => {});
  });
  await pumpFor(seconds);
  const window_ = lines.slice(from);
  const perf = window_.map((l) => PERF.exec(l)).filter(Boolean).map((m) => m.slice(1).map(Number));
  const perFrame = window_.map((l) => PER_FRAME.exec(l)).filter(Boolean).map((m) => m.slice(1).map(Number));
  const gpu = window_.map((l) => GPU_MEMORY.exec(l)).filter(Boolean).at(-1);
  const mean = (rows, i) => (rows.length ? rows.reduce((a, r) => a + r[i], 0) / rows.length : null);
  const page_ = await page.evaluate(() => {
    const r = {
      wasmBytes: window.Module?._skyrt_memory_bytes?.() ?? window.Module?.HEAPU8?.buffer.byteLength,
      jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
    };
    const m = window.__uaMemory;
    if (m) {
      r.uaBytes = m.bytes;
      r.uaBreakdown = m.breakdown
        .filter((b) => b.bytes > 0)
        .sort((a, b) => b.bytes - a.bytes)
        .slice(0, 6)
        .map((b) => ({ bytes: b.bytes, types: b.types.join(','), scope: b.attribution.map((a) => a.scope).join(',') }));
    }
    return r;
  });
  const row = {
    label,
    at: Number(elapsed().toFixed(1)),
    reports: perf.length,
    fps: mean(perf, 0),
    drawsPerFrame: mean(perf, 1),
    rendererExecutingPct: mean(perf, 2),
    rendererGpuWaitPct: mean(perf, 3),
    rendererGameWaitPct: mean(perf, 4),
    submitsPerFrame: mean(perFrame, 0),
    writeBuffersPerFrame: mean(perFrame, 1),
    writeBufferKiBPerFrame: mean(perFrame, 2),
    writeTexturesPerFrame: mean(perFrame, 3),
    gpuLiveMiB: gpu ? Number(gpu[1]) : null,
    gpuLiveObjects: gpu ? Number(gpu[2]) : null,
    textureCacheMiB: gpu ? Number(gpu[3]) : null,
    textureCacheTextures: gpu ? Number(gpu[4]) : null,
    gpuDetail: gpu?.[5] ?? null,
    wasmMiB: page_.wasmBytes ? page_.wasmBytes / 2 ** 20 : null,
    jsHeapMiB: page_.jsHeapBytes != null ? page_.jsHeapBytes / 2 ** 20 : null,
    uaSpecificMiB: page_.uaBytes != null ? page_.uaBytes / 2 ** 20 : null,
    uaBreakdown: page_.uaBreakdown ?? null,
    processPssMiB: Object.fromEntries(Object.entries(processMemory()).map(([k, v]) => [k, v / 1024])),
    driverGpuMiB: driverGpuMemory(),
    largestMappings: largestMappings(),
    allocators: allocatorDumps ? await allocatorDump().catch((e) => ({ error: e.message })) : null,
  };
  measurements.push(row);
  note(`[${elapsed().toFixed(1)} s] measured "${label}": ${JSON.stringify(row)}`);
}

function writeSummary(status) {
  const f = (v, d = 1) => (v == null ? '-' : Number(v).toFixed(d));
  const table = [
    ['', ...measurements.map((m) => m.label)],
    ['fps', ...measurements.map((m) => f(m.fps))],
    ['draws/frame', ...measurements.map((m) => f(m.drawsPerFrame, 0))],
    ['renderer executing %', ...measurements.map((m) => f(m.rendererExecutingPct, 0))],
    ['renderer waiting for GPU %', ...measurements.map((m) => f(m.rendererGpuWaitPct, 0))],
    ['submits/frame', ...measurements.map((m) => f(m.submitsPerFrame, 0))],
    ['write_buffer/frame', ...measurements.map((m) => f(m.writeBuffersPerFrame, 0))],
    ['write_buffer KiB/frame', ...measurements.map((m) => f(m.writeBufferKiBPerFrame, 0))],
    ['write_texture/frame', ...measurements.map((m) => f(m.writeTexturesPerFrame, 0))],
    ['GPU objects live MiB', ...measurements.map((m) => f(m.gpuLiveMiB))],
    ['texture cache MiB', ...measurements.map((m) => f(m.textureCacheMiB))],
    ['wasm memory MiB', ...measurements.map((m) => f(m.wasmMiB, 0))],
    ['JS heap MiB', ...measurements.map((m) => f(m.jsHeapMiB))],
    // measureUserAgentSpecificMemory counts the shared wasm memory once per worker: show one realm.
    ['largest JS realm MiB', ...measurements.map((m) => f(m.uaBreakdown?.[0]?.bytes / 2 ** 20, 0))],
    ['all processes PSS MiB', ...measurements.map((m) => f(Object.values(m.processPssMiB).reduce((a, b) => a + b, 0), 0))],
    ['renderer process PSS MiB', ...measurements.map((m) => f(m.processPssMiB.renderer, 0))],
    ['GPU process PSS MiB', ...measurements.map((m) => f(m.processPssMiB['gpu-process'], 0))],
    ['driver GPU MiB', ...measurements.map((m) => f(m.driverGpuMiB, 0))],
  ];
  const widths = table[0].map((_, c) => Math.max(...table.map((r) => String(r[c]).length)));
  const text = table.map((r) => r.map((v, c) => String(v).padStart(widths[c])).join('  ')).join('\n');
  note(`summary (${status}):\n${text}`);
  writeFileSync(join(out, 'summary.json'), JSON.stringify({ status, steps: steps.map((s) => s.text), measurements }, null, 2));
}

/// Whether `step`'s conditions hold, looking at log lines from index `from` on.
async function holds(step, from, since) {
  for (const c of step.conditions) {
    if (c.kind === 'log') {
      if (!lines.slice(from).some((l) => c.re.test(l))) return false;
    } else if (c.kind === 'after') {
      if (elapsed() - since < c.seconds) return false;
    } else {
      let v;
      if (c.kind === 'swaps') {
        v = await swapCount();
      } else {
        const report = lines.slice(from).map((l) => PERF.exec(l)).filter(Boolean).at(-1);
        if (!report) return false;
        v = Number(c.kind === 'fps' ? report[1] : report[2]);
      }
      const ok = { '>=': v >= c.value, '<=': v <= c.value, '>': v > c.value, '<': v < c.value }[c.op];
      if (!ok) return false;
    }
  }
  return true;
}

let status = 'ok';
if (steps.length === 0) {
  await pumpFor(seconds);
} else {
  // A `tap ... every` of the step before, repeated until this step's condition holds.
  let repeat = null;
  run: for (const [i, step] of steps.entries()) {
    const from = lines.length;
    const since = elapsed();
    note(`[${since.toFixed(1)} s] step ${i + 1}: waiting for ${step.text}`);
    while (!(await holds(step, from, since))) {
      if (elapsed() - since > stepTimeout || elapsed() > seconds) {
        status = `step ${i + 1} (${step.text}) did not happen within ${elapsed() - since > stepTimeout ? `${stepTimeout} s` : `the run's ${seconds} s`}`;
        break run;
      }
      if (repeat && elapsed() >= repeat.next) {
        await tap(repeat.name, repeat.mask);
        repeat.next = elapsed() + repeat.every;
      }
      await sleep(250);
      await pump();
    }
    repeat = null;
    note(`[${elapsed().toFixed(1)} s] step ${i + 1} reached`);
    for (const a of step.actions) {
      if (a.kind === 'tap') {
        await tap(a.name, a.mask);
        if (a.every > 0) repeat = { ...a, next: elapsed() + a.every };
      } else if (a.kind === 'shot') {
        await screenshot(a.name);
      } else if (a.kind === 'measure') {
        await measure(a.seconds, a.label);
      } else if (a.kind === 'wait') {
        await pumpFor(a.seconds);
      } else if (a.kind === 'gc') {
        await collectGarbage();
      } else if (a.kind === 'exit') {
        break run;
      }
    }
  }
}
if (status !== 'ok') note(`FAILED: ${status}`);
if (steps.length > 0) {
  await screenshot('final');
  writeSummary(status);
}
await context.close();
server.close();
process.exit(status === 'ok' ? 0 : 1);

/// Samples every worker's CPU for `seconds` and logs its top functions by self time.
async function profileWorkers(seconds) {
  const cdp = await context.newCDPSession(page);
  const workers = new Map();
  const pending = new Map();
  let nextId = 1;
  cdp.on('Target.attachedToTarget', (e) => workers.set(e.sessionId, e.targetInfo));
  cdp.on('Target.receivedMessageFromTarget', (e) => {
    const msg = JSON.parse(e.message);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  });
  const send = (sessionId, method, params = {}) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      cdp
        .send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method, params }) })
        .catch(() => resolve(null));
      setTimeout(() => resolve(null), 30000);
    });
  await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: false });
  await new Promise((r) => setTimeout(r, 1000));
  for (const id of workers.keys()) {
    await send(id, 'Profiler.enable');
    await send(id, 'Profiler.setSamplingInterval', { interval: 500 });
    await send(id, 'Profiler.start');
  }
  note(`profiling ${workers.size} workers for ${seconds} s`);
  await new Promise((r) => setTimeout(r, seconds * 1000));
  for (const id of workers.keys()) {
    const reply = await send(id, 'Profiler.stop');
    const profile = reply?.result?.profile;
    if (!profile) continue;
    const self = new Map();
    let total = 0;
    for (const n of profile.nodes) {
      const name = n.callFrame.functionName || `(${n.callFrame.url.split('/').pop()})`;
      self.set(name, (self.get(name) ?? 0) + (n.hitCount ?? 0));
      total += n.hitCount ?? 0;
    }
    const busy = total - (self.get('(idle)') ?? 0) - (self.get('(program)') ?? 0);
    if (busy <= 0) continue;
    const top = [...self.entries()].filter(([n]) => n !== '(idle)').sort((a, b) => b[1] - a[1]).slice(0, 15);
    note(`profile of ${workers.get(id)?.title ?? id} (${total} samples):\n` +
      top.map(([n, c]) => `  ${((100 * c) / total).toFixed(1).padStart(5)}%  ${n}`).join('\n'));
  }
  await cdp.detach();
}

/// Collects garbage in the page and every worker. Workers that allocate little JS rarely collect
/// on their own, so their dead ArrayBuffers (and WebGPU wrappers) can hold native memory.
async function collectGarbage() {
  const cdp = await context.newCDPSession(page);
  const workers = new Set();
  const pending = new Map();
  let nextId = 1;
  cdp.on('Target.attachedToTarget', (e) => workers.add(e.sessionId));
  cdp.on('Target.receivedMessageFromTarget', (e) => {
    const msg = JSON.parse(e.message);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  });
  const send = (sessionId, method, params = {}) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      cdp
        .send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method, params }) })
        .catch(() => resolve(null));
      setTimeout(() => resolve(null), 10000);
    });
  await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: false });
  await sleep(1000);
  await cdp.send('HeapProfiler.collectGarbage');
  for (const id of workers) await send(id, 'HeapProfiler.collectGarbage');
  await cdp.detach();
  note(`[${elapsed().toFixed(1)} s] step: collected garbage in the page and ${workers.size} workers`);
}

/// Pauses every worker and logs its call stack.
async function dumpWorkerStacks() {
  const cdp = await context.newCDPSession(page);
  const workers = new Map();
  const pending = new Map();
  let nextId = 1;
  cdp.on('Target.attachedToTarget', (e) => workers.set(e.sessionId, e.targetInfo));
  cdp.on('Target.receivedMessageFromTarget', (e) => {
    const msg = JSON.parse(e.message);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    } else if (msg.method === 'Debugger.paused') {
      const frames = msg.params.callFrames.slice(0, 25).map((f) => `    ${f.functionName || '(anonymous)'}`);
      note(`stack of ${workers.get(e.sessionId)?.title ?? e.sessionId}:\n${frames.join('\n')}`);
    }
  });
  const send = (sessionId, method, params = {}) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      cdp
        .send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method, params }) })
        .catch(() => resolve(null));
      setTimeout(() => resolve(null), 3000);
    });
  await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: false });
  await new Promise((r) => setTimeout(r, 1000));
  note(`dumping stacks of ${workers.size} workers`);
  for (const sessionId of workers.keys()) {
    await send(sessionId, 'Debugger.enable');
    await send(sessionId, 'Debugger.pause');
    await new Promise((r) => setTimeout(r, 1500));
    await send(sessionId, 'Debugger.resume');
  }
  await cdp.detach();
}
