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
//          [--stacks-at S[,S...]] [--audio] [--profile-at S:SECONDS]
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
import { createServer } from 'node:http';
import { appendFileSync, createReadStream, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
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

const dist = resolve(option('dist'));
const game = resolve(option('game'));
const profile = resolve(option('profile'));
const out = resolve(option('out'));
const seconds = Number(option('seconds', '120'));
const shotEvery = Number(option('shot-every', '10'));
const runtimeArgs = option('args', '--log info').split(/\s+/).filter(Boolean);
// Fixed so the origin (and with it the imported game in OPFS) is the same on every run.
const port = Number(option('port', '8091'));
const reloadAfter = Number(option('reload-after', '0'));
const audio = process.argv.includes('--audio');
const [profileAt, profileFor] = option('profile-at', '0:0').split(':').map(Number);
const stacksAt = option('stacks-at', '').split(',').filter(Boolean).map(Number);
mkdirSync(out, { recursive: true });
const logPath = join(out, 'log.txt');
writeFileSync(logPath, '');

const require = createRequire(join(process.env.PW_DIR ?? '.', 'noop.js'));
const { chromium } = require('playwright');

function listFiles(dir) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...listFiles(path));
    else if (entry.isFile()) files.push({ path: relative(game, path).split(sep).join('/'), size: statSync(path).size });
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
  const [root, name] = path.startsWith('/__game/') ? [game, path.slice(8)] : [dist, path.slice(1) || 'index.html'];
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
const imported = await page.evaluate(async () => {
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
let nextShot = shotEvery;
let reloaded = false;
let stacksDumped = 0;
let profiled = false;
while ((Date.now() - start) / 1000 < seconds) {
  await new Promise((r) => setTimeout(r, 1000));
  for (const l of await page.evaluate(() => window.__lines.splice(0))) note(l);
  const t = (Date.now() - start) / 1000;
  if (audio && Math.round(t) % 5 === 0) {
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
  if (t >= nextShot) {
    const shot = join(out, `canvas_${String(Math.round(t)).padStart(4, '0')}s.png`);
    await canvas.screenshot({ path: shot, timeout: 30000 }).then(
      () => note(`screenshot ${shot}`),
      (e) => note(`screenshot failed: ${e.message.split('\n')[0]}`),
    );
    nextShot += shotEvery;
  }
}
await context.close();
server.close();

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
