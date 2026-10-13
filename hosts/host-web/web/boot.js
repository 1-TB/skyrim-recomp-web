// The page's controller: imports the user's game files into OPFS (game/) or streams them from the
// server, keeps the settings, and boots host-web.js. The runtime reads OPFS through WasmFS.
'use strict';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const statusEl = $('status');

// Settings, kept in this browser (localStorage may be unavailable: then they last this visit).
const SETTINGS_KEY = 'skyrt-settings';
const DEFAULTS = {
  source: 'local', fps: '30', log: 'info', touch: 'auto', sensitivity: 1, volume: 1, perf: false, upload: true,
  debug: [], extra: '',
};
const settings = { ...DEFAULTS };
try {
  Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}'));
} catch {}
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {}
}

// The whole log (the page only shows its tail), and what hasn't been sent to the server yet.
const fullLog = [];
let unsent = [];
// The dev server (`cargo xtask serve`) keeps each page session's log in target/web-logs/.
const logSession = `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(36).slice(2, 8)}`;
// Off when the user says so, or when the server has no log endpoint.
let logUpload = settings.upload;

// Runtime lines that feed the performance overlay:
// "perf: 58.1 fps, 2260 draws/frame; renderer thread: 72% executing, ...".
const PERF_LINE = /perf: ([\d.]+) fps, (\d+) draws\/frame; renderer thread: (\d+)% executing/;

function log(line) {
  const perf = PERF_LINE.exec(line);
  if (perf) $('perf').textContent = `${perf[1]} fps · ${perf[2]} draws · renderer ${perf[3]}% busy`;
  fullLog.push(line);
  unsent.push(line);
  logEl.textContent += line + '\n';
  if (logEl.textContent.length > 2_000_000) logEl.textContent = logEl.textContent.slice(-1_000_000);
  logEl.scrollTop = logEl.scrollHeight;
}

async function sendLog() {
  if (!logUpload || !settings.upload || unsent.length === 0) return;
  const lines = unsent;
  unsent = [];
  try {
    const r = await fetch(`log?session=${logSession}`, { method: 'POST', body: lines.join('\n') + '\n' });
    // Another server (no log endpoint): stop trying.
    if (r.status === 404 || r.status === 405) logUpload = false;
  } catch {
    unsent = lines.concat(unsent);
  }
}
setInterval(sendLog, 3000);
addEventListener('pagehide', sendLog);

$('copy-log').addEventListener('click', async () => {
  const text = fullLog.join('\n') + '\n';
  try {
    await navigator.clipboard.writeText(text);
    $('copy-log').textContent = `Copied ${fullLog.length} lines`;
  } catch {
    // Clipboard denied: offer the log as a file instead.
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = `skyrim-web-${logSession}.log`;
    a.click();
  }
  setTimeout(() => ($('copy-log').textContent = 'Copy log'), 2000);
});

// Saves live in OPFS under content/. The dev server (`cargo xtask serve`) can hold a copy in
// target/web-saves/: "Export saves" uploads every file there, "Import saves" (before booting:
// the running title keeps its files open) copies them back.
async function contentDir(create) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('content', { create });
}

async function* walk(dir, prefix = '') {
  for await (const [name, handle] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === 'directory') yield* walk(handle, path);
    else yield [path, handle];
  }
}

const savesUrl = (path) => 'saves/' + path.split('/').map(encodeURIComponent).join('/');

$('export-saves').addEventListener('click', async () => {
  try {
    let files = 0, bytes = 0;
    for await (const [path, handle] of walk(await contentDir(false))) {
      const file = await handle.getFile();
      const r = await fetch(savesUrl(path), { method: 'POST', body: file });
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      files += 1;
      bytes += file.size;
    }
    log(`Exported ${files} save files (${(bytes / 2 ** 20).toFixed(1)} MiB) to the server.`);
  } catch (e) {
    log(`Export failed: ${e.message}`);
  }
});

$('import-saves').addEventListener('click', async () => {
  try {
    const list = (await (await fetch('saves/')).text()).split('\n').filter(Boolean);
    const content = await contentDir(true);
    for (const path of list) {
      const parts = path.split('/');
      let dir = content;
      for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part, { create: true });
      const data = await (await fetch(savesUrl(path))).arrayBuffer();
      const out = await (await dir.getFileHandle(parts.at(-1), { create: true })).createWritable();
      await out.write(data);
      await out.close();
    }
    log(`Imported ${list.length} save files from the server.`);
  } catch (e) {
    log(`Import failed (import before booting): ${e.message}`);
  }
});

async function gameDir(create) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('game', { create });
}

async function copyTree(src, dst, counter) {
  for await (const [name, handle] of src.entries()) {
    if (handle.kind === 'directory') {
      await copyTree(handle, await dst.getDirectoryHandle(name, { create: true }), counter);
    } else {
      const file = await handle.getFile();
      const out = await (await dst.getFileHandle(name, { create: true })).createWritable();
      await file.stream().pipeTo(out);
      counter.files += 1;
      counter.bytes += file.size;
      statusEl.textContent = `${counter.files} files, ${(counter.bytes / 2 ** 20).toFixed(1)} MiB copied`;
    }
  }
}

const gib = (bytes) => `${(bytes / 2 ** 30).toFixed(2)} GiB`;

$('pick').onclick = async () => {
  if (!window.showDirectoryPicker) {
    statusEl.textContent = 'This browser has no folder picker: import a .zip of the game folder, or stream it from the server.';
    return;
  }
  try {
    const src = await window.showDirectoryPicker();
    const counter = { files: 0, bytes: 0 };
    await copyTree(src, await gameDir(true), counter);
    statusEl.textContent = `Imported ${counter.files} files (${gib(counter.bytes)}).`;
  } catch (e) {
    statusEl.textContent = `Import failed: ${e.message}`;
  }
};

$('pick-zip').onclick = () => $('zip-file').click();
$('zip-file').onchange = async () => {
  const file = $('zip-file').files[0];
  $('zip-file').value = '';
  if (!file) return;
  try {
    const { extractZip } = await import('./zip.js');
    const done = await extractZip(file, await gameDir(true), (files, bytes, name) => {
      statusEl.textContent = `${files} files, ${(bytes / 2 ** 20).toFixed(1)} MiB extracted (${name})`;
    });
    statusEl.textContent = `Imported ${done.files} files (${gib(done.bytes)}) from ${file.name}.`;
  } catch (e) {
    statusEl.textContent = `Zip import failed: ${e.message}`;
  }
};

$('clear').onclick = async () => {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry('game', { recursive: true }).catch(() => {});
  statusEl.textContent = 'Cleared.';
};

// Where the game files come from: OPFS, or the dev server (`cargo xtask serve --game-dir DIR`),
// which lists them at game/manifest and serves byte ranges of game/files/PATH.
const STREAM_BASE = new URL('game/', location.href).href;

async function streamAvailable() {
  try {
    const r = await fetch(STREAM_BASE + 'manifest', { method: 'HEAD' });
    return r.ok;
  } catch {
    return false;
  }
}

async function localGameReady() {
  try {
    await (await gameDir(false)).getFileHandle('default.xex');
    return true;
  } catch {
    return false;
  }
}

async function showSource() {
  const stream = settings.source === 'stream';
  $('local-controls').hidden = stream;
  $('source-status').textContent = stream
    ? ((await streamAvailable()) ? 'The server offers the game files.' : 'The server offers no game files (start it with --game-dir).')
    : ((await localGameReady()) ? '' : 'Nothing imported yet.');
}

// Settings panel: each control mirrors one setting.
function bindSetting(id, key, parse = (v) => v, onChange = () => {}) {
  const el = $(id);
  const isBox = el.type === 'checkbox';
  if (isBox) el.checked = settings[key]; else el.value = settings[key];
  el.addEventListener(isBox || el.tagName === 'SELECT' ? 'change' : 'input', () => {
    settings[key] = isBox ? el.checked : parse(el.value);
    saveSettings();
    onChange(settings[key]);
  });
}

let gain = null;
let touch = null;
const touchWanted = () => settings.touch === 'on' || (settings.touch === 'auto' && matchMedia('(pointer: coarse)').matches);

bindSetting('source', 'source', String, showSource);
bindSetting('s-fps', 'fps');
bindSetting('s-log', 'log');
bindSetting('s-touch', 'touch', String, () => { if (touch) touch.visible = touchWanted(); });
bindSetting('s-sens', 'sensitivity', Number);
bindSetting('s-volume', 'volume', Number, (v) => { if (gain) gain.gain.value = v; });
bindSetting('s-perf', 'perf', Boolean, (v) => { $('perf').hidden = !v; });
bindSetting('s-upload', 'upload', Boolean);
bindSetting('s-extra', 'extra', String);
for (const box of document.querySelectorAll('[data-debug]')) {
  box.checked = settings.debug.includes(box.dataset.debug);
  box.addEventListener('change', () => {
    settings.debug = [...document.querySelectorAll('[data-debug]')].filter((b) => b.checked).map((b) => b.dataset.debug);
    saveSettings();
  });
}
$('perf').hidden = !settings.perf;
showSource();

// Vblank interrupts per second for each cap: the title presents every second vblank.
const VBLANK_HZ = { 30: 60, 60: 120, unlimited: 1000 };

function runtimeArguments() {
  const args = ['--log', settings.log];
  const hz = VBLANK_HZ[settings.fps];
  if (hz && hz !== 60) args.push('--vblank-hz', String(hz));
  for (const mode of settings.debug) args.push('--render-debug', mode);
  if (settings.source === 'stream') args.push('--stream-game', STREAM_BASE);
  return args.concat(settings.extra.split(/\s+/).filter(Boolean));
}

// Fullscreen shows the game (and the touch controls) alone, in landscape on phones.
// Without the Fullscreen API for elements (iPhone Safari) the stage fills the viewport instead.
function setFilled(on) {
  $('stage').classList.toggle('filled', on);
  $('exit-fill').hidden = !on;
}
$('exit-fill').onclick = () => setFilled(false);

$('fullscreen').onclick = async () => {
  const stage = $('stage');
  if (document.fullscreenElement || document.webkitFullscreenElement) {
    await (document.exitFullscreen?.() ?? document.webkitExitFullscreen?.());
    return;
  }
  const request = stage.requestFullscreen ?? stage.webkitRequestFullscreen;
  if (!request) {
    setFilled(true);
    return;
  }
  try {
    await request.call(stage, { navigationUI: 'hide' });
    await screen.orientation?.lock?.('landscape').catch(() => {});
  } catch (e) {
    log(`Fullscreen unavailable (${e.message}); filling the window instead.`);
    setFilled(true);
  }
};

$('boot').onclick = async () => {
  if (!crossOriginIsolated) {
    log('Page is not cross-origin isolated (COOP/COEP); serve it with `cargo xtask serve`.');
    return;
  }
  if (settings.source === 'stream') {
    if (!(await streamAvailable())) {
      log('The server offers no game files: start it with `cargo xtask serve --game-dir DIR`, or import the game.');
      return;
    }
  } else if (!(await localGameReady())) {
    log('No game/default.xex in this browser; import the game folder (or a .zip of it) first, or stream it.');
    return;
  }
  $('boot').disabled = true;
  // Audio needs a user gesture to start: set it up during this click.
  let audioNode = null;
  try {
    const context = new AudioContext({ sampleRate: 48000 });
    await context.audioWorklet.addModule('audio-worklet.js');
    audioNode = new AudioWorkletNode(context, 'skyrt-audio', { outputChannelCount: [2] });
    gain = new GainNode(context, { gain: settings.volume });
    audioNode.connect(gain).connect(context.destination);
  } catch (e) {
    log(`Audio unavailable: ${e.message}`);
  }
  const args = runtimeArguments();
  log(`Booting with: ${args.join(' ')}`);
  window.Module = {
    // Handed to the renderer thread as an OffscreenCanvas.
    canvas: $('canvas'),
    arguments: args,
    print: log,
    printErr: log,
    async onRuntimeInitialized() {
      const { createTouch } = await import('./touch.js');
      touch = createTouch($('stage'));
      touch.visible = touchWanted();
      const { startInput } = await import('./input.js');
      startInput(window.Module, { sensitivity: () => settings.sensitivity, touch });
      audioNode?.port.postMessage({ memory: window.Module.HEAPU8.buffer, header: window.Module._skyrt_audio_ring() });
    },
  };
  const script = document.createElement('script');
  script.src = 'host-web.js';
  document.body.appendChild(script);
};
