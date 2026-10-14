// The page's controller: imports the user's game files into OPFS (game/) or streams them from the
// server, keeps the settings, and boots host-web.js. The runtime reads OPFS through WasmFS.
// The load status (status.js), the controls help (controls.js) and the save manager (saves.js)
// are modules loaded beside it; the in-game menu lives here.
'use strict';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const statusEl = $('status');

// Settings, kept in this browser (localStorage may be unavailable: then they last this visit).
const SETTINGS_KEY = 'skyrt-settings';
const DEFAULTS = {
  source: 'local', sync: true, fps: '30', log: 'info', touch: 'auto', sensitivity: 1, volume: 1, perf: false, upload: true,
  debug: [], extra: '', theme: 'auto', scale: '1', keys: null, tab: 'controls',
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

// Page modules, loaded at once (they only define things until used).
const modules = Promise.all([import('./status.js'), import('./controls.js'), import('./saves.js')])
  .then(([status, controls, saves]) => ({ status, controls, saves }));
// The boot's load status (status.js), once booting.
let bootStatus = null;
// Whether the title has been started in this page (it can't be stopped short of a reload).
let running = false;

// A short confirmation at the bottom of the page.
let toastTimer = 0;
function toast(text) {
  const t = $('toast');
  t.textContent = text;
  t.classList.remove('fade');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('fade'), 3000);
}

// The current phase beside the Boot button (state: 'busy', 'ok', 'err' or '' for idle).
function showPhase(text, state) {
  $('boot-phase').textContent = text;
  $('boot-dot').className = `dot ${state}`;
}

// Lines meaning a thread died (a Rust panic, an abort, a worker's uncaught error): the game may
// keep running without a renderer, which otherwise looks like a frozen picture.
const FATAL_LINE = /panicked at |^Aborted\(|worker sent an error|RuntimeError: /;
let fatalShown = false;

// Shows `message` over the page (once) and sends the log at once.
function fatal(message) {
  if (!fatalShown) {
    fatalShown = true;
    const banner = document.createElement('div');
    banner.className = 'fatal-banner';
    banner.setAttribute('role', 'alert');
    const text = document.createElement('div');
    text.className = 'msg';
    text.textContent = `The game stopped with an error: ${message}. Reload the page to retry; the log has been sent.`;
    const reload = document.createElement('button');
    reload.className = 'small';
    reload.textContent = 'Reload';
    reload.onclick = () => location.reload();
    const dismiss = document.createElement('button');
    dismiss.className = 'small';
    dismiss.textContent = 'Dismiss';
    dismiss.onclick = () => banner.remove();
    banner.append(text, reload, dismiss);
    // Inside the stage while it is fullscreen (only the fullscreen element is shown).
    (document.fullscreenElement ?? document.body).appendChild(banner);
    bootStatus?.fail(message);
    showPhase('Stopped with an error', 'err');
  }
  sendLog();
}

function log(line) {
  if (FATAL_LINE.test(line)) queueMicrotask(() => fatal(line));
  const perf = PERF_LINE.exec(line);
  if (perf) $('perf').textContent = `${perf[1]} fps · ${perf[2]} draws · renderer ${perf[3]}% busy`;
  bootStatus?.line(line);
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
// Often: a phone closing the tab for using too much memory loses whatever wasn't sent.
setInterval(sendLog, 1000);
addEventListener('pagehide', sendLog);
// Uncaught errors (e.g. host-web.js failing to create its memory) would otherwise leave no trace.
addEventListener('error', (e) => {
  log(`Error: ${e.message} (${e.filename}:${e.lineno})`);
  sendLog();
});
addEventListener('unhandledrejection', (e) => {
  log(`Unhandled rejection: ${e.reason?.stack ?? e.reason}`);
  sendLog();
});

async function copyLog() {
  const text = fullLog.join('\n') + '\n';
  try {
    await navigator.clipboard.writeText(text);
    toast(`Copied ${fullLog.length} log lines`);
  } catch {
    // Clipboard denied: offer the log as a file instead.
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = `skyrim-web-${logSession}.log`;
    a.click();
  }
}
$('copy-log').addEventListener('click', copyLog);
$('log-note').textContent = `Session ${logSession}`;

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
  if (!confirm('Delete the game files imported into this browser? Saves are kept.')) return;
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

// The runtime is recompiled against one default.xex (with its title update), so an imported copy
// goes stale whenever the server's game folder is rebuilt. Before booting imported files, copy
// default.xex and $TitleUpdate/ from the server when they are missing or differ in size, or when
// the server's default.xex changed since the last copy (its ETag is remembered).
const SYNCED_KEY = 'skyrt-synced-xex';

async function syncFromServer() {
  if (!(await streamAvailable())) return;
  const manifest = (await (await fetch(STREAM_BASE + 'manifest')).text())
    .split('\n')
    .map((line) => line.split('\t'))
    .filter(([size, path]) => path && (path === 'default.xex' || path.startsWith('$TitleUpdate/')));
  const head = await fetch(STREAM_BASE + 'files/default.xex', { method: 'HEAD' });
  const xexStamp = head.headers.get('ETag') ?? head.headers.get('Content-Length');
  let synced = null;
  try {
    synced = localStorage.getItem(SYNCED_KEY);
  } catch {}
  const root = await gameDir(true);
  let copied = 0;
  for (const [size, path] of manifest) {
    const parts = path.split('/');
    let dir = root;
    for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part, { create: true });
    const name = parts[parts.length - 1];
    let local = null;
    try {
      local = await (await dir.getFileHandle(name)).getFile();
    } catch {}
    const stale = !local || local.size !== Number(size) || (path === 'default.xex' && synced !== xexStamp);
    if (!stale) continue;
    statusEl.textContent = `Updating ${path} from the server…`;
    const response = await fetch(STREAM_BASE + 'files/' + parts.map(encodeURIComponent).join('/'));
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const out = await (await dir.getFileHandle(name, { create: true })).createWritable();
    await response.body.pipeTo(out);
    copied += 1;
  }
  try {
    localStorage.setItem(SYNCED_KEY, xexStamp);
  } catch {}
  statusEl.textContent = copied ? `Updated ${copied} file(s) from the server.` : '';
  if (copied) log(`Updated ${copied} file(s) (default.xex / title update) from the server.`);
}

async function showSource() {
  const stream = settings.source === 'stream';
  $('local-controls').hidden = stream;
  $('source-status').textContent = stream
    ? ((await streamAvailable()) ? 'The server offers the game files.' : 'The server offers no game files (start it with --game-dir).')
    : ((await localGameReady()) ? '' : 'Nothing imported yet.');
}

// Settings: every control with `data-setting="key"` mirrors that setting (the menu repeats some
// of the panel's), `data-output="key"` shows a slider's value, and `onSetting[key]` applies a
// change at once (the others apply at the next boot).
let gain = null;
let touch = null;
const touchWanted = () => settings.touch === 'on' || (settings.touch === 'auto' && matchMedia('(pointer: coarse)').matches);

function setTouchVisible() {
  const on = touchWanted();
  if (touch) touch.visible = on;
  $('stage').classList.toggle('touch', !!touch && on);
}

function applyTheme() {
  const theme = settings.theme;
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  for (const b of document.querySelectorAll('[data-theme-choice]')) b.setAttribute('aria-pressed', String(b.dataset.themeChoice === theme));
}

const onSetting = {
  source: showSource,
  touch: setTouchVisible,
  volume: (v) => { if (gain) gain.gain.value = v; },
  perf: (v) => { $('perf').hidden = !v || !running; },
};
const FORMAT = {
  volume: (v) => `${Math.round(v * 100)}%`,
  sensitivity: (v) => `${Number(v).toFixed(2)}×`,
};

function showSetting(key) {
  for (const el of document.querySelectorAll(`[data-setting="${key}"]`)) {
    if (el.type === 'checkbox') el.checked = !!settings[key];
    else el.value = String(settings[key]);
  }
  for (const out of document.querySelectorAll(`[data-output="${key}"]`)) out.textContent = FORMAT[key]?.(settings[key]) ?? settings[key];
}

function setSetting(key, value) {
  settings[key] = value;
  saveSettings();
  showSetting(key);
  onSetting[key]?.(value);
}

for (const el of document.querySelectorAll('[data-setting]')) {
  const key = el.dataset.setting;
  const parse = typeof DEFAULTS[key] === 'number' ? Number : (v) => v;
  el.addEventListener(el.type === 'range' || el.type === 'text' ? 'input' : 'change', () => {
    setSetting(key, el.type === 'checkbox' ? el.checked : parse(el.value));
  });
}
for (const key of new Set([...document.querySelectorAll('[data-setting]')].map((el) => el.dataset.setting))) showSetting(key);
for (const box of document.querySelectorAll('[data-debug]')) {
  box.checked = settings.debug.includes(box.dataset.debug);
  box.addEventListener('change', () => {
    settings.debug = [...document.querySelectorAll('[data-debug]')].filter((b) => b.checked).map((b) => b.dataset.debug);
    saveSettings();
  });
}
for (const b of document.querySelectorAll('[data-theme-choice]')) {
  b.addEventListener('click', () => { settings.theme = b.dataset.themeChoice; saveSettings(); applyTheme(); });
}
applyTheme();

// Tabs: controls, saves, log.
let saveManager = null;
function showTab(name) {
  for (const b of document.querySelectorAll('[data-tab]')) {
    const on = b.dataset.tab === name;
    b.setAttribute('aria-selected', String(on));
    b.tabIndex = on ? 0 : -1;
    $(b.getAttribute('aria-controls')).hidden = !on;
  }
  if (name === 'saves') saveManager?.refresh();
  if (name === 'log') logEl.scrollTop = logEl.scrollHeight;
}
for (const b of document.querySelectorAll('[data-tab]')) {
  b.addEventListener('click', () => { settings.tab = b.dataset.tab; saveSettings(); showTab(b.dataset.tab); });
}
showTab(['controls', 'saves', 'log'].includes(settings.tab) ? settings.tab : 'controls');

modules.then(({ controls, saves }) => {
  controls.createControls({
    overrides: () => settings.keys,
    save: (overrides) => { settings.keys = overrides; saveSettings(); },
  });
  saveManager = saves.createSaveManager({ log, toast, running: () => running });
}).catch((e) => log(`Page modules failed to load: ${e.message}`));

showSource();

// Vblank interrupts per second for each cap: the title presents every second vblank.
const VBLANK_HZ = { 30: 60, 60: 120, unlimited: 1000 };

// The renderer's internal resolution, as a multiple of 1280×720: 1 to 3, or for "display" the
// multiple that matches the canvas's size on screen in device pixels now.
function resolutionScale() {
  if (settings.scale === 'display') {
    const height = $('canvas').getBoundingClientRect().height * devicePixelRatio;
    return Math.min(3, Math.max(1, Math.round(height / 720)));
  }
  return Math.min(3, Math.max(1, Number(settings.scale) || 1));
}

function runtimeArguments() {
  const args = ['--log', settings.log];
  const hz = VBLANK_HZ[settings.fps];
  if (hz && hz !== 60) args.push('--vblank-hz', String(hz));
  // Only when above the default: builds from before the flag reject it.
  const scale = resolutionScale();
  if (scale > 1) args.push('--resolution-scale', String(scale));
  for (const mode of settings.debug) args.push('--render-debug', mode);
  if (settings.source === 'stream') args.push('--stream-game', STREAM_BASE);
  return args.concat(settings.extra.split(/\s+/).filter(Boolean));
}

// Whether this browser lets a canvas hand control to an OffscreenCanvas (the renderer thread
// needs it), tried on a throwaway canvas in the document.
function probeCanvasTransfer() {
  if (!HTMLCanvasElement.prototype.transferControlToOffscreen) return 'unsupported';
  const probe = document.createElement('canvas');
  probe.hidden = true;
  document.body.append(probe);
  try {
    const offscreen = probe.transferControlToOffscreen();
    return `ok (WebGPU context on OffscreenCanvas: ${typeof offscreen.getContext === 'function' && !!navigator.gpu})`;
  } catch (e) {
    return `failed: ${e}`;
  } finally {
    probe.remove();
  }
}

// Logs the page's memory use every 20 s: phone browsers close tabs that use too much, without
// an error, so the last line before a silent stop shows how close it was.
function watchMemory() {
  const mib = (b) => `${(b / 2 ** 20).toFixed(0)} MiB`;
  let measuring = false;
  const report = () => {
    const wasm = window.Module?.HEAPU8?.buffer.byteLength;
    const heap = performance.memory ? `, JS heap ${mib(performance.memory.usedJSHeapSize)}` : '';
    log(`memory: wasm ${wasm ? mib(wasm) : '?'}${heap}`);
    // The whole page's use (Chromium; resolves after a garbage collection, which may take a while).
    if (performance.measureUserAgentSpecificMemory && !measuring) {
      measuring = true;
      performance.measureUserAgentSpecificMemory()
        .then((m) => log(`memory: page total ${mib(m.bytes)}`))
        .catch(() => {})
        .finally(() => { measuring = false; });
    }
  };
  setInterval(report, 20000);
}

// Fullscreen shows the game (and the touch controls) alone, in landscape on phones.
// Without the Fullscreen API for elements (iPhone Safari) the stage fills the viewport instead.
function setFilled(on) {
  $('stage').classList.toggle('filled', on);
  $('exit-fill').hidden = !on;
}
$('exit-fill').onclick = () => setFilled(false);

async function toggleFullscreen() {
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
}
$('fullscreen').onclick = toggleFullscreen;

// The in-game menu: settings that matter while playing, fullscreen, saves and the controls help.
// The game keeps running (there is no way to pause it from here) but gets no input meanwhile.
let menuOpen = false;
function setMenu(open) {
  if (!running) open = false;
  menuOpen = open;
  $('menu').hidden = !open;
  if (open) {
    document.exitPointerLock?.();
    showMenuView('main');
    $('menu-resume').focus();
  }
}
function showMenuView(view) {
  $('menu-main').hidden = view !== 'main';
  $('menu-controls-view').hidden = view !== 'controls';
}
$('menu-button').onclick = () => setMenu(true);
$('open-menu').onclick = () => setMenu(true);
$('menu-resume').onclick = () => setMenu(false);
$('menu-controls').onclick = () => showMenuView('controls');
$('menu-back').onclick = () => showMenuView('main');
$('menu-fullscreen').onclick = toggleFullscreen;
$('menu-log').onclick = copyLog;
$('menu-export').onclick = async () => {
  const { saves } = await modules;
  try {
    if (await saves.serverKeepsSaves()) {
      const { files, bytes } = await saves.exportAllToServer();
      toast(`Exported ${files} save files (${(bytes / 2 ** 20).toFixed(1)} MiB) to the server.`);
    } else {
      toast(`Downloading ${await saves.downloadAll()} save files as a zip.`);
    }
  } catch (e) {
    toast(`Export failed: ${e.message}`);
    log(`Export failed: ${e.message}`);
  }
};
// Clicking the dimmed backdrop closes the menu.
$('menu').addEventListener('click', (e) => { if (e.target === $('menu')) setMenu(false); });
// F1 toggles the menu (Esc is the browser's: it releases the mouse and leaves fullscreen) and
// Esc closes it. Registered before input.js's handler, which skips handled events.
addEventListener('keydown', (e) => {
  if (!running) return;
  if (e.code === 'F1' || (menuOpen && e.code === 'Escape')) {
    e.preventDefault();
    setMenu(e.code === 'F1' ? !menuOpen : false);
  }
});

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
  } else {
    if (settings.sync) {
      try {
        await syncFromServer();
      } catch (e) {
        log(`Updating from the server failed (${e.message}); booting the imported files as they are.`);
      }
    }
    if (!(await localGameReady())) {
      log('No game/default.xex in this browser; import the game folder (or a .zip of it) first, or stream it.');
      return;
    }
  }
  $('boot').disabled = true;
  running = true;
  bootStatus = (await modules).status.createStatus(showPhase);
  bootStatus.start({ logLevel: settings.log, streaming: settings.source === 'stream' });
  $('menu-button').hidden = false;
  $('open-menu').disabled = false;
  $('perf').hidden = !settings.perf;
  saveManager?.refresh();
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
  log(`Browser: ${navigator.userAgent}; device memory ${navigator.deviceMemory ?? '?'} GiB; ${navigator.hardwareConcurrency} threads`);
  watchMemory();
  log(`Canvas transfer probe: ${probeCanvasTransfer()}`);
  // A fresh element: Safari on iOS refused to transfer the page's original canvas
  // (InvalidStateError, i.e. it already had a context) while desktop WebKit accepts it.
  const canvas = $('canvas').cloneNode(false);
  $('canvas').replaceWith(canvas);
  window.Module = {
    // Handed to the renderer thread as an OffscreenCanvas.
    canvas,
    arguments: args,
    print: log,
    printErr: log,
    onAbort: (what) => fatal(`aborted (${what})`),
    // Compiles the module with progress in the log, sent at once: phones may close the tab
    // while compiling (the module is about 200 MB) before the periodic upload.
    instantiateWasm(imports, receive) {
      const started = performance.now();
      const secs = () => ((performance.now() - started) / 1000).toFixed(1);
      const memory = Object.values(imports).flatMap(Object.values).find((v) => v instanceof WebAssembly.Memory);
      log(`Startup: memory ${memory ? memory.buffer.byteLength / 2 ** 20 : '?'} MiB; compiling host_web.wasm`);
      sendLog();
      WebAssembly.compileStreaming(fetch('host_web.wasm'))
        .then(async (module) => {
          log(`Startup: compiled in ${secs()} s; instantiating`);
          sendLog();
          const instance = await WebAssembly.instantiate(module, imports);
          log(`Startup: instantiated after ${secs()} s`);
          sendLog();
          receive(instance, module);
        })
        .catch((e) => {
          log(`Startup failed after ${secs()} s: ${e}`);
          sendLog();
        });
      return {};
    },
    async onRuntimeInitialized() {
      const { createTouch } = await import('./touch.js');
      touch = createTouch($('stage'));
      setTouchVisible();
      const { startInput } = await import('./input.js');
      startInput(window.Module, {
        sensitivity: () => settings.sensitivity,
        keys: () => settings.keys,
        suspended: () => menuOpen,
        touch,
      });
      audioNode?.port.postMessage({ memory: window.Module.HEAPU8.buffer, header: window.Module._skyrt_audio_ring() });
    },
  };
  const script = document.createElement('script');
  script.src = 'host-web.js';
  document.body.appendChild(script);
};
