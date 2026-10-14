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
  debug: [], extra: '', theme: 'auto', scale: '1', keys: null, tab: 'controls', gpu: 'xenos',
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
  if (perf) $('perf').textContent = `${perf[1]} fps · ${perf[2]} draws · renderer ${perf[3]}% busy · ${activeRenderer?.short ?? ''}`;
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

// Crash breadcrumbs: iOS kills a page that uses too much memory without any event, and up to a
// second of log is lost with it. The last lines go to localStorage four times a second; the
// next page load sends them as "previous session ended: ..." (a clean unload is marked).
const CRUMBS_KEY = 'skyrt-crumbs';
try {
  const previous = JSON.parse(localStorage.getItem(CRUMBS_KEY) ?? 'null');
  localStorage.removeItem(CRUMBS_KEY);
  if (previous?.lines?.length) {
    // Not through log(): an old panic line must not raise this page's error banner.
    const lines = [`previous session ended: ${previous.session}, ${previous.clean ? 'unloaded normally' : 'killed or crashed (no unload)'}, `
      + `${(previous.up / 1000).toFixed(1)} s after page load, at ${new Date(previous.at).toISOString()}; its last lines:`]
      .concat(previous.lines.map((l) => `  | ${l}`));
    fullLog.push(...lines);
    unsent.push(...lines);
    logEl.textContent += lines.join('\n') + '\n';
  }
} catch {}
function saveCrumbs(clean = false) {
  if (!running) return;
  try {
    const M = window.Module;
    const mib = (b) => (b / 2 ** 20).toFixed(0);
    // The page's HEAPU8 view is not updated when a worker grows memory: ask the runtime.
    const wasm = M?._skyrt_memory_bytes?.() ?? M?.HEAPU8?.buffer.byteLength;
    const heap = performance.memory?.usedJSHeapSize;
    const io = M?._skyrt_io_stats ? `, files read ${mib(M._skyrt_io_stats(0))} MiB (${mib(M._skyrt_io_stats(1))} MiB via Blobs), ${M._skyrt_io_stats(2)} open` : '';
    const lines = fullLog.slice(-40);
    lines.push(`(crumb: wasm ${wasm ? mib(wasm) : '?'} MiB${heap ? `, JS heap ${mib(heap)} MiB` : ''}${io})`);
    localStorage.setItem(CRUMBS_KEY, JSON.stringify({ session: logSession, at: Date.now(), up: performance.now(), clean, lines }));
  } catch {}
}
setInterval(saveCrumbs, 250);
addEventListener('pagehide', () => saveCrumbs(true));
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

// Copies files of the server's game folder into OPFS game/: the whole folder ("Import from
// server") or, before each boot of imported files, default.xex and $TitleUpdate/ (the runtime is
// recompiled against one default.xex, so an imported copy goes stale whenever the server's game
// folder is rebuilt). A file is fetched when it is missing or differs in size, and default.xex also
// when the server's changed since the last copy (its ETag is remembered), so an interrupted import
// resumes where it stopped. Each file is written under a temporary name and renamed into place
// once complete: a partial file never passes the size check.
const SYNCED_KEY = 'skyrt-synced-xex';
const PART_SUFFIX = '.skyrt-part';

// The server's manifest: [{ path, size }].
async function serverManifest() {
  const r = await fetch(STREAM_BASE + 'manifest');
  if (!r.ok) throw new Error(`game/manifest: HTTP ${r.status}`);
  return (await r.text()).split('\n').map((line) => line.split('\t'))
    .filter(([size, path]) => path && /^\d+$/.test(size))
    .map(([size, path]) => ({ path, size: Number(size) }));
}

class Cancelled extends Error {
  constructor() {
    super('cancelled');
    this.name = 'AbortError';
  }
}

// Writes `body` (a ReadableStream) to `dir`/`name` through a temporary file. `counted(n)` is
// called for every chunk; `signal` cancels (the temporary file is removed).
async function writeFileAtomically(dir, name, body, counted, signal) {
  const partName = name + PART_SUFFIX;
  const part = await dir.getFileHandle(partName, { create: true });
  try {
    const out = await part.createWritable();
    const counter = new TransformStream({
      transform(chunk, controller) {
        counted(chunk.byteLength);
        controller.enqueue(chunk);
      },
    });
    await body.pipeThrough(counter).pipeTo(out, { signal });
    if (typeof part.move === 'function') {
      await dir.removeEntry(name).catch(() => {});
      await part.move(name);
    } else {
      // No FileSystemHandle.move (older Safari): copy the finished file into place.
      const final = await (await dir.getFileHandle(name, { create: true })).createWritable();
      await (await part.getFile()).stream().pipeTo(final);
      await dir.removeEntry(partName);
    }
  } catch (e) {
    await dir.removeEntry(partName).catch(() => {});
    throw signal?.aborted ? new Cancelled() : e;
  }
}

// Copies the server files whose path passes `filter`. `options`: { signal, prune (also delete
// local files the server doesn't list; only for the whole folder), progress({ files, totalFiles,
// bytes, totalBytes, current, rate }) }. Returns { copied, bytes (fetched), deleted }.
async function copyFromServer(filter, options = {}) {
  const { signal, progress = () => {} } = options;
  const manifest = (await serverManifest()).filter((f) => filter(f.path));
  const head = await fetch(STREAM_BASE + 'files/default.xex', { method: 'HEAD' });
  const xexStamp = head.headers.get('ETag') ?? head.headers.get('Content-Length');
  let synced = null;
  try {
    synced = localStorage.getItem(SYNCED_KEY);
  } catch {}
  const root = await gameDir(true);
  const dirs = new Map([['', root]]);
  async function dirOf(parts) {
    let key = '';
    let dir = root;
    for (const part of parts) {
      key += '/' + part;
      if (!dirs.has(key)) dirs.set(key, await dir.getDirectoryHandle(part, { create: true }));
      dir = dirs.get(key);
    }
    return dir;
  }

  // What needs fetching.
  const todo = [];
  let doneFiles = 0, doneBytes = 0;
  for (const f of manifest) {
    const parts = f.path.split('/');
    const dir = await dirOf(parts.slice(0, -1));
    let local = null;
    try {
      local = await (await dir.getFileHandle(parts.at(-1))).getFile();
    } catch {}
    const stale = !local || local.size !== f.size || (f.path === 'default.xex' && synced !== xexStamp);
    if (stale) {
      todo.push({ ...f, dir, name: parts.at(-1), replacing: local?.size ?? 0 });
    } else {
      doneFiles += 1;
      doneBytes += f.size;
    }
  }
  const totalFiles = manifest.length;
  const totalBytes = manifest.reduce((n, f) => n + f.size, 0);

  // Room for it: what is fetched, less what it replaces (the old file goes once a new one is in place).
  const needed = todo.reduce((n, f) => n + f.size, 0);
  const largest = todo.reduce((n, f) => Math.max(n, f.size), 0);
  if (needed > 0 && navigator.storage?.estimate) {
    const { usage = 0, quota = Infinity } = await navigator.storage.estimate();
    const growth = needed - todo.reduce((n, f) => n + f.replacing, 0) + largest;
    if (growth > quota - usage) {
      throw new Error(`not enough storage: ${gib(needed)} to copy, ${gib(Math.max(0, quota - usage))} available to this page`);
    }
  }

  const started = performance.now();
  let fetched = 0, lastReport = 0, current = '';
  const report = (force) => {
    const now = performance.now();
    if (!force && now - lastReport < 250) return;
    lastReport = now;
    const rate = fetched / 2 ** 20 / Math.max(0.001, (now - started) / 1000);
    progress({ files: doneFiles, totalFiles, bytes: doneBytes, totalBytes, current, rate });
  };
  report(true);
  let copied = 0;
  for (const f of todo) {
    if (signal?.aborted) throw new Cancelled();
    current = f.path;
    report(true);
    const response = await fetch(STREAM_BASE + 'files/' + f.path.split('/').map(encodeURIComponent).join('/'), { signal })
      .catch((e) => { throw signal?.aborted ? new Cancelled() : e; });
    if (!response.ok) throw new Error(`${f.path}: HTTP ${response.status}`);
    try {
      await writeFileAtomically(f.dir, f.name, response.body, (n) => {
        fetched += n;
        doneBytes += n;
        report(false);
      }, signal);
    } catch (e) {
      if (e.name === 'QuotaExceededError') throw new Error(`storage is full (while writing ${f.path}); free some space and import again to resume`);
      throw e;
    }
    copied += 1;
    doneFiles += 1;
  }
  current = '';
  report(true);
  if (manifest.some((f) => f.path === 'default.xex')) {
    try {
      localStorage.setItem(SYNCED_KEY, xexStamp);
    } catch {}
  }

  let deleted = 0;
  if (options.prune) {
    const listed = new Set(manifest.map((f) => f.path));
    const prune = async (dir, prefix) => {
      for await (const [name, handle] of dir.entries()) {
        const path = prefix ? `${prefix}/${name}` : name;
        if (handle.kind === 'directory') {
          await prune(handle, path);
        } else if (!listed.has(path)) {
          await dir.removeEntry(name);
          deleted += 1;
        }
      }
    };
    await prune(root, '');
  }
  return { copied, bytes: fetched, deleted };
}

// Before booting imported files: default.xex and the title update.
async function syncFromServer() {
  if (!(await streamAvailable())) return;
  const { copied } = await copyFromServer((path) => path === 'default.xex' || path.startsWith('$TitleUpdate/'), {
    progress: (p) => {
      statusEl.textContent = p.current ? `Updating ${p.current} from the server…` : '';
    },
  });
  statusEl.textContent = copied ? `Updated ${copied} file(s) from the server.` : '';
  if (copied) log(`Updated ${copied} file(s) (default.xex / title update) from the server.`);
}

// "Import from server": the whole game folder, with progress and Cancel.
let serverImport = null;
$('pick-server').onclick = async () => {
  if (serverImport || running) return;
  serverImport = new AbortController();
  const bar = $('import-progress');
  bar.hidden = false;
  $('pick-server').disabled = true;
  $('import-bar').style.width = '0%';
  statusEl.textContent = '';
  const mib = (b) => `${(b / 2 ** 20).toFixed(0)} MiB`;
  try {
    const persisted = await navigator.storage?.persist?.().catch(() => false);
    if (!persisted) log('Storage is not marked persistent: the browser may evict the imported game when space runs low.');
    const done = await copyFromServer(() => true, {
      signal: serverImport.signal,
      prune: $('import-prune').checked,
      progress: (p) => {
        $('import-bar').style.width = `${(100 * p.bytes / Math.max(1, p.totalBytes)).toFixed(1)}%`;
        $('import-text').textContent = `${p.files} of ${p.totalFiles} files · ${gib(p.bytes)} of ${gib(p.totalBytes)}`
          + (p.current ? ` · ${p.rate.toFixed(1)} MiB/s` : '');
        $('import-file').textContent = p.current;
      },
    });
    const text = done.copied
      ? `Imported ${done.copied} file(s) (${mib(done.bytes)}) from the server; the rest already matched.`
      : 'Every file already matches the server.';
    statusEl.textContent = text + (done.deleted ? ` Deleted ${done.deleted} local file(s) the server doesn't have.` : '');
    log(statusEl.textContent);
  } catch (e) {
    statusEl.textContent = e.name === 'AbortError'
      ? 'Import cancelled; importing again resumes it.'
      : `Import from the server failed: ${e.message}`;
    log(statusEl.textContent);
  } finally {
    serverImport = null;
    bar.hidden = true;
    $('pick-server').disabled = running;
    showSource();
  }
};
$('import-cancel').onclick = () => serverImport?.abort();

async function showSource() {
  const stream = settings.source === 'stream';
  $('local-controls').hidden = stream;
  const offered = await streamAvailable();
  $('server-import-controls').hidden = !offered;
  $('source-status').textContent = stream
    ? (offered ? 'The server offers the game files.' : 'The server offers no game files (start it with --game-dir).')
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

// Renderers (`--gpu MODE`): the default renders the GPU command stream (Xenos emulation), d3d
// renders from the title's Direct3D calls.
const RENDERERS = {
  xenos: { name: 'Classic (Xenos emulation)', short: 'Xenos' },
  d3d: { name: 'Direct3D (experimental)', short: 'D3D' },
};
// The renderer of this page's boot, once booted.
let activeRenderer = null;

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
  if (settings.gpu === 'd3d') args.push('--gpu', 'd3d');
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

// Whether WebAssembly runs compiled: browsers can run it in an interpreter instead (Edge's
// "Enhance your security on the web" mode does on unfamiliar sites such as a LAN address), which
// makes the game about ten times slower. A tiny loop takes a few ms compiled (about 4 ms on a
// desktop, under 15 ms on an old phone) and far longer interpreted; the best of three runs
// counts, so a garbage collection or Safari's interpreter tier before tier-up don't.
const JIT_LOOP_ITERATIONS = 5_000_000;
const JIT_SLOW_MS = 30;

function checkWasmJit() {
  const warn = (text) => {
    $('jit-warning').textContent = text;
    $('jit-warning').hidden = false;
  };
  if (typeof WebAssembly === 'undefined') {
    log('Startup check: WebAssembly is unavailable in this browser.');
    warn('This browser has WebAssembly turned off, so the game can\'t run. In Edge, turn off '
      + '"Enhance your security on the web" for this site.');
    return;
  }
  // (func (param $n i32) (result i32) (local $acc i32)
  //   loop  acc = (acc + n) * 155; n -= 1; br_if n  end  acc)
  const bytes = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f, // type: (i32) -> i32
    0x03, 0x02, 0x01, 0x00, // function 0 has type 0
    0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x00, // export "f"
    0x0a, 0x1f, 0x01, 0x1d, 0x01, 0x01, 0x7f, // code: one i32 local
    0x03, 0x40, 0x20, 0x01, 0x20, 0x00, 0x6a, 0x41, 0x9b, 0x01, 0x6c, 0x21, 0x01,
    0x20, 0x00, 0x41, 0x01, 0x6b, 0x22, 0x00, 0x0d, 0x00, 0x0b, 0x20, 0x01, 0x0b,
  ]);
  try {
    const f = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports.f;
    let best = Infinity;
    for (let i = 0; i < 3; i++) {
      const t = performance.now();
      f(JIT_LOOP_ITERATIONS);
      best = Math.min(best, performance.now() - t);
    }
    const slow = best > JIT_SLOW_MS;
    log(`Startup check: WebAssembly test loop took ${best.toFixed(1)} ms (over ${JIT_SLOW_MS} ms means no JIT)${slow ? ': interpreted' : ''}`);
    if (slow) {
      warn('This browser is running WebAssembly without its JIT, so the game will be very slow. In Edge, turn off '
        + '"Enhance your security on the web" for this site.');
    }
  } catch (e) {
    log(`Startup check: WebAssembly test failed: ${e.message}`);
  }
}
setTimeout(checkWasmJit, 300);

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
  if (serverImport) {
    log('Wait for the import from the server to finish (or cancel it) before booting.');
    return;
  }
  $('boot').disabled = true;
  running = true;
  // Shown beside the phase, in the menu and in the perf overlay, so logs and screenshots say which ran.
  activeRenderer = RENDERERS[settings.gpu] ?? RENDERERS.xenos;
  $('boot-renderer').textContent = `${activeRenderer.short} renderer`;
  $('boot-renderer').title = activeRenderer.name;
  $('boot-renderer').hidden = false;
  $('menu-renderer').textContent = `Running with the ${activeRenderer.name} renderer.`;
  $('pick-server').disabled = true;
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
  log(`Renderer: ${activeRenderer.name}`);
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
