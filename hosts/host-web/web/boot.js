// Copies the user's game folder into OPFS (game/) and boots host-web.js.
// The runtime reads the files through WasmFS's OPFS backend.
'use strict';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const statusEl = $('status');

// The whole log (the page only shows its tail), and what hasn't been sent to the server yet.
const fullLog = [];
let unsent = [];
// The dev server (`cargo xtask serve`) keeps each page session's log in target/web-logs/.
const logSession = `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(36).slice(2, 8)}`;
let logUpload = true;

function log(line) {
  fullLog.push(line);
  unsent.push(line);
  logEl.textContent += line + '\n';
  if (logEl.textContent.length > 2_000_000) logEl.textContent = logEl.textContent.slice(-1_000_000);
  logEl.scrollTop = logEl.scrollHeight;
}

async function sendLog() {
  if (!logUpload || unsent.length === 0) return;
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

$('pick').onclick = async () => {
  if (!window.showDirectoryPicker) {
    statusEl.textContent = 'This browser has no directory picker; use a Chromium-based browser.';
    return;
  }
  try {
    const src = await window.showDirectoryPicker();
    const counter = { files: 0, bytes: 0 };
    await copyTree(src, await gameDir(true), counter);
    statusEl.textContent = `Imported ${counter.files} files (${(counter.bytes / 2 ** 30).toFixed(2)} GiB).`;
  } catch (e) {
    statusEl.textContent = `Import failed: ${e.message}`;
  }
};

$('clear').onclick = async () => {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry('game', { recursive: true }).catch(() => {});
  statusEl.textContent = 'Cleared.';
};

$('boot').onclick = async () => {
  if (!crossOriginIsolated) {
    log('Page is not cross-origin isolated (COOP/COEP); serve it with `cargo xtask serve`.');
    return;
  }
  try {
    await (await gameDir(false)).getFileHandle('default.xex');
  } catch {
    log('No game/default.xex in OPFS; import the game folder first.');
    return;
  }
  $('boot').disabled = true;
  // Audio needs a user gesture to start: set it up during this click.
  let audioNode = null;
  try {
    const context = new AudioContext({ sampleRate: 48000 });
    await context.audioWorklet.addModule('audio-worklet.js');
    audioNode = new AudioWorkletNode(context, 'skyrt-audio', { outputChannelCount: [2] });
    audioNode.connect(context.destination);
  } catch (e) {
    log(`Audio unavailable: ${e.message}`);
  }
  window.Module = {
    // Handed to the renderer thread as an OffscreenCanvas.
    canvas: $('canvas'),
    arguments: $('args').value.split(/\s+/).filter(Boolean),
    print: log,
    printErr: log,
    onRuntimeInitialized() {
      import('./input.js').then((m) => m.startInput(window.Module));
      audioNode?.port.postMessage({ memory: window.Module.HEAPU8.buffer, header: window.Module._skyrt_audio_ring() });
    },
  };
  const script = document.createElement('script');
  script.src = 'host-web.js';
  document.body.appendChild(script);
};
