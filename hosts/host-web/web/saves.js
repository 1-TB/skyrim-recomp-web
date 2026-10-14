// Save manager: lists the title content stored in this browser's OPFS content/ (where the runtime
// keeps it), and exports, imports and deletes packages. The runtime's layout (kernel-hle
// xam/content.rs, xam/profile.rs):
//   content/<XUID:016X>/<TitleID:08X>/<ContentType:08X>/<FileName>/...  one package (a save is
//                                                                         type 1: Savegame.dat)
//   content/<XUID:016X>/<TitleID:08X>/<ContentType:08X>/<FileName>.title its display name (UTF-8)
//   content/<XUID:016X>/<TitleID:08X>/profile_settings.bin               the title's profile settings
// Exports are zips of paths relative to content/, the same layout as the dev server's
// target/web-saves/ (Export/Import all), so either can be imported again.
'use strict';

import { createZip, extractZip } from './zip.js';

const HEX16 = /^[0-9A-F]{16}$/i;
const HEX8 = /^[0-9A-F]{8}$/i;
const CONTENT_TYPES = { 1: 'Save', 2: 'Downloadable content', 3: 'Publisher content' };

async function contentDir(create) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('content', { create });
}

// Every file below `dir`: [path relative to it, FileSystemFileHandle].
async function* walk(dir, prefix = '') {
  for await (const [name, handle] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === 'directory') yield* walk(handle, path);
    else yield [path, handle];
  }
}

async function entries(dir) {
  const out = [];
  for await (const entry of dir.entries()) out.push(entry);
  return out;
}

// The packages in content/: [{ path (relative to content/), xuid, title, type, fileName,
// displayName, size, modified, files: [[path relative to content/, handle]] }], newest first.
export async function listPackages() {
  let content;
  try {
    content = await contentDir(false);
  } catch {
    return [];
  }
  const packages = [];
  for (const [xuid, user] of await entries(content)) {
    if (user.kind !== 'directory' || !HEX16.test(xuid)) continue;
    for (const [title, titleDir] of await entries(user)) {
      if (titleDir.kind !== 'directory' || !HEX8.test(title)) continue;
      for (const [type, typeDir] of await entries(titleDir)) {
        if (typeDir.kind !== 'directory' || !HEX8.test(type)) continue;
        const items = await entries(typeDir);
        const titles = new Map(items.filter(([n, h]) => h.kind === 'file' && n.endsWith('.title')));
        for (const [fileName, dir] of items) {
          if (dir.kind !== 'directory') continue;
          const base = `${xuid}/${title}/${type}`;
          const files = [];
          let size = 0, modified = 0;
          for await (const [path, handle] of walk(dir, `${base}/${fileName}`)) {
            const file = await handle.getFile();
            files.push([path, handle]);
            size += file.size;
            modified = Math.max(modified, file.lastModified);
          }
          let displayName = fileName;
          const titleHandle = titles.get(`${fileName}.title`);
          if (titleHandle) {
            const file = await titleHandle.getFile();
            files.push([`${base}/${fileName}.title`, titleHandle]);
            size += file.size;
            displayName = (await file.text()).trim() || fileName;
          }
          packages.push({
            path: `${base}/${fileName}`, xuid, title, type: parseInt(type, 16), fileName, displayName, size,
            modified, files, typeDir,
          });
        }
      }
    }
  }
  packages.sort((a, b) => b.modified - a.modified);
  return packages;
}

// A zip of `files` ([path, handle]).
async function zipOf(files) {
  const list = [];
  for (const [name, handle] of files) {
    const file = await handle.getFile();
    list.push({ name, data: new Uint8Array(await file.arrayBuffer()), modified: file.lastModified });
  }
  return createZip(list);
}

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

const savesUrl = (path) => 'saves/' + path.split('/').map(encodeURIComponent).join('/');
const mib = (bytes) => bytes >= 2 ** 20 ? `${(bytes / 2 ** 20).toFixed(1)} MiB` : `${Math.max(1, Math.round(bytes / 1024))} KiB`;
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
const safeName = (s) => s.replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'save';

// Skyrim's display names read "Save 3 - Prisoner, Skyrim, 000.07.53": name, location, time played.
function describe(displayName) {
  const m = /^(.*), (.*), (\d+)\.(\d\d)\.(\d\d)$/.exec(displayName);
  if (!m) return [displayName, ''];
  return [m[1], `${m[2]} · ${Number(m[3])}h ${m[4]}m played`];
}

// Exports every file in content/ to the dev server (`cargo xtask serve` keeps them in
// target/web-saves/). Returns { files, bytes }.
export async function exportAllToServer() {
  let files = 0, bytes = 0;
  for await (const [path, handle] of walk(await contentDir(true))) {
    const file = await handle.getFile();
    const r = await fetch(savesUrl(path), { method: 'POST', body: file });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    files += 1;
    bytes += file.size;
  }
  return { files, bytes };
}

// Downloads every file in content/ as one zip.
export async function downloadAll() {
  const files = [];
  for await (const entry of walk(await contentDir(true))) files.push(entry);
  if (files.length === 0) throw new Error('nothing stored in this browser yet');
  download(await zipOf(files), `skyrim-saves-${stamp()}.zip`);
  return files.length;
}

// Whether the page's server keeps saves (`cargo xtask serve`'s /saves/).
export async function serverKeepsSaves() {
  try {
    const r = await fetch('saves/');
    return r.ok && (r.headers.get('Content-Type') ?? '').startsWith('text/plain');
  } catch {
    return false;
  }
}

// `options`: { log(line), toast(text), running(): whether the title runs (it keeps its content
// open, so importing and deleting wait for the next page load) }.
export function createSaveManager(options) {
  const $ = (id) => document.getElementById(id);
  const listEl = $('saves-list');
  const noteEl = $('saves-note');
  const say = (text) => {
    options.log(text);
    options.toast(text);
  };

  async function refresh() {
    let packages;
    try {
      packages = await listPackages();
    } catch (e) {
      listEl.replaceChildren(Object.assign(document.createElement('div'), { className: 'empty', textContent: `Can't read this browser's storage: ${e.message}` }));
      return;
    }
    const running = options.running();
    noteEl.textContent = running
      ? 'The game is running: import and delete are available again after reloading the page (the title keeps its saves open).'
      : 'Saves are stored in this browser only. Download them before clearing site data.';
    $('saves-import-zip').disabled = running;
    $('import-saves').disabled = running;
    if (packages.length === 0) {
      listEl.replaceChildren(Object.assign(document.createElement('div'), { className: 'empty table-wrap', textContent: 'No saves in this browser yet.' }));
      return;
    }
    const table = document.createElement('table');
    table.className = 'saves-table';
    table.innerHTML = '<thead><tr><th>Save</th><th>Size</th><th>Modified</th><th></th></tr></thead>';
    const body = document.createElement('tbody');
    for (const p of packages) {
      const [name, sub] = describe(p.displayName);
      const tr = document.createElement('tr');
      const nameTd = document.createElement('td');
      nameTd.className = 'name';
      const kind = CONTENT_TYPES[p.type] ?? `Type 0x${p.type.toString(16)}`;
      nameTd.innerHTML = '<div class="save-name"></div><div class="save-sub"></div>';
      nameTd.firstChild.textContent = name;
      nameTd.lastChild.textContent = [sub, p.type === 1 ? '' : kind].filter(Boolean).join(' · ') || p.fileName;
      nameTd.title = p.path;
      const size = document.createElement('td');
      size.className = 'num';
      size.textContent = mib(p.size);
      const date = document.createElement('td');
      date.className = 'num';
      date.textContent = p.modified ? new Date(p.modified).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '';
      const act = document.createElement('td');
      act.className = 'act';
      const exportButton = document.createElement('button');
      exportButton.className = 'small';
      exportButton.textContent = 'Download';
      exportButton.onclick = async () => {
        try {
          download(await zipOf(p.files), `${safeName(name)}.zip`);
        } catch (e) {
          say(`Download of "${name}" failed: ${e.message}`);
        }
      };
      const del = document.createElement('button');
      del.className = 'small danger';
      del.textContent = 'Delete';
      del.disabled = running;
      del.onclick = async () => {
        if (!confirm(`Delete "${p.displayName}" from this browser? Download it first to keep a copy.`)) return;
        try {
          await p.typeDir.removeEntry(p.fileName, { recursive: true });
          await p.typeDir.removeEntry(`${p.fileName}.title`).catch(() => {});
          say(`Deleted "${p.displayName}".`);
        } catch (e) {
          say(`Delete failed: ${e.message}`);
        }
        refresh();
      };
      act.append(exportButton, ' ', del);
      tr.append(nameTd, size, date, act);
      body.append(tr);
    }
    table.append(body);
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    wrap.append(table);
    listEl.replaceChildren(wrap);
  }

  $('saves-refresh').onclick = refresh;
  $('saves-download-all').onclick = async () => {
    try {
      say(`Downloading ${await downloadAll()} files as a zip.`);
    } catch (e) {
      say(`Download failed: ${e.message}`);
    }
  };
  $('saves-import-zip').onclick = () => $('saves-zip-file').click();
  $('saves-zip-file').onchange = async () => {
    const file = $('saves-zip-file').files[0];
    $('saves-zip-file').value = '';
    if (!file) return;
    try {
      // Paths relative to content/: <XUID>/<TitleID>/... (a download from here, or the server's layout).
      const accept = (parts) => {
        if (parts.length < 3 || !HEX16.test(parts[0]) || !HEX8.test(parts[1])) {
          throw new Error(`${parts.join('/')} is not in the content layout (<user>/<title id>/...); use a zip downloaded from this page`);
        }
      };
      const done = await extractZip(file, await contentDir(true), null, { strip: false, accept });
      say(`Imported ${done.files} files (${mib(done.bytes)}) from ${file.name}.`);
    } catch (e) {
      say(`Import failed: ${e.message}`);
    }
    refresh();
  };

  $('export-saves').onclick = async () => {
    try {
      const { files, bytes } = await exportAllToServer();
      say(`Exported ${files} save files (${mib(bytes)}) to the server.`);
    } catch (e) {
      say(`Export failed: ${e.message}`);
    }
  };
  $('import-saves').onclick = async () => {
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
      say(`Imported ${list.length} save files from the server.`);
    } catch (e) {
      say(`Import failed (import before booting): ${e.message}`);
    }
    refresh();
  };
  serverKeepsSaves().then((yes) => { $('server-saves').hidden = !yes; });

  refresh();
  return { refresh };
}
