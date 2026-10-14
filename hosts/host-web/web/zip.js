// Extracts a .zip of the game folder straight into an OPFS directory, one entry at a time, so
// the archive never has to fit in memory: only its central directory is read up front. Handles
// zip64 (game archives exceed 4 GiB) and the stored and deflate methods. If every entry sits
// under one top-level folder (a zipped "Skyrim/" folder), that folder is stripped unless the
// caller says not to. Also writes small stored (uncompressed) archives, for save exports.
'use strict';

const u16 = (v, o) => v.getUint16(o, true);
const u32 = (v, o) => v.getUint32(o, true);
const u64 = (v, o) => Number(v.getBigUint64(o, true));

async function bytes(file, start, end) {
  return new DataView(await file.slice(start, end).arrayBuffer());
}

// The end of central directory record (and its zip64 counterpart): where the directory is.
async function directoryLocation(file) {
  const tailStart = Math.max(0, file.size - (65535 + 22));
  const tail = await bytes(file, tailStart, file.size);
  let eocd = -1;
  for (let i = tail.byteLength - 22; i >= 0; i--) {
    if (u32(tail, i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('not a zip file (no end of central directory)');
  let count = u16(tail, eocd + 10);
  let size = u32(tail, eocd + 12);
  let offset = u32(tail, eocd + 16);
  if (count === 0xffff || size === 0xffffffff || offset === 0xffffffff) {
    // Zip64 locator just before the EOCD record.
    const loc = eocd - 20;
    if (loc < 0 || u32(tail, loc) !== 0x07064b50) throw new Error('zip64 locator missing');
    const recordAt = u64(tail, loc + 8);
    const record = await bytes(file, recordAt, recordAt + 56);
    if (u32(record, 0) !== 0x06064b50) throw new Error('zip64 end of central directory missing');
    count = u64(record, 32);
    size = u64(record, 40);
    offset = u64(record, 48);
  }
  return { count, size, offset };
}

function* entries(dir, count) {
  const decoder = new TextDecoder();
  let p = 0;
  for (let i = 0; i < count; i++) {
    if (u32(dir, p) !== 0x02014b50) throw new Error(`bad central directory entry ${i}`);
    const flags = u16(dir, p + 8);
    const method = u16(dir, p + 10);
    let compressed = u32(dir, p + 20);
    let size = u32(dir, p + 24);
    const nameLen = u16(dir, p + 28);
    const extraLen = u16(dir, p + 30);
    const commentLen = u16(dir, p + 32);
    let local = u32(dir, p + 42);
    const name = decoder.decode(new Uint8Array(dir.buffer, dir.byteOffset + p + 46, nameLen));
    // Zip64 extended information: the 0xFFFFFFFF fields, in this order.
    let e = p + 46 + nameLen;
    const end = e + extraLen;
    while (e + 4 <= end) {
      const id = u16(dir, e);
      const len = u16(dir, e + 2);
      if (id === 0x0001) {
        let q = e + 4;
        if (size === 0xffffffff) { size = u64(dir, q); q += 8; }
        if (compressed === 0xffffffff) { compressed = u64(dir, q); q += 8; }
        if (local === 0xffffffff) { local = u64(dir, q); q += 8; }
      }
      e += 4 + len;
    }
    if (flags & 1) throw new Error(`${name}: encrypted entries are not supported`);
    yield { name, method, compressed, size, local };
    p = end + commentLen;
  }
}

// Extracts `file` (a zip) into OPFS directory `dst`. `progress(files, bytes, name)` is called
// as it goes; `options.strip: false` keeps a shared top-level folder, and `options.accept(parts)`
// can refuse entries (it throws for a bad archive before anything is written). Returns
// { files, bytes }.
export async function extractZip(file, dst, progress, options = {}) {
  const where = await directoryLocation(file);
  const dir = await bytes(file, where.offset, where.offset + where.size);
  const list = [...entries(dir, where.count)].filter((e) => !e.name.endsWith('/'));
  for (const e of list) {
    if (e.method !== 0 && e.method !== 8) throw new Error(`${e.name}: compression method ${e.method} is not supported`);
  }
  // Strip one shared top-level folder, unless the game is already at the root.
  const tops = new Set(list.map((e) => e.name.split('/')[0]));
  const strip = options.strip !== false && tops.size === 1 && list.every((e) => e.name.includes('/'))
    ? [...tops][0].length + 1 : 0;
  if (options.accept) {
    for (const e of list) options.accept(e.name.slice(strip).split('/').filter((s) => s && s !== '.'));
  }

  let files = 0, total = 0;
  for (const e of list) {
    const parts = e.name.slice(strip).split('/').filter((s) => s && s !== '.');
    if (parts.length === 0 || parts.includes('..')) continue;
    let out = dst;
    for (const part of parts.slice(0, -1)) out = await out.getDirectoryHandle(part, { create: true });
    const header = await bytes(file, e.local, e.local + 30);
    if (u32(header, 0) !== 0x04034b50) throw new Error(`${e.name}: bad local header`);
    const start = e.local + 30 + u16(header, 26) + u16(header, 28);
    let data = file.slice(start, start + e.compressed).stream();
    if (e.method === 8) data = data.pipeThrough(new DecompressionStream('deflate-raw'));
    const writable = await (await out.getFileHandle(parts.at(-1), { create: true })).createWritable();
    await data.pipeTo(writable);
    files += 1;
    total += e.size;
    progress?.(files, total, parts.join('/'));
  }
  return { files, bytes: total };
}

// CRC-32 (IEEE), as zip entries record it.
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(data) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// MS-DOS date and time of `ms` (local time), as zip headers store them.
function dosTime(ms) {
  const d = new Date(ms);
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = (Math.max(0, d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return [time, date];
}

// A stored zip of `files` ([{ name, data: Uint8Array, modified: ms }], each under 4 GiB, the
// whole archive too: saves are a few MiB). Returns a Blob.
export function createZip(files) {
  const encoder = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;
  for (const f of files) {
    const name = encoder.encode(f.name);
    const crc = crc32(f.data);
    const [time, date] = dosTime(f.modified ?? Date.now());
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true); // UTF-8 names
    local.setUint16(8, 0, true);
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, f.data.length, true);
    local.setUint32(22, f.data.length, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    parts.push(local, name, f.data);
    const entry = new DataView(new ArrayBuffer(46));
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);
    entry.setUint16(6, 20, true);
    entry.setUint16(8, 0x0800, true);
    entry.setUint16(10, 0, true);
    entry.setUint16(12, time, true);
    entry.setUint16(14, date, true);
    entry.setUint32(16, crc, true);
    entry.setUint32(20, f.data.length, true);
    entry.setUint32(24, f.data.length, true);
    entry.setUint16(28, name.length, true);
    entry.setUint32(42, offset, true);
    central.push(entry, name);
    offset += 30 + name.length + f.data.length;
  }
  const centralSize = central.reduce((n, p) => n + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], { type: 'application/zip' });
}
