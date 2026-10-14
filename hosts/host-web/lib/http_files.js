// Game files streamed from the server (src/stream.rs): synchronous requests, made on the worker
// of the guest thread that reads (synchronous XHR is allowed in workers, and the guest expects
// file reads to block).
addToLibrary({
  // GET `url` as text; returns a malloc'd UTF-8 string, or null on failure.
  skyrt_http_get_text__deps: ['$UTF8ToString', '$stringToNewUTF8'],
  skyrt_http_get_text: (url) => {
    const target = UTF8ToString(url);
    try {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', target, false);
      xhr.send();
      if (xhr.status !== 200) {
        err(`GET ${target}: HTTP ${xhr.status}`);
        return 0;
      }
      return stringToNewUTF8(xhr.responseText);
    } catch (e) {
      err(`GET ${target}: ${e.message}`);
      return 0;
    }
  },

  // Starts reading `len` bytes of the file at `base` + `path` (path segments are URL-encoded
  // here) from byte `offset` into memory at `dst` with fetch(), on the calling worker, which
  // must return to its event loop; then calls `done(arg, n)` (a table index) with the number of
  // bytes read (fewer only at the end of the file), or -1 on failure. Transient failures (a
  // phone switching networks) are retried.
  //
  // Not a synchronous XMLHttpRequest from the reading thread: Blink keeps each of those
  // responses (MiBs) alive until the worker's garbage collector runs, which with its small JS
  // heap is rarely, and a world load piled up ~1 GiB of renderer memory that way. Here the body
  // is read in pieces into one reused buffer.
  skyrt_fetch_range__deps: ['$UTF8ToString', '$getWasmTableEntry', '$callUserCallback'],
  skyrt_fetch_range: (base, path, offset, dst, len, done, arg) => {
    const target = UTF8ToString(base) + UTF8ToString(path).split('/').map(encodeURIComponent).join('/');
    const finish = (n) => callUserCallback(() => getWasmTableEntry(done)(arg, n));
    // Copies the body bytes [skip, skip + len) to dst; returns how many there were.
    const readBody = async (body, skip) => {
      let got = 0;
      let pos = 0;
      const store = (chunk) => {
        const from = Math.max(0, skip - pos);
        const to = Math.min(chunk.length, skip + len - pos);
        if (to > from) {
          HEAPU8.set(chunk.subarray(from, to), dst + got);
          got += to - from;
        }
        pos += chunk.length;
      };
      // Bodies that aren't byte streams (older browsers) hand out chunks of their own.
      let reader;
      let byob = true;
      try {
        reader = body.getReader({ mode: 'byob' });
      } catch {
        reader = body.getReader();
        byob = false;
      }
      let finished = false;
      while (got < len) {
        let result;
        if (byob) {
          const buffer = globalThis.skyrtFetchBuffer ?? new ArrayBuffer(1 << 20);
          globalThis.skyrtFetchBuffer = null;
          result = await reader.read(new Uint8Array(buffer));
          // The buffer comes back (transferred) with the bytes read.
          if (result.value) globalThis.skyrtFetchBuffer = result.value.buffer;
        } else {
          result = await reader.read();
        }
        if (result.done) {
          finished = true;
          break;
        }
        store(result.value);
      }
      if (!finished) await reader.cancel();
      return got;
    };
    (async () => {
      let last = '';
      for (let attempt = 0; attempt < 4; attempt++) {
        try {
          const response = await fetch(target, { headers: { Range: `bytes=${offset}-${offset + len - 1}` } });
          if (response.status === 206 || response.status === 200) {
            // A server that ignores ranges sends the whole file.
            return finish(await readBody(response.body, response.status === 200 ? offset : 0));
          }
          await response.body?.cancel();
          if (response.status === 416) return finish(0);
          last = `HTTP ${response.status}`;
          if (response.status >= 400 && response.status < 500) break;
        } catch (e) {
          last = e.message;
        }
      }
      err(`reading ${target} at ${offset}: ${last}`);
      finish(-1);
    })();
  },
});
