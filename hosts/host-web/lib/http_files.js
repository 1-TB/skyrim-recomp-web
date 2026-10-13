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

  // Reads `len` bytes of the file at `base` + `path` (path segments are URL-encoded here) from
  // byte `offset` into memory at `dst`. Returns the number of bytes read (fewer only at the end
  // of the file), or -1 on failure. Transient failures (a phone switching networks) are retried.
  skyrt_http_read__deps: ['$UTF8ToString'],
  skyrt_http_read: (base, path, offset, dst, len) => {
    const target = UTF8ToString(base) + UTF8ToString(path).split('/').map(encodeURIComponent).join('/');
    let last = '';
    for (let attempt = 0; attempt < 4; attempt++) {
      try {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', target, false);
        xhr.responseType = 'arraybuffer';
        xhr.setRequestHeader('Range', `bytes=${offset}-${offset + len - 1}`);
        xhr.send();
        let body;
        if (xhr.status === 206) {
          body = new Uint8Array(xhr.response);
        } else if (xhr.status === 200) {
          // A server that ignores ranges sends the whole file.
          body = new Uint8Array(xhr.response).subarray(offset, offset + len);
        } else if (xhr.status === 416) {
          return 0;
        } else {
          last = `HTTP ${xhr.status}`;
          if (xhr.status >= 400 && xhr.status < 500) break;
          continue;
        }
        const n = Math.min(body.length, len);
        HEAPU8.set(body.subarray(0, n), dst);
        return n;
      } catch (e) {
        last = e.message;
      }
    }
    err(`reading ${target} at ${offset}: ${last}`);
    return -1;
  },
});
