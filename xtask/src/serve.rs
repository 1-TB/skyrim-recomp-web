//! Static file server for the web build.
//!
//! SharedArrayBuffer (wasm threads) requires a cross-origin isolated page, so
//! every response carries COOP: same-origin and COEP: require-corp.
//!
//! `POST /log?session=ID` appends the request body to `target/web-logs/ID.log` (relative to the
//! directory the server runs in): the page sends its runtime log there as it goes.
//!
//! `POST /saves/PATH` stores the body as `target/web-saves/PATH` (the page's "Export saves"), and
//! `GET /saves/` lists what's there (one path per line) for "Import saves", which then fetches
//! each with `GET /saves/PATH`.
//!
//! Static files carry an ETag (size and modification time) and conditional requests are
//! answered with 304, so browsers keep the large module cached; a `.gz` copy next to a file (made
//! by `build-web`) is sent instead to clients that accept gzip.
//!
//! With `--game-dir DIR`, the game files are offered for streaming (the page's "stream from the
//! server" game source, for devices that can't hold a copy): `GET /game/manifest` lists every
//! file below DIR as `size<TAB>path` lines, and `GET /game/files/PATH` serves one, honouring
//! single `Range: bytes=` requests. The server only listens on the loopback interface; it is
//! reached from other devices through a private tunnel (e.g. `tailscale serve`), never directly.

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};

fn mime(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "map" => "application/json",
        _ => "application/octet-stream",
    }
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8], head_only: bool) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {ctype}\r\n\
         Content-Length: {}\r\n\
         Cross-Origin-Opener-Policy: same-origin\r\n\
         Cross-Origin-Embedder-Policy: require-corp\r\n\
         Cross-Origin-Resource-Policy: same-origin\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    if !head_only {
        stream.write_all(body)?;
    }
    Ok(())
}

/// Maps a request path onto `root`, refusing anything that escapes it.
fn resolve(root: &Path, url_path: &str) -> Option<PathBuf> {
    let path = url_path.split(['?', '#']).next().unwrap_or("/");
    let mut out = root.to_path_buf();
    for c in Path::new(path.trim_start_matches('/')).components() {
        match c {
            Component::Normal(p) => out.push(p),
            Component::CurDir => {}
            _ => return None,
        }
    }
    if out.is_dir() {
        out.push("index.html");
    }
    Some(out)
}

/// Largest log upload accepted in one request.
const MAX_LOG_BODY: usize = 8 << 20;

/// Appends `body` to the session's log file.
fn append_log(target: &str, body: &[u8]) -> Result<(), &'static str> {
    let session = target
        .split_once('?')
        .and_then(|(_, q)| q.split('&').find_map(|kv| kv.strip_prefix("session=")))
        .filter(|s| !s.is_empty() && s.len() <= 64)
        .filter(|s| s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
        .ok_or("bad session")?;
    let dir = Path::new("target/web-logs");
    std::fs::create_dir_all(dir).map_err(|_| "cannot create log directory")?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(format!("{session}.log")))
        .map_err(|_| "cannot open log")?;
    file.write_all(body).map_err(|_| "cannot write log")
}

/// Largest save file accepted.
const MAX_SAVE_BODY: usize = 64 << 20;
const SAVES_DIR: &str = "target/web-saves";

/// `target/web-saves/` + a request path below `/saves/`, refusing anything that escapes it.
fn save_path(url_path: &str) -> Option<PathBuf> {
    let rel = url_path.split(['?', '#']).next()?.strip_prefix("/saves/")?;
    let mut out = PathBuf::from(SAVES_DIR);
    for part in rel.split('/').filter(|p| !p.is_empty()) {
        let part = percent_decode(part)?;
        if part == "." || part == ".." || part.contains(['\\', '\0']) {
            return None;
        }
        out.push(part);
    }
    Some(out)
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `GET /game/manifest`: every file below the game directory with its size.
fn game_manifest(game: &Path) -> String {
    let mut files = Vec::new();
    list_files(game, "", &mut files);
    files.sort();
    let mut out = String::new();
    for f in files {
        if let Ok(m) = std::fs::metadata(game.join(&f)) {
            out.push_str(&format!("{}\t{f}\n", m.len()));
        }
    }
    out
}

/// A `bytes=START-END` / `bytes=START-` / `bytes=-SUFFIX` range of a `len`-byte file, as a
/// half-open interval; `None` if unsatisfiable or not a single range.
fn parse_range(value: &str, len: u64) -> Option<(u64, u64)> {
    let spec = value.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let (start, end) = match (a.trim(), b.trim()) {
        ("", suffix) => {
            let n: u64 = suffix.parse().ok()?;
            (len.saturating_sub(n), len)
        }
        (a, "") => (a.parse().ok()?, len),
        (a, b) => (a.parse().ok()?, b.parse::<u64>().ok()?.saturating_add(1).min(len)),
    };
    (start < end).then_some((start, end))
}

/// `GET /game/files/PATH`, streamed from disk (game archives are gigabytes).
fn serve_game_file(
    stream: &mut TcpStream,
    game: &Path,
    rel: &str,
    range: Option<&str>,
    head: bool,
) -> std::io::Result<()> {
    let mut path = game.to_path_buf();
    for part in rel.split('/').filter(|p| !p.is_empty()) {
        match percent_decode(part) {
            Some(p) if p != "." && p != ".." && !p.contains(['\\', '\0']) => path.push(p),
            _ => return respond(stream, "400 Bad Request", "text/plain", b"bad path", false),
        }
    }
    let Ok(mut file) = std::fs::File::open(&path) else {
        return respond(stream, "404 Not Found", "text/plain", b"not found", head);
    };
    let meta = file.metadata()?;
    let len = meta.len();
    // Same form as static files' ETag: the page compares it to notice a rebuilt default.xex.
    let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let etag = format!("\"{:x}-{:x}\"", len, modified.map_or(0, |d| d.as_nanos()));
    let (status, start, end) = match range {
        None => ("200 OK", 0, len),
        Some(r) => match parse_range(r, len) {
            Some((s, e)) => ("206 Partial Content", s, e),
            None => {
                let header = format!(
                    "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{len}\r\nContent-Length: 0\r\n\
                     Cross-Origin-Resource-Policy: same-origin\r\nConnection: close\r\n\r\n"
                );
                return stream.write_all(header.as_bytes());
            }
        },
    };
    let content_range =
        if range.is_some() { format!("Content-Range: bytes {start}-{}/{len}\r\n", end - 1) } else { String::new() };
    let header = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/octet-stream\r\n\
         Content-Length: {}\r\n\
         {content_range}\
         ETag: {etag}\r\n\
         Accept-Ranges: bytes\r\n\
         Cross-Origin-Opener-Policy: same-origin\r\n\
         Cross-Origin-Embedder-Policy: require-corp\r\n\
         Cross-Origin-Resource-Policy: same-origin\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n\r\n",
        end - start
    );
    stream.write_all(header.as_bytes())?;
    if head {
        return Ok(());
    }
    file.seek(SeekFrom::Start(start))?;
    std::io::copy(&mut file.take(end - start), stream)?;
    Ok(())
}

/// Every file below `dir`, as paths relative to it with `/` separators.
fn list_files(dir: &Path, prefix: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
        if e.path().is_dir() {
            list_files(&e.path(), &rel, out);
        } else {
            out.push(rel);
        }
    }
}

fn handle(mut stream: TcpStream, root: &Path, game: Option<&Path>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let mut content_length = 0usize;
    let mut range = None;
    let mut if_none_match = None;
    let mut accepts_gzip = false;
    let mut line = String::new();
    while reader.read_line(&mut line)? > 2 {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            } else if name.trim().eq_ignore_ascii_case("range") {
                range = Some(value.trim().to_string());
            } else if name.trim().eq_ignore_ascii_case("if-none-match") {
                if_none_match = Some(value.trim().to_string());
            } else if name.trim().eq_ignore_ascii_case("accept-encoding") {
                accepts_gzip = value.split(',').any(|e| e.trim().starts_with("gzip"));
            }
        }
        line.clear();
    }
    let mut parts = request.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");
    if method == "POST" && target.split('?').next() == Some("/log") {
        if content_length > MAX_LOG_BODY {
            return respond(&mut stream, "413 Payload Too Large", "text/plain", b"too large", false);
        }
        let mut body = vec![0; content_length];
        reader.read_exact(&mut body)?;
        return match append_log(target, &body) {
            Ok(()) => respond(&mut stream, "204 No Content", "text/plain", b"", false),
            Err(e) => respond(&mut stream, "400 Bad Request", "text/plain", e.as_bytes(), false),
        };
    }
    if target.starts_with("/saves/") {
        let Some(path) = save_path(target) else {
            return respond(&mut stream, "400 Bad Request", "text/plain", b"bad path", false);
        };
        if method == "POST" {
            if content_length > MAX_SAVE_BODY {
                return respond(&mut stream, "413 Payload Too Large", "text/plain", b"too large", false);
            }
            let mut body = vec![0; content_length];
            reader.read_exact(&mut body)?;
            let written =
                path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&path, &body));
            return match written {
                Ok(()) => respond(&mut stream, "204 No Content", "text/plain", b"", false),
                Err(_) => respond(&mut stream, "500 Internal Server Error", "text/plain", b"write failed", false),
            };
        }
        if method == "GET" && path == Path::new(SAVES_DIR) {
            let mut files = Vec::new();
            list_files(&path, "", &mut files);
            files.sort();
            return respond(&mut stream, "200 OK", "text/plain", (files.join("\n") + "\n").as_bytes(), false);
        }
        if method == "GET" {
            return match std::fs::read(&path) {
                Ok(body) => respond(&mut stream, "200 OK", "application/octet-stream", &body, false),
                Err(_) => respond(&mut stream, "404 Not Found", "text/plain", b"not found", false),
            };
        }
    }
    if method != "GET" && method != "HEAD" {
        return respond(&mut stream, "405 Method Not Allowed", "text/plain", b"method not allowed", false);
    }
    let head = method == "HEAD";
    let url_path = target.split(['?', '#']).next().unwrap_or("/");
    if url_path == "/game/manifest" || url_path.starts_with("/game/files/") {
        let Some(game) = game else {
            return respond(&mut stream, "404 Not Found", "text/plain", b"no --game-dir given to the server", head);
        };
        return match url_path.strip_prefix("/game/files/") {
            Some(rel) => serve_game_file(&mut stream, game, rel, range.as_deref(), head),
            None => respond(&mut stream, "200 OK", "text/plain; charset=utf-8", game_manifest(game).as_bytes(), head),
        };
    }
    let Some(path) = resolve(root, target).filter(|p| p.is_file()) else {
        return respond(&mut stream, "404 Not Found", "text/plain", b"not found", head);
    };
    serve_static(&mut stream, &path, if_none_match.as_deref(), accepts_gzip, head)
}

/// A static file, with an ETag, 304 for a matching `If-None-Match`, and the `.gz` copy when the
/// client accepts gzip and the copy is non-empty and at least as new.
fn serve_static(
    stream: &mut TcpStream,
    path: &Path,
    if_none_match: Option<&str>,
    accepts_gzip: bool,
    head: bool,
) -> std::io::Result<()> {
    let meta = std::fs::metadata(path)?;
    let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let etag = format!("\"{:x}-{:x}\"", meta.len(), modified.map_or(0, |d| d.as_nanos()));
    let common = "Cross-Origin-Opener-Policy: same-origin\r\n\
                  Cross-Origin-Embedder-Policy: require-corp\r\n\
                  Cross-Origin-Resource-Policy: same-origin\r\n\
                  Cache-Control: no-cache\r\n\
                  Vary: Accept-Encoding\r\n\
                  Connection: close\r\n";
    if if_none_match.is_some_and(|v| v.split(',').any(|t| t.trim() == etag)) {
        let header = format!("HTTP/1.1 304 Not Modified\r\nETag: {etag}\r\n{common}\r\n");
        return stream.write_all(header.as_bytes());
    }
    let gz = path.with_file_name(format!("{}.gz", path.file_name().unwrap_or_default().to_string_lossy()));
    // Not an empty copy (an interrupted compression) nor one older than the file.
    let gz_fresh = accepts_gzip
        && std::fs::metadata(&gz)
            .ok()
            .filter(|g| g.len() > 0)
            .and_then(|g| g.modified().ok())
            .zip(meta.modified().ok())
            .is_some_and(|(g, m)| g >= m);
    let (file_path, encoding) = if gz_fresh { (gz.as_path(), "Content-Encoding: gzip\r\n") } else { (path, "") };
    let mut file = std::fs::File::open(file_path)?;
    let len = file.metadata()?.len();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {len}\r\nETag: {etag}\r\n{encoding}{common}\r\n",
        mime(path)
    );
    stream.write_all(header.as_bytes())?;
    if !head {
        std::io::copy(&mut file, stream)?;
    }
    Ok(())
}

pub fn serve(root: &Path, port: u16, game: Option<&Path>) -> Result<(), String> {
    let root = root.canonicalize().map_err(|e| format!("{}: {e}", root.display()))?;
    let game = game.map(|g| g.canonicalize().map_err(|e| format!("{}: {e}", g.display()))).transpose()?;
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    println!("serving {} at http://127.0.0.1:{port}/ (cross-origin isolated)", root.display());
    if let Some(g) = &game {
        println!("streaming game files from {}", g.display());
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let (root, game) = (root.clone(), game.clone());
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &root, game.as_deref()) {
                eprintln!("serve: {e}");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn ranges() {
        assert_eq!(parse_range("bytes=0-99", 1000), Some((0, 100)));
        assert_eq!(parse_range("bytes=990-2000", 1000), Some((990, 1000)));
        assert_eq!(parse_range("bytes=500-", 1000), Some((500, 1000)));
        assert_eq!(parse_range("bytes=-10", 1000), Some((990, 1000)));
        assert_eq!(parse_range("bytes=1000-", 1000), None);
        assert_eq!(parse_range("bytes=0-1,5-6", 1000), None);
        assert_eq!(parse_range("items=0-1", 1000), None);
    }
}
