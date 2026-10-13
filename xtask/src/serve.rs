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

use std::io::{BufRead, BufReader, Read, Write};
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

fn handle(mut stream: TcpStream, root: &Path) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let mut content_length = 0usize;
    let mut line = String::new();
    while reader.read_line(&mut line)? > 2 {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().unwrap_or(0);
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
            let written = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&path, &body));
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
    match resolve(root, target).and_then(|p| std::fs::read(&p).ok().map(|b| (p, b))) {
        Some((path, body)) => respond(&mut stream, "200 OK", mime(&path), &body, head),
        None => respond(&mut stream, "404 Not Found", "text/plain", b"not found", head),
    }
}

pub fn serve(root: &Path, port: u16) -> Result<(), String> {
    let root = root.canonicalize().map_err(|e| format!("{}: {e}", root.display()))?;
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    println!("serving {} at http://127.0.0.1:{port}/ (cross-origin isolated)", root.display());
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let root = root.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &root) {
                eprintln!("serve: {e}");
            }
        });
    }
    Ok(())
}
