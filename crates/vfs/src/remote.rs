//! Game files read over HTTP instead of from a local directory: the browser host streams them
//! from the server the page came from (`cargo xtask serve --game-dir`), so a device with little
//! storage (a phone) needn't hold a copy.
//!
//! The server lists the files in a manifest (one `size<TAB>path` line per file, `/`-separated);
//! file contents are read with range requests through a [`RangeReader`], in [`CHUNK_BYTES`]
//! chunks kept in a bounded LRU cache. Remote trees are read-only.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::sync::{Arc, Mutex};

/// Reads bytes of a remote file.
pub trait RangeReader: Send + Sync {
    /// Fills `buf` with the bytes of `path` (as listed in the manifest) from `offset` on;
    /// returns how many it got (fewer only at the end of the file).
    fn read(&self, path: &str, offset: u64, buf: &mut [u8]) -> io::Result<usize>;
}

/// Bytes per cached chunk (and per request, unless a read spans several).
pub const CHUNK_BYTES: u64 = 2 << 20;
/// Default cache size in chunks.
pub const DEFAULT_CACHE_CHUNKS: usize = 96;

#[derive(Debug, Clone)]
enum Entry {
    File { path: String, size: u64 },
    Dir { path: String },
}

/// A read-only directory tree described by a manifest, with file contents fetched on demand.
pub struct RemoteTree {
    /// By lower-cased path (`""` is the root).
    entries: HashMap<String, Entry>,
    /// Lower-cased directory path -> child names (original casing).
    children: HashMap<String, BTreeMap<String, ()>>,
    reader: Arc<dyn RangeReader>,
    cache: Mutex<ChunkCache>,
}

/// What a remote path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteKind {
    File { size: u64 },
    Dir,
}

impl RemoteTree {
    /// Builds the tree from manifest text. Lines that don't parse are skipped.
    pub fn from_manifest(manifest: &str, reader: Arc<dyn RangeReader>, cache_chunks: usize) -> Self {
        let mut tree = RemoteTree {
            entries: HashMap::new(),
            children: HashMap::new(),
            reader,
            cache: Mutex::new(ChunkCache::new(cache_chunks.max(4))),
        };
        tree.entries.insert(String::new(), Entry::Dir { path: String::new() });
        for line in manifest.lines() {
            let Some((size, path)) = line.split_once('\t') else { continue };
            let (Ok(size), path) = (size.trim().parse::<u64>(), path.trim_matches('/')) else { continue };
            if path.is_empty() || path.split('/').any(|c| c.is_empty() || c == "." || c == "..") {
                continue;
            }
            tree.add_file(path, size);
        }
        tree
    }

    fn add_file(&mut self, path: &str, size: u64) {
        let parts: Vec<&str> = path.split('/').collect();
        let mut dir = String::new();
        for (i, part) in parts.iter().enumerate() {
            let full = parts[..=i].join("/");
            let key = full.to_ascii_lowercase();
            self.children.entry(dir.to_ascii_lowercase()).or_default().insert((*part).to_string(), ());
            if i + 1 == parts.len() {
                self.entries.insert(key, Entry::File { path: full, size });
            } else {
                self.entries.entry(key).or_insert(Entry::Dir { path: full.clone() });
                dir = full;
            }
        }
    }

    /// The kind of `path` (any casing, `/`-separated, relative to the root), and its listed
    /// spelling.
    pub fn lookup(&self, path: &str) -> Option<(RemoteKind, String)> {
        match self.entries.get(&path.trim_matches('/').to_ascii_lowercase())? {
            Entry::File { path, size } => Some((RemoteKind::File { size: *size }, path.clone())),
            Entry::Dir { path } => Some((RemoteKind::Dir, path.clone())),
        }
    }

    /// Entries of directory `path`: (name, kind).
    pub fn list(&self, path: &str) -> Option<Vec<(String, RemoteKind)>> {
        let key = path.trim_matches('/').to_ascii_lowercase();
        if !matches!(self.entries.get(&key)?, Entry::Dir { .. }) {
            return None;
        }
        let names = self.children.get(&key).map(|c| c.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
        Some(
            names
                .into_iter()
                .filter_map(|name| {
                    let child = if key.is_empty() { name.clone() } else { format!("{key}/{name}") };
                    let kind = self.lookup(&child)?.0;
                    Some((name, kind))
                })
                .collect(),
        )
    }

    /// Reads `buf.len()` bytes of file `path` (listed spelling) at `offset` (fewer at the end).
    pub fn read_at(&self, path: &str, size: u64, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= size || buf.is_empty() {
            return Ok(0);
        }
        let end = (offset + buf.len() as u64).min(size);
        let first = offset / CHUNK_BYTES;
        let last = (end - 1) / CHUNK_BYTES;
        // The range's chunks: cached ones, then the missing ones fetched (one request per
        // contiguous run). Kept here so a read larger than the cache still has all of them.
        let mut chunks: HashMap<u64, Arc<[u8]>> = HashMap::new();
        {
            let mut cache = self.cache.lock().unwrap();
            for c in first..=last {
                if let Some(data) = cache.get(path, c) {
                    chunks.insert(c, data);
                }
            }
        }
        let missing: Vec<u64> = (first..=last).filter(|c| !chunks.contains_key(c)).collect();
        let mut i = 0;
        while i < missing.len() {
            let mut j = i;
            while j + 1 < missing.len() && missing[j + 1] == missing[j] + 1 {
                j += 1;
            }
            let start = missing[i] * CHUNK_BYTES;
            let stop = ((missing[j] + 1) * CHUNK_BYTES).min(size);
            let mut data = vec![0u8; (stop - start) as usize];
            let mut got = 0;
            while got < data.len() {
                let n = self.reader.read(path, start + got as u64, &mut data[got..])?;
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("{path}: short read")));
                }
                got += n;
            }
            let mut cache = self.cache.lock().unwrap();
            for (k, chunk) in data.chunks(CHUNK_BYTES as usize).enumerate() {
                let chunk: Arc<[u8]> = chunk.into();
                cache.insert(path, missing[i] + k as u64, chunk.clone());
                chunks.insert(missing[i] + k as u64, chunk);
            }
            i = j + 1;
        }
        let mut done = 0usize;
        let mut pos = offset;
        while pos < end {
            let chunk_index = pos / CHUNK_BYTES;
            let chunk = &chunks[&chunk_index];
            let within = (pos - chunk_index * CHUNK_BYTES) as usize;
            let n = ((end - pos) as usize).min(chunk.len() - within);
            buf[done..done + n].copy_from_slice(&chunk[within..within + n]);
            done += n;
            pos += n as u64;
        }
        Ok(done)
    }

    /// Reads a whole file (any casing).
    pub fn read_file(&self, path: &str) -> io::Result<Vec<u8>> {
        match self.lookup(path) {
            Some((RemoteKind::File { size }, listed)) => {
                let mut data = vec![0u8; size as usize];
                let n = self.read_at(&listed, size, 0, &mut data)?;
                data.truncate(n);
                Ok(data)
            }
            _ => Err(io::Error::from(io::ErrorKind::NotFound)),
        }
    }
}

/// Least recently used chunks, at most `capacity`.
struct ChunkCache {
    capacity: usize,
    tick: u64,
    chunks: HashMap<(String, u64), (Arc<[u8]>, u64)>,
}

impl ChunkCache {
    fn new(capacity: usize) -> Self {
        ChunkCache { capacity, tick: 0, chunks: HashMap::new() }
    }

    fn get(&mut self, path: &str, index: u64) -> Option<Arc<[u8]>> {
        self.tick += 1;
        let tick = self.tick;
        self.chunks.get_mut(&(path.to_string(), index)).map(|(data, used)| {
            *used = tick;
            data.clone()
        })
    }

    fn insert(&mut self, path: &str, index: u64, data: Arc<[u8]>) {
        // Room for this read's chunks too: evict down to capacity before inserting.
        while self.chunks.len() >= self.capacity {
            let Some(oldest) = self.chunks.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| k.clone()) else {
                break;
            };
            self.chunks.remove(&oldest);
        }
        self.tick += 1;
        self.chunks.insert((path.to_string(), index), (data, self.tick));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Files whose byte at offset `o` is `(o * 7 + seed) as u8`; counts requests.
    struct Pattern {
        requests: AtomicUsize,
    }

    impl RangeReader for Pattern {
        fn read(&self, path: &str, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
            self.requests.fetch_add(1, Ordering::Relaxed);
            let seed = path.len() as u64;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = ((offset + i as u64) * 7 + seed) as u8;
            }
            Ok(buf.len())
        }
    }

    fn tree(cache: usize) -> (RemoteTree, Arc<Pattern>) {
        let reader = Arc::new(Pattern { requests: AtomicUsize::new(0) });
        let manifest = format!(
            "{}\tData/Skyrim - Meshes.bsa\n12\tdefault.xex\nbad line\n5\tData/Video/BGS_LOGO.BIK\n",
            3 * CHUNK_BYTES + 10
        );
        (RemoteTree::from_manifest(&manifest, reader.clone(), cache), reader)
    }

    #[test]
    fn lookups_and_listings_are_case_insensitive() {
        let (t, _) = tree(8);
        assert_eq!(t.lookup("DATA/skyrim - meshes.BSA").unwrap().1, "Data/Skyrim - Meshes.bsa");
        assert_eq!(t.lookup("data").unwrap().0, RemoteKind::Dir);
        assert!(t.lookup("nope").is_none());
        let root: Vec<String> = t.list("").unwrap().into_iter().map(|e| e.0).collect();
        assert_eq!(root, ["Data", "default.xex"]);
        let data: Vec<(String, RemoteKind)> = t.list("data").unwrap();
        assert_eq!(data[0], ("Skyrim - Meshes.bsa".to_string(), RemoteKind::File { size: 3 * CHUNK_BYTES + 10 }));
        assert_eq!(data[1].1, RemoteKind::Dir);
    }

    #[test]
    fn reads_span_chunks_and_are_cached_with_a_bound() {
        let (t, reader) = tree(4);
        let path = "Data/Skyrim - Meshes.bsa";
        let size = 3 * CHUNK_BYTES + 10;
        let mut buf = vec![0u8; 100];
        let offset = CHUNK_BYTES - 50;
        assert_eq!(t.read_at(path, size, offset, &mut buf).unwrap(), 100);
        let seed = path.len() as u64;
        assert!(buf.iter().enumerate().all(|(i, b)| *b == ((offset + i as u64) * 7 + seed) as u8));
        assert_eq!(reader.requests.load(Ordering::Relaxed), 1, "two adjacent chunks in one request");
        t.read_at(path, size, offset + 10, &mut buf).unwrap();
        assert_eq!(reader.requests.load(Ordering::Relaxed), 1, "cached");
        // Reading at the end returns what's there.
        assert_eq!(t.read_at(path, size, size - 4, &mut buf).unwrap(), 4);
        assert_eq!(t.read_at(path, size, size, &mut buf).unwrap(), 0);
        // Many chunks through a 4-chunk cache stay correct.
        let mut whole = vec![0u8; size as usize];
        assert_eq!(t.read_at(path, size, 0, &mut whole[..CHUNK_BYTES as usize * 3]).unwrap(), 3 * CHUNK_BYTES as usize);
        assert_eq!(t.read_file("default.xex").unwrap().len(), 12);
        assert!(t.cache.lock().unwrap().chunks.len() <= 4);
    }
}
