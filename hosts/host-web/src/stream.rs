//! The game files streamed from the server the page came from (`cargo xtask serve --game-dir`)
//! instead of copied into OPFS: requests go through lib/http_files.js.

use std::ffi::{CStr, CString, c_char, c_void};
use std::io;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use gpu_webgpu::web::{Next, run_event_loop};
use vfs::remote::{DEFAULT_CACHE_CHUNKS, RangeReader, RemoteTree};

unsafe extern "C" {
    fn skyrt_http_get_text(url: *const c_char) -> *mut c_char;
    fn skyrt_fetch_range(
        base: *const c_char,
        path: *const c_char,
        offset: f64,
        dst: *mut u8,
        len: u32,
        done: extern "C" fn(*mut c_void, i32),
        arg: *mut c_void,
    );
    fn emscripten_futex_wake(addr: *const c_void, count: i32) -> i32;
    fn free(ptr: *mut c_void);
}

/// One range read: filled in by the fetch thread, waited for by the reading thread.
struct Job {
    path: CString,
    offset: u64,
    dst: usize,
    len: u32,
    result: Mutex<Option<i32>>,
    finished: Condvar,
}

/// Reads queued for the fetch thread. `posted` counts them; the thread sleeps until it changes.
struct Queue {
    jobs: Mutex<Vec<Arc<Job>>>,
    posted: AtomicU32,
}

/// Range reads of `<base>files/<path>`, made with fetch() on a thread of their own (see
/// `skyrt_fetch_range` in lib/http_files.js for why not a synchronous request) while the
/// reading thread blocks.
struct HttpRanges {
    queue: Arc<Queue>,
}

impl HttpRanges {
    fn new(files: CString) -> io::Result<Self> {
        let queue = Arc::new(Queue { jobs: Mutex::new(Vec::new()), posted: AtomicU32::new(0) });
        let thread_queue = queue.clone();
        std::thread::Builder::new().name("streaming".into()).stack_size(256 << 10).spawn(move || {
            // Requests run concurrently in the thread's event loop; each finishes in `done`.
            run_event_loop(move || {
                let seen = thread_queue.posted.load(Ordering::Acquire);
                for job in std::mem::take(&mut *thread_queue.jobs.lock().unwrap()) {
                    extern "C" fn done(arg: *mut c_void, n: i32) {
                        // SAFETY: `arg` is the Arc leaked when the request started.
                        let job = unsafe { Arc::from_raw(arg.cast::<Job>()) };
                        *job.result.lock().unwrap() = Some(n);
                        job.finished.notify_all();
                    }
                    let (path, offset, dst, len) = (job.path.as_ptr(), job.offset as f64, job.dst as *mut u8, job.len);
                    // SAFETY: valid C strings (kept alive by the job, which `done` releases);
                    // the reading thread keeps `dst` valid until the job finishes.
                    unsafe {
                        skyrt_fetch_range(
                            files.as_ptr(),
                            path,
                            offset,
                            dst,
                            len,
                            done,
                            Arc::into_raw(job).cast_mut().cast(),
                        )
                    };
                }
                Next::OnChange { addr: &thread_queue.posted, value: seen, timeout_ms: 1000 }
            });
        })?;
        Ok(HttpRanges { queue })
    }
}

impl RangeReader for HttpRanges {
    fn read(&self, path: &str, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let path_c = CString::new(path).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // Each request is capped; the tree asks again for the rest. Offsets fit an f64 exactly.
        let len = buf.len().min(MAX_REQUEST) as u32;
        let started = std::time::Instant::now();
        let job = Arc::new(Job {
            path: path_c,
            offset,
            dst: buf.as_mut_ptr() as usize,
            len,
            result: Mutex::new(None),
            finished: Condvar::new(),
        });
        self.queue.jobs.lock().unwrap().push(job.clone());
        self.queue.posted.fetch_add(1, Ordering::Release);
        // SAFETY: wakes the fetch thread's Atomics.waitAsync on the counter.
        unsafe { emscripten_futex_wake(std::ptr::from_ref(&self.queue.posted).cast(), i32::MAX) };
        let mut result = job.result.lock().unwrap();
        while result.is_none() {
            result = job.finished.wait(result).unwrap();
        }
        let n = result.expect("set above");
        let took = started.elapsed();
        STATS.note(len, took);
        if took >= SLOW_REQUEST {
            // Slow links (a phone on mobile data) otherwise look like a hang.
            log::info!(
                "streaming: {} bytes of {} took {:.1} s ({:.1} MiB/s)",
                len,
                path,
                took.as_secs_f64(),
                f64::from(len) / took.as_secs_f64() / f64::from(1u32 << 20)
            );
        }
        if n < 0 {
            return Err(io::Error::other(format!("streaming {path}: request failed")));
        }
        Ok(n as usize)
    }
}

/// Requests and the time readers spent waiting for them, logged every [`STATS_PERIOD`] while
/// requests are made: a title thread blocked on streaming shows up as hitches.
struct Stats {
    requests: AtomicU32,
    bytes: std::sync::atomic::AtomicU64,
    wait_us: std::sync::atomic::AtomicU64,
    max_us: std::sync::atomic::AtomicU64,
    last_report: Mutex<Option<std::time::Instant>>,
}

static STATS: Stats = Stats {
    requests: AtomicU32::new(0),
    bytes: std::sync::atomic::AtomicU64::new(0),
    wait_us: std::sync::atomic::AtomicU64::new(0),
    max_us: std::sync::atomic::AtomicU64::new(0),
    last_report: Mutex::new(None),
};

const STATS_PERIOD: std::time::Duration = std::time::Duration::from_secs(10);

impl Stats {
    fn note(&self, len: u32, took: std::time::Duration) {
        let us = took.as_micros() as u64;
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(u64::from(len), Ordering::Relaxed);
        self.wait_us.fetch_add(us, Ordering::Relaxed);
        self.max_us.fetch_max(us, Ordering::Relaxed);
        let mut last = self.last_report.lock().unwrap();
        let now = std::time::Instant::now();
        let since = *last.get_or_insert(now);
        if now - since < STATS_PERIOD {
            return;
        }
        *last = Some(now);
        let n = self.requests.swap(0, Ordering::Relaxed);
        log::info!(
            "streaming: {n} requests, {:.1} MiB in the last {:.0} s; readers waited {:.2} s in all (avg {:.1} ms, max {:.0} ms)",
            self.bytes.swap(0, Ordering::Relaxed) as f64 / f64::from(1u32 << 20),
            (now - since).as_secs_f64(),
            self.wait_us.load(Ordering::Relaxed) as f64 / 1e6,
            self.wait_us.swap(0, Ordering::Relaxed) as f64 / 1e3 / f64::from(n.max(1)),
            self.max_us.swap(0, Ordering::Relaxed) as f64 / 1e3
        );
    }
}

/// Requests taking at least this long are logged.
const SLOW_REQUEST: std::time::Duration = std::time::Duration::from_secs(2);

/// Largest single range request.
const MAX_REQUEST: usize = 16 << 20;

/// Fetches `<base>manifest` and returns the tree it describes.
pub fn open(base: &str) -> Result<Arc<RemoteTree>, String> {
    let base = if base.ends_with('/') { base.to_string() } else { format!("{base}/") };
    log::info!("streaming: fetching the game file list from {base}manifest");
    let url = CString::new(format!("{base}manifest")).map_err(|e| e.to_string())?;
    // SAFETY: a valid C string; the result is a malloc'd C string (or null) that we free.
    let manifest = unsafe {
        let text = skyrt_http_get_text(url.as_ptr());
        if text.is_null() {
            return Err(format!("no game manifest at {base}manifest (is the server running with --game-dir?)"));
        }
        let s = CStr::from_ptr(text).to_string_lossy().into_owned();
        free(text.cast());
        s
    };
    let files = CString::new(format!("{base}files/")).map_err(|e| e.to_string())?;
    let ranges = HttpRanges::new(files).map_err(|e| format!("starting the streaming thread: {e}"))?;
    let tree = RemoteTree::from_manifest(&manifest, Arc::new(ranges), DEFAULT_CACHE_CHUNKS);
    let Some((vfs::remote::RemoteKind::File { size }, _)) = tree.lookup("default.xex") else {
        return Err(format!("the streamed game at {base} has no default.xex"));
    };
    log::info!(
        "streaming the game from {base}: {} files listed; reading default.xex ({:.1} MiB) first",
        manifest.lines().count(),
        size as f64 / f64::from(1u32 << 20)
    );
    Ok(Arc::new(tree))
}
