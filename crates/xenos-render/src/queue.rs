//! Ordered hand-off from the command processor to the renderer thread.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::Command;

#[derive(Default)]
struct State {
    pending: VecDeque<(u64, Command)>,
    /// Sequence number of the last submitted command.
    submitted: u64,
    /// Sequence number of the last command the renderer finished.
    completed: u64,
    /// The renderer is gone; nothing will complete any more.
    closed: bool,
    /// The renderer is blocked in `take`, and how many threads are in `wait_idle`: wakeups are
    /// only sent to someone waiting (each is a futex wake, and on the web may wake a worker; a
    /// frame queues thousands of commands).
    renderer_waiting: bool,
    idle_waiters: u32,
    /// When each swap not yet executed was submitted, oldest first (for frame-latency reports).
    swaps: VecDeque<std::time::Instant>,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    /// Signalled on submission (renderer side) and on completion (waiter side).
    changed: Condvar,
    /// Bumped after every submission, for a renderer that can't block (on the web it waits for
    /// a change of this word from its event loop: see `park`).
    submissions: AtomicU32,
    /// Such a renderer is waiting on `submissions`: submissions wake it.
    parked: AtomicBool,
    /// `State::completed`, readable without the lock (polled by the command processor).
    completed: AtomicU64,
}

#[cfg(target_os = "emscripten")]
unsafe extern "C" {
    fn emscripten_futex_wake(addr: *const std::ffi::c_void, count: i32) -> i32;
}

/// Submits work to the renderer thread. Cheap to clone; usable from any thread.
#[derive(Clone, Default)]
pub struct RendererHandle {
    shared: Arc<Shared>,
}

impl RendererHandle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues `cmd`; returns its sequence number.
    pub fn submit(&self, cmd: Command) -> u64 {
        let mut s = self.shared.state.lock().unwrap();
        s.submitted += 1;
        let seq = s.submitted;
        if !s.closed {
            if matches!(cmd, Command::Swap { .. }) {
                s.swaps.push_back(std::time::Instant::now());
            }
            s.pending.push_back((seq, cmd));
        }
        let wake = s.renderer_waiting;
        drop(s);
        if wake {
            self.shared.changed.notify_all();
        }
        self.shared.submissions.fetch_add(1, Ordering::SeqCst);
        #[cfg(target_os = "emscripten")]
        if self.shared.parked.load(Ordering::SeqCst) {
            // SAFETY: a live word; waking waiters on it has no other effect.
            unsafe { emscripten_futex_wake(std::ptr::from_ref(&self.shared.submissions).cast(), i32::MAX) };
        }
        seq
    }

    /// Blocks until every command submitted so far has been processed (or the renderer has
    /// stopped).
    pub fn wait_idle(&self) {
        let mut s = self.shared.state.lock().unwrap();
        let target = s.submitted;
        s.idle_waiters += 1;
        while s.completed < target && !s.closed {
            s = self.shared.changed.wait(s).unwrap();
        }
        s.idle_waiters -= 1;
    }

    /// Blocks until the command with sequence number `seq` has been processed (or the renderer
    /// has stopped).
    pub fn wait_for(&self, seq: u64) {
        let mut s = self.shared.state.lock().unwrap();
        s.idle_waiters += 1;
        while s.completed < seq && !s.closed {
            s = self.shared.changed.wait(s).unwrap();
        }
        s.idle_waiters -= 1;
    }

    /// Renderer side: when the swap it is executing was submitted.
    pub(crate) fn take_swap_time(&self) -> Option<std::time::Instant> {
        self.shared.state.lock().unwrap().swaps.pop_front()
    }

    pub fn submitted(&self) -> u64 {
        self.shared.state.lock().unwrap().submitted
    }

    /// Sequence number of the last command the renderer finished (lock-free).
    pub fn completed(&self) -> u64 {
        self.shared.completed.load(Ordering::Acquire)
    }

    /// Renderer side: takes everything queued, waiting up to `timeout` if nothing is.
    pub(crate) fn take(&self, timeout: Option<Duration>) -> VecDeque<(u64, Command)> {
        let mut s = self.shared.state.lock().unwrap();
        if s.pending.is_empty()
            && let Some(t) = timeout
        {
            s.renderer_waiting = true;
            s = self.shared.changed.wait_timeout_while(s, t, |s| s.pending.is_empty() && !s.closed).unwrap().0;
            s.renderer_waiting = false;
        }
        std::mem::take(&mut s.pending)
    }

    /// Renderer side, for a renderer that waits from an event loop: if nothing is queued, marks
    /// it parked and returns the word to wait on and its current value (it changes with the next
    /// submission, which also wakes waiters on it). `None` if work is queued.
    #[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
    pub(crate) fn park(&self) -> Option<(&AtomicU32, u32)> {
        // Read the word before looking at the queue: a submission in between changes it.
        let value = self.shared.submissions.load(Ordering::SeqCst);
        self.shared.parked.store(true, Ordering::SeqCst);
        if !self.shared.state.lock().unwrap().pending.is_empty() {
            self.unpark();
            return None;
        }
        Some((&self.shared.submissions, value))
    }

    #[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
    pub(crate) fn unpark(&self) {
        self.shared.parked.store(false, Ordering::SeqCst);
    }

    /// Renderer side: everything up to `seq` is done.
    pub(crate) fn complete(&self, seq: u64) {
        let mut s = self.shared.state.lock().unwrap();
        s.completed = s.completed.max(seq);
        self.shared.completed.store(s.completed, Ordering::Release);
        let wake = s.idle_waiters > 0;
        drop(s);
        if wake {
            self.shared.changed.notify_all();
        }
    }

    /// Renderer side: the renderer stopped; release all waiters for good.
    pub(crate) fn close(&self) {
        let mut s = self.shared.state.lock().unwrap();
        s.closed = true;
        s.pending.clear();
        drop(s);
        self.shared.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swap() -> Command {
        Command::Swap { fetch: [0; 6], width: 1, height: 1 }
    }

    #[test]
    fn wait_idle_returns_once_everything_submitted_completes() {
        let h = RendererHandle::new();
        assert_eq!(h.submit(swap()), 1);
        assert_eq!(h.submit(swap()), 2);
        let renderer = h.clone();
        let t = std::thread::spawn(move || {
            let work = renderer.take(Some(Duration::from_secs(5)));
            assert_eq!(work.iter().map(|w| w.0).collect::<Vec<_>>(), [1, 2]);
            std::thread::sleep(Duration::from_millis(20));
            renderer.complete(2);
        });
        h.wait_idle();
        assert_eq!(h.completed(), 2);
        t.join().unwrap();
    }

    #[test]
    fn closing_releases_waiters() {
        let h = RendererHandle::new();
        h.submit(swap());
        let renderer = h.clone();
        let t = std::thread::spawn(move || renderer.close());
        h.wait_idle();
        t.join().unwrap();
        // Submissions after closing are dropped and never block.
        h.submit(swap());
        h.wait_idle();
    }
}
