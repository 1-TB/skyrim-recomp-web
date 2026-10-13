//! XOVERLAPPED completion for asynchronous XAM calls.
//!
//! Every XAM operation here finishes before the call returns, so "async"
//! requests complete immediately: the result is stored, the event signalled
//! and the completion routine queued as an APC, as Xenia does.

use std::time::Duration;

use crate::Kernel;
use crate::objects::UserApc;

/// Pause before a deferred completion lands. Xenia uses 25 ms: at least one
/// title (5454082B) loops forever if such a request finishes instantly.
const DEFERRED_DELAY: Duration = Duration::from_millis(25);

/// XOVERLAPPED (0x1C bytes).
mod field {
    pub const RESULT: u32 = 0x0;
    pub const LENGTH: u32 = 0x4;
    /// Handle of the thread that receives the completion APC.
    pub const CONTEXT: u32 = 0x8;
    pub const EVENT: u32 = 0xC;
    pub const COMPLETION_ROUTINE: u32 = 0x10;
    pub const EXTENDED_ERROR: u32 = 0x18;
}

impl Kernel {
    /// Completes `overlapped` with `result`; the length reads 0 on success and
    /// 0xFFFFFFFF on failure (some titles test it instead of the result).
    pub fn complete_overlapped(&self, overlapped: u32, result: u32) {
        let length = if result == 0 { 0 } else { u32::MAX };
        self.complete_overlapped_ex(overlapped, result, result, length);
    }

    pub fn complete_overlapped_ex(&self, overlapped: u32, result: u32, extended_error: u32, length: u32) {
        let mem = self.mem();
        mem.write_u32(overlapped + field::RESULT, result);
        mem.write_u32(overlapped + field::EXTENDED_ERROR, extended_error);
        mem.write_u32(overlapped + field::LENGTH, length);
        let event = mem.read_u32(overlapped + field::EVENT);
        if event != 0 {
            match self.objects.lookup(event, None) {
                Some(ev) => {
                    self.objects.set_event(mem, &ev);
                }
                None => log::warn!("XOVERLAPPED {overlapped:#010X}: event handle {event:#X} is not an object"),
            }
        }
        let routine = mem.read_u32(overlapped + field::COMPLETION_ROUTINE);
        if routine != 0 {
            let thread_handle = mem.read_u32(overlapped + field::CONTEXT);
            let current = crate::thread::current();
            let thread = self
                .objects
                .lookup(thread_handle, current.as_ref().map(|t| &t.object))
                .and_then(|o| self.thread_for_object(&o));
            match thread {
                Some(t) => self
                    .objects
                    .queue_apc(t.tid(), UserApc { routine, context: result, arg1: length, arg2: overlapped }),
                None => log::warn!(
                    "XOVERLAPPED {overlapped:#010X}: completion routine but no thread for handle {thread_handle:#X}"
                ),
            }
        }
    }

    /// Completes `overlapped` from another thread, as a system UI or service
    /// would: the request reads as pending until `run` returns (result,
    /// extended error, length). `pre`/`post` run around it (Xenia's
    /// CompleteOverlappedDeferredEx).
    pub fn complete_overlapped_deferred(
        &'static self,
        overlapped: u32,
        pre: impl FnOnce() + Send + 'static,
        run: impl FnOnce() -> (u32, u32, u32) + Send + 'static,
        post: impl FnOnce() + Send + 'static,
    ) {
        let mem = self.mem();
        mem.write_u32(overlapped + field::RESULT, crate::status::win32::IO_PENDING);
        // The completion APC goes to the thread that made the request.
        if let Some(t) = crate::thread::current() {
            let h = self.objects.insert_handle(&t.object);
            mem.write_u32(overlapped + field::CONTEXT, h);
        }
        let event = mem.read_u32(overlapped + field::EVENT);
        if event != 0
            && let Some(ev) = self.objects.lookup(event, None)
        {
            self.objects.reset_event(mem, &ev);
        }
        let spawned = std::thread::Builder::new().name("xam-dispatch".into()).spawn(move || {
            pre();
            std::thread::sleep(DEFERRED_DELAY);
            let (result, extended_error, length) = run();
            self.complete_overlapped_ex(overlapped, result, extended_error, length);
            post();
        });
        if let Err(e) = spawned {
            log::error!("deferred XOVERLAPPED completion: {e}");
        }
    }

    /// Result of a completed XOVERLAPPED.
    pub fn overlapped_result(&self, overlapped: u32) -> u32 {
        self.mem().read_u32(overlapped + field::RESULT)
    }
}
