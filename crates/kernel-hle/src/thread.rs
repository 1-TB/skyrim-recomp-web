//! Guest threads as seen by the kernel.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use runtime_core::PPCContext;
use runtime_core::layout::{kpcr, kthread};
use runtime_core::thread::{ThreadBlock, creation_flags, spawn_host_thread};
use runtime_core::timebase;

use crate::Kernel;
use crate::dispatch::Call;
use crate::objects::{KObject, ObjectType, Payload, SyncState, UserApc};
use crate::status;

pub struct KThread {
    pub block: ThreadBlock,
    pub object: Arc<KObject>,
    pub start_address: u32,
    pub start_context: u32,
    pub xapi_startup: u32,
    pub creation_flags: u32,
    suspend_count: Mutex<u32>,
    resumed: Condvar,
    pub priority: AtomicI32,
    /// Kernel calls this thread has made (see [`note_kernel_call`]).
    kernel_calls: Arc<AtomicU64>,
    pub is_main: bool,
    /// Set by the title through the MSVC thread-naming exception.
    pub name: Mutex<Option<String>>,
}

impl KThread {
    pub fn tid(&self) -> u32 {
        self.block.thread_id
    }

    /// Returns the previous suspend count.
    pub fn resume(&self) -> u32 {
        let mut n = self.suspend_count.lock().unwrap();
        let prev = *n;
        if *n > 0 {
            *n -= 1;
            if *n == 0 {
                self.resumed.notify_all();
            }
        }
        prev
    }

    fn wait_until_resumed(&self) {
        let mut n = self.suspend_count.lock().unwrap();
        while *n > 0 {
            n = self.resumed.wait(n).unwrap();
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<KThread>>> = const { RefCell::new(None) };
}

pub fn current() -> Option<Arc<KThread>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Counts a kernel call made by the calling guest thread.
pub(crate) fn note_kernel_call() {
    CURRENT.with(|c| {
        if let Some(t) = &*c.borrow() {
            t.kernel_calls.fetch_add(1, Ordering::Release);
        }
    })
}

/// Kernel calls the calling guest thread has made (0 on host threads).
pub(crate) fn kernel_call_count() -> u64 {
    CURRENT.with(|c| c.borrow().as_ref().map_or(0, |t| t.kernel_calls.load(Ordering::Relaxed)))
}

/// The calling guest thread's kernel-call counter (None on host threads).
pub(crate) fn current_kernel_calls() -> Option<Arc<AtomicU64>> {
    CURRENT.with(|c| c.borrow().as_ref().map(|t| t.kernel_calls.clone()))
}

fn set_current(t: Option<Arc<KThread>>) {
    CURRENT.with(|c| *c.borrow_mut() = t);
}

/// Unwind payload used by ExTerminateThread to leave guest code.
pub struct ThreadExit(pub u32);

/// Terminates the calling guest thread by unwinding to its entry frame.
pub fn exit_current_thread(code: u32) -> ! {
    resume_unwind(Box::new(ThreadExit(code)))
}

/// NT timeout (100 ns units; negative = relative, positive = absolute
/// FILETIME, null pointer = infinite) to a deadline.
pub fn deadline_from_timeout(timeout: Option<i64>) -> Option<Instant> {
    let t = timeout?;
    let now = Instant::now();
    let ticks = if t < 0 { t.unsigned_abs() } else { (t as u64).saturating_sub(timebase::system_time()) };
    Some(now + Duration::from_nanos(ticks.saturating_mul(100)))
}

pub fn read_timeout(mem: &runtime_core::GuestMemory, ptr: u32) -> Option<i64> {
    (ptr != 0).then(|| mem.read_i64(ptr))
}

impl Kernel {
    fn new_kthread(&self, block: ThreadBlock, start: (u32, u32, u32), flags: u32, is_main: bool) -> Arc<KThread> {
        let mem = self.mem();
        let object = self.objects.register(
            mem,
            block.kthread,
            false,
            false,
            ObjectType::Thread,
            Some(SyncState::Thread { exited: false }),
            Payload::Thread { tid: block.thread_id },
            None,
        );
        // Reference held by the running thread itself.
        self.objects.add_ref(&object);
        let base_priority = mem.read_u8(block.kthread + kthread::BASE_PRIORITY) as i32;
        let t = Arc::new(KThread {
            block,
            object,
            start_address: start.0,
            start_context: start.1,
            xapi_startup: start.2,
            creation_flags: flags,
            suspend_count: Mutex::new((flags & creation_flags::SUSPENDED != 0) as u32),
            resumed: Condvar::new(),
            priority: AtomicI32::new(base_priority),
            kernel_calls: Arc::new(AtomicU64::new(0)),
            is_main,
            name: Mutex::new(None),
        });
        self.threads.lock().unwrap().insert(t.tid(), t.clone());
        t
    }

    /// ExCreateThread: allocates the guest structures and starts a host thread
    /// (blocked until resumed if created suspended).
    pub fn create_thread(
        &'static self,
        stack_size: u32,
        flags: u32,
        xapi_startup: u32,
        start_address: u32,
        start_context: u32,
    ) -> Result<Arc<KThread>, u32> {
        let stack_size = if stack_size == 0 { self.default_thread_stack_size() } else { stack_size };
        let parent_cpu = current().map(|p| self.mem().read_u8(p.block.pcr + kpcr::PRCB_CURRENT_CPU) as u32);
        let block = self
            .rt
            .threads
            .create(&self.rt.space, stack_size, flags, start_address, parent_cpu)
            .map_err(|_| status::NO_MEMORY)?;
        let t = self.new_kthread(block, (start_address, start_context, xapi_startup), flags, false);
        let runner = t.clone();
        spawn_host_thread(format!("guest-{:04X}", t.tid()), move || {
            runner.wait_until_resumed();
            self.run_thread(runner);
        })
        .map_err(|e| {
            log::error!("host thread creation failed: {e}");
            status::NO_MEMORY
        })?;
        Ok(t)
    }

    fn default_thread_stack_size(&self) -> u32 {
        let process = self.rt.threads.process();
        self.mem().read_u32(process + runtime_core::layout::kprocess::KERNEL_STACK_SIZE)
    }

    /// Starts the title's entry point on a new guest thread. Returns once the
    /// thread is running; [`Kernel::wait_for_exit`] blocks until the title ends.
    pub fn start_main_thread(&'static self) -> Result<Arc<KThread>, u32> {
        let entry = self.rt.image.entry_point;
        let block = self
            .rt
            .threads
            .create(
                &self.rt.space,
                self.rt.image.default_stack_size,
                creation_flags::TITLE_EXECUTION_THREAD,
                entry,
                None,
            )
            .map_err(|_| status::NO_MEMORY)?;
        let t = self.new_kthread(block, (entry, 0, 0), creation_flags::TITLE_EXECUTION_THREAD, true);
        let runner = t.clone();
        spawn_host_thread("guest-main".to_string(), move || self.run_thread(runner)).map_err(|e| {
            log::error!("host thread creation failed: {e}");
            status::NO_MEMORY
        })?;
        Ok(t)
    }

    /// Makes the calling host thread a guest thread (with its own stack, PCR and
    /// KTHREAD) without running guest code. Used by tests and host-side tools
    /// that call kernel functions directly.
    pub fn adopt_current_thread(&'static self) -> Result<Arc<KThread>, u32> {
        if let Some(t) = current() {
            return Ok(t);
        }
        let block = self.rt.threads.create(&self.rt.space, 0x10000, 0, 0, None).map_err(|_| status::NO_MEMORY)?;
        let t = self.new_kthread(block, (0, 0, 0), 0, false);
        set_current(Some(t.clone()));
        Ok(t)
    }

    /// A context for kernel calls made from an adopted thread.
    pub fn context_for(&self, t: &KThread) -> Box<PPCContext> {
        self.rt.new_context(&t.block)
    }

    fn run_thread(&'static self, t: Arc<KThread>) {
        set_current(Some(t.clone()));

        let mem = self.mem();
        mem.write_u8(t.block.kthread + kthread::THREAD_STATE, kthread::STATE_RUNNING);

        let mut ctx = self.rt.new_context(&t.block);
        let entry = if t.xapi_startup != 0 {
            ctx.set_gpr(3, t.start_address as u64);
            ctx.set_gpr(4, t.start_context as u64);
            t.xapi_startup
        } else {
            ctx.set_gpr(3, t.start_context as u64);
            t.start_address
        };
        log::info!(
            "thread {:#X} starting at {entry:#010X} (stack {:#010X}-{:#010X}, pcr {:#010X}, hardware thread {})",
            t.tid(),
            t.block.stack.limit,
            t.block.stack.base,
            t.block.pcr,
            mem.read_u8(t.block.pcr + kpcr::PRCB_CURRENT_CPU)
        );

        let outcome = catch_unwind(AssertUnwindSafe(|| self.rt.call(&mut ctx, entry)));
        let code = match outcome {
            Ok(Ok(())) => {
                // Without XAPI startup the entry's return value is the exit code.
                if t.xapi_startup == 0 { ctx.gpr32(3) } else { 0 }
            }
            Ok(Err(e)) => {
                log::error!("thread {:#X}: {e}", t.tid());
                status::UNSUCCESSFUL
            }
            Err(payload) => match payload.downcast::<ThreadExit>() {
                Ok(exit) => exit.0,
                Err(payload) => {
                    let msg = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unknown panic".into());
                    log::error!("kernel panic on guest thread {:#X}: {msg}", t.tid());
                    self.log_import_summary();
                    std::process::abort();
                }
            },
        };
        self.finish_thread(&t, code);
        set_current(None);
    }

    fn finish_thread(&self, t: &Arc<KThread>, code: u32) {
        let mem = self.mem();
        let k = t.block.kthread;
        mem.write_u8(k + kthread::TERMINATED, 1);
        mem.write_u32(k + kthread::EXIT_STATUS, code);
        mem.write_u64(k + kthread::EXIT_TIME, timebase::system_time());
        mem.write_u8(k + kthread::THREAD_STATE, kthread::STATE_TERMINATED);
        self.objects.thread_exited(mem, &t.object, t.tid());
        log::info!("thread {:#X} exited with {code:#X}", t.tid());
        if let Err(e) = self.rt.threads.release(&self.rt.space, &t.block) {
            log::error!("thread {:#X}: releasing stack/TLS failed: {e}", t.tid());
        }
        self.threads.lock().unwrap().remove(&t.tid());
        self.objects.release(&self.rt.space, &t.object);
        if t.is_main {
            self.request_exit(code);
        }
    }

    pub fn thread_by_id(&self, tid: u32) -> Option<Arc<KThread>> {
        self.threads.lock().unwrap().get(&tid).cloned()
    }

    pub fn thread_for_object(&self, obj: &KObject) -> Option<Arc<KThread>> {
        self.thread_by_id(obj.thread_id()?)
    }

    /// Runs queued user APCs on the calling thread. Returns how many ran.
    pub fn deliver_apcs(&self, call: &mut Call) -> usize {
        let tid = call.tid();
        let apcs = self.objects.take_apcs(tid);
        for UserApc { routine, context, arg1, arg2 } in &apcs {
            let saved = call.ctx.clone();
            call.ctx.set_gpr(3, *context as u64);
            call.ctx.set_gpr(4, *arg1 as u64);
            call.ctx.set_gpr(5, *arg2 as u64);
            if let Err(e) = self.rt.call(call.ctx, *routine) {
                log::error!("APC routine: {e}");
            }
            *call.ctx = saved;
        }
        apcs.len()
    }
}
