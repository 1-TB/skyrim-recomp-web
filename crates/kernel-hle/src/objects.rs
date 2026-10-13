//! Kernel objects, the handle table, and the wait engine.
//!
//! Every object has a guest address: either a dispatcher structure the title
//! owns (a KEVENT on its stack, a critical section in .data — "native"
//! objects), or one the kernel allocated so `ObReferenceObjectByHandle` has
//! something to return.
//!
//! Synchronization state for all waitable objects lives under one lock so a
//! multi-object wait is evaluated atomically. Waiters register a wait block;
//! signalling an object satisfies blocked waiters in FIFO order (acquiring on
//! their behalf), as the NT dispatcher does, which keeps pulse and auto-reset
//! semantics exact. Each wait block has its own wake slot: a signal wakes only
//! the threads it satisfied, and they resume without retaking the dispatcher
//! lock. (A shared condition variable woke every blocked thread on every
//! signal; the resulting stampede on the dispatcher lock delayed the satisfied
//! thread by milliseconds.)
//!
//! Resetting a notification event first waits (bounded) until every thread
//! involved in its signals since the last reset -- the threads they released
//! and the threads that sent them -- has caught up, i.e. made another kernel
//! call. On hardware each of those threads keeps running on its own hardware
//! thread (a released one within microseconds), long before another thread
//! can come round to reset the event for the next round, and titles rely on
//! that. On the host either side can sit preempted for milliseconds: a woken
//! thread waiting for a CPU, or the signaller preempted by the thread it just
//! woke. Skyrim's job system does not survive that: a worker that is late
//! leaving a frame's job loop (after setting or passing the last barrier's
//! event), or late joining it (released by a frame event partway through),
//! enters the next frame's job list after the main thread has cleared those
//! events and restarted the list, the job barriers then count one
//! participant too many, and the frame deadlocks.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use runtime_core::AddressSpace;
use runtime_core::layout::{dispatch_header as dh, object_type};
use runtime_core::memory::GuestMemory;

use crate::status;

pub type ObjId = u64;

pub const HANDLE_BASE: u32 = 0xF800_0000;
pub const CURRENT_PROCESS: u32 = 0xFFFF_FFFF;
pub const CURRENT_THREAD: u32 = 0xFFFF_FFFE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    Event,
    Semaphore,
    Mutant,
    Thread,
    File,
    /// KTIMER: armed by [`crate::timer`], waited on like an event
    /// (notification timers stay signaled, synchronization timers auto-reset).
    Timer,
    /// Opened by NtOpenSymbolicLinkObject; not waitable.
    SymbolicLink,
    /// Dispatcher type the kernel can wait on but never signals (queues):
    /// only the title's own code changes its state.
    Other(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    Event { manual_reset: bool, signaled: bool },
    Semaphore { count: i32, limit: i32 },
    Mutant { owner: Option<u32>, recursion: u32, abandoned: bool },
    Thread { exited: bool },
    AlwaysSignaled,
}

impl SyncState {
    /// Value mirrored into DISPATCHER_HEADER.SignalState.
    fn signal_state(&self) -> i32 {
        match *self {
            SyncState::Event { signaled, .. } => signaled as i32,
            SyncState::Semaphore { count, .. } => count,
            SyncState::Mutant { owner: None, .. } => 1,
            SyncState::Mutant { recursion, .. } => 1 - recursion as i32,
            SyncState::Thread { exited } => exited as i32,
            SyncState::AlwaysSignaled => 1,
        }
    }

    fn signaled_for(&self, tid: u32) -> bool {
        match *self {
            SyncState::Event { signaled, .. } => signaled,
            SyncState::Semaphore { count, .. } => count > 0,
            SyncState::Mutant { owner, .. } => owner.is_none_or(|o| o == tid),
            SyncState::Thread { exited } => exited,
            SyncState::AlwaysSignaled => true,
        }
    }

    /// Consumes the signal for `tid`; returns true if a mutant was abandoned.
    fn acquire(&mut self, tid: u32) -> bool {
        match self {
            SyncState::Event { manual_reset: false, signaled } => *signaled = false,
            SyncState::Semaphore { count, .. } => *count -= 1,
            SyncState::Mutant { owner, recursion, abandoned } => {
                *owner = Some(tid);
                *recursion += 1;
                return std::mem::take(abandoned);
            }
            _ => {}
        }
        false
    }
}

/// Per-type extra data.
pub enum Payload {
    None,
    File(crate::io::FileObject),
    Thread { tid: u32 },
    NotifyListener(Arc<crate::xam::notify::Listener>),
    Enumerator(crate::xam::content::Enumerator),
    // A symbolic link's target (`\Device\...`).
    SymbolicLink(String),
}

pub struct KObject {
    pub id: ObjId,
    pub ty: ObjectType,
    pub guest_ptr: u32,
    /// Allocated by the kernel (freed when the last reference goes away).
    owns_guest_ptr: bool,
    /// Lives in title-owned memory; never destroyed.
    pub native: bool,
    refs: AtomicU32,
    pub name: Option<String>,
    pub payload: Payload,
}

impl KObject {
    pub fn file(&self) -> Option<&crate::io::FileObject> {
        match &self.payload {
            Payload::File(f) => Some(f),
            _ => None,
        }
    }

    pub fn thread_id(&self) -> Option<u32> {
        match self.payload {
            Payload::Thread { tid } => Some(tid),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserApc {
    pub routine: u32,
    pub context: u32,
    pub arg1: u32,
    pub arg2: u32,
}

const NO_RESULT: u64 = u64::MAX;

/// A thread involved in a signal, tracked until it has caught up (made a
/// kernel call since it was armed; host threads, which have no counter, catch
/// up as soon as they are armed).
struct Participant {
    kernel_calls: Option<Arc<AtomicU64>>,
    armed: AtomicBool,
    mark: AtomicU64,
}

impl Participant {
    /// The calling thread, not yet armed.
    fn current() -> Self {
        Participant {
            kernel_calls: crate::thread::current_kernel_calls(),
            armed: AtomicBool::new(false),
            mark: AtomicU64::new(0),
        }
    }

    /// Starts counting from the thread's position now (called on the
    /// participant's own thread).
    fn arm(&self) {
        if let Some(calls) = &self.kernel_calls {
            self.mark.store(calls.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        self.armed.store(true, Ordering::Release);
    }

    fn caught_up(&self) -> bool {
        self.armed.load(Ordering::Acquire)
            && self.kernel_calls.as_ref().is_none_or(|c| c.load(Ordering::Acquire) != self.mark.load(Ordering::Relaxed))
    }
}

/// Where a blocked thread sleeps; set by whoever satisfies (or alerts) it.
struct WaitSlot {
    result: AtomicU64,
    apc: AtomicBool,
    /// The waiting thread, armed when it returns from the wait.
    waiter: Arc<Participant>,
    sleep: Mutex<()>,
    cv: Condvar,
}

/// Upper bound on how long resetting an event waits for the threads involved
/// in its signals (see the module docs); only a badly overloaded host gets
/// near it.
const RESET_GRACE: Duration = Duration::from_millis(50);

impl WaitSlot {
    fn new() -> Self {
        WaitSlot {
            result: AtomicU64::new(NO_RESULT),
            apc: AtomicBool::new(false),
            waiter: Arc::new(Participant::current()),
            sleep: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    fn mark_resumed(&self) {
        self.waiter.arm();
    }

    fn result(&self) -> Option<u32> {
        let r = self.result.load(Ordering::Acquire);
        (r != NO_RESULT).then_some(r as u32)
    }

    fn wake(&self) {
        drop(self.sleep.lock().unwrap());
        self.cv.notify_one();
    }

    fn post_result(&self, r: u32) {
        self.result.store(r as u64, Ordering::Release);
        self.wake();
    }

    fn post_apc(&self) {
        self.apc.store(true, Ordering::Release);
        self.wake();
    }
}

/// A blocked wait. Blocks stay in the dispatcher's list only while unsatisfied.
struct WaitBlock {
    id: u64,
    tid: u32,
    objects: Vec<ObjId>,
    wait_all: bool,
    alertable: bool,
    timed: bool,
    /// Guest return addresses of the waiting call, for diagnostics.
    site: Vec<u32>,
    slot: Arc<WaitSlot>,
}

/// A blocked wait, for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaiterInfo {
    pub tid: u32,
    pub wait_all: bool,
    pub timed: bool,
    /// Guest return addresses of the waiting call, innermost first.
    pub site: Vec<u32>,
    /// (guest pointer, state) of each object waited on.
    pub objects: Vec<(u32, Option<SyncState>)>,
}

#[derive(Default)]
struct DispState {
    sync: HashMap<ObjId, SyncState>,
    /// Guest dispatcher header for each waitable object (state mirror target).
    headers: HashMap<ObjId, u32>,
    waiters: Vec<WaitBlock>,
    apcs: HashMap<u32, VecDeque<UserApc>>,
    next_wait: u64,
    /// Threads involved in each notification event's signals since its last
    /// reset that may not have caught up yet.
    involved: HashMap<ObjId, Vec<Arc<Participant>>>,
}

impl DispState {
    fn mirror(&self, mem: &GuestMemory, id: ObjId) {
        if let (Some(s), Some(&ptr)) = (self.sync.get(&id), self.headers.get(&id)) {
            mem.write_i32(ptr + dh::SIGNAL_STATE, s.signal_state());
        }
    }

    /// Acquires the objects for `tid` if the wait can be satisfied now.
    fn try_satisfy(&mut self, mem: &GuestMemory, tid: u32, objects: &[ObjId], wait_all: bool) -> Option<u32> {
        if wait_all {
            if !objects.iter().all(|o| self.sync.get(o).is_some_and(|s| s.signaled_for(tid))) {
                return None;
            }
            let mut abandoned = false;
            for o in objects {
                abandoned |= self.sync.get_mut(o).unwrap().acquire(tid);
                self.mirror(mem, *o);
            }
            return Some(if abandoned { status::ABANDONED_WAIT_0 } else { status::SUCCESS });
        }
        for (i, o) in objects.iter().enumerate() {
            let s = self.sync.get_mut(o)?;
            if s.signaled_for(tid) {
                let abandoned = s.acquire(tid);
                self.mirror(mem, *o);
                let base = if abandoned { status::ABANDONED_WAIT_0 } else { status::SUCCESS };
                return Some(base + i as u32);
            }
        }
        None
    }

    /// Satisfies blocked waiters in arrival order after a state change, waking
    /// each satisfied thread directly. Returns whether any was satisfied.
    fn wake_waiters(&mut self, mem: &GuestMemory) -> bool {
        let mut woke = false;
        let mut i = 0;
        while i < self.waiters.len() {
            let (tid, wait_all) = (self.waiters[i].tid, self.waiters[i].wait_all);
            let objects = std::mem::take(&mut self.waiters[i].objects);
            let r = self.try_satisfy(mem, tid, &objects, wait_all);
            self.waiters[i].objects = objects;
            match r {
                Some(r) => {
                    let w = self.waiters.remove(i);
                    self.note_released(&w, r);
                    w.slot.post_result(r);
                    woke = true;
                }
                None => i += 1,
            }
        }
        woke
    }

    /// Records a satisfied waiter against the notification events that
    /// released it (the one that satisfied a wait-any, every object of a
    /// wait-all).
    fn note_released(&mut self, w: &WaitBlock, result: u32) {
        let index = if result >= status::ABANDONED_WAIT_0 {
            result - status::ABANDONED_WAIT_0
        } else {
            result.wrapping_sub(status::SUCCESS)
        } as usize;
        let sources: &[ObjId] = if w.wait_all { &w.objects } else { w.objects.get(index..=index).unwrap_or(&[]) };
        for id in sources {
            self.note_involved(*id, &w.slot.waiter);
        }
    }

    /// Adds a participant to a notification event's current round.
    fn note_involved(&mut self, id: ObjId, p: &Arc<Participant>) {
        if matches!(self.sync.get(&id), Some(SyncState::Event { manual_reset: true, .. })) {
            let list = self.involved.entry(id).or_default();
            list.retain(|p| !p.caught_up());
            list.push(p.clone());
        }
    }

    /// Takes a wait block out of the list; false if a signal already
    /// satisfied (and removed) it.
    fn remove_waiter(&mut self, id: u64) -> bool {
        match self.waiters.iter().position(|w| w.id == id) {
            Some(pos) => {
                self.waiters.remove(pos);
                true
            }
            None => false,
        }
    }
}

#[derive(Default)]
struct Tables {
    handles: HashMap<u32, Arc<KObject>>,
    by_ptr: HashMap<u32, Arc<KObject>>,
    names: HashMap<String, Arc<KObject>>,
    next_handle: u32,
}

pub struct ObjectManager {
    disp: Mutex<DispState>,
    /// Read-mostly (every handle and pointer lookup), so a reader/writer lock.
    tables: std::sync::RwLock<Tables>,
    next_id: AtomicU64,
}

/// Outcome of a wait, before APC delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitOutcome {
    /// NTSTATUS (SUCCESS+index, ABANDONED_WAIT_0+index).
    Satisfied(u32),
    Timeout,
    /// Alertable wait interrupted by a queued user APC.
    Apc,
}

impl Default for ObjectManager {
    fn default() -> Self {
        Self::new()
    }
}

fn header_size(ty: ObjectType) -> u32 {
    match ty {
        ObjectType::Event => 0x10,
        ObjectType::Semaphore => 0x14,
        ObjectType::Mutant => 0x20,
        // Opaque body for objects that are only referenced by pointer.
        _ => 0x40,
    }
}

fn type_byte(state: &SyncState) -> Option<u8> {
    Some(match state {
        SyncState::Event { manual_reset: true, .. } => object_type::EVENT_NOTIFICATION,
        SyncState::Event { manual_reset: false, .. } => object_type::EVENT_SYNCHRONIZATION,
        SyncState::Semaphore { .. } => object_type::SEMAPHORE,
        SyncState::Mutant { .. } => object_type::MUTANT,
        SyncState::Thread { .. } => object_type::THREAD,
        SyncState::AlwaysSignaled => return None,
    })
}

impl ObjectManager {
    pub fn new() -> Self {
        ObjectManager {
            disp: Mutex::new(DispState::default()),
            tables: std::sync::RwLock::new(Tables { next_handle: 1, ..Default::default() }),
            next_id: AtomicU64::new(1),
        }
    }

    fn lock(&self) -> MutexGuard<'_, DispState> {
        self.disp.lock().unwrap()
    }

    /// Creates a kernel-owned object with its own guest body. The object starts
    /// with no references; callers add a handle or pointer reference.
    pub fn create(
        &self,
        space: &AddressSpace,
        ty: ObjectType,
        sync: Option<SyncState>,
        payload: Payload,
        name: Option<String>,
    ) -> Result<Arc<KObject>, u32> {
        let ptr = space.system_alloc(header_size(ty), 16).map_err(|_| status::NO_MEMORY)?;
        let mem = &space.mem;
        if let Some(s) = &sync {
            if let Some(t) = type_byte(s) {
                mem.write_u8(ptr + dh::TYPE, t);
                mem.write_u8(ptr + dh::SIZE_FIELD, (header_size(ty) / 4) as u8);
                runtime_core::thread::init_list_head(mem, ptr + dh::WAIT_LIST_FLINK);
            }
            if let SyncState::Semaphore { limit, .. } = s {
                mem.write_i32(ptr + 0x10, *limit);
            }
        }
        Ok(self.register(mem, ptr, true, false, ty, sync, payload, name))
    }

    /// Registers an object whose guest structure already exists (a KTHREAD the
    /// runtime allocated, or a title-owned dispatcher object).
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        mem: &GuestMemory,
        guest_ptr: u32,
        owns_guest_ptr: bool,
        native: bool,
        ty: ObjectType,
        sync: Option<SyncState>,
        payload: Payload,
        name: Option<String>,
    ) -> Arc<KObject> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let obj = Arc::new(KObject {
            id,
            ty,
            guest_ptr,
            owns_guest_ptr,
            native,
            refs: AtomicU32::new(0),
            name: name.clone(),
            payload,
        });
        if let Some(s) = sync {
            let mut d = self.lock();
            d.sync.insert(id, s);
            d.headers.insert(id, guest_ptr);
            d.mirror(mem, id);
        }
        let mut t = self.tables.write().unwrap();
        t.by_ptr.insert(guest_ptr, obj.clone());
        if let Some(n) = name {
            t.names.insert(n.to_ascii_lowercase(), obj.clone());
        }
        obj
    }

    pub fn add_ref(&self, obj: &KObject) {
        obj.refs.fetch_add(1, Ordering::AcqRel);
    }

    /// Drops one reference; destroys kernel-owned objects at zero.
    pub fn release(&self, space: &AddressSpace, obj: &Arc<KObject>) {
        let prev = obj.refs.fetch_sub(1, Ordering::AcqRel);
        if prev != 1 || obj.native {
            return;
        }
        {
            let mut t = self.tables.write().unwrap();
            if t.by_ptr.get(&obj.guest_ptr).is_some_and(|o| o.id == obj.id) {
                t.by_ptr.remove(&obj.guest_ptr);
            }
            if let Some(n) = &obj.name {
                let key = n.to_ascii_lowercase();
                if t.names.get(&key).is_some_and(|o| o.id == obj.id) {
                    t.names.remove(&key);
                }
            }
        }
        {
            let mut d = self.lock();
            d.sync.remove(&obj.id);
            d.headers.remove(&obj.id);
            d.involved.remove(&obj.id);
        }
        if obj.owns_guest_ptr {
            let _ = space.system_free(obj.guest_ptr);
        }
    }

    pub fn insert_handle(&self, obj: &Arc<KObject>) -> u32 {
        self.add_ref(obj);
        let mut t = self.tables.write().unwrap();
        let handle = loop {
            let h = HANDLE_BASE + t.next_handle * 4;
            t.next_handle = t.next_handle.wrapping_add(1).max(1);
            if !t.handles.contains_key(&h) {
                break h;
            }
        };
        t.handles.insert(handle, obj.clone());
        handle
    }

    /// Resolves a handle. `current_thread` answers the NtCurrentThread pseudo-handle.
    pub fn lookup(&self, handle: u32, current_thread: Option<&Arc<KObject>>) -> Option<Arc<KObject>> {
        if handle == CURRENT_THREAD {
            return current_thread.cloned();
        }
        self.tables.read().unwrap().handles.get(&handle).cloned()
    }

    pub fn close_handle(&self, space: &AddressSpace, handle: u32) -> bool {
        let obj = self.tables.write().unwrap().handles.remove(&handle);
        match obj {
            Some(o) => {
                self.release(space, &o);
                true
            }
            None => false,
        }
    }

    pub fn by_guest_ptr(&self, ptr: u32) -> Option<Arc<KObject>> {
        self.tables.read().unwrap().by_ptr.get(&ptr).cloned()
    }

    pub fn find_named(&self, name: &str) -> Option<Arc<KObject>> {
        self.tables.read().unwrap().names.get(&name.to_ascii_lowercase()).cloned()
    }

    /// Object for a dispatcher structure in guest memory, created from the
    /// header contents on first use (titles often have pre-initialized objects
    /// in their data sections that never pass through Ke*Initialize*).
    pub fn native(&self, mem: &GuestMemory, ptr: u32) -> Arc<KObject> {
        if let Some(o) = self.by_guest_ptr(ptr) {
            return o;
        }
        let ty_byte = mem.read_u8(ptr + dh::TYPE);
        let signal = mem.read_i32(ptr + dh::SIGNAL_STATE);
        let (ty, sync) = match ty_byte {
            object_type::EVENT_NOTIFICATION => {
                (ObjectType::Event, SyncState::Event { manual_reset: true, signaled: signal != 0 })
            }
            object_type::EVENT_SYNCHRONIZATION => {
                (ObjectType::Event, SyncState::Event { manual_reset: false, signaled: signal != 0 })
            }
            // A KTHREAD whose kernel object was already released has exited;
            // its header still carries the mirrored signal state.
            object_type::THREAD => (ObjectType::Thread, SyncState::Thread { exited: signal != 0 }),
            // A timer that never went through KeInitializeTimerEx (pre-initialized data):
            // KeSetTimer arms it like any other.
            timer @ (object_type::TIMER_NOTIFICATION | object_type::TIMER_SYNCHRONIZATION) => {
                let manual_reset = timer == object_type::TIMER_NOTIFICATION;
                (ObjectType::Timer, SyncState::Event { manual_reset, signaled: signal != 0 })
            }
            object_type::SEMAPHORE => {
                (ObjectType::Semaphore, SyncState::Semaphore { count: signal, limit: mem.read_i32(ptr + 0x10) })
            }
            object_type::MUTANT => {
                if signal <= 0 {
                    log::warn!("native mutant {ptr:#010X} first seen in owned state; owner unknown, treating as free");
                }
                (ObjectType::Mutant, SyncState::Mutant { owner: None, recursion: 0, abandoned: false })
            }
            other => {
                log::warn!("waiting on dispatcher object {ptr:#010X} of type {other}, which this kernel never signals");
                (ObjectType::Other(other), SyncState::Event { manual_reset: true, signaled: signal != 0 })
            }
        };
        // Two threads may get here at once for a new object (e.g. both contending for a critical
        // section): only one registration may win, or one would wait on an object nobody signals.
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let obj = Arc::new(KObject {
            id,
            ty,
            guest_ptr: ptr,
            owns_guest_ptr: false,
            native: true,
            refs: AtomicU32::new(0),
            name: None,
            payload: Payload::None,
        });
        {
            // Sync state first, so whoever finds the object in the table finds it complete.
            let mut d = self.lock();
            d.sync.insert(id, sync);
            d.headers.insert(id, ptr);
        }
        let mut t = self.tables.write().unwrap();
        if let Some(existing) = t.by_ptr.get(&ptr).cloned() {
            drop(t);
            let mut d = self.lock();
            d.sync.remove(&id);
            d.headers.remove(&id);
            return existing;
        }
        t.by_ptr.insert(ptr, obj.clone());
        obj
    }

    /// (Re)initializes a native object in place (Ke*Initialize*).
    pub fn init_native(&self, mem: &GuestMemory, ptr: u32, ty: ObjectType, sync: SyncState) -> Arc<KObject> {
        if let Some(o) = self.by_guest_ptr(ptr)
            && o.ty == ty
        {
            let mut d = self.lock();
            d.sync.insert(o.id, sync);
            d.mirror(mem, o.id);
            return o;
        }
        self.register(mem, ptr, false, true, ty, Some(sync), Payload::None, None)
    }

    pub fn sync_state(&self, obj: &KObject) -> Option<SyncState> {
        self.lock().sync.get(&obj.id).copied()
    }

    fn update<R>(&self, mem: &GuestMemory, obj: &KObject, f: impl FnOnce(&mut SyncState) -> R) -> Option<R> {
        let mut d = self.lock();
        let r = f(d.sync.get_mut(&obj.id)?);
        d.mirror(mem, obj.id);
        d.wake_waiters(mem);
        Some(r)
    }

    /// KeSetEvent / NtSetEvent; returns the previous state. A signal that
    /// releases waiters involves the calling thread too (see the module docs).
    pub fn set_event(&self, mem: &GuestMemory, obj: &KObject) -> Option<i32> {
        let mut d = self.lock();
        let prev = match d.sync.get_mut(&obj.id)? {
            SyncState::Event { signaled, .. } => std::mem::replace(signaled, true) as i32,
            _ => 0,
        };
        d.mirror(mem, obj.id);
        if d.wake_waiters(mem) {
            let signaller = Arc::new(Participant::current());
            signaller.arm();
            d.note_involved(obj.id, &signaller);
        }
        Some(prev)
    }

    /// KeResetEvent / NtClearEvent; returns the previous state. First waits
    /// for the threads involved in the event's signals to catch up (see the
    /// module docs).
    pub fn reset_event(&self, mem: &GuestMemory, obj: &KObject) -> Option<i32> {
        let limit = Instant::now() + RESET_GRACE;
        let mut d = self.lock();
        loop {
            let lagging: Vec<Arc<Participant>> = match d.involved.remove(&obj.id) {
                Some(list) => list.into_iter().filter(|p| !p.caught_up()).collect(),
                None => Vec::new(),
            };
            if lagging.is_empty() {
                break;
            }
            // Wait outside the dispatcher lock (a released thread may need it
            // on its way out), then look again: the event may have been
            // signalled again meanwhile.
            drop(d);
            for p in &lagging {
                while !p.caught_up() && Instant::now() < limit {
                    std::thread::yield_now();
                }
            }
            d = self.lock();
            if Instant::now() >= limit {
                log::warn!(
                    "event {:#010X} reset while a thread involved in its signal has not caught up after {RESET_GRACE:?}",
                    obj.guest_ptr
                );
                d.involved.remove(&obj.id);
                break;
            }
        }
        let prev = match d.sync.get_mut(&obj.id)? {
            SyncState::Event { signaled, .. } => std::mem::replace(signaled, false) as i32,
            _ => 0,
        };
        d.mirror(mem, obj.id);
        Some(prev)
    }

    /// Sets a timer's signal state (expiry, or KeSetTimer clearing it); returns
    /// the previous state. Unlike [`ObjectManager::reset_event`] a clear does
    /// not wait for released threads: timers are re-armed by the thread that
    /// waits on them, not reset under a running round of waiters.
    pub fn set_timer_signaled(&self, mem: &GuestMemory, obj: &KObject, signaled: bool) -> Option<bool> {
        let mut d = self.lock();
        let prev = match d.sync.get_mut(&obj.id)? {
            SyncState::Event { signaled: s, .. } => std::mem::replace(s, signaled),
            _ => return None,
        };
        d.mirror(mem, obj.id);
        if signaled {
            d.wake_waiters(mem);
        }
        Some(prev)
    }

    /// Releases every current waiter that the signal can satisfy, then resets.
    pub fn pulse_event(&self, mem: &GuestMemory, obj: &KObject) -> Option<i32> {
        let mut d = self.lock();
        let prev = match d.sync.get_mut(&obj.id)? {
            SyncState::Event { signaled, .. } => std::mem::replace(signaled, true) as i32,
            _ => return Some(0),
        };
        d.wake_waiters(mem);
        if let Some(SyncState::Event { signaled, .. }) = d.sync.get_mut(&obj.id) {
            *signaled = false;
        }
        d.mirror(mem, obj.id);
        Some(prev)
    }

    /// Returns the previous count, or SEMAPHORE_LIMIT_EXCEEDED.
    pub fn release_semaphore(&self, mem: &GuestMemory, obj: &KObject, n: i32) -> Result<i32, u32> {
        self.update(mem, obj, |s| match s {
            SyncState::Semaphore { count, limit } => {
                if n <= 0 || *count as i64 + n as i64 > *limit as i64 {
                    return Err(status::SEMAPHORE_LIMIT_EXCEEDED);
                }
                let prev = *count;
                *count += n;
                Ok(prev)
            }
            _ => Err(status::OBJECT_TYPE_MISMATCH),
        })
        .unwrap_or(Err(status::INVALID_HANDLE))
    }

    /// Returns the previous signal state, or MUTANT_NOT_OWNED.
    pub fn release_mutant(&self, mem: &GuestMemory, obj: &KObject, tid: u32, abandon: bool) -> Result<i32, u32> {
        self.update(mem, obj, |s| {
            let prev = s.signal_state();
            match s {
                SyncState::Mutant { owner, recursion, abandoned } => {
                    if abandon {
                        *owner = None;
                        *recursion = 0;
                        *abandoned = true;
                        return Ok(prev);
                    }
                    if *owner != Some(tid) {
                        return Err(status::MUTANT_NOT_OWNED);
                    }
                    *recursion -= 1;
                    if *recursion == 0 {
                        *owner = None;
                    }
                    Ok(prev)
                }
                _ => Err(status::OBJECT_TYPE_MISMATCH),
            }
        })
        .unwrap_or(Err(status::INVALID_HANDLE))
    }

    /// Marks a thread object signaled and abandons the mutants it still owns.
    pub fn thread_exited(&self, mem: &GuestMemory, thread_obj: &KObject, tid: u32) {
        let mut d = self.lock();
        let ids: Vec<ObjId> = d.sync.keys().copied().collect();
        for id in ids {
            let mut changed = false;
            if let Some(SyncState::Mutant { owner, recursion, abandoned }) = d.sync.get_mut(&id)
                && *owner == Some(tid)
            {
                *owner = None;
                *recursion = 0;
                *abandoned = true;
                changed = true;
            }
            if changed {
                d.mirror(mem, id);
            }
        }
        if let Some(SyncState::Thread { exited }) = d.sync.get_mut(&thread_obj.id) {
            *exited = true;
        }
        d.mirror(mem, thread_obj.id);
        d.apcs.remove(&tid);
        d.wake_waiters(mem);
    }

    /// Blocks `tid` until the wait is satisfied, the deadline passes, or (when
    /// alertable) a user APC is queued for the thread. An empty object list
    /// is a pure delay. `site` (the caller's guest return addresses) is kept
    /// for [`ObjectManager::waiters`].
    #[allow(clippy::too_many_arguments)]
    pub fn wait(
        &self,
        mem: &GuestMemory,
        tid: u32,
        objects: &[ObjId],
        wait_all: bool,
        deadline: Option<Instant>,
        alertable: bool,
        site: Vec<u32>,
    ) -> WaitOutcome {
        let mut d = self.lock();
        if !objects.is_empty()
            && let Some(r) = d.try_satisfy(mem, tid, objects, wait_all)
        {
            return WaitOutcome::Satisfied(r);
        }
        let apc_pending = |d: &DispState| d.apcs.get(&tid).is_some_and(|q| !q.is_empty());
        if alertable && apc_pending(&d) {
            return WaitOutcome::Apc;
        }
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            return WaitOutcome::Timeout;
        }
        let id = d.next_wait;
        d.next_wait += 1;
        let slot = Arc::new(WaitSlot::new());
        d.waiters.push(WaitBlock {
            id,
            tid,
            objects: objects.to_vec(),
            wait_all,
            alertable,
            timed: deadline.is_some(),
            site,
            slot: slot.clone(),
        });
        drop(d);

        let interrupted = loop {
            if let Some(r) = slot.result() {
                slot.mark_resumed();
                return WaitOutcome::Satisfied(r);
            }
            if alertable && slot.apc.load(Ordering::Acquire) {
                break WaitOutcome::Apc;
            }
            let now = Instant::now();
            if deadline.is_some_and(|dl| now >= dl) {
                break WaitOutcome::Timeout;
            }
            let guard = slot.sleep.lock().unwrap();
            if slot.result.load(Ordering::Acquire) != NO_RESULT || (alertable && slot.apc.load(Ordering::Acquire)) {
                continue;
            }
            match deadline {
                Some(dl) => drop(slot.cv.wait_timeout(guard, dl - now).unwrap()),
                None => drop(slot.cv.wait(guard).unwrap()),
            }
        };
        // Leaving without a result: withdraw the wait block, unless a signal
        // satisfied it in the meantime (then that result stands).
        let withdrawn = self.lock().remove_waiter(id);
        slot.mark_resumed();
        match slot.result() {
            Some(r) if !withdrawn => WaitOutcome::Satisfied(r),
            _ => interrupted,
        }
    }

    /// Threads currently blocked in a wait (for hang diagnostics).
    pub fn waiters(&self) -> Vec<WaiterInfo> {
        let d = self.lock();
        d.waiters
            .iter()
            .map(|w| WaiterInfo {
                tid: w.tid,
                wait_all: w.wait_all,
                timed: w.timed,
                site: w.site.clone(),
                objects: w
                    .objects
                    .iter()
                    .map(|o| (d.headers.get(o).copied().unwrap_or(0), d.sync.get(o).copied()))
                    .collect(),
            })
            .collect()
    }

    pub fn queue_apc(&self, tid: u32, apc: UserApc) {
        let mut d = self.lock();
        d.apcs.entry(tid).or_default().push_back(apc);
        for w in d.waiters.iter().filter(|w| w.tid == tid && w.alertable) {
            w.slot.post_apc();
        }
    }

    pub fn take_apcs(&self, tid: u32) -> Vec<UserApc> {
        self.lock().apcs.get_mut(&tid).map(|q| q.drain(..).collect()).unwrap_or_default()
    }
}
