//! Guest thread model: every guest thread is a host thread with its own
//! `PPCContext`, a stack in guest memory, a PCR (r13), a KTHREAD and a TLS block.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::address_space::{AddressSpace, AllocResult, StackAllocation};
use crate::layout::{dispatch_header, kpcr, kprocess, kthread, object_type};
use crate::memory::GuestMemory;
use crate::timebase;

/// `ExCreateThread` creation flags.
pub mod creation_flags {
    pub const SUSPENDED: u32 = 0x0000_0001;
    pub const SYSTEM_THREAD: u32 = 0x0000_0002;
    pub const TLS_STATIC: u32 = 0x0000_0008;
    pub const PRIORITY_CLASS1: u32 = 0x0000_0020;
    pub const PRIORITY_CLASS2: u32 = 0x0000_0040;
    pub const RETURN_KTHREAD_PTR: u32 = 0x0000_0080;
    pub const TITLE_EXECUTION_THREAD: u32 = 0x0000_0100;
    pub const HIDDEN: u32 = 0x0000_0400;
    /// Bits 24..29 select the hardware thread.
    pub const AFFINITY_SHIFT: u32 = 24;
    pub const AFFINITY_MASK: u32 = 0x3F << AFFINITY_SHIFT;
}

pub const HARDWARE_THREADS: u32 = 6;
/// Dynamic TLS slots when the XEX doesn't specify a count (Xenia's default).
pub const DEFAULT_TLS_SLOTS: u32 = 1024;
pub const MIN_STACK_SIZE: u32 = 0x4000;
/// Bytes left above the initial r1 for the entry function's linkage area.
pub const INITIAL_STACK_HEADROOM: u32 = 0x100;

/// Host stack for threads running recompiled code. Guest calls nest as host
/// calls, so this bounds guest call depth.
#[cfg(not(target_os = "emscripten"))]
pub const HOST_STACK_SIZE: usize = 16 << 20;
/// On wasm the shadow stack comes out of the host's 1 GiB budget; most frame
/// state lives in wasm locals, so a smaller shadow stack suffices.
#[cfg(target_os = "emscripten")]
pub const HOST_STACK_SIZE: usize = 4 << 20;

/// Static TLS template from the XEX TLS directory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TlsTemplate {
    pub slot_count: u32,
    pub raw_data_address: u32,
    pub data_size: u32,
    pub raw_data_size: u32,
}

/// Guest-memory structures belonging to one thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadBlock {
    pub thread_id: u32,
    pub cpu: u32,
    pub stack: StackAllocation,
    pub pcr: u32,
    pub kthread: u32,
    /// Start of the TLS block; PCR.tls_ptr points here (static TLS data).
    pub tls_static: u32,
    /// Dynamic slot array (KeTlsGetValue/SetValue), after the static data.
    pub tls_dynamic: u32,
    pub tls_slots: u32,
}

impl ThreadBlock {
    /// Initial r1 for the thread's entry call.
    pub fn initial_sp(&self) -> u32 {
        self.stack.base - INITIAL_STACK_HEADROOM
    }

    pub fn tls_get(&self, mem: &GuestMemory, slot: u32) -> Option<u32> {
        (slot < self.tls_slots).then(|| mem.read_u32(self.tls_dynamic + slot * 4))
    }

    pub fn tls_set(&self, mem: &GuestMemory, slot: u32, value: u32) -> bool {
        if slot >= self.tls_slots {
            return false;
        }
        mem.write_u32(self.tls_dynamic + slot * 4, value);
        true
    }
}

pub struct ThreadFactory {
    next_thread_id: AtomicU32,
    next_cpu: AtomicU32,
    tls: TlsTemplate,
    process: u32,
}

/// Write an empty LIST_ENTRY (both links point at itself).
pub fn init_list_head(mem: &GuestMemory, addr: u32) {
    mem.write_u32(addr, addr);
    mem.write_u32(addr + 4, addr);
}

impl ThreadFactory {
    pub fn new(tls: TlsTemplate, process: u32) -> Self {
        ThreadFactory { next_thread_id: AtomicU32::new(1), next_cpu: AtomicU32::new(0), tls, process }
    }

    pub fn process(&self) -> u32 {
        self.process
    }

    pub fn tls_template(&self) -> TlsTemplate {
        self.tls
    }

    /// Hardware thread for a new thread: the one its creation flags name, else
    /// the creating thread's (as on hardware, where a thread without an
    /// explicit processor stays with its parent), else round robin.
    fn pick_cpu(&self, flags: u32, parent_cpu: Option<u32>) -> u32 {
        let mask = (flags & creation_flags::AFFINITY_MASK) >> creation_flags::AFFINITY_SHIFT;
        hardware_thread_for(mask)
            .or(parent_cpu)
            .unwrap_or_else(|| self.next_cpu.fetch_add(1, Ordering::Relaxed) % HARDWARE_THREADS)
    }

    /// Allocates and initializes stack, TLS, PCR and KTHREAD for a new thread.
    /// `parent_cpu` is the creating guest thread's hardware thread, if any.
    pub fn create(
        &self,
        space: &AddressSpace,
        stack_size: u32,
        flags: u32,
        start_address: u32,
        parent_cpu: Option<u32>,
    ) -> AllocResult<ThreadBlock> {
        let mem = &space.mem;
        let stack_size = stack_size.max(MIN_STACK_SIZE).div_ceil(0x1000) * 0x1000;
        let stack = space.alloc_stack(stack_size)?;

        let tls_slots = if self.tls.slot_count != 0 { self.tls.slot_count } else { DEFAULT_TLS_SLOTS };
        let tls_static = space.system_alloc(self.tls.data_size + tls_slots * 4, 16)?;
        let tls_dynamic = tls_static + self.tls.data_size;
        if self.tls.raw_data_size != 0 {
            mem.copy_within(tls_static, self.tls.raw_data_address, self.tls.raw_data_size);
        }

        let pcr = space.system_alloc(kpcr::SIZE, 16)?;
        let kthread = space.system_alloc(kthread::SIZE, 16)?;
        let thread_id = self.next_thread_id.fetch_add(1, Ordering::Relaxed);
        let cpu = self.pick_cpu(flags, parent_cpu);
        let prcb = pcr + kpcr::PRCB_DATA;

        // KTHREAD (Xenia XThread::InitializeGuestObject).
        let process = self.process;
        mem.write_u8(kthread + dispatch_header::TYPE, object_type::THREAD);
        mem.write_u8(kthread + dispatch_header::SIZE_FIELD, (kthread::SIZE / 4) as u8);
        init_list_head(mem, kthread + dispatch_header::WAIT_LIST_FLINK);
        init_list_head(mem, kthread + kthread::MUTANTS_LIST);
        mem.write_u8(kthread + kthread::SUSPEND_COUNT, (flags & creation_flags::SUSPENDED != 0) as u8);
        mem.write_u32(kthread + kthread::STACK_BASE, stack.base);
        mem.write_u32(kthread + kthread::STACK_LIMIT, stack.limit);
        mem.write_u32(kthread + kthread::STACK_KERNEL, stack.base - 240);
        mem.write_u32(kthread + kthread::TLS_ADDRESS, tls_dynamic);
        mem.write_u8(kthread + kthread::THREAD_STATE, kthread::STATE_INITIALIZED);
        let process_type = mem.read_u8(process + kprocess::PROCESS_TYPE);
        mem.write_u8(kthread + kthread::PROCESS_TYPE_DUP, process_type);
        mem.write_u8(kthread + kthread::PROCESS_TYPE, process_type);
        init_list_head(mem, kthread + kthread::APC_LISTS);
        init_list_head(mem, kthread + kthread::APC_LISTS + 8);
        let base_priority = mem.read_u8(process + kprocess::DEFAULT_THREAD_PRIORITY);
        mem.write_u8(
            kthread + kthread::PROCESS_PRIORITY_CLASS,
            mem.read_u8(process + kprocess::PROCESS_PRIORITY_CLASS),
        );
        mem.write_u8(kthread + kthread::BASE_PRIORITY_COPY, base_priority);
        mem.write_u8(kthread + kthread::BASE_PRIORITY, base_priority);
        mem.write_u8(kthread + kthread::PRIORITY, base_priority);
        mem.write_u8(kthread + kthread::MAX_DYNAMIC_PRIORITY, mem.read_u8(process + kprocess::MAX_DYNAMIC_PRIORITY));
        mem.write_u32(kthread + kthread::QUANTUM, mem.read_u32(process + kprocess::QUANTUM));
        mem.write_u32(kthread + kthread::A_PRCB_PTR, prcb);
        mem.write_u32(kthread + kthread::ANOTHER_PRCB_PTR, prcb);
        mem.write_u8(kthread + kthread::MAY_QUEUE_APCS, 1);
        mem.write_u32(kthread + kthread::MSR_MASK, 0xFDFF_D7FF);
        mem.write_u32(kthread + kthread::PROCESS, process);
        mem.write_u32(kthread + kthread::STACK_ALLOC_BASE, stack.base);
        mem.write_u8(kthread + kthread::CURRENT_CPU, cpu as u8);
        mem.write_u64(kthread + kthread::CREATE_TIME, timebase::system_time());
        mem.write_u32(kthread + kthread::THREAD_ID, thread_id);
        mem.write_u32(kthread + kthread::START_ADDRESS, start_address);
        mem.write_u32(kthread + kthread::CREATION_FLAGS, flags);
        mem.atomic_add_i32(process + kprocess::THREAD_COUNT, 1);

        // PCR (Xenia XThread::Create).
        mem.write_u32(pcr + kpcr::TLS_PTR, tls_static);
        mem.write_u64(pcr + kpcr::PCR_PTR, pcr as u64);
        mem.write_u32(pcr + kpcr::PRCB_CURRENT_THREAD, kthread);
        mem.write_u8(pcr + kpcr::PRCB_CURRENT_CPU, cpu as u8);
        mem.write_u32(pcr + kpcr::PRCB_PROCESSOR_MASK, 1 << cpu);
        mem.write_u32(pcr + kpcr::PRCB, prcb);
        mem.write_u32(pcr + kpcr::STACK_BASE_PTR, stack.base);
        mem.write_u32(pcr + kpcr::STACK_END_PTR, stack.limit);
        mem.write_u32(pcr + kpcr::PRCB_DPC_ACTIVE, 0);

        Ok(ThreadBlock { thread_id, cpu, stack, pcr, kthread, tls_static, tls_dynamic, tls_slots })
    }

    /// Frees the stack and TLS. The KTHREAD stays allocated: other threads may
    /// still hold references to it (it is a waitable object).
    pub fn release(&self, space: &AddressSpace, block: &ThreadBlock) -> AllocResult<()> {
        space.free_stack(block.stack.alloc_base)?;
        space.system_free(block.tls_static)?;
        space.system_free(block.pcr)?;
        space.mem.atomic_add_i32(self.process + kprocess::THREAD_COUNT, -1);
        Ok(())
    }
}

static THREAD_START_HOOK: std::sync::OnceLock<fn()> = std::sync::OnceLock::new();

/// Registers a function every guest host thread runs first (hosts use it for
/// per-thread setup such as an alternate signal stack). Only the first
/// registration takes effect.
pub fn set_thread_start_hook(hook: fn()) {
    let _ = THREAD_START_HOOK.set(hook);
}

/// Spawns the host thread that will run a guest thread.
pub fn spawn_host_thread<F>(name: String, f: F) -> std::io::Result<std::thread::JoinHandle<()>>
where
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new().name(name).stack_size(HOST_STACK_SIZE).spawn(move || {
        if let Some(hook) = THREAD_START_HOOK.get() {
            hook();
        }
        f()
    })
}

/// The hardware thread an affinity mask selects: its highest set bit among the
/// six hardware threads (Xenia GetFakeCpuNumber), or None for an empty mask.
pub fn hardware_thread_for(mask: u32) -> Option<u32> {
    let mask = mask & ((1 << HARDWARE_THREADS) - 1);
    (mask != 0).then(|| 31 - mask.leading_zeros())
}
