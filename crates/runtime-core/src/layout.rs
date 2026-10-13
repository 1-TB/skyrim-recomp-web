//! Offsets of kernel structures that live in guest memory and that title code
//! reads directly (through r13 or pointers the kernel hands out). Layouts follow
//! Xenia's reverse-engineered definitions (src/xenia/kernel/xthread.h,
//! kernel_state.h, xobject.h).

/// DISPATCHER_HEADER (0x10 bytes), the prefix of every waitable object.
pub mod dispatch_header {
    pub const TYPE: u32 = 0x0;
    /// Also `abandoned` / spin count / `signalling` depending on type.
    pub const ABSOLUTE: u32 = 0x1;
    pub const SIZE_FIELD: u32 = 0x2;
    pub const INSERTED: u32 = 0x3;
    pub const SIGNAL_STATE: u32 = 0x4;
    pub const WAIT_LIST_FLINK: u32 = 0x8;
    pub const WAIT_LIST_BLINK: u32 = 0xC;
    pub const SIZE: u32 = 0x10;
}

/// X_OBJECT_TYPES values stored in [`dispatch_header::TYPE`].
pub mod object_type {
    pub const EVENT_NOTIFICATION: u8 = 0x0;
    pub const EVENT_SYNCHRONIZATION: u8 = 0x1;
    pub const MUTANT: u8 = 0x2;
    pub const PROCESS: u8 = 0x3;
    pub const QUEUE: u8 = 0x4;
    pub const SEMAPHORE: u8 = 0x5;
    pub const THREAD: u8 = 0x6;
    pub const TIMER_NOTIFICATION: u8 = 0x8;
    pub const TIMER_SYNCHRONIZATION: u8 = 0x9;
}

/// Processor control region, pointed to by r13 (0x2D8 bytes).
pub mod kpcr {
    pub const TLS_PTR: u32 = 0x0;
    pub const MSR_MASK: u32 = 0x4;
    pub const PROCESSTYPE_VALUE_IN_DPC: u32 = 0xC;
    pub const CURRENT_IRQL: u32 = 0x18;
    /// 64-bit self pointer.
    pub const PCR_PTR: u32 = 0x30;
    pub const STACK_BASE_PTR: u32 = 0x70;
    pub const STACK_END_PTR: u32 = 0x74;
    /// Embedded KPRCB.
    pub const PRCB_DATA: u32 = 0x100;
    pub const PRCB_CURRENT_THREAD: u32 = PRCB_DATA;
    pub const PRCB_CURRENT_CPU: u32 = PRCB_DATA + 0xC;
    pub const PRCB_PROCESSOR_MASK: u32 = PRCB_DATA + 0x10;
    pub const PRCB_DPC_ACTIVE: u32 = PRCB_DATA + 0x50;
    /// Pointer to the KPRCB (== PCR + PRCB_DATA).
    pub const PRCB: u32 = 0x2A8;
    pub const SIZE: u32 = 0x2D8;
}

/// KTHREAD (0xAB0 bytes).
pub mod kthread {
    pub const HEADER: u32 = 0x0;
    pub const MUTANTS_LIST: u32 = 0x10;
    pub const STACK_BASE: u32 = 0x5C;
    pub const STACK_LIMIT: u32 = 0x60;
    pub const STACK_KERNEL: u32 = 0x64;
    pub const TLS_ADDRESS: u32 = 0x68;
    pub const THREAD_STATE: u32 = 0x6C;
    pub const ALERTABLE: u32 = 0x6F;
    pub const PRIORITY: u32 = 0x70;
    pub const FPU_EXCEPTIONS_ON: u32 = 0x71;
    pub const PROCESS_TYPE_DUP: u32 = 0x72;
    pub const PROCESS_TYPE: u32 = 0x73;
    pub const APC_LISTS: u32 = 0x74;
    pub const PROCESS: u32 = 0x84;
    pub const MAY_QUEUE_APCS: u32 = 0x8B;
    pub const MSR_MASK: u32 = 0x9C;
    pub const WAIT_RESULT: u32 = 0xA0;
    pub const QUANTUM: u32 = 0xB4;
    pub const BASE_PRIORITY: u32 = 0xB9;
    pub const SUSPEND_COUNT: u32 = 0xBC;
    pub const TERMINATED: u32 = 0xBE;
    pub const CURRENT_CPU: u32 = 0xBF;
    pub const A_PRCB_PTR: u32 = 0xC0;
    pub const ANOTHER_PRCB_PTR: u32 = 0xC4;
    pub const PROCESS_PRIORITY_CLASS: u32 = 0xC8;
    pub const BASE_PRIORITY_COPY: u32 = 0xC9;
    pub const MAX_DYNAMIC_PRIORITY: u32 = 0xCA;
    pub const STACK_ALLOC_BASE: u32 = 0xD0;
    pub const CREATE_TIME: u32 = 0x130;
    pub const EXIT_TIME: u32 = 0x138;
    pub const EXIT_STATUS: u32 = 0x140;
    pub const THREAD_ID: u32 = 0x14C;
    pub const START_ADDRESS: u32 = 0x150;
    pub const LAST_ERROR: u32 = 0x160;
    pub const FIBER_PTR: u32 = 0x164;
    pub const CREATION_FLAGS: u32 = 0x16C;
    pub const SIZE: u32 = 0xAB0;

    pub const STATE_INITIALIZED: u8 = 0;
    pub const STATE_READY: u8 = 1;
    pub const STATE_RUNNING: u8 = 2;
    pub const STATE_TERMINATED: u8 = 4;
    pub const STATE_WAITING: u8 = 5;
}

/// KPROCESS (0x60 bytes).
pub mod kprocess {
    pub const THREAD_LIST_SPINLOCK: u32 = 0x0;
    pub const THREAD_LIST: u32 = 0x4;
    pub const QUANTUM: u32 = 0xC;
    pub const THREAD_COUNT: u32 = 0x14;
    pub const PROCESS_PRIORITY_CLASS: u32 = 0x18;
    pub const DEFAULT_THREAD_PRIORITY: u32 = 0x19;
    pub const MAX_DYNAMIC_PRIORITY: u32 = 0x1A;
    pub const KERNEL_STACK_SIZE: u32 = 0x1C;
    pub const TLS_STATIC_DATA_ADDRESS: u32 = 0x20;
    pub const TLS_DATA_SIZE: u32 = 0x24;
    pub const TLS_RAW_DATA_SIZE: u32 = 0x28;
    pub const TLS_SLOT_SIZE: u32 = 0x2C;
    pub const PROCESS_TYPE: u32 = 0x2F;
    pub const SIZE: u32 = 0x60;
}

pub const PROCTYPE_IDLE: u8 = 0;
pub const PROCTYPE_TITLE: u8 = 1;
pub const PROCTYPE_SYSTEM: u8 = 2;
