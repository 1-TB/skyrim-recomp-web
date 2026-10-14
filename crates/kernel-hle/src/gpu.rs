//! The Xenos GPU front end: command processor, registers and interrupts.
//!
//! - GPU registers live in the MMIO window at [`REGISTER_BASE`] (plain guest
//!   memory, so title code reads and writes them directly). Registers past the
//!   window (shader constants, fetch constants) are host-side only.
//! - A command-processor thread drains the primary ring buffer whenever the
//!   title moves CP_RB_WPTR, executing the PM4 packets that have CPU-visible
//!   effects itself (register and memory writes, waits, fences, interrupts)
//!   and forwarding rendering work (register state, shader loads, draws,
//!   swaps) in stream order to the host's [`GpuBackend`], if one is set.
//!   Before every effect the CPU can observe, the CP waits for the backend to
//!   catch up, so GPU writes to memory (resolves) precede the fences after them.
//! - A vblank thread raises the graphics interrupt at 60 Hz (or `Gpu::set_vblank_hz`).
//!
//! Interrupt callbacks run on those threads as DPCs, as on hardware. The packet
//! semantics follow Xenia's command processor (src/xenia/gpu).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

use runtime_core::PPCContext;
use runtime_core::layout::{PROCTYPE_TITLE, kpcr, kthread};
use runtime_core::memory::GuestMemory;
use runtime_core::thread::spawn_host_thread;

use crate::Kernel;
use crate::xboxkrnl::call_as_dpc;

/// Guest address of GPU register 0.
pub const REGISTER_BASE: u32 = 0x7FC8_0000;
/// Registers reachable through the MMIO window (64 KiB).
const MMIO_REGISTERS: u32 = 0x4000;
/// Size of the full register file, including constant registers.
pub const REGISTER_COUNT: u32 = 0x5003;

/// Register indices (Xenia register_table.inc).
pub mod reg {
    pub const CP_RB_WPTR: u32 = 0x01C5;
    pub const SCRATCH_UMSK: u32 = 0x01DC;
    pub const SCRATCH_ADDR: u32 = 0x01DD;
    pub const SCRATCH_REG0: u32 = 0x0578;
    pub const SCRATCH_REG7: u32 = 0x057F;
    pub const WRITEBACK_START: u32 = 0x0A04;
    pub const WRITEBACK_SIZE: u32 = 0x0A05;
    pub const COHER_STATUS_HOST: u32 = 0x0A31;
    pub const RB_EDRAM_TIMING: u32 = 0x0F00;
    pub const RB_BC_CONTROL: u32 = 0x0F01;
    pub const RB_HSIO_INTERFACE_ALIGNER_VALUE: u32 = 0x0F2D;
    pub const D1MODE_V_COUNTER: u32 = 0x194C;
    pub const D1MODE_INTERRUPT_STATUS: u32 = 0x1951;
    pub const D1MODE_VIEWPORT_SIZE: u32 = 0x1961;
    pub const VGT_EVENT_INITIATOR: u32 = 0x21F9;
    pub const VGT_DMA_BASE: u32 = 0x21FA;
    pub const VGT_DMA_SIZE: u32 = 0x21FB;
    pub const VGT_DRAW_INITIATOR: u32 = 0x21FC;
    pub const RB_SAMPLE_COUNT_ADDR: u32 = 0x2325;
    pub const SHADER_CONSTANT_000_X: u32 = 0x4000;
    pub const SHADER_CONSTANT_FETCH_00_0: u32 = 0x4800;
    pub const SHADER_CONSTANT_BOOL_000_031: u32 = 0x4900;
    pub const SHADER_CONSTANT_LOOP_00: u32 = 0x4908;
    pub const REGISTERS_BASE: u32 = 0x2000;
}

/// PM4 type-3 opcodes (Xenia xenos.h).
pub mod pm4 {
    pub const NOP: u32 = 0x10;
    pub const REG_RMW: u32 = 0x21;
    pub const DRAW_INDX: u32 = 0x22;
    pub const VIZ_QUERY: u32 = 0x23;
    pub const SET_STATE: u32 = 0x25;
    pub const WAIT_FOR_IDLE: u32 = 0x26;
    pub const IM_LOAD: u32 = 0x27;
    pub const IM_LOAD_IMMEDIATE: u32 = 0x2B;
    pub const IM_STORE: u32 = 0x2C;
    pub const SET_CONSTANT: u32 = 0x2D;
    pub const LOAD_CONSTANT_CONTEXT: u32 = 0x2E;
    pub const LOAD_ALU_CONSTANT: u32 = 0x2F;
    pub const DRAW_INDX_BIN: u32 = 0x34;
    pub const DRAW_INDX_2_BIN: u32 = 0x35;
    pub const DRAW_INDX_2: u32 = 0x36;
    pub const INDIRECT_BUFFER_PFD: u32 = 0x37;
    pub const INVALIDATE_STATE: u32 = 0x3B;
    pub const WAIT_REG_MEM: u32 = 0x3C;
    pub const MEM_WRITE: u32 = 0x3D;
    pub const REG_TO_MEM: u32 = 0x3E;
    pub const INDIRECT_BUFFER: u32 = 0x3F;
    pub const COND_EXEC: u32 = 0x44;
    pub const COND_WRITE: u32 = 0x45;
    pub const EVENT_WRITE: u32 = 0x46;
    pub const ME_INIT: u32 = 0x48;
    pub const SET_SHADER_BASES: u32 = 0x4A;
    pub const SET_BIN_BASE_OFFSET: u32 = 0x4B;
    pub const MEM_WRITE_CNTR: u32 = 0x4F;
    pub const SET_BIN_MASK: u32 = 0x50;
    pub const SET_BIN_SELECT: u32 = 0x51;
    pub const WAIT_REG_EQ: u32 = 0x52;
    pub const WAIT_REG_GTE: u32 = 0x53;
    pub const INTERRUPT: u32 = 0x54;
    pub const SET_CONSTANT2: u32 = 0x55;
    pub const SET_SHADER_CONSTANTS: u32 = 0x56;
    pub const EVENT_WRITE_SHD: u32 = 0x58;
    pub const EVENT_WRITE_CFL: u32 = 0x59;
    pub const EVENT_WRITE_EXT: u32 = 0x5A;
    pub const EVENT_WRITE_ZPD: u32 = 0x5B;
    pub const WAIT_UNTIL_READ: u32 = 0x5C;
    pub const WAIT_IB_PFD_COMPLETE: u32 = 0x5D;
    pub const CONTEXT_UPDATE: u32 = 0x5E;
    pub const SET_BIN_MASK_LO: u32 = 0x60;
    pub const SET_BIN_MASK_HI: u32 = 0x61;
    pub const SET_BIN_SELECT_LO: u32 = 0x62;
    pub const SET_BIN_SELECT_HI: u32 = 0x63;
    /// Not a hardware opcode: VdSwap emits it (as Xenia does) to mark a present.
    pub const XE_SWAP: u32 = 0x64;
}

/// 'SWAP', the first word of an XE_SWAP packet.
pub const SWAP_SIGNATURE: u32 = u32::from_be_bytes(*b"SWAP");

pub fn packet_type0(index: u32, count: u32) -> u32 {
    ((count - 1) & 0x3FFF) << 16 | (index & 0x7FFF)
}

pub fn packet_type2() -> u32 {
    2 << 30
}

pub fn packet_type3(opcode: u32, count: u32) -> u32 {
    3 << 30 | ((count - 1) & 0x3FFF) << 16 | (opcode & 0x7F) << 8
}

/// Byte order applied to a GPU memory access, from the low two address bits.
fn gpu_swap(value: u32, endian: u32) -> u32 {
    match endian & 3 {
        1 => ((value << 8) & 0xFF00_FF00) | ((value >> 8) & 0x00FF_00FF),
        2 => value.swap_bytes(),
        3 => value.rotate_left(16),
        _ => value,
    }
}

/// Xenia's MatchValueAndRef: the low three bits of `wait_info` pick the
/// comparison (never, <, <=, ==, !=, >=, >, always).
fn match_value_and_ref(value: u32, reference: u32, wait_info: u32) -> bool {
    match wait_info & 7 {
        0 => false,
        1 => value < reference,
        2 => value <= reference,
        3 => value == reference,
        4 => value != reference,
        5 => value >= reference,
        6 => value > reference,
        _ => true,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RingConfig {
    /// Guest address of the primary ring buffer (0 until initialized).
    base: u32,
    size_dwords: u32,
    /// Physical address the read pointer is written back to (0 = off).
    rptr_writeback: u32,
    /// Bumped by every VdInitializeRingBuffer so the CP restarts at 0.
    generation: u32,
}

#[derive(Debug, Clone, Copy, Default)]
struct InterruptCallback {
    routine: u32,
    user_data: u32,
}

/// The renderer, fed by the command processor in stream order.
pub trait GpuBackend: Send + Sync {
    /// Register writes `(index, value)` from the command stream, in order.
    fn registers(&self, writes: Vec<(u32, u32)>);
    /// Shader microcode (host-order dwords) for the vertex (`pixel == false`) or pixel stage.
    fn shader(&self, pixel: bool, code: Arc<[u32]>);
    /// A draw; VGT_DRAW_INITIATOR (and VGT_DMA_BASE / VGT_DMA_SIZE for indexed draws) were
    /// sent with the registers before it.
    fn draw(&self);
    /// A present of the front buffer described by texture fetch constant 0 (`fetch`, physical
    /// base address).
    fn swap(&self, fetch: [u32; 6], width: u32, height: u32);
    /// Blocks until everything sent so far has been executed, including writes to guest memory.
    fn wait_idle(&self);
    /// `--gpu d3d`: a D3D call for the D3D-level renderer; returns its sequence number.
    fn d3d(&self, _cmd: d3d_hle::command::Command) -> u64 {
        0
    }
    /// Blocks until the command with sequence number `seq` has been executed.
    fn wait_for(&self, _seq: u64) {
        self.wait_idle();
    }
}

pub struct Gpu {
    backend: OnceLock<Arc<dyn GpuBackend>>,
    /// Registers at and above [`MMIO_REGISTERS`].
    high_registers: Box<[AtomicU32]>,
    ring: Mutex<RingConfig>,
    interrupt: Mutex<InterruptCallback>,
    /// Xenia's command-processor counter: bumped per swap and per vblank,
    /// readable through EVENT_WRITE_SHD.
    counter: AtomicU32,
    swaps: AtomicU64,
    vblanks: AtomicU64,
    command_processor: Once,
    vblank: Once,
    /// Vblank interrupts per second. Titles pace frames by vblanks (Skyrim presents every second
    /// one: 30 fps at the hardware's 60 Hz), so a higher rate lifts their frame rate cap, like
    /// Xenia's `vsync` option.
    vblank_hz: AtomicU32,
    /// Bumped whenever the command processor writes back its read pointer; title code waiting
    /// for ring space sleeps on it (see [`skyrt_gpu_wait_progress`]).
    progress: AtomicU32,
    progress_waiters: AtomicU32,
    progress_lock: Mutex<()>,
    progress_changed: Condvar,
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpu {
    pub fn new() -> Self {
        Gpu {
            backend: OnceLock::new(),
            high_registers: (MMIO_REGISTERS..REGISTER_COUNT).map(|_| AtomicU32::new(0)).collect(),
            ring: Mutex::new(RingConfig::default()),
            interrupt: Mutex::new(InterruptCallback::default()),
            counter: AtomicU32::new(0),
            swaps: AtomicU64::new(0),
            vblanks: AtomicU64::new(0),
            command_processor: Once::new(),
            vblank: Once::new(),
            vblank_hz: AtomicU32::new(DEFAULT_VBLANK_HZ),
            progress: AtomicU32::new(0),
            progress_waiters: AtomicU32::new(0),
            progress_lock: Mutex::new(()),
            progress_changed: Condvar::new(),
        }
    }

    /// Records command-processor progress and wakes threads waiting for it.
    fn note_progress(&self) {
        self.progress.fetch_add(1, Ordering::SeqCst);
        if self.progress_waiters.load(Ordering::SeqCst) != 0 {
            drop(self.progress_lock.lock().unwrap());
            self.progress_changed.notify_all();
        }
    }

    /// Waits until progress moves past `seen` or `timeout` passes.
    fn wait_progress(&self, seen: u32, timeout: Duration) {
        self.progress_waiters.fetch_add(1, Ordering::SeqCst);
        let guard = self.progress_lock.lock().unwrap();
        if self.progress.load(Ordering::SeqCst) == seen {
            let _ = self
                .progress_changed
                .wait_timeout_while(guard, timeout, |_| self.progress.load(Ordering::SeqCst) == seen);
        }
        self.progress_waiters.fetch_sub(1, Ordering::SeqCst);
    }

    /// Sets the vblank interrupt rate (default 60, the hardware's); takes effect at the next
    /// vblank.
    pub fn set_vblank_hz(&self, hz: u32) {
        self.vblank_hz.store(hz.clamp(1, 10_000), Ordering::Relaxed);
    }

    /// Seeds the registers titles read back during GPU bring-up with the values
    /// Xenia's GraphicsSystem::ReadRegister reports.
    pub fn init_registers(&self, mem: &GuestMemory) {
        for (r, v) in [
            (reg::RB_EDRAM_TIMING, 0x0810_0748),
            (reg::RB_BC_CONTROL, 0x0000_200E),
            (reg::RB_HSIO_INTERFACE_ALIGNER_VALUE, 0x00BB_BBBB),
            // Interrupt status: vblank.
            (reg::D1MODE_INTERRUPT_STATUS, 1),
            // 1280x720.
            (reg::D1MODE_VIEWPORT_SIZE, 0x0500_02D0),
        ] {
            self.write_register(mem, r, v);
        }
    }

    pub fn read_register(&self, mem: &GuestMemory, index: u32) -> u32 {
        if index < MMIO_REGISTERS {
            mem.atomic_load_u32(REGISTER_BASE + index * 4)
        } else {
            self.high_registers.get((index - MMIO_REGISTERS) as usize).map_or(0, |r| r.load(Ordering::Relaxed))
        }
    }

    pub fn write_register(&self, mem: &GuestMemory, index: u32, value: u32) {
        if index < MMIO_REGISTERS {
            mem.atomic_store_u32(REGISTER_BASE + index * 4, value);
        } else if let Some(r) = self.high_registers.get((index - MMIO_REGISTERS) as usize) {
            r.store(value, Ordering::Relaxed);
        } else {
            log::warn!("GPU: write to register {index:#06X} beyond the register file");
        }
    }

    /// Installs the renderer. Must happen before the title starts the command processor; only
    /// the first backend is kept.
    pub fn set_backend(&self, backend: Arc<dyn GpuBackend>) -> bool {
        self.backend.set(backend).is_ok()
    }

    pub fn swap_count(&self) -> u64 {
        self.swaps.load(Ordering::Relaxed)
    }

    pub fn vblank_count(&self) -> u64 {
        self.vblanks.load(Ordering::Relaxed)
    }
}

impl Kernel {
    /// VdInitializeRingBuffer: `ptr` is the ring's physical address (only the
    /// low 29 bits count, as for every address the GPU sees); its size is
    /// 2^(size_log2 + 3) bytes. Starts the command processor on first use.
    pub fn gpu_initialize_ring_buffer(&'static self, ptr: u32, size_log2: u32) {
        let mem = self.mem();
        let Some(base) = self.rt.space.translate_physical(ptr) else {
            log::error!("VdInitializeRingBuffer: {ptr:#010X} is not GPU-visible memory");
            return;
        };
        let size_bytes = 1u32 << (size_log2 + 3).min(31);
        mem.fill(base, size_bytes, 0);
        {
            let mut ring = self.gpu.ring.lock().unwrap();
            ring.base = base;
            ring.size_dwords = size_bytes / 4;
            ring.generation = ring.generation.wrapping_add(1);
        }
        self.gpu.write_register(mem, reg::CP_RB_WPTR, 0);
        log::info!("GPU: primary ring buffer at {base:#010X}, {size_bytes:#X} bytes");
        self.gpu.command_processor.call_once(|| self.spawn_kernel_thread("gpu-cp", command_processor_main));
    }

    /// VdEnableRingBufferRPtrWriteBack: `ptr` is physical, like the ring's.
    pub fn gpu_enable_read_pointer_writeback(&self, ptr: u32, block_size_log2: u32) {
        let phys = ptr & 0x1FFF_FFFF;
        self.gpu.ring.lock().unwrap().rptr_writeback = phys;
        log::info!("GPU: read pointer writeback to {phys:#010X} (block 2^{block_size_log2})");
    }

    /// VdSetGraphicsInterruptCallback. Starts vblank interrupts on first use.
    pub fn gpu_set_interrupt_callback(&'static self, routine: u32, user_data: u32) {
        *self.gpu.interrupt.lock().unwrap() = InterruptCallback { routine, user_data };
        log::info!("GPU: interrupt callback {routine:#010X} (user data {user_data:#010X})");
        if routine != 0 {
            self.gpu.vblank.call_once(|| self.spawn_kernel_thread("gpu-vblank", vblank_main));
        }
    }

    /// Counts a present (VdSwap's XE_SWAP packet reached the command processor).
    fn gpu_swap(&self, frontbuffer: u32, width: u32, height: u32) -> u64 {
        self.gpu.counter.fetch_add(1, Ordering::Relaxed);
        let n = self.gpu.swaps.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_multiple_of(600) {
            log::info!("GPU: swap #{n} (front buffer {frontbuffer:#010X}, {width}x{height})");
        }
        n
    }

    /// Runs the graphics interrupt callback as `callback(source, user_data)` in
    /// DPC context on `ctx`'s thread, presenting as processor `cpu`.
    fn gpu_dispatch_interrupt(&self, ctx: &mut PPCContext, source: u32, cpu: u32) {
        let cb = *self.gpu.interrupt.lock().unwrap();
        if cb.routine == 0 {
            return;
        }
        let mem = self.mem();
        let pcr = ctx.gpr32(13);
        mem.write_u8(pcr + kpcr::PRCB_CURRENT_CPU, cpu as u8);
        let k = mem.read_u32(pcr + kpcr::PRCB_CURRENT_THREAD);
        if k != 0 {
            mem.write_u8(k + kthread::CURRENT_CPU, cpu as u8);
        }
        call_as_dpc(self, ctx, cb.routine, &[source, cb.user_data], Some(PROCTYPE_TITLE));
    }

    /// Starts a host thread that is also a guest thread (it runs guest
    /// callbacks), passing it a context of its own.
    pub(crate) fn spawn_kernel_thread(&'static self, name: &str, body: fn(&'static Kernel, &mut PPCContext)) {
        let thread_name = name.to_string();
        let spawned = spawn_host_thread(thread_name.clone(), move || {
            let t = match self.adopt_current_thread() {
                Ok(t) => t,
                Err(s) => {
                    log::error!("{thread_name}: creating guest thread state failed: {s:#010X}");
                    return;
                }
            };
            let mut ctx = self.context_for(&t);
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| body(self, &mut ctx))) {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "guest code left the thread".into());
                log::error!("{thread_name} stopped: {msg}");
            }
        });
        if let Err(e) = spawned {
            log::error!("starting {name}: {e}");
        }
    }
}

/// Draws in a frame that mark a game world being shown (menus and loading screens draw ~100).
const FULL_FRAME_DRAWS: u32 = 1000;

/// The hardware's refresh rate.
pub const DEFAULT_VBLANK_HZ: u32 = 60;

fn vblank_main(k: &'static Kernel, ctx: &mut PPCContext) {
    let mem = k.mem();
    let period = || Duration::from_nanos(1_000_000_000 / u64::from(k.gpu.vblank_hz.load(Ordering::Relaxed).max(1)));
    let mut next = Instant::now() + period();
    loop {
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        }
        next += period();
        // Don't try to catch up after a stall (debugger, slow callback).
        if next < Instant::now() {
            next = Instant::now() + period();
        }
        let v = k.gpu.read_register(mem, reg::D1MODE_V_COUNTER);
        k.gpu.write_register(mem, reg::D1MODE_V_COUNTER, v.wrapping_add(1));
        k.gpu.counter.fetch_add(1, Ordering::Relaxed);
        k.gpu.vblanks.fetch_add(1, Ordering::Relaxed);
        k.gpu_dispatch_interrupt(ctx, 0, 2);
    }
}

fn command_processor_main(k: &'static Kernel, ctx: &mut PPCContext) {
    let mem = k.mem();
    let mut cp = CommandProcessor {
        k,
        mem,
        ctx,
        bin_mask: 0xFFFF_FFFF,
        bin_select: 0xFFFF_FFFF,
        zpd_samples: 0,
        backend: k.gpu.backend.get().cloned(),
        pending: Vec::new(),
        syncs: [0; 128],
        sync_wait: Duration::ZERO,
        frame_draws: 0,
        full_frame_seen: false,
        d3d_marker: None,
        d3d_writes: Vec::new(),
        d3d_packet: (0, 0),
        d3d_last_writer: std::collections::HashMap::new(),
        d3d_draw_next: false,
        d3d_last_draw: 0,
        d3d_draws: (0, 0),
    };
    let mut read_index = 0u32;
    let mut generation = 0u32;
    let mut idle_spins = 0u32;
    loop {
        let ring = *k.gpu.ring.lock().unwrap();
        if ring.generation != generation {
            generation = ring.generation;
            read_index = 0;
        }
        let write_index = k.gpu.read_register(mem, reg::CP_RB_WPTR) % ring.size_dwords.max(1);
        if ring.base == 0 || write_index == read_index {
            // Titles kick the ring by storing CP_RB_WPTR, which nothing
            // intercepts, so poll it: spin briefly, then back off to sleeping.
            idle_spins += 1;
            if idle_spins < 500 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(500));
            }
            continue;
        }
        idle_spins = 0;
        let count = (write_index + ring.size_dwords - read_index) % ring.size_dwords;
        log::trace!("ring {read_index:#X}..{write_index:#X}");
        let writeback = match ring.rptr_writeback {
            0 => None,
            phys => k.rt.space.translate_physical(phys).or_else(|| {
                log::warn!("GPU: read pointer writeback address {phys:#010X} unmapped");
                None
            }),
        };
        let mut reader = Reader { base: ring.base, capacity: ring.size_dwords, pos: read_index, remaining: count };
        // Report progress after every packet, as the hardware's periodic writeback does, rather
        // than after the whole batch: a title waiting for ring space (D3D spins on it) can go on
        // while the rest of the batch, which may wait for the renderer at fences, is executed.
        while reader.remaining > 0 {
            if !cp.packet(&mut reader) {
                // Malformed stream: drop the rest of this batch.
                reader.skip(reader.remaining);
            }
            if let Some(a) = writeback {
                mem.write_u32(a, reader.pos);
                k.gpu.note_progress();
            }
        }
        read_index = write_index;
    }
}

/// A window of command dwords in guest memory, wrapping at `capacity`.
struct Reader {
    base: u32,
    capacity: u32,
    pos: u32,
    remaining: u32,
}

impl Reader {
    fn read(&mut self, mem: &GuestMemory) -> u32 {
        let v = mem.read_u32(self.base + self.pos * 4);
        self.pos = (self.pos + 1) % self.capacity;
        self.remaining -= 1;
        v
    }

    fn skip(&mut self, n: u32) {
        let n = n.min(self.remaining);
        self.pos = (self.pos + n) % self.capacity;
        self.remaining -= n;
    }
}

struct CommandProcessor<'a> {
    k: &'static Kernel,
    mem: &'a GuestMemory,
    ctx: &'a mut PPCContext,
    bin_mask: u64,
    bin_select: u64,
    zpd_samples: u32,
    backend: Option<Arc<dyn GpuBackend>>,
    /// Register writes not yet sent to the backend.
    pending: Vec<(u32, u32)>,
    /// Waits for the renderer since the last report (see `report_syncs`).
    syncs: [u32; 128],
    sync_wait: Duration,
    /// Draws since the last swap, and whether a frame with many draws was reported (the end of
    /// a world load, for load-time measurements).
    frame_draws: u32,
    full_frame_seen: bool,
    /// Sequence number of the last `--gpu d3d-verify` marker, until the draw after it, and the
    /// register writes since.
    d3d_marker: Option<u32>,
    d3d_writes: Vec<(u32, u32)>,
    /// `--gpu d3d-verify`: the packet being executed (type 0/1 as 0x100/0x101, else the type-3
    /// opcode) and its address, and per register the last one that wrote it.
    d3d_packet: (u32, u32),
    d3d_last_writer: std::collections::HashMap<u32, (u32, u32)>,
    /// `--gpu d3d`: draw packets are the title's (a draw marker with a new sequence number came
    /// before them, and no D3D call or indirect buffer since: one call can emit several draw
    /// packets), and the last draw sequence number seen.
    d3d_draw_next: bool,
    d3d_last_draw: u32,
    /// `--gpu d3d`: draw packets run and left out since the last report.
    d3d_draws: (u64, u64),
}

/// Register-file base of each SET_CONSTANT / LOAD_ALU_CONSTANT type.
fn constant_base(kind: u32) -> Option<u32> {
    Some(match kind {
        0 => reg::SHADER_CONSTANT_000_X,
        1 => reg::SHADER_CONSTANT_FETCH_00_0,
        2 => reg::SHADER_CONSTANT_BOOL_000_031,
        3 => reg::SHADER_CONSTANT_LOOP_00,
        4 => reg::REGISTERS_BASE,
        _ => return None,
    })
}

impl CommandProcessor<'_> {
    /// Sends the buffered register writes to the backend.
    fn flush_registers(&mut self) {
        if let Some(b) = &self.backend
            && !self.pending.is_empty()
        {
            b.registers(std::mem::take(&mut self.pending));
        }
    }

    /// Lets the backend catch up before an effect the CPU can observe.
    fn sync(&mut self, opcode: u32) {
        self.flush_registers();
        if let Some(b) = &self.backend {
            let started = Instant::now();
            b.wait_idle();
            self.syncs[opcode as usize & 0x7F] += 1;
            self.sync_wait += started.elapsed();
        }
    }

    /// Logs how often and how long the command processor waited for the renderer per frame,
    /// with the swap report every 600 frames.
    fn report_syncs(&mut self, swaps: u64) {
        if swaps.is_multiple_of(600) {
            let total: u32 = self.syncs.iter().sum();
            let by_opcode: Vec<String> = (0..128)
                .filter(|&op| self.syncs[op] != 0)
                .map(|op| format!("{op:#04X}: {:.1}", f64::from(self.syncs[op]) / 600.0))
                .collect();
            log::info!(
                "GPU: {:.1} renderer waits per frame ({}), {:.2} ms blocked per frame",
                f64::from(total) / 600.0,
                by_opcode.join(", "),
                self.sync_wait.as_secs_f64() * 1000.0 / 600.0
            );
            self.syncs = [0; 128];
            self.sync_wait = Duration::ZERO;
        }
    }

    /// Reads `count` dwords of guest physical memory (big-endian words).
    fn read_physical(&self, addr: u32, count: u32) -> Option<Vec<u32>> {
        let base = self.k.rt.space.translate_physical(addr & 0x1FFF_FFFF)?;
        Some((0..count).map(|i| self.mem.read_u32(base + i * 4)).collect())
    }

    fn draw(&mut self, r: &mut Reader, count: u32, viz_token: bool) {
        let mem = self.mem;
        let mut remaining = count;
        if viz_token {
            r.read(mem);
            remaining -= 1;
        }
        if remaining == 0 {
            log::warn!("GPU: draw packet without VGT_DRAW_INITIATOR");
            return;
        }
        let initiator = r.read(mem);
        remaining -= 1;
        self.write_register(reg::VGT_DRAW_INITIATOR, initiator);
        match (initiator >> 6) & 3 {
            // DMA: index buffer address and size follow.
            0 => {
                if remaining < 2 {
                    log::warn!("GPU: indexed draw packet without VGT_DMA_BASE/SIZE");
                    return;
                }
                let base = r.read(mem) & 0x1FFF_FFFF;
                let size = r.read(mem);
                self.write_register(reg::VGT_DMA_BASE, base);
                self.write_register(reg::VGT_DMA_SIZE, size);
            }
            2 => {}
            1 => {
                log::warn!("GPU: draw with immediate indices is not supported");
                return;
            }
            other => {
                log::warn!("GPU: draw with invalid source select {other}");
                return;
            }
        }
        self.flush_registers();
        if let Some(b) = &self.backend {
            b.draw();
        }
    }

    fn load_shader(&mut self, pixel: bool, code: Vec<u32>) {
        self.flush_registers();
        if let Some(b) = &self.backend {
            b.shader(pixel, code.into());
        }
    }

    fn execute(&mut self, r: &mut Reader) {
        while r.remaining > 0 {
            if !self.packet(r) {
                // Malformed stream: drop the rest of this buffer.
                r.skip(r.remaining);
            }
        }
    }

    fn packet(&mut self, r: &mut Reader) -> bool {
        let start = r.base + r.pos * 4;
        let packet = r.read(self.mem);
        log::trace!("{start:#010X}: {packet:08X} ({} dwords left)", r.remaining);
        if packet == 0 || packet == 0x0BAD_F00D {
            return true;
        }
        self.d3d_packet = match packet >> 30 {
            0 => (0x100, start),
            1 => (0x101, start),
            _ => ((packet >> 8) & 0x7F, start),
        };
        match packet >> 30 {
            0 => {
                let count = ((packet >> 16) & 0x3FFF) + 1;
                if count > r.remaining {
                    return self.overflow(start, packet, count, r.remaining);
                }
                let base = packet & 0x7FFF;
                let one_register = packet & 0x8000 != 0;
                for i in 0..count {
                    let v = r.read(self.mem);
                    self.write_register(if one_register { base } else { base + i }, v);
                }
                true
            }
            1 => {
                if r.remaining < 2 {
                    return self.overflow(start, packet, 2, r.remaining);
                }
                let (a, b) = (r.read(self.mem), r.read(self.mem));
                self.write_register(packet & 0x7FF, a);
                self.write_register((packet >> 11) & 0x7FF, b);
                true
            }
            2 => true,
            _ => {
                let opcode = (packet >> 8) & 0x7F;
                let count = ((packet >> 16) & 0x3FFF) + 1;
                if count > r.remaining {
                    return self.overflow(start, packet, count, r.remaining);
                }
                if packet & 1 != 0 && ((self.bin_select & self.bin_mask) == 0 || opcode == pm4::XE_SWAP) {
                    r.skip(count);
                    return true;
                }
                let end = r.remaining - count;
                let ok = self.type3(r, opcode, count);
                // Every handler consumes exactly its payload; enforce it so a
                // short read can't desynchronize the stream.
                if r.remaining > end {
                    r.skip(r.remaining - end);
                }
                ok
            }
        }
    }

    fn overflow(&self, at: u32, packet: u32, count: u32, remaining: u32) -> bool {
        log::error!("GPU: packet {packet:#010X} at {at:#010X} needs {count} dwords, {remaining} left");
        false
    }

    fn type3(&mut self, r: &mut Reader, opcode: u32, count: u32) -> bool {
        let mem = self.mem;
        match opcode {
            pm4::INTERRUPT => {
                let cpu_mask = r.read(mem);
                self.sync(pm4::INTERRUPT);
                log::trace!("INTERRUPT cpu mask {cpu_mask:#X}");
                for cpu in 0..6 {
                    if cpu_mask & (1 << cpu) != 0 {
                        self.k.gpu_dispatch_interrupt(self.ctx, 1, cpu);
                    }
                }
            }
            pm4::XE_SWAP => {
                let signature = r.read(mem);
                if signature != SWAP_SIGNATURE {
                    log::warn!("GPU: XE_SWAP with bad signature {signature:#010X}");
                }
                let (fb, w, h) = (r.read(mem), r.read(mem), r.read(mem));
                self.flush_registers();
                if crate::d3d::gpu_mode() == crate::d3d::GpuMode::D3d {
                    // The present is a D3D call of its own (calls whose marker didn't fit go
                    // with the next marker: the title queues calls ahead of this point, so
                    // they can't all be delivered here).
                    let n = self.k.gpu.swaps.load(Ordering::Relaxed) + 1;
                    if n.is_multiple_of(600) {
                        let (run, left) = std::mem::take(&mut self.d3d_draws);
                        log::info!("GPU: d3d: {run} draw packets run, {left} left out over 600 swaps");
                    }
                } else if let Some(b) = &self.backend {
                    let fetch = std::array::from_fn(|i| self.read_register(reg::SHADER_CONSTANT_FETCH_00_0 + i as u32));
                    b.swap(fetch, w, h);
                }
                let swaps = self.k.gpu_swap(fb, w, h);
                self.report_syncs(swaps);
                if !self.full_frame_seen && self.frame_draws >= FULL_FRAME_DRAWS {
                    self.full_frame_seen = true;
                    log::info!("GPU: first frame with {} draws (swap #{swaps})", self.frame_draws);
                }
                self.frame_draws = 0;
            }
            pm4::INDIRECT_BUFFER | pm4::INDIRECT_BUFFER_PFD => {
                self.d3d_draw_next = false;
                let ptr = r.read(mem) & 0x1FFF_FFFF;
                let length = r.read(mem) & 0xF_FFFF;
                log::trace!("indirect buffer {ptr:#010X}, {length} dwords");
                if length != 0 {
                    match self.k.rt.space.translate_physical(ptr) {
                        Some(base) => {
                            let mut ib = Reader { base, capacity: length, pos: 0, remaining: length };
                            self.execute(&mut ib);
                        }
                        None => log::error!("GPU: indirect buffer at unmapped {ptr:#010X}"),
                    }
                }
            }
            pm4::WAIT_REG_MEM => {
                let (info, addr, reference, mask, wait) =
                    (r.read(mem), r.read(mem), r.read(mem), r.read(mem), r.read(mem));
                log::trace!(
                    "WAIT_REG_MEM info {info:#X} addr {addr:#010X} (at {:?}) ref {reference:#X} mask {mask:#X} wait {wait:#X}",
                    (info & 0x10 != 0).then(|| self.k.rt.space.translate_physical(addr & !3)).flatten()
                );
                // Usually already satisfied (Skyrim issues ~30 per frame): then the renderer
                // needn't catch up first, which would stall the command processor.
                if !self.reg_mem_matches(info, addr, reference, mask) {
                    self.sync(pm4::WAIT_REG_MEM);
                    self.wait_reg_mem(info, addr, reference, mask, wait);
                }
            }
            pm4::REG_RMW => {
                let (info, and_mask, or_mask) = (r.read(mem), r.read(mem), r.read(mem));
                let index = info & 0x1FFF;
                let mut v = self.read_register(index);
                v &= if info >> 31 != 0 { self.read_register(and_mask & 0x1FFF) } else { and_mask };
                v |= if (info >> 30) & 1 != 0 { self.read_register(or_mask & 0x1FFF) } else { or_mask };
                self.write_register(index, v);
            }
            pm4::REG_TO_MEM => {
                let (index, addr) = (r.read(mem), r.read(mem));
                self.sync(pm4::REG_TO_MEM);
                let v = self.read_register(index);
                self.store(addr, v);
            }
            pm4::MEM_WRITE => {
                self.sync(pm4::MEM_WRITE);
                let mut addr = r.read(mem);
                for _ in 1..count {
                    let v = r.read(mem);
                    self.store(addr, v);
                    addr = addr.wrapping_add(4);
                }
            }
            pm4::COND_WRITE => {
                let (info, poll, reference, mask, write_addr, data) =
                    (r.read(mem), r.read(mem), r.read(mem), r.read(mem), r.read(mem), r.read(mem));
                self.sync(pm4::COND_WRITE);
                let value = if info & 0x10 != 0 { self.load(poll) } else { self.read_register(poll) };
                if match_value_and_ref(value & mask, reference, info) {
                    if info & 0x100 != 0 {
                        self.store(write_addr, data);
                    } else {
                        self.write_register(write_addr, data);
                    }
                }
            }
            pm4::EVENT_WRITE => {
                let initiator = r.read(mem);
                self.write_register(reg::VGT_EVENT_INITIATOR, initiator & 0x3F);
            }
            pm4::EVENT_WRITE_SHD => {
                let (initiator, addr, value) = (r.read(mem), r.read(mem), r.read(mem));
                self.write_register(reg::VGT_EVENT_INITIATOR, initiator & 0x3F);
                self.sync(pm4::EVENT_WRITE_SHD);
                let data = if initiator >> 31 != 0 { self.k.gpu.counter.load(Ordering::Relaxed) } else { value };
                self.event_write_shd(addr, data);
            }
            pm4::EVENT_WRITE_EXT => {
                let (initiator, addr) = (r.read(mem), r.read(mem));
                self.write_register(reg::VGT_EVENT_INITIATOR, initiator & 0x3F);
                // No wait for the renderer: the extents are constants, not a fence (Skyrim
                // issues ~500 of these per frame, and waiting at each one serialized the command
                // processor with the renderer).
                // Screen extents: min/max x, min/max y (in 8-pixel units), min/max z.
                const MAX_SIZE: u16 = 8192;
                let extents = [0, MAX_SIZE >> 3, 0, MAX_SIZE >> 3, 0, 1];
                match self.k.rt.space.translate_physical(addr & !3) {
                    Some(a) => {
                        for (i, e) in extents.into_iter().enumerate() {
                            mem.write_u16(a + i as u32 * 2, e);
                        }
                    }
                    None => log::warn!("GPU: EVENT_WRITE_EXT to unmapped {addr:#010X}"),
                }
            }
            pm4::EVENT_WRITE_ZPD => {
                let initiator = r.read(mem);
                self.write_register(reg::VGT_EVENT_INITIATOR, initiator & 0x3F);
                self.sync(pm4::EVENT_WRITE_ZPD);
                self.write_zpd_report();
            }
            pm4::SET_CONSTANT => {
                let offset_type = r.read(mem);
                let index = offset_type & 0x7FF;
                let Some(base) = constant_base((offset_type >> 16) & 0xFF) else {
                    log::debug!("GPU: SET_CONSTANT type {}", (offset_type >> 16) & 0xFF);
                    r.skip(count - 1);
                    return true;
                };
                for i in 0..count - 1 {
                    let v = r.read(mem);
                    self.write_register(base + index + i, v);
                }
            }
            pm4::LOAD_ALU_CONSTANT => {
                let (address, offset_type, size) = (r.read(mem) & 0x3FFF_FFFF, r.read(mem), r.read(mem) & 0xFFF);
                let index = offset_type & 0x7FF;
                match (constant_base((offset_type >> 16) & 0xFF), self.read_physical(address, size)) {
                    (Some(base), Some(values)) => {
                        for (i, v) in values.into_iter().enumerate() {
                            self.write_register(base + index + i as u32, v);
                        }
                    }
                    (None, _) => log::debug!("GPU: LOAD_ALU_CONSTANT type {}", (offset_type >> 16) & 0xFF),
                    (_, None) => log::warn!("GPU: LOAD_ALU_CONSTANT from unmapped {address:#010X}"),
                }
            }
            pm4::IM_LOAD => {
                let (addr_type, start_size) = (r.read(mem), r.read(mem));
                let size = start_size & 0xFFFF;
                match self.read_physical(addr_type & !3, size) {
                    Some(code) => self.load_shader(addr_type & 3 == 1, code),
                    None => log::warn!("GPU: IM_LOAD from unmapped {:#010X}", addr_type & !3),
                }
            }
            pm4::IM_LOAD_IMMEDIATE => {
                let (shader_type, start_size) = (r.read(mem), r.read(mem));
                let size = (start_size & 0xFFFF).min(count - 2);
                let code: Vec<u32> = (0..size).map(|_| r.read(mem)).collect();
                self.load_shader(shader_type & 3 == 1, code);
            }
            pm4::DRAW_INDX | pm4::DRAW_INDX_2 if !self.d3d_title_draw() => {
                self.d3d_draws.1 += 1;
                r.skip(count);
            }
            pm4::DRAW_INDX => {
                self.d3d_draws.0 += 1;
                self.frame_draws += 1;
                self.verify_d3d_draw();
                self.draw(r, count, true)
            }
            pm4::DRAW_INDX_2 => {
                self.d3d_draws.0 += 1;
                self.frame_draws += 1;
                self.verify_d3d_draw();
                self.draw(r, count, false)
            }
            // A `--gpu d3d` marker (see crate::d3d).
            pm4::NOP if count >= 3 && crate::d3d::gpu_mode() == crate::d3d::GpuMode::D3d => {
                if r.read(mem) == crate::d3d::CALL_MARKER_SIGNATURE {
                    let (kind, value) = (r.read(mem), r.read(mem));
                    self.d3d_marker_reached(kind, value);
                }
            }
            // A `--gpu d3d-verify` marker before a draw's packets (see crate::d3d).
            pm4::NOP if count >= 2 && crate::d3d::gpu_mode() == crate::d3d::GpuMode::D3dVerify => {
                if r.read(mem) == d3d_hle::MARKER_SIGNATURE {
                    self.d3d_marker = Some(r.read(mem));
                    self.d3d_writes.clear();
                }
            }
            pm4::SET_CONSTANT2 | pm4::SET_SHADER_CONSTANTS => {
                let offset_type = r.read(mem);
                let index = offset_type & 0xFFFF;
                for i in 0..count - 1 {
                    let v = r.read(mem);
                    self.write_register(index + i, v);
                }
            }
            pm4::SET_BIN_MASK_LO => self.bin_mask = (self.bin_mask & !0xFFFF_FFFF) | r.read(mem) as u64,
            pm4::SET_BIN_MASK_HI => self.bin_mask = (self.bin_mask & 0xFFFF_FFFF) | (r.read(mem) as u64) << 32,
            pm4::SET_BIN_SELECT_LO => self.bin_select = (self.bin_select & !0xFFFF_FFFF) | r.read(mem) as u64,
            pm4::SET_BIN_SELECT_HI => {
                self.bin_select = (self.bin_select & 0xFFFF_FFFF) | (r.read(mem) as u64) << 32;
            }
            pm4::SET_BIN_MASK if count == 2 => {
                let (hi, lo) = (r.read(mem) as u64, r.read(mem) as u64);
                self.bin_mask = hi << 32 | lo;
            }
            pm4::SET_BIN_SELECT if count == 2 => {
                let (hi, lo) = (r.read(mem) as u64, r.read(mem) as u64);
                self.bin_select = hi << 32 | lo;
            }
            // Packets with nothing to emulate here: pipeline syncs (the backend
            // executes in order), binned/predicated variants and state saves.
            pm4::ME_INIT
            | pm4::NOP
            | pm4::WAIT_FOR_IDLE
            | pm4::WAIT_IB_PFD_COMPLETE
            | pm4::WAIT_UNTIL_READ
            | pm4::CONTEXT_UPDATE
            | pm4::DRAW_INDX_BIN
            | pm4::DRAW_INDX_2_BIN
            | pm4::VIZ_QUERY
            | pm4::SET_STATE
            | pm4::LOAD_CONSTANT_CONTEXT
            | pm4::IM_STORE
            | pm4::INVALIDATE_STATE
            | pm4::SET_SHADER_BASES
            | pm4::SET_BIN_BASE_OFFSET
            | pm4::EVENT_WRITE_CFL
            | pm4::COND_EXEC
            | pm4::MEM_WRITE_CNTR
            | pm4::WAIT_REG_EQ
            | pm4::WAIT_REG_GTE => r.skip(count),
            other => {
                log::warn!("GPU: unimplemented PM4 opcode {other:#04X} ({count} dwords)");
                r.skip(count);
                return false;
            }
        }
        true
    }

    fn read_register(&self, index: u32) -> u32 {
        self.k.gpu.read_register(self.mem, index)
    }

    /// `--gpu d3d`: whether the draw packet at hand is one of the title's. Other modes run every
    /// draw.
    fn d3d_title_draw(&self) -> bool {
        crate::d3d::gpu_mode() != crate::d3d::GpuMode::D3d || self.d3d_draw_next
    }

    fn d3d_marker_reached(&mut self, kind: u32, value: u32) {
        match kind {
            crate::d3d::MARKER_DRAW => {
                // Tile replays bring old numbers back: those draws already ran.
                self.d3d_draw_next = value.wrapping_sub(self.d3d_last_draw) as i32 > 0;
                if self.d3d_draw_next {
                    self.d3d_last_draw = value;
                }
            }
            crate::d3d::MARKER_CALL => {
                self.d3d_draw_next = false;
                self.d3d_deliver(value);
            }
            _ => {}
        }
    }

    /// Sends the queued D3D calls up to `id` to the renderer, after the register writes before
    /// them.
    fn d3d_deliver(&mut self, id: u32) {
        let calls = crate::d3d::take_calls(id);
        if calls.is_empty() {
            return;
        }
        self.flush_registers();
        if let Some(b) = &self.backend {
            for call in calls {
                b.d3d(call);
            }
        }
    }

    /// Compares the registers with the device shadow copied at the marker before this draw.
    fn verify_d3d_draw(&mut self) {
        if crate::d3d::gpu_mode() == crate::d3d::GpuMode::D3dVerify && self.d3d_marker.is_none() {
            crate::d3d::unmarked_draw(self.d3d_packet.1);
        }
        if let Some(seq) = self.d3d_marker.take() {
            let writes = std::mem::take(&mut self.d3d_writes);
            let writers = &self.d3d_last_writer;
            crate::d3d::verify_draw(
                seq,
                &writes,
                |index| self.k.gpu.read_register(self.mem, index),
                |index| writers.get(&index).copied().unwrap_or((0, 0)),
            );
            self.d3d_writes = writes;
        }
    }

    /// Register write from the command stream, with the side effects of the
    /// registers that have them.
    fn write_register(&mut self, index: u32, value: u32) {
        if self.backend.is_some() {
            self.pending.push((index, value));
        }
        if self.d3d_marker.is_some() {
            self.d3d_writes.push((index, value));
        }
        if crate::d3d::gpu_mode() == crate::d3d::GpuMode::D3dVerify {
            self.d3d_last_writer.insert(index, self.d3d_packet);
        }
        let gpu = &self.k.gpu;
        let mem = self.mem;
        if (reg::SCRATCH_REG0..=reg::SCRATCH_REG7).contains(&index) {
            gpu.write_register(mem, index, value);
            let n = index - reg::SCRATCH_REG0;
            if gpu.read_register(mem, reg::SCRATCH_UMSK) & (1 << n) != 0 {
                let addr = gpu.read_register(mem, reg::SCRATCH_ADDR).wrapping_add(n * 4);
                match self.k.rt.space.translate_physical(addr) {
                    Some(a) => mem.write_u32(a, value),
                    None => log::warn!("GPU: scratch writeback to unmapped {addr:#010X}"),
                }
            }
        } else if index == reg::COHER_STATUS_HOST {
            // A coherency request; with no GPU caches it completes at once.
            gpu.write_register(mem, index, 0);
        } else {
            gpu.write_register(mem, index, value);
        }
    }

    /// GPU-side 32-bit load from physical memory with the address's endian mode.
    fn load(&self, addr: u32) -> u32 {
        match self.k.rt.space.translate_physical(addr & !3) {
            Some(a) => gpu_swap(self.mem.read_u32(a).swap_bytes(), addr & 3),
            None => {
                log::warn!("GPU: read from unmapped {addr:#010X}");
                0
            }
        }
    }

    fn store(&self, addr: u32, value: u32) {
        match self.k.rt.space.translate_physical(addr & !3) {
            Some(a) => self.mem.write_u32(a, gpu_swap(value, addr & 3).swap_bytes()),
            None => log::warn!("GPU: write to unmapped {addr:#010X}"),
        }
    }

    /// Whether WAIT_REG_MEM's condition holds now.
    fn reg_mem_matches(&self, info: u32, addr: u32, reference: u32, mask: u32) -> bool {
        let value = if info & 0x10 != 0 { self.load(addr) } else { self.read_register(addr) };
        match_value_and_ref(value & mask, reference, info)
    }

    fn wait_reg_mem(&self, info: u32, addr: u32, reference: u32, mask: u32, wait: u32) {
        let is_memory = info & 0x10 != 0;
        let started = Instant::now();
        let mut warned = false;
        loop {
            let value = if is_memory { self.load(addr) } else { self.read_register(addr) };
            if match_value_and_ref(value & mask, reference, info) {
                return;
            }
            if wait >= 0x100 {
                std::thread::sleep(Duration::from_millis((wait / 0x100) as u64));
            } else {
                std::thread::yield_now();
            }
            if !warned && started.elapsed() > Duration::from_secs(5) {
                warned = true;
                log::warn!(
                    "GPU: WAIT_REG_MEM on {} {addr:#010X} (value {value:#X} & {mask:#X}, ref {reference:#X}, op {}) still waiting after 5 s",
                    if is_memory { "memory" } else { "register" },
                    info & 7
                );
            }
        }
    }

    fn event_write_shd(&self, addr: u32, data: u32) {
        let endian = addr & 3;
        let addr = addr & !3;
        if addr > 0x1FFF_FFFF {
            // Inside the writeback window the address is a CPU (virtual)
            // address rather than a physical one.
            let start = self.read_register(reg::WRITEBACK_START);
            let size = self.read_register(reg::WRITEBACK_SIZE);
            if start != 0 && addr.wrapping_sub(start) < size && self.k.rt.space.is_committed(addr) {
                self.mem.write_u32(addr, gpu_swap(data, endian).swap_bytes());
                return;
            }
        }
        self.store(addr | endian, data);
    }

    /// Occlusion query sample (Xenia's fake ZPD path). D3D brackets a query
    /// with two of these reports and takes end minus begin, so a running
    /// counter that grows on every report makes every query visible.
    fn write_zpd_report(&mut self) {
        let report = self.read_register(reg::RB_SAMPLE_COUNT_ADDR);
        if report == 0 {
            return;
        }
        self.zpd_samples = self.zpd_samples.wrapping_add(ZPD_SAMPLES_PER_REPORT);
        // xe_gpu_depth_sample_counts: Total, ZFail, ZPass, StencilFail as
        // little-endian A/B pairs.
        let Some(a) = self.k.rt.space.translate_physical(report) else {
            log::warn!("GPU: occlusion report to unmapped {report:#010X}");
            return;
        };
        let counts = [self.zpd_samples, 0, 0, 0, self.zpd_samples, 0, 0, 0];
        for (i, v) in counts.into_iter().enumerate() {
            self.mem.write_u32(a + i as u32 * 4, v.swap_bytes());
        }
    }
}

/// Samples each fake occlusion report adds (an on-screen 32x32 quad).
const ZPD_SAMPLES_PER_REPORT: u32 = 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_swap_modes() {
        assert_eq!(gpu_swap(0x1122_3344, 0), 0x1122_3344);
        assert_eq!(gpu_swap(0x1122_3344, 1), 0x2211_4433);
        assert_eq!(gpu_swap(0x1122_3344, 2), 0x4433_2211);
        assert_eq!(gpu_swap(0x1122_3344, 3), 0x3344_1122);
    }

    #[test]
    fn wait_comparisons() {
        assert!(!match_value_and_ref(1, 1, 0));
        assert!(match_value_and_ref(1, 2, 1));
        assert!(match_value_and_ref(2, 2, 2));
        assert!(match_value_and_ref(2, 2, 3));
        assert!(match_value_and_ref(1, 2, 4));
        assert!(match_value_and_ref(2, 2, 5));
        assert!(match_value_and_ref(3, 2, 6));
        assert!(match_value_and_ref(0, 9, 7));
        assert!(!match_value_and_ref(3, 2, 1));
    }

    #[test]
    fn packet_headers() {
        assert_eq!(packet_type3(pm4::XE_SWAP, 4), 0xC003_6400);
        assert_eq!(packet_type0(reg::SHADER_CONSTANT_FETCH_00_0, 6), 0x0005_4800);
        assert_eq!(SWAP_SIGNATURE, 0x5357_4150);
    }
}

/// Command-processor progress counter, for title code that waits for ring space (guest-sys
/// wraps D3D's ring-space poll with these instead of letting it spin; see its build script).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_gpu_progress() -> u32 {
    Kernel::try_global().map_or(0, |k| k.gpu.progress.load(Ordering::SeqCst))
}

/// Sleeps until the command processor's progress counter differs from `seen`, at most
/// `timeout_us` microseconds.
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_gpu_wait_progress(seen: u32, timeout_us: u32) {
    if let Some(k) = Kernel::try_global() {
        k.gpu.wait_progress(seen, Duration::from_micros(u64::from(timeout_us)));
    }
}
