//! Loading the title image and running guest code.

use std::fmt;

use xex::Xex;

use crate::address_space::{AddressSpace, AllocError};
use crate::context::PPCContext;
use crate::guest::GuestModule;
use crate::layout::{PROCTYPE_TITLE, kprocess};
use crate::memory::GuestMemory;
use crate::thread::{ThreadBlock, ThreadFactory, TlsTemplate, init_list_head};

#[derive(Debug)]
pub enum LoadError {
    Alloc(AllocError),
    /// The XEX is not the one the guest code was recompiled from.
    ImageMismatch {
        xex_base: u32,
        xex_size: u32,
        recompiled_base: u32,
        recompiled_size: u32,
    },
    NoEntryPoint,
    Xex(xex::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Alloc(e) => write!(f, "allocation failed: {e}"),
            LoadError::ImageMismatch { xex_base, xex_size, recompiled_base, recompiled_size } => write!(
                f,
                "XEX image {xex_base:#010X}+{xex_size:#X} does not match the recompiled image \
                 {recompiled_base:#010X}+{recompiled_size:#X}; rebuild against this XEX"
            ),
            LoadError::NoEntryPoint => write!(f, "XEX has no entry point"),
            LoadError::Xex(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<AllocError> for LoadError {
    fn from(e: AllocError) -> Self {
        LoadError::Alloc(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallError {
    /// No recompiled function starts at this address.
    NoFunction(u32),
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallError::NoFunction(a) => write!(f, "no recompiled function at {a:#010X}"),
        }
    }
}

impl std::error::Error for CallError {}

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub base: u32,
    pub size: u32,
    pub entry_point: u32,
    /// Guest copy of the XEX header (RtlImageXexHeaderField operates on it).
    pub header_addr: u32,
    pub header_size: u32,
    pub default_stack_size: u32,
}

pub struct Runtime {
    pub space: AddressSpace,
    pub module: GuestModule,
    pub threads: ThreadFactory,
    pub image: ImageInfo,
    pub xex: Xex,
}

/// Title process defaults (Xenia KernelState::InitializeProcess for the title).
const TITLE_PRIORITY_CLASS: u8 = 10;
const TITLE_DEFAULT_PRIORITY: u8 = 13;
const TITLE_MAX_DYNAMIC_PRIORITY: u8 = 17;
const TITLE_QUANTUM: u32 = 60;
const MIN_KERNEL_STACK_SIZE: u32 = 16 * 1024;
const DEFAULT_MAIN_STACK_SIZE: u32 = 256 * 1024;

/// KPROCESS.KernelStackSize, the size ExCreateThread uses for a zero stack
/// size: the XEX default stack size (page rounded, at least 16 KiB), as the
/// loader sets it (Xenia KernelState::SetExecutableModule).
pub fn process_stack_size(xex_default: Option<u32>) -> u32 {
    match xex_default {
        Some(n) if n != 0 => n.div_ceil(0x1000).saturating_mul(0x1000).max(MIN_KERNEL_STACK_SIZE),
        _ => MIN_KERNEL_STACK_SIZE,
    }
}

impl Runtime {
    pub fn load(module: GuestModule, xex: Xex) -> Result<Self, LoadError> {
        if xex.image_base() != module.image_base || xex.image_size() != module.image_size {
            return Err(LoadError::ImageMismatch {
                xex_base: xex.image_base(),
                xex_size: xex.image_size(),
                recompiled_base: module.image_base,
                recompiled_size: module.image_size,
            });
        }
        let entry_point = xex.entry_point().ok_or(LoadError::NoEntryPoint)?;
        let space = AddressSpace::new()?;
        let installed = module.install(&space)?;
        log::info!(
            "image {:#010X}+{:#X}, {installed} recompiled functions, call table at {:#010X}",
            module.image_base,
            module.image_size,
            module.table_base()
        );
        space.mem.write_bytes(module.image_base, xex.image());

        let header = xex.header_bytes();
        let header_addr = space.alloc_xex_anywhere(header.len() as u32)?;
        space.mem.write_bytes(header_addr, header);

        let tls = xex
            .tls_info()
            .map(|t| TlsTemplate {
                slot_count: t.slot_count,
                raw_data_address: t.raw_data_address,
                data_size: t.data_size,
                raw_data_size: t.raw_data_size,
            })
            .unwrap_or_default();
        let default_stack_size = xex.default_stack_size().unwrap_or(DEFAULT_MAIN_STACK_SIZE);
        let process = Self::create_title_process(&space, tls, process_stack_size(xex.default_stack_size()))?;

        let image = ImageInfo {
            base: module.image_base,
            size: module.image_size,
            entry_point,
            header_addr,
            header_size: header.len() as u32,
            default_stack_size,
        };
        Ok(Runtime { space, module, threads: ThreadFactory::new(tls, process), image, xex })
    }

    fn create_title_process(space: &AddressSpace, tls: TlsTemplate, stack_size: u32) -> Result<u32, AllocError> {
        let mem = &space.mem;
        let p = space.system_alloc(kprocess::SIZE, 16)?;
        init_list_head(mem, p + kprocess::THREAD_LIST);
        mem.write_u32(p + kprocess::QUANTUM, TITLE_QUANTUM);
        mem.write_u8(p + kprocess::PROCESS_PRIORITY_CLASS, TITLE_PRIORITY_CLASS);
        mem.write_u8(p + kprocess::DEFAULT_THREAD_PRIORITY, TITLE_DEFAULT_PRIORITY);
        mem.write_u8(p + kprocess::MAX_DYNAMIC_PRIORITY, TITLE_MAX_DYNAMIC_PRIORITY);
        mem.write_u32(p + kprocess::KERNEL_STACK_SIZE, stack_size);
        mem.write_u32(p + kprocess::TLS_STATIC_DATA_ADDRESS, tls.raw_data_address);
        mem.write_u32(p + kprocess::TLS_DATA_SIZE, tls.data_size);
        mem.write_u32(p + kprocess::TLS_RAW_DATA_SIZE, tls.raw_data_size);
        mem.write_u16(p + kprocess::TLS_SLOT_SIZE, (tls.slot_count * 4) as u16);
        mem.write_u8(p + kprocess::PROCESS_TYPE, PROCTYPE_TITLE);
        Ok(p)
    }

    pub fn mem(&self) -> &GuestMemory {
        &self.space.mem
    }

    pub fn base(&self) -> *mut u8 {
        self.space.mem.base_ptr()
    }

    /// A context ready to run on `block`'s thread: FP state captured from the
    /// host, r1 at the top of the guest stack, r13 at the PCR.
    pub fn new_context(&self, block: &ThreadBlock) -> Box<PPCContext> {
        let mut ctx = PPCContext::new_zeroed();
        // SAFETY: the shim placement-constructs a PPCContext over our allocation,
        // whose size and alignment are checked against the C++ type at build time.
        unsafe { (self.module.init_context)(&mut *ctx) };
        ctx.set_gpr(1, block.initial_sp() as u64);
        ctx.set_gpr(13, block.pcr as u64);
        ctx
    }

    /// Calls the recompiled function at `addr` with the registers already in `ctx`.
    pub fn call(&self, ctx: &mut PPCContext, addr: u32) -> Result<(), CallError> {
        let f = self.module.lookup(&self.space.mem, addr).ok_or(CallError::NoFunction(addr))?;
        // SAFETY: `f` is a recompiled function taking (ctx, base); base is the
        // guest memory every recompiled access is relative to.
        unsafe { f(ctx, self.base()) };
        Ok(())
    }
}
