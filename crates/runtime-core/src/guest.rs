//! The interface to recompiled guest code.
//!
//! guest-sys builds the XenonRecomp output and fills in a [`GuestModule`];
//! everything here depends only on that descriptor, so this crate can be built
//! and tested without any generated code.

use crate::address_space::{AddressSpace, AllocResult};
use crate::context::PPCContext;
use crate::memory::GuestMemory;

/// `void f(PPCContext& __restrict ctx, uint8_t* base)`. "C-unwind" because guest
/// thread termination unwinds through recompiled frames.
pub type PPCFunc = unsafe extern "C-unwind" fn(ctx: *mut PPCContext, base: *mut u8);

/// One entry of XenonRecomp's `PPCFuncMappings[]` (`{ size_t guest; PPCFunc* host; }`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FuncMapping {
    pub guest: usize,
    pub host: Option<PPCFunc>,
}

/// Constants from the generated ppc_config.h plus the entry points guest-sys exports.
#[derive(Clone, Copy)]
pub struct GuestModule {
    pub image_base: u32,
    pub image_size: u32,
    pub code_base: u32,
    pub code_size: u32,
    /// Terminated by an entry with `guest == 0`.
    pub functions: &'static [FuncMapping],
    /// Constructs a `PPCContext` in place and captures the host FP control word.
    pub init_context: unsafe extern "C" fn(ctx: *mut PPCContext),
}

impl GuestModule {
    /// Start of the indirect-call table (`PPC_LOOKUP_FUNC`): one pointer-sized
    /// slot per guest instruction, 8 bytes apart, right after the image.
    pub fn table_base(&self) -> u32 {
        self.image_base + self.image_size
    }

    pub fn table_size(&self) -> u32 {
        self.code_size * 2
    }

    fn table_slot(&self, guest_addr: u32) -> Option<u32> {
        let off = guest_addr.checked_sub(self.code_base)?;
        (off < self.code_size && off % 4 == 0).then(|| self.table_base() + off * 2)
    }

    /// Reserves the image + table range and fills the table from the mappings.
    /// Returns the number of functions installed.
    pub fn install(&self, space: &AddressSpace) -> AllocResult<usize> {
        space.alloc_xex(self.image_base, self.image_size + self.table_size())?;
        let mut installed = 0;
        for m in self.functions.iter().take_while(|m| m.guest != 0) {
            if let Some(host) = m.host
                && self.set(&space.mem, m.guest as u32, host)
            {
                installed += 1;
            }
        }
        Ok(installed)
    }

    /// Points the table slot for `guest_addr` at `host`.
    pub fn set(&self, mem: &GuestMemory, guest_addr: u32, host: PPCFunc) -> bool {
        let Some(slot) = self.table_slot(guest_addr) else { return false };
        // SAFETY: the slot lies in the committed table range; it holds a host
        // function pointer in native byte order, as PPC_LOOKUP_FUNC reads it.
        unsafe { std::ptr::write_unaligned(mem.ptr(slot) as *mut usize, host as usize) };
        true
    }

    pub fn lookup(&self, mem: &GuestMemory, guest_addr: u32) -> Option<PPCFunc> {
        let slot = self.table_slot(guest_addr)?;
        // SAFETY: as in `set`; zero means no function was recompiled there.
        let raw = unsafe { std::ptr::read_unaligned(mem.ptr(slot) as *const usize) };
        if raw == 0 {
            return None;
        }
        // SAFETY: non-zero slots are only ever written from PPCFunc values.
        Some(unsafe { std::mem::transmute::<usize, PPCFunc>(raw) })
    }
}
