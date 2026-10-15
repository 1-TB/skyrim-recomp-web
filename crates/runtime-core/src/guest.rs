//! The interface to recompiled guest code.
//!
//! guest-sys builds the XenonRecomp output and fills in a [`GuestModule`];
//! everything here depends only on that descriptor, so this crate can be built
//! and tested without any generated code.

use crate::address_space::{AddressSpace, AllocResult};
use crate::context::PPCContext;

/// `void f(PPCContext& __restrict ctx, uint8_t* base)`. "C-unwind" because guest
/// thread termination unwinds through recompiled frames.
pub type PPCFunc = unsafe extern "C-unwind" fn(ctx: *mut PPCContext, base: *mut u8);

/// Calls the recompiled function that starts at guest address `addr`, which must be one of
/// [`GuestModule::functions`] (guest-sys's generated `skyrt_call_func`, which recompiled code's
/// indirect calls also go through).
pub type CallFunc = unsafe extern "C-unwind" fn(addr: u32, ctx: *mut PPCContext, base: *mut u8);

/// Constants from the generated ppc_config.h plus the entry points guest-sys exports.
#[derive(Clone, Copy)]
pub struct GuestModule {
    pub image_base: u32,
    pub image_size: u32,
    pub code_base: u32,
    pub code_size: u32,
    /// Start address of every recompiled function, ascending.
    pub functions: &'static [u32],
    pub call: CallFunc,
    /// Constructs a `PPCContext` in place and captures the host FP control word.
    pub init_context: unsafe extern "C" fn(ctx: *mut PPCContext),
}

impl GuestModule {
    /// Reserves the image range. Returns the number of recompiled functions.
    pub fn install(&self, space: &AddressSpace) -> AllocResult<usize> {
        space.alloc_xex(self.image_base, self.image_size)?;
        Ok(self.functions.len())
    }

    /// Whether a recompiled function starts at `guest_addr`.
    pub fn has_function(&self, guest_addr: u32) -> bool {
        self.functions.binary_search(&guest_addr).is_ok()
    }
}

/// Called by `skyrt_call_func` for an address no recompiled function starts at (a guest
/// indirect call through a bad pointer, or code the recompiler didn't find).
#[unsafe(no_mangle)]
pub extern "C" fn skyrt_missing_function(addr: u32) -> ! {
    panic!("guest called {addr:#010X}, where no recompiled function starts");
}
