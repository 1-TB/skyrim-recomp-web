//! Core runtime for recompiled Xbox 360 code: the guest address space, PPC
//! context layout, clocks, kernel structure layouts and the thread model.

pub mod address_space;
pub mod context;
pub mod dirty;
pub mod guest;
pub mod heap;
pub mod layout;
pub mod memory;
pub mod runtime;
pub mod thread;
pub mod timebase;

pub use address_space::AddressSpace;
pub use context::PPCContext;
pub use guest::{FuncMapping, GuestModule, PPCFunc};
pub use memory::GuestMemory;
pub use runtime::Runtime;
