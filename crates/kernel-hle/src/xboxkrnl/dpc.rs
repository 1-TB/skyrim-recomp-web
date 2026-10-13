//! Deferred procedure calls.
//!
//! A DPC queued on hardware runs later at DISPATCH_LEVEL on its target
//! processor. Here it runs inline on the queuing thread, impersonating DPC
//! context (raised IRQL, KPRCB.DpcActive set) the way Xenia does: DPC routines
//! reach per-CPU state through r13, which the calling thread already has.
#![allow(non_snake_case)]

use runtime_core::PPCContext;
use runtime_core::layout::kpcr;
use runtime_core::memory::GuestMemory;

use crate::Kernel;
use crate::dispatch::{Call, Export, ModuleId};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] =
    &[export!(K, 0x06F, KeInitializeDpc), export!(K, 0x07B, KeInsertQueueDpc), export!(K, 0x08E, KeRemoveQueueDpc)];

/// KDPC layout.
mod kdpc {
    pub const TYPE: u32 = 0x0;
    pub const SELECTED_CPU_NUMBER: u32 = 0x2;
    pub const DESIRED_CPU_NUMBER: u32 = 0x3;
    pub const LIST_ENTRY: u32 = 0x4;
    pub const ROUTINE: u32 = 0xC;
    pub const CONTEXT: u32 = 0x10;
    pub const ARG1: u32 = 0x14;
    pub const ARG2: u32 = 0x18;
    pub const SIZE: u32 = 0x1C;
}

/// DPC object type (the value KeInitializeDpc stores in `type`).
const DPC_OBJECT_TYPE: u16 = 19;
const IRQL_DISPATCH: u8 = 2;

/// Previous IRQL of an impersonated DPC; `None` when the thread was already
/// running a DPC and nothing was changed.
pub struct DpcScope {
    pcr: u32,
    previous_irql: Option<u8>,
}

/// Enters DPC context on the processor whose PCR is `pcr`.
pub fn begin_dpc(mem: &GuestMemory, pcr: u32) -> DpcScope {
    if mem.read_u32(pcr + kpcr::PRCB_DPC_ACTIVE) != 0 {
        return DpcScope { pcr, previous_irql: None };
    }
    let previous = mem.read_u8(pcr + kpcr::CURRENT_IRQL);
    mem.write_u8(pcr + kpcr::CURRENT_IRQL, IRQL_DISPATCH);
    mem.write_u32(pcr + kpcr::PRCB_DPC_ACTIVE, 1);
    DpcScope { pcr, previous_irql: Some(previous) }
}

pub fn end_dpc(mem: &GuestMemory, scope: DpcScope) {
    if let Some(irql) = scope.previous_irql {
        mem.write_u8(scope.pcr + kpcr::CURRENT_IRQL, irql);
        mem.write_u32(scope.pcr + kpcr::PRCB_DPC_ACTIVE, 0);
    }
}

/// Calls the guest function `routine` with `args` in r3.. as a DPC, leaving
/// `ctx` as it was (apart from guest memory the routine changed).
pub fn call_as_dpc(kernel: &Kernel, ctx: &mut PPCContext, routine: u32, args: &[u32], process_type: Option<u8>) {
    let mem = kernel.mem();
    let pcr = ctx.gpr32(13);
    let scope = begin_dpc(mem, pcr);
    let saved = ctx.clone();
    for (i, &a) in args.iter().enumerate() {
        ctx.set_gpr(3 + i, a as u64);
    }
    if let Some(ty) = process_type {
        mem.write_u8(pcr + kpcr::PROCESSTYPE_VALUE_IN_DPC, ty);
    }
    if let Err(e) = kernel.rt.call(ctx, routine) {
        log::error!("DPC routine {routine:#010X}: {e}");
    }
    if process_type.is_some() {
        mem.write_u8(pcr + kpcr::PROCESSTYPE_VALUE_IN_DPC, runtime_core::layout::PROCTYPE_IDLE);
    }
    *ctx = saved;
    end_dpc(mem, scope);
}

/// (KDPC* Dpc, PKDEFERRED_ROUTINE Routine, void* Context)
fn KeInitializeDpc(c: &mut Call) -> u64 {
    let (dpc, routine, context) = (c.arg(0), c.arg(1), c.arg(2));
    let mem = c.mem();
    mem.fill(dpc, kdpc::SIZE, 0);
    mem.write_u16(dpc + kdpc::TYPE, DPC_OBJECT_TYPE);
    mem.write_u8(dpc + kdpc::SELECTED_CPU_NUMBER, 0);
    mem.write_u8(dpc + kdpc::DESIRED_CPU_NUMBER, 0);
    mem.write_u32(dpc + kdpc::ROUTINE, routine);
    mem.write_u32(dpc + kdpc::CONTEXT, context);
    0
}

/// (KDPC* Dpc, void* Arg1, void* Arg2) -> TRUE. Runs the routine as
/// Routine(Dpc, Context, Arg1, Arg2) before returning.
fn KeInsertQueueDpc(c: &mut Call) -> u64 {
    let (dpc, arg1, arg2) = (c.arg(0), c.arg(1), c.arg(2));
    run_dpc(c.kernel, c.ctx, dpc, arg1, arg2, None);
    1
}

/// Delivers a KDPC with its system arguments: runs Routine(Dpc, Context, Arg1, Arg2) on
/// `ctx`'s thread, as KeInsertQueueDpc and expiring timers do.
pub(crate) fn run_dpc(kernel: &Kernel, ctx: &mut PPCContext, dpc: u32, arg1: u32, arg2: u32, process_type: Option<u8>) {
    let mem = kernel.mem();
    mem.write_u32(dpc + kdpc::ARG1, arg1);
    mem.write_u32(dpc + kdpc::ARG2, arg2);
    // Never left on a queue: the list entry stays unlinked.
    mem.write_u32(dpc + kdpc::LIST_ENTRY, 0);
    mem.write_u32(dpc + kdpc::LIST_ENTRY + 4, 0);
    let routine = mem.read_u32(dpc + kdpc::ROUTINE);
    let context = mem.read_u32(dpc + kdpc::CONTEXT);
    if routine != 0 {
        call_as_dpc(kernel, ctx, routine, &[dpc, context, arg1, arg2], process_type);
    }
}

/// Queued DPCs have always run by the time KeInsertQueueDpc returns, so there
/// is never one to remove.
fn KeRemoveQueueDpc(_c: &mut Call) -> u64 {
    0
}
