//! Object manager exports.
#![allow(non_snake_case)]

use super::sync::handle_object;
use crate::dispatch::{Call, Export, ModuleId};
use crate::objects::{CURRENT_PROCESS, ObjectType};
use crate::status;

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x0CF, NtClose),
    export!(K, 0x0DA, NtDuplicateObject),
    export!(K, 0x110, ObReferenceObjectByHandle),
    export!(K, 0x105, ObDereferenceObject),
];

const DUPLICATE_CLOSE_SOURCE: u32 = 0x1;

fn NtClose(c: &mut Call) -> u64 {
    let handle = c.arg(0);
    if c.kernel.objects.close_handle(&c.kernel.rt.space, handle) {
        status::SUCCESS as u64
    } else {
        status::INVALID_HANDLE as u64
    }
}

fn NtDuplicateObject(c: &mut Call) -> u64 {
    let (handle, out_ptr, options) = (c.arg(0), c.arg(1), c.arg(2));
    let Some(obj) = handle_object(c, handle) else { return status::INVALID_HANDLE as u64 };
    let new = c.kernel.objects.insert_handle(&obj);
    if out_ptr != 0 {
        c.mem().write_u32(out_ptr, new);
    }
    if options & DUPLICATE_CLOSE_SOURCE != 0 {
        c.kernel.objects.close_handle(&c.kernel.rt.space, handle);
    }
    status::SUCCESS as u64
}

/// Returns the object's guest body and takes a pointer reference. When an
/// object type (one of the Ex*ObjectType exports) is given it must match.
fn ObReferenceObjectByHandle(c: &mut Call) -> u64 {
    let (handle, type_ptr, out_ptr) = (c.arg(0), c.arg(1), c.arg(2));
    if handle == CURRENT_PROCESS {
        return status::INVALID_HANDLE as u64;
    }
    let Some(obj) = handle_object(c, handle) else { return status::INVALID_HANDLE as u64 };
    if type_ptr != 0 {
        let v = &c.kernel.vars;
        let expected = match obj.ty {
            ObjectType::Thread => v.thread_object_type,
            ObjectType::Event => v.event_object_type,
            ObjectType::Mutant => v.mutant_object_type,
            ObjectType::Semaphore => v.semaphore_object_type,
            ObjectType::File => v.io_file_object_type,
            ObjectType::Other(_) => 0,
        };
        if type_ptr != expected {
            return status::OBJECT_TYPE_MISMATCH as u64;
        }
    }
    c.kernel.objects.add_ref(&obj);
    if out_ptr != 0 {
        c.mem().write_u32(out_ptr, obj.guest_ptr);
    }
    status::SUCCESS as u64
}

fn ObDereferenceObject(c: &mut Call) -> u64 {
    let ptr = c.arg(0);
    match c.kernel.objects.by_guest_ptr(ptr) {
        Some(obj) => c.kernel.objects.release(&c.kernel.rt.space, &obj),
        None => log::warn!("ObDereferenceObject({ptr:#010X}): unknown object"),
    }
    0
}
