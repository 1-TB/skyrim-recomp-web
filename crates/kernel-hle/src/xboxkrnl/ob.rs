//! Object manager exports.
#![allow(non_snake_case)]

use super::sync::handle_object;
use crate::dispatch::{Call, Export, ModuleId};
use crate::objects::{CURRENT_PROCESS, ObjectType, Payload};
use crate::status;
use crate::strings::{read_object_attributes, write_string_header};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x0CF, NtClose),
    export!(K, 0x0DA, NtDuplicateObject),
    export!(K, 0x110, ObReferenceObjectByHandle),
    export!(K, 0x105, ObDereferenceObject),
    export!(K, 0x0E0, NtOpenSymbolicLinkObject),
    export!(K, 0x0EC, NtQuerySymbolicLinkObject),
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
            ObjectType::Timer | ObjectType::SymbolicLink | ObjectType::Other(_) => 0,
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

/// (HANDLE* Handle, OBJECT_ATTRIBUTES* Attributes): opens a drive link such as
/// `\??\update:` so its target can be queried; titles use it to ask whether a
/// device (an installed title update) is present. Unknown links are
/// STATUS_NO_SUCH_FILE, as in Xenia.
fn NtOpenSymbolicLinkObject(c: &mut Call) -> u64 {
    let (handle_ptr, attrs) = (c.arg(0), c.arg(1));
    let Some(name) = read_object_attributes(c.mem(), attrs).and_then(|a| a.name) else {
        return status::INVALID_PARAMETER as u64;
    };
    if handle_ptr == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    if !name.is_ascii() {
        return status::OBJECT_NAME_INVALID as u64;
    }
    let link = vfs::path::strip_object_prefix(&name).trim_end_matches(['\\', '/']);
    let Some(target) = c.kernel.vfs.symlink_target(link) else {
        log::info!("NtOpenSymbolicLinkObject({name:?}): no such link");
        return status::NO_SUCH_FILE as u64;
    };
    let payload = Payload::SymbolicLink(target.clone());
    match c.kernel.objects.create(&c.kernel.rt.space, ObjectType::SymbolicLink, None, payload, None) {
        Ok(obj) => {
            let h = c.kernel.objects.insert_handle(&obj);
            c.mem().write_u32(handle_ptr, h);
            log::info!("NtOpenSymbolicLinkObject({name:?}) -> {target}");
            status::SUCCESS as u64
        }
        Err(s) => s as u64,
    }
}

/// (HANDLE Link, ANSI_STRING* Target): copies the target device path into the
/// caller's buffer, truncated to MaximumLength (Xenia's behaviour).
fn NtQuerySymbolicLinkObject(c: &mut Call) -> u64 {
    let (handle, out) = (c.arg(0), c.arg(1));
    let Some(obj) = handle_object(c, handle) else { return status::NO_SUCH_FILE as u64 };
    let Payload::SymbolicLink(target) = &obj.payload else { return status::OBJECT_TYPE_MISMATCH as u64 };
    if out == 0 {
        return status::INVALID_PARAMETER as u64;
    }
    let mem = c.mem();
    let max = mem.read_u16(out + 2) as usize;
    let buf = mem.read_u32(out + 4);
    let len = max.min(target.len());
    if len > 0 && buf != 0 {
        mem.write_bytes(buf, &target.as_bytes()[..len]);
    }
    write_string_header(mem, out, len as u16, max as u16, buf);
    status::SUCCESS as u64
}
