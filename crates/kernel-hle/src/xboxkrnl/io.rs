//! File I/O exports, backed by the VFS. All I/O completes synchronously; the
//! completion event is set and the completion APC queued before returning.
#![allow(non_snake_case)]

use std::sync::Arc;

use vfs::{Disposition, Kind, Node, OpenRequest, matches_pattern};

use super::sync::handle_object;
use crate::dispatch::{Call, Export, ModuleId};
use crate::io::{
    DirectoryScan, FILE_USE_FILE_POINTER_POSITION, FileObject, NETWORK_OPEN_INFO_SIZE, directory_info_size,
    vfs_error_status, write_basic_info, write_directory_info, write_iosb, write_network_open_info, write_standard_info,
};
use crate::objects::{KObject, ObjectType, Payload, SyncState, UserApc};
use crate::status;
use crate::strings::{read_ansi_lossy, read_object_attributes};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x021, FscSetCacheElementCount),
    export!(K, 0x0D2, NtCreateFile),
    export!(K, 0x0DF, NtOpenFile),
    export!(K, 0x0F0, NtReadFile),
    export!(K, 0x0FF, NtWriteFile),
    export!(K, 0x0E8, NtQueryInformationFile),
    export!(K, 0x0F7, NtSetInformationFile),
    export!(K, 0x0E7, NtQueryFullAttributesFile),
    export!(K, 0x0EF, NtQueryVolumeInformationFile),
    export!(K, 0x0E4, NtQueryDirectoryFile),
    export!(K, 0x0DB, NtFlushBuffersFile),
];

const GENERIC_WRITE: u32 = 0x4000_0000;
const GENERIC_ALL: u32 = 0x1000_0000;
const FILE_WRITE_DATA: u32 = 0x0002;
const FILE_APPEND_DATA: u32 = 0x0004;
const FILE_DIRECTORY_FILE: u32 = 0x0001;
const FILE_NON_DIRECTORY_FILE: u32 = 0x0040;
const FILE_SYNCHRONOUS_IO_ALERT: u32 = 0x10;
const FILE_SYNCHRONOUS_IO_NONALERT: u32 = 0x20;

mod info_class {
    pub const BASIC: u32 = 4;
    pub const STANDARD: u32 = 5;
    pub const INTERNAL: u32 = 6;
    pub const POSITION: u32 = 14;
    pub const MODE: u32 = 16;
    pub const ALIGNMENT: u32 = 17;
    pub const ALLOCATION: u32 = 19;
    pub const END_OF_FILE: u32 = 20;
    pub const NETWORK_OPEN: u32 = 34;
}

mod fs_info_class {
    pub const VOLUME: u32 = 1;
    pub const SIZE: u32 = 3;
    pub const DEVICE: u32 = 4;
    pub const ATTRIBUTE: u32 = 5;
}

/// RootDirectory pseudo-handle for the \?? (DOS devices) directory: names
/// under it are drive-letter paths like `game:\Data`, resolved as absolute.
const OB_DOS_DEVICES: u32 = 0xFFFF_FFFD;

/// Full guest path for OBJECT_ATTRIBUTES, honoring RootDirectory.
fn object_path(c: &Call, attrs: u32) -> Result<String, u32> {
    let a = read_object_attributes(c.mem(), attrs).ok_or(status::INVALID_PARAMETER)?;
    let name = a.name.unwrap_or_default();
    if a.root_directory == 0 || a.root_directory == OB_DOS_DEVICES {
        return Ok(name);
    }
    let root = handle_object(c, a.root_directory).ok_or(status::INVALID_HANDLE)?;
    let file = root.file().ok_or(status::OBJECT_TYPE_MISMATCH)?;
    Ok(format!("{}\\{}", file.guest_path, name.trim_start_matches(['\\', '/'])))
}

fn file_from_handle(c: &Call, handle: u32) -> Result<Arc<KObject>, u32> {
    let obj = handle_object(c, handle).ok_or(status::INVALID_HANDLE)?;
    if obj.file().is_none() {
        return Err(status::OBJECT_TYPE_MISMATCH);
    }
    Ok(obj)
}

fn open(c: &mut Call, handle_ptr: u32, access: u32, attrs: u32, iosb: u32, disposition: u32, options: u32) -> u64 {
    let path = match object_path(c, attrs) {
        Ok(p) => p,
        Err(s) => return s as u64,
    };
    let Some(disposition) = Disposition::from_nt(disposition) else { return status::INVALID_PARAMETER as u64 };
    let kind = if options & FILE_DIRECTORY_FILE != 0 {
        Kind::Directory
    } else if options & FILE_NON_DIRECTORY_FILE != 0 {
        Kind::NonDirectory
    } else {
        Kind::Any
    };
    let write = access & (GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA | FILE_APPEND_DATA) != 0;
    let req = OpenRequest { write, disposition, kind };

    match c.kernel.vfs.open(&path, req) {
        Ok((node, action)) => {
            let guest_path = match &node {
                Node::File(f) => f.guest_path.clone(),
                Node::Directory(d) => d.guest_path.clone(),
            };
            log::info!("open {path:?} -> {guest_path} ({action:?}{})", if write { ", write" } else { "" });
            let synchronous = options & (FILE_SYNCHRONOUS_IO_ALERT | FILE_SYNCHRONOUS_IO_NONALERT) != 0;
            let file = FileObject::new(node, guest_path, synchronous);
            let obj = match c.kernel.objects.create(
                &c.kernel.rt.space,
                ObjectType::File,
                Some(SyncState::AlwaysSignaled),
                Payload::File(file),
                None,
            ) {
                Ok(o) => o,
                Err(s) => return s as u64,
            };
            let h = c.kernel.objects.insert_handle(&obj);
            if handle_ptr != 0 {
                c.mem().write_u32(handle_ptr, h);
            }
            write_iosb(c.mem(), iosb, status::SUCCESS, action as u32);
            status::SUCCESS as u64
        }
        Err(e) => {
            let s = vfs_error_status(&e);
            log::info!("open {path:?} failed: {e} ({s:#010X})");
            write_iosb(c.mem(), iosb, s, 0);
            s as u64
        }
    }
}

/// (FileHandle*, DesiredAccess, ObjectAttributes*, IoStatusBlock*,
///  AllocationSize*, FileAttributes, ShareAccess, CreateDisposition, CreateOptions)
fn NtCreateFile(c: &mut Call) -> u64 {
    let (handle_ptr, access, attrs, iosb) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3));
    let (disposition, options) = (c.arg(7), c.arg(8));
    open(c, handle_ptr, access, attrs, iosb, disposition, options)
}

/// (FileHandle*, DesiredAccess, ObjectAttributes*, IoStatusBlock*, ShareAccess, OpenOptions)
fn NtOpenFile(c: &mut Call) -> u64 {
    let (handle_ptr, access, attrs, iosb, options) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(5));
    open(c, handle_ptr, access, attrs, iosb, 1, options)
}

/// Completion side effects shared by read/write.
fn complete_io(c: &Call, event: u32, apc_routine: u32, apc_context: u32, iosb: u32, s: u32, info: u32) {
    write_iosb(c.mem(), iosb, s, info);
    if event != 0
        && let Some(ev) = handle_object(c, event)
    {
        c.kernel.objects.set_event(c.mem(), &ev);
    }
    // Bit 0 of the routine is a flag, not part of the address (Xenia: "probably
    // means do not queue to IO ports"); as there, the APC is only queued for a
    // successful request with a context.
    let routine = apc_routine & !1;
    if routine != 0 && apc_context != 0 && s == status::SUCCESS {
        c.kernel.objects.queue_apc(c.tid(), UserApc { routine, context: apc_context, arg1: iosb, arg2: 0 });
    }
}

fn io_offset(c: &Call, file: &FileObject, offset_ptr: u32) -> u64 {
    if offset_ptr != 0 {
        let off = c.mem().read_i64(offset_ptr);
        if off != FILE_USE_FILE_POINTER_POSITION && off >= 0 {
            return off as u64;
        }
    }
    *file.position.lock().unwrap()
}

/// (FileHandle, Event, ApcRoutine, ApcContext, IoStatusBlock*, Buffer, Length, ByteOffset*)
fn NtReadFile(c: &mut Call) -> u64 {
    let (handle, event, apc, apc_ctx, iosb, buffer, length, offset_ptr) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6), c.arg(7));
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let file = obj.file().unwrap();
    let Node::File(f) = &file.node else { return status::INVALID_PARAMETER as u64 };
    let offset = io_offset(c, file, offset_ptr);
    if buffer as u64 + length as u64 > runtime_core::memory::GUEST_SPACE_SIZE {
        return status::ACCESS_VIOLATION as u64;
    }
    // The save menu reads only the start of each save; loading one reads on from there.
    if offset > 0
        && let Node::File(f) = &file.node
        && f.host_path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("Savegame.dat"))
        && !file.save_load_logged.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        let package = f.host_path.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned());
        log::info!("content: loading save {}", package.unwrap_or_default());
    }
    // The read writes guest memory directly: the renderer must see it as written.
    runtime_core::dirty::mark(buffer, length);
    // Straight into the guest buffer, one read per piece that is contiguous in host memory
    // (normally the whole buffer).
    let mut result: std::io::Result<u64> = Ok(0);
    c.mem().runs(buffer, length, |off, ptr, len| {
        let Ok(done) = result else { return };
        if done < off as u64 {
            return; // an earlier piece came up short: end of file
        }
        // SAFETY: the destination is the guest buffer the title passed for this read,
        // bounds-checked against the 4 GiB space above; `runs` yields host-contiguous pieces.
        let dest = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        result = f.read_at(dest, offset + off as u64).map(|n| done + n as u64);
    });
    let (s, n) = match result {
        Ok(0) if length > 0 => (status::END_OF_FILE, 0),
        Ok(n) => (status::SUCCESS, n as u32),
        Err(e) => {
            log::warn!("NtReadFile({}): {e}", file.guest_path);
            (status::UNSUCCESSFUL, 0)
        }
    };
    *file.position.lock().unwrap() = offset + n as u64;
    complete_io(c, event, apc, apc_ctx, iosb, s, n);
    // Asynchronous handles report the request as pending even though it is
    // complete; the title collects the result from the IO_STATUS_BLOCK or the
    // event (as on hardware and in Xenia). End of file is reported directly.
    if !file.synchronous && s != status::END_OF_FILE { status::PENDING as u64 } else { s as u64 }
}

/// (FileHandle, Event, ApcRoutine, ApcContext, IoStatusBlock*, Buffer, Length, ByteOffset*)
fn NtWriteFile(c: &mut Call) -> u64 {
    let (handle, event, apc, apc_ctx, iosb, buffer, length, offset_ptr) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6), c.arg(7));
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let file = obj.file().unwrap();
    let Node::File(f) = &file.node else { return status::INVALID_PARAMETER as u64 };
    let offset = io_offset(c, file, offset_ptr);
    let data = c.mem().read_vec(buffer, length);
    let (s, n) = match f.write_at(&data, offset) {
        Ok(n) => (status::SUCCESS, n as u32),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => (status::ACCESS_DENIED, 0),
        Err(e) => {
            log::warn!("NtWriteFile({}): {e}", file.guest_path);
            (status::UNSUCCESSFUL, 0)
        }
    };
    *file.position.lock().unwrap() = offset + n as u64;
    complete_io(c, event, apc, apc_ctx, iosb, s, n);
    if !file.synchronous { status::PENDING as u64 } else { s as u64 }
}

/// (FileHandle, IoStatusBlock*, FileInformation, Length, FileInformationClass)
fn NtQueryInformationFile(c: &mut Call) -> u64 {
    let (handle, iosb, info, length, class) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let file = obj.file().unwrap();
    let meta = match file.metadata() {
        Ok(m) => m,
        Err(e) => {
            log::warn!("NtQueryInformationFile({}): {e}", file.guest_path);
            return status::UNSUCCESSFUL as u64;
        }
    };
    let mem = c.mem();
    let size = match class {
        info_class::BASIC => 0x28,
        info_class::STANDARD => 0x18,
        info_class::INTERNAL | info_class::POSITION => 8,
        info_class::MODE | info_class::ALIGNMENT => 4,
        info_class::NETWORK_OPEN => NETWORK_OPEN_INFO_SIZE,
        _ => {
            log::warn!("NtQueryInformationFile({}): unsupported class {class}", file.guest_path);
            write_iosb(mem, iosb, status::INVALID_INFO_CLASS, 0);
            return status::INVALID_INFO_CLASS as u64;
        }
    };
    if length < size {
        write_iosb(mem, iosb, status::INFO_LENGTH_MISMATCH, 0);
        return status::INFO_LENGTH_MISMATCH as u64;
    }
    match class {
        info_class::BASIC => write_basic_info(mem, info, &meta),
        info_class::STANDARD => write_standard_info(mem, info, &meta),
        info_class::INTERNAL => mem.write_u64(info, obj.id),
        info_class::POSITION => mem.write_u64(info, *file.position.lock().unwrap()),
        info_class::MODE => mem.write_u32(info, 0),
        info_class::ALIGNMENT => mem.write_u32(info, 0),
        _ => write_network_open_info(mem, info, &meta),
    }
    write_iosb(mem, iosb, status::SUCCESS, size);
    status::SUCCESS as u64
}

/// (FileHandle, IoStatusBlock*, FileInformation, Length, FileInformationClass)
fn NtSetInformationFile(c: &mut Call) -> u64 {
    let (handle, iosb, info, length, class) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let file = obj.file().unwrap();
    let mem = c.mem();
    let s = match class {
        info_class::POSITION if length >= 8 => {
            *file.position.lock().unwrap() = mem.read_u64(info);
            status::SUCCESS
        }
        info_class::END_OF_FILE if length >= 8 => match &file.node {
            Node::File(f) => match f.set_len(mem.read_u64(info)) {
                Ok(()) => status::SUCCESS,
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => status::ACCESS_DENIED,
                Err(_) => status::UNSUCCESSFUL,
            },
            Node::Directory(_) => status::INVALID_PARAMETER,
        },
        // Preallocation is a hint; the host filesystem allocates on write.
        info_class::ALLOCATION if length >= 8 => status::SUCCESS,
        info_class::POSITION | info_class::END_OF_FILE | info_class::ALLOCATION => status::INFO_LENGTH_MISMATCH,
        _ => {
            log::warn!("NtSetInformationFile({}): unsupported class {class}", file.guest_path);
            status::INVALID_INFO_CLASS
        }
    };
    write_iosb(mem, iosb, s, 0);
    s as u64
}

/// (ObjectAttributes*, FILE_NETWORK_OPEN_INFORMATION*)
fn NtQueryFullAttributesFile(c: &mut Call) -> u64 {
    let (attrs, out) = (c.arg(0), c.arg(1));
    let path = match object_path(c, attrs) {
        Ok(p) => p,
        Err(s) => return s as u64,
    };
    match c.kernel.vfs.metadata(&path) {
        Ok(m) => {
            write_network_open_info(c.mem(), out, &m);
            status::SUCCESS as u64
        }
        Err(e) => {
            log::info!("attributes {path:?}: {e}");
            vfs_error_status(&e) as u64
        }
    }
}

/// (FileHandle, IoStatusBlock*, FsInformation, Length, FsInformationClass)
fn NtQueryVolumeInformationFile(c: &mut Call) -> u64 {
    let (handle, iosb, info, length, class) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let mem = c.mem();
    let read_only = match &obj.file().unwrap().node {
        Node::File(f) => f.read_only,
        Node::Directory(d) => d.read_only,
    };
    let (s, written) = match class {
        fs_info_class::VOLUME => {
            let label = b"SKYRIM";
            let size = 0x12 + label.len() as u32;
            if length < size {
                (status::BUFFER_OVERFLOW, 0)
            } else {
                mem.write_u64(info, runtime_core::timebase::system_time());
                mem.write_u32(info + 8, 0x5348_4952);
                mem.write_u32(info + 12, label.len() as u32);
                mem.write_u8(info + 16, 0);
                mem.write_bytes(info + 17, label);
                (status::SUCCESS, size)
            }
        }
        fs_info_class::SIZE if length >= 24 => {
            // 8 GiB volume, 64 KiB clusters (128 x 512-byte sectors).
            mem.write_u64(info, 0x2_0000);
            mem.write_u64(info + 8, if read_only { 0 } else { 0x1_0000 });
            mem.write_u32(info + 16, 128);
            mem.write_u32(info + 20, 512);
            (status::SUCCESS, 24)
        }
        fs_info_class::DEVICE if length >= 8 => {
            const FILE_DEVICE_CD_ROM: u32 = 0x02;
            const FILE_DEVICE_DISK: u32 = 0x07;
            const FILE_READ_ONLY_DEVICE: u32 = 0x02;
            mem.write_u32(info, if read_only { FILE_DEVICE_CD_ROM } else { FILE_DEVICE_DISK });
            mem.write_u32(info + 4, if read_only { FILE_READ_ONLY_DEVICE } else { 0 });
            (status::SUCCESS, 8)
        }
        fs_info_class::ATTRIBUTE => {
            let name: &[u8] = if read_only { b"GDFX" } else { b"FATX" };
            let size = 12 + name.len() as u32;
            if length < size {
                (status::BUFFER_OVERFLOW, 0)
            } else {
                mem.write_u32(info, 0);
                mem.write_i32(info + 4, 255);
                mem.write_u32(info + 8, name.len() as u32);
                mem.write_bytes(info + 12, name);
                (status::SUCCESS, size)
            }
        }
        fs_info_class::SIZE | fs_info_class::DEVICE => (status::INFO_LENGTH_MISMATCH, 0),
        _ => {
            log::warn!("NtQueryVolumeInformationFile: unsupported class {class}");
            (status::INVALID_INFO_CLASS, 0)
        }
    };
    write_iosb(mem, iosb, s, written);
    s as u64
}

/// (FileHandle, Event, ApcRoutine, ApcContext, IoStatusBlock*, FileInformation,
///  Length, FileName (ANSI_STRING* pattern), RestartScan)
fn NtQueryDirectoryFile(c: &mut Call) -> u64 {
    let (handle, event, apc, apc_ctx, iosb, info, length, pattern_ptr, restart) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6), c.arg(7), c.arg(8) != 0);
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let file = obj.file().unwrap();
    let Node::Directory(dir) = &file.node else { return status::INVALID_PARAMETER as u64 };
    let mem = c.mem();

    let mut scan = file.scan.lock().unwrap();
    let first = scan.is_none();
    if first || restart {
        let pattern = if pattern_ptr != 0 { read_ansi_lossy(mem, pattern_ptr) } else { String::new() };
        let pattern = if pattern.is_empty() { "*".to_string() } else { pattern };
        let entries = match dir.entries() {
            Ok(e) => e.into_iter().filter(|e| matches_pattern(&e.name, &pattern)).collect(),
            Err(e) => {
                log::warn!("NtQueryDirectoryFile({}): {e}", file.guest_path);
                return status::UNSUCCESSFUL as u64;
            }
        };
        *scan = Some(DirectoryScan { entries, next: 0 });
    }
    let scan = scan.as_mut().unwrap();

    if scan.next >= scan.entries.len() {
        let s = if first && scan.entries.is_empty() { status::NO_SUCH_FILE } else { status::NO_MORE_FILES };
        complete_io(c, event, apc, apc_ctx, iosb, s, 0);
        return s as u64;
    }
    // Pack as many entries as fit, each 8-byte aligned and chained.
    let mut offset = 0u32;
    let mut prev: Option<u32> = None;
    while scan.next < scan.entries.len() {
        let e = &scan.entries[scan.next];
        let size = directory_info_size(e);
        let start = (offset + 7) & !7;
        if start + size > length {
            break;
        }
        if let Some(p) = prev {
            mem.write_u32(info + p, start - p);
        }
        write_directory_info(mem, info + start, scan.next as u32, e);
        prev = Some(start);
        offset = start + size;
        scan.next += 1;
    }
    let s = if prev.is_none() { status::BUFFER_OVERFLOW } else { status::SUCCESS };
    complete_io(c, event, apc, apc_ctx, iosb, s, offset);
    s as u64
}

/// (FileHandle, IoStatusBlock*)
fn NtFlushBuffersFile(c: &mut Call) -> u64 {
    let (handle, iosb) = (c.arg(0), c.arg(1));
    let obj = match file_from_handle(c, handle) {
        Ok(o) => o,
        Err(s) => return s as u64,
    };
    let s = match &obj.file().unwrap().node {
        Node::File(f) => {
            if f.flush().is_ok() {
                status::SUCCESS
            } else {
                status::UNSUCCESSFUL
            }
        }
        Node::Directory(_) => status::SUCCESS,
    };
    write_iosb(c.mem(), iosb, s, 0);
    s as u64
}

/// (u32, u32): file-system cache sizing; the host OS does the caching.
fn FscSetCacheElementCount(_c: &mut Call) -> u64 {
    status::SUCCESS as u64
}
