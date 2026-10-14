//! File objects and the FILE_*_INFORMATION encodings shared by the Nt*File exports.

use std::sync::Mutex;

use runtime_core::memory::GuestMemory;
use runtime_core::timebase::filetime_from_system_time;
use vfs::{DirEntry, Metadata, Node, VfsError};

use crate::status;

pub mod attributes {
    pub const READONLY: u32 = 0x01;
    pub const DIRECTORY: u32 = 0x10;
    pub const NORMAL: u32 = 0x80;
}

/// Position value meaning "use the file pointer" in NtReadFile/NtWriteFile.
pub const FILE_USE_FILE_POINTER_POSITION: i64 = -2;

pub struct DirectoryScan {
    pub entries: Vec<DirEntry>,
    pub next: usize,
}

pub struct FileObject {
    pub node: Node,
    pub guest_path: String,
    pub position: Mutex<u64>,
    pub scan: Mutex<Option<DirectoryScan>>,
    /// Opened with FILE_SYNCHRONOUS_IO_*: reads and writes report their
    /// result directly instead of STATUS_PENDING.
    pub synchronous: bool,
    /// "content: loading save" was logged for this open (see NtReadFile).
    pub save_load_logged: std::sync::atomic::AtomicBool,
}

impl FileObject {
    pub fn new(node: Node, guest_path: String, synchronous: bool) -> Self {
        FileObject {
            node,
            guest_path,
            position: Mutex::new(0),
            scan: Mutex::new(None),
            synchronous,
            save_load_logged: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn metadata(&self) -> std::io::Result<Metadata> {
        match &self.node {
            Node::File(f) => f.metadata(),
            Node::Directory(d) => d.metadata(),
        }
    }
}

pub fn vfs_error_status(e: &VfsError) -> u32 {
    match e {
        VfsError::NoSuchDevice(_) | VfsError::PathNotFound => status::OBJECT_PATH_NOT_FOUND,
        VfsError::NotFound => status::OBJECT_NAME_NOT_FOUND,
        VfsError::AlreadyExists => status::OBJECT_NAME_COLLISION,
        VfsError::AccessDenied => status::ACCESS_DENIED,
        VfsError::NotADirectory => status::NOT_A_DIRECTORY,
        VfsError::IsADirectory => status::FILE_IS_A_DIRECTORY,
        VfsError::Io(_) => status::UNSUCCESSFUL,
    }
}

pub fn attributes_of(m: &Metadata) -> u32 {
    let mut a = if m.is_dir { attributes::DIRECTORY } else { 0 };
    if m.read_only {
        a |= attributes::READONLY;
    }
    if a == 0 { attributes::NORMAL } else { a }
}

pub fn allocation_size(m: &Metadata) -> u64 {
    m.size.div_ceil(0x1000) * 0x1000
}

pub struct Times {
    pub creation: u64,
    pub access: u64,
    pub write: u64,
    pub change: u64,
}

pub fn times_of(m: &Metadata) -> Times {
    let write = filetime_from_system_time(m.modified);
    Times {
        creation: filetime_from_system_time(m.created),
        access: filetime_from_system_time(m.accessed),
        write,
        change: write,
    }
}

/// FILE_NETWORK_OPEN_INFORMATION (0x38 bytes).
pub const NETWORK_OPEN_INFO_SIZE: u32 = 0x38;

pub fn write_network_open_info(mem: &GuestMemory, p: u32, m: &Metadata) {
    let t = times_of(m);
    mem.write_u64(p, t.creation);
    mem.write_u64(p + 0x08, t.access);
    mem.write_u64(p + 0x10, t.write);
    mem.write_u64(p + 0x18, t.change);
    mem.write_u64(p + 0x20, allocation_size(m));
    mem.write_u64(p + 0x28, m.size);
    mem.write_u32(p + 0x30, attributes_of(m));
    mem.write_u32(p + 0x34, 0);
}

/// FILE_BASIC_INFORMATION (0x28 bytes).
pub fn write_basic_info(mem: &GuestMemory, p: u32, m: &Metadata) {
    let t = times_of(m);
    mem.write_u64(p, t.creation);
    mem.write_u64(p + 0x08, t.access);
    mem.write_u64(p + 0x10, t.write);
    mem.write_u64(p + 0x18, t.change);
    mem.write_u32(p + 0x20, attributes_of(m));
    mem.write_u32(p + 0x24, 0);
}

/// FILE_STANDARD_INFORMATION (0x18 bytes).
pub fn write_standard_info(mem: &GuestMemory, p: u32, m: &Metadata) {
    mem.write_u64(p, allocation_size(m));
    mem.write_u64(p + 0x08, m.size);
    mem.write_u32(p + 0x10, 1);
    mem.write_u8(p + 0x14, 0);
    mem.write_u8(p + 0x15, m.is_dir as u8);
    mem.write_u16(p + 0x16, 0);
}

/// X_FILE_DIRECTORY_INFORMATION header size; the ANSI name follows.
pub const DIRECTORY_INFO_HEADER: u32 = 0x40;

/// Writes one directory entry at `p` and returns its unpadded size.
pub fn write_directory_info(mem: &GuestMemory, p: u32, index: u32, e: &DirEntry) -> u32 {
    let t = times_of(&e.metadata);
    let name: Vec<u8> = e.name.bytes().map(|b| if b.is_ascii() { b } else { b'?' }).collect();
    mem.write_u32(p, 0);
    mem.write_u32(p + 0x04, index);
    mem.write_u64(p + 0x08, t.creation);
    mem.write_u64(p + 0x10, t.access);
    mem.write_u64(p + 0x18, t.write);
    mem.write_u64(p + 0x20, t.change);
    mem.write_u64(p + 0x28, e.metadata.size);
    mem.write_u64(p + 0x30, allocation_size(&e.metadata));
    mem.write_u32(p + 0x38, attributes_of(&e.metadata));
    mem.write_u32(p + 0x3C, name.len() as u32);
    mem.write_bytes(p + DIRECTORY_INFO_HEADER, &name);
    DIRECTORY_INFO_HEADER + name.len() as u32
}

pub fn directory_info_size(e: &DirEntry) -> u32 {
    DIRECTORY_INFO_HEADER + e.name.len() as u32
}

/// IO_STATUS_BLOCK { u32 Status; u32 Information; }
pub fn write_iosb(mem: &GuestMemory, iosb: u32, status: u32, information: u32) {
    if iosb != 0 {
        mem.write_u32(iosb, status);
        mem.write_u32(iosb + 4, information);
    }
}
