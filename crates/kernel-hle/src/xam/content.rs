//! Title content (save games and the like) and XAM enumerators.
//!
//! Content packages are plain host directories under the content root:
//! `<root>/<XUID:016X>/<TitleID:08X>/<ContentType:08X>/<FileName>/`, with the
//! display name beside it in `<FileName>.title` (UTF-8). XamContentCreate
//! mounts a package as its own read-write device and links the title's root
//! name (`save:`) to it. Behavior follows Xenia's xam_content.cc / xam_enum.cc.
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::Kernel;
use crate::dispatch::{Call, Export, ModuleId};
use crate::input::MAX_USERS;
use crate::objects::{ObjectType, Payload};
use crate::status::{hresult, win32};

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x250, XamEnumerate),
    export!(X, 0x258, XamContentCreate),
    export!(X, 0x259, XamContentCreateEx),
    export!(X, 0x25A, XamContentClose),
    export!(X, 0x25B, XamContentDelete),
    export!(X, 0x25C, XamContentCreateEnumerator),
    export!(X, 0x25E, XamContentGetDeviceData),
    export!(X, 0x25F, XamContentGetDeviceName),
    export!(X, 0x265, XamContentGetDeviceState),
    export!(X, 0x266, XamContentGetLicenseMask),
    export!(X, 0x267, XamContentFlush),
    export!(X, 0x25D, XamContentCreateDeviceEnumerator),
    export!(X, 0x269, XamContentOpenFile),
    export!(X, 0x270, XamContentDeleteInternal),
    export!(X, 0x271, XamContentCreateInternal),
];

/// Storage device ids (Xenia's dummy devices).
pub const DEVICE_HDD: u32 = 1;
pub const DEVICE_ODD: u32 = 2;

/// sizeof(XCONTENT_DATA).
pub const CONTENT_DATA_SIZE: u32 = 0x134;
/// XCONTENT_DATA_INTERNAL (XCONTENT_DATA plus ownership fields): title id offset.
const INTERNAL_TITLE_ID: u32 = 0x140;
/// Title id meaning "the running title" in XCONTENT_DATA_INTERNAL.
const CURRENT_TITLE: u32 = 0xFFFF_FFFF;
/// The user index of the *Internal content calls: no user (shared content).
const USER_INDEX_NONE: u32 = 0xFE;
/// sizeof(XCONTENT_DEVICE_DATA).
const DEVICE_DATA_SIZE: u32 = 0x50;
const DISPLAY_NAME_CHARS: usize = 128;
const FILE_NAME_LEN: usize = 42;

/// XContentFlag: creation dispositions (low nibble) and enumeration flags.
mod flag {
    pub const CREATE_NEW: u32 = 1;
    pub const CREATE_ALWAYS: u32 = 2;
    pub const OPEN_EXISTING: u32 = 3;
    pub const OPEN_ALWAYS: u32 = 4;
    pub const TRUNCATE_EXISTING: u32 = 5;
    pub const EXCLUDE_COMMON: u32 = 0x1000;
    pub const EXCLUDE_READ_ONLY_DEVICES: u32 = 0x8000_0000;
}

/// XamContentCreate dispositions.
const DISPOSITION_CREATE: u32 = 1;
const DISPOSITION_OPEN: u32 = 2;

/// ERROR_INVALID_NAME.
const ERROR_INVALID_NAME: u32 = 123;

/// One package as XCONTENT_DATA describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentData {
    pub device_id: u32,
    pub content_type: u32,
    pub display_name: Vec<u16>,
    pub file_name: String,
}

impl ContentData {
    fn read(c: &Call, at: u32) -> Self {
        let mem = c.mem();
        let mut display_name = mem.read_utf16(at + 8, DISPLAY_NAME_CHARS as u32);
        if let Some(end) = display_name.iter().position(|&u| u == 0) {
            display_name.truncate(end);
        }
        let raw = mem.read_vec(at + 0x108, FILE_NAME_LEN as u32);
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        ContentData {
            device_id: mem.read_u32(at),
            content_type: mem.read_u32(at + 4),
            display_name,
            file_name: String::from_utf8_lossy(&raw[..end]).trim().to_string(),
        }
    }

    /// The XCONTENT_DATA bytes (big-endian, as the guest reads them).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; CONTENT_DATA_SIZE as usize];
        b[0..4].copy_from_slice(&self.device_id.to_be_bytes());
        b[4..8].copy_from_slice(&self.content_type.to_be_bytes());
        for (i, u) in self.display_name.iter().take(DISPLAY_NAME_CHARS - 1).enumerate() {
            b[8 + i * 2..10 + i * 2].copy_from_slice(&u.to_be_bytes());
        }
        let name = self.file_name.as_bytes();
        let n = name.len().min(FILE_NAME_LEN);
        b[0x108..0x108 + n].copy_from_slice(&name[..n]);
        b
    }
}

/// A file name is usable as a single host path component.
fn valid_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

/// Fixed-size items handed out by XamEnumerate.
pub struct Enumerator {
    item_size: u32,
    per_call: u32,
    items: Vec<Vec<u8>>,
    cursor: Mutex<usize>,
}

impl Enumerator {
    pub fn new(item_size: u32, per_call: u32, items: Vec<Vec<u8>>) -> Self {
        Enumerator { item_size, per_call: per_call.max(1), items, cursor: Mutex::new(0) }
    }

    /// Copies the next batch into `buffer`; returns the item count, or a
    /// Win32 error (NO_MORE_FILES once exhausted).
    fn write_items(&self, c: &Call, buffer: u32, buffer_len: u32) -> Result<u32, u32> {
        let mut cursor = self.cursor.lock().unwrap();
        let remaining = self.items.len() - *cursor;
        let count = remaining.min(self.per_call as usize);
        if count == 0 {
            return Err(win32::NO_MORE_FILES);
        }
        if (buffer_len / self.item_size.max(1)) < count as u32 {
            return Err(win32::INSUFFICIENT_BUFFER);
        }
        for (i, item) in self.items[*cursor..*cursor + count].iter().enumerate() {
            c.mem().write_bytes(buffer + i as u32 * self.item_size, item);
        }
        *cursor += count;
        Ok(count as u32)
    }
}

struct OpenPackage {
    device: String,
}

#[derive(Default)]
pub struct ContentManager {
    root: Mutex<Option<PathBuf>>,
    /// Lower-cased root name (`save`) -> mounted package.
    open: Mutex<HashMap<String, OpenPackage>>,
    next_device: AtomicU32,
}

impl ContentManager {
    /// Host directory that holds every user's content.
    pub fn set_root(&self, root: impl Into<PathBuf>) {
        *self.root.lock().unwrap() = Some(root.into());
    }

    pub fn root(&self) -> Option<PathBuf> {
        self.root.lock().unwrap().clone()
    }
}

fn root_key(root_name: &str) -> String {
    root_name.trim_end_matches(':').to_ascii_lowercase()
}

impl Kernel {
    pub(crate) fn title_id(&self) -> u32 {
        self.rt.xex.execution_info().map_or(0, |i| i.title_id)
    }

    fn content_type_dir(&self, xuid: u64, content_type: u32) -> Option<PathBuf> {
        self.content_dir_for(xuid, self.title_id(), content_type)
    }

    fn content_dir_for(&self, xuid: u64, title: u32, content_type: u32) -> Option<PathBuf> {
        let root = self.content.root()?;
        Some(root.join(format!("{xuid:016X}")).join(format!("{title:08X}")).join(format!("{content_type:08X}")))
    }

    fn package_path_for(&self, xuid: u64, title: u32, data: &ContentData) -> Option<PathBuf> {
        Some(self.content_dir_for(xuid, title, data.content_type)?.join(&data.file_name))
    }

    /// Packages of one type stored for `xuid` on the hard drive.
    pub fn list_content(&self, xuid: u64, content_type: u32) -> Vec<ContentData> {
        let Some(dir) = self.content_type_dir(xuid, content_type) else { return Vec::new() };
        let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
        let mut out: Vec<ContentData> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .filter(|n| valid_file_name(n) && n.len() <= FILE_NAME_LEN)
            .map(|file_name| {
                let title = std::fs::read_to_string(dir.join(format!("{file_name}.title"))).unwrap_or_default();
                let display = if title.is_empty() { file_name.clone() } else { title };
                ContentData {
                    device_id: DEVICE_HDD,
                    content_type,
                    display_name: display.trim_end_matches('\n').encode_utf16().collect(),
                    file_name,
                }
            })
            .collect();
        out.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        out
    }

    fn mount_package(&self, root_name: &str, path: &Path) {
        let n = self.content.next_device.fetch_add(1, Ordering::Relaxed);
        let device = format!("\\Device\\Content_{n}");
        self.vfs.mount(&device, path, false);
        self.vfs.add_symlink(&format!("{}:", root_name.trim_end_matches(':')), &device);
        self.content.open.lock().unwrap().insert(root_key(root_name), OpenPackage { device });
    }

    /// XamContentClose: unlinks and unmounts. Win32 result.
    pub fn close_content(&self, root_name: &str) -> u32 {
        match self.content.open.lock().unwrap().remove(&root_key(root_name)) {
            Some(p) => {
                self.vfs.remove_symlink(root_name);
                self.vfs.unmount(&p.device);
                win32::SUCCESS
            }
            None => win32::FILE_NOT_FOUND,
        }
    }
}

/// Finishes a call that may complete through an XOVERLAPPED.
fn finish(c: &Call, overlapped: u32, result: u32, length: u32) -> u64 {
    if overlapped == 0 {
        return result as u64;
    }
    let overlapped_result = if result == 0 { 0 } else { win32::FUNCTION_FAILED };
    c.kernel.complete_overlapped_ex(overlapped, overlapped_result, hresult::from_win32(result), length);
    win32::IO_PENDING as u64
}

/// The XUID content is filed under for `user` (0 = shared by all users).
fn user_xuid(c: &Call, user: u32) -> Result<u64, u32> {
    if user as usize >= MAX_USERS {
        return Ok(0);
    }
    c.kernel.users.get(user).map(|p| p.xuid).ok_or(win32::NO_SUCH_USER)
}

/// (u32 UserIndex, u32 DeviceId, u32 ContentType, u32 Flags, u32 ItemsPerCall,
///  u32* BufferSize, HANDLE* Enumerator)
fn XamContentCreateEnumerator(c: &mut Call) -> u64 {
    let (user, device, content_type, flags, per_call, size_ptr, handle_ptr) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6));
    let xuid = match user_xuid(c, user) {
        Ok(x) => x,
        Err(e) => return e as u64,
    };
    if (device != 0 && device != DEVICE_HDD && device != DEVICE_ODD) || handle_ptr == 0 {
        if size_ptr != 0 {
            c.mem().write_u32(size_ptr, 0);
        }
        return hresult::E_INVALIDARG as u64;
    }
    if size_ptr != 0 {
        c.mem().write_u32(size_ptr, CONTENT_DATA_SIZE * per_call);
    }
    let mut items = Vec::new();
    if device != DEVICE_ODD {
        if xuid != 0 {
            items.extend(c.kernel.list_content(xuid, content_type));
        }
        if xuid == 0 || flags & flag::EXCLUDE_COMMON == 0 {
            items.extend(c.kernel.list_content(0, content_type));
        }
        items.dedup();
    }
    log::info!("XamContentCreateEnumerator: type {content_type:#X}, {} item(s)", items.len());
    let e = Enumerator::new(CONTENT_DATA_SIZE, per_call, items.iter().map(ContentData::to_bytes).collect());
    match c.kernel.objects.create(&c.kernel.rt.space, ObjectType::Other(0), None, Payload::Enumerator(e), None) {
        Ok(obj) => {
            let h = c.kernel.objects.insert_handle(&obj);
            c.mem().write_u32(handle_ptr, h);
            win32::SUCCESS as u64
        }
        Err(s) => {
            log::error!("XamContentCreateEnumerator: {s:#010X}");
            win32::NOT_ENOUGH_MEMORY as u64
        }
    }
}

/// (HANDLE Enumerator, u32 Flags, void* Buffer, u32 BufferLength,
///  u32* ItemsReturned, XOVERLAPPED* Overlapped)
fn XamEnumerate(c: &mut Call) -> u64 {
    let (handle, buffer, buffer_len, returned_ptr, overlapped) = (c.arg(0), c.arg(2), c.arg(3), c.arg(4), c.arg(5));
    let Some(obj) = c.kernel.objects.lookup(handle, None) else { return win32::INVALID_HANDLE as u64 };
    let Payload::Enumerator(e) = &obj.payload else { return win32::INVALID_HANDLE as u64 };
    let (result, count) = if buffer == 0 {
        (win32::INVALID_PARAMETER, 0)
    } else {
        match e.write_items(c, buffer, buffer_len) {
            Ok(n) => (win32::SUCCESS, n),
            Err(err) => (err, 0),
        }
    };
    if overlapped == 0 && returned_ptr != 0 {
        c.mem().write_u32(returned_ptr, count);
    }
    finish(c, overlapped, result, count)
}

/// (u32* Mask, XOVERLAPPED* Overlapped). A disc title has no content
/// licenses.
fn XamContentGetLicenseMask(c: &mut Call) -> u64 {
    let (mask_ptr, overlapped) = (c.arg(0), c.arg(1));
    if mask_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    c.mem().write_u32(mask_ptr, 0);
    finish(c, overlapped, win32::SUCCESS, 0)
}

#[allow(clippy::too_many_arguments)]
fn content_create(
    c: &Call,
    title: u32,
    user: u32,
    root_name_ptr: u32,
    data_ptr: u32,
    flags: u32,
    disposition_ptr: u32,
    license_ptr: u32,
    overlapped: u32,
) -> u64 {
    let xuid = match user_xuid(c, user) {
        Ok(x) => x,
        Err(e) => return e as u64,
    };
    let root_name = if root_name_ptr == 0 {
        String::new()
    } else {
        String::from_utf8_lossy(&c.mem().read_cstr(root_name_ptr, 64)).into_owned()
    };
    if root_name.is_empty() {
        return ERROR_INVALID_NAME as u64;
    }
    if c.kernel.vfs.symlink_target(&root_name).is_some() {
        return win32::INVALID_PARAMETER as u64;
    }
    let data = ContentData::read(c, data_ptr);
    if !valid_file_name(&data.file_name) {
        return ERROR_INVALID_NAME as u64;
    }
    if overlapped != 0 && disposition_ptr != 0 {
        c.mem().write_u32(disposition_ptr, 0);
    }
    let Some(path) = c.kernel.package_path_for(xuid, title, &data) else {
        log::warn!("XamContentCreate({root_name}): no content root configured");
        return finish(c, overlapped, win32::ACCESS_DENIED, 0);
    };
    let exists = path.is_dir();
    let remove = |p: &Path| {
        let _ = std::fs::remove_dir_all(p);
    };
    let disposition = match flags & 0xF {
        flag::CREATE_NEW if exists => Err(win32::ALREADY_EXISTS),
        flag::CREATE_NEW => Ok(DISPOSITION_CREATE),
        flag::CREATE_ALWAYS => {
            if exists {
                remove(&path);
            }
            Ok(DISPOSITION_CREATE)
        }
        flag::OPEN_EXISTING if !exists => Err(win32::PATH_NOT_FOUND),
        flag::OPEN_EXISTING => Ok(DISPOSITION_OPEN),
        flag::OPEN_ALWAYS => Ok(if exists { DISPOSITION_OPEN } else { DISPOSITION_CREATE }),
        flag::TRUNCATE_EXISTING if !exists => Err(win32::PATH_NOT_FOUND),
        flag::TRUNCATE_EXISTING => {
            remove(&path);
            Ok(DISPOSITION_CREATE)
        }
        other => {
            log::warn!("XamContentCreate({root_name}): unknown disposition {other}");
            Err(win32::INVALID_PARAMETER)
        }
    };
    let (result, disposition) = match disposition {
        Ok(d) => {
            let made = if d == DISPOSITION_CREATE {
                std::fs::create_dir_all(&path).and_then(|_| {
                    let title = String::from_utf16_lossy(&data.display_name);
                    std::fs::write(path.with_file_name(format!("{}.title", data.file_name)), title)
                })
            } else {
                Ok(())
            };
            match made {
                Ok(()) => {
                    c.kernel.mount_package(&root_name, &path);
                    log::info!(
                        "content {root_name}: -> {} ({})",
                        path.display(),
                        if d == DISPOSITION_CREATE { "created" } else { "opened" }
                    );
                    (win32::SUCCESS, d)
                }
                Err(e) => {
                    log::error!("XamContentCreate({root_name}): {}: {e}", path.display());
                    (win32::ACCESS_DENIED, 0)
                }
            }
        }
        Err(e) => (e, 0),
    };
    if license_ptr != 0 && result == win32::SUCCESS {
        c.mem().write_u32(license_ptr, 0);
    }
    if disposition_ptr != 0 {
        c.mem().write_u32(disposition_ptr, disposition);
    }
    finish(c, overlapped, result, disposition)
}

/// (u32 UserIndex, char* RootName, XCONTENT_DATA* Data, u32 Flags,
///  u32* Disposition, u32* LicenseMask, XOVERLAPPED* Overlapped)
fn XamContentCreate(c: &mut Call) -> u64 {
    content_create(c, c.kernel.title_id(), c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6))
}

/// As XamContentCreate, plus (u32 CacheSize, u64 ContentSize) before the
/// XOVERLAPPED (the ninth argument, on the stack).
fn XamContentCreateEx(c: &mut Call) -> u64 {
    content_create(c, c.kernel.title_id(), c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(8))
}

/// The title an XCONTENT_DATA_INTERNAL names (the running title for 0 / -1).
fn internal_title(c: &Call, data_ptr: u32) -> u32 {
    match c.mem().read_u32(data_ptr + INTERNAL_TITLE_ID) {
        0 | CURRENT_TITLE => c.kernel.title_id(),
        t => t,
    }
}

/// (char* RootName, XCONTENT_DATA_INTERNAL* Data, u32 Flags, u32* Disposition,
///  u32* LicenseMask, u32 CacheSize, u64 ContentSize, XOVERLAPPED* Overlapped):
/// XamContentCreate for no particular user (shared content), in the title the
/// data names (Xenia's xeXamContentCreate with XUserIndexNone).
fn XamContentCreateInternal(c: &mut Call) -> u64 {
    let data_ptr = c.arg(1);
    if data_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    let title = internal_title(c, data_ptr);
    content_create(c, title, USER_INDEX_NONE, c.arg(0), data_ptr, c.arg(2), c.arg(3), c.arg(4), c.arg(7))
}

/// (char* RootName, XOVERLAPPED* Overlapped)
fn XamContentClose(c: &mut Call) -> u64 {
    let (root_ptr, overlapped) = (c.arg(0), c.arg(1));
    let root_name = String::from_utf8_lossy(&c.mem().read_cstr(root_ptr, 64)).into_owned();
    let result = c.kernel.close_content(&root_name);
    if overlapped != 0 {
        c.kernel.complete_overlapped(overlapped, result);
        return win32::IO_PENDING as u64;
    }
    result as u64
}

/// (char* RootName, XOVERLAPPED* Overlapped): writes go straight to the host.
fn XamContentFlush(c: &mut Call) -> u64 {
    let (root_ptr, overlapped) = (c.arg(0), c.arg(1));
    let root_name = String::from_utf8_lossy(&c.mem().read_cstr(root_ptr, 64)).into_owned();
    if !c.kernel.content.open.lock().unwrap().contains_key(&root_key(&root_name)) {
        return crate::status::INVALID_PARAMETER as u64;
    }
    if overlapped != 0 {
        c.kernel.complete_overlapped(overlapped, win32::SUCCESS);
        return win32::IO_PENDING as u64;
    }
    win32::SUCCESS as u64
}

/// (u32 UserIndex, XCONTENT_DATA* Data, XOVERLAPPED* Overlapped)
fn XamContentDelete(c: &mut Call) -> u64 {
    let (user, data_ptr, overlapped) = (c.arg(0), c.arg(1), c.arg(2));
    content_delete(c, c.kernel.title_id(), user, data_ptr, overlapped)
}

/// (XCONTENT_DATA_INTERNAL* Data, XOVERLAPPED* Overlapped): XamContentDelete of
/// shared content in the title the data names.
fn XamContentDeleteInternal(c: &mut Call) -> u64 {
    let (data_ptr, overlapped) = (c.arg(0), c.arg(1));
    if data_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    let title = internal_title(c, data_ptr);
    content_delete(c, title, USER_INDEX_NONE, data_ptr, overlapped)
}

fn content_delete(c: &Call, title: u32, user: u32, data_ptr: u32, overlapped: u32) -> u64 {
    let xuid = match user_xuid(c, user) {
        Ok(x) => x,
        Err(e) => return e as u64,
    };
    let data = ContentData::read(c, data_ptr);
    let result = match c.kernel.package_path_for(xuid, title, &data) {
        Some(p) if valid_file_name(&data.file_name) && p.is_dir() => {
            let _ = std::fs::remove_file(p.with_file_name(format!("{}.title", data.file_name)));
            match std::fs::remove_dir_all(&p) {
                Ok(()) => win32::SUCCESS,
                Err(e) => {
                    log::error!("XamContentDelete: {}: {e}", p.display());
                    win32::ACCESS_DENIED
                }
            }
        }
        _ => win32::FILE_NOT_FOUND,
    };
    if overlapped != 0 {
        c.kernel.complete_overlapped(overlapped, result);
        return win32::IO_PENDING as u64;
    }
    result as u64
}

/// The storage devices: (id, XContentDeviceType, name, total, free). Sizes
/// are Xenia's dummy devices'; the hard drive lives in the content root.
const DEVICES: [(u32, u32, &str, u64, u64); 2] =
    [(DEVICE_HDD, 1, "Dummy HDD", 20 << 30, 10 << 30), (DEVICE_ODD, 4, "Dummy ODD", 7 << 30, 0)];

fn device(id: u32) -> Option<(u32, u32, &'static str, u64, u64)> {
    DEVICES.iter().copied().find(|d| d.0 == id)
}

/// XCONTENT_DEVICE_DATA (big-endian): id, type, total and free bytes, name.
fn device_data((id, ty, name, total, free): (u32, u32, &str, u64, u64)) -> Vec<u8> {
    let mut b = vec![0u8; DEVICE_DATA_SIZE as usize];
    b[0..4].copy_from_slice(&id.to_be_bytes());
    b[4..8].copy_from_slice(&ty.to_be_bytes());
    b[8..0x10].copy_from_slice(&total.to_be_bytes());
    b[0x10..0x18].copy_from_slice(&free.to_be_bytes());
    for (i, u) in name.encode_utf16().take(27).enumerate() {
        b[0x18 + i * 2..0x1A + i * 2].copy_from_slice(&u.to_be_bytes());
    }
    b
}

/// (u32 DeviceId, XCONTENT_DEVICE_DATA* Data), 0x50 bytes.
fn XamContentGetDeviceData(c: &mut Call) -> u64 {
    let (id, out) = (c.arg(0), c.arg(1));
    let Some(d) = device(id) else { return win32::DEVICE_NOT_CONNECTED as u64 };
    c.mem().write_bytes(out, &device_data(d));
    win32::SUCCESS as u64
}

/// (u32 ContentType, u32 ContentFlags, u32 MaxCount, u32* BufferSize, HANDLE* Enumerator):
/// an enumerator of the storage devices (XCONTENT_DEVICE_DATA items), without the
/// read-only disc drive when the flags exclude read-only devices (Xenia).
fn XamContentCreateDeviceEnumerator(c: &mut Call) -> u64 {
    let (flags, max_count, size_ptr, handle_ptr) = (c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    if handle_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    if size_ptr != 0 {
        c.mem().write_u32(size_ptr, DEVICE_DATA_SIZE * max_count);
    }
    let items: Vec<Vec<u8>> = DEVICES
        .iter()
        .filter(|d| !(d.0 == DEVICE_ODD && flags & flag::EXCLUDE_READ_ONLY_DEVICES != 0))
        .take(max_count as usize)
        .map(|&d| device_data(d))
        .collect();
    let e = Enumerator::new(DEVICE_DATA_SIZE, max_count, items);
    match c.kernel.objects.create(&c.kernel.rt.space, ObjectType::Other(0), None, Payload::Enumerator(e), None) {
        Ok(obj) => {
            let h = c.kernel.objects.insert_handle(&obj);
            c.mem().write_u32(handle_ptr, h);
            win32::SUCCESS as u64
        }
        Err(s) => {
            log::error!("XamContentCreateDeviceEnumerator: {s:#010X}");
            win32::NOT_ENOUGH_MEMORY as u64
        }
    }
}

/// (u32 UserIndex, char* RootName, char* Path, u32 Flags, u32* Disposition,
///  u32* LicenseMask, XOVERLAPPED* Overlapped): mounts a content package *file*
/// as RootName. Content here is stored as plain directories, so there is never
/// a package file to open: ERROR_FILE_NOT_FOUND, as Xenia.
fn XamContentOpenFile(c: &mut Call) -> u64 {
    let (root_ptr, path_ptr) = (c.arg(1), c.arg(2));
    let read =
        |p: u32| if p == 0 { String::new() } else { String::from_utf8_lossy(&c.mem().read_cstr(p, 260)).into_owned() };
    log::info!("XamContentOpenFile({:?} as {:?}): no package files", read(path_ptr), read(root_ptr));
    win32::FILE_NOT_FOUND as u64
}

/// (u32 DeviceId, char16_t* Name, u32 Capacity)
fn XamContentGetDeviceName(c: &mut Call) -> u64 {
    let (id, buffer, capacity) = (c.arg(0), c.arg(1), c.arg(2));
    let Some((_, _, name, _, _)) = device(id) else { return win32::DEVICE_NOT_CONNECTED as u64 };
    let mut units: Vec<u16> = name.encode_utf16().collect();
    if (capacity as usize) < units.len() + 1 {
        return win32::INSUFFICIENT_BUFFER as u64;
    }
    units.push(0);
    c.mem().write_utf16(buffer, &units);
    win32::SUCCESS as u64
}

/// (u32 DeviceId, XOVERLAPPED* Overlapped): whether the device is present.
fn XamContentGetDeviceState(c: &mut Call) -> u64 {
    let (id, overlapped) = (c.arg(0), c.arg(1));
    let present = device(id).is_some();
    if overlapped != 0 {
        if present {
            c.kernel.complete_overlapped(overlapped, win32::SUCCESS);
        } else {
            c.kernel.complete_overlapped_ex(overlapped, win32::FUNCTION_FAILED, win32::DEVICE_NOT_CONNECTED, 0);
        }
        return win32::IO_PENDING as u64;
    }
    if present { win32::SUCCESS as u64 } else { win32::DEVICE_NOT_CONNECTED as u64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_data_layout() {
        let d = ContentData {
            device_id: DEVICE_HDD,
            content_type: 1,
            display_name: "Save 1".encode_utf16().collect(),
            file_name: "Save1".into(),
        };
        let b = d.to_bytes();
        assert_eq!(b.len(), 0x134);
        assert_eq!(&b[0..8], &[0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(&b[8..12], &[0, b'S', 0, b'a']);
        assert_eq!(&b[0x108..0x10D], b"Save1");
        assert!(valid_file_name("Save1"));
        assert!(!valid_file_name("..") && !valid_file_name("a/b") && !valid_file_name(""));
    }
}
