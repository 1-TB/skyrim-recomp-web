//! ANSI_STRING / UNICODE_STRING / OBJECT_ATTRIBUTES helpers.

use runtime_core::memory::GuestMemory;

/// `{ u16 Length; u16 MaximumLength; u32 Buffer; }` (both string kinds).
pub const STRING_SIZE: u32 = 8;

pub fn read_ansi(mem: &GuestMemory, string_ptr: u32) -> Vec<u8> {
    if string_ptr == 0 {
        return Vec::new();
    }
    let len = mem.read_u16(string_ptr) as u32;
    let buf = mem.read_u32(string_ptr + 4);
    if buf == 0 || len == 0 {
        return Vec::new();
    }
    mem.read_vec(buf, len)
}

pub fn read_ansi_lossy(mem: &GuestMemory, string_ptr: u32) -> String {
    String::from_utf8_lossy(&read_ansi(mem, string_ptr)).into_owned()
}

pub fn read_unicode(mem: &GuestMemory, string_ptr: u32) -> Vec<u16> {
    if string_ptr == 0 {
        return Vec::new();
    }
    let len = mem.read_u16(string_ptr) as u32 / 2;
    let buf = mem.read_u32(string_ptr + 4);
    if buf == 0 {
        return Vec::new();
    }
    mem.read_utf16(buf, len)
}

pub fn write_string_header(mem: &GuestMemory, string_ptr: u32, len: u16, max: u16, buf: u32) {
    mem.write_u16(string_ptr, len);
    mem.write_u16(string_ptr + 2, max);
    mem.write_u32(string_ptr + 4, buf);
}

/// `{ u32 RootDirectory; u32 ObjectName (PANSI_STRING); u32 Attributes; }`
pub struct ObjectAttributes {
    pub root_directory: u32,
    pub name: Option<String>,
    pub attributes: u32,
}

pub fn read_object_attributes(mem: &GuestMemory, ptr: u32) -> Option<ObjectAttributes> {
    if ptr == 0 {
        return None;
    }
    let name_ptr = mem.read_u32(ptr + 4);
    let name = (name_ptr != 0).then(|| read_ansi_lossy(mem, name_ptr)).filter(|n| !n.is_empty());
    Some(ObjectAttributes { root_directory: mem.read_u32(ptr), name, attributes: mem.read_u32(ptr + 8) })
}

/// Code page 1252-compatible narrowing used by the Rtl conversion routines:
/// characters above 0xFF become '?'.
pub fn narrow(units: &[u16]) -> Vec<u8> {
    units.iter().map(|&u| if u <= 0xFF { u as u8 } else { b'?' }).collect()
}

pub fn widen(bytes: &[u8]) -> Vec<u16> {
    bytes.iter().map(|&b| b as u16).collect()
}

pub fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}
