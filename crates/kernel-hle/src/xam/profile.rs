//! Profile settings (XamUserRead/WriteProfileSettings): dashboard options
//! such as controller vibration, and the title's own per-profile blobs.
//!
//! A setting id carries its type and maximum size (type << 28 | size << 16 |
//! index). Values a title writes are kept per (XUID, title) and saved under
//! the content root in `<XUID>/<TitleID>/profile_settings.bin`; anything never
//! written reads back as its default. Behavior follows Xenia's xam_user.cc,
//! user_settings.* and user_data.*.
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::Kernel;
use crate::dispatch::{Call, Export, ModuleId};
use crate::input::MAX_USERS;
use crate::status::{hresult, win32};

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x219, XamUserReadProfileSettings),
    export!(X, 0x21A, XamUserWriteProfileSettings),
    export!(X, 0x22A, XamUserReadProfileSettingsEx),
];

/// X_USER_DATA_TYPE.
mod data_type {
    pub const CONTEXT: u8 = 0;
    pub const INT32: u8 = 1;
    pub const INT64: u8 = 2;
    pub const DOUBLE: u8 = 3;
    pub const WSTRING: u8 = 4;
    pub const FLOAT: u8 = 5;
    pub const BINARY: u8 = 6;
    pub const DATETIME: u8 = 7;
}

/// X_USER_PROFILE_SETTING_SOURCE.
const SOURCE_DEFAULT: u32 = 1;
const SOURCE_TITLE: u32 = 2;

/// sizeof(X_USER_PROFILE_SETTING) and X_USER_READ_PROFILE_SETTINGS.
const SETTING_SIZE: u32 = 40;
const READ_HEADER_SIZE: u32 = 8;
/// Largest value a setting can hold.
const MAX_DATA_SIZE: usize = 0x3E8;
/// Settings with an index below this are always valid (Xenia kMaxUserSettingId).
const MAX_INDEXED_SETTING: u32 = 0x58;
/// kXNotificationSystemProfileSettingChanged.
const PROFILE_SETTING_CHANGED: u32 = 0x0000_000E;

const XPROFILE_OPTION_CONTROLLER_VIBRATION: u32 = 0x1004_0003;
const XPROFILE_GAMER_TIER: u32 = 0x1004_003A;
const XPROFILE_GAMERCARD_PICTURE_KEY: u32 = 0x4064_000F;
/// AccountSubscriptionTier::kSubscriptionTierGold.
const TIER_GOLD: i32 = 6;

/// Known settings whose index is at or above MAX_INDEXED_SETTING (from
/// Xenia's known_settings): web, Crux (gamercard) and title-specific blobs.
const KNOWN_HIGH_SETTINGS: [u32; 39] = [
    0x1004_200B,
    0x1004_2000,
    0x1004_2004,
    0x1004_2005,
    0x1004_2006,
    0x1004_2007,
    0x1004_2008,
    0x1004_2009,
    0x1004_2003,
    0x1004_2001,
    0x1004_200C,
    0x1004_200A,
    0x1004_2002,
    0x1004_200D,
    0x4064_03FC,
    0x4064_03FE,
    0x4064_03FB,
    0x4064_03FD,
    0x43E8_03FA,
    0x1004_03F3,
    0x7008_03F4,
    0x4100_03F6,
    0x4064_03E8,
    0x1004_03EA,
    0x1004_03EB,
    0x1004_03EC,
    0x6034_03F2,
    0x1004_03ED,
    0x1004_03EE,
    0x1004_03EF,
    0x1004_03F0,
    0x1004_03F1,
    0x6010_03F7,
    0x6010_03F8,
    0x6010_03F9,
    0x60A8_03F5,
    0x63E8_3FFF,
    0x63E8_3FFE,
    0x63E8_3FFD,
];

fn setting_type(id: u32) -> u8 {
    (id >> 28) as u8
}

fn max_size(id: u32) -> u32 {
    (id >> 16) & 0xFFF
}

fn is_setting_valid(id: u32) -> bool {
    (id & 0xFFFF) < MAX_INDEXED_SETTING || KNOWN_HIGH_SETTINGS.contains(&id)
}

fn has_extended_data(ty: u8) -> bool {
    ty == data_type::WSTRING || ty == data_type::BINARY
}

/// One setting value: the X_USER_DATA union bytes (big-endian) plus, for
/// strings and binary blobs, the bytes it points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Setting {
    pub id: u32,
    pub source: u32,
    pub ty: u8,
    pub data: [u8; 8],
    pub ext: Vec<u8>,
}

impl Setting {
    fn with_u32(id: u32, v: u32) -> Self {
        let mut data = [0u8; 8];
        data[..4].copy_from_slice(&v.to_be_bytes());
        Setting { id, source: SOURCE_DEFAULT, ty: setting_type(id), data, ext: Vec::new() }
    }

    /// A string value: UTF-16BE with its terminator.
    fn with_string(id: u32, s: &str) -> Self {
        let mut ext: Vec<u8> = s.encode_utf16().take(MAX_DATA_SIZE / 2 - 1).flat_map(u16::to_be_bytes).collect();
        ext.extend_from_slice(&[0, 0]);
        let mut data = [0u8; 8];
        data[..4].copy_from_slice(&(ext.len() as u32).to_be_bytes());
        Setting { id, source: SOURCE_DEFAULT, ty: data_type::WSTRING, data, ext }
    }

    /// The value a setting has before the title writes one.
    pub fn default_for(id: u32) -> Self {
        match id {
            XPROFILE_OPTION_CONTROLLER_VIBRATION => Setting::with_u32(id, 3),
            XPROFILE_GAMER_TIER => Setting::with_u32(id, TIER_GOLD as u32),
            XPROFILE_GAMERCARD_PICTURE_KEY => Setting::with_string(id, "FFFE07D10002000200010002"),
            _ => match setting_type(id) {
                data_type::WSTRING => Setting::with_string(id, ""),
                data_type::BINARY => {
                    Setting { id, source: SOURCE_DEFAULT, ty: data_type::BINARY, data: [0; 8], ext: Vec::new() }
                }
                data_type::CONTEXT | data_type::INT64 | data_type::DOUBLE | data_type::FLOAT | data_type::DATETIME => {
                    Setting { id, source: SOURCE_DEFAULT, ty: setting_type(id), data: [0; 8], ext: Vec::new() }
                }
                // INT32, and types outside the enum, read as a zero int.
                _ => Setting { id, source: SOURCE_DEFAULT, ty: data_type::INT32, data: [0; 8], ext: Vec::new() },
            },
        }
    }

    /// Reads an X_USER_PROFILE_SETTING the title passed in.
    fn read_guest(c: &Call, at: u32) -> Self {
        let mem = c.mem();
        let id = mem.read_u32(at + 0x10);
        let ty = mem.read_u8(at + 0x18);
        let mut data = [0u8; 8];
        mem.read_bytes(at + 0x20, &mut data);
        let mut ext = Vec::new();
        if has_extended_data(ty) {
            let size = (u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize).min(MAX_DATA_SIZE);
            let ptr = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
            if size != 0 && ptr != 0 {
                ext = mem.read_vec(ptr, size as u32);
            }
            data[..4].copy_from_slice(&(ext.len() as u32).to_be_bytes());
        }
        Setting { id, source: SOURCE_TITLE, ty, data, ext }
    }

    /// Fills the value part of an X_USER_PROFILE_SETTING; strings and blobs
    /// go to `*ext_cursor`, which advances by the setting's maximum size.
    fn write_guest(&self, c: &Call, at: u32, ext_cursor: &mut u32) {
        let mem = c.mem();
        mem.write_u32(at, self.source);
        mem.write_u32(at + 0x10, self.id);
        mem.write_u8(at + 0x18, self.ty);
        mem.write_bytes(at + 0x20, &self.data);
        if has_extended_data(self.ty) && !self.ext.is_empty() {
            mem.write_u32(at + 0x20, self.ext.len() as u32);
            mem.write_u32(at + 0x24, *ext_cursor);
            mem.write_bytes(*ext_cursor, &self.ext);
            *ext_cursor += max_size(self.id);
        }
    }

    fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.id.to_be_bytes());
        out.push(self.ty);
        out.extend_from_slice(&self.data);
        out.extend_from_slice(&(self.ext.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.ext);
    }

    fn deserialize(b: &[u8]) -> Option<(Self, usize)> {
        let id = u32::from_be_bytes(b.get(0..4)?.try_into().ok()?);
        let ty = *b.get(4)?;
        let data: [u8; 8] = b.get(5..13)?.try_into().ok()?;
        let len = u32::from_be_bytes(b.get(13..17)?.try_into().ok()?) as usize;
        let ext = b.get(17..17 + len)?.to_vec();
        Some((Setting { id, source: SOURCE_TITLE, ty, data, ext }, 17 + len))
    }
}

/// Title-written settings, loaded lazily per (XUID, title).
#[derive(Default)]
pub struct ProfileSettings {
    stores: Mutex<HashMap<(u64, u32), HashMap<u32, Setting>>>,
}

impl Kernel {
    fn settings_file(&self, xuid: u64, title: u32) -> Option<PathBuf> {
        Some(
            self.content.root()?.join(format!("{xuid:016X}")).join(format!("{title:08X}")).join("profile_settings.bin"),
        )
    }

    fn with_settings<R>(&self, xuid: u64, title: u32, f: impl FnOnce(&mut HashMap<u32, Setting>) -> R) -> R {
        let mut stores = self.profile_settings.stores.lock().unwrap();
        let store = stores.entry((xuid, title)).or_insert_with(|| {
            let mut map = HashMap::new();
            if let Some(bytes) = self.settings_file(xuid, title).and_then(|p| std::fs::read(p).ok()) {
                let mut at = 0;
                while let Some((s, used)) = Setting::deserialize(&bytes[at..]) {
                    map.insert(s.id, s);
                    at += used;
                }
            }
            map
        });
        f(store)
    }

    /// The value the title would read: what it wrote, else the default.
    pub fn profile_setting(&self, xuid: u64, title: u32, id: u32) -> Setting {
        self.with_settings(xuid, title, |m| m.get(&id).cloned()).unwrap_or_else(|| Setting::default_for(id))
    }

    fn store_profile_settings(&self, xuid: u64, title: u32, settings: Vec<Setting>) {
        let bytes = self.with_settings(xuid, title, |m| {
            for s in settings {
                m.insert(s.id, s);
            }
            let mut ids: Vec<&u32> = m.keys().collect();
            ids.sort();
            let mut out = Vec::new();
            for id in ids {
                m[id].serialize(&mut out);
            }
            out
        });
        if let Some(path) = self.settings_file(xuid, title) {
            let written =
                path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|_| std::fs::write(&path, bytes));
            if let Err(e) = written {
                log::error!("saving profile settings to {}: {e}", path.display());
            }
        }
    }

    fn current_title(&self, title: u32) -> u32 {
        if title != 0 { title } else { self.rt.xex.execution_info().map_or(0, |i| i.title_id) }
    }
}

/// Shared by XamUserReadProfileSettings(Ex).
#[allow(clippy::too_many_arguments)]
fn read_settings(
    c: &Call,
    title: u32,
    user: u32,
    xuid_count: u32,
    xuids: u32,
    count: u32,
    ids_ptr: u32,
    size_ptr: u32,
    buffer: u32,
    overlapped: u32,
) -> u64 {
    if !(1..=32).contains(&count) || size_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    let mem = c.mem();
    let buffer_size = mem.read_u32(size_ptr);
    if buffer_size != 0 && buffer == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    let ids: Vec<u32> = (0..count).map(|i| mem.read_u32(ids_ptr + i * 4)).collect();
    let data_size: u32 = ids.iter().filter(|&&id| has_extended_data(setting_type(id))).map(|&id| max_size(id)).sum();
    let users = if xuids != 0 { xuid_count.max(1) } else { 1 };
    let needed = READ_HEADER_SIZE + (count * SETTING_SIZE + data_size) * users;
    if buffer == 0 || buffer_size < needed {
        if buffer_size == 0 {
            mem.write_u32(size_ptr, needed);
        }
        return win32::INSUFFICIENT_BUFFER as u64;
    }

    let profile = if xuids != 0 {
        let xuid = mem.read_u64(xuids);
        c.kernel.users.index_of(xuid).and_then(|i| c.kernel.users.get(i as u32)).map(|p| (p, None))
    } else if (user as usize) < MAX_USERS {
        c.kernel.users.get(user).map(|p| (p, Some(user)))
    } else {
        None
    };
    let Some((profile, by_index)) = profile else {
        if overlapped != 0 {
            c.kernel.complete_overlapped_ex(
                overlapped,
                win32::FUNCTION_FAILED,
                hresult::from_win32(win32::NO_SUCH_USER),
                0,
            );
            return win32::IO_PENDING as u64;
        }
        return win32::FUNCTION_FAILED as u64;
    };

    let title = c.kernel.current_title(title);
    let settings_ptr = buffer + READ_HEADER_SIZE;
    mem.fill(settings_ptr, count * SETTING_SIZE, 0);
    let mut ext_cursor = settings_ptr + count * SETTING_SIZE;
    let mut out = settings_ptr;
    let mut written = 0;
    for &id in &ids {
        if !is_setting_valid(id) {
            if id != 0 {
                log::warn!("XamUserReadProfileSettings: unknown setting {id:#010X}");
            }
            continue;
        }
        c.kernel.profile_setting(profile.xuid, title, id).write_guest(c, out, &mut ext_cursor);
        match by_index {
            Some(index) => {
                mem.write_u64(out + 8, u64::MAX);
                mem.write_u32(out + 8, index);
            }
            None => mem.write_u64(out + 8, profile.xuid),
        }
        out += SETTING_SIZE;
        written += 1;
    }
    mem.write_u32(buffer, written);
    mem.write_u32(buffer + 4, settings_ptr);
    if overlapped != 0 {
        c.kernel.complete_overlapped_ex(overlapped, 0, 0, 0);
        return win32::IO_PENDING as u64;
    }
    win32::SUCCESS as u64
}

/// (u32 TitleId, u32 UserIndex, u32 XuidCount, u64* Xuids, u32 SettingCount,
///  u32* SettingIds, u32* BufferSize, X_USER_READ_PROFILE_SETTINGS* Buffer,
///  XOVERLAPPED* Overlapped)
fn XamUserReadProfileSettings(c: &mut Call) -> u64 {
    read_settings(c, c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), c.arg(6), c.arg(7), c.arg(8))
}

/// As XamUserReadProfileSettings with a second buffer-size pointer (used when
/// the first is null) before the buffer.
fn XamUserReadProfileSettingsEx(c: &mut Call) -> u64 {
    let size_ptr = if c.arg(6) != 0 { c.arg(6) } else { c.arg(7) };
    read_settings(c, c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4), c.arg(5), size_ptr, c.arg(8), c.arg(9))
}

/// (u32 TitleId, u32 UserIndex, u32 SettingCount, X_USER_PROFILE_SETTING*
///  Settings, XOVERLAPPED* Overlapped)
fn XamUserWriteProfileSettings(c: &mut Call) -> u64 {
    let (title, user, count, settings_ptr, overlapped) = (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    if count == 0 || settings_ptr == 0 {
        return win32::INVALID_PARAMETER as u64;
    }
    let result = match c.kernel.users.get(user) {
        None => win32::NO_SUCH_USER,
        Some(profile) => {
            let settings: Vec<Setting> = (0..count)
                .map(|i| Setting::read_guest(c, settings_ptr + i * SETTING_SIZE))
                .filter(|s| s.ty == setting_type(s.id) && is_setting_valid(s.id))
                .collect();
            c.kernel.store_profile_settings(profile.xuid, c.kernel.current_title(title), settings);
            c.kernel.broadcast_notification(PROFILE_SETTING_CHANGED, (1 << user) & 0xF);
            win32::SUCCESS
        }
    };
    if overlapped != 0 {
        c.kernel.complete_overlapped_ex(overlapped, result, hresult::from_win32(result), 0);
        return win32::IO_PENDING as u64;
    }
    result as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_ids_encode_type_and_size() {
        assert_eq!(setting_type(0x63E8_3FFF), data_type::BINARY);
        assert_eq!(max_size(0x63E8_3FFF), 0x3E8);
        assert!(is_setting_valid(0x1004_0003));
        assert!(is_setting_valid(0x63E8_3FFE));
        assert!(!is_setting_valid(0x1004_0099));
        let v = Setting::default_for(XPROFILE_OPTION_CONTROLLER_VIBRATION);
        assert_eq!(&v.data[..4], &3u32.to_be_bytes());
        let pic = Setting::default_for(XPROFILE_GAMERCARD_PICTURE_KEY);
        assert_eq!(pic.ext.len(), 25 * 2);
        assert_eq!(&pic.ext[..2], &[0, b'F']);
        assert!(Setting::default_for(0x63E8_3FFF).ext.is_empty());
        let mut buf = Vec::new();
        pic.serialize(&mut buf);
        let (back, used) = Setting::deserialize(&buf).unwrap();
        assert_eq!(used, buf.len());
        assert_eq!((back.id, back.ext), (pic.id, pic.ext));
    }
}
