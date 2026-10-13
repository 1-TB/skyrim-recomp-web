//! Signed-in users (local offline profiles) and the XamUser* identity exports.
//! Behavior follows Xenia's xam_user.cc.
#![allow(non_snake_case)]

use std::sync::Mutex;

use crate::dispatch::{Call, Export, ModuleId};
use crate::input::MAX_USERS;
use crate::status::{hresult, win32};

const X: ModuleId = ModuleId::XAM;

pub(super) const EXPORTS: &[Export] = &[
    export!(X, 0x20A, XamUserGetXUID),
    export!(X, 0x20C, XamUserGetGamerTag),
    export!(X, 0x20E, XamUserGetName),
    export!(X, 0x210, XamUserGetSigninState),
    export!(X, 0x227, XamUserGetSigninInfo),
];

/// eXUserSigninState.
pub const SIGNIN_NOT_SIGNED_IN: u32 = 0;
pub const SIGNIN_LOCALLY: u32 = 1;

/// Gamertags are at most 15 characters.
const MAX_NAME: usize = 15;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub xuid: u64,
    pub name: String,
}

impl Profile {
    /// An offline profile. The XUID is derived from the name (in Xenia's
    /// offline range, 0xE03 << 52) so saves stored under it are found again
    /// on the next run.
    pub fn offline(name: &str) -> Self {
        let name: String = name.chars().filter(|c| c.is_ascii() && !c.is_ascii_control()).take(MAX_NAME).collect();
        let mut hash: u32 = 0x811C_9DC5;
        for b in name.bytes() {
            hash = (hash ^ b as u32).wrapping_mul(0x0100_0193);
        }
        Profile { xuid: (0xE03u64 << 52) | (hash & 0x7FFF_FFFF) as u64, name }
    }
}

pub struct Users {
    slots: Mutex<[Option<Profile>; MAX_USERS]>,
}

impl Default for Users {
    /// User 0 signed in as "Player", as Xenia does with its default profile.
    fn default() -> Self {
        let users = Users { slots: Mutex::new(Default::default()) };
        users.sign_in(0, Profile::offline("Player"));
        users
    }
}

impl Users {
    pub fn sign_in(&self, user: usize, profile: Profile) {
        if let Some(slot) = self.slots.lock().unwrap().get_mut(user) {
            *slot = Some(profile);
        }
    }

    pub fn sign_out(&self, user: usize) {
        if let Some(slot) = self.slots.lock().unwrap().get_mut(user) {
            *slot = None;
        }
    }

    pub fn get(&self, user: u32) -> Option<Profile> {
        self.slots.lock().unwrap().get(user as usize).cloned().flatten()
    }

    pub fn index_of(&self, xuid: u64) -> Option<usize> {
        self.slots.lock().unwrap().iter().position(|p| p.as_ref().is_some_and(|p| p.xuid == xuid))
    }
}

/// (u32 UserIndex, u32 TypeMask, u64* Xuid)
fn XamUserGetXUID(c: &mut Call) -> u64 {
    let (user, type_mask, out) = (c.arg(0), c.arg(1), c.arg(2));
    if out == 0 {
        return hresult::E_INVALIDARG as u64;
    }
    c.mem().write_u64(out, 0);
    if user as usize >= MAX_USERS {
        return hresult::E_INVALIDARG as u64;
    }
    let Some(p) = c.kernel.users.get(user) else {
        return hresult::from_win32(win32::NO_SUCH_USER) as u64;
    };
    // Profiles are offline (1) and, as in Xenia, also answer online (2) and
    // guest (4) requests with the same XUID.
    if type_mask & 7 == 0 {
        return hresult::from_win32(win32::NO_SUCH_USER) as u64;
    }
    c.mem().write_u64(out, p.xuid);
    hresult::S_OK as u64
}

/// (u32 UserIndex, char16_t* Buffer, u32 BufferLength): UTF-16 gamertag.
fn XamUserGetGamerTag(c: &mut Call) -> u64 {
    let (user, buffer, len) = (c.arg(0), c.arg(1), c.arg(2));
    if buffer == 0 || len < 16 || user as usize >= MAX_USERS {
        return hresult::E_INVALIDARG as u64;
    }
    let Some(p) = c.kernel.users.get(user) else {
        return hresult::from_win32(win32::NO_SUCH_USER) as u64;
    };
    let mut units: Vec<u16> = p.name.encode_utf16().take(15).collect();
    units.push(0);
    c.mem().write_utf16(buffer, &units);
    hresult::S_OK as u64
}

/// (u32 UserIndex, char* Buffer, u32 BufferLength)
fn XamUserGetName(c: &mut Call) -> u64 {
    let (user, buffer, len) = (c.arg(0), c.arg(1), c.arg(2));
    if user as usize >= MAX_USERS {
        return win32::INVALID_PARAMETER as u64;
    }
    let Some(p) = c.kernel.users.get(user) else {
        if buffer != 0 && len != 0 {
            c.mem().write_u8(buffer, 0);
        }
        return win32::NO_SUCH_USER as u64;
    };
    if buffer != 0 && len != 0 {
        let n = (len as usize - 1).min(p.name.len());
        c.mem().write_bytes(buffer, &p.name.as_bytes()[..n]);
        c.mem().write_u8(buffer + n as u32, 0);
    }
    win32::SUCCESS as u64
}

fn XamUserGetSigninState(c: &mut Call) -> u64 {
    let user = c.arg(0);
    if user as usize >= MAX_USERS {
        return SIGNIN_NOT_SIGNED_IN as u64;
    }
    (if c.kernel.users.get(user).is_some() { SIGNIN_LOCALLY } else { SIGNIN_NOT_SIGNED_IN }) as u64
}

/// XUSER_GET_SIGNIN_INFO_OFFLINE_XUID_ONLY.
const SIGNIN_INFO_OFFLINE_XUID_ONLY: u32 = 2;

/// (u32 UserIndex, u32 Flags, XUSER_SIGNIN_INFO* Info), 40 bytes.
fn XamUserGetSigninInfo(c: &mut Call) -> u64 {
    let (user, flags, info) = (c.arg(0), c.arg(1), c.arg(2));
    if info == 0 {
        return hresult::E_INVALIDARG as u64;
    }
    let mem = c.mem();
    mem.fill(info, 40, 0);
    if user as usize >= MAX_USERS {
        return hresult::from_win32(win32::NO_SUCH_USER) as u64;
    }
    let Some(p) = c.kernel.users.get(user) else {
        return hresult::from_win32(win32::NO_SUCH_USER) as u64;
    };
    if flags == 0 || flags & SIGNIN_INFO_OFFLINE_XUID_ONLY != 0 {
        mem.write_u64(info, p.xuid);
    }
    mem.write_u32(info + 0xC, SIGNIN_LOCALLY);
    mem.write_bytes(info + 0x18, p.name.as_bytes());
    hresult::S_OK as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_xuid_is_stable_and_in_the_offline_range() {
        let a = Profile::offline("Dovahkiin");
        assert_eq!(a, Profile::offline("Dovahkiin"));
        assert_eq!(a.xuid >> 52, 0xE03);
        assert_ne!(a.xuid, Profile::offline("Lydia").xuid);
        assert_eq!(Profile::offline("ThisNameIsWayTooLongForXbox").name.len(), 15);
        let users = Users::default();
        assert_eq!(users.get(0).unwrap().name, "Player");
        assert!(users.get(1).is_none());
        assert_eq!(users.index_of(users.get(0).unwrap().xuid), Some(0));
    }
}
