//! Console configuration store read by ExGetXConfigSetting.
//!
//! Category blobs use the console's big-endian layouts (offsets from Xenia
//! xconfig.h). Defaults describe a retail NTSC-U console set to English/US,
//! 720p over HDMI, UTC.

use std::sync::Mutex;

pub mod category {
    pub const SECURED: u16 = 0x02;
    pub const USER: u16 = 0x03;
}

const SECURED_SIZE: usize = 0x200;
const USER_SIZE: usize = 0x1FD;

/// (category, setting, offset, size)
const FIELDS: &[(u16, u16, usize, usize)] = &[
    (category::SECURED, 0x01, 0x20, 6), // MAC address
    (category::SECURED, 0x02, 0x28, 4), // AV region
    (category::SECURED, 0x03, 0x2C, 2), // game region
    (category::SECURED, 0x04, 0x34, 4), // DVD region
    (category::SECURED, 0x05, 0x38, 4), // reset key
    (category::SECURED, 0x06, 0x3C, 4), // system flags
    (category::SECURED, 0x07, 0x40, 2), // power mode
    (category::SECURED, 0x08, 0x08, 4), // online network id
    (category::SECURED, 0x09, 0x42, 2), // power VCS control
    (category::USER, 0x01, 0x008, 4),   // time zone bias
    (category::USER, 0x02, 0x00C, 4),   // TZ standard name
    (category::USER, 0x03, 0x010, 4),   // TZ daylight name
    (category::USER, 0x04, 0x014, 4),   // TZ standard date
    (category::USER, 0x05, 0x018, 4),   // TZ daylight date
    (category::USER, 0x06, 0x01C, 4),   // TZ standard bias
    (category::USER, 0x07, 0x020, 4),   // TZ daylight bias
    (category::USER, 0x08, 0x024, 8),   // default profile
    (category::USER, 0x09, 0x02C, 4),   // language
    (category::USER, 0x0A, 0x030, 4),   // video flags
    (category::USER, 0x0B, 0x034, 4),   // audio flags
    (category::USER, 0x0C, 0x038, 4),   // retail flags
    (category::USER, 0x0D, 0x03C, 4),   // devkit flags
    (category::USER, 0x0E, 0x040, 1),   // country
    (category::USER, 0x0F, 0x041, 1),   // parental control flags
    (category::USER, 0x10, 0x044, 256), // SMB config
    (category::USER, 0x11, 0x144, 8),   // Live PUID
    (category::USER, 0x12, 0x14C, 16),  // Live credentials
    (category::USER, 0x13, 0x15C, 4),   // AV screen size (HDMI/composite)
    (category::USER, 0x14, 0x160, 4),   // AV screen size (component)
    (category::USER, 0x15, 0x164, 4),   // AV screen size (VGA)
    (category::USER, 0x16, 0x168, 4),   // PC game
    (category::USER, 0x17, 0x16C, 4),   // PC password
    (category::USER, 0x18, 0x170, 4),   // PC movie
    (category::USER, 0x19, 0x174, 4),   // PC game rating
    (category::USER, 0x1A, 0x178, 4),   // PC movie rating
    (category::USER, 0x1B, 0x17C, 1),   // PC hint
    (category::USER, 0x1C, 0x17D, 32),  // PC hint answer
    (category::USER, 0x1D, 0x19D, 32),  // PC override
    (category::USER, 0x1E, 0x1BD, 4),   // music playback mode
    (category::USER, 0x1F, 0x1C1, 4),   // music volume (float)
    (category::USER, 0x20, 0x1C5, 4),   // music flags
    (category::USER, 0x21, 0x1C9, 4),   // arcade flags
    (category::USER, 0x22, 0x1CD, 4),   // PC version
    (category::USER, 0x23, 0x1D1, 4),   // PC TV
    (category::USER, 0x24, 0x1D5, 4),   // PC TV rating
    (category::USER, 0x25, 0x1D9, 4),   // PC explicit video
    (category::USER, 0x26, 0x1DD, 4),   // PC explicit video rating
    (category::USER, 0x27, 0x1E1, 4),   // PC unrated video
    (category::USER, 0x28, 0x1E5, 4),   // PC unrated video rating
    (category::USER, 0x29, 0x1E9, 4),   // video output black levels
    (category::USER, 0x2A, 0x1ED, 1),   // video player display mode
    (category::USER, 0x2B, 0x1EE, 4),   // alternate video timing id
    (category::USER, 0x2C, 0x1F2, 4),   // video driver options
    (category::USER, 0x2D, 0x1F6, 4),   // music UI flags
    (category::USER, 0x2E, 0x1FA, 1),   // video media source type
    (category::USER, 0x2F, 0x1FB, 1),   // music media source type
    (category::USER, 0x30, 0x1FC, 1),   // photo media source type
];

pub const LANGUAGE_ENGLISH: u32 = 1;
pub const COUNTRY_UNITED_STATES: u8 = 103;
pub const AV_REGION_NTSCM: u32 = 0x0040_0100;
pub const GAME_REGION_ALL: u16 = 0xFFFF;

pub struct XConfig {
    secured: Mutex<Vec<u8>>,
    user: Mutex<Vec<u8>>,
}

impl Default for XConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl XConfig {
    pub fn new() -> Self {
        let mut secured = vec![0u8; SECURED_SIZE];
        secured[0x20..0x26].copy_from_slice(&[0x00, 0x22, 0x48, 0x00, 0x00, 0x01]);
        secured[0x28..0x2C].copy_from_slice(&AV_REGION_NTSCM.to_be_bytes());
        secured[0x2C..0x2E].copy_from_slice(&GAME_REGION_ALL.to_be_bytes());
        secured[0x34..0x38].copy_from_slice(&1u32.to_be_bytes());

        let mut user = vec![0u8; USER_SIZE];
        user[0x0C..0x10].copy_from_slice(b"UTC\0");
        user[0x10..0x14].copy_from_slice(b"UTC\0");
        user[0x2C..0x30].copy_from_slice(&LANGUAGE_ENGLISH.to_be_bytes());
        // Dolby Digital | Dolby Pro Logic
        user[0x34..0x38].copy_from_slice(&0x0001_0001u32.to_be_bytes());
        // Dashboard initialized
        user[0x38..0x3C].copy_from_slice(&0x0000_0040u32.to_be_bytes());
        user[0x40] = COUNTRY_UNITED_STATES;
        // Screen sizes encoded as (width << 16) | height.
        user[0x15C..0x160].copy_from_slice(&((1280u32 << 16) | 720).to_be_bytes());
        user[0x160..0x164].copy_from_slice(&((1280u32 << 16) | 720).to_be_bytes());
        user[0x164..0x168].copy_from_slice(&((1280u32 << 16) | 720).to_be_bytes());
        user[0x1C1..0x1C5].copy_from_slice(&0.7f32.to_bits().to_be_bytes());

        XConfig { secured: Mutex::new(secured), user: Mutex::new(user) }
    }

    fn field(category: u16, setting: u16) -> Option<(usize, usize)> {
        FIELDS.iter().find(|f| f.0 == category && f.1 == setting).map(|f| (f.2, f.3))
    }

    fn blob(&self, category: u16) -> Option<&Mutex<Vec<u8>>> {
        match category {
            category::SECURED => Some(&self.secured),
            category::USER => Some(&self.user),
            _ => None,
        }
    }

    /// Size of a setting; 0 for settings this store doesn't have (the kernel
    /// reports success with a zero size for those, as Xenia does).
    pub fn size(&self, category: u16, setting: u16) -> usize {
        Self::field(category, setting).map_or(0, |f| f.1)
    }

    pub fn read(&self, category: u16, setting: u16) -> Option<Vec<u8>> {
        let (off, len) = Self::field(category, setting)?;
        Some(self.blob(category)?.lock().unwrap()[off..off + len].to_vec())
    }

    pub fn write(&self, category: u16, setting: u16, data: &[u8]) -> bool {
        let Some((off, len)) = Self::field(category, setting) else { return false };
        let Some(blob) = self.blob(category) else { return false };
        let n = len.min(data.len());
        blob.lock().unwrap()[off..off + n].copy_from_slice(&data[..n]);
        true
    }
}
