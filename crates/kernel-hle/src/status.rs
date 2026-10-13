//! NTSTATUS and Win32 error values returned to the guest.

pub const SUCCESS: u32 = 0x0000_0000;
pub const ABANDONED_WAIT_0: u32 = 0x0000_0080;
pub const USER_APC: u32 = 0x0000_00C0;
pub const ALERTED: u32 = 0x0000_0101;
pub const TIMEOUT: u32 = 0x0000_0102;
pub const PENDING: u32 = 0x0000_0103;
pub const OBJECT_NAME_EXISTS: u32 = 0x4000_0000;
pub const BUFFER_OVERFLOW: u32 = 0x8000_0005;
pub const NO_MORE_FILES: u32 = 0x8000_0006;
pub const UNSUCCESSFUL: u32 = 0xC000_0001;
pub const NOT_IMPLEMENTED: u32 = 0xC000_0002;
pub const INVALID_INFO_CLASS: u32 = 0xC000_0003;
pub const INFO_LENGTH_MISMATCH: u32 = 0xC000_0004;
pub const ACCESS_VIOLATION: u32 = 0xC000_0005;
pub const INVALID_HANDLE: u32 = 0xC000_0008;
pub const INVALID_PARAMETER: u32 = 0xC000_000D;
pub const NO_SUCH_FILE: u32 = 0xC000_000F;
pub const END_OF_FILE: u32 = 0xC000_0011;
pub const NO_MEMORY: u32 = 0xC000_0017;
pub const CONFLICTING_ADDRESSES: u32 = 0xC000_0018;
pub const ALREADY_COMMITTED: u32 = 0xC000_0021;
pub const ACCESS_DENIED: u32 = 0xC000_0022;
pub const BUFFER_TOO_SMALL: u32 = 0xC000_0023;
pub const OBJECT_TYPE_MISMATCH: u32 = 0xC000_0024;
pub const OBJECT_NAME_INVALID: u32 = 0xC000_0033;
pub const OBJECT_NAME_NOT_FOUND: u32 = 0xC000_0034;
pub const OBJECT_NAME_COLLISION: u32 = 0xC000_0035;
pub const OBJECT_PATH_NOT_FOUND: u32 = 0xC000_003A;
pub const INVALID_PAGE_PROTECTION: u32 = 0xC000_0045;
pub const MUTANT_NOT_OWNED: u32 = 0xC000_0046;
pub const SEMAPHORE_LIMIT_EXCEEDED: u32 = 0xC000_0047;
pub const MEMORY_NOT_ALLOCATED: u32 = 0xC000_00A0;
pub const FILE_IS_A_DIRECTORY: u32 = 0xC000_00BA;
pub const NOT_SUPPORTED: u32 = 0xC000_00BB;
pub const INVALID_PARAMETER_1: u32 = 0xC000_00EF;
pub const INVALID_PARAMETER_2: u32 = 0xC000_00F0;
pub const INVALID_PARAMETER_3: u32 = 0xC000_00F1;
pub const NOT_A_DIRECTORY: u32 = 0xC000_0103;
pub const NOT_FOUND: u32 = 0xC000_0225;

pub fn is_error(status: u32) -> bool {
    status >= 0xC000_0000
}

/// Win32 errors (XAM returns these directly).
pub mod win32 {
    pub const SUCCESS: u32 = 0;
    pub const FILE_NOT_FOUND: u32 = 2;
    pub const PATH_NOT_FOUND: u32 = 3;
    pub const ACCESS_DENIED: u32 = 5;
    pub const INVALID_HANDLE: u32 = 6;
    pub const NOT_ENOUGH_MEMORY: u32 = 8;
    pub const NO_MORE_FILES: u32 = 18;
    pub const HANDLE_EOF: u32 = 38;
    pub const NOT_SUPPORTED: u32 = 50;
    pub const FILE_EXISTS: u32 = 80;
    pub const INVALID_PARAMETER: u32 = 87;
    pub const INSUFFICIENT_BUFFER: u32 = 122;
    pub const ALREADY_EXISTS: u32 = 183;
    pub const NOT_FOUND: u32 = 1168;
    pub const MR_MID_NOT_FOUND: u32 = 317;
    pub const BAD_ARGUMENTS: u32 = 160;
    pub const IO_INCOMPLETE: u32 = 996;
    pub const IO_PENDING: u32 = 997;
    pub const CANCELLED: u32 = 1223;
    pub const DEVICE_NOT_CONNECTED: u32 = 1167;
    pub const NO_SUCH_USER: u32 = 1525;
    pub const FUNCTION_FAILED: u32 = 1627;
}

/// HRESULTs returned by XAM.
pub mod hresult {
    pub const S_OK: u32 = 0;
    pub const E_FAIL: u32 = 0x8000_4005;
    pub const E_INVALIDARG: u32 = 0x8007_0057;
    /// HRESULT_FROM_WIN32(ERROR_NOT_FOUND).
    pub const E_NOTFOUND: u32 = 0x8007_0490;

    /// HRESULT_FROM_WIN32.
    pub const fn from_win32(code: u32) -> u32 {
        if code == 0 { 0 } else { (code & 0xFFFF) | 0x8007_0000 }
    }
}

/// RtlNtStatusToDosError for the statuses this kernel produces, falling back to
/// the NT convention of returning ERROR_MR_MID_NOT_FOUND for unknown codes.
pub fn to_dos_error(status: u32) -> u32 {
    match status {
        SUCCESS => win32::SUCCESS,
        TIMEOUT => 1460,
        PENDING => 997,
        BUFFER_OVERFLOW => 234,
        NO_MORE_FILES => win32::NO_MORE_FILES,
        UNSUCCESSFUL => 31,
        NOT_IMPLEMENTED => 1,
        INVALID_INFO_CLASS | INVALID_PARAMETER | INVALID_PARAMETER_1 | INVALID_PARAMETER_2 | INVALID_PARAMETER_3 => {
            win32::INVALID_PARAMETER
        }
        INFO_LENGTH_MISMATCH => 24,
        ACCESS_VIOLATION => 998,
        INVALID_HANDLE => win32::INVALID_HANDLE,
        NO_SUCH_FILE | OBJECT_NAME_NOT_FOUND => win32::FILE_NOT_FOUND,
        OBJECT_PATH_NOT_FOUND => win32::PATH_NOT_FOUND,
        END_OF_FILE => win32::HANDLE_EOF,
        NO_MEMORY => win32::NOT_ENOUGH_MEMORY,
        ACCESS_DENIED => win32::ACCESS_DENIED,
        BUFFER_TOO_SMALL => win32::INSUFFICIENT_BUFFER,
        OBJECT_TYPE_MISMATCH => 1,
        OBJECT_NAME_INVALID => 123,
        OBJECT_NAME_COLLISION => win32::ALREADY_EXISTS,
        MUTANT_NOT_OWNED => 288,
        SEMAPHORE_LIMIT_EXCEEDED => 298,
        FILE_IS_A_DIRECTORY | NOT_A_DIRECTORY => 267,
        NOT_SUPPORTED => win32::NOT_SUPPORTED,
        NOT_FOUND => win32::NOT_FOUND,
        s if s & 0xFFFF_0000 == 0x8007_0000 || s & 0xFFFF_0000 == 0xC007_0000 => s & 0xFFFF,
        _ => win32::MR_MID_NOT_FOUND,
    }
}
