//! XEX2 container parsing.
//!
//! Only decrypted executables are accepted: the retail AES key is console
//! material and is deliberately not part of this project. Uncompressed and
//! "basic" (zero-run) compressed images are supported; LZX-compressed images
//! must be decompressed beforehand (e.g. `xextool -e u -c u default.xex`).

pub mod builder;

use std::fmt;

/// Optional header keys. The low byte encodes the payload size: 0x00/0x01 mean
/// the value is stored inline, 0xFF means the payload starts with its own byte
/// length, anything else is a dword count.
pub mod keys {
    pub const RESOURCE_INFO: u32 = 0x0000_02FF;
    pub const FILE_FORMAT_INFO: u32 = 0x0000_03FF;
    pub const ORIGINAL_BASE_ADDRESS: u32 = 0x0001_0001;
    pub const ENTRY_POINT: u32 = 0x0001_0100;
    pub const IMAGE_BASE_ADDRESS: u32 = 0x0001_0201;
    pub const IMPORT_LIBRARIES: u32 = 0x0001_03FF;
    pub const ORIGINAL_PE_NAME: u32 = 0x0001_83FF;
    pub const TLS_INFO: u32 = 0x0002_0104;
    pub const DEFAULT_STACK_SIZE: u32 = 0x0002_0200;
    pub const DEFAULT_FILESYSTEM_CACHE_SIZE: u32 = 0x0002_0301;
    pub const DEFAULT_HEAP_SIZE: u32 = 0x0002_0401;
    pub const SYSTEM_FLAGS: u32 = 0x0003_0000;
    pub const EXECUTION_INFO: u32 = 0x0004_0006;
    pub const TITLE_WORKSPACE_SIZE: u32 = 0x0004_0201;
    pub const GAME_RATINGS: u32 = 0x0004_0310;
}

pub const MAGIC: [u8; 4] = *b"XEX2";

/// Offset of the first optional header entry inside the XEX header.
pub const OPT_HEADER_TABLE_OFFSET: u32 = 0x18;

/// Size of the security info block this parser reads (up to and including the
/// page descriptor count).
pub const SECURITY_INFO_SIZE: u32 = 0x184;

const ENCRYPTION_NONE: u16 = 0;
const COMPRESSION_NONE: u16 = 0;
const COMPRESSION_BASIC: u16 = 1;
const COMPRESSION_NORMAL: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    BadMagic,
    Truncated(&'static str),
    Malformed(&'static str),
    Encrypted,
    LzxCompressed,
    UnsupportedCompression(u16),
    AddressOutOfImage(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic => write!(f, "not an XEX2 file"),
            Error::Truncated(what) => write!(f, "file truncated while reading {what}"),
            Error::Malformed(what) => write!(f, "malformed {what}"),
            Error::Encrypted => write!(f, "XEX is encrypted; supply a decrypted copy (xextool -e u -c u default.xex)"),
            Error::LzxCompressed => {
                write!(f, "XEX uses LZX compression; supply a decompressed copy (xextool -e u -c u default.xex)")
            }
            Error::UnsupportedCompression(kind) => write!(f, "unsupported XEX compression type {kind}"),
            Error::AddressOutOfImage(addr) => write!(f, "address {addr:#010X} is outside the image"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn be16(data: &[u8], off: usize, what: &'static str) -> Result<u16> {
    data.get(off..off + 2).map(|b| u16::from_be_bytes([b[0], b[1]])).ok_or(Error::Truncated(what))
}

fn be32(data: &[u8], off: usize, what: &'static str) -> Result<u32> {
    data.get(off..off + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])).ok_or(Error::Truncated(what))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptHeader {
    pub key: u32,
    /// Inline value, or the offset of the payload from the start of the file.
    pub value: u32,
}

/// Payload of an optional header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptValue<'a> {
    Inline(u32),
    Data { offset: u32, bytes: &'a [u8] },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityInfo {
    pub header_size: u32,
    pub image_size: u32,
    pub image_flags: u32,
    pub load_address: u32,
    pub import_table_count: u32,
    pub export_table: u32,
    pub region: u32,
    pub allowed_media_types: u32,
    pub page_descriptor_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsInfo {
    pub slot_count: u32,
    pub raw_data_address: u32,
    pub data_size: u32,
    pub raw_data_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionInfo {
    pub media_id: u32,
    pub version: u32,
    pub base_version: u32,
    pub title_id: u32,
    pub platform: u8,
    pub executable_type: u8,
    pub disc_number: u8,
    pub disc_count: u8,
    pub savegame_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportLibrary {
    pub name: String,
    pub id: u32,
    pub version: u32,
    pub min_version: u32,
    /// Guest addresses of the import records, in file order.
    pub records: Vec<u32>,
}

/// One resolved import. Every import has a record slot; function imports are
/// additionally followed by a 16-byte thunk that the code branches to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub library: String,
    pub ordinal: u16,
    pub record_addr: u32,
    pub thunk_addr: Option<u32>,
}

impl Import {
    pub fn is_function(&self) -> bool {
        self.thunk_addr.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct Xex {
    header: Vec<u8>,
    module_flags: u32,
    opt_headers: Vec<OptHeader>,
    security: SecurityInfo,
    image: Vec<u8>,
}

impl Xex {
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.get(0..4) != Some(&MAGIC[..]) {
            return Err(Error::BadMagic);
        }
        let module_flags = be32(data, 0x04, "XEX header")?;
        let header_size = be32(data, 0x08, "XEX header")? as usize;
        let security_offset = be32(data, 0x10, "XEX header")? as usize;
        let header_count = be32(data, 0x14, "XEX header")? as usize;
        if header_size > data.len() {
            return Err(Error::Truncated("XEX header"));
        }
        let header = data[..header_size].to_vec();

        let mut opt_headers = Vec::with_capacity(header_count);
        for i in 0..header_count {
            let off = OPT_HEADER_TABLE_OFFSET as usize + i * 8;
            opt_headers.push(OptHeader {
                key: be32(&header, off, "optional header table")?,
                value: be32(&header, off + 4, "optional header table")?,
            });
        }

        let s = security_offset;
        let security = SecurityInfo {
            header_size: be32(&header, s, "security info")?,
            image_size: be32(&header, s + 0x004, "security info")?,
            image_flags: be32(&header, s + 0x10C, "security info")?,
            load_address: be32(&header, s + 0x110, "security info")?,
            import_table_count: be32(&header, s + 0x128, "security info")?,
            export_table: be32(&header, s + 0x160, "security info")?,
            region: be32(&header, s + 0x178, "security info")?,
            allowed_media_types: be32(&header, s + 0x17C, "security info")?,
            page_descriptor_count: be32(&header, s + 0x180, "security info")?,
        };

        let mut xex = Xex { header, module_flags, opt_headers, security, image: Vec::new() };
        xex.image = xex.extract_image(&data[header_size..])?;
        Ok(xex)
    }

    fn extract_image(&self, payload: &[u8]) -> Result<Vec<u8>> {
        let image_size = self.security.image_size as usize;
        let Some(OptValue::Data { bytes: info, .. }) = self.opt_header(keys::FILE_FORMAT_INFO) else {
            // No file format info means a raw image.
            let mut image = vec![0u8; image_size];
            let n = payload.len().min(image_size);
            image[..n].copy_from_slice(&payload[..n]);
            return Ok(image);
        };
        let info_size = be32(info, 0, "file format info")? as usize;
        let encryption = be16(info, 4, "file format info")?;
        let compression = be16(info, 6, "file format info")?;
        if encryption != ENCRYPTION_NONE {
            return Err(Error::Encrypted);
        }
        match compression {
            COMPRESSION_NONE => {
                let mut image = vec![0u8; image_size];
                let n = payload.len().min(image_size);
                image[..n].copy_from_slice(&payload[..n]);
                Ok(image)
            }
            COMPRESSION_BASIC => {
                if info_size < 8 || info.len() < info_size {
                    return Err(Error::Malformed("basic compression info"));
                }
                let block_count = (info_size - 8) / 8;
                let mut image = Vec::with_capacity(image_size);
                let mut src = 0usize;
                for i in 0..block_count {
                    let data_size = be32(info, 8 + i * 8, "basic compression block")? as usize;
                    let zero_size = be32(info, 12 + i * 8, "basic compression block")? as usize;
                    let chunk = payload.get(src..src + data_size).ok_or(Error::Truncated("basic compression data"))?;
                    image.extend_from_slice(chunk);
                    image.resize(image.len() + zero_size, 0);
                    src += data_size;
                }
                image.resize(image_size.max(image.len()), 0);
                Ok(image)
            }
            COMPRESSION_NORMAL => Err(Error::LzxCompressed),
            other => Err(Error::UnsupportedCompression(other)),
        }
    }

    /// Raw XEX header bytes (everything before the image payload). Header-relative
    /// offsets returned by [`Xex::opt_header_field`] index into this buffer.
    pub fn header_bytes(&self) -> &[u8] {
        &self.header
    }

    pub fn module_flags(&self) -> u32 {
        self.module_flags
    }

    pub fn opt_headers(&self) -> &[OptHeader] {
        &self.opt_headers
    }

    pub fn security(&self) -> &SecurityInfo {
        &self.security
    }

    /// The loaded image (PE headers and sections), to be mapped at [`Xex::image_base`].
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    pub fn image_base(&self) -> u32 {
        match self.opt_header(keys::IMAGE_BASE_ADDRESS) {
            Some(OptValue::Inline(v)) => v,
            _ => self.security.load_address,
        }
    }

    pub fn image_size(&self) -> u32 {
        self.security.image_size
    }

    pub fn opt_header(&self, key: u32) -> Option<OptValue<'_>> {
        let entry = self.opt_headers.iter().find(|h| h.key == key)?;
        match key & 0xFF {
            0x00 | 0x01 => Some(OptValue::Inline(entry.value)),
            0xFF => {
                let off = entry.value as usize;
                let len = be32(&self.header, off, "optional header").ok()? as usize;
                let bytes = self.header.get(off..off + len)?;
                Some(OptValue::Data { offset: entry.value, bytes })
            }
            dwords => {
                let off = entry.value as usize;
                let bytes = self.header.get(off..off + dwords as usize * 4)?;
                Some(OptValue::Data { offset: entry.value, bytes })
            }
        }
    }

    /// What `RtlImageXexHeaderField` returns, relative to the header base: either
    /// the inline value (key low byte 0x00), or the header offset of the value
    /// (0x01: the opt-table slot itself; otherwise the payload offset).
    pub fn opt_header_field(&self, key: u32) -> Option<HeaderField> {
        let index = self.opt_headers.iter().position(|h| h.key == key)?;
        let entry = self.opt_headers[index];
        Some(match key & 0xFF {
            0x00 => HeaderField::Value(entry.value),
            0x01 => HeaderField::Offset(OPT_HEADER_TABLE_OFFSET + index as u32 * 8 + 4),
            _ => HeaderField::Offset(entry.value),
        })
    }

    pub fn entry_point(&self) -> Option<u32> {
        match self.opt_header(keys::ENTRY_POINT)? {
            OptValue::Inline(v) => Some(v),
            OptValue::Data { .. } => None,
        }
    }

    pub fn system_flags(&self) -> u32 {
        match self.opt_header(keys::SYSTEM_FLAGS) {
            Some(OptValue::Inline(v)) => v,
            _ => 0,
        }
    }

    pub fn default_stack_size(&self) -> Option<u32> {
        match self.opt_header(keys::DEFAULT_STACK_SIZE)? {
            OptValue::Inline(v) => Some(v),
            OptValue::Data { .. } => None,
        }
    }

    pub fn tls_info(&self) -> Option<TlsInfo> {
        let OptValue::Data { bytes, .. } = self.opt_header(keys::TLS_INFO)? else { return None };
        Some(TlsInfo {
            slot_count: be32(bytes, 0, "TLS info").ok()?,
            raw_data_address: be32(bytes, 4, "TLS info").ok()?,
            data_size: be32(bytes, 8, "TLS info").ok()?,
            raw_data_size: be32(bytes, 12, "TLS info").ok()?,
        })
    }

    pub fn execution_info(&self) -> Option<ExecutionInfo> {
        let OptValue::Data { bytes, .. } = self.opt_header(keys::EXECUTION_INFO)? else { return None };
        if bytes.len() < 24 {
            return None;
        }
        Some(ExecutionInfo {
            media_id: be32(bytes, 0, "execution info").ok()?,
            version: be32(bytes, 4, "execution info").ok()?,
            base_version: be32(bytes, 8, "execution info").ok()?,
            title_id: be32(bytes, 12, "execution info").ok()?,
            platform: bytes[16],
            executable_type: bytes[17],
            disc_number: bytes[18],
            disc_count: bytes[19],
            savegame_id: be32(bytes, 20, "execution info").ok()?,
        })
    }

    /// Big-endian dword at a guest address inside the image.
    pub fn read_image_u32(&self, addr: u32) -> Result<u32> {
        let off = addr.checked_sub(self.image_base()).ok_or(Error::AddressOutOfImage(addr))? as usize;
        be32(&self.image, off, "image").map_err(|_| Error::AddressOutOfImage(addr))
    }

    pub fn import_libraries(&self) -> Result<Vec<ImportLibrary>> {
        let Some(OptValue::Data { bytes, .. }) = self.opt_header(keys::IMPORT_LIBRARIES) else {
            return Ok(Vec::new());
        };
        let string_table_size = be32(bytes, 4, "import header")? as usize;
        let library_count = be32(bytes, 8, "import header")? as usize;

        let table = bytes.get(12..12 + string_table_size).ok_or(Error::Truncated("import string table"))?;
        let mut names = Vec::new();
        let mut off = 0usize;
        while off < table.len() && names.len() < library_count {
            let end = table[off..].iter().position(|&b| b == 0).map_or(table.len(), |p| off + p);
            names.push(String::from_utf8_lossy(&table[off..end]).into_owned());
            // Each string is NUL-terminated and padded to a dword boundary.
            off = (end + 1 + 3) & !3;
        }

        let mut libraries = Vec::with_capacity(library_count);
        let mut lib_off = 12 + string_table_size;
        for i in 0..library_count {
            let size = be32(bytes, lib_off, "import library")? as usize;
            let id = be32(bytes, lib_off + 0x18, "import library")?;
            let version = be32(bytes, lib_off + 0x1C, "import library")?;
            let min_version = be32(bytes, lib_off + 0x20, "import library")?;
            let name_index = be16(bytes, lib_off + 0x24, "import library")? as usize;
            let count = be16(bytes, lib_off + 0x26, "import library")? as usize;
            if size < 0x28 + count * 4 {
                return Err(Error::Malformed("import library size"));
            }
            let mut records = Vec::with_capacity(count);
            for r in 0..count {
                records.push(be32(bytes, lib_off + 0x28 + r * 4, "import records")?);
            }
            let name = names.get(name_index).or_else(|| names.get(i)).cloned().unwrap_or_default();
            libraries.push(ImportLibrary { name, id, version, min_version, records });
            lib_off += size;
        }
        Ok(libraries)
    }

    /// Resolves import records against the image. Record value layout (after
    /// byte swap): bits 24..31 type (0 = record slot, 1 = thunk), bits 16..23
    /// library hint, bits 0..15 ordinal. A function import is a slot followed by
    /// a thunk with the same ordinal; a lone slot is a variable import.
    pub fn imports(&self) -> Result<Vec<Import>> {
        let mut out: Vec<Import> = Vec::new();
        for library in self.import_libraries()? {
            let start = out.len();
            for &record_addr in &library.records {
                let value = self.read_image_u32(record_addr)?;
                let kind = value >> 24;
                let ordinal = (value & 0xFFFF) as u16;
                match kind {
                    0 => out.push(Import { library: library.name.clone(), ordinal, record_addr, thunk_addr: None }),
                    1 => {
                        let last = out[start..].last_mut();
                        match last {
                            Some(prev) if prev.ordinal == ordinal && prev.thunk_addr.is_none() => {
                                prev.thunk_addr = Some(record_addr);
                            }
                            _ => return Err(Error::Malformed("import thunk without a matching record")),
                        }
                    }
                    _ => return Err(Error::Malformed("import record type")),
                }
            }
        }
        Ok(out)
    }

    /// Function extents from the `.pdata` exception directory, sorted by start
    /// address: `(start, size_in_bytes)`. Leaf functions without unwind data
    /// are absent.
    pub fn pdata_functions(&self) -> Result<Vec<(u32, u32)>> {
        let Some(pdata) = self.sections()?.into_iter().find(|s| s.name == ".pdata") else {
            return Ok(Vec::new());
        };
        let start = pdata.virtual_address as usize;
        let bytes = self.image.get(start..start + pdata.virtual_size as usize).ok_or(Error::Truncated(".pdata"))?;
        let mut out: Vec<(u32, u32)> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .filter_map(|e| {
                let begin = u32::from_be_bytes([e[0], e[1], e[2], e[3]]);
                let data = u32::from_be_bytes([e[4], e[5], e[6], e[7]]);
                // Data: PrologLength:8 | FunctionLength:22 (in instructions) | flags:2
                (begin != 0).then_some((begin, ((data >> 8) & 0x3F_FFFF) * 4))
            })
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    /// PE section table of the image (little-endian PE headers).
    pub fn sections(&self) -> Result<Vec<Section>> {
        let img = &self.image;
        let le32 = |off: usize| -> Result<u32> {
            img.get(off..off + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or(Error::Truncated("PE headers"))
        };
        let le16 = |off: usize| -> Result<u16> {
            img.get(off..off + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or(Error::Truncated("PE headers"))
        };
        let nt = le32(0x3C)? as usize;
        if img.get(nt..nt + 4) != Some(&b"PE\0\0"[..]) {
            return Err(Error::Malformed("PE signature"));
        }
        let count = le16(nt + 6)? as usize;
        let optional_size = le16(nt + 20)? as usize;
        let table = nt + 24 + optional_size;
        let mut sections = Vec::with_capacity(count);
        for i in 0..count {
            let off = table + i * 40;
            let raw_name = img.get(off..off + 8).ok_or(Error::Truncated("PE section table"))?;
            let name_len = raw_name.iter().position(|&b| b == 0).unwrap_or(8);
            sections.push(Section {
                name: String::from_utf8_lossy(&raw_name[..name_len]).into_owned(),
                virtual_size: le32(off + 8)?,
                virtual_address: le32(off + 12)?,
                characteristics: le32(off + 36)?,
            });
        }
        Ok(sections)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderField {
    Value(u32),
    Offset(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub virtual_size: u32,
    /// Relative to the image base.
    pub virtual_address: u32,
    pub characteristics: u32,
}

impl Section {
    pub const CODE: u32 = 0x0000_0020;

    pub fn is_code(&self) -> bool {
        self.characteristics & Self::CODE != 0
    }
}
