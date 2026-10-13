//! Writers for minimal PE images and XEX2 containers.
//!
//! These exist so the toolchain (XenonRecomp, shim generation, linking, boot)
//! can be exercised end to end with synthetic titles instead of game files.
//! The output follows the same layout rules the parser and XenonRecomp's
//! loader read: little-endian PE headers, big-endian XEX headers.

use crate::{ExecutionInfo, TlsInfo, keys};

pub const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
pub const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
pub const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
pub const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
pub const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

const PE_HEADER_OFFSET: usize = 0x80;
const OPTIONAL_HEADER_SIZE: usize = 0xE0;
const MACHINE_POWERPCBE: u16 = 0x01F2;

#[derive(Debug, Clone)]
pub struct PeSection {
    /// At most 7 bytes; XenonRecomp reads section names as C strings.
    pub name: String,
    pub rva: u32,
    pub virtual_size: u32,
    pub characteristics: u32,
    pub data: Vec<u8>,
}

/// Lays out a PE image of `image_size` bytes with the given sections copied to
/// their RVAs.
pub fn build_pe_image(sections: &[PeSection], image_size: u32) -> Vec<u8> {
    let mut img = vec![0u8; image_size as usize];
    img[0..2].copy_from_slice(b"MZ");
    img[0x3C..0x40].copy_from_slice(&(PE_HEADER_OFFSET as u32).to_le_bytes());

    let nt = PE_HEADER_OFFSET;
    img[nt..nt + 4].copy_from_slice(b"PE\0\0");
    let fh = nt + 4;
    img[fh..fh + 2].copy_from_slice(&MACHINE_POWERPCBE.to_le_bytes());
    img[fh + 2..fh + 4].copy_from_slice(&(sections.len() as u16).to_le_bytes());
    img[fh + 16..fh + 18].copy_from_slice(&(OPTIONAL_HEADER_SIZE as u16).to_le_bytes());
    img[fh + 18..fh + 20].copy_from_slice(&0x0102u16.to_le_bytes());
    let oh = fh + 20;
    img[oh..oh + 2].copy_from_slice(&0x010Bu16.to_le_bytes());
    img[oh + 56..oh + 60].copy_from_slice(&image_size.to_le_bytes());

    let table = oh + OPTIONAL_HEADER_SIZE;
    for (i, s) in sections.iter().enumerate() {
        assert!(s.name.len() < 8, "section name {:?} must be shorter than 8 bytes", s.name);
        let off = table + i * 40;
        img[off..off + s.name.len()].copy_from_slice(s.name.as_bytes());
        img[off + 8..off + 12].copy_from_slice(&s.virtual_size.to_le_bytes());
        img[off + 12..off + 16].copy_from_slice(&s.rva.to_le_bytes());
        img[off + 16..off + 20].copy_from_slice(&(s.data.len() as u32).to_le_bytes());
        img[off + 20..off + 24].copy_from_slice(&s.rva.to_le_bytes());
        img[off + 36..off + 40].copy_from_slice(&s.characteristics.to_le_bytes());
        let start = s.rva as usize;
        img[start..start + s.data.len()].copy_from_slice(&s.data);
    }
    img
}

#[derive(Debug, Clone)]
pub struct ImportLibrarySpec {
    pub name: String,
    pub id: u32,
    pub version: u32,
    pub min_version: u32,
    /// Guest addresses of the record slots/thunks; their values live in the image.
    pub records: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct XexBuilder {
    pub image: Vec<u8>,
    pub load_address: u32,
    pub entry_point: u32,
    pub module_flags: u32,
    pub imports: Vec<ImportLibrarySpec>,
    pub tls: Option<TlsInfo>,
    pub system_flags: Option<u32>,
    pub default_stack_size: Option<u32>,
    pub execution_info: Option<ExecutionInfo>,
    /// Encode the payload with "basic" compression (zero-run elision).
    pub basic_compression: bool,
}

fn push_be32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn align(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

impl XexBuilder {
    pub fn build(&self) -> Vec<u8> {
        const BLOCK: usize = 0x8000;

        let mut opt: Vec<(u32, Payload)> = Vec::new();
        opt.push((keys::ENTRY_POINT, Payload::Inline(self.entry_point)));
        opt.push((keys::IMAGE_BASE_ADDRESS, Payload::Inline(self.load_address)));
        if let Some(flags) = self.system_flags {
            opt.push((keys::SYSTEM_FLAGS, Payload::Inline(flags)));
        }
        if let Some(size) = self.default_stack_size {
            opt.push((keys::DEFAULT_STACK_SIZE, Payload::Inline(size)));
        }

        // Payload (the image as stored after the header) and file format info.
        let mut ffi = Vec::new();
        let mut payload = Vec::new();
        if self.basic_compression {
            let mut blocks = Vec::new();
            for chunk in self.image.chunks(BLOCK) {
                let data_len = chunk.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
                payload.extend_from_slice(&chunk[..data_len]);
                blocks.push((data_len as u32, (chunk.len() - data_len) as u32));
            }
            push_be32(&mut ffi, 8 + 8 * blocks.len() as u32);
            ffi.extend_from_slice(&0u16.to_be_bytes());
            ffi.extend_from_slice(&1u16.to_be_bytes());
            for (data, zero) in blocks {
                push_be32(&mut ffi, data);
                push_be32(&mut ffi, zero);
            }
        } else {
            payload.extend_from_slice(&self.image);
            push_be32(&mut ffi, 8);
            ffi.extend_from_slice(&0u16.to_be_bytes());
            ffi.extend_from_slice(&0u16.to_be_bytes());
        }
        opt.push((keys::FILE_FORMAT_INFO, Payload::Data(ffi)));

        if !self.imports.is_empty() {
            let mut strings = Vec::new();
            for lib in &self.imports {
                strings.extend_from_slice(lib.name.as_bytes());
                strings.push(0);
                strings.resize(align(strings.len(), 4), 0);
            }
            let mut libs = Vec::new();
            for (i, lib) in self.imports.iter().enumerate() {
                push_be32(&mut libs, 0x28 + 4 * lib.records.len() as u32);
                libs.extend_from_slice(&[0u8; 0x14]);
                push_be32(&mut libs, lib.id);
                push_be32(&mut libs, lib.version);
                push_be32(&mut libs, lib.min_version);
                libs.extend_from_slice(&(i as u16).to_be_bytes());
                libs.extend_from_slice(&(lib.records.len() as u16).to_be_bytes());
                for &r in &lib.records {
                    push_be32(&mut libs, r);
                }
            }
            let mut data = Vec::new();
            push_be32(&mut data, (12 + strings.len() + libs.len()) as u32);
            push_be32(&mut data, strings.len() as u32);
            push_be32(&mut data, self.imports.len() as u32);
            data.extend_from_slice(&strings);
            data.extend_from_slice(&libs);
            opt.push((keys::IMPORT_LIBRARIES, Payload::Data(data)));
        }

        if let Some(tls) = self.tls {
            let mut data = Vec::new();
            push_be32(&mut data, tls.slot_count);
            push_be32(&mut data, tls.raw_data_address);
            push_be32(&mut data, tls.data_size);
            push_be32(&mut data, tls.raw_data_size);
            opt.push((keys::TLS_INFO, Payload::Data(data)));
        }

        if let Some(info) = self.execution_info {
            let mut data = Vec::new();
            push_be32(&mut data, info.media_id);
            push_be32(&mut data, info.version);
            push_be32(&mut data, info.base_version);
            push_be32(&mut data, info.title_id);
            data.extend_from_slice(&[info.platform, info.executable_type, info.disc_number, info.disc_count]);
            push_be32(&mut data, info.savegame_id);
            opt.push((keys::EXECUTION_INFO, Payload::Data(data)));
        }

        opt.sort_by_key(|(k, _)| *k);

        // Header: fixed fields, opt table, security info, then opt payloads.
        let table_end = 0x18 + opt.len() * 8;
        let security_offset = align(table_end, 8);
        let mut cursor = security_offset + crate::SECURITY_INFO_SIZE as usize;
        let mut entries = Vec::with_capacity(opt.len());
        let mut blobs: Vec<(usize, Vec<u8>)> = Vec::new();
        for (key, payload) in opt {
            match payload {
                Payload::Inline(v) => entries.push((key, v)),
                Payload::Data(bytes) => {
                    cursor = align(cursor, 8);
                    entries.push((key, cursor as u32));
                    let len = bytes.len();
                    blobs.push((cursor, bytes));
                    cursor += len;
                }
            }
        }
        let header_size = align(cursor, 0x1000);

        let mut out = vec![0u8; header_size];
        out[0..4].copy_from_slice(&crate::MAGIC);
        out[4..8].copy_from_slice(&self.module_flags.to_be_bytes());
        out[8..12].copy_from_slice(&(header_size as u32).to_be_bytes());
        out[16..20].copy_from_slice(&(security_offset as u32).to_be_bytes());
        out[20..24].copy_from_slice(&(entries.len() as u32).to_be_bytes());
        for (i, (key, value)) in entries.iter().enumerate() {
            let off = 0x18 + i * 8;
            out[off..off + 4].copy_from_slice(&key.to_be_bytes());
            out[off + 4..off + 8].copy_from_slice(&value.to_be_bytes());
        }
        let s = security_offset;
        out[s..s + 4].copy_from_slice(&crate::SECURITY_INFO_SIZE.to_be_bytes());
        out[s + 4..s + 8].copy_from_slice(&(self.image.len() as u32).to_be_bytes());
        out[s + 0x110..s + 0x114].copy_from_slice(&self.load_address.to_be_bytes());
        let import_count: usize = self.imports.len();
        out[s + 0x128..s + 0x12C].copy_from_slice(&(import_count as u32).to_be_bytes());
        out[s + 0x178..s + 0x17C].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        for (off, bytes) in blobs {
            out[off..off + bytes.len()].copy_from_slice(&bytes);
        }
        out.extend_from_slice(&payload);
        out
    }
}

enum Payload {
    Inline(u32),
    Data(Vec<u8>),
}
