use xex::builder::{
    IMAGE_SCN_CNT_CODE, IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, ImportLibrarySpec,
    PeSection, XexBuilder, build_pe_image,
};
use xex::{Error, HeaderField, TlsInfo, Xex, keys};

const BASE: u32 = 0x8200_0000;

fn sample(basic_compression: bool) -> Vec<u8> {
    // .rdata at RVA 0x1000 holds the import records:
    //   0x1000: slot  (type 0, ordinal 0x0C)  -> variable import
    //   0x1004: slot  (type 0, ordinal 0xCC)  -> function import record
    // .text at RVA 0x2000 holds the thunk for 0xCC (type 1).
    let mut rdata = vec![0u8; 0x10];
    rdata[0..4].copy_from_slice(&0x0000_000Cu32.to_be_bytes());
    rdata[4..8].copy_from_slice(&0x0000_00CCu32.to_be_bytes());
    let mut text = vec![0u8; 0x20];
    text[0..4].copy_from_slice(&0x0100_00CCu32.to_be_bytes());
    text[0x10..0x14].copy_from_slice(&0x4E80_0020u32.to_be_bytes());
    let image = build_pe_image(
        &[
            PeSection {
                name: ".rdata".into(),
                rva: 0x1000,
                virtual_size: 0x10,
                characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
                data: rdata,
            },
            PeSection {
                name: ".text".into(),
                rva: 0x2000,
                virtual_size: 0x20,
                characteristics: IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ,
                data: text,
            },
        ],
        0x10000,
    );
    XexBuilder {
        image,
        load_address: BASE,
        entry_point: BASE + 0x2010,
        module_flags: 1,
        imports: vec![ImportLibrarySpec {
            name: "xboxkrnl.exe".into(),
            id: 0,
            version: 0x2000_4400,
            min_version: 0x2000_0000,
            records: vec![BASE + 0x1000, BASE + 0x1004, BASE + 0x2000],
        }],
        tls: Some(TlsInfo { slot_count: 64, raw_data_address: BASE + 0x1008, data_size: 0x20, raw_data_size: 0x8 }),
        system_flags: Some(0x0000_0400),
        default_stack_size: Some(0x40000),
        execution_info: None,
        basic_compression,
    }
    .build()
}

#[test]
fn parses_built_xex() {
    for basic in [false, true] {
        let xex = Xex::parse(&sample(basic)).expect("parse");
        assert_eq!(xex.image_base(), BASE);
        assert_eq!(xex.image_size(), 0x10000);
        assert_eq!(xex.image().len(), 0x10000);
        assert_eq!(xex.entry_point(), Some(BASE + 0x2010));
        assert_eq!(xex.system_flags(), 0x400);
        assert_eq!(xex.default_stack_size(), Some(0x40000));
        assert_eq!(xex.read_image_u32(BASE + 0x2010).unwrap(), 0x4E80_0020);
        let tls = xex.tls_info().unwrap();
        assert_eq!(tls.slot_count, 64);
        assert_eq!(tls.raw_data_size, 8);

        let sections = xex.sections().unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[1].name, ".text");
        assert!(sections[1].is_code());
        assert!(!sections[0].is_code());
    }
}

#[test]
fn pairs_import_records_with_thunks() {
    let xex = Xex::parse(&sample(false)).unwrap();
    let imports = xex.imports().unwrap();
    assert_eq!(imports.len(), 2);
    assert_eq!(imports[0].library, "xboxkrnl.exe");
    assert_eq!(imports[0].ordinal, 0x0C);
    assert!(!imports[0].is_function());
    assert_eq!(imports[1].ordinal, 0xCC);
    assert_eq!(imports[1].record_addr, BASE + 0x1004);
    assert_eq!(imports[1].thunk_addr, Some(BASE + 0x2000));
}

#[test]
fn header_fields_resolve_like_rtl_image_xex_header_field() {
    let xex = Xex::parse(&sample(false)).unwrap();
    assert_eq!(xex.opt_header_field(keys::SYSTEM_FLAGS), Some(HeaderField::Value(0x400)));
    // 0x01-class keys resolve to the opt-table slot holding the value.
    let Some(HeaderField::Offset(off)) = xex.opt_header_field(keys::IMAGE_BASE_ADDRESS) else {
        panic!("expected offset")
    };
    let hdr = xex.header_bytes();
    let off = off as usize;
    assert_eq!(u32::from_be_bytes(hdr[off..off + 4].try_into().unwrap()), BASE);
    let Some(HeaderField::Offset(tls_off)) = xex.opt_header_field(keys::TLS_INFO) else { panic!("expected offset") };
    let tls_off = tls_off as usize;
    assert_eq!(u32::from_be_bytes(hdr[tls_off..tls_off + 4].try_into().unwrap()), 64);
    assert_eq!(xex.opt_header_field(keys::GAME_RATINGS), None);
}

#[test]
fn rejects_encrypted_and_lzx() {
    let mut data = sample(false);
    let xex = Xex::parse(&data).unwrap();
    let Some(xex::OptValue::Data { offset, .. }) = xex.opt_header(keys::FILE_FORMAT_INFO) else {
        panic!("no file format info")
    };
    let off = offset as usize;
    data[off + 4..off + 6].copy_from_slice(&1u16.to_be_bytes());
    assert_eq!(Xex::parse(&data).unwrap_err(), Error::Encrypted);
    data[off + 4..off + 6].copy_from_slice(&0u16.to_be_bytes());
    data[off + 6..off + 8].copy_from_slice(&2u16.to_be_bytes());
    assert_eq!(Xex::parse(&data).unwrap_err(), Error::LzxCompressed);
    assert_eq!(Xex::parse(b"XEX1....").unwrap_err(), Error::BadMagic);
}
