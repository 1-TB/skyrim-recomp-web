mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use common::{
    NTCLOSE_SLOT, NTCLOSE_THUNK, RECORDER, TIMESTAMP_SLOT, UNKNOWN_VAR_SLOT, guest, kernel, kernel_with,
    recorded_with_arg1, scratch,
};
use kernel_hle::dispatch::{ModuleId, StubConfig};
use kernel_hle::{all_exports, status};

#[test]
fn unimplemented_imports_return_configured_stub_values() {
    let mut stubs = StubConfig { default_return: 0xC000_0002, ..Default::default() };
    let (key, value) = StubConfig::parse_override("xam:0x123=0x7").unwrap();
    stubs.overrides.insert(key, value);
    let k = kernel_with(stubs);
    let mut ctx = guest(k);

    ctx.set_gpr(3, 0x1111);
    k.dispatch(ModuleId::XBOXKRNL, 0x3FE, &mut ctx);
    assert_eq!(ctx.gpr(3), 0xC000_0002);
    k.dispatch(ModuleId::XAM, 0x123, &mut ctx);
    assert_eq!(ctx.gpr(3), 7);
    k.dispatch(ModuleId::XAM, 0x123, &mut ctx);

    let stats = k.import_stats();
    let xam = stats.iter().find(|s| s.0 == ModuleId::XAM && s.1 == 0x123).unwrap();
    assert_eq!(xam.3.calls, 2);
    assert!(!xam.3.implemented);
    assert_eq!(xam.2, "xam_123");
    // Names from the build-time table are used when known.
    assert_eq!(k.import_name(ModuleId::XBOXKRNL, 0x3FF), "SyntheticMissingVariable");
}

#[test]
fn implemented_imports_are_counted() {
    let k = kernel();
    let mut ctx = guest(k);
    k.call_export("KeQueryPerformanceFrequency", &mut ctx, &[]);
    assert_eq!(ctx.gpr(3), 50_000_000);
    let s = k.import_stats().into_iter().find(|s| s.2 == "KeQueryPerformanceFrequency").unwrap();
    assert!(s.3.implemented);
    assert_eq!(s.3.calls, 1);
}

#[test]
fn import_slots_are_patched_at_boot() {
    let k = kernel();
    let mem = k.mem();
    assert_eq!(mem.read_u32(TIMESTAMP_SLOT), k.vars.ke_time_stamp_bundle);
    assert_eq!(mem.read_u32(NTCLOSE_SLOT), NTCLOSE_THUNK);
    let placeholder = mem.read_u32(UNKNOWN_VAR_SLOT);
    assert_ne!(placeholder, 0);
    assert_eq!(mem.read_u32(placeholder), 0);
    // The bundle's clock is live.
    let t0 = mem.read_u64(k.vars.ke_time_stamp_bundle + 8);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(mem.read_u64(k.vars.ke_time_stamp_bundle + 8) > t0);
}

/// Cross-checks every registered (module, ordinal, name) against XenonRecomp's
/// copy of the export tables. Set XENON_RECOMP_DIR to the XenonRecomp checkout.
#[test]
fn export_ordinals_match_export_tables() {
    let Some(dir) = std::env::var_os("XENON_RECOMP_DIR").map(PathBuf::from) else {
        eprintln!("XENON_RECOMP_DIR not set; skipping export table cross-check");
        return;
    };
    let mut table: HashMap<(u32, u16), String> = HashMap::new();
    for (file, module) in [("xboxkrnl_table.inc", 0u32), ("xam_table.inc", 1u32)] {
        let text = std::fs::read_to_string(dir.join("XenonUtils/xbox").join(file)).expect("read export table");
        for line in text.lines().filter(|l| l.starts_with("XE_EXPORT(")) {
            let parts: Vec<&str> = line["XE_EXPORT(".len()..].split(',').map(str::trim).collect();
            let ordinal = u16::from_str_radix(parts[1].trim_start_matches("0x"), 16).unwrap();
            table.insert((module, ordinal), parts[2].to_string());
        }
    }
    let mut bad = Vec::new();
    for e in all_exports() {
        match table.get(&(e.module.0, e.ordinal)) {
            Some(name) if name == e.name => {}
            other => bad.push(format!("{:?}: table has {:?}", e, other)),
        }
    }
    assert!(bad.is_empty(), "mismatched exports:\n{}", bad.join("\n"));
}

#[test]
fn virtual_memory_round_trip() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x1000);
    let (base_ptr, size_ptr, info) = (io, io + 4, io + 0x10);
    k.mem().write_u32(base_ptr, 0);
    k.mem().write_u32(size_ptr, 0x1234);
    // MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
    assert_eq!(
        k.call_export("NtAllocateVirtualMemory", &mut ctx, &[base_ptr, size_ptr, 0x3000, 4, 0]) as u32,
        status::SUCCESS
    );
    let base = k.mem().read_u32(base_ptr);
    assert_eq!(k.mem().read_u32(size_ptr), 0x10000, "unspecified base rounds size to 64 KiB");
    k.mem().write_u32(base + 0x100, 42);

    assert_eq!(k.call_export("NtQueryVirtualMemory", &mut ctx, &[base, info, 0]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_u32(info), base);
    assert_eq!(k.mem().read_u32(info + 12), 0x10000);
    assert_eq!(k.mem().read_u32(info + 16), 0x1000, "MEM_COMMIT");
    assert_eq!(k.mem().read_u32(info + 20), 4);
    assert_eq!(k.call_export("MmQueryAllocationSize", &mut ctx, &[base]), 0x10000);

    let old = io + 0x30;
    k.mem().write_u32(base_ptr, base);
    k.mem().write_u32(size_ptr, 0x1000);
    assert_eq!(
        k.call_export("NtProtectVirtualMemory", &mut ctx, &[base_ptr, size_ptr, 2, old, 0]) as u32,
        status::SUCCESS
    );
    assert_eq!(k.mem().read_u32(old), 4);
    assert_eq!(k.call_export("MmQueryAddressProtect", &mut ctx, &[base]), 2);

    // Committing an unaligned range commits every page it touches.
    k.mem().write_u32(base_ptr, 0);
    k.mem().write_u32(size_ptr, 0x10000);
    k.call_export("NtAllocateVirtualMemory", &mut ctx, &[base_ptr, size_ptr, 0x2000, 4, 0]);
    let reserved = k.mem().read_u32(base_ptr);
    k.mem().write_u32(base_ptr, reserved + 0x800);
    k.mem().write_u32(size_ptr, 0x1000);
    assert_eq!(
        k.call_export("NtAllocateVirtualMemory", &mut ctx, &[base_ptr, size_ptr, 0x1000, 4, 0]) as u32,
        status::SUCCESS
    );
    assert_eq!((k.mem().read_u32(base_ptr), k.mem().read_u32(size_ptr)), (reserved, 0x2000));
    assert!(k.rt.space.is_committed(reserved + 0x1000));
    assert!(!k.rt.space.is_committed(reserved + 0x2000));

    k.mem().write_u32(base_ptr, base);
    k.mem().write_u32(size_ptr, 0);
    assert_eq!(
        k.call_export("NtFreeVirtualMemory", &mut ctx, &[base_ptr, size_ptr, 0x8000, 0]) as u32,
        status::SUCCESS
    );
    assert_eq!(k.mem().read_u32(size_ptr), 0x10000);
    k.call_export("NtQueryVirtualMemory", &mut ctx, &[base, info, 0]);
    assert_eq!(k.mem().read_u32(info + 16), 0x10000, "MEM_FREE");
    assert_eq!(
        k.call_export("NtFreeVirtualMemory", &mut ctx, &[base_ptr, size_ptr, 0x8000, 0]) as u32,
        status::MEMORY_NOT_ALLOCATED
    );
}

#[test]
fn physical_memory_and_pool() {
    let k = kernel();
    let mut ctx = guest(k);
    // PAGE_READWRITE | MEM_LARGE_PAGES -> 64 KiB view at 0xA0000000.
    let p = k.call_export("MmAllocatePhysicalMemoryEx", &mut ctx, &[0, 0x20000, 0x2000_0004, 0, u32::MAX, 0]) as u32;
    assert!((0xA000_0000..0xC000_0000).contains(&p), "{p:#X}");
    assert_eq!(p % 0x10000, 0);
    assert_eq!(k.call_export("MmGetPhysicalAddress", &mut ctx, &[p]) as u32, p - 0xA000_0000);
    k.mem().write_u32(p + 0x1FFFC, 1);
    k.call_export("MmFreePhysicalMemory", &mut ctx, &[0, p]);
    assert_eq!(
        k.call_export("MmAllocatePhysicalMemoryEx", &mut ctx, &[0, 0x1000, 0x1, 0, u32::MAX, 0]),
        0,
        "NOACCESS rejected"
    );

    let small = k.call_export("ExAllocatePoolWithTag", &mut ctx, &[0x40, u32::from_be_bytes(*b"Test")]) as u32;
    assert_ne!(small, 0);
    assert_eq!(k.mem().read_u32(small - 4), u32::from_be_bytes(*b"Test"));
    let big = k.call_export("ExAllocatePool", &mut ctx, &[0x5000]) as u32;
    assert_eq!(big % 0x1000, 0);
    k.call_export("ExFreePool", &mut ctx, &[small]);
    k.call_export("ExFreePool", &mut ctx, &[big]);
}

/// Builds `OBJECT_ATTRIBUTES` with an ANSI path at `at`.
fn object_attributes(k: &kernel_hle::Kernel, at: u32, path: &str) -> u32 {
    let mem = k.mem();
    mem.write_bytes(at + 0x40, path.as_bytes());
    mem.write_u16(at + 0x10, path.len() as u16);
    mem.write_u16(at + 0x12, path.len() as u16 + 1);
    mem.write_u32(at + 0x14, at + 0x40);
    mem.write_u32(at, 0);
    mem.write_u32(at + 4, at + 0x10);
    mem.write_u32(at + 8, 0);
    at
}

#[test]
fn files_open_read_and_query_through_the_vfs() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x4000);
    let (handle_ptr, iosb, buf, info, offset) = (io, io + 8, io + 0x100, io + 0x200, io + 0x300);
    let attrs = object_attributes(k, io + 0x1000, "game:\\data\\SKYRIM.ESM");

    // GENERIC_READ, FILE_OPEN, FILE_SYNCHRONOUS_IO_NONALERT | FILE_NON_DIRECTORY_FILE
    // (9th argument on the stack)
    let s = k.call_export("NtCreateFile", &mut ctx, &[handle_ptr, 0x8000_0000, attrs, iosb, 0, 0, 1, 1, 0x60]) as u32;
    assert_eq!(s, status::SUCCESS);
    assert_eq!(k.mem().read_u32(iosb + 4), 1, "FILE_OPENED");
    let h = k.mem().read_u32(handle_ptr);

    k.mem().write_i64(offset, 5);
    assert_eq!(k.call_export("NtReadFile", &mut ctx, &[h, 0, 0, 0, iosb, buf, 4, offset]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_vec(buf, 4), b"0123");
    assert_eq!(k.mem().read_u32(iosb + 4), 4);
    // Continue from the file pointer (no offset).
    assert_eq!(k.call_export("NtReadFile", &mut ctx, &[h, 0, 0, 0, iosb, buf, 100, 0]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_u32(iosb + 4), 6);
    assert_eq!(k.mem().read_vec(buf, 6), b"456789");
    assert_eq!(k.call_export("NtReadFile", &mut ctx, &[h, 0, 0, 0, iosb, buf, 100, 0]) as u32, status::END_OF_FILE);

    // FileStandardInformation
    assert_eq!(k.call_export("NtQueryInformationFile", &mut ctx, &[h, iosb, info, 0x18, 5]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_u64(info + 8), 15, "end of file");
    // FilePositionInformation
    assert_eq!(k.call_export("NtQueryInformationFile", &mut ctx, &[h, iosb, info, 8, 14]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_u64(info), 15);
    assert_eq!(
        k.call_export("NtQueryInformationFile", &mut ctx, &[h, iosb, info, 4, 5]) as u32,
        status::INFO_LENGTH_MISMATCH
    );
    // An I/O APC routine's low bit is a flag: the APC runs RECORDER(context,
    // iosb) at the next alertable wait.
    const APC_CONTEXT: u32 = 0xA9C0_0001;
    k.mem().write_i64(offset, 0);
    let r = k.call_export("NtReadFile", &mut ctx, &[h, 0, RECORDER | 1, APC_CONTEXT, iosb, buf, 4, offset]);
    assert_eq!(r as u32, status::SUCCESS);
    assert!(recorded_with_arg1(iosb).is_empty(), "not delivered before an alertable wait");
    let delay = offset + 8;
    k.mem().write_i64(delay, -10_000);
    k.call_export("KeDelayExecutionThread", &mut ctx, &[1, 1, delay]);
    let calls = recorded_with_arg1(iosb);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].args[0], APC_CONTEXT);
    assert_eq!(k.call_export("NtClose", &mut ctx, &[h]) as u32, status::SUCCESS);

    // Without FILE_SYNCHRONOUS_IO_*, a read completes but reports
    // STATUS_PENDING; the real result is in the IO_STATUS_BLOCK.
    let s = k.call_export("NtCreateFile", &mut ctx, &[handle_ptr, 0x8000_0000, attrs, iosb, 0, 0, 1, 1, 0x40]) as u32;
    assert_eq!(s, status::SUCCESS);
    let ah = k.mem().read_u32(handle_ptr);
    k.mem().write_i64(offset, 0);
    assert_eq!(k.call_export("NtReadFile", &mut ctx, &[ah, 0, 0, 0, iosb, buf, 4, offset]) as u32, status::PENDING);
    assert_eq!((k.mem().read_u32(iosb), k.mem().read_u32(iosb + 4)), (status::SUCCESS, 4));
    assert_eq!(k.mem().read_vec(buf, 4), b"TES4");
    k.call_export("NtClose", &mut ctx, &[ah]);

    let missing = object_attributes(k, io + 0x1000, "game:\\Data\\Missing.esp");
    assert_eq!(
        k.call_export("NtCreateFile", &mut ctx, &[handle_ptr, 0x8000_0000, missing, iosb, 0, 0, 1, 1, 0]) as u32,
        status::OBJECT_NAME_NOT_FOUND
    );
    let write = object_attributes(k, io + 0x1000, "d:\\Data\\new.txt");
    // GENERIC_WRITE, FILE_CREATE on the read-only disc
    assert_eq!(
        k.call_export("NtCreateFile", &mut ctx, &[handle_ptr, 0x4000_0000, write, iosb, 0, 0, 0, 2, 0]) as u32,
        status::ACCESS_DENIED
    );

    let full = object_attributes(k, io + 0x1000, "\\??\\game:\\Data\\Skyrim.esm");
    assert_eq!(k.call_export("NtQueryFullAttributesFile", &mut ctx, &[full, info]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_u64(info + 0x28), 15);
    assert_eq!(k.mem().read_u32(info + 0x30) & 0x10, 0, "not a directory");

    // RootDirectory = ObDosDevices (0xFFFFFFFD): the name is a DOS-device path.
    let dos = object_attributes(k, io + 0x1000, "game:\\Data\\Skyrim.esm");
    k.mem().write_u32(dos, 0xFFFF_FFFD);
    assert_eq!(k.call_export("NtQueryFullAttributesFile", &mut ctx, &[dos, info]) as u32, status::SUCCESS);
    assert_eq!(
        k.call_export("NtCreateFile", &mut ctx, &[handle_ptr, 0x8000_0000, dos, iosb, 0, 0, 1, 1, 0x40]) as u32,
        status::SUCCESS
    );
    assert_eq!(k.call_export("NtClose", &mut ctx, &[k.mem().read_u32(handle_ptr)]) as u32, status::SUCCESS);
}

#[test]
fn directory_enumeration_with_pattern() {
    let k = kernel();
    let mut ctx = guest(k);
    let io = scratch(k, 0x4000);
    let (handle_ptr, iosb, info, pattern) = (io, io + 8, io + 0x2000, io + 0x100);
    let attrs = object_attributes(k, io + 0x1000, "game:\\Data");
    // FILE_DIRECTORY_FILE
    assert_eq!(
        k.call_export("NtOpenFile", &mut ctx, &[handle_ptr, 0x8000_0000, attrs, iosb, 1, 1]) as u32,
        status::SUCCESS
    );
    let h = k.mem().read_u32(handle_ptr);
    k.mem().write_bytes(pattern + 0x10, b"*.esm");
    k.mem().write_u16(pattern, 5);
    k.mem().write_u16(pattern + 2, 6);
    k.mem().write_u32(pattern + 4, pattern + 0x10);

    // One entry fits in 0x48 bytes; collect them one call at a time.
    let mut names = Vec::new();
    loop {
        let s = k.call_export("NtQueryDirectoryFile", &mut ctx, &[h, 0, 0, 0, iosb, info, 0x50, pattern, 0]) as u32;
        if s == status::NO_MORE_FILES {
            break;
        }
        assert_eq!(s, status::SUCCESS);
        let len = k.mem().read_u32(info + 0x3C);
        names.push(String::from_utf8(k.mem().read_vec(info + 0x40, len)).unwrap());
        assert_eq!(k.mem().read_u32(info), 0, "single entry has no next offset");
    }
    assert_eq!(names, ["Skyrim.esm", "Update.esm"]);

    // Restart with a large buffer: both entries in one call, chained.
    let s = k.call_export("NtQueryDirectoryFile", &mut ctx, &[h, 0, 0, 0, iosb, info, 0x1000, pattern, 1]) as u32;
    assert_eq!(s, status::SUCCESS);
    let next = k.mem().read_u32(info);
    assert_eq!(next % 8, 0);
    assert_eq!(k.mem().read_vec(info + next + 0x40, 10), b"Update.esm");
}

#[test]
fn rtl_strings_and_time() {
    let k = kernel();
    let mut ctx = guest(k);
    let m = scratch(k, 0x1000);
    k.mem().write_bytes(m + 0x100, b"hello\0");
    k.call_export("RtlInitAnsiString", &mut ctx, &[m, m + 0x100]);
    assert_eq!((k.mem().read_u16(m), k.mem().read_u16(m + 2), k.mem().read_u32(m + 4)), (5, 6, m + 0x100));

    // ANSI -> UNICODE (allocated) -> ANSI (allocated)
    assert_eq!(k.call_export("RtlAnsiStringToUnicodeString", &mut ctx, &[m + 8, m, 1]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_u16(m + 8), 10);
    let ubuf = k.mem().read_u32(m + 12);
    assert_eq!(k.mem().read_u16(ubuf), b'h' as u16);
    assert_eq!(k.call_export("RtlUnicodeStringToAnsiString", &mut ctx, &[m + 16, m + 8, 1]) as u32, status::SUCCESS);
    assert_eq!(k.mem().read_vec(k.mem().read_u32(m + 20), 6), b"hello\0");
    k.call_export("RtlFreeUnicodeString", &mut ctx, &[m + 8]);
    k.call_export("RtlFreeAnsiString", &mut ctx, &[m + 16]);
    assert_eq!(k.mem().read_u32(m + 12), 0);

    assert_eq!(k.call_export("RtlCompareString", &mut ctx, &[m, m, 0]) as i32, 0);
    assert_eq!(k.call_export("RtlUpcaseUnicodeChar", &mut ctx, &[b'q' as u32]) as u32, b'Q' as u32);
    assert_eq!(k.call_export("RtlNtStatusToDosError", &mut ctx, &[status::OBJECT_NAME_NOT_FOUND]) as u32, 2);

    // 2011-11-11 11:11:11.111 (Friday) round trip.
    let fields = m + 0x200;
    for (i, v) in [2011u16, 11, 11, 11, 11, 11, 111, 0].iter().enumerate() {
        k.mem().write_u16(fields + i as u32 * 2, *v);
    }
    assert_eq!(k.call_export("RtlTimeFieldsToTime", &mut ctx, &[fields, m + 0x300]), 1);
    assert_eq!(k.mem().read_u64(m + 0x300), 129_654_834_711_110_000);
    k.call_export("RtlTimeToTimeFields", &mut ctx, &[m + 0x300, m + 0x400]);
    let back: Vec<u16> = (0..8).map(|i| k.mem().read_u16(m + 0x400 + i * 2)).collect();
    assert_eq!(back, [2011, 11, 11, 11, 11, 11, 111, 5]);
}

#[test]
fn system_queries() {
    let k = kernel();
    let mut ctx = guest(k);
    let m = scratch(k, 0x1000);
    // The synthetic XEX has system flags 0x400 (bit 10).
    assert_eq!(k.call_export("XexCheckExecutablePrivilege", &mut ctx, &[10]), 1);
    assert_eq!(k.call_export("XexCheckExecutablePrivilege", &mut ctx, &[11]), 0);
    assert_eq!(k.call_export("XGetAVPack", &mut ctx, &[]), 8);
    assert_eq!(k.call_export("XGetLanguage", &mut ctx, &[]), 1);

    let header = k.rt.image.header_addr;
    assert_eq!(k.call_export("RtlImageXexHeaderField", &mut ctx, &[header, xex::keys::SYSTEM_FLAGS]), 0x400);
    let base_slot = k.call_export("RtlImageXexHeaderField", &mut ctx, &[header, xex::keys::IMAGE_BASE_ADDRESS]) as u32;
    assert_eq!(k.mem().read_u32(base_slot), common::BASE);
    assert_eq!(k.call_export("RtlImageXexHeaderField", &mut ctx, &[header, 0x1234_5600]), 0);

    // Language setting: category USER (3), setting 9.
    let (buf, req) = (m, m + 0x10);
    assert_eq!(k.call_export("ExGetXConfigSetting", &mut ctx, &[3, 9, buf, 4, req]) as u32, status::SUCCESS);
    assert_eq!((k.mem().read_u32(buf), k.mem().read_u16(req)), (1, 4));
    assert_eq!(k.call_export("ExGetXConfigSetting", &mut ctx, &[3, 9, buf, 2, req]) as u32, status::BUFFER_TOO_SMALL);

    let hmod = m + 0x20;
    assert_eq!(k.call_export("XexGetModuleHandle", &mut ctx, &[0, hmod]), 0);
    let ldr = k.mem().read_u32(hmod);
    assert_eq!(k.mem().read_u32(ldr + 0x18), common::BASE);
    assert_eq!(k.mem().read_u32(ldr + 0x58), header);

    k.call_export("KeQuerySystemTime", &mut ctx, &[m + 0x30]);
    // After 2020-01-01 in FILETIME.
    assert!(k.mem().read_u64(m + 0x30) > 132_223_104_000_000_000);
}

#[test]
fn guest_printf_formatting() {
    let k = kernel();
    let m = scratch(k, 0x1000);
    k.mem().write_bytes(m, b"skyrim\0");
    let args = [42u64, m as u64, 0xBEEF, (-7i32) as u32 as u64, 1.5f64.to_bits(), b'Z' as u64];
    let mut it = args.into_iter();
    let s = kernel_hle::dbgprint::format(k.mem(), b"%d %s %08x %+d %.2f %c %%", || it.next().unwrap());
    assert_eq!(s, "42 skyrim 0000beef -7 1.50 Z %");
}
