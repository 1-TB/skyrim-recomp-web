//! A synthetic title for end-to-end testing of the toolchain without game
//! files: hand-assembled PPC code in a PE image wrapped as a decrypted XEX,
//! with real xboxkrnl/xam import tables.
//!
//! The program reports what it observed through DbgPrint ("synth: ...") and
//! returns 7 from its entry point. `EXPECTED_LINES` lists what a correct
//! runtime prints when run with `--stub-return 0xDEADBEEF`.

use std::collections::HashMap;
use std::path::Path;

use xex::builder::{
    IMAGE_SCN_CNT_CODE, IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE,
    ImportLibrarySpec, PeSection, XexBuilder, build_pe_image,
};

use crate::ppc::*;

pub const BASE: u32 = 0x8200_0000;
const IMAGE_SIZE: u32 = 0x10000;
const RDATA: u32 = 0x1000;
const DATA: u32 = 0x2000;
const PDATA: u32 = 0x3000;
const TEXT: u32 = 0x4000;

pub const EXIT_CODE: i32 = 7;
pub const STUB_RETURN: u32 = 0xDEAD_BEEF;
pub const FILE_CONTENTS: &str = "hello-from-vfs";

/// Lines the runtime must log (DbgPrint output), in order.
pub fn expected_lines() -> Vec<String> {
    vec![
        "synth: nonvolatile=1005 r31=77".into(),
        "synth: switch=12 99 21 98".into(),
        "synth: tls slot=0 value=55".into(),
        "synth: counter=3".into(),
        "synth: views load=1234 atomic=1 vector=13 dcbz=0".into(),
        "synth: unpack4=3F800000 C0000000 3F000000 47FFE000 unpack2=3F000000 3F800000".into(),
        "synth: pack=3C00C000 33334444 blrl=107".into(),
        "synth: indirect=105".into(),
        "synth: worker 42".into(),
        "synth: thread create=00000000 wait=00000000".into(),
        format!("synth: open=00000000 read=00000000 data={FILE_CONTENTS}"),
        format!("synth: stubs={STUB_RETURN:08X} {STUB_RETURN:08X} {STUB_RETURN:08X} avpack=8"),
    ]
}

struct Import {
    lib: usize,
    ordinal: u16,
    label: &'static str,
    function: bool,
}

const LIBS: [&str; 2] = ["xboxkrnl.exe", "xam.xex"];

const IMPORTS: &[Import] = &[
    Import { lib: 0, ordinal: 0x0CC, label: "NtAllocateVirtualMemory", function: true },
    Import { lib: 0, ordinal: 0x0BA, label: "MmAllocatePhysicalMemoryEx", function: true },
    Import { lib: 0, ordinal: 0x152, label: "KeTlsAlloc", function: true },
    Import { lib: 0, ordinal: 0x155, label: "KeTlsSetValue", function: true },
    Import { lib: 0, ordinal: 0x154, label: "KeTlsGetValue", function: true },
    Import { lib: 0, ordinal: 0x12E, label: "RtlInitializeCriticalSection", function: true },
    Import { lib: 0, ordinal: 0x125, label: "RtlEnterCriticalSection", function: true },
    Import { lib: 0, ordinal: 0x130, label: "RtlLeaveCriticalSection", function: true },
    Import { lib: 0, ordinal: 0x00D, label: "ExCreateThread", function: true },
    Import { lib: 0, ordinal: 0x019, label: "ExTerminateThread", function: true },
    Import { lib: 0, ordinal: 0x0FD, label: "NtWaitForSingleObjectEx", function: true },
    Import { lib: 0, ordinal: 0x0CF, label: "NtClose", function: true },
    Import { lib: 0, ordinal: 0x003, label: "DbgPrint", function: true },
    Import { lib: 0, ordinal: 0x0D2, label: "NtCreateFile", function: true },
    Import { lib: 0, ordinal: 0x0F0, label: "NtReadFile", function: true },
    // Named in the export tables but not implemented: exercises the stub path.
    Import { lib: 0, ordinal: 0x0D9, label: "NtDeviceIoControlFile", function: true },
    // Not in the export tables at all: XenonRecomp emits a plain sub_ stub.
    Import { lib: 0, ordinal: 0x3FF, label: "Unnamed3FF", function: true },
    Import { lib: 0, ordinal: 0x0AD, label: "KeTimeStampBundle", function: false },
    Import { lib: 1, ordinal: 0x3CB, label: "XGetAVPack", function: true },
    Import { lib: 1, ordinal: 0x212, label: "XamUserCheckPrivilege", function: true },
];

struct Data {
    bytes: Vec<u8>,
    base: u32,
    labels: HashMap<String, u32>,
}

impl Data {
    fn new(base: u32) -> Self {
        Data { bytes: Vec::new(), base, labels: HashMap::new() }
    }

    fn align(&mut self, a: usize) {
        while !self.bytes.len().is_multiple_of(a) {
            self.bytes.push(0);
        }
    }

    fn put(&mut self, label: &str, data: &[u8]) -> u32 {
        self.align(8);
        let addr = self.base + self.bytes.len() as u32;
        self.labels.insert(label.into(), addr);
        self.bytes.extend_from_slice(data);
        addr
    }

    fn reserve(&mut self, label: &str, size: usize) -> u32 {
        self.put(label, &vec![0u8; size])
    }

    fn cstr(&mut self, label: &str, s: &str) -> u32 {
        let mut v = s.as_bytes().to_vec();
        v.push(0);
        self.put(label, &v)
    }
}

pub struct Synthetic {
    pub xex: Vec<u8>,
    pub savegprlr_14: u32,
    pub restgprlr_14: u32,
}

pub fn build() -> Synthetic {
    // .rdata: import record slots, then strings.
    let mut rdata = Data::new(BASE + RDATA);
    let mut slots = HashMap::new();
    for imp in IMPORTS {
        let record = ((imp.lib as u32) << 16) | imp.ordinal as u32;
        let addr = rdata.put(&format!("slot_{}", imp.label), &record.to_be_bytes());
        slots.insert(imp.label, addr);
    }
    for (label, s) in [
        ("fmt_alloc", "synth: alloc status=%08X base=%08X\n"),
        ("fmt_tls", "synth: tls slot=%d value=%X\n"),
        ("fmt_counter", "synth: counter=%d\n"),
        ("fmt_views", "synth: views load=%X atomic=%d vector=%X dcbz=%X\n"),
        ("fmt_unpack", "synth: unpack4=%08X %08X %08X %08X unpack2=%08X %08X\n"),
        ("fmt_pack", "synth: pack=%08X %08X blrl=%d\n"),
        ("fmt_indirect", "synth: indirect=%d\n"),
        ("fmt_worker", "synth: worker %d\n"),
        ("fmt_thread", "synth: thread create=%08X wait=%08X\n"),
        ("fmt_file", "synth: open=%08X read=%08X data=%s\n"),
        ("fmt_stub", "synth: stubs=%08X %08X %08X avpack=%d\n"),
        ("fmt_bundle", "synth: bundle=%08X\n"),
        ("fmt_nv", "synth: nonvolatile=%d r31=%d\n"),
        ("fmt_switch", "synth: switch=%d %d %d %d\n"),
        ("path", "game:\\SYNTH.TXT"),
    ] {
        rdata.cstr(label, s);
    }

    let rel_table = rdata.reserve("rel_table", 4);

    // .data: mutable state, plus a prebuilt OBJECT_ATTRIBUTES for the file.
    let mut data = Data::new(BASE + DATA);
    for (label, size) in [
        ("counter", 4),
        ("cs", 28),
        ("thread_handle", 4),
        ("thread_id", 4),
        ("file_handle", 4),
        ("iosb", 8),
        ("file_buf", 32),
        ("alloc_base", 4),
        ("alloc_size", 4),
    ] {
        data.reserve(label, size);
    }
    // Xenos halves for vupkd3d128/vpkd3d128: PPC halfwords 4..7 are 1.0, -2.0,
    // 0.5 and 0x7FFF (131008; no infinity in the Xenos encoding).
    data.align(16);
    let mut halves = Vec::new();
    for w in [0x1111_2222u32, 0x3333_4444, 0x3C00_C000, 0x3800_7FFF] {
        halves.extend_from_slice(&w.to_be_bytes());
    }
    data.put("halves", &halves);
    data.align(16);
    data.reserve("vec_out", 16);
    let path = rdata.labels["path"];
    let path_len = "game:\\SYNTH.TXT".len() as u16;
    let mut ansi = Vec::new();
    ansi.extend_from_slice(&path_len.to_be_bytes());
    ansi.extend_from_slice(&(path_len + 1).to_be_bytes());
    ansi.extend_from_slice(&path.to_be_bytes());
    let ansi_addr = data.put("ansi_path", &ansi);
    let mut attrs = Vec::new();
    attrs.extend_from_slice(&0u32.to_be_bytes());
    attrs.extend_from_slice(&ansi_addr.to_be_bytes());
    attrs.extend_from_slice(&0u32.to_be_bytes());
    data.put("obj_attr", &attrs);

    // .text: one 16-byte thunk per function import, then the code.
    let mut a = Asm::new(BASE + TEXT);
    let mut thunks = Vec::new();
    for imp in IMPORTS.iter().filter(|i| i.function) {
        let addr = a.label(imp.label);
        a.word((1 << 24) | ((imp.lib as u32) << 16) | imp.ordinal as u32);
        a.word(0x6000_0000);
        a.word(0x6000_0000);
        a.blr();
        thunks.push((imp, addr));
    }
    for (name, addr) in rdata.labels.iter().chain(data.labels.iter()) {
        a.extern_label(name, *addr);
    }

    let main = a.label("main");
    a.prologue(0x100);
    // Non-volatile registers survive a call that saves/restores them through
    // the compiler's __savegprlr/__restgprlr helpers.
    a.li(R31, 77);
    a.li(R3, 5);
    a.bl("nvtest");
    a.mr(R4, R3);
    a.mr(R5, R31);
    a.la(R3, "fmt_nv");
    a.bl("DbgPrint");
    // Jump tables: an absolute inline table in a leaf function, and a
    // byte-offset table in a .pdata function; each called in and out of range.
    for (i, (f, arg)) in [("sw_abs", 2), ("sw_abs", 7), ("sw_rel", 1), ("sw_rel", 9)].into_iter().enumerate() {
        a.li(R3, arg);
        a.bl(f);
        a.stw(R3, 0x60 + 4 * i as i32, R1);
    }
    a.lwz(R4, 0x60, R1);
    a.lwz(R5, 0x64, R1);
    a.lwz(R6, 0x68, R1);
    a.lwz(R7, 0x6C, R1);
    a.la(R3, "fmt_switch");
    a.bl("DbgPrint");
    // NtAllocateVirtualMemory(&base, &size, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE, 0)
    a.la(R3, "alloc_base");
    a.li(R0, 0);
    a.stw(R0, 0, R3);
    a.la(R4, "alloc_size");
    a.li32(R0, 0x10000);
    a.stw(R0, 0, R4);
    a.li(R5, 0x3000);
    a.li(R6, 4);
    a.li(R7, 0);
    a.bl("NtAllocateVirtualMemory");
    a.mr(R4, R3);
    a.la(R5, "alloc_base");
    a.lwz(R5, 0, R5);
    a.la(R3, "fmt_alloc");
    a.bl("DbgPrint");
    // TLS
    a.bl("KeTlsAlloc");
    a.stw(R3, 0x60, R1);
    a.li(R4, 0x55);
    a.bl("KeTlsSetValue");
    a.lwz(R3, 0x60, R1);
    a.bl("KeTlsGetValue");
    a.mr(R5, R3);
    a.lwz(R4, 0x60, R1);
    a.la(R3, "fmt_tls");
    a.bl("DbgPrint");
    // Critical section
    a.la(R3, "cs");
    a.bl("RtlInitializeCriticalSection");
    a.la(R3, "cs");
    a.bl("RtlEnterCriticalSection");
    a.la(R3, "cs");
    a.bl("RtlLeaveCriticalSection");
    // Three lwarx/stwcx. increments (raw `base + ea` accesses in generated code).
    a.la(R9, "counter");
    a.li(R10, 0);
    for i in 0..3 {
        let l = format!("atomic{i}");
        a.label(&l);
        a.lwarx(R11, R9, R10);
        a.addi(R11, R11, 1);
        a.stwcx(R11, R9, R10);
        a.bne(&l);
    }
    a.la(R4, "counter");
    a.lwz(R4, 0, R4);
    a.la(R3, "fmt_counter");
    a.bl("DbgPrint");
    // One physical page through all three views (0xE0000000 + x is physical
    // x + 0x1000; 0xA0000000/0xC0000000 + x are physical x): plain loads and
    // stores (PPC_LOAD/PPC_STORE) and the accesses XenonRecomp emits inline
    // (lwarx/stwcx., lvx/stvx, dcbz) must all reach the same memory.
    // MmAllocatePhysicalMemoryEx(0, 0x1000, PAGE_READWRITE, 0, ~0, 0): 4 KiB view.
    a.li(R3, 0);
    a.li32(R4, 0x1000);
    a.li(R5, 4);
    a.li(R6, 0);
    a.li(R7, -1);
    a.li(R8, 0);
    a.bl("MmAllocatePhysicalMemoryEx");
    a.li32(R9, 0x2000_1000);
    a.add(R9, R3, R9); // physical address
    a.li32(R10, 0xA000_0000);
    a.add(R10, R9, R10); // 64 KiB view
    a.li32(R12, 0xC000_0000);
    a.add(R12, R9, R12); // 16 MiB view
    // Store through 0xE..., load through 0xC...
    a.li32(R0, 0x1234);
    a.stw(R0, 0, R3);
    a.lwz(R11, 0, R12);
    a.stw(R11, 0x70, R1);
    // Atomic increment through 0xA..., read back through 0xE...
    a.li(R8, 4);
    a.label("view_atomic");
    a.lwarx(R11, R10, R8);
    a.addi(R11, R11, 1);
    a.stwcx(R11, R10, R8);
    a.bne("view_atomic");
    a.lwz(R11, 4, R3);
    a.stw(R11, 0x74, R1);
    // Four words through 0xE..., lvx through 0xC..., stvx through 0xA...,
    // last word read back through 0xE...
    for (i, v) in [0x10, 0x11, 0x12, 0x13].into_iter().enumerate() {
        a.li(R0, v);
        a.stw(R0, 0x10 + 4 * i as i32, R3);
    }
    a.li(R8, 0x10);
    a.lvx(V0, R12, R8);
    a.li(R8, 0x30);
    a.stvx(V0, R10, R8);
    a.lwz(R11, 0x3C, R3);
    a.stw(R11, 0x78, R1);
    // Non-zero through 0xA..., dcbz through 0xC..., read through 0xE...
    a.li(R8, 0x40);
    a.li(R0, -1);
    a.stwx(R0, R10, R8);
    a.dcbz(R12, R8);
    a.lwz(R11, 0x40, R3);
    a.stw(R11, 0x7C, R1);
    a.lwz(R4, 0x70, R1);
    a.lwz(R5, 0x74, R1);
    a.lwz(R6, 0x78, R1);
    a.lwz(R7, 0x7C, R1);
    a.la(R3, "fmt_views");
    a.bl("DbgPrint");
    // D3D vector unpack/pack (Xenos half floats) and a call through LR.
    a.la(R9, "halves");
    a.la(R10, "vec_out");
    a.li(R8, 0);
    a.lvx(V0, R9, R8);
    a.vupkd3d128(V1, V0, 5 << 2); // float16_4
    a.stvx(V1, R10, R8);
    for i in 0..4 {
        a.lwz(R11, 4 * i, R10);
        a.stw(R11, 0x70 + 4 * i, R1);
    }
    a.vupkd3d128(V2, V0, 3 << 2); // float16_2
    a.stvx(V2, R10, R8);
    a.lwz(R9, 0, R10);
    a.lwz(R11, 12, R10);
    a.stw(R11, 0x80, R1);
    a.lwz(R4, 0x70, R1);
    a.lwz(R5, 0x74, R1);
    a.lwz(R6, 0x78, R1);
    a.lwz(R7, 0x7C, R1);
    a.mr(R8, R9);
    a.lwz(R9, 0x80, R1);
    a.la(R3, "fmt_unpack");
    a.bl("DbgPrint");
    // float16_2 of (1.0, -2.0) into word 0 of a copy of the halves (pack
    // mode 1, shift 3); words 1-3 keep their values.
    a.la(R9, "halves");
    a.la(R10, "vec_out");
    a.li(R8, 0);
    a.lvx(V3, R9, R8);
    a.vpkd3d128(V3, V1, 3, 1, 3);
    a.stvx(V3, R10, R8);
    a.lwz(R11, 0, R10);
    a.stw(R11, 0x70, R1);
    a.lwz(R11, 4, R10);
    a.stw(R11, 0x74, R1);
    a.la(R11, "helper");
    a.mtlr(R11);
    a.li(R3, 7);
    a.blrl();
    a.mr(R6, R3);
    a.lwz(R4, 0x70, R1);
    a.lwz(R5, 0x74, R1);
    a.la(R3, "fmt_pack");
    a.bl("DbgPrint");
    // Indirect call through ctr (PPC_CALL_INDIRECT_FUNC -> call table).
    a.la(R11, "helper");
    a.mtctr(R11);
    a.li(R3, 5);
    a.bctrl();
    a.mr(R4, R3);
    a.la(R3, "fmt_indirect");
    a.bl("DbgPrint");
    // ExCreateThread(&handle, 0, &tid, 0, worker, 41, 0); wait; close.
    a.la(R3, "thread_handle");
    a.li(R4, 0);
    a.la(R5, "thread_id");
    a.li(R6, 0);
    a.la(R7, "worker");
    a.li(R8, 41);
    a.li(R9, 0);
    a.bl("ExCreateThread");
    a.stw(R3, 0x60, R1);
    a.la(R3, "thread_handle");
    a.lwz(R3, 0, R3);
    a.li(R4, 1);
    a.li(R5, 0);
    a.li(R6, 0);
    a.bl("NtWaitForSingleObjectEx");
    a.mr(R5, R3);
    a.lwz(R4, 0x60, R1);
    a.la(R3, "fmt_thread");
    a.bl("DbgPrint");
    a.la(R3, "thread_handle");
    a.lwz(R3, 0, R3);
    a.bl("NtClose");
    // NtCreateFile(&h, GENERIC_READ, &attrs, &iosb, 0, 0, 1, FILE_OPEN,
    //              FILE_SYNCHRONOUS_IO_NONALERT | FILE_NON_DIRECTORY_FILE)
    a.li(R0, 0x60);
    a.stw(R0, 0x54, R1);
    a.la(R3, "file_handle");
    a.li32(R4, 0x8000_0000);
    a.la(R5, "obj_attr");
    a.la(R6, "iosb");
    a.li(R7, 0);
    a.li(R8, 0);
    a.li(R9, 1);
    a.li(R10, 1);
    a.bl("NtCreateFile");
    a.stw(R3, 0x60, R1);
    // NtReadFile(h, 0, 0, 0, &iosb, buf, 31, NULL)
    a.la(R3, "file_handle");
    a.lwz(R3, 0, R3);
    a.li(R4, 0);
    a.li(R5, 0);
    a.li(R6, 0);
    a.la(R7, "iosb");
    a.la(R8, "file_buf");
    a.li(R9, 31);
    a.li(R10, 0);
    a.bl("NtReadFile");
    a.stw(R3, 0x64, R1);
    a.lwz(R4, 0x60, R1);
    a.lwz(R5, 0x64, R1);
    a.la(R6, "file_buf");
    a.la(R3, "fmt_file");
    a.bl("DbgPrint");
    a.la(R3, "file_handle");
    a.lwz(R3, 0, R3);
    a.bl("NtClose");
    // Unimplemented imports: named, unnamed, and XAM; plus an implemented XAM call.
    a.li(R3, 0x11);
    a.li(R4, 0x22);
    a.bl("NtDeviceIoControlFile");
    a.stw(R3, 0x60, R1);
    a.li(R3, 0x33);
    a.bl("Unnamed3FF");
    a.stw(R3, 0x64, R1);
    a.li(R3, 0);
    a.bl("XamUserCheckPrivilege");
    a.stw(R3, 0x68, R1);
    a.bl("XGetAVPack");
    a.mr(R7, R3);
    a.lwz(R4, 0x60, R1);
    a.lwz(R5, 0x64, R1);
    a.lwz(R6, 0x68, R1);
    a.la(R3, "fmt_stub");
    a.bl("DbgPrint");
    // Variable import: the patched slot points at KeTimeStampBundle.
    a.la(R11, "slot_KeTimeStampBundle");
    a.lwz(R4, 0, R11);
    a.la(R3, "fmt_bundle");
    a.bl("DbgPrint");
    a.li(R3, EXIT_CODE);
    a.epilogue(0x100);
    let main_end = a.pc();

    let worker = a.label("worker");
    a.prologue(0x80);
    a.addi(R3, R3, 1);
    a.stw(R3, 0x60, R1);
    a.mr(R4, R3);
    a.la(R3, "fmt_worker");
    a.bl("DbgPrint");
    a.lwz(R3, 0x60, R1);
    a.bl("ExTerminateThread");
    a.epilogue(0x80);
    let worker_end = a.pc();

    let helper = a.label("helper");
    a.addi(R3, R3, 100);
    a.blr();
    let helper_end = a.pc();

    // Leaf function without .pdata: absolute table right after the bctr.
    a.label("sw_abs");
    a.cmplwi(CR6, R3, 3);
    a.bgt(CR6, "sw_abs_default");
    a.la(R12, "sw_abs_table");
    a.slwi(R0, R3, 2);
    a.lwzx(R0, R12, R0);
    a.mtctr(R0);
    a.bctr();
    a.label("sw_abs_table");
    for c in 0..4 {
        a.addr_word(&format!("sw_abs_case{c}"));
    }
    for c in 0..4 {
        a.label(&format!("sw_abs_case{c}"));
        a.li(R3, 10 + c);
        a.blr();
    }
    a.label("sw_abs_default");
    a.li(R3, 99);
    a.blr();

    // .pdata function: code base + (byte offset << 2) from a .rdata table.
    let sw_rel = a.label("sw_rel");
    a.cmplwi(CR6, R3, 3);
    a.bgt(CR6, "sw_rel_default");
    a.la(R12, "rel_table");
    a.lbzx(R0, R12, R3);
    a.slwi(R0, R0, 2);
    a.la(R12, "sw_rel_base");
    a.add(R12, R12, R0);
    a.mtctr(R12);
    a.bctr();
    let rel_cases: Vec<u32> = (0..4)
        .map(|c| {
            let at = a.label(&format!("sw_rel_case{c}"));
            if c == 0 {
                a.extern_label("sw_rel_base", at);
            }
            a.li(R3, 20 + c);
            a.blr();
            at
        })
        .collect();
    a.label("sw_rel_default");
    a.li(R3, 98);
    a.blr();
    let sw_rel_end = a.pc();

    let nvtest = a.label("nvtest");
    a.mflr(R12);
    a.bl("__savegprlr_29");
    a.stwu(R1, -0x70, R1);
    a.mr(R31, R3);
    a.addi(R31, R31, 1000);
    a.mr(R3, R31);
    a.addi(R1, R1, 0x70);
    a.b("__restgprlr_29");
    let nvtest_end = a.pc();

    // The compiler's register save/restore helpers: each __savegprlr_N entry
    // falls through to the next register, then saves LR (in r12).
    let savegpr = a.pc();
    for r in 14..=31u32 {
        a.label(&format!("__savegprlr_{r}"));
        a.std(r, -(8 * (32 - r as i32) + 8), R1);
    }
    a.stw(R12, -8, R1);
    a.blr();
    let restgpr = a.pc();
    for r in 14..=31u32 {
        a.label(&format!("__restgprlr_{r}"));
        a.ld(r, -(8 * (32 - r as i32) + 8), R1);
    }
    a.lwz(R12, -8, R1);
    a.mtlr(R12);
    a.blr();

    let (text, _labels) = a.finish();

    // .pdata: IMAGE_CE_RUNTIME_FUNCTION { BeginAddress, PrologLen:8 | FuncLen:22 | 32bit:1 | Exc:1 }
    let mut pdata = Vec::new();
    for (start, end, prolog) in [
        (main, main_end, 3u32),
        (worker, worker_end, 3),
        (helper, helper_end, 0),
        (nvtest, nvtest_end, 3),
        (sw_rel, sw_rel_end, 0),
    ] {
        pdata.extend_from_slice(&start.to_be_bytes());
        let words = (end - start) / 4;
        pdata.extend_from_slice(&(prolog | (words << 8) | (1 << 30)).to_be_bytes());
    }

    let off = (rel_table - (BASE + RDATA)) as usize;
    for (c, &case) in rel_cases.iter().enumerate() {
        rdata.bytes[off + c] = ((case - rel_cases[0]) >> 2) as u8;
    }

    let rw = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE;
    let sections = vec![
        PeSection {
            name: ".rdata".into(),
            rva: RDATA,
            virtual_size: rdata.bytes.len() as u32,
            characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
            data: rdata.bytes,
        },
        PeSection {
            name: ".data".into(),
            rva: DATA,
            virtual_size: data.bytes.len() as u32,
            characteristics: rw,
            data: data.bytes,
        },
        PeSection {
            name: ".pdata".into(),
            rva: PDATA,
            virtual_size: pdata.len() as u32,
            characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
            data: pdata,
        },
        PeSection {
            name: ".text".into(),
            rva: TEXT,
            virtual_size: text.len() as u32,
            characteristics: IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ,
            data: text,
        },
    ];
    let image = build_pe_image(&sections, IMAGE_SIZE);

    let mut records: Vec<Vec<u32>> = vec![Vec::new(); LIBS.len()];
    for imp in IMPORTS {
        records[imp.lib].push(slots[imp.label]);
        if let Some((_, thunk)) = thunks.iter().find(|(i, _)| i.label == imp.label) {
            records[imp.lib].push(*thunk);
        }
    }
    let xex = XexBuilder {
        image,
        load_address: BASE,
        entry_point: main,
        module_flags: 1,
        imports: LIBS
            .iter()
            .zip(records)
            .map(|(name, records)| ImportLibrarySpec {
                name: name.to_string(),
                id: 0,
                version: 0x2000_4497,
                min_version: 0x2000_0000,
                records,
            })
            .collect(),
        tls: None,
        system_flags: Some(0),
        default_stack_size: Some(0x40000),
        execution_info: None,
        basic_compression: true,
    }
    .build();
    Synthetic { xex, savegprlr_14: savegpr, restgprlr_14: restgpr }
}

fn recomp_toml(s: &Synthetic) -> String {
    format!(
        "# XenonRecomp config for the synthetic test title (generated by `cargo xtask synth`).
# The synthetic code has no FPR/VMX save/restore helpers, so XenonRecomp reports
# those six addresses as unspecified; that is expected here.
[main]
file_path = \"default.xex\"
out_directory_path = \"recomp\"
switch_table_file_path = \"switch_tables.toml\"
savegprlr_14_address = {:#010X}
restgprlr_14_address = {:#010X}
",
        s.savegprlr_14, s.restgprlr_14
    )
}

/// Writes default.xex, the recompiler config and a game directory into `dir`.
pub fn write(dir: &Path) -> std::io::Result<()> {
    let s = build();
    std::fs::create_dir_all(dir.join("game"))?;
    std::fs::create_dir_all(dir.join("recomp"))?;
    std::fs::write(dir.join("default.xex"), &s.xex)?;
    std::fs::write(dir.join("game/default.xex"), &s.xex)?;
    std::fs::write(dir.join("game/synth.txt"), FILE_CONTENTS)?;
    std::fs::write(dir.join("synthetic.toml"), recomp_toml(&s))?;
    let xex = xex::Xex::parse(&s.xex).map_err(std::io::Error::other)?;
    let report = crate::switches::analyze(&xex).map_err(std::io::Error::other)?;
    if !report.unresolved.is_empty() || report.tables.len() != 2 || report.functions.len() != 1 {
        return Err(std::io::Error::other(format!(
            "switch analysis of the synthetic title: {} tables, leaf extents {:X?}, unresolved {:?}",
            report.tables.len(),
            report.functions,
            report.unresolved
        )));
    }
    std::fs::write(dir.join("switch_tables.toml"), crate::switches::toml(&report))?;
    Ok(())
}
