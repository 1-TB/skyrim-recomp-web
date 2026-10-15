//! Test harness: a kernel around a synthetic title (no recompiled code).

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use kernel_hle::Kernel;
use kernel_hle::dispatch::{ImportName, StubConfig};
use runtime_core::layout::kpcr;
use runtime_core::{GuestModule, PPCContext, Runtime};
use xex::builder::{
    IMAGE_SCN_CNT_CODE, IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE,
    ImportLibrarySpec, PeSection, XexBuilder, build_pe_image,
};

pub const BASE: u32 = 0x8200_0000;
pub const IMAGE_SIZE: u32 = 0x10000;
pub const CODE_RVA: u32 = 0x2000;
/// Record slot of the KeTimeStampBundle variable import.
pub const TIMESTAMP_SLOT: u32 = BASE + 0x1000;
/// Record slot of an import with no implementation (xboxkrnl ordinal 0x3FF).
pub const UNKNOWN_VAR_SLOT: u32 = BASE + 0x1004;
/// Record slot and thunk of the NtClose function import.
pub const NTCLOSE_SLOT: u32 = BASE + 0x1008;
pub const NTCLOSE_THUNK: u32 = BASE + CODE_RVA;

/// A guest function (implemented on the host) that records each call.
pub const RECORDER: u32 = BASE + CODE_RVA + 0x100;

static FUNCS: [u32; 1] = [RECORDER];

/// The module's function dispatch: [`RECORDER`] is the only function.
unsafe extern "C-unwind" fn call(addr: u32, ctx: *mut PPCContext, base: *mut u8) {
    assert_eq!(addr, RECORDER);
    // SAFETY: as received from Runtime::call.
    unsafe { recorder(ctx, base) }
}

/// What [`RECORDER`] saw: r3..r6 and the DPC state of its processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedCall {
    pub args: [u32; 4],
    pub irql: u8,
    pub dpc_active: u32,
    pub process_type_in_dpc: u8,
    pub cpu: u8,
}

pub static RECORDED: Mutex<Vec<RecordedCall>> = Mutex::new(Vec::new());

/// Recorded calls whose second argument is `tag`.
pub fn recorded_with_arg1(tag: u32) -> Vec<RecordedCall> {
    RECORDED.lock().unwrap().iter().filter(|r| r.args[1] == tag).copied().collect()
}

unsafe extern "C-unwind" fn recorder(ctx: *mut PPCContext, base: *mut u8) {
    // SAFETY: called by Runtime::call with a live context and the guest base.
    let ctx = unsafe { &mut *ctx };
    // Guest addresses go through the same translation as recompiled code's (skyrt_ea).
    let pcr = runtime_core::memory::linear(ctx.gpr32(13)) as usize;
    // SAFETY: r13 is the thread's PCR, inside committed guest memory (one frame).
    let (irql, dpc_active, process_type_in_dpc, cpu) = unsafe {
        let p = base.add(pcr);
        (
            *p.add(kpcr::CURRENT_IRQL as usize),
            u32::from_be_bytes(*(p.add(kpcr::PRCB_DPC_ACTIVE as usize) as *const [u8; 4])),
            *p.add(kpcr::PROCESSTYPE_VALUE_IN_DPC as usize),
            *p.add(kpcr::PRCB_CURRENT_CPU as usize),
        )
    };
    let args = [ctx.gpr32(3), ctx.gpr32(4), ctx.gpr32(5), ctx.gpr32(6)];
    RECORDED.lock().unwrap().push(RecordedCall { args, irql, dpc_active, process_type_in_dpc, cpu });
    // Clobber volatile registers like real code would.
    ctx.set_gpr(3, 0xDEAD);
}

unsafe extern "C" fn init_context(_ctx: *mut PPCContext) {}

fn synthetic_xex() -> xex::Xex {
    let mut rdata = vec![0u8; 0x10];
    rdata[0..4].copy_from_slice(&0x0000_00ADu32.to_be_bytes());
    rdata[4..8].copy_from_slice(&0x0000_03FFu32.to_be_bytes());
    rdata[8..12].copy_from_slice(&0x0000_00CFu32.to_be_bytes());
    let mut text = vec![0u8; 0x20];
    text[0..4].copy_from_slice(&0x0100_00CFu32.to_be_bytes());
    text[0x10..0x14].copy_from_slice(&0x4E80_0020u32.to_be_bytes());
    let image = build_pe_image(
        &[
            PeSection {
                name: ".rdata".into(),
                rva: 0x1000,
                virtual_size: 0x10,
                characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE,
                data: rdata,
            },
            PeSection {
                name: ".text".into(),
                rva: CODE_RVA,
                virtual_size: 0x20,
                characteristics: IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ,
                data: text,
            },
        ],
        IMAGE_SIZE,
    );
    let bytes = XexBuilder {
        image,
        load_address: BASE,
        entry_point: BASE + CODE_RVA + 0x10,
        module_flags: 1,
        imports: vec![ImportLibrarySpec {
            name: "xboxkrnl.exe".into(),
            id: 0,
            version: 0x2000_4497,
            min_version: 0x2000_0000,
            records: vec![TIMESTAMP_SLOT, UNKNOWN_VAR_SLOT, NTCLOSE_SLOT, NTCLOSE_THUNK],
        }],
        tls: Some(xex::TlsInfo { slot_count: 64, raw_data_address: 0, data_size: 0, raw_data_size: 0 }),
        system_flags: Some(0x0000_0400),
        default_stack_size: Some(0x40000),
        execution_info: None,
        basic_compression: false,
    }
    .build();
    xex::Xex::parse(&bytes).expect("synthetic xex")
}

pub fn game_dir() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "kernel-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("Data")).unwrap();
    std::fs::write(dir.join("Data/Skyrim.esm"), b"TES4 0123456789").unwrap();
    std::fs::write(dir.join("Data/Skyrim.ini"), b"[General]\n").unwrap();
    std::fs::write(dir.join("Data/Update.esm"), b"TES4").unwrap();
    std::fs::write(dir.join("default.xex"), b"XEX2").unwrap();
    dir
}

static NAMES: [ImportName; 1] = [ImportName { module: 0, ordinal: 0x3FF, name: "SyntheticMissingVariable" }];

pub fn kernel_with(stubs: StubConfig) -> &'static Kernel {
    let _ = env_logger_init();
    let module = GuestModule {
        image_base: BASE,
        image_size: IMAGE_SIZE,
        code_base: BASE + CODE_RVA,
        code_size: 0x1000,
        functions: &FUNCS,
        call,
        init_context,
    };
    let rt = Runtime::load(module, synthetic_xex()).expect("runtime");
    Kernel::new(rt, vfs::Vfs::with_game_root(game_dir()), stubs, &NAMES).expect("kernel")
}

pub fn kernel() -> &'static Kernel {
    kernel_with(StubConfig::default())
}

/// The calling host thread as a guest thread, with a context for kernel calls.
pub fn guest(k: &'static Kernel) -> Box<PPCContext> {
    let t = k.adopt_current_thread().expect("adopt thread");
    k.context_for(&t)
}

/// Zeroed scratch memory in the virtual heap.
pub fn scratch(k: &Kernel, size: u32) -> u32 {
    k.rt.space.alloc_virtual(size, 0x1000, true, 0x04, false).unwrap()
}

/// Writes a relative NT timeout (milliseconds) and returns its address.
pub fn timeout_ms(k: &Kernel, at: u32, ms: i64) -> u32 {
    k.mem().write_i64(at, -ms * 10_000);
    at
}

fn env_logger_init() -> Result<(), log::SetLoggerError> {
    struct Stderr;
    impl log::Log for Stderr {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::Level::Warn
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("[{}] {}", r.level(), r.args());
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: Stderr = Stderr;
    log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Warn))
}
