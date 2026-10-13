//! System queries, debug output, and title termination.
#![allow(non_snake_case)]

use runtime_core::layout::{kpcr, kthread};

use crate::dispatch::{Call, Export, ModuleId};
use crate::status;

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x194, XexCheckExecutablePrivilege),
    export!(K, 0x195, XexGetModuleHandle),
    export!(K, 0x010, ExGetXConfigSetting),
    export!(K, 0x003, DbgPrint),
    export!(K, 0x052, KeBugCheck),
    export!(K, 0x053, KeBugCheckEx),
    export!(K, 0x028, HalReturnToFirmware),
    export!(K, 0x05D, KeEnableFpuExceptions),
    export!(K, 0x015, ExRegisterTitleTerminateNotification),
    export!(K, 0x136, RtlRaiseException),
];

fn XexCheckExecutablePrivilege(c: &mut Call) -> u64 {
    let privilege = c.arg(0);
    if privilege >= 32 {
        return 0;
    }
    ((c.kernel.rt.xex.system_flags() >> privilege) & 1) as u64
}

/// (const char* ModuleName, HMODULE* OutModule). A NULL name or the title's own
/// image name yields the executable's LDR entry.
fn XexGetModuleHandle(c: &mut Call) -> u64 {
    let (name_ptr, out) = (c.arg(0), c.arg(1));
    let name = if name_ptr == 0 { Vec::new() } else { c.mem().read_cstr(name_ptr, 260) };
    let name = String::from_utf8_lossy(&name).into_owned();
    let is_self = name.is_empty() || name.eq_ignore_ascii_case("default.xex");
    if is_self {
        if out != 0 {
            c.mem().write_u32(out, c.kernel.vars.executable_ldr_entry);
        }
        return status::win32::SUCCESS as u64;
    }
    log::warn!("XexGetModuleHandle({name:?}): no loaded module by that name");
    if out != 0 {
        c.mem().write_u32(out, 0);
    }
    status::win32::NOT_FOUND as u64
}

/// (u16 Category, u16 Setting, void* Buffer, u16 BufferSize, u16* RequiredSize)
fn ExGetXConfigSetting(c: &mut Call) -> u64 {
    let (category, setting, buffer, buffer_size, required_ptr) =
        (c.arg(0) as u16, c.arg(1) as u16, c.arg(2), c.arg(3) as u16, c.arg(4));
    let mem = c.mem();
    if buffer == 0 && buffer_size != 0 {
        return status::INVALID_PARAMETER_3 as u64;
    }
    let size = c.kernel.xconfig.size(category, setting);
    if size == 0 {
        log::info!("ExGetXConfigSetting: setting {category:#X}/{setting:#X} not in the config store");
    }
    if buffer != 0 {
        if (buffer_size as usize) < size {
            return status::BUFFER_TOO_SMALL as u64;
        }
        if let Some(data) = c.kernel.xconfig.read(category, setting) {
            mem.write_bytes(buffer, &data);
        }
    }
    if required_ptr != 0 {
        mem.write_u16(required_ptr, size as u16);
    }
    status::SUCCESS as u64
}

/// DbgPrint(const char* format, ...): variadic slots follow the format in
/// r4..r10, then the caller's parameter area.
fn DbgPrint(c: &mut Call) -> u64 {
    let fmt_ptr = c.arg(0);
    let fmt = c.mem().read_cstr(fmt_ptr, 4096);
    let mut n = 1;
    let text = crate::dbgprint::format(c.kernel.mem(), &fmt, || {
        let v = c.arg64(n);
        n += 1;
        v
    });
    log::info!(target: "guest", "{}", text.trim_end_matches(['\r', '\n']));
    status::SUCCESS as u64
}

fn bugcheck(c: &mut Call, code: u32, params: [u32; 4]) -> ! {
    log::error!(
        "KeBugCheck {code:#010X} ({:#010X}, {:#010X}, {:#010X}, {:#010X}) on thread {:#X}, guest stack {:08X?}",
        params[0],
        params[1],
        params[2],
        params[3],
        c.tid(),
        c.guest_backtrace(16)
    );
    c.kernel.log_import_summary();
    c.kernel.request_exit(code);
    crate::thread::exit_current_thread(code)
}

fn KeBugCheck(c: &mut Call) -> u64 {
    let code = c.arg(0);
    bugcheck(c, code, [0; 4])
}

fn KeBugCheckEx(c: &mut Call) -> u64 {
    let (code, p) = (c.arg(0), [c.arg(1), c.arg(2), c.arg(3), c.arg(4)]);
    bugcheck(c, code, p)
}

/// The title asked to leave (reboot, return to dashboard, launch another XEX).
fn HalReturnToFirmware(c: &mut Call) -> u64 {
    let routine = c.arg(0);
    log::info!("HalReturnToFirmware({routine}): title exiting");
    c.kernel.request_exit(0);
    crate::thread::exit_current_thread(0)
}

/// (BOOL Enable): recorded per thread; recompiled code never traps on FP.
fn KeEnableFpuExceptions(c: &mut Call) -> u64 {
    let enable = c.arg(0) != 0;
    let pcr = c.ctx.gpr32(13);
    let k = c.mem().read_u32(pcr + kpcr::PRCB_CURRENT_THREAD);
    if k != 0 {
        c.mem().write_u8(k + kthread::FPU_EXCEPTIONS_ON, enable as u8);
    }
    0
}

/// (EX_TITLE_TERMINATE_REGISTRATION* Reg { routine, priority, list }, BOOL Create).
/// Kept so unregistering works; as in Xenia the routines are not invoked when
/// the title exits (they may block on threads that are already gone).
fn ExRegisterTitleTerminateNotification(c: &mut Call) -> u64 {
    let (reg, create) = (c.arg(0), c.arg(1));
    let (routine, priority) = (c.mem().read_u32(reg), c.mem().read_u32(reg + 4));
    let mut list = c.kernel.terminate_notifications.lock().unwrap();
    if create != 0 {
        list.push((routine, priority));
    } else {
        list.retain(|&(r, _)| r != routine);
    }
    0
}

/// EXCEPTION_RECORD fields.
const EXCEPTION_CODE: u32 = 0x0;
const EXCEPTION_ADDRESS: u32 = 0xC;
const EXCEPTION_PARAM_COUNT: u32 = 0x10;
const EXCEPTION_INFO: u32 = 0x14;
/// MSVC SetThreadName.
const EXCEPTION_SET_THREAD_NAME: u32 = 0x406D_1388;
/// MSVC C++ throw.
const EXCEPTION_CPP: u32 = 0xE06D_7363;

/// (EXCEPTION_RECORD* Record). Recompiled code has no structured exception
/// unwinding, so as in Xenia the call returns to the raiser: thread naming is
/// honored, anything else is reported.
fn RtlRaiseException(c: &mut Call) -> u64 {
    let record = c.arg(0);
    let mem = c.mem();
    let code = mem.read_u32(record + EXCEPTION_CODE);
    let info = record + EXCEPTION_INFO;
    match code {
        EXCEPTION_SET_THREAD_NAME => {
            // THREADNAME_INFO { type = 0x1000, name, thread id (-1 = caller), flags }
            let name_ptr = mem.read_u32(info + 4);
            let tid = mem.read_u32(info + 8);
            let name: String = String::from_utf8_lossy(&mem.read_cstr(name_ptr, 64))
                .chars()
                .map(|ch| if (' '..='~').contains(&ch) { ch } else { '?' })
                .collect();
            let thread = if tid == u32::MAX { Some(c.thread()) } else { c.kernel.thread_by_id(tid) };
            match thread {
                Some(t) => {
                    log::info!("thread {:#X} named {name:?}", t.tid());
                    *t.name.lock().unwrap() = Some(name);
                }
                None => log::warn!("SetThreadName({tid:#X}, {name:?}): no such thread"),
            }
        }
        EXCEPTION_CPP => {
            let thrown = mem.read_u32(info + 4);
            let throw_info = mem.read_u32(info + 8);
            log::error!(
                "guest threw a C++ exception (object {thrown:#010X}, throw info {throw_info:#010X}) on thread {:#X}; no handler can run, continuing",
                c.tid()
            );
        }
        _ => {
            let params = mem.read_u32(record + EXCEPTION_PARAM_COUNT).min(15);
            let args: Vec<String> = (0..params).map(|i| format!("{:08X}", mem.read_u32(info + i * 4))).collect();
            log::error!(
                "guest raised exception {code:#010X} at {:#010X} [{}] on thread {:#X}; continuing",
                mem.read_u32(record + EXCEPTION_ADDRESS),
                args.join(" "),
                c.tid()
            );
        }
    }
    0
}
