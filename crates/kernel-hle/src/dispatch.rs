//! Import dispatch keyed by (module, ordinal).
//!
//! The generated shim (guest-sys) defines every import thunk as a call to
//! [`skyrt_kernel_call`] with the import's module id and ordinal. Implemented
//! exports run their handler; anything else is logged with its arguments and
//! returns the configured stub value.

use std::collections::HashMap;
use std::fmt;

use runtime_core::PPCContext;
use runtime_core::memory::GuestMemory;

use crate::Kernel;
use crate::thread::KThread;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleId(pub u32);

impl ModuleId {
    pub const XBOXKRNL: ModuleId = ModuleId(0);
    pub const XAM: ModuleId = ModuleId(1);
    /// Ids from 2 up are other import libraries, numbered by guest-sys in the
    /// order they appear in the XEX.
    pub const FIRST_OTHER: u32 = 2;

    pub fn for_library(name: &str) -> Option<ModuleId> {
        match name.to_ascii_lowercase().as_str() {
            "xboxkrnl.exe" => Some(ModuleId::XBOXKRNL),
            "xam.xex" => Some(ModuleId::XAM),
            _ => None,
        }
    }
}

/// Name of an import as known to the build (from the XEX + export tables).
#[derive(Debug, Clone, Copy)]
pub struct ImportName {
    pub module: u32,
    pub ordinal: u16,
    pub name: &'static str,
}

pub type Handler = fn(&mut Call) -> u64;

#[derive(Clone, Copy)]
pub struct Export {
    pub module: ModuleId,
    pub ordinal: u16,
    pub name: &'static str,
    pub handler: Handler,
}

impl fmt::Debug for Export {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{:#05X} {}", self.module.0, self.ordinal, self.name)
    }
}

/// Arguments and state for one import call.
pub struct Call<'a> {
    pub kernel: &'static Kernel,
    pub ctx: &'a mut PPCContext,
    pub module: ModuleId,
    pub ordinal: u16,
}

impl Call<'_> {
    pub fn mem(&self) -> &GuestMemory {
        self.kernel.mem()
    }

    /// 32-bit argument `n` (0-based): r3..r10, then the caller's parameter area
    /// at r1+0x54+8*(n-8) (low word of each 8-byte slot).
    pub fn arg(&self, n: usize) -> u32 {
        if n < 8 {
            self.ctx.gpr32(3 + n)
        } else {
            let sp = self.ctx.gpr32(1);
            self.mem().read_u32(sp + 0x54 + (n as u32 - 8) * 8)
        }
    }

    pub fn arg_i32(&self, n: usize) -> i32 {
        self.arg(n) as i32
    }

    pub fn arg64(&self, n: usize) -> u64 {
        if n < 8 {
            self.ctx.gpr(3 + n)
        } else {
            let sp = self.ctx.gpr32(1);
            self.mem().read_u64(sp + 0x50 + (n as u32 - 8) * 8)
        }
    }

    /// The calling guest thread.
    pub fn thread(&self) -> std::sync::Arc<KThread> {
        crate::thread::current().expect("kernel call from a host thread that is not a guest thread")
    }

    pub fn tid(&self) -> u32 {
        self.thread().block.thread_id
    }

    /// Return addresses of the calling guest code, innermost first: LR, then
    /// the saved LR of each frame on the back chain (stored at -8 from the
    /// caller's stack pointer, as Xbox 360 prologues do). For diagnostics.
    pub fn guest_backtrace(&self, depth: usize) -> Vec<u32> {
        let stack = crate::thread::current().map(|t| t.block.stack);
        let mut out = vec![self.ctx.lr as u32];
        let mut sp = self.ctx.gpr32(1);
        while out.len() < depth {
            let Some(s) = stack else { break };
            if sp < s.limit || sp >= s.base {
                break;
            }
            let next = self.mem().read_u32(sp);
            if next <= sp || next > s.base {
                break;
            }
            out.push(self.mem().read_u32(next - 8));
            sp = next;
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ImportStats {
    pub calls: u64,
    pub implemented: bool,
}

/// Kernel calls taking at least this many milliseconds are logged with their caller (0 = off;
/// `--slow-calls MS`), for finding what a stalled title thread waits on.
pub static SLOW_CALL_MS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Unimplemented imports log every call up to this many, then every Nth.
const UNIMPLEMENTED_LOG_ALL: u64 = 64;
const UNIMPLEMENTED_LOG_EVERY: u64 = 1024;

#[derive(Debug, Clone, Default)]
pub struct StubConfig {
    /// Returned in r3 by unimplemented imports.
    pub default_return: u64,
    pub overrides: HashMap<(ModuleId, u16), u64>,
}

impl StubConfig {
    /// Parses `module:ordinal=value` (module is `xboxkrnl`, `xam`, or a numeric
    /// id; numbers accept `0x`).
    pub fn parse_override(spec: &str) -> Result<((ModuleId, u16), u64), String> {
        let (key, value) = spec.split_once('=').ok_or("expected module:ordinal=value")?;
        let (module, ordinal) = key.split_once(':').ok_or("expected module:ordinal")?;
        let module = match module.to_ascii_lowercase().as_str() {
            "xboxkrnl" | "xboxkrnl.exe" => ModuleId::XBOXKRNL,
            "xam" | "xam.xex" => ModuleId::XAM,
            other => ModuleId(parse_num(other)? as u32),
        };
        Ok(((module, parse_num(ordinal)? as u16), parse_num(value)?))
    }
}

pub fn parse_num(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<i64>().map(|v| v as u64).or_else(|_| s.parse::<u64>()),
    };
    r.map_err(|e| format!("bad number {s:?}: {e}"))
}

impl Kernel {
    pub fn import_name(&self, module: ModuleId, ordinal: u16) -> String {
        if let Some(e) = self.exports.get(&(module, ordinal)) {
            return e.name.to_string();
        }
        if let Some(n) = self.import_names.get(&(module, ordinal)) {
            return n.to_string();
        }
        format!("{}_{:03X}", self.module_name(module).trim_end_matches(".exe").trim_end_matches(".xex"), ordinal)
    }

    pub fn module_name(&self, module: ModuleId) -> &str {
        self.module_names.get(module.0 as usize).map(String::as_str).unwrap_or("unknown")
    }

    /// Runs one import call. The return value goes to r3.
    pub fn dispatch(&'static self, module: ModuleId, ordinal: u16, ctx: &mut PPCContext) {
        crate::thread::note_kernel_call();
        let export = self.exports.get(&(module, ordinal)).copied();
        let counter = self.stats.read().unwrap().get(&(module, ordinal)).cloned();
        let counter =
            counter.unwrap_or_else(|| self.stats.write().unwrap().entry((module, ordinal)).or_default().clone());
        let calls = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let args: [u32; 8] = std::array::from_fn(|i| ctx.gpr32(3 + i));

        match export {
            Some(e) => {
                if calls == 1 {
                    log::info!("first call: {}!{} ({:#05X})", self.module_name(module), e.name, ordinal);
                }
                let mut call = Call { kernel: self, ctx, module, ordinal };
                let slow_ms = SLOW_CALL_MS.load(std::sync::atomic::Ordering::Relaxed);
                let started = (slow_ms != 0).then(std::time::Instant::now);
                if slow_ms != 0 {
                    crate::thread::set_caller_lr(call.ctx.lr as u32);
                }
                let ret = (e.handler)(&mut call);
                if let Some(started) = started {
                    let took = started.elapsed();
                    if took.as_millis() >= u128::from(slow_ms) {
                        log::info!(
                            "slow call: {}({:08X} {:08X} {:08X}) took {:.0} ms, from {:08X?}",
                            e.name,
                            args[0],
                            args[1],
                            args[2],
                            took.as_secs_f64() * 1000.0,
                            call.guest_backtrace(6)
                        );
                    }
                }
                log::trace!(
                    "{}!{}({:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}) = {:#X}",
                    self.module_name(module),
                    e.name,
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    args[7],
                    ret
                );
                call.ctx.set_gpr(3, ret);
            }
            None => {
                let ret = self.stubs.overrides.get(&(module, ordinal)).copied().unwrap_or(self.stubs.default_return);
                if calls <= UNIMPLEMENTED_LOG_ALL || calls % UNIMPLEMENTED_LOG_EVERY == 0 {
                    log::warn!(
                        "unimplemented import {}!{} (ordinal {:#05X}, call #{calls}) args \
                         [{:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}] sp={:08X} -> stub {:#X}",
                        self.module_name(module),
                        self.import_name(module, ordinal),
                        ordinal,
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[5],
                        args[6],
                        args[7],
                        ctx.gpr32(1),
                        ret
                    );
                }
                ctx.set_gpr(3, ret);
            }
        }
    }

    /// Every import called so far, sorted by module and ordinal.
    pub fn import_stats(&self) -> Vec<(ModuleId, u16, String, ImportStats)> {
        let stats = self.stats.read().unwrap();
        let mut v: Vec<_> = stats
            .iter()
            .map(|(&(m, o), n)| {
                let s = ImportStats {
                    calls: n.load(std::sync::atomic::Ordering::Relaxed),
                    implemented: self.exports.contains_key(&(m, o)),
                };
                (m, o, self.import_name(m, o), s)
            })
            .collect();
        v.sort_by_key(|(m, o, _, _)| (*m, *o));
        v
    }

    /// Logs every guest thread blocked in a wait, with the objects it waits on.
    pub fn log_waiters(&self) {
        for w in self.objects.waiters() {
            if w.timed {
                continue;
            }
            let objects: Vec<String> = w.objects.iter().map(|(p, s)| format!("{p:#010X} {s:?}")).collect();
            log::info!(
                "thread {:#X} waiting ({}) on [{}] from {:08X?}",
                w.tid,
                if w.wait_all { "all" } else { "any" },
                objects.join(", "),
                w.site
            );
        }
    }

    pub fn log_import_summary(&self) {
        let stats = self.import_stats();
        let unimplemented = stats.iter().filter(|s| !s.3.implemented).count();
        log::info!("imports called: {} ({} unimplemented)", stats.len(), unimplemented);
        for (m, o, name, s) in stats {
            log::info!(
                "  {:<12} {:#05X} {:<40} {:>10} calls{}",
                self.module_name(m),
                o,
                name,
                s.calls,
                if s.implemented { "" } else { "  UNIMPLEMENTED" }
            );
        }
    }
}

/// Called by the generated import thunks.
///
/// # Safety
/// `ctx` must point to the calling thread's live `PPCContext`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn skyrt_kernel_call(module: u32, ordinal: u32, ctx: *mut PPCContext, _base: *mut u8) {
    let kernel = Kernel::global();
    // SAFETY: the shim passes the context of the thread executing the call.
    let ctx = unsafe { &mut *ctx };
    kernel.dispatch(ModuleId(module), ordinal as u16, ctx);
}
