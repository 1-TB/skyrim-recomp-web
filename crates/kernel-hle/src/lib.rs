//! High-level emulation of the Xbox 360 kernel (xboxkrnl.exe) and XAM (xam.xex)
//! imports for recompiled titles.

/// `export!(module, ordinal, Name)` registers the handler function `Name`.
macro_rules! export {
    ($module:expr, $ordinal:expr, $name:ident) => {
        $crate::dispatch::Export { module: $module, ordinal: $ordinal, name: stringify!($name), handler: $name }
    };
}

pub mod audio;
pub mod d3d;
pub mod dbgprint;
pub mod dispatch;
pub mod gpu;
pub mod input;
pub mod io;
pub mod logging;
pub mod objects;
pub mod overlapped;
pub mod status;
pub mod strings;
pub mod thread;
pub mod variables;
pub mod xam;
pub mod xboxkrnl;
pub mod xconfig;
pub mod xma;
pub mod zlib;

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use runtime_core::Runtime;
use runtime_core::memory::GuestMemory;
use vfs::Vfs;

use dispatch::{Export, ImportName, ModuleId, StubConfig};
use objects::ObjectManager;
use thread::KThread;
use variables::Variables;
use xconfig::XConfig;

/// Every export this kernel implements.
pub fn all_exports() -> impl Iterator<Item = &'static Export> {
    xboxkrnl::EXPORTS.iter().chain(xam::EXPORTS.iter())
}

/// Module ids for a XEX's import libraries, in library order: xboxkrnl.exe is 0,
/// xam.xex is 1, any other library gets the next id from 2 upwards. guest-sys
/// uses this same function when generating the import thunks.
pub fn assign_module_ids(library_names: &[String]) -> Vec<ModuleId> {
    let mut next = ModuleId::FIRST_OTHER;
    let mut others: HashMap<String, ModuleId> = HashMap::new();
    library_names
        .iter()
        .map(|n| {
            ModuleId::for_library(n).unwrap_or_else(|| {
                *others.entry(n.to_ascii_lowercase()).or_insert_with(|| {
                    let id = ModuleId(next);
                    next += 1;
                    id
                })
            })
        })
        .collect()
}

pub struct Kernel {
    pub rt: Runtime,
    pub vfs: Vfs,
    pub objects: ObjectManager,
    pub vars: Variables,
    pub xconfig: XConfig,
    pub gpu: gpu::Gpu,
    pub audio: audio::AudioSystem,
    pub xma: xma::XmaDecoder,
    pub input: input::InputSystem,
    pub users: xam::user::Users,
    pub content: xam::content::ContentManager,
    pub(crate) timers: xboxkrnl::Timers,
    pub(crate) ldi: xboxkrnl::LdiContexts,
    pub(crate) notifications: xam::notify::Notifications,
    pub(crate) media: xam::msg::MediaPlayer,
    pub(crate) profile_settings: xam::profile::ProfileSettings,
    /// ExRegisterTitleTerminateNotification registrations (routine, priority).
    pub(crate) terminate_notifications: Mutex<Vec<(u32, u32)>>,
    pub(crate) stubs: StubConfig,
    pub(crate) exports: HashMap<(ModuleId, u16), Export>,
    pub(crate) import_names: HashMap<(ModuleId, u16), &'static str>,
    pub(crate) module_names: Vec<String>,
    /// Call counters per import; read-mostly, so counting never parks a thread.
    pub(crate) stats: std::sync::RwLock<HashMap<(ModuleId, u16), Arc<std::sync::atomic::AtomicU64>>>,
    pub(crate) threads: Mutex<HashMap<u32, Arc<KThread>>>,
    /// Allocated dynamic TLS slots.
    pub(crate) tls_slots: Mutex<Vec<bool>>,
    exit_code: Mutex<Option<u32>>,
    exit_cv: Condvar,
}

static GLOBAL: OnceLock<&'static Kernel> = OnceLock::new();

impl Kernel {
    /// Builds the kernel for a loaded title: creates the exported variables,
    /// patches the import table and starts the clock updater. The kernel lives
    /// for the rest of the process.
    pub fn new(
        mut rt: Runtime,
        vfs: Vfs,
        stubs: StubConfig,
        import_names: &'static [ImportName],
    ) -> Result<&'static Kernel, String> {
        let vars =
            Variables::create(&rt.space, rt.image.base, rt.image.size, rt.image.entry_point, rt.image.header_addr);

        let libraries = rt.xex.import_libraries().map_err(|e| e.to_string())?;
        let lib_names: Vec<String> = libraries.iter().map(|l| l.name.clone()).collect();
        let ids = assign_module_ids(&lib_names);
        let mut module_names = vec!["xboxkrnl.exe".to_string(), "xam.xex".to_string()];
        for (name, id) in lib_names.iter().zip(&ids) {
            let i = id.0 as usize;
            if module_names.len() <= i {
                module_names.resize(i + 1, String::new());
            }
            module_names[i] = name.clone();
        }

        // The image is in guest memory now; its copy (tens of MiB) only served the import records.
        let imports = rt.xex.imports().map_err(|e| e.to_string())?;
        rt.xex.discard_image();

        let tls_count = rt.threads.tls_template().slot_count;
        let tls_count = if tls_count != 0 { tls_count } else { runtime_core::thread::DEFAULT_TLS_SLOTS };

        let kernel: &'static Kernel = Box::leak(Box::new(Kernel {
            vfs,
            objects: ObjectManager::new(),
            xconfig: XConfig::new(),
            gpu: gpu::Gpu::new(),
            audio: audio::AudioSystem::default(),
            xma: xma::XmaDecoder::default(),
            input: input::InputSystem::new(),
            users: xam::user::Users::default(),
            content: xam::content::ContentManager::default(),
            timers: xboxkrnl::Timers::default(),
            ldi: xboxkrnl::LdiContexts::default(),
            notifications: xam::notify::Notifications::default(),
            media: xam::msg::MediaPlayer::default(),
            profile_settings: xam::profile::ProfileSettings::default(),
            terminate_notifications: Mutex::new(Vec::new()),
            stubs,
            exports: all_exports().map(|e| ((e.module, e.ordinal), *e)).collect(),
            import_names: import_names.iter().map(|n| ((ModuleId(n.module), n.ordinal), n.name)).collect(),
            module_names,
            stats: std::sync::RwLock::new(HashMap::new()),
            threads: Mutex::new(HashMap::new()),
            tls_slots: Mutex::new(vec![false; tls_count as usize]),
            exit_code: Mutex::new(None),
            exit_cv: Condvar::new(),
            vars,
            rt,
        }));

        kernel.patch_imports(&ids, &imports)?;
        // Video driver state the title reads before calling any Vd function.
        xboxkrnl::init_critical_section_at(
            kernel,
            kernel.vars.vd_hsio_calibration_lock,
            xboxkrnl::spin_count_field(10000),
        );
        kernel.gpu.init_registers(kernel.mem());
        variables::spawn_time_stamp_updater(kernel.mem(), kernel.vars.ke_time_stamp_bundle);
        Ok(kernel)
    }

    fn patch_imports(&self, ids: &[ModuleId], imports: &[xex::Import]) -> Result<(), String> {
        let libraries = self.rt.xex.import_libraries().map_err(|e| e.to_string())?;
        let id_of =
            |lib: &str| libraries.iter().position(|l| l.name == lib).map(|i| ids[i]).unwrap_or(ModuleId(u32::MAX));
        let mem = self.mem();
        let (mut functions, mut variables) = (0, 0);
        for imp in imports {
            let module = id_of(&imp.library);
            match imp.thunk_addr {
                Some(thunk) => {
                    mem.write_u32(imp.record_addr, thunk);
                    functions += 1;
                }
                None => {
                    let addr = match self.vars.address_of(module, imp.ordinal) {
                        Some(a) => a,
                        None => {
                            let a = self
                                .rt
                                .space
                                .system_alloc(0x100, 16)
                                .map_err(|e| format!("allocating placeholder variable: {e}"))?;
                            log::warn!(
                                "unimplemented variable import {}!{} (ordinal {:#05X}): slot {:#010X} -> zeroed block {a:#010X}",
                                imp.library,
                                self.import_name(module, imp.ordinal),
                                imp.ordinal,
                                imp.record_addr
                            );
                            a
                        }
                    };
                    mem.write_u32(imp.record_addr, addr);
                    variables += 1;
                }
            }
        }
        log::info!("patched {functions} function and {variables} variable import slots");
        Ok(())
    }

    /// Makes this kernel the one the generated import thunks call into.
    pub fn install_global(&'static self) {
        if GLOBAL.set(self).is_err() {
            panic!("a kernel is already installed");
        }
    }

    /// The installed kernel, if boot has got that far.
    pub fn try_global() -> Option<&'static Kernel> {
        GLOBAL.get().copied()
    }

    pub fn global() -> &'static Kernel {
        GLOBAL.get().expect("kernel import called before Kernel::install_global")
    }

    pub fn mem(&self) -> &GuestMemory {
        &self.rt.space.mem
    }

    /// Records the title's exit (main thread finished, or a terminate-title call).
    pub fn request_exit(&self, code: u32) {
        let mut c = self.exit_code.lock().unwrap();
        if c.is_none() {
            *c = Some(code);
        }
        self.exit_cv.notify_all();
    }

    /// Blocks until the title exits; returns its exit code.
    pub fn wait_for_exit(&self) -> u32 {
        let mut c = self.exit_code.lock().unwrap();
        loop {
            if let Some(code) = *c {
                return code;
            }
            c = self.exit_cv.wait(c).unwrap();
        }
    }

    /// Like [`Kernel::wait_for_exit`], giving up after `timeout`.
    pub fn wait_for_exit_timeout(&self, timeout: std::time::Duration) -> Option<u32> {
        let c = self.exit_code.lock().unwrap();
        let (c, _) = self.exit_cv.wait_timeout_while(c, timeout, |c| c.is_none()).unwrap();
        *c
    }

    /// Calls an implemented export by name from host code (tests, tools).
    pub fn call_export(&'static self, name: &str, ctx: &mut runtime_core::PPCContext, args: &[u32]) -> u64 {
        let e = self.exports.values().find(|e| e.name == name).unwrap_or_else(|| panic!("no export named {name}"));
        for (i, &a) in args.iter().enumerate() {
            if i < 8 {
                ctx.set_gpr(3 + i, a as u64);
            } else {
                // Caller's parameter area, as Call::arg reads it.
                let sp = ctx.gpr32(1);
                self.mem().write_u32(sp + 0x54 + (i as u32 - 8) * 8, a);
            }
        }
        self.dispatch(e.module, e.ordinal, ctx);
        ctx.gpr(3)
    }
}
