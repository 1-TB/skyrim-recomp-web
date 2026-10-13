//! Exported kernel variables and import-table patching.
//!
//! XenonRecomp only turns function imports into calls; every import's record
//! slot still holds the loader's ordinal word. At boot each slot is patched:
//! variable imports get the guest address of the variable, function imports
//! get the address of their thunk (so code that loads the slot and calls
//! through it lands in the same recompiled thunk).

use std::time::Duration;

use runtime_core::AddressSpace;
use runtime_core::address_space::PAGE_4K;
use runtime_core::heap::protect;
use runtime_core::memory::GuestMemory;
use runtime_core::timebase;

use crate::dispatch::ModuleId;
use crate::strings::{utf16, write_string_header};

pub mod ordinals {
    pub const EX_CONSOLE_GAME_REGION: u16 = 0x00C;
    pub const EX_EVENT_OBJECT_TYPE: u16 = 0x00E;
    pub const EX_MUTANT_OBJECT_TYPE: u16 = 0x012;
    pub const EX_SEMAPHORE_OBJECT_TYPE: u16 = 0x017;
    pub const EX_THREAD_OBJECT_TYPE: u16 = 0x01B;
    pub const EX_TIMER_OBJECT_TYPE: u16 = 0x01C;
    pub const IO_COMPLETION_OBJECT_TYPE: u16 = 0x036;
    pub const IO_DEVICE_OBJECT_TYPE: u16 = 0x03A;
    pub const IO_FILE_OBJECT_TYPE: u16 = 0x03E;
    pub const KE_DEBUG_MONITOR_DATA: u16 = 0x059;
    pub const KE_TIME_STAMP_BUNDLE: u16 = 0x0AD;
    pub const OB_DIRECTORY_OBJECT_TYPE: u16 = 0x106;
    pub const OB_SYMBOLIC_LINK_OBJECT_TYPE: u16 = 0x112;
    pub const XBOX_HARDWARE_INFO: u16 = 0x156;
    pub const XBOX_KRNL_BASE_VERSION: u16 = 0x157;
    pub const XBOX_KRNL_VERSION: u16 = 0x158;
    pub const XEX_EXECUTABLE_MODULE_HANDLE: u16 = 0x193;
    pub const VD_GLOBAL_DEVICE: u16 = 0x1BE;
    pub const VD_GLOBAL_XAM_DEVICE: u16 = 0x1BF;
    pub const VD_GPU_CLOCK_IN_MHZ: u16 = 0x1C0;
    pub const VD_HSIO_CALIBRATION_LOCK: u16 = 0x1C1;
    pub const EX_LOADED_COMMAND_LINE: u16 = 0x1AE;
    pub const EX_LOADED_IMAGE_NAME: u16 = 0x1AF;
    pub const KE_CERT_MONITOR_DATA: u16 = 0x266;
}

pub const LOADED_IMAGE_NAME: &str = "\\Device\\Cdrom0\\default.xex";
pub const COMMAND_LINE: &str = "\"default.xex\"";

/// Size of X_OBJECT_TYPE.
const OBJECT_TYPE_SIZE: u32 = 0x1C;

/// Guest addresses of everything the kernel exports or hands out.
#[derive(Debug, Clone)]
pub struct Variables {
    pub ex_console_game_region: u32,
    pub ke_time_stamp_bundle: u32,
    pub xbox_hardware_info: u32,
    pub xbox_krnl_version: u32,
    pub xbox_krnl_base_version: u32,
    pub xex_executable_module_handle: u32,
    pub ex_loaded_image_name: u32,
    pub ex_loaded_command_line: u32,
    pub ke_debug_monitor_data: u32,
    pub ke_cert_monitor_data: u32,
    pub thread_object_type: u32,
    pub event_object_type: u32,
    pub mutant_object_type: u32,
    pub semaphore_object_type: u32,
    pub timer_object_type: u32,
    pub io_completion_object_type: u32,
    pub io_device_object_type: u32,
    pub io_file_object_type: u32,
    pub ob_directory_object_type: u32,
    pub ob_symbolic_link_object_type: u32,
    /// LDR_DATA_TABLE_ENTRY of the title executable (its HMODULE).
    pub executable_ldr_entry: u32,
    /// Pointer to the video driver's device block (physical memory).
    pub vd_global_device: u32,
    pub vd_global_xam_device: u32,
    pub vd_gpu_clock_in_mhz: u32,
    /// RTL_CRITICAL_SECTION; initialized by [`crate::Kernel::new`].
    pub vd_hsio_calibration_lock: u32,
}

/// Size of RTL_CRITICAL_SECTION.
pub const CRITICAL_SECTION_SIZE: u32 = 0x1C;
/// Size of the block VdGlobalDevice points at (as in Xenia).
const VD_DEVICE_SIZE: u32 = 40;
const GPU_CLOCK_MHZ: u32 = 500;

fn alloc(space: &AddressSpace, size: u32) -> u32 {
    space.system_alloc(size, 16).expect("system heap exhausted while creating kernel variables")
}

fn object_type(space: &AddressSpace, pool_tag: &[u8; 4]) -> u32 {
    let p = alloc(space, OBJECT_TYPE_SIZE);
    space.mem.write_bytes(p + 0x18, pool_tag);
    p
}

fn write_kernel_version(mem: &GuestMemory, addr: u32, build: u16) {
    mem.write_u16(addr, 2);
    mem.write_u16(addr + 2, 0);
    mem.write_u16(addr + 4, build);
    mem.write_u16(addr + 6, 0);
}

impl Variables {
    pub fn create(space: &AddressSpace, image_base: u32, image_size: u32, entry: u32, header_addr: u32) -> Self {
        let mem = &space.mem;

        let ex_console_game_region = alloc(space, 4);
        mem.write_u16(ex_console_game_region, 0xFFFF);

        // { u64 InterruptTime; u64 SystemTime; u32 TickCount; u32 Padding; }
        let ke_time_stamp_bundle = alloc(space, 0x18);
        update_time_stamp_bundle(mem, ke_time_stamp_bundle);

        // { u32 Flags; u8 NumberOfProcessors; ... } (16 bytes)
        let xbox_hardware_info = alloc(space, 0x10);
        mem.write_u32(xbox_hardware_info, 0x20);
        mem.write_u8(xbox_hardware_info + 4, 6);

        let xbox_krnl_version = alloc(space, 8);
        write_kernel_version(mem, xbox_krnl_version, 17559);
        let xbox_krnl_base_version = alloc(space, 8);
        write_kernel_version(mem, xbox_krnl_base_version, 1888);

        // Pointers to debugger/cert monitor callbacks: none attached.
        let ke_debug_monitor_data = alloc(space, 4);
        let ke_cert_monitor_data = alloc(space, 4);

        let ex_loaded_image_name = alloc(space, LOADED_IMAGE_NAME.len() as u32 + 1);
        mem.write_bytes(ex_loaded_image_name, LOADED_IMAGE_NAME.as_bytes());
        let ex_loaded_command_line = alloc(space, COMMAND_LINE.len() as u32 + 1);
        mem.write_bytes(ex_loaded_command_line, COMMAND_LINE.as_bytes());

        let thread_object_type = object_type(space, b"Thre");
        let event_object_type = object_type(space, b"Even");
        let mutant_object_type = object_type(space, b"Muta");
        let semaphore_object_type = object_type(space, b"Sema");
        let timer_object_type = object_type(space, b"Time");
        let io_completion_object_type = object_type(space, b"Comp");
        let io_device_object_type = object_type(space, b"Devi");
        let io_file_object_type = object_type(space, b"File");
        let ob_directory_object_type = object_type(space, b"Dire");
        let ob_symbolic_link_object_type = object_type(space, b"Symb");

        let executable_ldr_entry = create_ldr_entry(space, image_base, image_size, entry, header_addr);
        let xex_executable_module_handle = alloc(space, 4);
        mem.write_u32(xex_executable_module_handle, executable_ldr_entry);

        let device = space
            .alloc_physical(VD_DEVICE_SIZE, PAGE_4K, 0, u32::MAX, 32, protect::READWRITE)
            .expect("physical memory exhausted while creating kernel variables");
        let vd_global_device = alloc(space, 4);
        mem.write_u32(vd_global_device, device);
        let vd_global_xam_device = alloc(space, 4);
        let vd_gpu_clock_in_mhz = alloc(space, 4);
        mem.write_u32(vd_gpu_clock_in_mhz, GPU_CLOCK_MHZ);
        let vd_hsio_calibration_lock = alloc(space, CRITICAL_SECTION_SIZE);
        Variables {
            ex_console_game_region,
            ke_time_stamp_bundle,
            xbox_hardware_info,
            xbox_krnl_version,
            xbox_krnl_base_version,
            ke_debug_monitor_data,
            ke_cert_monitor_data,
            ex_loaded_image_name,
            ex_loaded_command_line,
            thread_object_type,
            event_object_type,
            mutant_object_type,
            semaphore_object_type,
            timer_object_type,
            io_completion_object_type,
            io_device_object_type,
            io_file_object_type,
            ob_directory_object_type,
            ob_symbolic_link_object_type,
            executable_ldr_entry,
            xex_executable_module_handle,
            vd_global_device,
            vd_global_xam_device,
            vd_gpu_clock_in_mhz,
            vd_hsio_calibration_lock,
        }
    }

    /// Guest address for an exported variable, if this kernel provides it.
    pub fn address_of(&self, module: ModuleId, ordinal: u16) -> Option<u32> {
        use ordinals::*;
        if module != ModuleId::XBOXKRNL {
            return None;
        }
        Some(match ordinal {
            EX_CONSOLE_GAME_REGION => self.ex_console_game_region,
            EX_EVENT_OBJECT_TYPE => self.event_object_type,
            EX_MUTANT_OBJECT_TYPE => self.mutant_object_type,
            EX_SEMAPHORE_OBJECT_TYPE => self.semaphore_object_type,
            EX_THREAD_OBJECT_TYPE => self.thread_object_type,
            EX_TIMER_OBJECT_TYPE => self.timer_object_type,
            IO_COMPLETION_OBJECT_TYPE => self.io_completion_object_type,
            IO_DEVICE_OBJECT_TYPE => self.io_device_object_type,
            IO_FILE_OBJECT_TYPE => self.io_file_object_type,
            KE_DEBUG_MONITOR_DATA => self.ke_debug_monitor_data,
            KE_TIME_STAMP_BUNDLE => self.ke_time_stamp_bundle,
            OB_DIRECTORY_OBJECT_TYPE => self.ob_directory_object_type,
            OB_SYMBOLIC_LINK_OBJECT_TYPE => self.ob_symbolic_link_object_type,
            XBOX_HARDWARE_INFO => self.xbox_hardware_info,
            XBOX_KRNL_BASE_VERSION => self.xbox_krnl_base_version,
            XBOX_KRNL_VERSION => self.xbox_krnl_version,
            XEX_EXECUTABLE_MODULE_HANDLE => self.xex_executable_module_handle,
            EX_LOADED_COMMAND_LINE => self.ex_loaded_command_line,
            EX_LOADED_IMAGE_NAME => self.ex_loaded_image_name,
            KE_CERT_MONITOR_DATA => self.ke_cert_monitor_data,
            VD_GLOBAL_DEVICE => self.vd_global_device,
            VD_GLOBAL_XAM_DEVICE => self.vd_global_xam_device,
            VD_GPU_CLOCK_IN_MHZ => self.vd_gpu_clock_in_mhz,
            VD_HSIO_CALIBRATION_LOCK => self.vd_hsio_calibration_lock,
            _ => return None,
        })
    }
}

/// LDR_DATA_TABLE_ENTRY layout per Xenia xmodule.h.
fn create_ldr_entry(space: &AddressSpace, image_base: u32, image_size: u32, entry: u32, header_addr: u32) -> u32 {
    let mem = &space.mem;
    let e = alloc(space, 0x64);
    for list in [0x0, 0x8, 0x10] {
        runtime_core::thread::init_list_head(mem, e + list);
    }
    mem.write_u32(e + 0x18, image_base);
    mem.write_u32(e + 0x1C, image_base);
    mem.write_u32(e + 0x20, image_size);
    let name = utf16("default.xex");
    let name_buf = alloc(space, (name.len() as u32 + 1) * 2);
    mem.write_utf16(name_buf, &name);
    let len = (name.len() * 2) as u16;
    write_string_header(mem, e + 0x24, len, len + 2, name_buf);
    write_string_header(mem, e + 0x2C, len, len + 2, name_buf);
    mem.write_u32(e + 0x38, image_size);
    mem.write_u32(e + 0x3C, entry);
    mem.write_u16(e + 0x40, 1);
    mem.write_u32(e + 0x44, image_base);
    mem.write_u32(e + 0x58, header_addr);
    e
}

pub fn update_time_stamp_bundle(mem: &GuestMemory, addr: u32) {
    mem.write_u64(addr, timebase::interrupt_time());
    mem.write_u64(addr + 8, timebase::system_time());
    mem.write_u32(addr + 0x10, timebase::tick_count());
}

/// Keeps KeTimeStampBundle current (the kernel's clock interrupt does this
/// every millisecond on hardware).
pub fn spawn_time_stamp_updater(mem: &'static GuestMemory, addr: u32) {
    let spawned = std::thread::Builder::new().name("ke-timestamp".into()).spawn(move || {
        loop {
            update_time_stamp_bundle(mem, addr);
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    if let Err(e) = spawned {
        log::error!("could not start KeTimeStampBundle updater: {e}");
    }
}
