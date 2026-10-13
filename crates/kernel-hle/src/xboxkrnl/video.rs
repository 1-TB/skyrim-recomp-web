//! Video driver (Vd*) exports, backed by the GPU stand-in in [`crate::gpu`].
//! Behavior follows Xenia's xboxkrnl_video.cc.
#![allow(non_snake_case)]

use runtime_core::address_space::PAGE_4K;
use runtime_core::heap::protect;

use crate::dispatch::{Call, Export, ModuleId};
use crate::gpu::{SWAP_SIGNATURE, packet_type0, packet_type2, packet_type3, pm4, reg};
use crate::xam::{display_size, write_video_mode};

const K: ModuleId = ModuleId::XBOXKRNL;

pub(super) const EXPORTS: &[Export] = &[
    export!(K, 0x1B1, VdCallGraphicsNotificationRoutines),
    export!(K, 0x1B4, VdEnableDisableClockGating),
    export!(K, 0x1B6, VdEnableRingBufferRPtrWriteBack),
    export!(K, 0x1B9, VdGetCurrentDisplayGamma),
    export!(K, 0x1BA, VdGetCurrentDisplayInformation),
    export!(K, 0x1BC, VdGetGraphicsAsicID),
    export!(K, 0x1BD, VdGetSystemCommandBuffer),
    export!(K, 0x1C2, VdInitializeEngines),
    export!(K, 0x1C3, VdInitializeRingBuffer),
    export!(K, 0x1C5, VdInitializeScalerCommandBuffer),
    export!(K, 0x1C6, VdIsHSIOTrainingSucceeded),
    export!(K, 0x1C7, VdPersistDisplay),
    export!(K, 0x1C9, VdQueryVideoFlags),
    export!(K, 0x1CA, VdQueryVideoMode),
    export!(K, 0x1D3, VdSetDisplayMode),
    export!(K, 0x1D4, VdSetDisplayModeOverride),
    export!(K, 0x1D5, VdSetGraphicsInterruptCallback),
    export!(K, 0x1D9, VdSetSystemCommandBufferGpuIdentifierAddress),
    export!(K, 0x1DC, VdShutdownEngines),
    export!(K, 0x25B, VdSwap),
    export!(K, 0x269, VdRetrainEDRAM),
    export!(K, 0x26A, VdRetrainEDRAMWorker),
    export!(K, 0x2D2, VdQueryRealVideoMode),
];

/// Display gamma: type 2 (power curve) with exponent 2.2222.
const GAMMA_TYPE: u32 = 2;
const GAMMA_POWER: f32 = 2.222_222_3;

fn VdCallGraphicsNotificationRoutines(_c: &mut Call) -> u64 {
    0
}

fn VdEnableDisableClockGating(_c: &mut Call) -> u64 {
    0
}

/// (void* Ptr, u32 BlockSizeLog2)
fn VdEnableRingBufferRPtrWriteBack(c: &mut Call) -> u64 {
    c.kernel.gpu_enable_read_pointer_writeback(c.arg(0), c.arg(1));
    0
}

/// (u32* Type, float* Power)
fn VdGetCurrentDisplayGamma(c: &mut Call) -> u64 {
    let (type_ptr, power_ptr) = (c.arg(0), c.arg(1));
    if type_ptr != 0 {
        c.mem().write_u32(type_ptr, GAMMA_TYPE);
    }
    if power_ptr != 0 {
        c.mem().write_f32(power_ptr, GAMMA_POWER);
    }
    0
}

/// Fills X_DISPLAY_INFO (0x58 bytes).
fn VdGetCurrentDisplayInformation(c: &mut Call) -> u64 {
    let out = c.arg(0);
    let mem = c.mem();
    let (w, h) = display_size(c.kernel);
    mem.fill(out, 0x58, 0);
    mem.write_u16(out, w as u16);
    mem.write_u16(out + 0x2, h as u16);
    // Scaler parameters at 0x8: source rect, scaled output size, filters.
    mem.write_u32(out + 0x8 + 0x8, w);
    mem.write_u32(out + 0x8 + 0xC, h);
    mem.write_u32(out + 0x8 + 0x10, w);
    mem.write_u32(out + 0x8 + 0x14, h);
    mem.write_u32(out + 0x8 + 0x18, 1);
    mem.write_u32(out + 0x8 + 0x28, 1);
    // Overscan: left, top, right, bottom.
    mem.write_u16(out + 0x40, 320);
    mem.write_u16(out + 0x42, 180);
    mem.write_u16(out + 0x44, 320);
    mem.write_u16(out + 0x46, 180);
    mem.write_u16(out + 0x48, w as u16);
    mem.write_u16(out + 0x4A, h as u16);
    mem.write_f32(out + 0x4C, 60.0);
    mem.write_u32(out + 0x50, 0);
    mem.write_u16(out + 0x56, w as u16);
    0
}

fn VdGetGraphicsAsicID(_c: &mut Call) -> u64 {
    0x11
}

/// (void* Buffer, u32* Id): the system command buffer the D3D runtime
/// reserves; contents only need to be consistent with VdSwap.
fn VdGetSystemCommandBuffer(c: &mut Call) -> u64 {
    let (p0, p1) = (c.arg(0), c.arg(1));
    let mem = c.mem();
    mem.fill(p0, 0x94, 0);
    mem.write_u32(p0, 0xBEEF_0000);
    mem.write_u32(p1, 0xBEEF_0001);
    0
}

fn VdInitializeEngines(_c: &mut Call) -> u64 {
    1
}

/// (void* Ptr, int SizeLog2)
fn VdInitializeRingBuffer(c: &mut Call) -> u64 {
    c.kernel.gpu_initialize_ring_buffer(c.arg(0), c.arg(1));
    0
}

/// Fills the scaler command words (argument 10, `count` in argument 11) with
/// type-2 NOP packets; returns the count.
fn VdInitializeScalerCommandBuffer(c: &mut Call) -> u64 {
    let (dest, count) = (c.arg(10), c.arg(11));
    let mem = c.mem();
    for i in 0..count {
        mem.write_u32(dest + i * 4, 0x8000_0000);
    }
    count as u64
}

fn VdIsHSIOTrainingSucceeded(_c: &mut Call) -> u64 {
    1
}

/// (u32 Unknown, u32* Out): hands back a small no-access allocation.
fn VdPersistDisplay(c: &mut Call) -> u64 {
    let out = c.arg(1);
    if out != 0 {
        match c.kernel.rt.space.alloc_physical(64, PAGE_4K, 0, u32::MAX, 32, protect::NOACCESS) {
            Ok(a) => c.mem().write_u32(out, a),
            Err(e) => {
                log::error!("VdPersistDisplay: {e}");
                c.mem().write_u32(out, 0);
            }
        }
    }
    1
}

fn VdQueryVideoFlags(c: &mut Call) -> u64 {
    let (w, h) = display_size(c.kernel);
    let widescreen = w * 9 >= h * 16;
    (widescreen as u64) | ((w >= 1280) as u64) << 1 | ((w >= 1920) as u64) << 2
}

fn VdQueryVideoMode(c: &mut Call) -> u64 {
    write_video_mode(c.kernel, c.arg(0));
    0
}

fn VdQueryRealVideoMode(c: &mut Call) -> u64 {
    write_video_mode(c.kernel, c.arg(0));
    0
}

fn VdSetDisplayMode(_c: &mut Call) -> u64 {
    0
}

fn VdSetDisplayModeOverride(_c: &mut Call) -> u64 {
    0
}

/// (PFN Callback, void* UserData)
fn VdSetGraphicsInterruptCallback(c: &mut Call) -> u64 {
    c.kernel.gpu_set_interrupt_callback(c.arg(0), c.arg(1));
    0
}

fn VdSetSystemCommandBufferGpuIdentifierAddress(_c: &mut Call) -> u64 {
    0
}

fn VdShutdownEngines(_c: &mut Call) -> u64 {
    0
}

fn VdRetrainEDRAM(_c: &mut Call) -> u64 {
    0
}

fn VdRetrainEDRAMWorker(_c: &mut Call) -> u64 {
    0
}

/// Words the D3D runtime reserves in the ring for VdSwap.
const SWAP_PACKET_DWORDS: u32 = 64;

/// (u32* RingSpace, u32* FetchConstant, void* Writeback, void* SystemBuffer,
///  u32* SystemId, u32* FrontBuffer, u32* Format, u32* ColorSpace,
///  u32* Width, u32* Height)
///
/// Writes the front buffer's texture fetch constant (rebased to its physical
/// address) and an XE_SWAP packet into the 64 ring words the caller reserved.
fn VdSwap(c: &mut Call) -> u64 {
    let (buffer, fetch_ptr, frontbuffer_ptr) = (c.arg(0), c.arg(1), c.arg(5));
    let (width_ptr, height_ptr) = (c.arg(8), c.arg(9));
    let mem = c.mem();
    let mut fetch: [u32; 6] = std::array::from_fn(|i| mem.read_u32(fetch_ptr + i as u32 * 4));
    let virt = fetch[1] & 0xFFFF_F000;
    if frontbuffer_ptr != 0 && mem.read_u32(frontbuffer_ptr) != virt {
        log::debug!(
            "VdSwap: front buffer {:#010X} differs from fetch base {virt:#010X}",
            mem.read_u32(frontbuffer_ptr)
        );
    }
    let Some(phys) = c.kernel.rt.space.physical_address(virt) else {
        log::error!("VdSwap: front buffer {virt:#010X} is not physical memory");
        return 0;
    };
    fetch[1] = (fetch[1] & 0xFFF) | (phys & 0xFFFF_F000);
    let width = if width_ptr != 0 { mem.read_u32(width_ptr) } else { 0 };
    let height = if height_ptr != 0 { mem.read_u32(height_ptr) } else { 0 };

    let mut words = Vec::with_capacity(SWAP_PACKET_DWORDS as usize);
    words.push(packet_type0(reg::SHADER_CONSTANT_FETCH_00_0, 6));
    words.extend_from_slice(&fetch);
    words.push(packet_type3(pm4::XE_SWAP, 4));
    words.extend_from_slice(&[SWAP_SIGNATURE, phys, width, height]);
    words.resize(SWAP_PACKET_DWORDS as usize, packet_type2());
    for (i, w) in words.into_iter().enumerate() {
        mem.write_u32(buffer + i as u32 * 4, w);
    }
    0
}
