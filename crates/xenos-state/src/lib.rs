//! Xenos register state decoding into host-API-independent draw, render target and resolve descriptions.
//!
//! The crate is pure: it reads register values through [`RegisterSource`] and guest physical memory
//! through [`GuestMemory`], and produces plain descriptions (topology, index buffers, viewport,
//! depth/stencil, blending, EDRAM bindings, resolves) that a WebGPU-style backend can consume.
//!
//! Module overview:
//! - [`registers`]: register indices for the whole Xenos register file.
//! - [`xenos`]: hardware enumerations and small helpers (endian swaps, float24).
//! - [`bitfields`]: typed decoders/encoders for the bitfield registers used by the draw state.
//! - [`primitive`]: draw initiator / index buffer description and guest-to-host topology conversion.
//! - [`shader`]: shader program setup (SQ_* registers, IM_LOAD packets).
//! - [`viewport`]: viewport, scissor, clipping and rasterizer state.
//! - [`depth_stencil`]: depth/stencil state mapped to WebGPU-style enums.
//! - [`blend`]: color output and blending mapped to WebGPU-style enums.
//! - [`edram`]: EDRAM geometry and render target bindings.
//! - [`resolve`]: EDRAM-to-memory copy (resolve) description.
//! - [`tiling`]: Xenos tiled texture address functions used by resolve destinations.

pub mod bitfields;
pub mod blend;
pub mod depth_stencil;
pub mod edram;
pub mod primitive;
pub mod registers;
pub mod resolve;
pub mod shader;
pub mod tiling;
pub mod viewport;
pub mod xenos;

/// Read access to the Xenos register file.
///
/// Values are host-order 32-bit words indexed by register number (see [`registers`]); float
/// registers hold the IEEE 754 bit pattern.
pub trait RegisterSource {
    /// Returns the value of the register, or 0 if the index is out of range.
    fn reg(&self, index: u32) -> u32;

    /// Returns the register reinterpreted as `f32`.
    fn reg_f32(&self, index: u32) -> f32 {
        f32::from_bits(self.reg(index))
    }
}

impl RegisterSource for [u32] {
    fn reg(&self, index: u32) -> u32 {
        self.get(index as usize).copied().unwrap_or(0)
    }
}

impl<const N: usize> RegisterSource for [u32; N] {
    fn reg(&self, index: u32) -> u32 {
        self.as_slice().reg(index)
    }
}

impl RegisterSource for Vec<u32> {
    fn reg(&self, index: u32) -> u32 {
        self.as_slice().reg(index)
    }
}

impl<T: RegisterSource + ?Sized> RegisterSource for &T {
    fn reg(&self, index: u32) -> u32 {
        (**self).reg(index)
    }
}

/// Adapts a closure `Fn(u32) -> u32` into a [`RegisterSource`], useful when part of the register
/// file lives in guest MMIO memory and part on the host.
pub struct RegisterFn<F>(pub F);

impl<F: Fn(u32) -> u32> RegisterSource for RegisterFn<F> {
    fn reg(&self, index: u32) -> u32 {
        (self.0)(index)
    }
}

/// Read access to guest physical memory (addresses as seen by the GPU, 0..512 MiB).
pub trait GuestMemory {
    /// Copies `out.len()` bytes starting at `address` into `out`. Returns `false` if any part of the
    /// range is not readable, in which case the contents of `out` are unspecified.
    fn read_physical(&self, address: u32, out: &mut [u8]) -> bool;
}

impl GuestMemory for [u8] {
    fn read_physical(&self, address: u32, out: &mut [u8]) -> bool {
        let start = address as usize;
        match start.checked_add(out.len()).and_then(|end| self.get(start..end)) {
            Some(src) => {
                out.copy_from_slice(src);
                true
            }
            None => false,
        }
    }
}

impl GuestMemory for Vec<u8> {
    fn read_physical(&self, address: u32, out: &mut [u8]) -> bool {
        self.as_slice().read_physical(address, out)
    }
}

impl<T: GuestMemory + ?Sized> GuestMemory for &T {
    fn read_physical(&self, address: u32, out: &mut [u8]) -> bool {
        (**self).read_physical(address, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_source_impls() {
        let mut regs = vec![0u32; 0x5003];
        regs[registers::RB_SURFACE_INFO as usize] = 0x1234;
        regs[registers::PA_CL_VPORT_XSCALE as usize] = 2.5f32.to_bits();
        assert_eq!(regs.reg(registers::RB_SURFACE_INFO), 0x1234);
        assert_eq!(regs.as_slice().reg_f32(registers::PA_CL_VPORT_XSCALE), 2.5);
        assert_eq!(regs.reg(0x9000), 0);
        let f = RegisterFn(|i| i * 2);
        assert_eq!(f.reg(21), 42);
        let by_ref: &dyn RegisterSource = &f;
        assert_eq!(by_ref.reg(1), 2);
    }

    #[test]
    fn guest_memory_slice() {
        let mem: Vec<u8> = (0..16).collect();
        let mut out = [0u8; 4];
        assert!(mem.read_physical(4, &mut out));
        assert_eq!(out, [4, 5, 6, 7]);
        assert!(!mem.read_physical(14, &mut out));
        assert!(!mem.read_physical(u32::MAX, &mut out));
    }

    #[test]
    fn register_table_lookup() {
        assert_eq!(registers::register_name(registers::RB_COPY_CONTROL), Some("RB_COPY_CONTROL"));
        let info = registers::register_info(registers::PA_CL_VPORT_ZOFFSET).unwrap();
        assert_eq!(info.kind, registers::RegisterKind::Float);
        assert_eq!(registers::register_info(0x0003), None);
        assert!(registers::REGISTERS.windows(2).all(|w| w[0].index < w[1].index));
        assert_eq!(registers::VGT_DRAW_INITIATOR, 0x21FC);
        assert_eq!(registers::SHADER_CONSTANT_FETCH_00_0, 0x4800);
        assert_eq!(registers::RB_BLENDCONTROL3, 0x220B);
    }
}
