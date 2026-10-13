//! Rust view of XenonRecomp's `PPCContext`.
//!
//! The layout depends on the recompiler's `*_as_local` / `skip_*` options; this
//! mirrors the configuration with all of them disabled (every register lives in
//! the context), which is what `config/skyrim.toml.template` specifies. The
//! guest-sys shim static_asserts these same offsets against the generated
//! header, so a config/layout mismatch fails the build.
//!
//! Field order in the C struct is r3, r0, r1, r2, r4..r31; [`PPCContext::gpr`]
//! hides that permutation.

/// One 64-bit GPR/FPR slot. The C side is a union; on little-endian hosts the
/// 32-bit view is the low half of the 64-bit value.
pub type Reg = u64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XerRegister {
    pub so: u8,
    pub ov: u8,
    pub ca: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CrField {
    pub lt: u8,
    pub gt: u8,
    pub eq: u8,
    pub so: u8,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VReg(pub [u8; 16]);

#[repr(C, align(64))]
#[derive(Clone)]
pub struct PPCContext {
    /// Storage order r3, r0, r1, r2, r4, ..., r31.
    gpr_storage: [Reg; 32],
    pub lr: u64,
    pub ctr: Reg,
    pub xer: XerRegister,
    pub reserved: Reg,
    pub msr: u32,
    pub cr: [CrField; 8],
    /// Cached host FP control word (`PPCFPSCRRegister::csr`).
    pub fpscr_csr: u32,
    pub fpr: [Reg; 32],
    pub vr: [VReg; 128],
}

pub mod offsets {
    pub const R3: usize = 0;
    pub const R0: usize = 8;
    pub const R1: usize = 16;
    pub const R2: usize = 24;
    pub const R4: usize = 32;
    pub const R13: usize = 104;
    pub const R31: usize = 248;
    pub const LR: usize = 256;
    pub const CTR: usize = 264;
    pub const XER: usize = 272;
    pub const RESERVED: usize = 280;
    pub const MSR: usize = 288;
    pub const CR0: usize = 292;
    pub const FPSCR: usize = 324;
    pub const F0: usize = 328;
    pub const F1: usize = 336;
    pub const V0: usize = 592;
    pub const V127: usize = 2624;
    pub const SIZE: usize = 2688;
    pub const ALIGN: usize = 64;
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(offset_of!(PPCContext, gpr_storage) == offsets::R3);
    assert!(offset_of!(PPCContext, lr) == offsets::LR);
    assert!(offset_of!(PPCContext, ctr) == offsets::CTR);
    assert!(offset_of!(PPCContext, xer) == offsets::XER);
    assert!(offset_of!(PPCContext, reserved) == offsets::RESERVED);
    assert!(offset_of!(PPCContext, msr) == offsets::MSR);
    assert!(offset_of!(PPCContext, cr) == offsets::CR0);
    assert!(offset_of!(PPCContext, fpscr_csr) == offsets::FPSCR);
    assert!(offset_of!(PPCContext, fpr) == offsets::F0);
    assert!(offset_of!(PPCContext, vr) == offsets::V0);
    assert!(size_of::<PPCContext>() == offsets::SIZE);
    assert!(align_of::<PPCContext>() == offsets::ALIGN);
};

/// Storage slot for architectural GPR `n`.
#[inline]
const fn gpr_slot(n: usize) -> usize {
    match n {
        3 => 0,
        0 => 1,
        1 => 2,
        2 => 3,
        n => n,
    }
}

impl PPCContext {
    /// A context in the state the C++ default constructor produces (all zero,
    /// msr = 0x200A000). Use the guest module's `init_context` to also capture the
    /// host FP control word before running guest code.
    pub fn new_zeroed() -> Box<Self> {
        let mut ctx = Box::new(PPCContext {
            gpr_storage: [0; 32],
            lr: 0,
            ctr: 0,
            xer: XerRegister::default(),
            reserved: 0,
            msr: 0x200A000,
            cr: [CrField::default(); 8],
            fpscr_csr: 0,
            fpr: [0; 32],
            vr: [VReg::default(); 128],
        });
        ctx.msr = 0x200A000;
        ctx
    }

    #[inline]
    pub fn gpr(&self, n: usize) -> u64 {
        self.gpr_storage[gpr_slot(n)]
    }

    #[inline]
    pub fn set_gpr(&mut self, n: usize, v: u64) {
        self.gpr_storage[gpr_slot(n)] = v;
    }

    #[inline]
    pub fn gpr32(&self, n: usize) -> u32 {
        self.gpr(n) as u32
    }

    #[inline]
    pub fn fpr_f64(&self, n: usize) -> f64 {
        f64::from_bits(self.fpr[n])
    }

    #[inline]
    pub fn set_fpr_f64(&mut self, n: usize, v: f64) {
        self.fpr[n] = v.to_bits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpr_permutation_matches_c_field_order() {
        let mut ctx = PPCContext::new_zeroed();
        for n in 0..32 {
            ctx.set_gpr(n, n as u64 + 100);
        }
        let raw = &ctx.gpr_storage;
        assert_eq!(raw[0], 103);
        assert_eq!(raw[1], 100);
        assert_eq!(raw[2], 101);
        assert_eq!(raw[3], 102);
        assert_eq!(raw[4], 104);
        assert_eq!(raw[13], 113);
        assert_eq!(ctx.gpr(1), 101);
        assert_eq!(ctx.msr, 0x200A000);
    }
}
