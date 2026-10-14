//! Layout of the D3D device object.
//!
//! The device keeps a copy of the GPU registers it controls and writes the changed ones into the
//! command buffer when a draw is issued: the draw functions ([`crate::HookKind::DrawLeaf`]) test
//! the dirty qwords at the start of the device and call a flush helper per register block, which
//! emits a type-0 packet for each run of dirty registers, reading register `first + i` from
//! `offset + 4 * i`. [`BLOCKS`] lists those calls (from `sub_823B7508`, which `sub_823B3798`,
//! `sub_823E2D60` and `sub_823AA6A0` repeat):
//!
//! | flush helper | dirty bits | blocks |
//! |---|---|---|
//! | `sub_823B73B8` (16 dwords per bit) | dirty[0], dirty[1] (64 each) | VS, PS float constants |
//! | `sub_823B7B28` (1 dword per bit) | dirty[2], dirty[3], dirty[4] | RB/PA/VGT/SQ blocks, bool/loop constants |
//! | `sub_823B7980` (6 dwords per bit) | dirty[3] low 32 bits | fetch constants |
//! | `sub_8240F100` (4 dwords per bit) | dirty[4] bits 49..54 | user clip planes |
//!
//! Everything else the draw emits (shader uploads in `sub_823B3D88`, the HiZ and surface packets
//! of `sub_82393070`, `sub_823A5850`) is computed from other device fields, not a register copy.
//! [`crate::verify`] checks the table against the command processor at every draw.

/// Dirty qwords (bit 63 = first register of a block's run).
pub const DIRTY: u32 = 0x0;
/// Inline render-state setter table (`SetRenderState` calls through it).
pub const RENDER_STATE_FUNCTIONS: u32 = 0x40;
/// Inline sampler-state setter table.
pub const SAMPLER_STATE_FUNCTIONS: u32 = 0x1D4;
/// Command buffer: address of the last dword written (packets are stored with `stwu`, so the
/// next dword goes at `+4`).
pub const COMMAND_WRITE: u32 = 48;
/// Command buffer end: helpers that write a run of registers stop before it.
pub const COMMAND_END: u32 = 52;
/// Command buffer soft limit: draws check `write <= limit` before emitting (or make space).
pub const COMMAND_LIMIT: u32 = 56;
/// Render target surface pointers (4), as SetRenderTarget stores them.
pub const RENDER_TARGETS: u32 = 0x3210;
/// Depth/stencil surface pointer, as SetDepthStencilSurface stores it.
pub const DEPTH_STENCIL: u32 = 0x3220;
/// Bytes of the device covered by the register shadow and the fields above.
pub const SIZE: u32 = 0x3800;

/// A run of registers stored consecutively in the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    /// Byte offset in the device of `first`'s value.
    pub offset: u32,
    /// First register (dword index of the Xenos register space).
    pub first: u32,
    pub count: u32,
    pub name: &'static str,
}

impl Block {
    pub const fn contains(&self, register: u32) -> bool {
        register >= self.first && register < self.first + self.count
    }

    /// Device byte offset of `register` (which must be in the block).
    pub const fn offset_of(&self, register: u32) -> u32 {
        self.offset + 4 * (register - self.first)
    }
}

/// The device's register shadow: every register a draw flushes, where it is kept.
pub const BLOCKS: &[Block] = &[
    Block { offset: 0x0780, first: 0x4000, count: 1024, name: "VS float constants" },
    Block { offset: 0x1780, first: 0x4400, count: 1024, name: "PS float constants" },
    Block { offset: 0x0480, first: 0x4800, count: 192, name: "fetch constants" },
    Block { offset: 10112, first: 0x4900, count: 40, name: "bool and loop constants" },
    Block { offset: 10272, first: 0x2388, count: 24, name: "PA_CL_UCP_0_X..PA_CL_UCP_5_W" },
    Block { offset: 10368, first: 0x2000, count: 16, name: "RB_SURFACE_INFO..PA_SC_SCREEN_SCISSOR_BR" },
    Block { offset: 10444, first: 0x2100, count: 21, name: "VGT_MAX_VTX_INDX..PA_CL_VPORT_ZOFFSET" },
    Block { offset: 10528, first: 0x2180, count: 5, name: "SQ_PROGRAM_CNTL..SQ_WRAPPING_1" },
    Block { offset: 10548, first: 0x2200, count: 12, name: "RB_DEPTHCONTROL..RB_BLENDCONTROL3" },
    Block { offset: 10596, first: 0x2280, count: 21, name: "PA_SU_POINT_SIZE..VGT_ENHANCE" },
    Block { offset: 10680, first: 0x2300, count: 38, name: "PA_SC_LINE_CNTL..RB_SAMPLE_COUNT_ADDR" },
    Block { offset: 10832, first: 0x2380, count: 8, name: "PA_SU_POLY_OFFSET_FRONT_SCALE..PA_CL_POINT_CULL_RAD" },
];

/// Where `register` is kept in the device, if it is.
pub fn shadow_offset(register: u32) -> Option<u32> {
    BLOCKS.iter().find(|b| b.contains(register)).map(|b| b.offset_of(register))
}

/// Every register of [`BLOCKS`] with its device offset, in table order.
pub fn shadowed_registers() -> impl Iterator<Item = (u32, u32)> {
    BLOCKS.iter().flat_map(|b| (0..b.count).map(move |i| (b.first + i, b.offset + 4 * i)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_do_not_overlap() {
        for (i, a) in BLOCKS.iter().enumerate() {
            for b in &BLOCKS[i + 1..] {
                let regs = a.first < b.first + b.count && b.first < a.first + a.count;
                let bytes = a.offset < b.offset + 4 * b.count && b.offset < a.offset + 4 * a.count;
                assert!(!regs && !bytes, "{a:?} overlaps {b:?}");
            }
            assert!(a.offset + 4 * a.count <= SIZE);
        }
        assert_eq!(shadow_offset(0x2200), Some(10548));
        assert_eq!(shadow_offset(0x4800 + 6), Some(0x0480 + 24));
        assert_eq!(shadow_offset(0x2080), None);
    }
}
