//! Skyrim's Xbox 360 Direct3D device (disc version, title update 0.0.10.2), for rendering at the
//! level of its D3D calls instead of its command stream (see `target/d3d-hle-report.md`).
//!
//! The functions hooked in the recompiled title ([`HOOKS`]; guest-sys emits the overrides), the
//! layout of the device's register shadow ([`device`]), and a check of that layout against the
//! command processor's registers at every draw ([`verify`]).

pub mod device;
pub mod verify;

/// What a hooked function does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookKind {
    /// Emits draw packets after flushing the device's dirty state (`r3` is the device).
    DrawLeaf,
    /// A draw entry point that reaches a [`HookKind::DrawLeaf`].
    Draw,
    Clear,
    Resolve,
    BeginTiling,
    EndTiling,
    SetRenderTarget,
    SetDepthStencilSurface,
    Present,
}

/// A recompiled function replaced by a pass-through hook.
#[derive(Clone, Copy, Debug)]
pub struct Hook {
    pub function: &'static str,
    pub kind: HookKind,
    /// What it is, from its code and calls (the report's section 2.1).
    pub name: &'static str,
}

/// Hooked functions; the index is the id the hooks pass to the host.
pub const HOOKS: &[Hook] = &[
    Hook { function: "sub_823B7508", kind: HookKind::DrawLeaf, name: "DrawIndexedVertices" },
    Hook { function: "sub_823B3798", kind: HookKind::DrawLeaf, name: "DrawVertices" },
    Hook { function: "sub_823E2D60", kind: HookKind::DrawLeaf, name: "DrawPrimitiveUP core" },
    Hook { function: "sub_823D9968", kind: HookKind::DrawLeaf, name: "DrawIndexed variant" },
    Hook { function: "sub_823AA6A0", kind: HookKind::DrawLeaf, name: "internal draw (resolve, tiling)" },
    Hook { function: "sub_823A3AD0", kind: HookKind::Draw, name: "DrawIndexedPrimitiveUP" },
    Hook { function: "sub_823D9C20", kind: HookKind::Draw, name: "DrawIndexed variant (outer)" },
    Hook { function: "sub_82366A10", kind: HookKind::Draw, name: "DrawPrimitiveUP" },
    Hook { function: "sub_823A7690", kind: HookKind::Draw, name: "internal draw (outer)" },
    Hook { function: "sub_823E6800", kind: HookKind::Clear, name: "Clear" },
    Hook { function: "sub_823A8A30", kind: HookKind::Resolve, name: "Resolve" },
    Hook { function: "sub_82405598", kind: HookKind::BeginTiling, name: "BeginTiling" },
    Hook { function: "sub_823EF368", kind: HookKind::EndTiling, name: "EndTiling" },
    Hook { function: "sub_8242E650", kind: HookKind::SetRenderTarget, name: "SetRenderTarget" },
    Hook { function: "sub_82385580", kind: HookKind::SetDepthStencilSurface, name: "SetDepthStencilSurface" },
    Hook { function: "sub_823785D8", kind: HookKind::Present, name: "Present" },
];

/// Title id and version the hooks and the device layout belong to.
pub const TITLE: (u32, u32) = (0x4253_07E6, 0x0000_0A02);

/// First word of the type-3 NOP the verifier puts before a draw's packets ('SKD3'); the second is
/// the draw's sequence number.
pub const MARKER_SIGNATURE: u32 = 0x534B_4433;

/// PM4 type-3 NOP header with `count` payload dwords.
pub const fn nop_header(count: u32) -> u32 {
    3 << 30 | ((count - 1) & 0x3FFF) << 16 | 0x10 << 8
}
