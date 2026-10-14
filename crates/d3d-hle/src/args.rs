//! Arguments of the hooked D3D calls and the guest objects they refer to.
//!
//! The title's ABI passes the first eight integer arguments in r3-r10 and floating-point ones in
//! f1..; a floating-point argument also takes an integer slot, and arguments past the eighth slot
//! are on the caller's stack at `r1 + 84 + 8 * (slot - 8)` (as the callers store them).

/// Reads the arguments of a call: integer registers, f1, and guest dwords.
pub trait CallRegs {
    /// `rN` (low 32 bits).
    fn gpr(&self, n: usize) -> u32;
    /// `fN`.
    fn fpr(&self, n: usize) -> f64;
    /// Guest dword at `address` (big-endian guest memory, as a value).
    fn read(&self, address: u32) -> u32;

    /// Integer argument `slot` (0 = r3): a register, or the caller's stack past the eighth.
    fn arg(&self, slot: usize) -> u32 {
        if slot < 8 { self.gpr(3 + slot) } else { self.read(self.gpr(1) + 84 + 8 * (slot as u32 - 8)) }
    }
}

/// `D3DCLEAR_*` flags.
pub mod clear {
    /// Render targets 0-3.
    pub const TARGETS: u32 = 0xF;
    pub const ZBUFFER: u32 = 0x10;
    pub const STENCIL: u32 = 0x20;
}

/// `Clear(device, count, rects, flags, color, z, stencil)`: `sub_823E6800`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClearArgs {
    /// Rectangles `[x1, y1, x2, y2)` to clear; empty = the whole viewport.
    pub rects: Vec<[i32; 4]>,
    pub flags: u32,
    /// D3DCOLOR (A8R8G8B8).
    pub color: u32,
    pub z: f32,
    pub stencil: u32,
}

impl ClearArgs {
    pub fn read(c: &impl CallRegs) -> Self {
        let (count, rects) = (c.arg(1), c.arg(2));
        let rects = if rects == 0 {
            Vec::new()
        } else {
            (0..count.min(64)).map(|i| std::array::from_fn(|k| c.read(rects + 16 * i + 4 * k as u32) as i32)).collect()
        };
        // Z is a float argument: it takes slot 5 (r8), so the stencil is in slot 6 (r9).
        ClearArgs { rects, flags: c.arg(3), color: c.arg(4), z: c.fpr(1) as f32, stencil: c.arg(6) }
    }

    /// The color as RGBA floats in [0, 1].
    pub fn rgba(&self) -> [f32; 4] {
        let c = self.color;
        [(c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF, c >> 24].map(|v| v as f32 / 255.0)
    }
}

/// `D3DRESOLVE_*` flags.
pub mod resolve {
    /// Source: render target 0-3, or 4 for the depth/stencil surface.
    pub const SOURCE: u32 = 0x7;
    pub const DEPTHSTENCIL: u32 = 0x4;
    pub const CLEAR_RENDERTARGET: u32 = 0x100;
    pub const CLEAR_DEPTHSTENCIL: u32 = 0x200;
}

/// `Resolve(device, flags, source_rect, dest_texture, dest_point, level, slice_or_face,
/// clear_color, clear_z, clear_stencil, parameters)`: `sub_823A7690`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolveArgs {
    pub flags: u32,
    /// `[x1, y1, x2, y2)` in the source surface; `None` = all of it.
    pub source_rect: Option<[i32; 4]>,
    /// The destination texture object (0: clear only).
    pub dest_texture: u32,
    pub dest_point: Option<[i32; 2]>,
    pub level: u32,
    pub slice: u32,
    pub clear_color: Option<[f32; 4]>,
    pub clear_z: f32,
    pub clear_stencil: u32,
    pub parameters: u32,
}

impl ResolveArgs {
    pub fn read(c: &impl CallRegs) -> Self {
        let rect = c.arg(2);
        let point = c.arg(4);
        let color = c.arg(7);
        ResolveArgs {
            flags: c.arg(1),
            source_rect: (rect != 0).then(|| std::array::from_fn(|k| c.read(rect + 4 * k as u32) as i32)),
            dest_texture: c.arg(3),
            dest_point: (point != 0).then(|| [c.read(point) as i32, c.read(point + 4) as i32]),
            level: c.arg(5),
            slice: c.arg(6),
            clear_color: (color != 0).then(|| std::array::from_fn(|k| f32::from_bits(c.read(color + 4 * k as u32)))),
            // Slot 8 holds the float clear Z (passed in f1).
            clear_z: c.fpr(1) as f32,
            clear_stencil: c.arg(9),
            parameters: c.arg(10),
        }
    }

    /// Render target index, or `None` for the depth/stencil surface.
    pub fn source_target(&self) -> Option<u32> {
        let s = self.flags & resolve::SOURCE;
        (s < resolve::DEPTHSTENCIL).then_some(s)
    }
}

/// A D3D surface (render target or depth/stencil buffer in EDRAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceHeader {
    pub address: u32,
    pub words: [u32; SURFACE_WORDS],
}

/// Dwords of a surface object read (the header and the fields after it).
pub const SURFACE_WORDS: usize = 12;

impl SurfaceHeader {
    pub fn read(address: u32, read: impl Fn(u32) -> u32) -> Self {
        SurfaceHeader { address, words: std::array::from_fn(|i| read(address + 4 * i as u32)) }
    }

    /// RB_SURFACE_INFO for the surface (pitch in pixels, MSAA): SetRenderTarget and
    /// SetDepthStencilSurface load it into the device.
    pub fn surface_info(&self) -> u32 {
        self.words[6]
    }

    /// RB_COLOR_INFO (color surfaces) or RB_DEPTH_INFO (depth surfaces): EDRAM base and format.
    pub fn info(&self) -> u32 {
        self.words[7]
    }
}

/// A D3D texture: the GPU fetch constant it binds as (texture fetch constant layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureHeader {
    pub address: u32,
    pub fetch: [u32; 6],
}

/// Byte offset of the fetch constant in a texture object.
pub const TEXTURE_FETCH: u32 = 28;

impl TextureHeader {
    pub fn read(address: u32, read: impl Fn(u32) -> u32) -> Self {
        TextureHeader { address, fetch: std::array::from_fn(|i| read(address + TEXTURE_FETCH + 4 * i as u32)) }
    }

    /// Physical address of the base level (fetch dword 1, 4 KiB pages).
    pub fn base_address(&self) -> u32 {
        self.fetch[1] & 0xFFFF_F000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        regs: [u32; 32],
        f1: f64,
        mem: std::collections::HashMap<u32, u32>,
    }

    impl CallRegs for Fake {
        fn gpr(&self, n: usize) -> u32 {
            self.regs[n]
        }
        fn fpr(&self, _: usize) -> f64 {
            self.f1
        }
        fn read(&self, a: u32) -> u32 {
            self.mem.get(&a).copied().unwrap_or(0)
        }
    }

    #[test]
    fn clear_and_resolve_arguments() {
        let mut f = Fake { regs: [0; 32], f1: 0.5, mem: Default::default() };
        f.regs[3] = 0x4000_0000;
        f.regs[4] = 1;
        f.regs[5] = 0x7000_0000;
        f.regs[6] = 0x3F;
        f.regs[7] = 0x80FF_4020;
        f.regs[9] = 7;
        for (i, v) in [1, 2, 640, 360].iter().enumerate() {
            f.mem.insert(0x7000_0000 + 4 * i as u32, *v);
        }
        let c = ClearArgs::read(&f);
        assert_eq!(c, ClearArgs { rects: vec![[1, 2, 640, 360]], flags: 0x3F, color: 0x80FF_4020, z: 0.5, stencil: 7 });
        assert_eq!(c.rgba()[0], 1.0);

        f.regs[4] = 0x204;
        f.regs[5] = 0;
        f.regs[6] = 0x51F1_3500;
        f.regs[7] = 0;
        f.regs[1] = 0x7010_0000;
        f.mem.insert(0x7010_0000 + 92, 0x80);
        f.mem.insert(0x7010_0000 + 100, 0x1234);
        let r = ResolveArgs::read(&f);
        assert_eq!((r.flags, r.dest_texture, r.clear_stencil, r.parameters), (0x204, 0x51F1_3500, 0x80, 0x1234));
        assert_eq!(r.source_target(), None);
        assert_eq!((r.source_rect, r.dest_point, r.clear_color), (None, None, None));
    }
}
