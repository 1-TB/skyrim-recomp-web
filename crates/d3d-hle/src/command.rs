//! The D3D calls a D3D-level renderer executes, decoded from the hooks, in call order.

use crate::args::{ClearArgs, ResolveArgs, SurfaceHeader, TextureHeader};

/// A render target or depth/stencil surface as bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surface {
    /// The surface object (identifies it).
    pub address: u32,
    pub header: SurfaceHeader,
    pub depth: bool,
}

impl Surface {
    /// RB_SURFACE_INFO: pitch in pixels.
    pub fn width(&self) -> u32 {
        (self.header.surface_info() & 0x3FFF).max(1)
    }

    /// Samples per pixel (RB_SURFACE_INFO MSAA field: 1, 2 or 4).
    pub fn samples(&self) -> u32 {
        1 << ((self.header.surface_info() >> 16) & 3).min(2)
    }

    /// EDRAM format: RB_COLOR_INFO / RB_DEPTH_INFO bits 16-19 (ColorRenderTargetFormat or
    /// DepthRenderTargetFormat).
    pub fn format(&self) -> u32 {
        (self.header.info() >> 16) & 0xF
    }

    /// Bytes per sample in EDRAM.
    pub fn bytes_per_sample(&self) -> u32 {
        // 16_16_16_16, 16_16_16_16_FLOAT and 32_32_FLOAT are 64-bit.
        if !self.depth && matches!(self.format(), 5 | 7 | 15) { 8 } else { 4 }
    }

    /// Rows the surface covers: its EDRAM size (object dword 11) over a row's bytes.
    pub fn height(&self) -> u32 {
        let row = self.width() * self.bytes_per_sample() * self.samples();
        (self.header.words[11] / row.max(1)).clamp(1, 8192)
    }
}

/// A D3D call for the renderer.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    SetRenderTarget {
        index: u32,
        surface: Option<Surface>,
    },
    SetDepthStencil {
        surface: Option<Surface>,
    },
    Clear(ClearArgs),
    Resolve {
        args: ResolveArgs,
        dest: Option<TextureHeader>,
    },
    /// Show the texture the frame was resolved to.
    Present {
        front: Option<TextureHeader>,
    },
}
