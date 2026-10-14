//! Host render targets standing in for Xenos EDRAM, and resolves (EDRAM copies) into guest memory.
//!
//! WebGPU has no rasterizer-ordered views, so EDRAM is emulated like Xenia's host render target
//! path: every [`RenderTargetKey`] (base, pitch, MSAA, depth/format) gets a host texture covering
//! one EDRAM addressing period, and a tile ownership map records which texture holds the latest
//! contents of each of the 2048 tiles. When a draw (or a resolve clear) uses tiles last written
//! through another key, their contents are converted through the EDRAM bit layout into the new
//! texture (an ownership transfer, a fullscreen pass discarding everything outside the tiles).
//!
//! Draw flow ([`EdramCache::targets`]): pass the draw's `RenderTargetSetup` and the encoder the
//! draw will be recorded in; transfers are encoded first, then use the returned attachments
//! (load + store) in the draw's render pass. [`DrawTargets`] also says what the pipeline must do:
//! - formats: `colors[i].1` / `depth.1`; sample count 1 or 4 (guest 2x MSAA runs on 4x targets in
//!   host samples 0 and 3, use [`DrawTargets::sample_mask`]);
//! - [`ColorTargetInfo`]: multiply outputs by `exp_bias_factor`, clamp to `output_range`, disable
//!   blending when not `blendable`. k_8_8_8_8_GAMMA targets are stored *linear* in rgba16float, so
//!   the shader writes linear color and blending happens in linear space (no gamma code needed);
//! - depth: host depth is the guest z (0..1) for both D24S8 and D24FS8; stencil is the guest stencil.
//!
//! Resolve flow ([`EdramCache::begin_resolve`] / [`EdramCache::poll_resolve`]): tiles of the source
//! rectangle are dumped from their owners into an EDRAM-layout storage buffer, a compute shader
//! does sample selection/averaging, PWL-gamma decoding, exp bias, red/blue swap and destination
//! format packing (Xenia's resolve shaders) into linear words, which are read back; the CPU then
//! applies the endian swap and the tiled 2D/3D destination layout and writes guest memory. Clears
//! are scissored draws into the target owning the cleared tiles (after a transfer of the rest).

mod cache;
mod dest;
mod format;
mod ownership;
mod shaders;

pub use cache::{DrawTargets, EdramCache, ExternalSource, PendingResolve, ResolveResult, SharedMemoryTarget};
pub use format::{
    ColorTargetInfo, DepthTargetInfo, EdramOptions, WGSL_LINEAR_TO_PWL_GAMMA, host_color_format, host_depth_format,
    linear_to_pwl_gamma, pwl_gamma_to_linear,
};
pub use ownership::OwnedRange;
pub use xenos_state::edram::RenderTargetKey;

/// Guest physical memory as seen by the GPU (byte order as in memory).
pub trait GuestMemoryAccess {
    fn read(&self, addr: u32, out: &mut [u8]);
    fn write(&self, addr: u32, data: &[u8]);
}

/// Why EDRAM work could not be done. Nothing is submitted when one is returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EdramError {
    /// MSAA mode 3.
    InvalidMsaa(u32),
    /// EDRAM color format without a host equivalent (raw value).
    UnsupportedFormat(u32),
    /// Tile-aligned surface wider than the host texture limit.
    PitchTooLarge {
        width: u32,
        max: u32,
    },
    /// No resolve conversion to this destination format.
    UnsupportedCopy(xenos_state::xenos::TextureFormat),
    /// The converted copy would exceed the GPU buffer limits.
    ResolveTooLarge {
        bytes: u64,
        max: u64,
    },
    /// Mapping or reading the resolve readback failed.
    Readback(String),
    DeviceLost(String),
}

impl std::fmt::Display for EdramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMsaa(v) => write!(f, "invalid MSAA mode {v}"),
            Self::UnsupportedFormat(v) => write!(f, "unsupported EDRAM color format {v}"),
            Self::PitchTooLarge { width, max } => write!(f, "surface width {width} exceeds the host limit {max}"),
            Self::UnsupportedCopy(format) => write!(f, "unsupported resolve destination format {format:?}"),
            Self::ResolveTooLarge { bytes, max } => write!(f, "resolve output of {bytes} bytes exceeds {max}"),
            Self::Readback(m) => write!(f, "resolve readback failed: {m}"),
            Self::DeviceLost(m) => write!(f, "device lost: {m}"),
        }
    }
}

impl std::error::Error for EdramError {}
