//! Xenos texture fetch constants, formats, tiling and conversion to host layouts.
//!
//! Entry points:
//! - [`TextureFetchConstant::decode`] / [`VertexFetchConstant::decode`] for fetch constants.
//! - [`TextureGuestInfo::from_fetch`] for the guest memory layout and exact byte ranges.
//! - [`host_mapping`] to choose the WebGPU format and CPU conversion for a guest format.
//! - [`load_texture`] to build upload-ready per-level data from guest memory.
//! - [`frontbuffer_to_rgba8`] for presenting the VdSwap front buffer.
//! - [`sampler_descriptor`] for the WebGPU sampler.
//!
//! Shader-side responsibilities that a host format cannot express are reported, not hidden:
//! per-component signedness ([`TextureFetchConstant::swizzled_signs`]; load the unsigned
//! and/or signed view as [`TextureFetchConstant::needed_views`] says), `UnsignedBiased`
//! (`2x - 1`), `Gamma` (Xenos piecewise-linear degamma), `NumFormat::Integer` (scale by
//! `2^bits - 1`), `exp_adjust` (multiply by `2^exp_adjust`), LOD bias, the fetch swizzle
//! composed over [`HostMapping::component_map`], and border colors.

pub mod address;
pub mod bc;
pub mod convert;
pub mod endian;
pub mod fetch;
pub mod format;
pub mod host;
pub mod layout;
pub mod load;
pub mod sampler;

pub use endian::{Endian, Endian128};
pub use fetch::{
    AnisoFilter, ArbitraryFilter, BorderColor, ClampMode, DataDimension, FetchConstantType, FetchSlot, NumFormat,
    SwizzleSource, TextureFetchConstant, TextureFilter, TextureSign, VertexFetchConstant,
};
pub use format::{FormatInfo, FormatKind, TextureFormat};
pub use host::{
    Conversion, HostCaps, HostChannel, HostFormat, HostMapping, MappingError, Support, host_mapping, support,
};
pub use layout::{GuestLayout, GuestRange, RangeKind, TextureGuestInfo, TextureSubresources};
pub use load::{
    GuestMemory, HostDimension, LoadError, LoadOptions, LoadedLevel, LoadedTexture, Rgba8Image, frontbuffer_to_rgba8,
    load_texture,
};
pub use sampler::{
    AddressMode, FetchInstructionFilters, FilterMode, SamplerDescriptor, SamplerLoss, sampler_descriptor,
};
