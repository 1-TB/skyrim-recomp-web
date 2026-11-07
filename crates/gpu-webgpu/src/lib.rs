//! WebGPU device layer over the standard `webgpu.h` C API, with one code path for both hosts:
//! natively it links wgpu-native, on `wasm32-unknown-emscripten` it links Dawn's emdawnwebgpu
//! port (which forwards to the browser's WebGPU).
//!
//! Threading: every object belongs to the thread that created it (on the web the underlying JS
//! objects are per-worker), so handles are neither `Send` nor `Sync`. Run the renderer on one
//! thread and feed it from others.
//!
//! Asynchrony: operations that complete later (adapter/device requests, buffer mapping, error
//! scopes, queue work) return a [`Pending`]. Its callback fires while the owning thread calls
//! [`Instance::process_events`] (natively also from [`Device::poll`]). On the web the thread
//! must also return to the browser event loop between polls (see [`web`]); natively
//! [`Instance::block_on`] and [`Device::block_on`] may wait in place.

macro_rules! flag_ops {
    ($t:ident) => {
        impl core::ops::BitOr for $t {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
        impl core::ops::BitOrAssign for $t {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
        impl core::ops::BitAnd for $t {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }
        impl $t {
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }
        }
    };
}

#[rustfmt::skip]
pub mod ffi;
pub mod counters;
mod desc;
mod error;
pub mod memory;
mod pending;
pub mod staging;
#[cfg(target_os = "emscripten")]
pub mod web;

// The webgpu.h backend (wgpu-native natively, emdawnwebgpu on the web).
#[cfg(not(metal_backend))]
mod buffer;
#[cfg(not(metal_backend))]
mod command;
#[cfg(not(metal_backend))]
mod device;
#[cfg(not(metal_backend))]
mod ext;
#[cfg(not(metal_backend))]
mod handle;
#[cfg(not(metal_backend))]
mod instance;
#[cfg(not(metal_backend))]
mod resource;
#[cfg(not(metal_backend))]
mod surface;
#[cfg(not(metal_backend))]
pub mod timing;
#[cfg(not(metal_backend))]
mod upscale;
#[cfg(not(metal_backend))]
pub use upscale::SpatialScaler;

#[cfg(not(metal_backend))]
pub use buffer::Buffer;
#[cfg(not(metal_backend))]
pub use command::{CommandBuffer, CommandEncoder, ComputePass, Queue, RenderPass, UNSHARED_UPLOADS};
#[cfg(not(metal_backend))]
pub use device::Device;
#[cfg(not(metal_backend))]
pub use instance::{Adapter, Instance};
#[cfg(not(metal_backend))]
pub use resource::{
    BindGroup, BindGroupLayout, ComputePipeline, PipelineLayout, RenderPipeline, Sampler, ShaderModule, Texture,
    TextureView,
};
#[cfg(not(metal_backend))]
pub use surface::{Surface, SurfaceFrame};

// The native Metal backend (Apple targets).
#[cfg(metal_backend)]
mod metal;
#[cfg(metal_backend)]
pub use metal::timing;
#[cfg(metal_backend)]
pub use metal::{
    Adapter, BindGroup, BindGroupLayout, Buffer, CommandBuffer, CommandEncoder, ComputePass, ComputePipeline, Device,
    Instance, PipelineLayout, Queue, RenderPass, RenderPipeline, Sampler, ShaderModule, SpatialScaler, Surface, SurfaceFrame,
    Texture, TextureView, UNSHARED_UPLOADS,
};

pub use desc::*;
pub use error::{Error, GpuError, Result};
pub use pending::Pending;

pub use ffi::{
    WGPUAdapterType as AdapterType, WGPUAddressMode as AddressMode, WGPUBackendType as BackendType,
    WGPUBlendFactor as BlendFactor, WGPUBlendOperation as BlendOperation, WGPUBufferBindingType as BufferBindingType,
    WGPUBufferUsage as BufferUsage, WGPUColor as Color, WGPUColorWriteMask as ColorWriteMask,
    WGPUCompareFunction as CompareFunction, WGPUCompositeAlphaMode as CompositeAlphaMode, WGPUCullMode as CullMode,
    WGPUErrorFilter as ErrorFilter, WGPUErrorType as ErrorType, WGPUExtent3D as Extent3d,
    WGPUFeatureName as FeatureName, WGPUFilterMode as FilterMode, WGPUFrontFace as FrontFace,
    WGPUIndexFormat as IndexFormat, WGPULimits as Limits, WGPULoadOp as LoadOp, WGPUMapMode as MapMode,
    WGPUMipmapFilterMode as MipmapFilterMode, WGPUOrigin3D as Origin3d, WGPUPowerPreference as PowerPreference,
    WGPUPresentMode as PresentMode, WGPUPrimitiveTopology as PrimitiveTopology,
    WGPUSamplerBindingType as SamplerBindingType, WGPUShaderStage as ShaderStage,
    WGPUStencilOperation as StencilOperation, WGPUStorageTextureAccess as StorageTextureAccess, WGPUStoreOp as StoreOp,
    WGPUTextureAspect as TextureAspect, WGPUTextureDimension as TextureDimension, WGPUTextureFormat as TextureFormat,
    WGPUTextureSampleType as TextureSampleType, WGPUTextureUsage as TextureUsage,
    WGPUTextureViewDimension as TextureViewDimension, WGPUVertexFormat as VertexFormat,
    WGPUVertexStepMode as VertexStepMode,
};

/// Where the Metal backend writes the MSL it generates for each pipeline stage (`None`: nowhere).
/// The webgpu.h backend compiles shaders inside its implementation and ignores this.
pub static MSL_DUMP_DIR: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

/// The GPU implementation in use, e.g. `wgpu-native 29.0.1.1`.
pub fn implementation() -> String {
    #[cfg(metal_backend)]
    {
        "native Metal (naga WGSL -> MSL)".to_string()
    }
    #[cfg(all(not(metal_backend), not(target_os = "emscripten")))]
    {
        let v = unsafe { ext::native::wgpuGetVersion() };
        format!("wgpu-native {}.{}.{}.{}", v >> 24, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff)
    }
    #[cfg(target_os = "emscripten")]
    {
        "emdawnwebgpu (browser WebGPU)".to_string()
    }
}

/// `Extent3d { width, height, depthOrArrayLayers }` shorthand.
pub const fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> Extent3d {
    Extent3d { width, height, depthOrArrayLayers: depth_or_array_layers }
}

/// `Origin3d { x, y, z }` shorthand.
pub const fn origin(x: u32, y: u32, z: u32) -> Origin3d {
    Origin3d { x, y, z }
}

/// Size in bytes of one texel (or one block for compressed formats) and the block edge in texels,
/// for the formats a copy may touch. `None` for depth/stencil formats without a copyable layout.
pub fn format_block_info(format: TextureFormat) -> Option<(u32, u32)> {
    use TextureFormat as F;
    Some(match format {
        F::R8Unorm | F::R8Snorm | F::R8Uint | F::R8Sint | F::Stencil8 => (1, 1),
        F::R16Uint | F::R16Sint | F::R16Float | F::RG8Unorm | F::RG8Snorm | F::RG8Uint | F::RG8Sint => (2, 1),
        F::R16Unorm | F::R16Snorm | F::Depth16Unorm => (2, 1),
        F::R32Float | F::R32Uint | F::R32Sint | F::RG16Uint | F::RG16Sint | F::RG16Float | F::RG16Unorm => (4, 1),
        F::RG16Snorm | F::RGBA8Unorm | F::RGBA8UnormSrgb | F::RGBA8Snorm | F::RGBA8Uint | F::RGBA8Sint => (4, 1),
        F::BGRA8Unorm | F::BGRA8UnormSrgb | F::RGB10A2Uint | F::RGB10A2Unorm | F::RG11B10Ufloat => (4, 1),
        F::RGB9E5Ufloat | F::Depth32Float => (4, 1),
        F::RG32Float | F::RG32Uint | F::RG32Sint | F::RGBA16Uint | F::RGBA16Sint | F::RGBA16Float => (8, 1),
        F::RGBA16Unorm | F::RGBA16Snorm => (8, 1),
        F::RGBA32Float | F::RGBA32Uint | F::RGBA32Sint => (16, 1),
        F::BC1RGBAUnorm | F::BC1RGBAUnormSrgb | F::BC4RUnorm | F::BC4RSnorm => (8, 4),
        F::BC2RGBAUnorm | F::BC2RGBAUnormSrgb | F::BC3RGBAUnorm | F::BC3RGBAUnormSrgb => (16, 4),
        F::BC5RGUnorm | F::BC5RGSnorm | F::BC6HRGBUfloat | F::BC6HRGBFloat | F::BC7RGBAUnorm => (16, 4),
        F::BC7RGBAUnormSrgb => (16, 4),
        _ => return None,
    })
}
