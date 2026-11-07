//! MetalFX spatial upscaling: an edge-aware upscaler, sharper than bicubic filtering, for
//! scaling a finished frame up to the screen. (The temporal scaler needs motion vectors, which a
//! recompiled console title doesn't produce.)

use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLResource, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage};
use objc2_metal_fx::{MTLFXSpatialScaler, MTLFXSpatialScalerBase, MTLFXSpatialScalerDescriptor};

use super::{CommandEncoder, Device, Id, conv};
use crate::{Texture, TextureFormat, TextureUsage, TextureView};

/// Scales `input` (render into it) up to `output` (sample it) with MetalFX.
pub struct SpatialScaler {
    raw: Id<dyn MTLFXSpatialScaler>,
    input: Texture,
    input_view: TextureView,
    output: Texture,
    output_view: TextureView,
}

impl std::fmt::Debug for SpatialScaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpatialScaler({}x{} -> {}x{})", self.input.width(), self.input.height(), self.output.width(), self.output.height())
    }
}

fn texture(
    device: &ProtocolObject<dyn MTLDevice>,
    format: MTLPixelFormat,
    (width, height): (u32, u32),
    usage: MTLTextureUsage,
    label: &str,
) -> Option<Id<dyn MTLTexture>> {
    // SAFETY: a plain 2D texture descriptor with non-zero sizes.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format,
            width as usize,
            height as usize,
            false,
        )
    };
    desc.setUsage(usage);
    desc.setStorageMode(MTLStorageMode::Private);
    let t = device.newTextureWithDescriptor(&desc)?;
    t.setLabel(Some(&super::ns(label)));
    Some(t)
}

impl Device {
    /// A MetalFX spatial scaler from `input` to `output` pixels in `format` (an 8-bit RGBA or
    /// BGRA format, or RGBA16Float); `None` if MetalFX can't on this device.
    pub fn create_spatial_scaler(&self, input: (u32, u32), output: (u32, u32), format: TextureFormat) -> Option<SpatialScaler> {
        let device = &self.shared.raw;
        if input.0 == 0 || input.1 == 0 || output.0 == 0 || output.1 == 0 {
            return None;
        }
        // SAFETY: a capability query.
        if !unsafe { MTLFXSpatialScalerDescriptor::supportsDevice(device) } {
            return None;
        }
        let pixel_format = conv::pixel_format(format)?;
        autoreleasepool(|_| {
            let desc = unsafe { MTLFXSpatialScalerDescriptor::new() };
            // SAFETY: setters of a fresh descriptor with sizes and a format MetalFX accepts.
            let raw = unsafe {
                desc.setColorTextureFormat(pixel_format);
                desc.setOutputTextureFormat(pixel_format);
                desc.setInputWidth(input.0 as usize);
                desc.setInputHeight(input.1 as usize);
                desc.setOutputWidth(output.0 as usize);
                desc.setOutputHeight(output.1 as usize);
                desc.newSpatialScalerWithDevice(device)
            }?;
            // SAFETY: usage queries of the scaler just made.
            let (in_usage, out_usage) = unsafe { (raw.colorTextureUsage(), raw.outputTextureUsage()) };
            let input_raw = texture(
                device,
                pixel_format,
                input,
                in_usage | MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
                "metalfx input",
            )?;
            let output_raw =
                texture(device, pixel_format, output, out_usage | MTLTextureUsage::ShaderRead, "metalfx output")?;
            let input = Texture::from_surface(input_raw, format, TextureUsage::RenderAttachment | TextureUsage::TextureBinding);
            let output = Texture::from_surface(output_raw, format, TextureUsage::TextureBinding);
            let input_view = input.create_view(None);
            let output_view = output.create_view(None);
            Some(SpatialScaler { raw, input, input_view, output, output_view })
        })
    }
}

impl SpatialScaler {
    /// The texture to render the frame into.
    pub fn input(&self) -> &TextureView {
        &self.input_view
    }

    /// The upscaled frame, after [`CommandEncoder::spatial_scale`].
    pub fn output(&self) -> &TextureView {
        &self.output_view
    }

    pub fn output_size(&self) -> (u32, u32) {
        (self.output.width(), self.output.height())
    }
}

impl CommandEncoder {
    /// Upscales the scaler's input into its output (after the passes recorded so far).
    pub fn spatial_scale(&mut self, scaler: &SpatialScaler) {
        let (Some(input), Some(output)) = (scaler.input.mtl(), scaler.output.mtl()) else { return };
        self.end_blit();
        // SAFETY: both textures were made with the usages the scaler asked for; the content
        // size is the whole input.
        unsafe {
            scaler.raw.setInputContentWidth(input.width());
            scaler.raw.setInputContentHeight(input.height());
            scaler.raw.setColorTexture(Some(&input));
            scaler.raw.setOutputTexture(Some(&output));
            scaler.raw.encodeToCommandBuffer(&self.raw);
        }
    }
}
