//! Presentation through the host's `CAMetalLayer`.

use std::cell::RefCell;
use std::rc::Rc;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDrawable};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use super::{Adapter, Device, Instance, Shared, conv, ns};
use crate::{
    CompositeAlphaMode, Error, PresentMode, Result, SurfaceCapabilities, SurfaceConfiguration, SurfaceTarget, Texture,
    TextureFormat, TextureUsage, ffi,
};

struct Config {
    format: TextureFormat,
    usage: TextureUsage,
    shared: Rc<Shared>,
}

pub(crate) struct SurfaceInner {
    layer: Retained<CAMetalLayer>,
    config: RefCell<Option<Config>>,
}

/// A presentable target: the host's `CAMetalLayer`.
#[derive(Clone)]
pub struct Surface {
    inner: Rc<SurfaceInner>,
}

super::resource::rc_handle!(Surface);

/// The texture to draw this frame; present it with [`SurfaceFrame::present`].
pub struct SurfaceFrame {
    pub texture: Texture,
    pub suboptimal: bool,
    surface: Surface,
    drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
}

impl std::fmt::Debug for SurfaceFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceFrame").field("texture", &self.texture).finish()
    }
}

impl Instance {
    /// # Safety
    /// The layer must stay valid for the surface's lifetime.
    pub unsafe fn create_surface(&self, target: SurfaceTarget<'_>) -> Result<Surface> {
        match target {
            SurfaceTarget::Metal { layer } => {
                let layer = unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }
                    .ok_or_else(|| Error::Surface("null CAMetalLayer".into()))?;
                Ok(Surface { inner: Rc::new(SurfaceInner { layer, config: RefCell::new(None) }) })
            }
            _ => Err(Error::Unsupported("the Metal backend presents only to a CAMetalLayer")),
        }
    }
}

impl Surface {
    pub fn capabilities(&self, _adapter: &Adapter) -> Result<SurfaceCapabilities> {
        let mut present_modes = vec![PresentMode::Fifo];
        if cfg!(target_os = "macos") {
            present_modes.push(PresentMode::Immediate);
        }
        Ok(SurfaceCapabilities {
            usages: TextureUsage::RenderAttachment | TextureUsage::CopySrc | TextureUsage::CopyDst | TextureUsage::TextureBinding,
            formats: vec![
                TextureFormat::BGRA8Unorm,
                TextureFormat::BGRA8UnormSrgb,
                TextureFormat::RGBA16Float,
                TextureFormat::RGB10A2Unorm,
            ],
            present_modes,
            alpha_modes: vec![CompositeAlphaMode::Opaque, CompositeAlphaMode::Premultiplied],
        })
    }

    pub fn configure(&self, device: &Device, config: &SurfaceConfiguration<'_>) {
        let layer = &self.inner.layer;
        let Some(format) = conv::pixel_format(config.format) else {
            device.shared.report(crate::ErrorType::Validation, format!("surface format {:?}", config.format));
            return;
        };
        layer.setDevice(Some(&device.shared.raw));
        layer.setPixelFormat(format);
        layer.setFramebufferOnly(config.usage == TextureUsage::RenderAttachment);
        layer.setOpaque(config.alpha_mode != CompositeAlphaMode::Premultiplied);
        layer.setMaximumDrawableCount(3);
        layer.setDrawableSize(objc2_core_foundation::CGSize::new(f64::from(config.width), f64::from(config.height)));
        layer.setAllowsNextDrawableTimeout(false);
        #[cfg(target_os = "macos")]
        layer.setDisplaySyncEnabled(config.present_mode != PresentMode::Immediate);
        *self.inner.config.borrow_mut() =
            Some(Config { format: config.format, usage: config.usage, shared: device.shared.clone() });
    }

    pub fn unconfigure(&self) {
        self.inner.config.borrow_mut().take();
    }

    /// The next drawable to render to.
    pub fn current_frame(&self) -> Result<SurfaceFrame> {
        let (format, usage) = match &*self.inner.config.borrow() {
            Some(c) => (c.format, c.usage),
            None => return Err(Error::SurfaceStatus(ffi::WGPUSurfaceGetCurrentTextureStatus::Lost)),
        };
        let drawable = autoreleasepool(|_| self.inner.layer.nextDrawable())
            .ok_or(Error::SurfaceStatus(ffi::WGPUSurfaceGetCurrentTextureStatus::Timeout))?;
        let texture = Texture::from_surface(drawable.texture(), format, usage);
        Ok(SurfaceFrame { texture, suboptimal: false, surface: self.clone(), drawable })
    }
}

impl SurfaceFrame {
    /// Queues the frame for display after the work submitted so far.
    pub fn present(self) {
        let config = self.surface.inner.config.borrow();
        let Some(c) = config.as_ref() else { return };
        autoreleasepool(|_| {
            let cb = c.shared.new_command_buffer();
            cb.setLabel(Some(&ns("present")));
            let drawable: &ProtocolObject<dyn MTLDrawable> = ProtocolObject::from_ref(&*self.drawable);
            // On ProMotion iPhones the pacing a drawable asks for is what raises the display
            // to 120 Hz (with a plain present it mostly stayed at 60 even with a display link
            // preferring 120).
            #[cfg(target_os = "ios")]
            cb.presentDrawable_afterMinimumDuration(drawable, 1.0 / 120.0);
            #[cfg(not(target_os = "ios"))]
            cb.presentDrawable(drawable);
            cb.commit();
        });
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }
}
