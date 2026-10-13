use std::ffi::c_void;
use std::ptr;

use crate::ext::{self, stype};
use crate::handle::handle;
use crate::instance::slice_or_empty;
use crate::{
    Adapter, CompositeAlphaMode, Device, Error, Instance, PresentMode, Result, Texture, TextureFormat, TextureUsage,
    ffi,
};

handle!(
    /// A presentable target: a window natively, a `<canvas>` (or its OffscreenCanvas) on the web.
    Surface,
    WGPUSurface,
    wgpuSurfaceAddRef,
    wgpuSurfaceRelease
);

/// Where a surface presents. Variants that the platform lacks fail with [`Error::Unsupported`].
#[derive(Clone, Copy, Debug)]
pub enum SurfaceTarget<'a> {
    /// CSS selector of a canvas (`"#canvas"`). From a pthread this resolves to the
    /// OffscreenCanvas that was transferred to it (see [`crate::web::spawn_canvas_thread`]).
    Canvas(&'a str),
    Xlib {
        display: *mut c_void,
        window: u64,
    },
    Xcb {
        connection: *mut c_void,
        window: u32,
    },
    Wayland {
        display: *mut c_void,
        surface: *mut c_void,
    },
    Windows {
        hinstance: *mut c_void,
        hwnd: *mut c_void,
    },
    Metal {
        layer: *mut c_void,
    },
    Android {
        window: *mut c_void,
    },
}

#[derive(Clone, Debug, Default)]
pub struct SurfaceCapabilities {
    pub usages: TextureUsage,
    /// Preferred format first.
    pub formats: Vec<TextureFormat>,
    pub present_modes: Vec<PresentMode>,
    pub alpha_modes: Vec<CompositeAlphaMode>,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceConfiguration<'a> {
    pub format: TextureFormat,
    pub usage: TextureUsage,
    /// On the web this also resizes the canvas' drawing buffer.
    pub width: u32,
    pub height: u32,
    pub view_formats: &'a [TextureFormat],
    pub alpha_mode: CompositeAlphaMode,
    /// The web supports only `Fifo` (or `Undefined`).
    pub present_mode: PresentMode,
}

/// The texture to draw this frame; present it with [`SurfaceFrame::present`].
#[derive(Debug)]
pub struct SurfaceFrame {
    pub texture: Texture,
    pub suboptimal: bool,
    surface: Surface,
}

impl Instance {
    /// # Safety
    /// Native window handles must stay valid for the surface's lifetime.
    pub unsafe fn create_surface(&self, target: SurfaceTarget<'_>) -> Result<Surface> {
        fn chain(s_type: ffi::WGPUSType) -> ffi::WGPUChainedStruct {
            ffi::WGPUChainedStruct { next: ptr::null_mut(), sType: s_type }
        }
        // Each source is a #[repr(C)] struct starting with its chain header.
        enum Source {
            #[cfg(target_os = "emscripten")]
            Canvas(ext::WGPUEmscriptenSurfaceSourceCanvasHTMLSelector),
            Xlib(ext::WGPUSurfaceSourceXlibWindow),
            Xcb(ext::WGPUSurfaceSourceXCBWindow),
            Wayland(ext::WGPUSurfaceSourceWaylandSurface),
            Windows(ext::WGPUSurfaceSourceWindowsHWND),
            Metal(ext::WGPUSurfaceSourceMetalLayer),
            Android(ext::WGPUSurfaceSourceAndroidNativeWindow),
        }
        let native_only = cfg!(target_os = "emscripten");
        // emdawnwebgpu reads the selector as a C string, ignoring the view's length.
        #[cfg(target_os = "emscripten")]
        let selector = match target {
            SurfaceTarget::Canvas(s) => {
                std::ffi::CString::new(s).map_err(|_| Error::Invalid("selector has NUL".into()))?
            }
            _ => std::ffi::CString::default(),
        };
        let mut source = match target {
            #[cfg(target_os = "emscripten")]
            SurfaceTarget::Canvas(_) => Source::Canvas(ext::WGPUEmscriptenSurfaceSourceCanvasHTMLSelector {
                chain: chain(stype::EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR),
                selector: ffi::WGPUStringView { data: selector.as_ptr(), length: selector.as_bytes().len() },
            }),
            #[cfg(not(target_os = "emscripten"))]
            SurfaceTarget::Canvas(_) => return Err(Error::Unsupported("canvas surfaces exist only on the web")),
            _ if native_only => return Err(Error::Unsupported("window-system surfaces exist only natively")),
            SurfaceTarget::Xlib { display, window } => Source::Xlib(ext::WGPUSurfaceSourceXlibWindow {
                chain: chain(stype::SURFACE_SOURCE_XLIB_WINDOW),
                display,
                window,
            }),
            SurfaceTarget::Xcb { connection, window } => Source::Xcb(ext::WGPUSurfaceSourceXCBWindow {
                chain: chain(stype::SURFACE_SOURCE_XCB_WINDOW),
                connection,
                window,
            }),
            SurfaceTarget::Wayland { display, surface } => Source::Wayland(ext::WGPUSurfaceSourceWaylandSurface {
                chain: chain(stype::SURFACE_SOURCE_WAYLAND_SURFACE),
                display,
                surface,
            }),
            SurfaceTarget::Windows { hinstance, hwnd } => Source::Windows(ext::WGPUSurfaceSourceWindowsHWND {
                chain: chain(stype::SURFACE_SOURCE_WINDOWS_HWND),
                hinstance,
                hwnd,
            }),
            SurfaceTarget::Metal { layer } => Source::Metal(ext::WGPUSurfaceSourceMetalLayer {
                chain: chain(stype::SURFACE_SOURCE_METAL_LAYER),
                layer,
            }),
            SurfaceTarget::Android { window } => Source::Android(ext::WGPUSurfaceSourceAndroidNativeWindow {
                chain: chain(stype::SURFACE_SOURCE_ANDROID_NATIVE_WINDOW),
                window,
            }),
        };
        let next: *mut ffi::WGPUChainedStruct = match &mut source {
            #[cfg(target_os = "emscripten")]
            Source::Canvas(s) => ptr::from_mut(s).cast(),
            Source::Xlib(s) => ptr::from_mut(s).cast(),
            Source::Xcb(s) => ptr::from_mut(s).cast(),
            Source::Wayland(s) => ptr::from_mut(s).cast(),
            Source::Windows(s) => ptr::from_mut(s).cast(),
            Source::Metal(s) => ptr::from_mut(s).cast(),
            Source::Android(s) => ptr::from_mut(s).cast(),
        };
        let desc = ffi::WGPUSurfaceDescriptor { nextInChain: next, ..Default::default() };
        let raw = unsafe { ffi::wgpuInstanceCreateSurface(self.raw(), &desc) };
        Surface::from_created(raw, "surface")
            .map_err(|_| Error::Surface(format!("cannot create a surface for {target:?}")))
    }
}

impl Surface {
    pub fn capabilities(&self, adapter: &Adapter) -> Result<SurfaceCapabilities> {
        let mut caps = ffi::WGPUSurfaceCapabilities::default();
        let status = unsafe { ffi::wgpuSurfaceGetCapabilities(self.raw(), adapter.raw(), &mut caps) };
        if status != ffi::WGPUStatus::Success {
            return Err(Error::Surface("querying capabilities failed".into()));
        }
        let out = SurfaceCapabilities {
            usages: caps.usages,
            formats: slice_or_empty(caps.formats, caps.formatCount).to_vec(),
            present_modes: slice_or_empty(caps.presentModes, caps.presentModeCount).to_vec(),
            alpha_modes: slice_or_empty(caps.alphaModes, caps.alphaModeCount).to_vec(),
        };
        unsafe { ffi::wgpuSurfaceCapabilitiesFreeMembers(caps) };
        Ok(out)
    }

    pub fn configure(&self, device: &Device, config: &SurfaceConfiguration<'_>) {
        let raw = ffi::WGPUSurfaceConfiguration {
            device: device.raw(),
            format: config.format,
            usage: config.usage,
            width: config.width,
            height: config.height,
            viewFormatCount: config.view_formats.len(),
            viewFormats: config.view_formats.as_ptr(),
            alphaMode: config.alpha_mode,
            presentMode: config.present_mode,
            ..Default::default()
        };
        unsafe { ffi::wgpuSurfaceConfigure(self.raw(), &raw) };
    }

    pub fn unconfigure(&self) {
        unsafe { ffi::wgpuSurfaceUnconfigure(self.raw()) };
    }

    /// The next texture to render to. `Outdated`/`Lost` statuses mean "reconfigure";
    /// they come back as [`Error::SurfaceStatus`].
    pub fn current_frame(&self) -> Result<SurfaceFrame> {
        let mut st = ffi::WGPUSurfaceTexture::default();
        unsafe { ffi::wgpuSurfaceGetCurrentTexture(self.raw(), &mut st) };
        let ok = st.status == ffi::WGPUSurfaceGetCurrentTextureStatus::SuccessOptimal
            || st.status == ffi::WGPUSurfaceGetCurrentTextureStatus::SuccessSuboptimal;
        if !ok || st.texture.is_null() {
            if !st.texture.is_null() {
                unsafe { ffi::wgpuTextureRelease(st.texture) };
            }
            return Err(Error::SurfaceStatus(st.status));
        }
        Ok(SurfaceFrame {
            texture: unsafe { Texture::from_raw(st.texture) },
            suboptimal: st.status == ffi::WGPUSurfaceGetCurrentTextureStatus::SuccessSuboptimal,
            surface: self.clone(),
        })
    }
}

impl SurfaceFrame {
    /// Natively queues the frame for display. On the web the browser shows whatever was drawn
    /// to the canvas texture when the thread returns to its event loop, so this only drops it.
    pub fn present(self) {
        #[cfg(not(target_os = "emscripten"))]
        unsafe {
            ffi::wgpuSurfacePresent(self.surface.raw());
        }
        drop(self);
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }
}
