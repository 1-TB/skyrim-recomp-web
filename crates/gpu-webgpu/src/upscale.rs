//! MetalFX spatial upscaling exists only on the Metal backend: here no scaler can be made.

use crate::{CommandEncoder, Device, TextureFormat, TextureView};

/// Never made on this backend (see the Metal backend's `SpatialScaler`).
#[derive(Debug)]
pub enum SpatialScaler {}

impl Device {
    /// Always `None`: upscaling with MetalFX needs the Metal backend.
    pub fn create_spatial_scaler(&self, _input: (u32, u32), _output: (u32, u32), _format: TextureFormat) -> Option<SpatialScaler> {
        None
    }
}

impl SpatialScaler {
    pub fn input(&self) -> &TextureView {
        match *self {}
    }

    pub fn output(&self) -> &TextureView {
        match *self {}
    }

    pub fn output_size(&self) -> (u32, u32) {
        match *self {}
    }
}

impl CommandEncoder {
    pub fn spatial_scale(&mut self, scaler: &SpatialScaler) {
        match *scaler {}
    }
}
