//! Headless native run of the test scene: render to an RGBA8 texture and read everything back.

use gpu_webgpu::*;

use super::scene::{self, Scene};

pub struct Output {
    pub adapter: AdapterInfo,
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8 rows.
    pub rgba: Vec<u8>,
    pub seq: Vec<u32>,
    pub uncaptured_errors: Vec<GpuError>,
}

pub fn run(width: u32, height: u32) -> Result<Output> {
    let instance = Instance::new()?;
    let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions::default()))?;
    let info = adapter.info();
    let device =
        instance.block_on(adapter.request_device(&DeviceDescriptor { label: "offscreen", ..Default::default() }))?;

    device.push_error_scope(ErrorFilter::Validation);
    let scene = Scene::new(&device, TextureFormat::RGBA8Unorm);
    let target = device.create_texture(&TextureDescriptor {
        label: "target",
        size: extent(width, height, 1),
        format: TextureFormat::RGBA8Unorm,
        usage: TextureUsage::RenderAttachment | TextureUsage::CopySrc,
        ..Default::default()
    });
    let view = target.create_view(None);

    let row = (width * 4).next_multiple_of(256);
    let readback = device.create_buffer(&BufferDescriptor {
        label: "readback",
        size: u64::from(row * height) + scene::STORAGE_SIZE,
        usage: BufferUsage::MapRead | BufferUsage::CopyDst,
        mapped_at_creation: false,
    });
    let pixels_size = u64::from(row * height);

    let mut encoder = device.create_command_encoder("frame");
    scene.encode(&mut encoder, &view, width, height);
    encoder.copy_texture_to_buffer(
        &TexelCopyTextureInfo::whole(&target),
        &TexelCopyBufferInfo { buffer: &readback, offset: 0, bytes_per_row: row, rows_per_image: None },
        extent(width, height, 1),
    );
    encoder.copy_buffer_to_buffer(&scene.storage, 0, &readback, pixels_size, scene::STORAGE_SIZE);
    device.queue().submit([encoder.finish("frame")]);
    if let Some(e) = device.block_on(device.pop_error_scope())? {
        return Err(e.into());
    }
    device.block_on(device.queue().on_submitted_work_done())?;

    device.block_on(readback.map_async(MapMode::Read, 0, None))?;
    let mut padded = vec![0u8; pixels_size as usize];
    readback.read_mapped(0, &mut padded)?;
    let mut storage = vec![0u8; scene::STORAGE_SIZE as usize];
    readback.read_mapped(pixels_size, &mut storage)?;
    readback.unmap();

    let rgba = padded.chunks(row as usize).flat_map(|r| &r[..width as usize * 4]).copied().collect();
    let seq = storage[16..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok(Output { adapter: info, width, height, rgba, seq, uncaptured_errors: device.take_errors() })
}

impl Output {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    pub fn write_png(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(file, self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(std::io::Error::other)?;
        w.write_image_data(&self.rgba).map_err(std::io::Error::other)
    }
}
