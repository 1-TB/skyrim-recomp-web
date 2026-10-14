//! Shared test support: guest memory and a single-texture render target cache.
#![allow(dead_code)]

use std::sync::Mutex;

use gpu_webgpu::*;
use xenos_render::PhysicalMemory;
use xenos_render::targets::{DrawTargets, RenderTargetCache, ResolveStatus, SharedMemoryTarget};
use xenos_state::edram::RenderTargetSetup;
use xenos_state::resolve::ResolveInfo;

pub struct Memory(pub Mutex<Vec<u8>>);

impl PhysicalMemory for Memory {
    fn read(&self, addr: u32, out: &mut [u8]) {
        let m = self.0.lock().unwrap();
        for (i, b) in out.iter_mut().enumerate() {
            *b = m.get(addr as usize + i).copied().unwrap_or(0);
        }
    }

    fn write(&self, addr: u32, data: &[u8]) {
        let mut m = self.0.lock().unwrap();
        for (i, b) in data.iter().enumerate() {
            if let Some(d) = m.get_mut(addr as usize + i) {
                *d = *b;
            }
        }
    }
}

impl xenos_texture::GuestMemory for Memory {
    fn read(&self, addr: u32, out: &mut [u8]) {
        PhysicalMemory::read(self, addr, out);
    }
}

/// One RGBA8 color target of fixed size for every setup; resolves are not supported.
pub struct OneTarget {
    pub texture: Texture,
    pub view: TextureView,
    pub size: u32,
    pub setups: std::rc::Rc<std::cell::RefCell<Vec<RenderTargetSetup>>>,
}

impl OneTarget {
    pub fn new(device: &Device, size: u32) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: "test target",
            size: extent(size, size, 1),
            format: TextureFormat::RGBA8Unorm,
            usage: TextureUsage::RenderAttachment | TextureUsage::CopySrc,
            ..Default::default()
        });
        let view = texture.create_view(None);
        OneTarget { texture, view, size, setups: Default::default() }
    }

    /// Reads the texture back as RGBA8 rows.
    pub fn read(device: &Device, texture: &Texture, size: u32) -> Vec<u8> {
        let row = (size * 4).next_multiple_of(256);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: "readback",
            size: u64::from(row * size),
            usage: BufferUsage::MapRead | BufferUsage::CopyDst,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder("readback");
        encoder.copy_texture_to_buffer(
            &TexelCopyTextureInfo::whole(texture),
            &TexelCopyBufferInfo { buffer: &buffer, offset: 0, bytes_per_row: row, rows_per_image: None },
            extent(size, size, 1),
        );
        device.queue().submit([encoder.finish("readback")]);
        device.block_on(buffer.map_async(MapMode::Read, 0, None)).unwrap();
        let mut padded = vec![0u8; (row * size) as usize];
        buffer.read_mapped(0, &mut padded).unwrap();
        padded.chunks(row as usize).flat_map(|r| &r[..size as usize * 4]).copied().collect()
    }
}

impl RenderTargetCache for OneTarget {
    fn targets(
        &mut self,
        _device: &Device,
        _encoder: &mut CommandEncoder,
        setup: &RenderTargetSetup,
        _min_height: u32,
    ) -> Option<DrawTargets> {
        self.setups.borrow_mut().push(setup.clone());
        let mut colors: [Option<(TextureView, TextureFormat)>; 4] = Default::default();
        colors[0] = setup.color[0].map(|_| (self.view.clone(), TextureFormat::RGBA8Unorm));
        Some(DrawTargets {
            colors,
            depth: None,
            width: self.size,
            height: self.size,
            sample_count: 1,
            sample_mask: !0,
            blendable: [true; 4],
        })
    }

    fn resolve(
        &mut self,
        _device: &Device,
        _info: &ResolveInfo,
        _memory: &dyn PhysicalMemory,
        _shared: Option<&SharedMemoryTarget<'_>>,
        _encoder: Option<&mut CommandEncoder>,
    ) -> ResolveStatus {
        ResolveStatus::Done { written: None, in_gpu_memory: false }
    }

    fn poll_resolve(&mut self, _device: &Device, _memory: &dyn PhysicalMemory) -> ResolveStatus {
        ResolveStatus::Done { written: None, in_gpu_memory: false }
    }

    fn end_frame(&mut self) {}
}

pub fn device() -> Device {
    let instance = Instance::new().expect("instance");
    let adapter = instance.block_on(instance.request_adapter(&RequestAdapterOptions::default())).expect("adapter");
    let features = xenos_render::wanted_features(&adapter.features());
    instance
        .block_on(adapter.request_device(&DeviceDescriptor {
            label: "test",
            required_features: &features,
            required_limits: Some(adapter.limits()),
        }))
        .expect("device")
}
