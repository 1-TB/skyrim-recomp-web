//! Web proof: a renderer pthread owns the page's `#canvas` (as an OffscreenCanvas), creates
//! the WebGPU device through emdawnwebgpu and draws the test scene every frame. On frame 5 it
//! also copies the canvas texture back and checks the probe pixels, printing
//! `WEBGPU_E2E_RESULT ok` (or the failures). Built and checked by tools/web/run-e2e.sh.

#[cfg(target_os = "emscripten")]
#[path = "common/scene.rs"]
mod scene;

#[cfg(target_os = "emscripten")]
mod renderer {
    use gpu_webgpu::*;

    use super::scene::{self, Scene};

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;
    const CHECK_FRAME: u64 = 5;

    enum Phase {
        Adapter(Pending<Result<Adapter>>),
        Device(Adapter, Pending<Result<Device>>),
        Run(Box<Running>),
        Failed,
    }

    struct Readback {
        buffer: Buffer,
        map: Pending<Result<()>>,
        row: u32,
        pixels_size: u64,
    }

    struct Running {
        device: Device,
        scene: Scene,
        format: TextureFormat,
        frame: u64,
        readback: Option<Readback>,
    }

    fn report(result: std::result::Result<(), Vec<String>>) {
        match result {
            Ok(()) => println!("WEBGPU_E2E_RESULT ok"),
            Err(failures) => println!("WEBGPU_E2E_RESULT fail: {}", failures.join("; ")),
        }
    }

    pub fn start() {
        println!("renderer thread: {}", implementation());
        let instance = match Instance::new() {
            Ok(i) => i,
            Err(e) => return report(Err(vec![e.to_string()])),
        };
        let surface = match unsafe { instance.create_surface(SurfaceTarget::Canvas("#canvas")) } {
            Ok(s) => s,
            Err(e) => return report(Err(vec![e.to_string()])),
        };
        let request = instance
            .request_adapter(&RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() });
        let mut phase = Phase::Adapter(request);
        web::run_frame_loop(move || {
            instance.process_events();
            phase = match std::mem::replace(&mut phase, Phase::Failed) {
                Phase::Adapter(p) => match p.try_take() {
                    None => Phase::Adapter(p),
                    Some(Ok(adapter)) => {
                        let info = adapter.info();
                        println!(
                            "adapter: {:?} {} {} ({:?})",
                            info.backend, info.vendor, info.architecture, info.description
                        );
                        let request = adapter.request_device(&DeviceDescriptor { label: "web", ..Default::default() });
                        Phase::Device(adapter, request)
                    }
                    Some(Err(e)) => {
                        report(Err(vec![e.to_string()]));
                        Phase::Failed
                    }
                },
                Phase::Device(adapter, p) => match p.try_take() {
                    None => Phase::Device(adapter, p),
                    Some(Ok(device)) => match configure(&surface, &adapter, device) {
                        Ok(running) => Phase::Run(Box::new(running)),
                        Err(e) => {
                            report(Err(vec![e.to_string()]));
                            Phase::Failed
                        }
                    },
                    Some(Err(e)) => {
                        report(Err(vec![e.to_string()]));
                        Phase::Failed
                    }
                },
                Phase::Run(mut running) => match running.frame(&surface) {
                    Ok(()) => Phase::Run(running),
                    Err(e) => {
                        report(Err(vec![e.to_string()]));
                        Phase::Failed
                    }
                },
                Phase::Failed => return false,
            };
            true
        });
    }

    fn configure(surface: &Surface, adapter: &Adapter, device: Device) -> Result<Running> {
        let caps = surface.capabilities(adapter)?;
        let format = *caps.formats.first().ok_or(Error::Surface("no surface formats".into()))?;
        println!("surface formats {:?}, using {:?}", caps.formats, format);
        surface.configure(
            &device,
            &SurfaceConfiguration {
                format,
                usage: TextureUsage::RenderAttachment | TextureUsage::CopySrc,
                width: WIDTH,
                height: HEIGHT,
                view_formats: &[],
                alpha_mode: CompositeAlphaMode::Opaque,
                present_mode: PresentMode::Fifo,
            },
        );
        let scene = Scene::new(&device, format);
        Ok(Running { device, scene, format, frame: 0, readback: None })
    }

    impl Running {
        fn frame(&mut self, surface: &Surface) -> Result<()> {
            self.frame += 1;
            let frame = surface.current_frame()?;
            let view = frame.texture.create_view(None);
            let mut encoder = self.device.create_command_encoder("frame");
            self.scene.encode(&mut encoder, &view, WIDTH, HEIGHT);
            if self.frame == CHECK_FRAME {
                let row = (WIDTH * 4).next_multiple_of(256);
                let pixels_size = u64::from(row * HEIGHT);
                let buffer = self.device.create_buffer(&BufferDescriptor {
                    label: "readback",
                    size: pixels_size + scene::STORAGE_SIZE,
                    usage: BufferUsage::MapRead | BufferUsage::CopyDst,
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    &TexelCopyTextureInfo::whole(&frame.texture),
                    &TexelCopyBufferInfo { buffer: &buffer, offset: 0, bytes_per_row: row, rows_per_image: None },
                    extent(WIDTH, HEIGHT, 1),
                );
                encoder.copy_buffer_to_buffer(&self.scene.storage, 0, &buffer, pixels_size, scene::STORAGE_SIZE);
                self.device.queue().submit([encoder.finish("frame")]);
                let map = buffer.map_async(MapMode::Read, 0, None);
                self.readback = Some(Readback { buffer, map, row, pixels_size });
            } else {
                self.device.queue().submit([encoder.finish("frame")]);
            }
            frame.present();

            if let Some(rb) = &self.readback
                && let Some(mapped) = rb.map.try_take()
            {
                mapped?;
                let result = self.verify(rb);
                rb.buffer.unmap();
                self.readback = None;
                println!("frame {}: readback checked", self.frame);
                report(result);
            }
            Ok(())
        }

        fn verify(&self, rb: &Readback) -> std::result::Result<(), Vec<String>> {
            let mut pixels = vec![0u8; rb.pixels_size as usize];
            let mut storage = vec![0u8; scene::STORAGE_SIZE as usize];
            let read = rb.buffer.read_mapped(0, &mut pixels).and(rb.buffer.read_mapped(rb.pixels_size, &mut storage));
            if let Err(e) = read {
                return Err(vec![e.to_string()]);
            }
            let bgra = matches!(self.format, TextureFormat::BGRA8Unorm | TextureFormat::BGRA8UnormSrgb);
            let mut failures = scene::check(WIDTH, HEIGHT, |x, y| {
                let i = (y * rb.row + x * 4) as usize;
                let p = [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
                if bgra { [p[2], p[1], p[0], p[3]] } else { p }
            });
            let seq: Vec<u32> = storage[16..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            if seq != scene::expected_seq() {
                failures.push(format!("compute output {seq:?}"));
            }
            failures.extend(self.device.take_errors().iter().map(|e| e.to_string()));
            if failures.is_empty() { Ok(()) } else { Err(failures) }
        }
    }
}

#[cfg(target_os = "emscripten")]
fn main() {
    println!("main thread: spawning the renderer");
    if let Err(e) = gpu_webgpu::web::spawn_canvas_thread("#canvas", 1 << 20, renderer::start) {
        println!("WEBGPU_E2E_RESULT fail: {e}");
        return;
    }
    // Returning would exit the runtime (EXIT_RUNTIME=1); the title's main blocks the same way.
    loop {
        std::thread::park();
    }
}

#[cfg(not(target_os = "emscripten"))]
fn main() {
    eprintln!("build for wasm32-unknown-emscripten; see crates/gpu-webgpu/tools/web/run-e2e.sh");
}
