//! Needs a WebGPU adapter: a GPU, or lavapipe via VK_ICD_FILENAMES.
#![cfg(not(target_os = "emscripten"))]

#[path = "../examples/common/mod.rs"]
mod common;

use common::{offscreen, scene};
use gpu_webgpu::*;

fn device() -> (Instance, Device) {
    let instance = Instance::new().expect("instance");
    let adapter = instance
        .block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .expect("no WebGPU adapter (set VK_ICD_FILENAMES to lavapipe on GPU-less machines)");
    let device = instance.block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");
    (instance, device)
}

#[test]
fn renders_textured_triangle_and_compute() {
    let out = offscreen::run(64, 64).expect("offscreen run");
    println!("{} on {} ({:?})", implementation(), out.adapter.device, out.adapter.backend);
    let png = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu-webgpu-offscreen.png");
    out.write_png(&png).expect("png");
    let failures = scene::check(out.width, out.height, |x, y| out.pixel(x, y));
    assert!(failures.is_empty(), "{failures:#?} (image: {})", png.display());
    assert_eq!(out.seq, scene::expected_seq());
    assert!(out.uncaptured_errors.is_empty(), "{:?}", out.uncaptured_errors);
}

#[test]
fn error_scope_captures_validation_error() {
    let (_instance, device) = device();
    device.push_error_scope(ErrorFilter::Validation);
    // MapRead may only be combined with CopyDst.
    let _bad = device.create_buffer(&BufferDescriptor {
        label: "invalid",
        size: 16,
        usage: BufferUsage::MapRead | BufferUsage::Storage,
        mapped_at_creation: false,
    });
    let err = device.block_on(device.pop_error_scope()).expect("pop").expect("validation error");
    assert_eq!(err.ty, ErrorType::Validation);
    assert!(device.block_on(device.pop_error_scope()).is_err(), "empty stack must be an error");
    assert_eq!(device.error_count(), 0);
}

#[test]
fn uncaptured_errors_are_collected() {
    let (_instance, device) = device();
    let _bad = device.create_buffer(&BufferDescriptor {
        label: "invalid",
        size: 16,
        usage: BufferUsage::MapWrite | BufferUsage::Uniform,
        mapped_at_creation: false,
    });
    let errors = device.take_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].ty, ErrorType::Validation);
}

#[test]
fn map_write_upload_roundtrip() {
    let (_instance, device) = device();
    let data: Vec<u8> = (0..=255).collect();
    let upload = device.create_buffer(&BufferDescriptor {
        label: "upload",
        size: 256,
        usage: BufferUsage::MapWrite | BufferUsage::CopySrc,
        mapped_at_creation: false,
    });
    device.block_on(upload.map_async(MapMode::Write, 0, None)).expect("map write");
    upload.write_mapped(0, &data).expect("write");
    upload.unmap();
    let init = device.create_buffer_init("init", BufferUsage::CopySrc, &data[..10]);
    let readback = device.create_buffer(&BufferDescriptor {
        label: "readback",
        size: 512,
        usage: BufferUsage::MapRead | BufferUsage::CopyDst,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder("copy");
    enc.copy_buffer_to_buffer(&upload, 0, &readback, 0, 256);
    enc.clear_buffer(&readback, 256, None);
    enc.copy_buffer_to_buffer(&init, 0, &readback, 256, 12);
    device.queue().write_buffer(&readback, 300, &[]);
    device.queue().submit([enc.finish("copy")]);
    device.block_on(readback.map_async(MapMode::Read, 0, None)).expect("map read");
    let mut back = vec![0u8; 512];
    readback.read_mapped(0, &mut back).expect("read");
    assert_eq!(&back[..256], &data[..]);
    assert_eq!(&back[256..266], &data[..10]);
    assert!(back[266..].iter().all(|&b| b == 0));
    assert!(readback.read_mapped(0, &mut [0u8; 4]).is_ok());
    readback.unmap();
    assert!(readback.read_mapped(0, &mut [0u8; 4]).is_err(), "unmapped buffers must not be readable");
    assert!(device.take_errors().is_empty());
}

#[test]
fn surfaces_report_platform_support() {
    let instance = Instance::new().expect("instance");
    let err = unsafe { instance.create_surface(SurfaceTarget::Canvas("#canvas")) }.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}
