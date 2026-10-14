//! Native end to end: a swap of a tiled front buffer in "guest memory" is rendered headlessly
//! and dumped; the PNG must match the CPU decode of the same front buffer.
#![cfg(not(target_os = "emscripten"))]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use xenos_render::native::{self, NativeOptions};
use xenos_render::{Command, PhysicalMemory};
use xenos_texture::address::tiled_2d;
use xenos_texture::{Endian, TextureFetchConstant, TextureFormat, frontbuffer_to_rgba8};

struct Memory(Vec<u8>);

impl PhysicalMemory for Memory {
    fn read(&self, addr: u32, out: &mut [u8]) {
        for (i, b) in out.iter_mut().enumerate() {
            *b = self.0.get(addr as usize + i).copied().unwrap_or(0);
        }
    }

    fn write(&self, _addr: u32, _data: &[u8]) {
        unreachable!("presenting never writes guest memory")
    }
}

impl xenos_texture::GuestMemory for Memory {
    fn read(&self, addr: u32, out: &mut [u8]) {
        PhysicalMemory::read(self, addr, out);
    }
}

const BASE: u32 = 0x10_0000;
const W: u32 = 128;
const H: u32 = 64;

fn quadrant_color(x: u32, y: u32) -> [u8; 4] {
    match (x < W / 2, y < H / 2) {
        (true, true) => [0xFF, 0x20, 0x10, 0xFF],
        (false, true) => [0x10, 0xE0, 0x30, 0xFF],
        (true, false) => [0x20, 0x30, 0xD0, 0xFF],
        (false, false) => [0xF0, 0xF0, 0x40, 0xFF],
    }
}

#[test]
fn swap_of_a_tiled_front_buffer_is_presented() {
    let mut mem = vec![0u8; (BASE + W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let offset = tiled_2d(x as i32, y as i32, W, 2) as u32;
            let at = (BASE + offset) as usize;
            mem[at..at + 4].copy_from_slice(&quadrant_color(x, y));
        }
    }
    let mut fetch = TextureFetchConstant::new_2d(TextureFormat::k_8_8_8_8, W, H);
    fetch.tiled = true;
    fetch.endianness = Endian::None;
    fetch.base_page = BASE >> 12;
    let memory = Arc::new(Memory(mem));
    let expected = frontbuffer_to_rgba8(&fetch, memory.as_ref()).expect("CPU decode");

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("present-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let handle = native::start(
        memory,
        NativeOptions { dump_dir: Some(dir.clone()), dump_every: 1, window: None },
        Box::new(|device| Box::new(common::OneTarget::new(device, 16))),
    )
    .expect("renderer");
    handle.submit(Command::Swap { fetch: fetch.encode(), width: W, height: H });
    handle.wait_idle();

    let png = dir.join("frame_000001.png");
    let decoder = png::Decoder::new(std::fs::File::open(&png).expect("frame dumped (is a WebGPU adapter available?)"));
    let mut reader = decoder.read_info().unwrap();
    let mut rgba = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut rgba).unwrap();
    assert_eq!((info.width, info.height), (W, H));
    for (x, y) in [(5, 5), (100, 10), (10, 50), (120, 60), (63, 31), (64, 32)] {
        let i = ((y * W + x) * 4) as usize;
        assert_eq!(&rgba[i..i + 4], &expected.data[i..i + 4], "pixel ({x},{y})");
    }
}
