//! Headless native proof: `cargo run -p gpu-webgpu --example offscreen [out.png]`.
//! Renders the test scene offscreen, checks probe pixels and the compute output, writes a PNG.

#[cfg(not(target_os = "emscripten"))]
mod common;

#[cfg(not(target_os = "emscripten"))]
fn main() -> std::process::ExitCode {
    use common::{offscreen, scene};
    let path = std::env::args().nth(1).unwrap_or_else(|| "offscreen.png".into());
    let out = match offscreen::run(64, 64) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("offscreen run failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    println!(
        "adapter: {} ({:?}, {:?}) {}",
        out.adapter.device, out.adapter.backend, out.adapter.adapter_type, out.adapter.description
    );
    if let Err(e) = out.write_png(std::path::Path::new(&path)) {
        eprintln!("writing {path}: {e}");
        return std::process::ExitCode::FAILURE;
    }
    let mut failures = scene::check(out.width, out.height, |x, y| out.pixel(x, y));
    if out.seq != scene::expected_seq() {
        failures.push(format!("compute output {:?}", out.seq));
    }
    failures.extend(out.uncaptured_errors.iter().map(|e| e.to_string()));
    for (fx, fy, _) in scene::PROBES {
        let (x, y) = ((fx * out.width as f32) as u32, (fy * out.height as f32) as u32);
        println!("pixel ({x:2},{y:2}) = {:?}", out.pixel(x, y));
    }
    println!("wrote {path}");
    if failures.is_empty() {
        println!("OK");
        std::process::ExitCode::SUCCESS
    } else {
        failures.iter().for_each(|f| eprintln!("FAIL {f}"));
        std::process::ExitCode::FAILURE
    }
}

#[cfg(target_os = "emscripten")]
fn main() {
    eprintln!("the offscreen example is native-only; see the web_triangle example");
}
