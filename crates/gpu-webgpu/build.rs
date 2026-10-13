//! Links the webgpu.h implementation.
//!
//! Native: wgpu-native from `WGPU_NATIVE_DIR` (an unpacked release), linked as a shared library
//! because its static archive carries a second Rust std (duplicate `rust_eh_personality`).
//! Binaries outside this crate need the library on their loader path (rpath or LD_LIBRARY_PATH).
//! Web: emdawnwebgpu is an Emscripten port and gets linked by emcc (`--use-port=emdawnwebgpu`);
//! the final binary's link flags are the host's responsibility, except for this crate's examples.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=WGPU_NATIVE_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("emscripten") {
        web_example_link_args();
        return;
    }
    let Some(dir) = env::var_os("WGPU_NATIVE_DIR").map(PathBuf::from) else {
        panic!(
            "WGPU_NATIVE_DIR is not set: point it at an unpacked wgpu-native release \
             (https://github.com/gfx-rs/wgpu-native/releases/tag/v29.0.1.1, wgpu-<os>-<arch>-release.zip)"
        );
    };
    let lib = dir.join("lib");
    let shared = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "wgpu_native.dll.lib",
        Ok("macos") => "libwgpu_native.dylib",
        _ => "libwgpu_native.so",
    };
    if !lib.join(shared).is_file() {
        panic!("{} not found (WGPU_NATIVE_DIR={})", lib.join(shared).display(), dir.display());
    }
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=wgpu_native");
    // Examples and tests of this crate run straight from target/.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!("cargo:rustc-link-arg-examples=-Wl,-rpath,{}", lib.display());
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{}", lib.display());
    }
}

/// emcc flags for `examples/` (mirrors hosts/host-web/build.rs plus what WebGPU needs).
fn web_example_link_args() {
    for arg in [
        "-pthread",
        "-sPROXY_TO_PTHREAD=1",
        "-sALLOW_MEMORY_GROWTH=1",
        "-Wno-pthreads-mem-growth",
        "-sMAXIMUM_MEMORY=4GB",
        "-sINITIAL_MEMORY=128MB",
        "-sGLOBAL_BASE=1048576",
        "-sSTACK_SIZE=1MB",
        "-sDEFAULT_PTHREAD_STACK_SIZE=1MB",
        "-sWASMFS=1",
        "-sFORCE_FILESYSTEM=1",
        "-fwasm-exceptions",
        "-sSUPPORT_LONGJMP=wasm",
        "-sEXIT_RUNTIME=1",
        "-sENVIRONMENT=web,worker,node",
        "-sEXPORTED_RUNTIME_METHODS=callMain",
        "-sEXPORTED_FUNCTIONS=_main",
        // WebGPU: webgpu.h on top of the browser API.
        "--use-port=emdawnwebgpu:cpp_bindings=false",
        // Lets pthread_create hand the page's <canvas> to the renderer thread
        // as an OffscreenCanvas; the transfer machinery lives in the GL library.
        "-sOFFSCREENCANVAS_SUPPORT=1",
        "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$GL",
    ] {
        println!("cargo:rustc-link-arg-examples={arg}");
    }
    let js = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("web/event_loop.js");
    println!("cargo:rerun-if-changed={}", js.display());
    println!("cargo:rustc-link-arg-examples=--js-library={}", js.display());
}
