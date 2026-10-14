//! Emscripten link settings for the web host.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("emscripten") {
        return;
    }
    for arg in [
        "-pthread",
        // main() blocks (it waits for the title to exit), so run it on a worker
        // and keep the browser thread free to service OPFS and spawn workers.
        "-sPROXY_TO_PTHREAD=1",
        // Guest memory is laid out compactly in linear memory (runtime-core memory.rs,
        // linear_map), which grows on demand.
        "-sALLOW_MEMORY_GROWTH=1",
        // emcc warns that JS heap views are re-checked after growth with
        // pthreads; the runtime touches the heap from JS only for I/O proxying.
        "-Wno-pthreads-mem-growth",
        // runtime-core's linear_map::SIZE rounded up (main.rs checks it); iOS Safari often
        // refuses to reserve a 3 GiB shared memory.
        "-sMAXIMUM_MEMORY=1538MB",
        "-sINITIAL_MEMORY=128MB",
        // Keep low addresses unused so stray guest null-pointer accesses land in
        // unused memory rather than on host data.
        "-sGLOBAL_BASE=1048576",
        "-sSTACK_SIZE=1MB",
        "-sDEFAULT_PTHREAD_STACK_SIZE=1MB",
        // OPFS (and the Node backend for headless runs) via WasmFS.
        "-sWASMFS=1",
        "-sFORCE_FILESYSTEM=1",
        // Thread termination unwinds Rust through C++ frames; setjmp/longjmp
        // are redirected to the host. Both use native wasm exceptions.
        "-fwasm-exceptions",
        "-sSUPPORT_LONGJMP=wasm",
        "-sEXIT_RUNTIME=1",
        "-sENVIRONMENT=web,worker,node",
        "-sEXPORTED_RUNTIME_METHODS=callMain,HEAPU8",
        // Called by web/input.js with controller state, and by tools/browser-run.mjs.
        "-sEXPORTED_FUNCTIONS=_main,_skyrt_set_pad,_skyrt_set_pad_connected,_skyrt_audio_ring,_skyrt_swap_count,_skyrt_memory_bytes",
        // WebGPU: webgpu.h on top of the browser API (crates/gpu-webgpu).
        "--use-port=emdawnwebgpu:cpp_bindings=false",
        // Lets the renderer thread take the page's <canvas> as an OffscreenCanvas; the
        // transfer machinery lives in the GL library.
        "-sOFFSCREENCANVAS_SUPPORT=1",
        // $stringToUTF8OnStack: emscripten 6.0.12's OPFS backend lists directories with it but
        // doesn't declare the dependency, so without it every OPFS readdir fails with EIO (the
        // content enumerator then finds no saves).
        "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$GL,$stringToUTF8OnStack",
    ] {
        println!("cargo:rustc-link-arg={arg}");
    }
    // The renderer's event-loop scheduler (gpu_webgpu::web::run_event_loop).
    let js = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/gpu-webgpu/web/event_loop.js");
    println!("cargo:rerun-if-changed={}", js.display());
    println!("cargo:rustc-link-arg=--js-library={}", js.display());
    // The on-screen keyboard's prompt (src/text_input.rs).
    let text_input = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lib/text_input.js");
    println!("cargo:rerun-if-changed={}", text_input.display());
    println!("cargo:rustc-link-arg=--js-library={}", text_input.display());
    // Streamed game files (src/stream.rs).
    let http_files = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lib/http_files.js");
    println!("cargo:rerun-if-changed={}", http_files.display());
    println!("cargo:rustc-link-arg=--js-library={}", http_files.display());
    // Device kind for memory budgets (src/main.rs).
    let device = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lib/device.js");
    println!("cargo:rerun-if-changed={}", device.display());
    println!("cargo:rustc-link-arg=--js-library={}", device.display());
}
