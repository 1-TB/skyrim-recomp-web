//! Tests run the native renderer straight from target/, so they need wgpu-native's directory on
//! their loader path (gpu-webgpu links it as a shared library).

fn main() {
    println!("cargo:rerun-if-env-changed=WGPU_NATIVE_DIR");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("emscripten") {
        return;
    }
    if let Some(dir) = std::env::var_os("WGPU_NATIVE_DIR") {
        let lib = std::path::Path::new(&dir).join("lib");
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{}", lib.display());
    }
}
