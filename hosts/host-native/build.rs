//! The binary links wgpu-native as a shared library (through gpu-webgpu); record its directory
//! as an rpath so the game runs without LD_LIBRARY_PATH.

fn main() {
    println!("cargo:rerun-if-env-changed=WGPU_NATIVE_DIR");
    if let Some(dir) = std::env::var_os("WGPU_NATIVE_DIR") {
        let lib = std::path::Path::new(&dir).join("lib");
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", lib.display());
    }
}
