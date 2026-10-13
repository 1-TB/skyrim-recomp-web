#!/usr/bin/env bash
# Builds the web_triangle example and checks it in headless Chromium.
# Without DISPLAY it runs a headed Chromium on Xvfb (see e2e.mjs for why).
# Env: PW_DIR (dir whose node_modules has playwright), EMSDK (sourced emsdk_env.sh),
#      CARGO_TARGET_DIR (optional), CHROME_ARGS (optional override of the WebGPU flags).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
target="${CARGO_TARGET_DIR:-$root/target}"
: "${PW_DIR:?set PW_DIR to a directory containing node_modules/playwright}"
: "${EMSDK:?source emsdk_env.sh first}"
cargo build --manifest-path "$root/Cargo.toml" -p gpu-webgpu --example web_triangle \
  --target wasm32-unknown-emscripten -Zbuild-std=std,panic_unwind
out="$target/wasm32-unknown-emscripten/debug/examples"
if [[ -z "${DISPLAY:-}" && -z "${E2E_HEADLESS:-}" ]]; then
  exec xvfb-run -a -s "-screen 0 1024x768x24" node "$here/e2e.mjs" "$out" "${1:-$target/gpu-webgpu-web.png}"
fi
node "$here/e2e.mjs" "$out" "${1:-$target/gpu-webgpu-web.png}"
