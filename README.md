# Skyrim (Xbox 360) in the browser

This repository is a static recompilation runtime for the Xbox 360 version of *The Elder Scrolls V: Skyrim*. It
contains **no game code, assets or Xbox SDK material**. You supply your own disc dump. The build then translates
the game's PowerPC code to C++ and compiles it to WebAssembly. A Rust runtime emulates the kernel, the GPU (on
WebGPU) and audio, and the page runs it in the browser.

The steps below go from your disc image to the game running in a browser tab.

## What you need

- **Your own Skyrim Xbox 360 disc image** (`.iso`, disc 1). Tested with title ID `425307E6`, disc version
  (no title update).
- **A Linux machine for building.** An x86-64 machine with about 16 GB of RAM and 20 GB of free disk.
  - Tools: `git`, `cmake`, `ninja`, `clang`/`clang++`, `python3`.
  - [rustup](https://rustup.rs). The pinned nightly toolchain installs itself from `rust-toolchain.toml`.
  - [emsdk](https://emscripten.org/docs/getting_started/downloads.html) with Emscripten **6.0.12**:
    ```sh
    git clone https://github.com/emscripten-core/emsdk ~/emsdk
    ~/emsdk/emsdk install 6.0.12 && ~/emsdk/emsdk activate 6.0.12
    ```
  - [extract-xiso](https://github.com/XboxDev/extract-xiso), to unpack the disc image.
- **A browser with WebGPU** to play in: desktop Chrome or Edge, or Android Chrome. A gamepad works; so do keyboard
  and mouse, or the on-screen touch controls.

## Quick start: `setup.sh`

`setup.sh` runs steps 1–6 below for you. It installs missing tools (system packages through
pacman/apt/dnf/zypper, the Rust nightly, emsdk 6.0.12, extract-xiso), then unpacks, decrypts, configures,
recompiles and builds:

```sh
./setup.sh path/to/skyrim1.iso --serve
```

Every step checks whether its work is already done and skips it if so, so you can re-run the script after an
interruption, a failure or a code change. It also checks the helper addresses against your executable before it
writes `config/skyrim.toml`, and stops if they don't match the tested version. Useful options:

| Option | Effect |
|---|---|
| `--status` | Show which steps are done. |
| `--only build` / `--skip deps` | Run or skip particular steps (comma-separated). |
| `-y` | Install missing dependencies without asking. |
| `-j N` | Limit parallel compile jobs (lower it if the build runs out of memory). |
| `--debug` | Build a debug web version instead of a release one. |
| `--reconfigure`, `--force-recompile` | Regenerate the config, or rerun the recompiler. |

Logs go to `generated/logs/`. `source generated/env.sh` sets up a shell for running the `cargo xtask` commands
yourself. The sections below describe what each step does.

## 1. Unpack the disc

From the repository root:

```sh
extract-xiso -x skyrim1.iso -d game
```

`game/` should now hold `default.xex` and `Data/` (about 3.9 GB). `game/` is gitignored, so nothing from it is
ever committed.

## 2. Set up XenonRecomp

This repo patches [XenonRecomp](https://github.com/hedge-dev/XenonRecomp) to support Skyrim. The command below
clones it to `~/tools/XenonRecomp` at a pinned commit, applies `tools/xenonrecomp/skyrt.patch` and builds it:

```sh
cargo xtask setup-xenonrecomp
```

## 3. Decrypt the executable

The disc's `default.xex` is encrypted and compressed. Keep the original and write a decrypted copy, which is the
file the runtime loads:

```sh
mkdir -p generated/xex
mv game/default.xex generated/xex/default.original.xex
cargo xtask decrypt --in generated/xex/default.original.xex --out game/default.xex
```

The decryption key comes from your XenonRecomp checkout; this repo doesn't contain it.

## 4. Recompile the game code

Create the recompiler configuration from the template:

```sh
cp config/skyrim.toml.template config/skyrim.toml
```

Edit `config/skyrim.toml`:

1. Uncomment `switch_table_file_path = "skyrim_switch_tables.toml"`.
2. Set the helper addresses. These are the values for the disc version. Run `cargo xtask scan --xex
   game/default.xex` to check them against your executable:

```toml
restgprlr_14_address = 0x82F91CB0
savegprlr_14_address = 0x82F91C60
restfpr_14_address   = 0x82F923CC
savefpr_14_address   = 0x82F92380
restvmx_14_address   = 0x82F928C8
savevmx_14_address   = 0x82F92630
restvmx_64_address   = 0x82F9295C
savevmx_64_address   = 0x82F926C4
longjmp_address      = 0x82F92BC0
setjmp_address       = 0x82F931B0
```

Then find the jump tables and run the recompiler, which writes C++ into `generated/recomp/`:

```sh
cargo xtask switches --xex game/default.xex --out config/skyrim_switch_tables.toml
cargo xtask recompile --config config/skyrim.toml
```

`recompile` ends by printing three `export` lines. Run them in your shell, since every build after this needs them:

```sh
export XENON_RECOMP_OUT=$PWD/generated/recomp
export XENON_RECOMP_DIR=$HOME/tools/XenonRecomp
export SKYRIM_XEX=$PWD/game/default.xex
```

## 5. Build the web version

```sh
source ~/emsdk/emsdk_env.sh
cargo xtask build-web --release
```

The build takes a while; the recompiled C++ is large. The page is written to `web-dist/`.

## 6. Serve it

WebAssembly threads need a cross-origin isolated page, and the bundled server sends the required headers:

```sh
cargo xtask serve --port 8080 --game-dir game
```

Open <http://127.0.0.1:8080/>.

The server only listens on localhost. To reach it from another device (a phone, another PC), put it behind a
private tunnel that gives you HTTPS, such as Tailscale: `tailscale serve --bg --https=8443 http://127.0.0.1:8080`.
Browsers only enable WebGPU and threads on HTTPS pages or on localhost. Don't expose the server to the internet:
with `--game-dir` it serves your game files.

## 7. Play

On the page:

1. **Game files.** Choose one source:
   - **Imported into this browser**: *Import game folder…* (Chromium desktop) or *Import .zip…* (a zip of the
     `game/` folder, which works on phones). The files are copied into the browser's private storage once, about
     4 GB.
   - **Streamed from the server**: nothing is copied. Files are fetched on demand from the `--game-dir` you served.
     This is the easiest option on phones.
2. **Settings** (optional, remembered by the browser): frame-rate cap (30 is the original; 60 or unlimited lift
   it), look sensitivity, volume, touch controls, a performance overlay, log level and diagnostics.
3. Press **Boot**, then **Fullscreen**. Saves are kept in the browser. *Export saves* and *Import saves* copy them
   to or from the dev server (`target/web-saves/`).

Controls:

| Input | How it works |
|---|---|
| Gamepad | The first connected gamepad with the standard mapping. |
| Keyboard and mouse | WASD to move. Click the game to capture the mouse for looking (Esc releases it). Left/right click are the right/left triggers. Space is A, Tab or Esc is B, R is X, F is Y, Q/E are the bumpers, Enter is Start, Backspace is Back, 1–4 are the D-pad. |
| Touch | Drag on the left half to move and on the right half to look. The on-screen buttons cover the rest of the controller. |

## Native debug build (optional)

A headless Linux build renders frames to PNG files, for debugging without a browser. It needs
[wgpu-native](https://github.com/gfx-rs/wgpu-native) (`WGPU_NATIVE_DIR`) and a Vulkan driver:

```sh
cargo build -p host-native
target/debug/skyrim-native --game-dir game --dump-frames out --dump-every 60
```

Run `skyrim-native` with no arguments to see every option.

## Repository layout

| Path | Contents |
|---|---|
| `crates/runtime-core` | Guest memory and threads. |
| `crates/kernel-hle` | Xbox 360 kernel and XAM emulation. |
| `crates/vfs` | Game and save file systems, including streaming. |
| `crates/xenos-*` | GPU emulation: registers, shader translation to WGSL, textures, EDRAM, the renderer. |
| `crates/gpu-webgpu` | A small safe layer over `webgpu.h`. |
| `crates/xma-decoder` | Audio decoding. |
| `crates/guest-sys` | Builds the recompiled C++ and links it in. |
| `hosts/host-web` | The browser host (`web/` holds the page). |
| `hosts/host-native` | The Linux debug host. |
| `xtask` | Build tooling. |
| `tools/` | The XenonRecomp patch and the XEX decryption tool. |
