# iOS port and Metal backend: tracker

As of 2026-11-07. The game runs on Mac and iPhone (18 Pro Max, A20 Pro, iOS 27.3 beta) through a
native Metal renderer. Missing clothing/grass (iPhone) and streaky roofs (Mac) are fixed: both were
z-fighting from host float32 depth where the Xbox stores float24. Open work is performance (target 120 fps).

## Status

| Area | Mac (M2) | iPhone 18 Pro Max |
| --- | --- | --- |
| Boot, menus, New Game, loading saves | Works | Works |
| Renderer | Native Metal (default); wgpu kept for A/B | Native Metal (default) |
| Riverwood save frame rate | ~42 fps at 1×, GPU-bound (see GPU budget) | ~34 fps at 2× (GPU ~30 ms) |
| Character clothing, grass (alpha-tested) | Correct | Correct (float24 depth) |
| Streaky thatch/roof textures | Fixed (float24 depth) | Correct |
| Frame cap | 120 fps (`--vblank-hz 240` default) | 120 fps + `CADisableMinimumFrameDurationOnPhone` |
| GPU errors in logs | None | None |

## Work log (newest first)

| When | Milestone | Key changes |
| --- | --- | --- |
| Nov 7 | Freeze fix, GPU profiling, compute passes | lwarx/stwcx. reservations carry a version (ABA in the guest's lock-free lists left a cycle in a job list: thread 0xE spun forever; 0/8 hangs after); constant patcher walks only its stage's records; store and texture-load shaders without variable-indexed local arrays; consecutive compute passes share one Metal encoder; naga loop bounding off; the native renderer blocks on GPU progress instead of spinning (renderer busy 63% → 32%); `--autoplay` in args.txt for scripted iOS runs |
| Nov 7 | Pixel shader speed, settings, MetalFX | Mac town 24 → 39.6 fps: branch-free `xe_tex_post` (format conversion and swizzle only when needed), PS parameter register written through a `switch` (a variable index put all registers in scratch memory); `--anisotropy N`, `--metalfx` (MetalFX spatial upscaling in the presenter); iOS settings panel at launch (`settings.txt`), FPS counter, full-resolution drawable (view scale factor was 1); GPU pass timing on Metal (`gpu-time`, per-target pass labels), `solid` debug mode; wgpu-native dropped on Apple |
| Nov 7 | Float24 depth (root cause of missing textures) | Pixel shaders round depth to Xbox D24FS8 float24 precision (`xe_float24_depth`, `DEPTH_FLOAT24` system flag, `[[depth(less)]]` conservative depth); iOS alpha workarounds removed; 240 Hz vblank default; debug modes `ps-probe`, `hash-pages`, `verify-loads`, `float32-depth`, `--gpu-sync-draws` |
| Nov 6 | Alpha-test investigation on Metal | Failed alpha tests clear the sample mask instead of `discard` (iOS default); debug modes `alpha-mask`, `alpha-discard`, `alpha-view-kill`, `alpha-depth-bias=N`; `cargo xtask build-ios --simulator` |
| Nov 6 | Native Metal backend | `crates/gpu-webgpu/src/metal/`: naga WGSL→MSL per pipeline, safe math, `preserveInvariance`, pipeline sample mask emulated in the shader, zero-initialised buffers/textures, upload ring, `--render-debug msl-dump`. Hosts always use it on Apple targets (wgpu-native support dropped Nov 7) |
| Nov 6 | Shared GPU types split | `gpu-webgpu` descriptors in `src/desc.rs`; backend chosen at compile time (`metal_backend` cfg = feature `metal` on Apple targets) |
| Before | `ios-port` commit f8fd449 | iOS app (UIKit scene host, touch, gamepads, Documents layout, logs, args.txt), d3d path default for TU 0x0A02, raised memory limits |
| Before | Crash fixes | XMA context-array register, ARM barriers for `sync`/`lwsync`/`eieio`, Darwin page discard zeroing, missing-function reports with guest state |
| Before | Rendering fixes | Black textures (handheld budgets removed), title update applied |

## Resolved: alpha-tested geometry missing on the A20 / streaky roofs on the Mac

Root cause: the Xbox stores depth as float24 (D24FS8), so nearly coplanar layers (clothing over a
body, thatch over roof boards, foliage) get equal depth and pass the game's GEQUAL tests. Host
float32 depth splits them per pixel, and the M2 and A20 rasterizers round differently, so each
device lost different layers. Fix: round depth to float24 in the pixel shader, as Xenia's
`depth_float24_convert_in_pixel_shader` does; `--render-debug float32-depth` turns it off.

Symptom was: the woman by the fence in Riverwood drew only head, hands and feet; grass and plants
were missing on the phone; thatch roofs streaked on the Mac.

| Experiment | Result on iPhone | What it rules out |
| --- | --- | --- |
| `no-alpha-test` (earlier, wgpu) | Clothing back | Confirms the alpha test is involved |
| `no-a2c`, `single-sample`, `no-d32s8` (wgpu) | Still missing | Alpha to coverage, MSAA, depth format |
| Native Metal backend, no fast math | Still missing | wgpu, wgpu's compile options, fast math |
| Kill via sample mask instead of `discard` | Still missing | The `discard` instruction itself |
| `cpu-texture` | Still missing (only ~25% of loads moved to the CPU) | Inconclusive |
| `alpha-view` (no kill, test result as colour) | Torso cyan = alpha ≈ 1, test passes — same as Mac | Wrong texture alpha / constants |
| `alpha-view-kill` (colour + kill) | Torso missing, trees cyan | Shows the kill changes the outcome |
| `alpha-depth-bias=256` | Still missing | Depth ties against depth-only passes |
| Mac in a window with iOS texture budget + `alpha-mask` | Correct | iOS host settings, window presentation |
| iOS Simulator (M2 GPU, iOS code) | Not run: the iOS 26.5 runtime stays on a white screen under Xcode 27.2 (Simulator.app is now DeviceHub.app) | — |
| `ps-probe` on the Mac (baseline) | Torso pixel: repeated draws of vs 6c / ps 6b, alpha 0.66–1.0, ref 0.502, func 4 (greater), mask all ones | Reference for the phone run |

## GPU budget (Mac, Riverwood, 1×, Metal System Trace)

Per 24 ms frame: pixel shaders 9.5 ms (scene pass ~4 ms, ALU-bound on integer emulation work; F32 is
~16% of ALU), EDRAM resolves 4.6 ms (~29/frame; full-screen 0.23 ms each), vertex 3.4 ms, blits
1.9 ms (texture-load copies, index gathers), texture loads 1.8 ms, constant patches 0.8 ms, idle
gaps between ~200 encoders ~4 ms. Float24 depth costs ~4 ms (no hidden-surface removal with shader
depth writes). Profile with `xcrun xctrace record --template 'Metal System Trace' --attach PID`
(add `--instrument 'Metal GPU Counters'` for limiters) and `--render-debug gpu-time` for labels.

## Repo organisation

Branches:

| Branch | At | Contents |
| --- | --- | --- |
| `main` | a0d62fa (= origin/main) | Upstream web port |
| `ios-port` | f8fd449 | iOS/macOS ports, ARM fixes, XMA fix |
| `metal-backend` | `ios-port` + Metal commit | Native Metal backend, float24 depth fix, debug tooling, simulator support (pushed to origin) |

Local, gitignored (never commit): `game/`, `generated/`, `config/*.toml`, `target/` (saves in
`target/phone-saves`, run outputs in `target/mac-*` and `target/phone-*`).

## How to run

```sh
source generated/env.sh
# Mac (Metal; release build of the guest takes ~15 min the first time)
cargo build --release -p host-native
./target/release/skyrim-native --game-dir game --content <copy of target/phone-saves/content> \
  --gpu-sync --vblank-hz 120 --run-for 150 --dump-frames <dir> --dump-every 200 \
  --tap START@15 --tap START@20 --tap A@30 --tap A@38

# iPhone
cargo xtask build-ios --release --launch
# Simulator (builds and installs; launching is unreliable with Xcode 27.2, see above)
cargo xtask build-ios --release --install --simulator
```

Phone files (`<id>`: the device UDID from `xcrun devicectl list devices`; `<bundle>`: `dev.skyrt.recomp.<team id, lowercase>`, as build-ios prints it):

```sh
xcrun devicectl device copy to   --device <id> --domain-type appDataContainer --domain-identifier <bundle> \
  --source args.txt --destination Documents/args.txt
xcrun devicectl device copy from --device <id> --domain-type appDataContainer --domain-identifier <bundle> \
  --source Documents/logs --destination <dir>      # also Documents/frames, Documents/target/shader-dump/msl
```

Useful `--render-debug` modes: `gpu-time` (GPU ms per pass label), `solid` (constant-colour pixel shaders), `ps-probe=FRAME:XxY,...` (per-draw alpha-test records at chosen pixels), `msl-dump`, `alpha-view`, `alpha-view-kill`, `alpha-mask`,
`alpha-discard`, `no-alpha-test`, `cpu-texture`, `trace-frame=N`, `texture-mib=N`.

## Backlog

| Priority | Item |
| --- | --- |
| 1 | Performance toward 120 fps: GPU-bound on translated pixel shaders (solid-colour shaders: 71 fps); float24 depth ~15%; renderer CPU ~15 ms/frame |
| 2 | Metal performance: direct writes into shared buffers instead of staged blits, cheaper per-draw binding, memoryless MSAA targets |
| 5 | Pre-existing e2e allocation-check failure |
