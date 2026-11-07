# iOS port and Metal backend: tracker

As of 2026-11-07. The game runs on Mac and iPhone (18 Pro Max, A20 Pro, iOS 27.3 beta) through a
native Metal renderer. Missing clothing/grass (iPhone) and streaky roofs (Mac) are fixed: both were
z-fighting from host float32 depth where the Xbox stores float24. Open work is performance (target 120 fps).

## Status

| Area | Mac (M2) | iPhone 18 Pro Max |
| --- | --- | --- |
| Boot, menus, New Game, loading saves | Works | Works |
| Renderer | Native Metal (default); wgpu kept for A/B | Native Metal (default) |
| Riverwood save frame rate | ~53 fps at 1× (Whiterun ~50), GPU-bound | 60 fps (cap) at 1×; uncapped ~56 in town, 95–110 in lighter scenes; renderer CPU-bound |
| Character clothing, grass (alpha-tested) | Correct | Correct (float24 depth) |
| Streaky thatch/roof textures | Fixed (float24 depth) | Correct |
| Frame cap | 120 fps (`--vblank-hz 240` default) | 120 fps + `CADisableMinimumFrameDurationOnPhone` |
| GPU errors in logs | None | None |

## Work log (newest first)

| When | Milestone | Key changes |
| --- | --- | --- |
| Nov 7 | Phone findings, more overlap | Texture loads write RGBA8/RGBA16F/R32F/RGBA32F textures directly (no copy); 1 MiB × 32 constant sets; user-interactive QoS for guest/renderer threads; iOS no longer waits at fences (`--gpu-sync` restores; it pinned the phone at 60 fps); perf line reports presenting separately. Mac 47 → 53 fps; phone holds 60 at 1× and is renderer-CPU-bound above that (~17 ms/frame of renderer work in town vs ~8.6 ms on the Mac) |
| Nov 7 | Submissions overlap, CPU-written constants | Fetch constants decoded on the CPU per draw; texture loads read the mirror in place (no gather blit); per-submission buffers rotate (constant blocks, index rings) so a submission no longer waits for the previous one; on Metal constant blocks are written straight into shared memory (no patch dispatch); float constant diff walks set bits; `--no-msaa` + iOS Anti-aliasing setting. Mac 42 → 47.4 fps |
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

Per 21 ms frame (47 fps): pixel shaders 8.7 ms (scene pass ~4.5 ms; ALU-bound on integer
emulation work, F32 ~16% of ALU), EDRAM resolves ~4.4 ms (~25/frame; size-independent cost,
likely dependency waits in the render-to-texture chains), vertex 4.3 ms, texture loads 1.6 ms,
blits 1.3 ms (texture-load copies, index gathers), GPU idle ~12% (encoder transitions). Float24
depth costs ~4 ms (no hidden-surface removal with shader depth writes) and is needed for
correctness. Renderer CPU ~8 ms/frame (37% of a core at 45 fps).

Phone (iPhone 18 Pro Max, 1×): GPU has headroom; the renderer thread is the limit in dense
scenes (~94% busy at ~56 fps). Renderer CPU per frame on the Mac (Time Profiler): Executor::draw
~6 ms (constants ~1.7, texture binding ~0.7, sync ~0.6, Metal encoding ~1.3), register writes
~1 ms. A Time Profiler run on the device (`xcrun xctrace record --device <udid> --attach
Skyrim --template 'Time Profiler'`) is the next step there.

Remaining ideas, largest first: render-to-texture shortcut (a resolve whose destination is
sampled with the same format becomes a texture copy instead of resolve → guest memory → texture
load: ~4–5 ms); cheaper SM3 multiply semantics in pixel shaders; explicit synchronization for
the guest-memory mirror (untracked buffers measured +1.4 fps). 120 fps at 1× needs the GPU frame
under 8.3 ms, i.e. pixel and vertex shaders about twice as fast as now; higher resolution scales
multiply the pixel work.

Profile with `xcrun xctrace record --template 'Metal System Trace' --attach PID` (add
`--instrument 'Metal GPU Counters'` for limiters; `--render-debug gpu-time` labels passes by
target) and `--template 'Time Profiler'` for the CPU.

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
