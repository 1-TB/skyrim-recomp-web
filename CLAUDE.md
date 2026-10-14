# CLAUDE.md

Static recompilation of Xbox 360 *Skyrim* (title `425307E6`, disc version) that runs in the browser.
XenonRecomp translates the PowerPC code to C++; it is compiled to wasm (Emscripten, pthreads) or
native x86-64. A Rust runtime emulates the kernel/XAM (HLE), the Xenos GPU on WebGPU and audio.
Goal: the game fully playable in a browser tab (desktop and phones). `README.md` has the user-facing
build steps from disc image to browser.

## Hard rules (from the owner)

- **Never commit or embed game code, assets, generated recompiler output or Xbox SDK material**,
  including the retail AES key (it is only referenced through the external XenonRecomp checkout's
  `XenonUtils/xex.h`). Users supply their own dumps. Gitignored: `/game/`, `/generated/`,
  `config/*.toml` (not the `.template`), `*.iso`, `*.xex`, `/target/`, `/web-dist/`, `perf.data*`.
  Audit `git status` / the diff before every commit (shader dumps and saves live under `target/`).
- Full working implementations; no placeholder/TODO stubs except the "unimplemented import"
  logging path.
- No READMEs or guides beyond code comments unless asked.
- Don't assume XenonRecomp's generated signatures or `PPCContext` layout; read its source.
- Commit only when asked. Commit messages end with the session's attribution trailer lines.
- Xenia (BSD) is the reference implementation: port its logic, don't vendor it. A sparse checkout
  of `src/xenia/gpu` is handy (`git clone --filter=blob:none --sparse
  https://github.com/xenia-project/xenia.git`) — keep it outside the repo.
- Code style: match the surrounding code; comment density like the existing files (doc comments
  explain *why*).

## Environment and commands

All paths relative to the repo root. Tools live outside the repo:
`~/tools/XenonRecomp` (patched, built), `~/emsdk` (Emscripten 6.0.12), `~/tools/mesa-lvp`
(lavapipe + `WGPU_NATIVE_DIR` for native WebGPU), `~/tools/extract-xiso`.

```sh
export PATH=$HOME/.cargo/bin:$PATH
# Every build of guest-sys / the hosts needs:
export XENON_RECOMP_OUT=$PWD/generated/recomp XENON_RECOMP_DIR=~/tools/XenonRecomp SKYRIM_XEX=$PWD/game/default.xex

# Native (headless; renders through lavapipe):
source ~/tools/mesa-lvp/env.sh
CARGO_TARGET_DIR=$PWD/target/lead cargo build -p host-native        # dev profile is opt-level 1
# Optimized build with frame pointers for perf (guest C++ at -O3 takes ~15 min):
RUSTFLAGS="-C force-frame-pointers=yes" CARGO_TARGET_DIR=$PWD/target/fprel cargo build --release -p host-native

# Web (writes web-dist/, which the dev server serves live; also writes host_web.wasm.gz):
source ~/emsdk/emsdk_env.sh
cargo xtask build-web --release                                      # ~4-9 min

# Tests (crates that need a GPU use lavapipe; lib tests need the wgpu-native path):
source ~/tools/mesa-lvp/env.sh
LD_LIBRARY_PATH=$WGPU_NATIVE_DIR/lib:$LD_LIBRARY_PATH CARGO_TARGET_DIR=$PWD/target/check cargo test -q -p <crate>
```

Other xtask commands: `setup-xenonrecomp`, `decrypt`, `scan`, `switches`, `recompile`, `synth`,
`e2e [--web]` (synthetic title end to end), `serve`.

### Dev server

`target/debug/xtask serve --port 8080 --game-dir game`, started detached:
`setsid nohup target/debug/xtask serve --port 8080 --game-dir game > target/serve.log 2>&1 < /dev/null &`
(find the old one with `ss -ltnp | grep :8080` and kill it first; rebuild with `cargo build -p xtask`).
It listens on 127.0.0.1 only and is exposed to the owner's devices with `tailscale serve` on port
8443 (HTTPS is required for WebGPU/threads off-localhost). Endpoints:

- static files with COOP/COEP/CORP, ETag + 304, and the `.gz` copy when the client accepts gzip;
- `POST /log?session=ID` → `target/web-logs/ID.log` (the page uploads its log every 3 s — read
  the user's sessions there; `gpu-bench-*.log` are results of `web-dist/gpu-bench.html`);
- `POST/GET /saves/...` → `target/web-saves/` (page buttons Export/Import saves);
- `GET /game/manifest` (`size<TAB>path` lines) and ranged `GET /game/files/PATH` for streaming.

### Reproducing the game natively

- `skyrim-native --game-dir game --content DIR --run-for SECONDS --dump-frames OUT --dump-every N
  --tap BUTTON@SECONDS ...` (tap times are seconds after boot; the box is slow, so menus take
  minutes). `--render-debug MODE` (repeatable): `scissor`, `nodraw`, `nopool`, `nomemo`,
  `cpu-resolve`, `cpu-texture`, `frames=N`, `trace-frame=N`, `trace-every=K`.
- The owner's saves (exported from the browser) are in `target/web-saves/...` and copied to
  `target/perf/save-content`. Scripts in `target/perf/` (not committed), e.g. `save-fix.sh`: copy
  the save content, `--tap START@40 --tap A@200 --tap A@260 --tap A@320` (Continue), dump every 50
  frames; the outdoor Helgen scene appears around frame 900-1300 (timing varies run to run).
- Native in-game rendering is ~1 fps (lavapipe); a 1500 s run reaches roughly frame 1200-1500.
- Kill natives with `pgrep -x skyrim-native`; never `pkill -f <pattern>` that matches your own
  shell's command line (it kills the tool shell). Wait loops: `until <cond>; do sleep 20; done`
  with `run_in_background`, not foreground sleeps.

### Browser harness

`PW_DIR=<scratchpad>/pw xvfb-run -a -s "-screen 0 1600x1200x24" node hosts/host-web/tools/browser-run.mjs
--dist web-dist --game game --profile target/browser-profile --out DIR --seconds N [--args "..."]
[--stream] [--audio] [--profile-at S:D] [--stacks-at S] [--reload-after S] [--saves DIR]
[--step "COND => ACTIONS" ...]`. Steps wait for log lines / perf conditions / swap counts and
then tap, screenshot, `measure S LABEL` or exit (grammar in the script's header); each run ends
with a summary table (fps, renderer busy %, uploads/frame, live GPU bytes by label, texture cache,
wasm memory, process PSS, driver GPU memory) also written to `OUT/summary.json`, and exits 1 if a
step times out. Title signal: the first `XamContentCreateEnumerator` call; main menu:
`XamContentCreateEnumerator: type 0x1, N item(s)`; Continue: `Savegame.dat` opens; in game:
`draws>=1000`. Chrome here only has
SwiftShader, so it checks correctness, not GPU performance (the owner measures on an NVIDIA desktop
and an Android phone).

## Architecture

| Path | What |
|---|---|
| `crates/xex` | XEX2 parsing. |
| `crates/runtime-core` | Guest memory (4 GiB space; on wasm guest address == linear address), threads, dirty-page tracking (`dirty.rs`, `skyrt_dirty_pages`). |
| `crates/kernel-hle` | xboxkrnl/XAM HLE, objects and sync, GPU command processor front end (`gpu.rs`), input, audio, content/saves, on-screen keyboard hook. |
| `crates/vfs` | Guest devices/paths; game root is a directory or a streamed `RemoteTree` (`remote.rs`, 2 MiB chunk LRU); OPFS mounting on the web. |
| `crates/guest-sys` | Builds the recompiled C++ (CMake) and links it; import name table. |
| `cpp/ppc_context.h` | Forked XenonRecomp header (rev 3): FPSCR emulation, timebase, `PPC_EA`/`skyrt_ea` view folding, `PPC_MARK_WRITTEN` dirty marking. |
| `tools/xenonrecomp/skyrt.patch` | Patch applied by `setup-xenonrecomp` (missing instructions, switch handling, EA routing, vupkd3d128/vpkd3d128 forms). |
| `crates/xenos-state` | Register decoding: draws, render targets, viewport, blend, resolves (port of Xenia's draw_util etc.). |
| `crates/xenos-shader` | Xenos microcode → WGSL translator, disassembler. |
| `crates/xenos-texture`, `xenos-texture-cache` | Formats/untiling; texture cache with GPU loads from the GPU copy of guest memory (`gpu_load.rs`), pooling, retirement. |
| `crates/xenos-edram` | EDRAM emulation: host render targets, ownership transfers between formats/pitches, resolves stored straight into the GPU copy of guest memory, clears. |
| `crates/xenos-render` | Renderer thread: command queue, `executor.rs` (draw path, constants, passes), `shared_memory.rs` (512 MiB guest memory mirrored in 4×128 MiB storage buffers), `float_patch.rs`, presenter, native/web loops, `debug.rs`. |
| `crates/gpu-webgpu` | Safe layer over `webgpu.h` (emdawnwebgpu on web, wgpu-native natively), op counters. |
| `crates/xma-decoder` | XMA audio (LGPL-2.1+ port of FFmpeg wmapro — licensing still needs the owner's review). |
| `hosts/common` | Shared CLI options and boot (`boot.rs`), GPU/audio glue. |
| `hosts/host-native` | Linux debug host. |
| `hosts/host-web` | Browser host: `src/main.rs`, `src/stream.rs`, `src/text_input.rs`; JS libraries in `lib/`; page in `web/` (`index.html`, `boot.js` settings/sources/fullscreen/memory log, `input.js`, `touch.js`, `zip.js`, `audio-worklet.js`, `gpu-bench.html`). |
| `xtask` | Build tooling and the dev server. |

Key facts:

- Skyrim XEX: image 0x82000000+0x17B0000, entry 0x82FAC510; helper addresses are in `README.md`.
- Memory map (runtime-core `memory.rs`): host owns < 0x40000000 on wasm; virtual heap
  0x40000000-0x70000000; stacks 0x70000000-0x7E000000; system 0x7E000000; MMIO 0x7FC00000;
  XEX 0x80000000; 4K-page heap 0x90000000; physical memory stored once at 0xA0000000 (views at
  0xC0000000/0xE0000000 folded by `fold_view`, which must match `skyrt_ea` in the C++ header).
  Consequence: wasm linear memory is 3 GiB from boot.
- Web threading: `PROXY_TO_PTHREAD`; the renderer owns the canvas as an OffscreenCanvas on its own
  worker and must yield to the event loop (pump budget; idle parks with `Atomics.waitAsync`).
  Chrome's WebGPU only flushes between tasks.
- WebGPU objects released in workers are freed only by GC in browsers: destroy textures/buffers
  explicitly (texture cache retires, `destroy_retired` at frame end); at most 2 frames in flight.
- Bind group caches keyed by raw handles must hold the handles (address reuse caused flicker).
- Resolves are stored into the GPU copy of guest memory (no readback), pages tracked as
  GPU-authoritative; uploads (`writeBuffer`) take effect before a submission's commands, hence
  the batch conflict tracking in `SharedMemory::sync`.
- Float constants: each draw's VS/PS block is a patch on the previous block (mask + changed
  vec4s), rebuilt by one compute dispatch submitted before the batch (`float_patch.rs`).
- Float24 depth (D24FS8): host depth holds half the guest value (Xenia's
  `full_float24_in_0_to_1`; `pack_depth`/`unpack_depth` in xenos-edram, viewport z ×0.5) so EDRAM
  tiles survive color↔depth ownership transfers — Skyrim's HUD passes alias depth over the frame
  (this was the red-sky bug).
- Color outputs are clamped per target format (`xe_color_range`; 7e3 [0, 31.875], fixed16 [-32, 32]).
- Web page settings persist in localStorage `skyrt-settings`; frame cap maps to `--vblank-hz`
  (30 → 60 default, 60 → 120, unlimited → 1000; Skyrim presents every second vblank).

## Debugging rendering

`--render-debug trace-frame=N [trace-every=K]` (native) logs every draw of the traced frames
(shaders, targets, flags, fetch constants, blend registers, PS c0-c7, first vertex bytes) and every
resolve (`ResolveInfo`), writes each shader's WGSL and disassembly to `target/shader-dump/`
(ids are per run — clear the directory between runs), and writes each traced pass's color targets
as PNG there. Those PNGs are whole host render targets: content outside the pass's rectangle is
stale from earlier passes. Narrow a bug by finding the first pass whose output (inside its
rectangle) is wrong, then compare its translation with the disassembly and Xenia.

Performance: the web perf line every 5 s (`perf: fps, draws/frame; renderer thread: % executing /
waiting for GPU / waiting for game` plus per-frame WebGPU op counts) and `ExecutorStats` every 300
frames (`renderer: frame N`). Native profiles mostly show lavapipe, so measure browser costs from the
owner's logs; their `gpu-bench.html` run showed upload bytes, not submit counts, dominate.

## Status (2026-10-13)

- Boots, menus, New Game and loading saves work; Helgen renders correctly on desktop at ~60 fps
  with the 60 fps cap (`--vblank-hz 120`), renderer thread ~73% busy before the float patcher.
- Phone (Android): imports via .zip work; runs die silently while loading the world — suspected tab
  OOM. The page now logs memory/device info every 20 s; read the next phone session log in
  `target/web-logs/` to confirm. Streaming from the server works but was slow on the phone
  (progress is logged now). The wasm module is 176 MiB of code (27 MB gzipped).

Open items:

1. Confirm the phone crash cause from the memory log, then cut memory (3 GiB linear memory reserve,
   512 MiB GPU mirror of guest memory, texture cache).
2. Measure the float patcher on the owner's GPU; next upload to cut is the ~0.9 MB/frame of other
   constant blocks (system constants per draw).
3. Possible pre-existing race: green macroblocks seen once in the Bethesda logo video under
   SwiftShader (blank YUV texels; video decode vs texture load). Unconfirmed.
4. LGPL xma-decoder licensing review by the owner.
