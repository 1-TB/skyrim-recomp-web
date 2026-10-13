#!/usr/bin/env bash
# One-shot setup and build: Skyrim (Xbox 360) disc image -> web build in web-dist/.
#
# Runs every step from README.md. Each step checks whether its work is already
# done and skips it if so, which makes the script safe to re-run: after an
# interruption, a failed step or a code change it picks up where it stopped.
#
#   ./setup.sh                      # everything, auto-detecting the .iso
#   ./setup.sh ~/isos/skyrim1.iso   # everything, with an explicit image
#   ./setup.sh --title-update tu... # ...with a title update package applied
#   ./setup.sh --serve              # ...then serve on http://127.0.0.1:8080/
#   ./setup.sh --status             # show which steps are done
#   ./setup.sh --only build         # just rebuild the web version
#
# Run ./setup.sh --help for every option.

set -Eeo pipefail

# --------------------------------------------------------------------------------------------
# Constants (from README.md / xtask)
# --------------------------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EMSCRIPTEN_VERSION="6.0.12"
TESTED_TITLE_ID="425307E6"
MIN_RAM_GB=16
MIN_DISK_GB=20

# Helper addresses for the tested disc version (README step 4). `scan` confirms
# the first eight against the executable; setjmp/longjmp it can't find itself.
declare -A KNOWN_ADDR=(
    [restgprlr_14_address]=0x82F91CB0
    [savegprlr_14_address]=0x82F91C60
    [restfpr_14_address]=0x82F923CC
    [savefpr_14_address]=0x82F92380
    [restvmx_14_address]=0x82F928C8
    [savevmx_14_address]=0x82F92630
    [restvmx_64_address]=0x82F9295C
    [savevmx_64_address]=0x82F926C4
    [longjmp_address]=0x82F92BC0
    [setjmp_address]=0x82F931B0
)
SCANNED_KEYS=(restgprlr_14_address savegprlr_14_address restfpr_14_address savefpr_14_address
    restvmx_14_address savevmx_14_address restvmx_64_address savevmx_64_address)

ALL_STEPS=(deps extract xenonrecomp decrypt update config switches recompile build serve)

GAME_DIR="$ROOT/game"
GAME_XEX="$GAME_DIR/default.xex"
ORIGINAL_XEX="$ROOT/generated/xex/default.original.xex"
# The decrypted disc executable, kept when a title update replaces game/default.xex.
BASE_XEX="$ROOT/generated/xex/default.base.xex"
TU_DIR="$ROOT/generated/tu"
# Must match vfs::TITLE_UPDATE_DIR: the runtime links update: to it.
TU_CONTENT="$GAME_DIR/\$TitleUpdate"
CONFIG="$ROOT/config/skyrim.toml"
CONFIG_TEMPLATE="$ROOT/config/skyrim.toml.template"
SWITCHES="$ROOT/config/skyrim_switch_tables.toml"
RECOMP_OUT="$ROOT/generated/recomp"
STATE_DIR="$ROOT/generated/.setup"
LOG_DIR="$ROOT/generated/logs"
ENV_FILE="$ROOT/generated/env.sh"

# Overridable through the environment.
XENON_RECOMP_DIR="${XENON_RECOMP_DIR:-$HOME/tools/XenonRecomp}"
EMSDK_DIR="${EMSDK_DIR:-${EMSDK:-$HOME/emsdk}}"
EXTRACT_XISO_DIR="${EXTRACT_XISO_DIR:-$HOME/tools/extract-xiso}"
export XENON_RECOMP_DIR

# --------------------------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------------------------

ISO="${SKYRIM_ISO:-}"
TITLE_UPDATE="${SKYRIM_TITLE_UPDATE:-}"
NO_TITLE_UPDATE=0
RELEASE=1
SERVE=0
PORT=8080
JOBS=""
ASSUME_YES=0
RECONFIGURE=0
FORCE_RECOMPILE=0
STATUS_ONLY=0
ONLY=""
SKIP=""

usage() {
    cat <<EOF
Usage: ./setup.sh [options] [DISC.iso]

Takes your Skyrim Xbox 360 disc image (disc 1) all the way to a web build in
web-dist/. Steps that are already done are skipped, so re-running is cheap.

Steps: ${ALL_STEPS[*]}
  deps         system packages (cmake, ninja, clang...), Rust nightly, emsdk $EMSCRIPTEN_VERSION, extract-xiso
  extract      unpack the .iso into game/
  xenonrecomp  clone, patch and build XenonRecomp in \$XENON_RECOMP_DIR
  decrypt      decrypt game/default.xex (original kept in generated/xex/)
  update       apply a title update package: patch game/default.xex, its files into
               game/\$TitleUpdate/ (the runtime's update: drive)
  config       write config/skyrim.toml, with addresses checked by \`cargo xtask scan\`
  switches     find jump tables -> config/skyrim_switch_tables.toml
  recompile    PowerPC -> C++ into generated/recomp/
  build        cargo xtask build-web -> web-dist/
  serve        cargo xtask serve (only with --serve or --only serve)

Options:
  --iso PATH          disc image (default: \$SKYRIM_ISO, else the only/Skyrim-named *.iso
                      in the repo, its parent directory, ~/Downloads or ~)
  --title-update PATH title update package (an STFS file such as tu00000002_00000000; default:
                      \$SKYRIM_TITLE_UPDATE, else one for this game found where disc images are
                      looked for, the newest if several)
  --no-title-update   use the disc version (undoes an applied title update)
  --debug             debug web build instead of release
  --serve             start the local server when the build is done
  --port N            server port (default $PORT)
  -j, --jobs N        parallel build jobs (default: picked from CPU count and RAM)
  -y, --yes           install missing dependencies without asking
  --reconfigure       regenerate config/skyrim.toml even if it exists
  --force-recompile   rerun the recompiler even if its output is up to date
  --only STEPS        run only these steps (comma-separated)
  --skip STEPS        skip these steps (comma-separated)
  --status            show the state of every step and exit
  -h, --help          this help

Environment: XENON_RECOMP_DIR (default ~/tools/XenonRecomp), EMSDK_DIR (default \$EMSDK or
~/emsdk), EXTRACT_XISO_DIR (default ~/tools/extract-xiso), SKYRIM_ISO, SKYRIM_TITLE_UPDATE.

Logs of every step go to generated/logs/. Afterwards, \`source generated/env.sh\` sets up a
shell for running cargo xtask commands by hand.
EOF
}

# --------------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------------

if [[ -t 1 ]]; then
    B=$'\e[1m' DIM=$'\e[2m' RED=$'\e[31m' GRN=$'\e[32m' YEL=$'\e[33m' BLU=$'\e[34m' RST=$'\e[0m'
else
    B="" DIM="" RED="" GRN="" YEL="" BLU="" RST=""
fi

CURRENT_STEP=""
CURRENT_LOG=""
START_TIME=$SECONDS

step()  { CURRENT_STEP="$1"; echo; echo "${B}${BLU}==> [$1]${RST}${B} $2${RST}"; }
info()  { echo "    $*"; }
ok()    { echo "    ${GRN}✓${RST} $*"; }
skip()  { echo "    ${DIM}✓ $* (skipped)${RST}"; }
warn()  { echo "    ${YEL}!${RST} $*" >&2; }
die()   { echo; echo "${RED}${B}error:${RST} $*" >&2; exit 1; }

on_error() {
    local code=$?
    echo >&2
    if [[ -n "$CURRENT_STEP" ]]; then
        echo "${RED}${B}Setup failed${RST} in step ${B}$CURRENT_STEP${RST} (exit $code)." >&2
    else
        echo "${RED}${B}Setup failed${RST} (exit $code)." >&2
    fi
    [[ -n "$CURRENT_LOG" ]] && echo "Full log: $CURRENT_LOG" >&2
    echo "Fix the problem and re-run ./setup.sh; finished steps are skipped." >&2
    exit "$code"
}
trap on_error ERR
trap 'echo; echo "${YEL}Interrupted.${RST} Re-run ./setup.sh to continue." >&2; exit 130' INT

confirm() {
    [[ $ASSUME_YES == 1 ]] && return 0
    [[ -t 0 ]] || return 1
    local reply
    read -r -p "    $1 [Y/n] " reply
    [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# run LOGNAME CMD...: run a command, showing its output and saving it to generated/logs/.
# A log is started fresh on its first use in a run and appended to after that.
declare -A LOG_STARTED=()
run() {
    local name="$1"; shift
    mkdir -p "$LOG_DIR"
    CURRENT_LOG="$LOG_DIR/$name.log"
    [[ -n "${LOG_STARTED[$name]:-}" ]] || : >"$CURRENT_LOG"
    LOG_STARTED[$name]=1
    echo "    ${DIM}\$ $*${RST}"
    echo "# $(date -Is)  $*" >>"$CURRENT_LOG"
    "$@" 2>&1 | tee -a "$CURRENT_LOG"
    CURRENT_LOG=""
}

have() { command -v "$1" >/dev/null 2>&1; }

need_arg() { [[ -n "${2:-}" ]] || die "$1 needs a value"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iso) need_arg "$1" "${2:-}"; ISO="$2"; shift ;;
        --title-update) need_arg "$1" "${2:-}"; TITLE_UPDATE="$2"; shift ;;
        --no-title-update) NO_TITLE_UPDATE=1 ;;
        --debug) RELEASE=0 ;;
        --release) RELEASE=1 ;;
        --serve) SERVE=1 ;;
        --port) need_arg "$1" "${2:-}"; PORT="$2"; shift ;;
        -j|--jobs) need_arg "$1" "${2:-}"; JOBS="$2"; shift ;;
        -y|--yes) ASSUME_YES=1 ;;
        --reconfigure) RECONFIGURE=1 ;;
        --force-recompile) FORCE_RECOMPILE=1 ;;
        --only) need_arg "$1" "${2:-}"; ONLY="$2"; shift ;;
        --skip) need_arg "$1" "${2:-}"; SKIP="$2"; shift ;;
        --status) STATUS_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option: $1 (see --help)" ;;
        *) ISO="$1" ;;
    esac
    shift
done
[[ "$PORT" =~ ^[0-9]+$ ]] || die "--port needs a number"
[[ -z "$JOBS" || "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs needs a positive number"
[[ -z "$TITLE_UPDATE" || $NO_TITLE_UPDATE == 0 ]] || die "--title-update and --no-title-update exclude each other"

# True if file $1 exists and is newer than every existing file among the rest.
newer_than_all() {
    local target="$1"; shift
    [[ -e "$target" ]] || return 1
    local f
    for f in "$@"; do
        [[ -e "$f" && "$f" -nt "$target" ]] && return 1
    done
    return 0
}

selected() {
    local s="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$s,"* ]] || return 1
    elif [[ "$s" == serve && $SERVE == 0 ]]; then
        return 1
    fi
    [[ ",$SKIP," != *",$s,"* ]]
}

for s in ${ONLY//,/ } ${SKIP//,/ }; do
    [[ " ${ALL_STEPS[*]} " == *" $s "* ]] || die "unknown step '$s' (steps: ${ALL_STEPS[*]})"
done

# --------------------------------------------------------------------------------------------
# XEX inspection (header only; mirrors crates/xex)
# --------------------------------------------------------------------------------------------

# Prints "<state> <title id> <version>", state one of: plain (loadable), encrypted, compressed,
# invalid.
xex_state() {
    python3 - "$1" <<'PY'
import struct, sys
try:
    with open(sys.argv[1], "rb") as f:
        head = f.read(0x18)
        if head[:4] != b"XEX2":
            raise ValueError
        header_size, = struct.unpack(">I", head[8:12])
        count, = struct.unpack(">I", head[0x14:0x18])
        f.seek(0)
        h = f.read(header_size)
    opts = {}
    for i in range(count):
        k, v = struct.unpack(">II", h[0x18 + i * 8 : 0x20 + i * 8])
        opts[k] = v
    title, version = "unknown", "unknown"
    if 0x00040006 in opts:
        o = opts[0x00040006]
        v, = struct.unpack(">I", h[o + 4 : o + 8])
        version = "%d.%d.%d.%d" % (v >> 28, (v >> 24) & 0xF, (v >> 8) & 0xFFFF, v & 0xFF)
        title = "%08X" % struct.unpack(">I", h[o + 12 : o + 16])
    state = "plain"
    if 0x000003FF in opts:
        o = opts[0x000003FF]
        enc, comp = struct.unpack(">HH", h[o + 4 : o + 8])
        if enc != 0:
            state = "encrypted"
        elif comp not in (0, 1):
            state = "compressed"
    print(state, title, version)
except Exception:
    print("invalid unknown unknown")
PY
}

# --------------------------------------------------------------------------------------------
# Environment for the builds
# --------------------------------------------------------------------------------------------

load_emsdk() {
    [[ -f "$EMSDK_DIR/emsdk_env.sh" ]] || return 1
    # shellcheck disable=SC1091
    source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1 || return 1
    have emcc
}

emsdk_version_ok() {
    load_emsdk || return 1
    emcc --version 2>/dev/null | head -n1 | grep -q " $EMSCRIPTEN_VERSION "
}

export_guest_env() {
    export XENON_RECOMP_OUT="$RECOMP_OUT"
    export XENON_RECOMP_DIR
    export SKYRIM_XEX="$GAME_XEX"
}

write_env_file() {
    mkdir -p "$(dirname "$ENV_FILE")"
    cat >"$ENV_FILE" <<EOF
# Written by setup.sh. \`source generated/env.sh\` to run cargo xtask build commands by hand.
export XENON_RECOMP_OUT="$RECOMP_OUT"
export XENON_RECOMP_DIR="$XENON_RECOMP_DIR"
export SKYRIM_XEX="$GAME_XEX"
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
EOF
}

pick_jobs() {
    if [[ -n "$JOBS" ]]; then
        export CARGO_BUILD_JOBS="$JOBS"
    elif [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
        # The recompiled C++ units are huge; budget ~2 GB of RAM per compiler process.
        local cpus mem_gb
        cpus=$(nproc 2>/dev/null || echo 4)
        mem_gb=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 8)
        JOBS=$(( mem_gb / 2 < cpus ? mem_gb / 2 : cpus ))
        (( JOBS < 1 )) && JOBS=1
        export CARGO_BUILD_JOBS="$JOBS"
    fi
}

# --------------------------------------------------------------------------------------------
# Step: deps
# --------------------------------------------------------------------------------------------

EXTRACT_XISO=""
find_extract_xiso() {
    if have extract-xiso; then EXTRACT_XISO="$(command -v extract-xiso)"; return 0; fi
    if [[ -x "$EXTRACT_XISO_DIR/build/extract-xiso" ]]; then EXTRACT_XISO="$EXTRACT_XISO_DIR/build/extract-xiso"; return 0; fi
    return 1
}

missing_system_tools() {
    local t absent=()
    for t in git cmake ninja clang clang++ python3 gzip; do have "$t" || absent+=("$t"); done
    echo "${absent[*]}"
}

install_system_packages() {
    local tools="$1" cmd
    if have pacman; then
        cmd=(sudo pacman -S --needed --noconfirm git cmake ninja clang python gzip)
    elif have apt-get; then
        cmd=(sudo apt-get install -y git cmake ninja-build clang python3 gzip)
    elif have dnf; then
        cmd=(sudo dnf install -y git cmake ninja-build clang python3 gzip)
    elif have zypper; then
        cmd=(sudo zypper install -y git cmake ninja clang python3 gzip)
    else
        die "missing tools: $tools. Install them with your package manager and re-run."
    fi
    warn "missing tools: $tools"
    confirm "Install them with: ${cmd[*]} ?" || die "missing tools: $tools (install them, or re-run with -y)"
    if ! [[ -t 0 ]] && ! sudo -n true 2>/dev/null; then
        die "sudo needs a password and there is no terminal to ask for it. Run: ${cmd[*]}"
    fi
    run deps-packages "${cmd[@]}"
}

deps_done() {
    [[ -z "$(missing_system_tools)" ]] && have rustup && find_extract_xiso && emsdk_version_ok
}

step_deps() {
    step deps "Checking build dependencies"

    # Machine size (README: ~16 GB RAM, 20 GB free disk).
    local mem_gb disk_gb
    mem_gb=$(awk '/MemTotal/ {print int(($2 + 524288) / 1048576)}' /proc/meminfo 2>/dev/null || echo 0)
    disk_gb=$(df -Pk "$ROOT" | awk 'NR == 2 {print int($4 / 1048576)}')
    if (( mem_gb < MIN_RAM_GB )); then
        warn "${mem_gb} GB of RAM; $MIN_RAM_GB GB is recommended (use -j 2 if the build runs out of memory)"
    else
        ok "${mem_gb} GB RAM"
    fi
    if (( disk_gb < MIN_DISK_GB )) && [[ ! -d "$RECOMP_OUT" ]]; then
        warn "${disk_gb} GB free disk; about $MIN_DISK_GB GB is needed"
    else
        ok "${disk_gb} GB free disk"
    fi
    [[ "$(uname -m)" == x86_64 ]] || warn "untested architecture $(uname -m); README targets x86-64"

    local missing
    missing="$(missing_system_tools)"
    if [[ -n "$missing" ]]; then
        install_system_packages "$missing"
        missing="$(missing_system_tools)"
        [[ -z "$missing" ]] || die "still missing after install: $missing"
    fi
    ok "git, cmake, ninja, clang, python3"

    # Rust: rustup installs the pinned nightly from rust-toolchain.toml.
    if ! have rustup; then
        warn "rustup is not installed"
        if confirm "Install rustup from https://sh.rustup.rs ?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
            # shellcheck disable=SC1091
            source "$HOME/.cargo/env"
        else
            die "install rustup (https://rustup.rs) and re-run"
        fi
    fi
    local channel
    channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
    if rustup toolchain list | grep -q "^$channel"; then
        ok "Rust $channel"
    else
        info "installing Rust $channel (from rust-toolchain.toml)"
    fi
    # Also adds any component/target the toolchain file lists that is missing.
    # (`rustup toolchain install` with no argument needs rustup >= 1.28; `rustup show` installs on older ones.)
    run deps-rust rustup toolchain install || run deps-rust rustup show
    have cargo || die "cargo not found on PATH after installing Rust; open a new shell or source ~/.cargo/env"

    # Emscripten.
    if emsdk_version_ok; then
        ok "Emscripten $EMSCRIPTEN_VERSION ($EMSDK_DIR)"
    else
        if [[ ! -x "$EMSDK_DIR/emsdk" ]]; then
            confirm "Install emsdk into $EMSDK_DIR ?" || die "install emsdk $EMSCRIPTEN_VERSION (see README) and re-run"
            run deps-emsdk git clone https://github.com/emscripten-core/emsdk "$EMSDK_DIR"
        fi
        info "installing Emscripten $EMSCRIPTEN_VERSION"
        run deps-emsdk "$EMSDK_DIR/emsdk" install "$EMSCRIPTEN_VERSION"
        run deps-emsdk "$EMSDK_DIR/emsdk" activate "$EMSCRIPTEN_VERSION"
        emsdk_version_ok || die "emcc $EMSCRIPTEN_VERSION not usable after install; see $LOG_DIR/deps-emsdk.log"
        ok "Emscripten $EMSCRIPTEN_VERSION"
    fi

    # extract-xiso: only needed while game/ hasn't been unpacked yet.
    if find_extract_xiso; then
        ok "extract-xiso ($EXTRACT_XISO)"
    elif [[ -d "$GAME_DIR/Data" ]] && ! [[ -e "$STATE_DIR/extract.partial" ]]; then
        skip "extract-xiso: game/ is already unpacked"
    else
        info "building extract-xiso into $EXTRACT_XISO_DIR"
        [[ -d "$EXTRACT_XISO_DIR/.git" ]] ||
            run deps-extract-xiso git clone --recursive https://github.com/XboxDev/extract-xiso "$EXTRACT_XISO_DIR"
        run deps-extract-xiso cmake -S "$EXTRACT_XISO_DIR" -B "$EXTRACT_XISO_DIR/build" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
        run deps-extract-xiso cmake --build "$EXTRACT_XISO_DIR/build"
        find_extract_xiso || die "extract-xiso build produced no binary in $EXTRACT_XISO_DIR/build"
        ok "extract-xiso ($EXTRACT_XISO)"
    fi
}

# --------------------------------------------------------------------------------------------
# Step: extract
# --------------------------------------------------------------------------------------------

extract_done() {
    [[ ! -e "$STATE_DIR/extract.partial" && -d "$GAME_DIR/Data" ]] &&
        [[ -f "$GAME_XEX" || -f "$ORIGINAL_XEX" ]]
}

find_iso() {
    if [[ -n "$ISO" ]]; then
        [[ -f "$ISO" ]] || die "disc image not found: $ISO"
        return
    fi
    local candidates=() f
    shopt -s nullglob nocaseglob
    for f in "$ROOT"/*.iso "$(dirname "$ROOT")"/*.iso "$HOME"/Downloads/*.iso "$HOME"/*.iso; do
        candidates+=("$(realpath "$f")")
    done
    # The same file can be found through two of those directories.
    mapfile -t candidates < <(printf '%s\n' "${candidates[@]}" | awk 'NF && !seen[$0]++')
    shopt -u nullglob nocaseglob
    if (( ${#candidates[@]} > 1 )); then
        local named=()
        for f in "${candidates[@]}"; do [[ "${f,,}" == *skyrim* ]] && named+=("$f"); done
        (( ${#named[@]} > 0 )) && candidates=("${named[@]}")
    fi
    case ${#candidates[@]} in
        0) die "no disc image found. Pass it: ./setup.sh /path/to/skyrim1.iso" ;;
        1) ISO="${candidates[0]}" ;;
        *)
            if [[ -t 0 ]]; then
                echo "    Several disc images found:"
                select f in "${candidates[@]}"; do [[ -n "$f" ]] && { ISO="$f"; break; }; done
            else
                die "several disc images found (${candidates[*]}); pass one explicitly"
            fi
            ;;
    esac
}

step_extract() {
    step extract "Unpacking the disc into game/"
    if extract_done; then skip "game/ is unpacked"; return; fi

    find_iso
    info "disc image: $ISO ($(du -h "$ISO" | cut -f1))"
    find_extract_xiso || die "extract-xiso not found; run the deps step first"

    mkdir -p "$STATE_DIR"
    if [[ -e "$STATE_DIR/extract.partial" ]]; then
        warn "a previous extraction was interrupted; starting over"
        rm -rf "$GAME_DIR"
    elif [[ -e "$GAME_DIR" && -n "$(ls -A "$GAME_DIR" 2>/dev/null)" ]]; then
        die "game/ exists but looks incomplete (no Data/ or default.xex). Move it away and re-run."
    fi
    touch "$STATE_DIR/extract.partial"
    run extract "$EXTRACT_XISO" -x "$ISO" -d "$GAME_DIR"
    [[ -d "$GAME_DIR/Data" && -f "$GAME_XEX" ]] ||
        die "game/ has no default.xex and Data/ after extraction; is this Skyrim disc 1?"
    rm -f "$STATE_DIR/extract.partial"
    ok "unpacked $(du -sh "$GAME_DIR" | cut -f1) into game/"
}

# --------------------------------------------------------------------------------------------
# Step: xenonrecomp
# --------------------------------------------------------------------------------------------

xenonrecomp_done() {
    [[ -x "$XENON_RECOMP_DIR/build/XenonRecomp/XenonRecomp" ]]
}

step_xenonrecomp() {
    step xenonrecomp "Setting up XenonRecomp in $XENON_RECOMP_DIR"
    # Always run: xtask checks the pinned commit and patch itself, and an up-to-date
    # build is a no-op.
    run xenonrecomp cargo xtask setup-xenonrecomp
    xenonrecomp_done || die "XenonRecomp binary missing after setup"
    ok "XenonRecomp ready"
}

# --------------------------------------------------------------------------------------------
# Step: decrypt
# --------------------------------------------------------------------------------------------

decrypt_done() {
    [[ -f "$GAME_XEX" ]] && [[ "$(xex_state "$GAME_XEX")" == plain\ * ]]
}

check_title() {
    local title="$1"
    if [[ "$title" == "$TESTED_TITLE_ID" ]]; then
        ok "title ID $title (the tested version)"
    else
        warn "title ID $title; this repo is tested with $TESTED_TITLE_ID. Continuing, but expect problems."
    fi
}

step_decrypt() {
    step decrypt "Decrypting the executable"
    local state title
    if [[ -f "$GAME_XEX" ]]; then
        read -r state title _ < <(xex_state "$GAME_XEX")
        case "$state" in
            plain) skip "game/default.xex is already decrypted"; return ;;
            invalid) die "game/default.xex is not a valid XEX2 file" ;;
        esac
        check_title "$title"
        mkdir -p "$(dirname "$ORIGINAL_XEX")"
        if [[ -f "$ORIGINAL_XEX" ]] && ! cmp -s "$GAME_XEX" "$ORIGINAL_XEX"; then
            mv "$ORIGINAL_XEX" "$ORIGINAL_XEX.bak"
            warn "replaced a different generated/xex/default.original.xex (old copy: .bak)"
        fi
        mv "$GAME_XEX" "$ORIGINAL_XEX"
        info "kept the original as generated/xex/default.original.xex"
    elif [[ -f "$ORIGINAL_XEX" ]]; then
        read -r state title _ < <(xex_state "$ORIGINAL_XEX")
        [[ "$state" != invalid ]] || die "$ORIGINAL_XEX is not a valid XEX2 file"
        check_title "$title"
    else
        die "no game/default.xex or generated/xex/default.original.xex; run the extract step"
    fi

    run decrypt cargo xtask decrypt --in "$ORIGINAL_XEX" --out "$GAME_XEX"
    decrypt_done || die "game/default.xex is still not decrypted after xex-decrypt"
    ok "wrote decrypted game/default.xex"
}

# --------------------------------------------------------------------------------------------
# Step: update
# --------------------------------------------------------------------------------------------

UPDATE_STAMP="$STATE_DIR/update.stamp"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

# Lists the title update packages for title $1 found in the given directories and one level
# below them, as "<version>\t<path>", newest first. Only the first bytes of each file are read.
list_title_updates() {
    local title="$1"; shift
    PYTHONDONTWRITEBYTECODE=1 python3 - "$ROOT/tools/stfs-extract.py" "$title" "$@" <<'PY2'
import glob, importlib.util, os, sys
spec = importlib.util.spec_from_file_location("stfs", sys.argv[1])
stfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stfs)
title, found = int(sys.argv[2], 16), {}
for d in sys.argv[3:]:
    for path in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, "*", "*")):
        try:
            if not os.path.isfile(path) or os.path.getsize(path) < 0xB000:
                continue
            with open(path, "rb") as f:
                if f.read(4) not in (b"LIVE", b"PIRS"):
                    continue
            pkg = stfs.Package(path)
            if pkg.content_type != 0xB0000 or pkg.title_id != title:
                continue
            ver = pkg.installer[2] if pkg.installer else pkg.version
            found[os.path.realpath(path)] = ver
        except (OSError, stfs.StfsError):
            pass
for path, ver in sorted(found.items(), key=lambda kv: -kv[1]):
    print(f"{stfs.version(ver)}\t{path}")
PY2
}

# Sets TU_PKG to the package to apply ("" for none): --title-update, else the newest one found
# where disc images are looked for.
TU_PKG=""
resolve_title_update() {
    TU_PKG=""
    [[ $NO_TITLE_UPDATE == 0 ]] || return 0
    if [[ -n "$TITLE_UPDATE" ]]; then
        [[ -f "$TITLE_UPDATE" ]] || die "title update package not found: $TITLE_UPDATE"
        TU_PKG="$(realpath "$TITLE_UPDATE")"
        return 0
    fi
    local xex="$GAME_XEX" title dirs=("$ROOT" "$(dirname "$ROOT")" "$HOME/Downloads" "$HOME")
    [[ -f "$BASE_XEX" ]] && xex="$BASE_XEX"
    [[ -f "$xex" ]] || return 0
    read -r _ title _ < <(xex_state "$xex")
    [[ "$title" != unknown ]] || return 0
    [[ -n "$ISO" && -f "$ISO" ]] && dirs=("$(dirname "$(realpath "$ISO")")" "${dirs[@]}")
    TU_PKG="$(list_title_updates "$title" "${dirs[@]}" | head -n1 | cut -f2)"
}

# The package's identity as recorded in the stamp: a changed or replaced file is re-applied.
package_id() { echo "$1 $(stat -c '%s %Y' "$1")"; }

title_update_applied() { [[ -f "$BASE_XEX" && -f "$GAME_XEX" ]] && ! cmp -s "$BASE_XEX" "$GAME_XEX"; }

update_done() {
    resolve_title_update
    if [[ -z "$TU_PKG" ]]; then
        # Without a package to apply, an applied update is only undone on request.
        [[ $NO_TITLE_UPDATE == 0 ]] || ! title_update_applied
    else
        [[ -f "$UPDATE_STAMP" && -d "$TU_CONTENT" && -f "$GAME_XEX" ]] &&
            [[ "$(sed -n 1p "$UPDATE_STAMP")" == "$(package_id "$TU_PKG")" ]] &&
            [[ "$(sed -n 2p "$UPDATE_STAMP")" == "$(sha_of "$GAME_XEX")" ]]
    fi
}

# generated/xex/default.base.xex: the decrypted disc executable, the input of every patch.
ensure_base_xex() {
    [[ -f "$BASE_XEX" ]] && return 0
    mkdir -p "$(dirname "$BASE_XEX")"
    if [[ -f "$ORIGINAL_XEX" ]]; then
        run update cargo xtask decrypt --in "$ORIGINAL_XEX" --out "$BASE_XEX"
    elif [[ -f "$GAME_XEX" && ! -f "$UPDATE_STAMP" ]]; then
        # Never patched, so this is still the disc executable (it was decrypted already on disc).
        cp "$GAME_XEX" "$BASE_XEX"
    else
        die "the disc executable is gone (no generated/xex/default.original.xex); re-extract the disc"
    fi
    [[ "$(xex_state "$BASE_XEX")" == plain\ * ]] || die "could not recreate $BASE_XEX"
}

# Replaces game/default.xex atomically, so an interruption never leaves half a file.
install_xex() { mv -f "$1" "$GAME_XEX.tmp" && mv -f "$GAME_XEX.tmp" "$GAME_XEX"; }

step_update() {
    step update "Title update"
    if update_done; then
        if [[ -n "$TU_PKG" ]]; then
            skip "title update $(sed -n 3p "$UPDATE_STAMP") is applied"
        elif title_update_applied; then
            skip "keeping the applied title update; its package wasn't found (--no-title-update removes it)"
        else
            skip "no title update package (pass one with --title-update; the disc version works too)"
        fi
        return
    fi

    if [[ -z "$TU_PKG" ]]; then
        cp "$BASE_XEX" "$GAME_XEX.new"
        install_xex "$GAME_XEX.new"
        rm -rf "$TU_CONTENT" "$TU_DIR" "$UPDATE_STAMP"
        ok "restored the disc executable and removed game/\$TitleUpdate/"
        return
    fi

    info "package: $TU_PKG"
    ensure_base_xex
    run update python3 "$ROOT/tools/stfs-extract.py" --info "$TU_PKG" >/dev/null
    local base_title tu_title tu_version
    read -r _ base_title _ < <(xex_state "$BASE_XEX")
    tu_title=$(sed -n 's/^  title ID \([0-9A-Fa-f]\{8\}\),.*/\1/p' "$LOG_DIR/update.log" | head -n1)
    tu_version=$(sed -n 's/.*installer base version [^,]*, version \([0-9.]*\).*/\1/p' "$LOG_DIR/update.log" | head -n1)
    [[ "${tu_title^^}" == "$base_title" ]] ||
        die "that package is for title ${tu_title:-unknown}, but the game is $base_title"

    # Extract straight into game/, then move the executable patch out: only the data files
    # belong on the update: drive.
    local partial="$TU_CONTENT.partial"
    rm -rf "$partial" "$TU_DIR"
    run update python3 "$ROOT/tools/stfs-extract.py" "$TU_PKG" "$partial"
    [[ -f "$partial/default.xexp" ]] || { rm -rf "$partial"; die "the package has no default.xexp; is it a title update?"; }
    mkdir -p "$TU_DIR"
    mv "$partial/default.xexp" "$TU_DIR/default.xexp"

    run update cargo xtask patch --xex "$BASE_XEX" --patch "$TU_DIR/default.xexp" --out "$GAME_XEX.new"
    local state title version
    read -r state title version < <(xex_state "$GAME_XEX.new")
    [[ "$state" == plain && "$title" == "$base_title" ]] ||
        die "the patched executable is not usable ($state, title $title)"
    install_xex "$GAME_XEX.new"
    rm -rf "$TU_CONTENT"
    mv "$partial" "$TU_CONTENT"

    mkdir -p "$STATE_DIR"
    printf '%s\n%s\n%s\n' "$(package_id "$TU_PKG")" "$(sha_of "$GAME_XEX")" "${tu_version:-$version}" >"$UPDATE_STAMP"
    ok "game/default.xex is now version $version; $(find "$TU_CONTENT" -type f | wc -l) files in game/\$TitleUpdate/ ($(du -sh "$TU_CONTENT" | cut -f1))"
}

# --------------------------------------------------------------------------------------------
# Step: config
# --------------------------------------------------------------------------------------------

config_problems() {
    [[ -f "$CONFIG" ]] || { echo "missing"; return; }
    local p=()
    grep -Eq '^[[:space:]]*switch_table_file_path[[:space:]]*=' "$CONFIG" || p+=("switch_table_file_path is commented out")
    grep -Eq '^[[:space:]]*[a-z0-9_]+_address[[:space:]]*=[[:space:]]*0x0+([[:space:]]|$)' "$CONFIG" &&
        p+=("some helper addresses are still 0x00000000")
    echo "${p[*]}"
}

# sha256 of the game/default.xex the config was written or checked for.
CONFIG_STAMP="$STATE_DIR/config.xex"
CONFIG_MARKER="# Written by setup.sh"

config_done() {
    [[ -z "$(config_problems)" ]] || return 1
    [[ ! -f "$CONFIG_STAMP" || ! -f "$GAME_XEX" ]] || [[ "$(cat "$CONFIG_STAMP")" == "$(sha_of "$GAME_XEX")" ]]
}

signed_hex() { if (( $1 < 0 )); then printf -- '-0x%X' $(( -$1 )); else printf '+0x%X' "$1"; fi; }

# Prints the value of address key $1 in config file $2.
config_value() { sed -n "s/^$1[[:space:]]*=[[:space:]]*\(0x[0-9A-Fa-f]*\).*/\1/p" "$2" | head -n1; }

# Prints the address the last `scan` found for key $1, or nothing.
scanned_value() { sed -n "s/^$1 = \(0x[0-9A-Fa-f]*\)$/\1/p" "$LOG_DIR/scan.log" | tail -n1; }

# Checks that setjmp and longjmp of the tested disc executable (generated/xex/default.base.xex)
# reappear $1 (decimal, signed) bytes further in game/default.xex: same instructions, apart from branch targets
# and immediates, which move with the code and data around them.
verify_relocated_jmp() {
    python3 - "$BASE_XEX" "$GAME_XEX" "$1" "${KNOWN_ADDR[longjmp_address]}" "${KNOWN_ADDR[setjmp_address]}" <<'PY2'
import struct, sys

def image(path):
    d = open(path, "rb").read()
    header_size, sec, count = struct.unpack(">I", d[8:12])[0], struct.unpack(">I", d[0x10:0x14])[0], struct.unpack(">I", d[0x14:0x18])[0]
    load, = struct.unpack(">I", d[sec + 0x110 : sec + 0x114])
    opts = dict(struct.unpack(">II", d[0x18 + i * 8 : 0x20 + i * 8]) for i in range(count))
    o = opts.get(0x000003FF)
    comp = struct.unpack(">H", d[o + 6 : o + 8])[0] if o else 0
    if comp == 0:
        return load, d[header_size:]
    if comp != 1:
        sys.exit(f"{path}: unexpected compression {comp}")
    # Basic compression: runs of stored bytes followed by runs of zeros.
    info_size, = struct.unpack(">I", d[o : o + 4])
    img, pos = bytearray(), header_size
    for b in range((info_size - 8) // 8):
        size, zeros = struct.unpack(">II", d[o + 8 + b * 8 : o + 16 + b * 8])
        img += d[pos : pos + size] + bytes(zeros)
        pos += size
    return load, bytes(img)

(base_load, base), (cur_load, cur) = image(sys.argv[1]), image(sys.argv[2])
delta = int(sys.argv[3])
for name, a in (("longjmp", int(sys.argv[4], 16)), ("setjmp", int(sys.argv[5], 16))):
    n = 64
    x = struct.unpack(f">{n}I", base[a - base_load : a - base_load + n * 4])
    y = struct.unpack(f">{n}I", cur[a + delta - cur_load : a + delta - cur_load + n * 4])
    exact = sum(p == q for p, q in zip(x, y))
    similar = all(p == q or p >> 16 == q >> 16 or (p >> 26 == 18 and q >> 26 == 18) for p, q in zip(x, y))
    if not similar or exact < n * 3 // 4:
        sys.exit(f"{name} is not at 0x{a + delta:08X} ({exact}/{n} instructions identical)")
PY2
}

step_config() {
    step config "Writing config/skyrim.toml"
    local key xex_sha
    xex_sha="$(sha_of "$GAME_XEX")"
    if [[ -f "$CONFIG" && $RECONFIGURE == 0 ]]; then
        local problems
        problems="$(config_problems)"
        [[ -z "$problems" ]] || die "config/skyrim.toml exists but $problems. Fix it, or re-run with --reconfigure."
        if [[ -f "$CONFIG_STAMP" && "$(cat "$CONFIG_STAMP")" == "$xex_sha" ]]; then
            skip "config/skyrim.toml matches game/default.xex (--reconfigure to regenerate)"
            return
        fi
        # Written for another executable, or by hand: keep it if its helpers are where the scan
        # finds them in this one.
        info "checking config/skyrim.toml against game/default.xex"
        run scan cargo xtask scan --xex "$GAME_XEX" >/dev/null
        local stale=() ours=1
        for key in "${SCANNED_KEYS[@]}"; do
            local have_v found_v
            have_v="$(config_value "$key" "$CONFIG")" found_v="$(scanned_value "$key")"
            [[ -z "$found_v" || $((have_v)) == $((found_v)) ]] || stale+=("$key")
        done
        if (( ${#stale[@]} == 0 )); then
            mkdir -p "$STATE_DIR"
            echo "$xex_sha" >"$CONFIG_STAMP"
            skip "config/skyrim.toml matches game/default.xex (--reconfigure to regenerate)"
            return
        fi
        # Regenerate only configs this script wrote (older ones carry the tested addresses).
        if ! grep -q "^$CONFIG_MARKER" "$CONFIG"; then
            for key in "${!KNOWN_ADDR[@]}"; do
                [[ $(( $(config_value "$key" "$CONFIG") )) == $(( KNOWN_ADDR[$key] )) ]] || ours=0
            done
        fi
        (( ours )) || die "config/skyrim.toml doesn't match game/default.xex (${stale[*]}). Fix it, or re-run with --reconfigure."
        info "game/default.xex changed since config/skyrim.toml was written; regenerating it"
    fi

    info "scanning game/default.xex for the save/restore helpers"
    run scan cargo xtask scan --xex "$GAME_XEX"

    declare -A addr
    local mismatches=() unresolved=() found deltas=()
    for key in "${!KNOWN_ADDR[@]}"; do addr[$key]="${KNOWN_ADDR[$key]}"; done
    for key in "${SCANNED_KEYS[@]}"; do
        found="$(scanned_value "$key")"
        if [[ -z "$found" ]]; then
            unresolved+=("$key")
        elif (( found != KNOWN_ADDR[$key] )); then
            mismatches+=("$key: scan 0x${found:2}, tested ${KNOWN_ADDR[$key]}")
            addr[$key]="$found"
        fi
        [[ -z "$found" ]] || deltas+=("$(( found - KNOWN_ADDR[$key] ))")
    done
    (( ${#unresolved[@]} == 0 )) ||
        warn "scan found no unique match for ${unresolved[*]}; using the tested values"

    # A title update shifts the runtime library as one block: if every helper moved by the same
    # offset, setjmp/longjmp (which scan can't find) moved by it too. That is verified against
    # the disc executable.
    local relocated=0
    if (( ${#mismatches[@]} > 0 && ${#unresolved[@]} == 0 )) &&
        [[ "$(printf '%s\n' "${deltas[@]}" | sort -u | wc -l)" == 1 ]] && [[ -f "$BASE_XEX" ]]; then
        local delta=${deltas[0]} why
        if why="$(verify_relocated_jmp "$delta" 2>&1)"; then
            for key in longjmp_address setjmp_address; do
                addr[$key]="$(printf '0x%08X' $(( KNOWN_ADDR[$key] + delta )))"
            done
            relocated=1
            info "this executable's runtime library sits $(signed_hex "$delta") bytes from the tested disc"
            info "version's; setjmp/longjmp verified at ${addr[setjmp_address]}/${addr[longjmp_address]}"
        else
            warn "helpers moved by $(signed_hex "$delta"), but: $why"
        fi
    fi
    if (( ${#mismatches[@]} > 0 && !relocated )); then
        warn "this executable differs from the tested disc version:"
        for key in "${mismatches[@]}"; do warn "  $key"; done
        warn "using the scanned helper addresses, but longjmp_address/setjmp_address must be"
        warn "found by hand (see the comments in config/skyrim.toml and the scan log)."
    fi

    mkdir -p "$(dirname "$CONFIG")" "$STATE_DIR"
    local args=() version
    read -r _ _ version < <(xex_state "$GAME_XEX")
    for key in "${!addr[@]}"; do args+=("$key=${addr[$key]}"); done
    python3 - "$CONFIG_TEMPLATE" "$CONFIG" "$CONFIG_MARKER for game/default.xex version $version; it is regenerated when that file changes." "${args[@]}" <<'PY2'
import re, sys
src, dst, marker, pairs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
text = open(src).read()
text = re.sub(r'^#\s*(switch_table_file_path\s*=)', r'\1', text, count=1, flags=re.M)
for pair in pairs:
    key, value = pair.split("=", 1)
    text, n = re.subn(rf'^({re.escape(key)}\s*=\s*)0x[0-9A-Fa-f]+', rf'\g<1>{value.upper().replace("0X", "0x")}', text, flags=re.M)
    if n != 1:
        sys.exit(f"template has no single '{key}' line")
open(dst, "w").write(marker + "\n" + text)
PY2
    echo "$xex_sha" >"$CONFIG_STAMP"
    config_done || die "generated config still has problems: $(config_problems)"
    (( ${#mismatches[@]} == 0 || relocated )) ||
        die "review longjmp_address/setjmp_address in config/skyrim.toml, then re-run ./setup.sh"
    ok "wrote config/skyrim.toml"
}

# --------------------------------------------------------------------------------------------
# Step: switches
# --------------------------------------------------------------------------------------------

switches_done() { newer_than_all "$SWITCHES" "$GAME_XEX"; }

step_switches() {
    step switches "Finding jump tables"
    if switches_done; then skip "config/skyrim_switch_tables.toml is up to date"; return; fi
    run switches cargo xtask switches --xex "$GAME_XEX" --out "$SWITCHES"
    [[ -s "$SWITCHES" ]] || die "no switch tables written"
    ok "wrote config/skyrim_switch_tables.toml"
}

# --------------------------------------------------------------------------------------------
# Step: recompile
# --------------------------------------------------------------------------------------------

recompile_inputs=("$CONFIG" "$SWITCHES" "$GAME_XEX" "$ROOT/cpp/ppc_context.h"
    "$XENON_RECOMP_DIR/build/XenonRecomp/XenonRecomp")

recompile_done() {
    [[ -f "$RECOMP_OUT/ppc_recomp_shared.h" ]] &&
        newer_than_all "$STATE_DIR/recompile.stamp" "${recompile_inputs[@]}"
}

step_recompile() {
    step recompile "Recompiling PowerPC code to C++"
    if [[ $FORCE_RECOMPILE == 0 ]] && recompile_done; then
        skip "generated/recomp/ is up to date (--force-recompile to redo)"
        return
    fi
    info "this takes a few minutes"
    rm -f "$STATE_DIR/recompile.stamp"
    run recompile cargo xtask recompile --config "$CONFIG"
    [[ -f "$RECOMP_OUT/ppc_recomp_shared.h" ]] || die "recompiler wrote no output"

    local missing_tables
    missing_tables=$(grep -c "no switch table entry present" "$LOG_DIR/recompile.log" || true)
    (( missing_tables == 0 )) ||
        warn "$missing_tables jump tables were not found by \`switches\` (see generated/logs/recompile.log)"

    mkdir -p "$STATE_DIR"
    touch "$STATE_DIR/recompile.stamp"
    write_env_file
    ok "$(find "$RECOMP_OUT" -name 'ppc_recomp.*.cpp' | wc -l) C++ files in generated/recomp/ ($(du -sh "$RECOMP_OUT" | cut -f1))"
}

# --------------------------------------------------------------------------------------------
# Step: build
# --------------------------------------------------------------------------------------------

build_done() {
    [[ -f "$ROOT/web-dist/index.html" ]] && newer_than_all "$ROOT/web-dist/host_web.wasm" "$STATE_DIR/recompile.stamp"
}

step_build() {
    local profile=release flag=--release
    [[ $RELEASE == 1 ]] || { profile=debug; flag=""; }
    step build "Building the web version ($profile)"
    [[ -f "$RECOMP_OUT/ppc_recomp_shared.h" ]] || die "generated/recomp/ is empty; run the recompile step"
    load_emsdk || die "emsdk not found in $EMSDK_DIR; run the deps step"
    export_guest_env
    write_env_file
    pick_jobs
    info "$CARGO_BUILD_JOBS parallel jobs (-j N to change). The first build takes a long time."
    local t=$SECONDS
    # shellcheck disable=SC2086
    run build cargo xtask build-web $flag
    build_done || die "web-dist/ is incomplete after the build"
    ok "web-dist/ built in $(( (SECONDS - t) / 60 ))m $(( (SECONDS - t) % 60 ))s ($(du -h "$ROOT/web-dist/host_web.wasm" | cut -f1) wasm)"
}

# --------------------------------------------------------------------------------------------
# Step: serve
# --------------------------------------------------------------------------------------------

step_serve() {
    step serve "Serving on http://127.0.0.1:$PORT/"
    build_done || die "web-dist/ is not built; run the build step"
    info "open ${B}http://127.0.0.1:$PORT/${RST} in Chrome/Edge (WebGPU). Ctrl+C to stop."
    info "for other devices use an HTTPS tunnel, e.g. tailscale serve --bg --https=8443 http://127.0.0.1:$PORT"
    trap - ERR INT
    exec cargo xtask serve --port "$PORT" --game-dir "$GAME_DIR"
}

# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

status_line() {
    local s="$1"
    if [[ "$s" == serve ]]; then
        printf "  ${DIM}%-6s %s${RST}\n" "-" "$s"
    elif "${s}_done"; then
        printf "  ${GRN}%-6s${RST} %s\n" "done" "$s"
    else
        printf "  ${YEL}%-6s${RST} %s\n" "todo" "$s"
    fi
}

if [[ $STATUS_ONLY == 1 ]]; then
    echo "${B}Setup status${RST} ($ROOT)"
    for s in "${ALL_STEPS[@]}"; do status_line "$s"; done
    exit 0
fi

echo "${B}Skyrim (Xbox 360) web build setup${RST}"
for s in "${ALL_STEPS[@]}"; do
    selected "$s" || continue
    "step_$s"
done

CURRENT_STEP=""
echo
echo "${GRN}${B}Done${RST} in $(( (SECONDS - START_TIME) / 60 ))m $(( (SECONDS - START_TIME) % 60 ))s."
if build_done && ! selected serve; then
    echo "Start the server with:  ${B}./setup.sh --only serve${RST}   (or: cargo xtask serve --port $PORT --game-dir game)"
    echo "then open http://127.0.0.1:$PORT/"
fi
