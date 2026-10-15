# Builds XenonRecomp's generated C++ plus the generated import shim into one
# static library. Driven by crates/guest-sys/build.rs; works with clang
# (native) and with the Emscripten toolchain file (wasm32).
cmake_minimum_required(VERSION 3.20)
project(skyrim_guest LANGUAGES CXX)

set(XENON_RECOMP_OUT "" CACHE PATH "XenonRecomp output directory (ppc_recomp.*.cpp, ppc_config.h, ...)")
set(SIMDE_DIR "" CACHE PATH "simde headers (XenonRecomp/thirdparty/simde)")
set(SKYRT_SHIM "" CACHE FILEPATH "Generated import shim (skyrt_imports.cpp)")
set(SKYRT_DISPATCH "" CACHE FILEPATH "Generated indirect-call dispatch (skyrt_dispatch.cpp)")
option(SKYRT_CHECK_FRAMES "Report scalar accesses straddling non-adjacent linear frames" OFF)

foreach(var XENON_RECOMP_OUT SIMDE_DIR SKYRT_SHIM SKYRT_DISPATCH)
    if(NOT ${var})
        message(FATAL_ERROR "${var} must be set")
    endif()
endforeach()

file(GLOB RECOMP_SOURCES CONFIGURE_DEPENDS "${XENON_RECOMP_OUT}/ppc_recomp.*.cpp")
if(NOT RECOMP_SOURCES)
    message(FATAL_ERROR "no ppc_recomp.*.cpp in ${XENON_RECOMP_OUT}; run `cargo xtask recompile` first")
endif()

add_library(skyrim_guest STATIC
    ${RECOMP_SOURCES}
    "${SKYRT_SHIM}"
    # Replaces ppc_func_mapping.cpp (see guest-sys build.rs, fn dispatch).
    "${SKYRT_DISPATCH}"
)
target_include_directories(skyrim_guest PRIVATE "${XENON_RECOMP_OUT}" "${SIMDE_DIR}")
set_target_properties(skyrim_guest PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
if(NOT EMSCRIPTEN)
    # Linked into a position-independent Rust executable. (On wasm the link is static and PIC
    # would only add a base-relative indirection to every global access.)
    set_target_properties(skyrim_guest PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_compile_options(skyrim_guest PRIVATE
    # Generated code type-puns through unions and casts and produces many
    # unused-label/variable diagnostics.
    -fno-strict-aliasing
    -w
)

if(EMSCRIPTEN)
    target_compile_options(skyrim_guest PRIVATE
        -pthread
        -msimd128
        # Guest threads leave recompiled code by unwinding (ExTerminateThread),
        # and setjmp/longjmp are redirected to the host; both use wasm EH.
        -fwasm-exceptions
        -sSUPPORT_LONGJMP=wasm
        # Guest memory starts at address 0 on wasm; `base + ea` must never be
        # treated as a null-pointer dereference.
        -fno-delete-null-pointer-checks
    )
else()
    target_compile_options(skyrim_guest PRIVATE
        -march=x86-64-v2
        # Rust unwinds through these frames on thread termination.
        -fexceptions
        -funwind-tables
    )
endif()

if(SKYRT_CHECK_FRAMES)
    target_compile_definitions(skyrim_guest PRIVATE SKYRT_CHECK_FRAMES_ENABLED)
endif()
