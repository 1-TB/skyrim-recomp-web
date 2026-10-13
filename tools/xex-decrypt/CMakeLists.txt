# Builds xex-decrypt from main.cpp plus the XenonUtils/thirdparty sources of a
# local XenonRecomp checkout (pass -DXENON_RECOMP_DIR=...). Driven by
# `cargo xtask decrypt`.
cmake_minimum_required(VERSION 3.20)
project(xex_decrypt C CXX)

set(XENON_RECOMP_DIR "" CACHE PATH "XenonRecomp checkout (cloned with --recursive)")
if(NOT XENON_RECOMP_DIR)
    message(FATAL_ERROR "XENON_RECOMP_DIR must be set")
endif()
set(UTILS "${XENON_RECOMP_DIR}/XenonUtils")
set(THIRDPARTY "${XENON_RECOMP_DIR}/thirdparty")

add_executable(xex-decrypt
    main.cpp
    "${UTILS}/xex_patcher.cpp"
    "${UTILS}/memory_mapped_file.cpp"
    "${THIRDPARTY}/libmspack/libmspack/mspack/lzxd.c"
    "${THIRDPARTY}/tiny-AES-c/aes.c"
)
target_include_directories(xex-decrypt PRIVATE
    "${UTILS}"
    "${THIRDPARTY}/libmspack/libmspack/mspack"
    "${THIRDPARTY}/tiny-AES-c"
    "${THIRDPARTY}/TinySHA1"
)
set_target_properties(xex-decrypt PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_compile_options(xex-decrypt PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-w>)
