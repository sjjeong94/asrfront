cmake_minimum_required(VERSION 3.18...3.30)
project(asrfront LANGUAGES C)

# scikit-build-core sets SKBUILD=2, which option() would not treat as ON.
if(SKBUILD)
  set(_asrfront_python_default ON)
else()
  set(_asrfront_python_default OFF)
endif()
option(ASRFRONT_BUILD_PYTHON "Build the nanobind Python extension" ${_asrfront_python_default})
option(ASRFRONT_BUILD_TESTS "Build C unit tests" OFF)
option(ASRFRONT_BUILD_BENCH "Build C benchmarks" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# C core: pure C11, links only against libm. No -ffast-math (reproducibility).
# ---------------------------------------------------------------------------
function(asrfront_c_options target)
  target_include_directories(${target} PUBLIC csrc/include PRIVATE csrc)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    # No FMA contraction: results must not depend on the compiler or target (explicit fmaf()
    # calls, where wanted, are unaffected).
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off)
  endif()
endfunction()

# SIMD kernels: the same source compiled per ISA, selected at runtime (kernels.c).
# No FMA flags anywhere, so every variant gives bit-identical results.
add_library(asrfront_kernels_generic OBJECT csrc/kernels_impl.c)
target_compile_definitions(asrfront_kernels_generic PRIVATE AF_ISA=generic)
asrfront_c_options(asrfront_kernels_generic)
set(ASRFRONT_KERNEL_OBJECTS $<TARGET_OBJECTS:asrfront_kernels_generic>)

# Plain-C variant (no vector extensions), i.e. the MSVC code path; selectable with
# ASRFRONT_KERNELS=scalar so it is tested on every platform.
add_library(asrfront_kernels_scalar OBJECT csrc/kernels_impl.c)
target_compile_definitions(asrfront_kernels_scalar PRIVATE AF_ISA=scalar AF_NO_VECTOR_EXT=1)
asrfront_c_options(asrfront_kernels_scalar)
list(APPEND ASRFRONT_KERNEL_OBJECTS $<TARGET_OBJECTS:asrfront_kernels_scalar>)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
  add_library(asrfront_kernels_avx2 OBJECT csrc/kernels_impl.c)
  target_compile_definitions(asrfront_kernels_avx2 PRIVATE AF_ISA=avx2)
  target_compile_options(asrfront_kernels_avx2 PRIVATE $<IF:$<C_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2>)
  asrfront_c_options(asrfront_kernels_avx2)
  list(APPEND ASRFRONT_KERNEL_OBJECTS $<TARGET_OBJECTS:asrfront_kernels_avx2>)
  set(ASRFRONT_HAVE_AVX2 ON)
endif()

add_library(asrfront_core STATIC
  csrc/audio_load.c
  csrc/common.c
  csrc/fft.c
  csrc/kernels.c
  csrc/melbank.c
  csrc/remix.c
  csrc/resample.c
  csrc/wav.c
  csrc/whisper.c
  csrc/sensevoice.c
  ${ASRFRONT_KERNEL_OBJECTS}
)
asrfront_c_options(asrfront_core)
if(ASRFRONT_HAVE_AVX2)
  target_compile_definitions(asrfront_core PRIVATE AF_HAVE_AVX2=1)
endif()
if(UNIX)
  target_link_libraries(asrfront_core PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Python extension
# ---------------------------------------------------------------------------
if(ASRFRONT_BUILD_PYTHON)
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  # SKBUILD_SABI_COMPONENT is "Development.SABIModule" when building a stable-ABI wheel.
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
               ${SKBUILD_SABI_COMPONENT})
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_ext STABLE_ABI NB_STATIC src/asrfront/_ext.cpp)
  target_include_directories(_ext PRIVATE csrc)  # internal headers (test hooks)
  target_link_libraries(_ext PRIVATE asrfront_core)
  install(TARGETS _ext LIBRARY DESTINATION asrfront)
endif()

# ---------------------------------------------------------------------------
# C unit tests
# ---------------------------------------------------------------------------
if(ASRFRONT_BUILD_TESTS)
  enable_testing()
  foreach(t test_api test_fft test_load)
    add_executable(${t} tests/c/${t}.c)
    target_include_directories(${t} PRIVATE csrc)  # internal headers
    target_link_libraries(${t} PRIVATE asrfront_core)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()

# ---------------------------------------------------------------------------
# C benchmarks
# ---------------------------------------------------------------------------
if(ASRFRONT_BUILD_BENCH)
  foreach(b bench_fft bench_frontend)
    add_executable(${b} bench/${b}.c)
    target_include_directories(${b} PRIVATE csrc)
    target_link_libraries(${b} PRIVATE asrfront_core)
  endforeach()
endif()
