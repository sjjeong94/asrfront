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
add_library(asrfront_core STATIC
  csrc/common.c
  csrc/fft.c
  csrc/whisper.c
  csrc/sensevoice.c
)
target_include_directories(asrfront_core PUBLIC csrc/include PRIVATE csrc)
set_target_properties(asrfront_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(asrfront_core PRIVATE /W4)
else()
  target_compile_options(asrfront_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
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
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_ext NB_STATIC src/asrfront/_ext.cpp)
  target_link_libraries(_ext PRIVATE asrfront_core)
  install(TARGETS _ext LIBRARY DESTINATION asrfront)
endif()

# ---------------------------------------------------------------------------
# C unit tests
# ---------------------------------------------------------------------------
if(ASRFRONT_BUILD_TESTS)
  enable_testing()
  foreach(t test_api test_fft)
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
  add_executable(bench_fft bench/bench_fft.c)
  target_include_directories(bench_fft PRIVATE csrc)
  target_link_libraries(bench_fft PRIVATE asrfront_core)
endif()
