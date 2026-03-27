cmake_minimum_required(VERSION 3.20)
project(slinky LANGUAGES CXX)

option(SLINKY_ENABLE_TESTS "Enable building and running tests and benchmarks" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Mirror the flags from .bazelrc
add_compile_options(
    -g
    -fstrict-aliasing
    -Wno-sign-compare
    -fno-rtti
    -fno-exceptions
)

if(SLINKY_ENABLE_TESTS)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
    )

    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest google_benchmark)

    # Suppress warnings in fetched dependencies
    target_compile_options(gtest PRIVATE -w)
    target_compile_options(gtest_main PRIVATE -w)
    target_compile_options(gmock PRIVATE -w)
    target_compile_options(benchmark PRIVATE -w)
    target_compile_options(benchmark_main PRIVATE -w)

    enable_testing()
    include(GoogleTest)
endif()

# All targets include from the repo root so that "slinky/base/foo.h" style
# includes resolve correctly.
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(slinky/base)
add_subdirectory(slinky/runtime)
add_subdirectory(slinky/builder)

if(SLINKY_ENABLE_TESTS)
    add_subdirectory(slinky/apps)
endif()
