cmake_minimum_required(VERSION 3.20)
project(slinky LANGUAGES CXX)

option(SLINKY_ENABLE_TESTS "Enable building and running tests and benchmarks" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Mirror the flags from .bazelrc
if(MSVC)
    add_compile_options(
        /Zi        # debug info
        /GR-       # disable RTTI
        /EHs-      # exception handling
        /EHa-
        /wd4018    # sign-compare warnings
    )
else()
    add_compile_options(
        -g
        -fstrict-aliasing
        -Wno-sign-compare
        -fno-rtti
        -fno-exceptions
    )
endif()

if(SLINKY_ENABLE_TESTS)
    include(FetchContent)

    if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
        )

        set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)
    endif()
    if(NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
        FetchContent_Declare(
            google_benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.5
        )

        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(google_benchmark)
    endif()

    # Suppress warnings in fetched dependencies
    set(DISABLE_WARNINGS_FLAG -w)
    if(MSVC)
        set(DISABLE_WARNINGS_FLAG /w)
    endif()
    target_compile_options(gtest PRIVATE ${DISABLE_WARNINGS_FLAG})
    target_compile_options(gtest_main PRIVATE ${DISABLE_WARNINGS_FLAG})
    target_compile_options(gmock PRIVATE ${DISABLE_WARNINGS_FLAG})
    target_compile_options(benchmark PRIVATE ${DISABLE_WARNINGS_FLAG})
    target_compile_options(benchmark_main PRIVATE ${DISABLE_WARNINGS_FLAG})

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
