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
    enable_testing()
    find_package(GTest REQUIRED)
    find_package(benchmark REQUIRED)
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
