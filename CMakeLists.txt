cmake_minimum_required(VERSION 3.23)
project(slinky LANGUAGES CXX)

option(SLINKY_ENABLE_TESTS
    "Enable building and running tests and benchmarks"
    ${PROJECT_IS_TOP_LEVEL}
)

# Don't impose a choice of language extensions on projects that include us.
if (PROJECT_IS_TOP_LEVEL)
    option(CMAKE_CXX_EXTENSIONS
        "Enable compiler-specific language extensions" OFF
    )
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SlinkyHelpers)

# Nothing marks symbols for export from a DLL yet, so a shared build on
# Windows would produce DLLs without import libraries.
if (WIN32 AND BUILD_SHARED_LIBS)
    message(FATAL_ERROR
        "Shared libraries are not yet supported on Windows. "
        "Configure with -DBUILD_SHARED_LIBS=OFF."
    )
endif ()

find_package(Threads REQUIRED)

if (SLINKY_ENABLE_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    find_package(benchmark REQUIRED)
    include(GoogleTest)
endif ()

add_subdirectory(slinky/base)
add_subdirectory(slinky/runtime)
add_subdirectory(slinky/builder)

if (SLINKY_ENABLE_TESTS)
    add_subdirectory(slinky/apps)
endif ()
