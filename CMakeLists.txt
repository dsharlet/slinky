cmake_minimum_required(VERSION 3.23)
project(slinky VERSION 0.1.0 LANGUAGES CXX)

include(CMakeDependentOption)

option(SLINKY_ENABLE_TESTS
    "Enable building and running tests and benchmarks"
    ${PROJECT_IS_TOP_LEVEL}
)

cmake_dependent_option(
    SLINKY_INSTALL_RULES "Include slinky's install() rules"
    ${PROJECT_IS_TOP_LEVEL} "NOT CMAKE_SKIP_INSTALL_RULES" OFF
)

# Let projects that include us choose the library type independently of the
# BUILD_SHARED_LIBS they use for everything else.
if (DEFINED SLINKY_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${SLINKY_SHARED_LIBS}")
endif ()

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
        "Configure with -DSLINKY_SHARED_LIBS=OFF."
    )
endif ()

# Static libraries that use std::thread need -pthread, not just -lpthread, on
# older glibc.
if (PROJECT_IS_TOP_LEVEL)
    option(THREADS_PREFER_PTHREAD_FLAG "Prefer -pthread to -lpthread" ON)
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

if (SLINKY_INSTALL_RULES)
    add_subdirectory(cmake/packaging)
endif ()
