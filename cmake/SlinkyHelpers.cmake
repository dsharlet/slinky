# Helpers that keep the per-directory CMakeLists.txt files declarative.

# Applies the flags that mirror .bazelrc to a target.
function(_slinky_set_defaults target)
    if (MSVC)
        target_compile_options(${target}
            PRIVATE
            /Zi  # debug info
            /GR-  # disable RTTI
            /EHs-  # disable C++ exception handling
            /EHa-  # disable structured (SEH) exception handling
            /wd4018  # sign-compare warnings
        )
    else ()
        target_compile_options(${target}
            PRIVATE
            -g  # debug info
            -fstrict-aliasing  # allow type-based alias analysis
            -Wno-sign-compare  # sign-compare warnings
            -fno-rtti  # disable RTTI
            -fno-exceptions  # disable exception handling
        )
    endif ()
endfunction()

# Adds the library slinky_<name> along with the alias slinky::<name>.
#
#   slinky_add_library(<name> [SOURCES <src>...] [HEADERS <hdr>...]
#                      [LINK [PUBLIC <lib>...] [PRIVATE <lib>...]])
#
# HEADERS become the public header set of the target, rooted at the repository
# root so that "slinky/base/foo.h" style includes resolve. Paths are relative to
# the calling directory. LINK is passed to target_link_libraries as is, so it
# takes the usual PUBLIC and PRIVATE keywords. Every library requires (and
# propagates) C++17.
function(slinky_add_library name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;HEADERS;LINK")

    set(target slinky_${name})
    add_library(${target} ${ARG_SOURCES})
    add_library(slinky::${name} ALIAS ${target})

    target_compile_features(${target} PUBLIC cxx_std_17)

    if (ARG_HEADERS)
        target_sources(
            ${target}
            PUBLIC
            FILE_SET HEADERS
            BASE_DIRS "${slinky_SOURCE_DIR}"
            FILES ${ARG_HEADERS}
        )
    endif ()

    if (ARG_LINK)
        target_link_libraries(${target} ${ARG_LINK})
    endif ()

    _slinky_set_defaults(${target})
endfunction()

# Adds a test or benchmark executable.
#
#   slinky_add_test(<name> [BENCHMARK] [SOURCES <src>...] [LINK <lib>...])
#
# By default this is a GoogleTest executable that links GTest::gtest_main and
# whose tests are registered with CTest. With BENCHMARK it is a Google
# Benchmark executable that links benchmark::benchmark_main and is not
# registered with CTest.
#
# The target is named slinky_<group>_<name>_test, or _benchmark with BENCHMARK,
# where <group> comes from the calling directory: its path relative to slinky/,
# with any "test" components dropped and the rest joined by underscores (e.g.
# slinky/builder/test/simplify -> builder_simplify). SOURCES defaults to
# <name>.cc.
function(slinky_add_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG BENCHMARK "" "SOURCES;LINK")

    if (NOT ARG_SOURCES)
        set(ARG_SOURCES ${name}.cc)
    endif ()

    cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR
        BASE_DIRECTORY "${slinky_SOURCE_DIR}/slinky"
        OUTPUT_VARIABLE group
    )
    string(REPLACE "/" ";" group "${group}")
    list(REMOVE_ITEM group test)
    list(JOIN group "_" group)

    if (ARG_BENCHMARK)
        set(target slinky_${group}_${name}_benchmark)
        set(main_lib benchmark::benchmark_main)
    else ()
        set(target slinky_${group}_${name}_test)
        set(main_lib GTest::gtest_main)
    endif ()

    add_executable(${target} ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE ${ARG_LINK} ${main_lib})
    _slinky_set_defaults(${target})

    if (NOT ARG_BENCHMARK)
        gtest_discover_tests(${target})
    endif ()
endfunction()
