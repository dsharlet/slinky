# A standalone project that consumes an installed slinky package, the way a
# downstream user would. It is driven by cmake/test/CMakeLists.txt (through
# `ctest --build-and-test`) and can also be run by hand against any prefix:
#
#   cmake -S cmake/test/consumer -B build-consumer -DCMAKE_PREFIX_PATH=<prefix> \
#       -DSLINKY_TEST_COMPONENT=static -DSLINKY_TEST_EXPECTED_TYPE=STATIC_LIBRARY
cmake_minimum_required(VERSION 3.23)
project(slinky_package_test LANGUAGES CXX)

set(SLINKY_TEST_VERSION "" CACHE STRING "Version to request from find_package")
set(SLINKY_TEST_COMPONENT "" CACHE STRING "Component to request (static or shared)")
set(SLINKY_TEST_EXPECTED_TYPE "" CACHE STRING "Expected type of the imported slinky::base target")
set(SLINKY_TEST_CONFLICTING_COMPONENT "" CACHE STRING "Opposite component that must be rejected after loading")

set(find_args ${SLINKY_TEST_VERSION} CONFIG REQUIRED)
if (SLINKY_TEST_COMPONENT)
    list(APPEND find_args COMPONENTS ${SLINKY_TEST_COMPONENT})
endif ()
find_package(slinky ${find_args})

set(libraries base thread_pool thread_pool_impl chrome_trace runtime visualize builder)
foreach (library IN LISTS libraries)
    if (NOT TARGET slinky::${library})
        message(FATAL_ERROR "The slinky package did not define slinky::${library}")
    endif ()
endforeach ()

# Test utilities depend on GoogleTest and are not part of the package.
foreach (library IN ITEMS base_test_util builder_test_util)
    if (TARGET slinky::${library})
        message(FATAL_ERROR "The slinky package exported slinky::${library}")
    endif ()
endforeach ()

if (SLINKY_TEST_EXPECTED_TYPE)
    foreach (library IN LISTS libraries)
        get_target_property(type slinky::${library} TYPE)
        if (NOT type STREQUAL SLINKY_TEST_EXPECTED_TYPE)
            message(FATAL_ERROR "Expected slinky::${library} to be ${SLINKY_TEST_EXPECTED_TYPE}, but it is ${type}")
        endif ()
    endforeach ()
endif ()

# A second request for the other type must fail without replacing the first.
if (SLINKY_TEST_CONFLICTING_COMPONENT)
    find_package(slinky QUIET CONFIG COMPONENTS ${SLINKY_TEST_CONFLICTING_COMPONENT})
    if (slinky_FOUND)
        message(FATAL_ERROR "slinky accepted conflicting component `${SLINKY_TEST_CONFLICTING_COMPONENT}`")
    endif ()
    get_target_property(type_after_conflict slinky::base TYPE)
    if (SLINKY_TEST_EXPECTED_TYPE AND NOT type_after_conflict STREQUAL SLINKY_TEST_EXPECTED_TYPE)
        message(FATAL_ERROR "A conflicting package request replaced slinky::base")
    endif ()
endif ()

# Consumers must not need to ask for C++17 themselves.
get_target_property(features slinky::base INTERFACE_COMPILE_FEATURES)
if (NOT "cxx_std_17" IN_LIST features)
    message(FATAL_ERROR "slinky::base does not require C++17")
endif ()

add_executable(slinky_package_smoke smoke.cc)
target_link_libraries(slinky_package_smoke PRIVATE slinky::builder slinky::visualize slinky::thread_pool_impl slinky::chrome_trace)

enable_testing()
add_test(NAME smoke COMMAND slinky_package_smoke)
