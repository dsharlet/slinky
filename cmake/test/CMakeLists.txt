# Tests for the installed package. Requires the install rules in cmake/packaging.

set(packaging_binary_dir "${PROJECT_BINARY_DIR}/cmake/packaging")

# Install into a staging prefix, then consume it from standalone projects,
# exactly as downstream users would. This tree builds one library type, so
# only that type is installed; see consumer/CMakeLists.txt for the consumer.
set(staging "${CMAKE_CURRENT_BINARY_DIR}/prefix")
add_test(
    NAME slinky_package_install
    COMMAND
    "${CMAKE_COMMAND}" "-DCMAKE_INSTALL_PREFIX=${staging}" "-DCMAKE_INSTALL_CONFIG_NAME=$<CONFIG>" -P
    "${packaging_binary_dir}/cmake_install.cmake"
)
set_tests_properties(slinky_package_install PROPERTIES FIXTURES_SETUP slinky_package)

if (BUILD_SHARED_LIBS)
    set(type shared)
    set(other static)
    set(expected_type SHARED_LIBRARY)
else ()
    set(type static)
    set(other shared)
    set(expected_type STATIC_LIBRARY)
endif ()

# Adds a test that configures, builds, and runs the consumer project with the
# given -D options. Pass REJECTED_WITH <regex> for requests the package must
# reject: the test then passes only if its output matches the regex, so that
# it cannot pass because of an unrelated failure.
function (slinky_add_package_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "PREFIX;REJECTED_WITH" OPTIONS)
    if (NOT ARG_PREFIX)
        set(ARG_PREFIX "${staging}")
    endif ()
    set(binary_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    add_test(
        NAME slinky_package_${name}
        COMMAND
        "${CMAKE_CTEST_COMMAND}" --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/consumer" "${binary_dir}"
        --build-generator "${CMAKE_GENERATOR}" --build-config "$<CONFIG>" --build-options
        "-DCMAKE_PREFIX_PATH=${ARG_PREFIX}" "-DCMAKE_BUILD_TYPE=$<CONFIG>"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}" ${ARG_OPTIONS} --test-command
        "${CMAKE_CTEST_COMMAND}" --output-on-failure -C "$<CONFIG>"
    )
    if (ARG_PREFIX STREQUAL staging)
        set(fixture slinky_package)
    else ()
        set(fixture slinky_package_both)
    endif ()
    set_tests_properties(slinky_package_${name} PROPERTIES FIXTURES_REQUIRED ${fixture})
    if (ARG_REJECTED_WITH)
        set_tests_properties(slinky_package_${name} PROPERTIES PASS_REGULAR_EXPRESSION "${ARG_REJECTED_WITH}")
    endif ()
endfunction ()

slinky_add_package_test(default OPTIONS -DSLINKY_TEST_EXPECTED_TYPE=${expected_type})
slinky_add_package_test(
    component
    OPTIONS
    -DSLINKY_TEST_COMPONENT=${type}
    -DSLINKY_TEST_EXPECTED_TYPE=${expected_type}
    -DSLINKY_TEST_CONFLICTING_COMPONENT=${other}
)
slinky_add_package_test(
    version
    OPTIONS
    -DSLINKY_TEST_VERSION=${slinky_VERSION_MAJOR}.${slinky_VERSION_MINOR}
    -DSLINKY_TEST_EXPECTED_TYPE=${expected_type}
)

# Only this tree's type is installed, and the next major version is not ours.
math(EXPR next_major "${slinky_VERSION_MAJOR} + 1")
slinky_add_package_test(
    reject_${other}
    REJECTED_WITH "slinky `${other}` libraries were requested but not found"
    OPTIONS -DSLINKY_TEST_COMPONENT=${other}
)
slinky_add_package_test(
    reject_version
    REJECTED_WITH "with requested version \"${next_major}.0\""
    OPTIONS -DSLINKY_TEST_VERSION=${next_major}.0
)

# The other library type is built and installed alongside this tree's own, so
# that the two coexist in one prefix. This is the only way to exercise the
# config file's choice between them, because a single tree installs one type.
set(staging_both "${CMAKE_CURRENT_BINARY_DIR}/prefix-both")
if (BUILD_SHARED_LIBS)
    set(other_shared_libs OFF)
else ()
    set(other_shared_libs ON)
endif ()
add_test(
    NAME slinky_package_install_both
    COMMAND
    "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DBINARY_DIR=${packaging_binary_dir}"
    "-DOTHER_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/other-build" "-DPREFIX=${staging_both}"
    "-DOTHER_SHARED_LIBS=${other_shared_libs}" "-DCONFIG=$<CONFIG>" "-DGENERATOR=${CMAKE_GENERATOR}"
    "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}" -P "${CMAKE_CURRENT_SOURCE_DIR}/install_both.cmake"
)
set_tests_properties(slinky_package_install_both PROPERTIES FIXTURES_SETUP slinky_package_both TIMEOUT 600)

# With no hint, static is preferred. Explicit requests are binding, and
# SLINKY_SHARED_LIBS outranks BUILD_SHARED_LIBS.
slinky_add_package_test(both_default PREFIX "${staging_both}" OPTIONS -DSLINKY_TEST_EXPECTED_TYPE=STATIC_LIBRARY)
slinky_add_package_test(
    both_build_shared_libs
    PREFIX "${staging_both}"
    OPTIONS -DBUILD_SHARED_LIBS=ON -DSLINKY_TEST_EXPECTED_TYPE=SHARED_LIBRARY
)
slinky_add_package_test(
    both_slinky_shared_libs
    PREFIX "${staging_both}"
    OPTIONS -DSLINKY_SHARED_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DSLINKY_TEST_EXPECTED_TYPE=SHARED_LIBRARY
)
slinky_add_package_test(
    both_static
    PREFIX "${staging_both}"
    OPTIONS
    -DSLINKY_TEST_COMPONENT=static
    -DSLINKY_TEST_CONFLICTING_COMPONENT=shared
    -DSLINKY_TEST_EXPECTED_TYPE=STATIC_LIBRARY
)
slinky_add_package_test(
    both_shared
    PREFIX "${staging_both}"
    OPTIONS
    -DSLINKY_TEST_COMPONENT=shared
    -DSLINKY_TEST_CONFLICTING_COMPONENT=static
    -DSLINKY_TEST_EXPECTED_TYPE=SHARED_LIBRARY
)
