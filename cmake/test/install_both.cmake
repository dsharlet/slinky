# Installs the already-built library type from this build tree, then configures,
# builds, and installs the other type from the same sources, into one prefix.
# This is how package managers produce a co-installed static and shared package.
#
# Run with -P and these variables: SOURCE_DIR, BINARY_DIR (this tree's slinky
# packaging binary dir), OTHER_BINARY_DIR, PREFIX, OTHER_SHARED_LIBS, GENERATOR,
# CXX_COMPILER. CONFIG is the build configuration, and is empty for
# single-configuration generators configured without a CMAKE_BUILD_TYPE.
foreach (variable IN ITEMS SOURCE_DIR BINARY_DIR OTHER_BINARY_DIR PREFIX OTHER_SHARED_LIBS GENERATOR CXX_COMPILER)
    if (NOT DEFINED ${variable})
        message(FATAL_ERROR "install_both.cmake requires ${variable}")
    endif ()
endforeach ()

function (run)
    execute_process(COMMAND ${ARGN} COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY)
endfunction ()

set(config_args)
if (CONFIG)
    set(config_args --config "${CONFIG}")
endif ()

file(REMOVE_RECURSE "${PREFIX}")
run("${CMAKE_COMMAND}" "-DCMAKE_INSTALL_PREFIX=${PREFIX}" "-DCMAKE_INSTALL_CONFIG_NAME=${CONFIG}" -P
    "${BINARY_DIR}/cmake_install.cmake")

run("${CMAKE_COMMAND}" -G "${GENERATOR}" -S "${SOURCE_DIR}" -B "${OTHER_BINARY_DIR}" -DSLINKY_ENABLE_TESTS=OFF
    -DSLINKY_INSTALL_RULES=ON "-DBUILD_SHARED_LIBS=${OTHER_SHARED_LIBS}" "-DCMAKE_BUILD_TYPE=${CONFIG}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
run("${CMAKE_COMMAND}" --build "${OTHER_BINARY_DIR}" ${config_args})
run("${CMAKE_COMMAND}" --install "${OTHER_BINARY_DIR}" ${config_args} --prefix "${PREFIX}")
