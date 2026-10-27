include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(SLINKY_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/slinky"
    CACHE STRING "Path to slinky CMake files"
)

# The libraries that make up the package. Excludes test utilities.
set(slinky_package_targets
    slinky_base
    slinky_thread_pool
    slinky_thread_pool_impl
    slinky_chrome_trace
    slinky_runtime
    slinky_visualize
    slinky_builder
)

# Shared libraries find each other next to themselves, regardless of the
# consumer's linker flags or the prefix they were installed to. Without this, a
# consumer linked with --as-needed (the default on Debian and Ubuntu) cannot
# load the transitive dependencies of the libraries it uses directly.
if (APPLE)
    set(origin "@loader_path")
else ()
    set(origin "$ORIGIN")
endif ()
set_target_properties(${slinky_package_targets} PROPERTIES INSTALL_RPATH "${origin}")

install(
    TARGETS ${slinky_package_targets}
    EXPORT slinky-targets
    RUNTIME COMPONENT slinky_Runtime
    LIBRARY COMPONENT slinky_Runtime NAMELINK_COMPONENT slinky_Development
    ARCHIVE COMPONENT slinky_Development
    FILE_SET HEADERS COMPONENT slinky_Development
)

# A single build tree never mixes library types, so each one installs only its
# own targets file. Downstream, find_package (via slinky-config.cmake) picks one.
if (BUILD_SHARED_LIBS)
    set(type shared)
else ()
    set(type static)
endif ()

install(
    EXPORT slinky-targets
    DESTINATION "${SLINKY_INSTALL_CMAKEDIR}"
    NAMESPACE slinky::
    FILE slinky-${type}-targets.cmake
    COMPONENT slinky_Development
)

write_basic_package_version_file(
    slinky-config-version.cmake
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/slinky-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/slinky-config-version.cmake"
    DESTINATION "${SLINKY_INSTALL_CMAKEDIR}"
    COMPONENT slinky_Development
)
