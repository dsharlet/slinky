include(CMakeFindDependencyMacro)

# Static libraries that use std::thread need -pthread, not just -lpthread, on
# some platforms, but respect an explicit choice by the consumer.
if (NOT DEFINED THREADS_PREFER_PTHREAD_FLAG)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
endif ()
find_dependency(Threads)

# The static and shared libraries are exported to separate targets files so
# that both can be installed to one prefix. Choose between them as follows:
#
#   1. The `static` or `shared` component, if requested (binding).
#   2. The SLINKY_SHARED_LIBS variable, if defined (binding).
#   3. The BUILD_SHARED_LIBS variable, if defined (preference).
#   4. Whichever is installed, preferring static.
#
# Binding choices fail the package if that type is not installed or if the other
# type was already loaded. Preferences fall back to the other type, and defer to
# whichever type was already loaded. Imported targets cannot be shadowed, so a
# second find_package() cannot be allowed to silently reuse the wrong type.

macro (_slinky_load_failure message)
    set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${message}")
    set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
    return()
endmacro ()

set(slinky_components "${${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS}")

foreach (component IN LISTS slinky_components)
    if (NOT component MATCHES "^(static|shared)$")
        _slinky_load_failure("slinky does not recognize component `${component}`.")
    endif ()
endforeach ()

if ("static" IN_LIST slinky_components AND "shared" IN_LIST slinky_components)
    _slinky_load_failure("slinky `static` and `shared` components are mutually exclusive.")
endif ()

set(slinky_static_targets "${CMAKE_CURRENT_LIST_DIR}/slinky-static-targets.cmake")
set(slinky_shared_targets "${CMAKE_CURRENT_LIST_DIR}/slinky-shared-targets.cmake")

macro (_slinky_load_targets type)
    if (NOT EXISTS "${slinky_${type}_targets}")
        _slinky_load_failure("slinky `${type}` libraries were requested but not found.")
    endif ()
    if (TARGET slinky::base)
        get_target_property(slinky_loaded_type slinky::base TYPE)
        if ("${type}" STREQUAL "static")
            set(slinky_requested_type STATIC_LIBRARY)
        else ()
            set(slinky_requested_type SHARED_LIBRARY)
        endif ()
        if (NOT slinky_loaded_type STREQUAL slinky_requested_type)
            _slinky_load_failure(
                "slinky `${type}` libraries were requested, but slinky::base was already loaded as a ${slinky_loaded_type} target."
            )
        endif ()
    endif ()
    include("${slinky_${type}_targets}")
endmacro ()

if ("static" IN_LIST slinky_components)
    _slinky_load_targets(static)
elseif ("shared" IN_LIST slinky_components)
    _slinky_load_targets(shared)
elseif (DEFINED SLINKY_SHARED_LIBS)
    if (SLINKY_SHARED_LIBS)
        _slinky_load_targets(shared)
    else ()
        _slinky_load_targets(static)
    endif ()
elseif (TARGET slinky::base)
    get_target_property(slinky_loaded_type slinky::base TYPE)
    if (slinky_loaded_type STREQUAL "SHARED_LIBRARY")
        _slinky_load_targets(shared)
    else ()
        _slinky_load_targets(static)
    endif ()
elseif (BUILD_SHARED_LIBS AND EXISTS "${slinky_shared_targets}")
    _slinky_load_targets(shared)
elseif (EXISTS "${slinky_static_targets}")
    _slinky_load_targets(static)
else ()
    _slinky_load_targets(shared)
endif ()
