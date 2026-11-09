# The root Torch build keeps its existing implementation during migration.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include("${CMAKE_CURRENT_LIST_DIR}/../cmake/legacy_kernels.cmake")
    return()
endif()

cmake_minimum_required(VERSION 3.20)
project(sgl_cann_ops LANGUAGES NONE)

# Validate the explicit build stage and architecture.
set(SGL_CANN_BUILD_STAGE CODEGEN CACHE STRING "CANN checkpoint: CODEGEN, TILING or KERNEL")
set_property(CACHE SGL_CANN_BUILD_STAGE PROPERTY STRINGS CODEGEN TILING KERNEL)
if(NOT SGL_CANN_BUILD_STAGE MATCHES "^(CODEGEN|TILING|KERNEL)$")
    message(FATAL_ERROR "Unknown SGL_CANN_BUILD_STAGE: ${SGL_CANN_BUILD_STAGE}")
endif()
set(SGL_ARCH_VERSION "" CACHE STRING "Explicit architecture: 2201 or 3510")
set(SGL_CANN_OPERATORS "ALL" CACHE STRING "ALL or a semicolon-separated list of operator IDs")
set(SGL_NATIVE_OPERATORS "" CACHE STRING "Optional unified CLI request; empty retains the CANN-only entry")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/architectures.json" SGL_ARCHITECTURES)
if(NOT SGL_ARCH_VERSION MATCHES "^(2201|3510)$")
    message(FATAL_ERROR "SGL_ARCH_VERSION must be explicitly set to 2201 or 3510")
endif()
string(JSON count LENGTH "${SGL_ARCHITECTURES}" "${SGL_ARCH_VERSION}" compute_units)
math(EXPR last "${count} - 1")
set(SGL_CANN_COMPUTE_UNITS)
foreach(index RANGE ${last})
    string(JSON unit GET "${SGL_ARCHITECTURES}" "${SGL_ARCH_VERSION}" compute_units ${index})
    list(APPEND SGL_CANN_COMPUTE_UNITS "${unit}")
endforeach()

# Discover declarations, then validate only the selected implementations.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/operators.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/domains.cmake")
sgl_add_host_operator_domains()
set(host_operators)
if(SGL_NATIVE_OPERATORS)
    # Keep the original unified request intact. Only the CANN request reaches
    # legacy declaration-local dependency expansion; ALL retains CANN's ALL scope.
    set(cann_operators)
    sgl_get_registered_operators(registered_host_operators EXECUTION_UNIT HOST)
    foreach(id IN LISTS SGL_NATIVE_OPERATORS)
        if(id IN_LIST registered_host_operators)
            list(APPEND host_operators "${id}")
        else()
            list(APPEND cann_operators "${id}")
        endif()
    endforeach()
    set(SGL_CANN_OPERATORS "${cann_operators}" CACHE STRING
        "ALL or a semicolon-separated list of CANN operator IDs" FORCE)
endif()
sgl_add_cann_operator_domains()
if(SGL_NATIVE_OPERATORS)
    # Read the expanded CANN request from the legacy variable, then replace
    # both local lists with the final selection. HOST IDs stay separate.
    sgl_select_native_operators(cann_operators host_operators
        ARCH "${SGL_ARCH_VERSION}"
        OPERATORS ${SGL_CANN_OPERATORS} ${host_operators})
else()
    sgl_select_cann_operators(cann_operators ARCH "${SGL_ARCH_VERSION}" OPERATORS ${SGL_CANN_OPERATORS})
endif()
string(REPLACE ";" "\n" host_selection_text "${host_operators}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/selected-host-operators.txt"
    CONTENT "${host_selection_text}\n" @ONLY)
message(STATUS "Selected HOST operators: ${host_operators}")
set(aicore_operators)
set(aicpu_operators)
foreach(id IN LISTS cann_operators)
    sgl_get_operator_field(unit "${id}" EXECUTION_UNIT)
    if(unit STREQUAL "AICORE")
        list(APPEND aicore_operators "${id}")
    else()
        list(APPEND aicpu_operators "${id}")
    endif()
endforeach()
# Hand the selected AICPU IDs to the existing Python package coordinator.
set(SGL_CANN_AICPU_OPERATORS "${aicpu_operators}" CACHE INTERNAL
    "AICPU operators selected for this CANN build" FORCE)
message(STATUS "Selected CANN operators: ${cann_operators}; ARCH: ${SGL_ARCH_VERSION}; SDK targets: ${SGL_CANN_COMPUTE_UNITS}")
string(REPLACE ";" "\n" cann_selection_text "${cann_operators}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/selected-operators.txt"
    CONTENT "${cann_selection_text}\n" @ONLY)
if(DEFINED SGL_CANN_CONFIGURED_OPERATORS
   AND NOT "${SGL_CANN_CONFIGURED_OPERATORS}" STREQUAL "${cann_operators}")
    message(FATAL_ERROR "CANN operator selection changed; run with --make-clean before rebuilding")
endif()
set(SGL_CANN_CONFIGURED_OPERATORS "${cann_operators}" CACHE INTERNAL "Configured operator selection" FORCE)

# Load the initialized CANN SDK after declaration validation.
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CANN code generation requires Linux")
endif()
if(NOT IS_DIRECTORY "$ENV{ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH must point to the initialized CANN SDK")
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
set(SGL_CANN_SDK_ROOT "${ASCEND_CANN_PACKAGE_PATH}" CACHE INTERNAL "Configured CANN SDK root" FORCE)
set(ASCEND_COMPUTE_UNIT "${SGL_CANN_COMPUTE_UNITS}")
set(vendor_name sgl_kernel_npu)
set(ENABLE_BINARY_PACKAGE ON)
set(ENABLE_SOURCE_PACKAGE ON)
set(ASCEND_SKIP_FAILED_COMPUTE_UNIT OFF)
find_package(ASC REQUIRED CONFIG PATHS
    "${ASCEND_CANN_PACKAGE_PATH}"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/lib64/cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    NO_DEFAULT_PATH
)

# Build interfaces, Tiling and Kernel/package targets in dependency order.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cann_ops.cmake")
if(SGL_CANN_BUILD_STAGE STREQUAL "KERNEL")
    npu_op_package(sgl_kernel_npu TYPE RUN
        CONFIG INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
endif()
sgl_generate_cann_interfaces("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${cann_operators})
if(NOT SGL_CANN_BUILD_STAGE STREQUAL "CODEGEN" AND aicore_operators)
    sgl_add_cann_tiling(${aicore_operators})
endif()
if(SGL_CANN_BUILD_STAGE STREQUAL "KERNEL")
    sgl_add_cann_host_libraries("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${cann_operators})
    set(native_build_targets)
    if(aicore_operators)
        sgl_add_cann_kernels("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${aicore_operators})
    endif()
    if(aicpu_operators)
        # A directory scope keeps the HCC compiler local to AICPU targets.
        add_subdirectory(cmake/aicpu aicpu)
        list(APPEND native_build_targets sgl_cann_aicpu sgl_cann_aicpu_metadata)
    endif()

    if(host_operators)
        add_subdirectory(cmake/host host)
        list(APPEND native_build_targets sgl_host_operators)
    endif()

    # Package and check the targets created by the Host and device builders.
    set(cann_host_libraries)
    foreach(target IN ITEMS sgl_cann_opapi sgl_cann_proto)
        if(TARGET ${target})
            list(APPEND cann_host_libraries ${target})
        endif()
    endforeach()
    set(SGL_CANN_HOST_LIBRARIES "${cann_host_libraries}" CACHE INTERNAL
        "Host libraries expected in this CANN package" FORCE)
    set(cann_package_libraries ${cann_host_libraries})
    if(aicore_operators)
        list(APPEND cann_package_libraries sgl_cann_tiling sgl_cann_kernels)
    endif()
    if(cann_package_libraries)
        npu_op_package_add(sgl_kernel_npu LIBRARY ${cann_package_libraries})
    endif()
    # CANN creates binary when the Kernel library is added to the package.
    if(aicore_operators)
        if(NOT TARGET binary)
            message(FATAL_ERROR "The SDK did not create the required binary build target")
        endif()
        list(APPEND native_build_targets binary)
    endif()
    add_custom_target(sgl_cann_kernel_check DEPENDS ${native_build_targets} ${cann_host_libraries})
endif()
message(STATUS "CANN checkpoint: ${SGL_CANN_BUILD_STAGE}; configuration never launches a Kernel or installs a package")
