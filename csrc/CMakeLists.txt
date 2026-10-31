# Route the parent Torch project to the existing build during migration.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include("${CMAKE_CURRENT_LIST_DIR}/../cmake/legacy_kernels.cmake")
    return()
endif()

# Validate build inputs and expand ARCH into the SDK compute-unit families.
cmake_minimum_required(VERSION 3.20)
project(sgl_cann_ops LANGUAGES NONE)

set(SGL_CANN_BUILD_STAGE CODEGEN CACHE STRING "CANN checkpoint: CODEGEN, TILING or KERNEL")
set_property(CACHE SGL_CANN_BUILD_STAGE PROPERTY STRINGS CODEGEN TILING KERNEL)
if(NOT SGL_CANN_BUILD_STAGE MATCHES "^(CODEGEN|TILING|KERNEL)$")
    message(FATAL_ERROR "Unknown SGL_CANN_BUILD_STAGE: ${SGL_CANN_BUILD_STAGE}")
endif()
set(SGL_ARCH_VERSION "" CACHE STRING "Explicit architecture: 2201 or 3510")
set(SGL_CANN_OPERATORS "ALL" CACHE STRING "ALL or a semicolon-separated list of operator IDs")
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

# Load shared declarations, discover domains, and select supported operators.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/operators.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/domains.cmake")
sgl_add_operator_domains()
sgl_select_operators(selected_operators ARCH "${SGL_ARCH_VERSION}" OPERATORS ${SGL_CANN_OPERATORS})
sgl_resolve_operators(ARCH "${SGL_ARCH_VERSION}" OPERATORS ${selected_operators})
# Record selection and reject incompatible reuse of this build directory.
message(STATUS "Selected CANN operators: ${selected_operators}; ARCH: ${SGL_ARCH_VERSION}; SDK targets: ${SGL_CANN_COMPUTE_UNITS}")
string(REPLACE ";" "\n" selection_text "${selected_operators}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/selected-operators.txt"
    CONTENT "${selection_text}\n" @ONLY)
if(DEFINED SGL_CANN_CONFIGURED_OPERATORS
   AND NOT "${SGL_CANN_CONFIGURED_OPERATORS}" STREQUAL "${selected_operators}")
    message(FATAL_ERROR "CANN operator selection changed; run with --make_clean before rebuilding")
endif()
set(SGL_CANN_CONFIGURED_OPERATORS "${selected_operators}" CACHE INTERNAL "Configured operator selection" FORCE)

# Check the Host environment before enabling C++ or loading the CANN SDK.
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CANN code generation requires Linux")
endif()
if(NOT IS_DIRECTORY "$ENV{ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH must point to the initialized CANN SDK")
endif()

# Initialize the C++ toolchain and load the selected SDK's CMake functions.
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

# Define the Run package for a full Kernel build, then dispatch each checkpoint.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cann_ops.cmake")
if(SGL_CANN_BUILD_STAGE STREQUAL "KERNEL")
    npu_op_package(sgl_kernel_npu TYPE RUN
        CONFIG INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
endif()
sgl_generate_cann_interfaces("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${selected_operators})
if(NOT SGL_CANN_BUILD_STAGE STREQUAL "CODEGEN")
    sgl_add_cann_tiling(${selected_operators})
endif()
if(SGL_CANN_BUILD_STAGE STREQUAL "KERNEL")
    sgl_add_cann_kernels("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${selected_operators})
endif()
message(STATUS "CANN checkpoint: ${SGL_CANN_BUILD_STAGE}; configuration never launches a Kernel or installs a package")
