# The root Torch build keeps its existing implementation during migration.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include("${CMAKE_CURRENT_LIST_DIR}/../cmake/legacy_kernels.cmake")
    return()
endif()

cmake_minimum_required(VERSION 3.20)
project(sgl_cann_ops LANGUAGES NONE)

option(SGL_CANN_DECLARATIONS_ONLY "Validate declarations without loading CANN" OFF)
set(SGL_CANN_BUILD_STAGE CODEGEN CACHE STRING "CANN checkpoint: CODEGEN, TILING or KERNEL")
set_property(CACHE SGL_CANN_BUILD_STAGE PROPERTY STRINGS CODEGEN TILING KERNEL)
if(NOT SGL_CANN_BUILD_STAGE MATCHES "^(CODEGEN|TILING|KERNEL)$")
    message(FATAL_ERROR "Unknown SGL_CANN_BUILD_STAGE: ${SGL_CANN_BUILD_STAGE}")
endif()
set(SGL_ARCH_VERSION "" CACHE STRING "Explicit architecture: 2201 or 3510")
set(SGL_CANN_OPERATORS "ALL" CACHE STRING "ALL or a semicolon-separated list of operator IDs")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/architectures.cmake")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/operators.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/domains.cmake")
sgl_add_operator_domains()
sgl_select_operators(selected_operators ARCH "${SGL_ARCH_VERSION}" OPERATORS ${SGL_CANN_OPERATORS})
message(STATUS "Selected CANN operators: ${selected_operators}; ARCH: ${SGL_ARCH_VERSION}; SDK targets: ${SGL_CANN_COMPUTE_UNITS}")
string(REPLACE ";" "\n" selection_text "${selected_operators}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/selected-operators.txt"
    CONTENT "${selection_text}\n" @ONLY)
if(DEFINED SGL_CANN_CONFIGURED_OPERATORS
   AND NOT "${SGL_CANN_CONFIGURED_OPERATORS}" STREQUAL "${selected_operators}")
    message(FATAL_ERROR "CANN operator selection changed; run with --make_clean before rebuilding")
endif()
set(SGL_CANN_CONFIGURED_OPERATORS "${selected_operators}" CACHE INTERNAL "Configured operator selection" FORCE)

if(SGL_CANN_DECLARATIONS_ONLY)
    message(STATUS "Declarations validated only; no SDK code generation or compilation was performed")
    return()
endif()

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CANN code generation requires Linux; use SGL_CANN_DECLARATIONS_ONLY on this host")
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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cann_ops.cmake")
if(SGL_CANN_BUILD_STAGE STREQUAL "KERNEL")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/packages.cmake")
    sgl_initialize_cann_package()
endif()
sgl_generate_cann_interfaces("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${selected_operators})
if(NOT SGL_CANN_BUILD_STAGE STREQUAL "CODEGEN")
    sgl_add_cann_tiling(${selected_operators})
endif()
if(SGL_CANN_BUILD_STAGE STREQUAL "KERNEL")
    sgl_add_cann_kernels("${CMAKE_CURRENT_BINARY_DIR}/autogen" ${selected_operators})
endif()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../tests/csrc" "${CMAKE_CURRENT_BINARY_DIR}/api-check")
message(STATUS "CANN checkpoint: ${SGL_CANN_BUILD_STAGE}; configuration never launches a Kernel or installs a package")
