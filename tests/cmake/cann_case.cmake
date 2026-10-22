cmake_minimum_required(VERSION 3.20)

if(NOT TEST_CHILD)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DCASE=${CASE}"
            "-DTEST_BINARY_DIR=${TEST_BINARY_DIR}" -DTEST_CHILD=ON
            -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if("${EXPECTED_ERROR}" STREQUAL "")
        if(NOT "${result}" STREQUAL "0")
            message(FATAL_ERROR "${CASE} failed:\n${output}\n${error}")
        endif()
    else()
        if("${result}" STREQUAL "0")
            message(FATAL_ERROR "${CASE} unexpectedly succeeded")
        endif()
        string(FIND "${output}\n${error}" "${EXPECTED_ERROR}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "${CASE} failed for the wrong reason:\n${output}\n${error}")
        endif()
    endif()
    return()
endif()

get_filename_component(source "${CMAKE_CURRENT_LIST_DIR}/../../csrc" ABSOLUTE)
if(CASE MATCHES "^header_")
    include("${source}/cmake/cann_ops.cmake")
    file(MAKE_DIRECTORY "${TEST_BINARY_DIR}")
    # Synthetic text tests header discovery only, never SDK generation or C++ ABI.
    file(WRITE "${TEST_BINARY_DIR}/aclnn_unrelated.h" "void unrelated();\n")
    if(NOT CASE STREQUAL "header_missing")
        file(WRITE "${TEST_BINARY_DIR}/aclnn_sdk_chosen_name.h"
            "void aclnnSglApplyTokenBitMaskGetWorkspaceSize();\n")
    endif()
    if(CASE STREQUAL "header_ambiguous")
        file(WRITE "${TEST_BINARY_DIR}/aclnn_second.h"
            "void aclnnSglApplyTokenBitMaskGetWorkspaceSize();\n")
    endif()
    sgl_find_bitmask_api_header(header "${TEST_BINARY_DIR}")
    if(NOT header STREQUAL "${TEST_BINARY_DIR}/aclnn_sdk_chosen_name.h")
        message(FATAL_ERROR "Wrong generated header selected: ${header}")
    endif()
    return()
endif()

# These names are CMake metadata fixtures, not NPU models or hardware claims.
set(soc fixture_soc_a)
set(confirmed_socs fixture_soc_a)
set(operators apply_token_bitmask)
set(stage CODEGEN)
if(CASE STREQUAL "missing_soc")
    set(soc "")
elseif(CASE STREQUAL "missing_confirmed_socs")
    set(confirmed_socs "")
elseif(CASE STREQUAL "unsupported_soc")
    set(soc fixture_soc_b)
elseif(CASE STREQUAL "invalid_soc")
    set(soc "not a compute unit")
elseif(CASE STREQUAL "multiple_targets")
    set(soc "fixture_soc_a;fixture_soc_b")
elseif(CASE STREQUAL "unknown_operator")
    set(operators unknown_op)
elseif(CASE STREQUAL "invalid_stage")
    set(stage UNKNOWN)
elseif(CASE STREQUAL "tiling_declarations")
    set(stage TILING)
elseif(CASE STREQUAL "kernel_declarations")
    set(stage KERNEL)
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${TEST_BINARY_DIR}"
        -DSGL_CANN_DECLARATIONS_ONLY=ON
        "-DSGL_CANN_SOC=${soc}"
        "-DSGL_BITMASK_CONFIRMED_SOCS=${confirmed_socs}"
        "-DSGL_CANN_OPERATORS=${operators}"
        "-DSGL_CANN_BUILD_STAGE=${stage}"
        -DCMAKE_DISABLE_FIND_PACKAGE_ASC=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_Torch=TRUE
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "${output}\n${error}")
endif()
if(NOT output MATCHES "Declarations validated only")
    message(FATAL_ERROR "Declaration-only configuration was not confirmed: ${output}")
endif()
file(READ "${TEST_BINARY_DIR}/CMakeCache.txt" cache)
if(cache MATCHES "CMAKE_CXX_COMPILER:")
    message(FATAL_ERROR "Declaration-only mode unexpectedly enabled C++ compilation")
endif()

# Inspect the production declaration, which now points to the CANN entry.
include("${source}/cmake/operators.cmake")
set(SGL_BITMASK_CONFIRMED_SOCS "${confirmed_socs}")
include("${source}/sampling/apply_token_bitmask/CMakeLists.txt")
sgl_get_operator(op_type apply_token_bitmask CANN_OP_TYPE)
sgl_get_operator(kernel apply_token_bitmask KERNEL_SOURCES)
sgl_get_operator(adapter apply_token_bitmask TORCH_ADAPTER)
if(NOT op_type STREQUAL "SglApplyTokenBitMask"
   OR NOT kernel STREQUAL "${source}/sampling/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp"
   OR NOT adapter STREQUAL "")
    message(FATAL_ERROR "Unexpected production declaration: ${op_type}; ${kernel}; ${adapter}")
endif()
