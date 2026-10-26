cmake_minimum_required(VERSION 3.20)
project(sgl_cmake_tests LANGUAGES NONE)
enable_testing()

function(add_operator_case name expected_error)
    add_test(NAME operators.${name}
        COMMAND "${CMAKE_COMMAND}"
            "-DCASE=${name}"
            "-DEXPECTED_ERROR=${expected_error}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/${name}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/operators_case.cmake")
endfunction()

add_operator_case(roundtrip "")
add_operator_case(no_torch_adapter "")
add_operator_case(duplicate_id "Duplicate operator ID: fixture_op")
add_operator_case(duplicate_op_type "Duplicate CANN_OP_TYPE: FixtureOp")
add_operator_case(missing_id "missing required field: ID")
add_operator_case(missing_op_type "missing required field: CANN_OP_TYPE")
add_operator_case(empty_arch_list "Operator fields need values: SUPPORTED_ARCHS")
add_operator_case(unknown_argument "Unknown operator arguments: TYPO_FIELD")
add_operator_case(missing_source "Operator source is not an existing file:")
add_operator_case(directory_source "Operator source is not an existing file:")
add_operator_case(aicpu_selected "Selected EXECUTION_UNIT backend is not implemented:")
add_operator_case(call_mode_rejected "Unknown operator arguments: CALL_MODE")
add_operator_case(unknown_operator "Unknown operator ID: unknown_op")
add_operator_case(unsupported_arch "does not support requested ARCH: 9999")
add_operator_case(missing_selection_arch "Selection is missing required field: ARCH")
add_operator_case(duplicate_selection "Duplicate requested operator: fixture_op")
add_operator_case(all_supported "")
add_operator_case(all_filtered "")
add_operator_case(multiple_selection "")
add_operator_case(all_empty "No CANN operators support requested ARCH: 9999")
add_operator_case(all_mixed "ALL cannot be combined with explicit operator IDs")
add_operator_case(multiple_unsupported "does not support requested ARCH: 2201")

function(add_cann_case name expected_error)
    add_test(NAME cann.${name}
        COMMAND "${CMAKE_COMMAND}"
            "-DCASE=${name}"
            "-DEXPECTED_ERROR=${expected_error}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cann-${name}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cann_case.cmake")
endfunction()

add_cann_case(declarations_only "")
add_cann_case(arch3510_declarations "")
add_cann_case(tiling_declarations "")
add_cann_case(kernel_declarations "")
add_cann_case(invalid_stage "Unknown SGL_CANN_BUILD_STAGE: UNKNOWN")
add_cann_case(missing_arch "SGL_ARCH_VERSION must be explicitly set to 2201 or 3510")
add_cann_case(unsupported_arch "SGL_ARCH_VERSION must be explicitly set to 2201 or 3510")
add_cann_case(multiple_targets "SGL_ARCH_VERSION must be explicitly set to 2201 or 3510")
add_cann_case(unknown_operator "Unknown operator ID: unknown_op")
add_cann_case(header_discovery "")
add_cann_case(header_missing "Expected one generated SglApplyTokenBitMask ACLNN header, found 0")
add_cann_case(header_ambiguous "Expected one generated SglApplyTokenBitMask ACLNN header, found 2")
foreach(arch IN ITEMS 2201 3510)
    add_test(NAME cann.sdk_mapping_${arch}
        COMMAND "${CMAKE_COMMAND}"
            -S "${CMAKE_CURRENT_SOURCE_DIR}/kernel_sources"
            -B "${CMAKE_CURRENT_BINARY_DIR}/cann-sdk-mapping-${arch}"
            "-DSGL_ARCH_VERSION=${arch}")
endforeach()
add_test(NAME cann.sdk_mapping_without_bitmask
    COMMAND "${CMAKE_COMMAND}"
        -S "${CMAKE_CURRENT_SOURCE_DIR}/kernel_sources"
        -B "${CMAKE_CURRENT_BINARY_DIR}/cann-sdk-mapping-without-bitmask"
        -DSGL_ARCH_VERSION=2201 -DTEST_WITHOUT_BITMASK=ON)
