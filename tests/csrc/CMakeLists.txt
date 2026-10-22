if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(sgl_portable_operator_tests LANGUAGES NONE)
    enable_testing()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME bitmask.kernel_preservation
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_bitmask_migration.py")
    add_test(NAME bitmask.package_verifier
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_cann_package.py")
    return()
endif()

add_library(sgl_bitmask_api_check OBJECT check_apply_token_bitmask_api.cpp)
target_include_directories(sgl_bitmask_api_check PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include"
)
target_compile_definitions(sgl_bitmask_api_check PRIVATE
    "SGL_BITMASK_ACLNN_HEADER=\"${SGL_BITMASK_ACLNN_HEADER}\""
)
