if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(sgl_portable_operator_tests LANGUAGES NONE)
    enable_testing()
    option(SGL_BITMASK_HOST_TEST_ONLY "Check the runtime test's data and comparisons without CANN" OFF)
    set(SGL_BITMASK_INSTALLED_VENDOR "" CACHE PATH "Installed sgl_kernel_npu vendor directory for the ACLNN test")
    if(SGL_BITMASK_HOST_TEST_ONLY OR SGL_BITMASK_INSTALLED_VENDOR)
        enable_language(CXX)
        add_executable(test_bitmask_aclnn test_bitmask_aclnn.cpp)
        target_compile_features(test_bitmask_aclnn PRIVATE cxx_std_17)
        set_target_properties(test_bitmask_aclnn PROPERTIES CXX_EXTENSIONS OFF)
        if(SGL_BITMASK_HOST_TEST_ONLY)
            if(SGL_BITMASK_INSTALLED_VENDOR)
                message(FATAL_ERROR "Select either Host-only checks or the installed CANN runtime test")
            endif()
            target_compile_definitions(test_bitmask_aclnn PRIVATE SGL_BITMASK_HOST_TEST_ONLY)
            add_test(NAME bitmask.runtime_reference COMMAND test_bitmask_aclnn)
        else()
            if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
                message(FATAL_ERROR "The installed CANN runtime test requires Linux")
            endif()
            set(SGL_CANN_SDK "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN SDK used by the installed package")
            set(opapi "${SGL_BITMASK_INSTALLED_VENDOR}/op_api")
            set(header "${opapi}/include/aclnn_sgl_apply_token_bit_mask.h")
            if(NOT IS_ABSOLUTE "${SGL_BITMASK_INSTALLED_VENDOR}" OR NOT EXISTS "${header}" OR
               NOT IS_ABSOLUTE "${SGL_CANN_SDK}" OR NOT EXISTS "${SGL_CANN_SDK}/include/acl/acl.h")
                message(FATAL_ERROR "Supply the installed vendor and initialized CANN SDK with their real headers")
            endif()
            find_library(SGL_BITMASK_OPAPI cust_opapi PATHS "${opapi}/lib" NO_DEFAULT_PATH REQUIRED)
            find_library(SGL_ACL_RUNTIME ascendcl PATHS "${SGL_CANN_SDK}/lib64" NO_DEFAULT_PATH REQUIRED)
            find_library(SGL_NNOPBASE nnopbase PATHS "${SGL_CANN_SDK}/lib64" NO_DEFAULT_PATH REQUIRED)
            target_include_directories(test_bitmask_aclnn PRIVATE "${SGL_CANN_SDK}/include")
            target_compile_definitions(test_bitmask_aclnn PRIVATE
                "SGL_BITMASK_ACLNN_HEADER=\"${header}\""
                "SGL_BITMASK_OPAPI_PATH=\"${SGL_BITMASK_OPAPI}\"")
            target_link_libraries(test_bitmask_aclnn PRIVATE
                "${SGL_BITMASK_OPAPI}" "${SGL_ACL_RUNTIME}" "${SGL_NNOPBASE}" ${CMAKE_DL_LIBS})
            message(STATUS "Runtime test uses installed operator library: ${SGL_BITMASK_OPAPI}")
            message(STATUS "Build only; run test_bitmask_aclnn explicitly on a selected NPU")
        endif()
        return()
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME bitmask.kernel_preservation
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_bitmask_migration.py")
    add_test(NAME bitmask.package_verifier
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_cann_package.py")
    add_test(NAME cann.build_orchestration
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_cann_build.py")
    return()
endif()

if(NOT "apply_token_bitmask" IN_LIST selected_operators)
    return()
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/check_bitmask_api.cmake")
sgl_find_bitmask_api_header(SGL_BITMASK_ACLNN_HEADER "${CMAKE_CURRENT_BINARY_DIR}/../autogen")
if(TARGET sgl_cann_kernel_check)
    add_custom_target(sgl_bitmask_kernel_check DEPENDS sgl_cann_kernel_check)
endif()

add_library(sgl_bitmask_api_check OBJECT check_apply_token_bitmask_api.cpp)
target_include_directories(sgl_bitmask_api_check PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include"
)
target_compile_definitions(sgl_bitmask_api_check PRIVATE
    "SGL_BITMASK_ACLNN_HEADER=\"${SGL_BITMASK_ACLNN_HEADER}\""
)
