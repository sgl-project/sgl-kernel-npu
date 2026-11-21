cmake_minimum_required(VERSION 3.16.0)
project(opp)
find_package(ASC REQUIRED)
option(DEEPEP_FUSEEP_A2 "Build experimental A2 DispatchFFNCombine only" OFF)
if(DEEPEP_FUSEEP_A2)
    if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910b")
        message(FATAL_ERROR "DEEPEP_FUSEEP_A2 requires ASCEND_COMPUTE_UNIT=ascend910b")
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/fuseep_a2.cmake)
    return()
endif()
set(package_name ${vendor_name})

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}/
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
    add_subdirectory(framework)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    add_subdirectory(op_kernel)
endif()
