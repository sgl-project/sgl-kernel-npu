cmake_minimum_required(VERSION 3.16.0)
project(opp)
find_package(ASC REQUIRED)
set(package_name ${vendor_name})

option(DEEPEP_TEST_1C1V "Launch FusedDeepMoE on Ascend950 with one AIV per block" OFF)
set(FUSED_DEEP_MOE_LAUNCH_DEFINITION "")
if(DEEPEP_TEST_1C1V)
    if(NOT "${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend950")
        message(FATAL_ERROR "DEEPEP_TEST_1C1V requires ASCEND_COMPUTE_UNIT=ascend950")
    endif()
    set(FUSED_DEEP_MOE_LAUNCH_DEFINITION DEEPEP_FUSED_MOE_1C1V)
endif()

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
