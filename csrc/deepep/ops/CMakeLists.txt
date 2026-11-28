cmake_minimum_required(VERSION 3.16.0)
project(opp)
find_package(ASC REQUIRED)
set(package_name ${vendor_name})

set(DEEPEP_FUSED_MOE_1C1V OFF)
set(DEEPEP_USE_MX_FUSED_MOE OFF)
if("${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend950")
    set(DEEPEP_USE_MX_FUSED_MOE ON)
elseif("${ASCEND_COMPUTE_UNIT}" MATCHES "^ascend960(dt|pr)$")
    set(DEEPEP_USE_MX_FUSED_MOE ON)
    set(DEEPEP_FUSED_MOE_1C1V ON)
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
