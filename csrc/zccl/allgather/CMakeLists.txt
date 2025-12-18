# this is the cmakelist file for deepep build
# deepep will be built as separated wheel package

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(DEEPEP_ARCH "x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(DEEPEP_ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported host processor: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(PROJECT_BUILD_PATH ${PROJECT_BINARY_DIR})
set(TARGET_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(ASCEND_HOME_PATH ${ASCEND_HOME_PATH})
set(SHMEM_HOME_PATH ${SHMEM_HOME_PATH})

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

set_target_properties(zccl_cpp PROPERTIES CXX_STANDARD 17)
target_include_directories( zccl_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/zccl

    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${TORCH_NPU_DIR}/include/third_party/acl/inc/acl
    ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${ASCEND_HOME_PATH}/include
    ${SHMEM_HOME_PATH}/shmem/include
)
target_link_directories(zccl_cpp PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${ASCEND_HOME_PATH}/lib64
        ${SHMEM_HOME_PATH}/shmem/lib
)
target_link_libraries(zccl_cpp PRIVATE
    torch_python
    ascendcl
    torch_npu
    shmem
)

message(STATUS "TARGET_INSTALL_DIR = ${TARGET_INSTALL_DIR}")
install(TARGETS zccl_cpp
    LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/lib
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)
