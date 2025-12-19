# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/zccl/allgather/op_host/allgather.cpp
)

# set the so name
set(OP_PLUGIN_NAME zccl)

# kernel side files without workspace
ascendc_library(zccl_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/zccl/allgather/op_kernel/allgather_kernel.cpp
)

ascendc_include_directories(zccl_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    ${ASCEND_INCLUDE_DIR}
)

# create shared library libzccl.so
add_library({OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        zccl_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
