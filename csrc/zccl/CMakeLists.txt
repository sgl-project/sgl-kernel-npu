# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/zccl/allgather/op_host/allgather.cpp
    ${PROJECT_OP_SRC_BASE}/zccl/reduce_scatter/op_host/reduce_scatter.cpp
    ${PROJECT_OP_SRC_BASE}/zccl/all_reduce/op_host/all_reduce.cpp
)

# set the so name
set(OP_PLUGIN_NAME zccl)

# kernel side files without workspace
ascendc_library(zccl_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/zccl/allgather/op_kernel/allgather_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/zccl/allgather/op_kernel/allgather_zero_buff_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/zccl/reduce_scatter/op_kernel/reduce_scatter_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/zccl/reduce_scatter/op_kernel/reduce_scatter_zero_buff_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/zccl/all_reduce/op_kernel/all_reduce_kernel.cpp
)

ascendc_include_directories(zccl_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/aclnn
    ${ASCEND_INCLUDE_DIR}/aclnn/opdev
    ${PROJECT_OP_SRC_BASE}/zccl/include
)

# create shared library libzccl.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        zccl_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
        shmem
        mf_smem
        mf_hybm_core
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${SHMEM_HOME_PATH}/shmem/lib
        ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
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
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
)
