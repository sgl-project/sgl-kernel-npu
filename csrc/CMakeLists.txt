# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/pytorch_extensions.cpp
    ${PROJECT_OP_SRC_BASE}/helloworld/op_host/helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_host/cache_loc_assign.cpp
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_host/alloc_extend_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_host/assign_cache.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_host/build_tree.cpp
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_host/mla_preprocess.cpp
    ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_host/batch_matmul_transpose.cpp
    ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_host/tiling/tiling_data.cpp
    ${PROJECT_OP_SRC_BASE}/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/bgmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/bgmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/lightning_indexer.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/tiling/lightning_indexer_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/tri_inv/op_host/tri_inv.cpp
    )
if(BUILD_CATLASS_MODULE)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_basic.cpp
    )
endif()

# set the so name
set(OP_PLUGIN_NAME sgl_kernel_npu)

# kernel side files without workspace
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/helloworld/op_kernel/kernel_helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_kernel/assign_cache_op.cpp
    ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/bgmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/bgmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/tri_inv/op_kernel/tri_inv_kernel.cpp
)

# kernel side files with workspace
set(WORKSPACE_KERNEL_SRCS
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_kernel/alloc_extend_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_kernel/build_tree_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_kernel/lightning_indexer_kernel.cpp
)
if(BUILD_CATLASS_MODULE)
    list(APPEND WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_matmul_basic_kernel.cpp
    )
endif()
ascendc_library(workspace_kernel STATIC ${WORKSPACE_KERNEL_SRCS})
if(BUILD_CATLASS_MODULE)
    ascendc_include_directories(workspace_kernel PRIVATE
        ${CATLASS_DIR}/include
    )
endif()

ascendc_compile_definitions(workspace_kernel PRIVATE
       -DHAVE_WORKSPACE
       -DHAVE_TILING
)

# create shared library libsgl_kernel_npu.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        workspace_kernel
        no_workspace_kernel
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
