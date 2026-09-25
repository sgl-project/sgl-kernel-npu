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
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmc_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmc_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/tiling/sgemmc_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/lightning_indexer.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/tiling/lightning_indexer_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/tri_inv/op_host/tri_inv.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/causal_conv1d_update.cpp
    ${PROJECT_OP_SRC_BASE}/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_host/causal_conv1d.cpp
    ${PROJECT_OP_SRC_BASE}/apply_token_bitmask/op_host/apply_token_bitmask.cpp
    )
if(BUILD_CATLASS_MODULE)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_basic.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_op_utils.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_gmm_fp8.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_fp8.cpp
    )
endif()
list(APPEND OP_SRCS
    ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_host/mega_chunk_gdn.cpp
)

# set the so name
set(OP_PLUGIN_NAME sgl_kernel_npu)

# kernel side files without workspace
set(NO_WORKSPACE_KERNEL_SRCS
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
    ${PROJECT_OP_SRC_BASE}/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp
)

if(BUILD_CATLASS_MODULE)
    list(APPEND NO_WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_gmm_fp8_kernel.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_matmul_fp8_kernel.cpp
    )
endif()
ascendc_library(no_workspace_kernel STATIC ${NO_WORKSPACE_KERNEL_SRCS})
ascendc_include_directories(no_workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils/kernel
)
if(BUILD_CATLASS_MODULE)
    ascendc_include_directories(no_workspace_kernel PRIVATE
        ${CATLASS_DIR}/include
    )
endif()

# kernel side files with workspace
set(WORKSPACE_KERNEL_SRCS
    ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_kernel/alloc_extend_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_kernel/build_tree_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_kernel/lightning_indexer_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel/causal_conv1d_update.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel/causal_conv1d.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmc_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmc_shrink_kernel.cpp
)
if(BUILD_CATLASS_MODULE)
    list(APPEND WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_matmul_basic_kernel.cpp
    )
endif()
ascendc_library(workspace_kernel STATIC ${WORKSPACE_KERNEL_SRCS})
ascendc_include_directories(workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel
)
if(BUILD_CATLASS_MODULE)
    ascendc_include_directories(workspace_kernel PRIVATE
        ${CATLASS_DIR}/include
    )
endif()

ascendc_include_directories(workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils/kernel
)

ascendc_compile_definitions(workspace_kernel PRIVATE
       -DHAVE_WORKSPACE
       -DHAVE_TILING
)

# These kernels use the A2/A3 implementation. Do not compile or register them
# for A5 (or other SoCs), whose kernel APIs and scheduling differ.
set(ATTENTION_KERNEL_TARGETS)
if(SOC_VERSION MATCHES "^Ascend910(B|_93)")
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/laser_attention/op_host/laser_attention.cpp
        ${PROJECT_OP_SRC_BASE}/ada_block_sparse_attention/op_host/ada_block_sparse_attention.cpp
        ${PROJECT_OP_SRC_BASE}/ada_block_sparse_attention/op_host/ada_block_sparse_attention_tiling.cpp
        ${PROJECT_OP_SRC_BASE}/sparse_block_estimate/op_host/sparse_block_estimate_tiling.cpp
    )
    ascendc_library(laser_attention_kernel STATIC
        ${PROJECT_OP_SRC_BASE}/laser_attention/op_kernel/laser_attention.cpp
    )
    ascendc_include_directories(laser_attention_kernel PRIVATE
        ${PROJECT_OP_SRC_BASE}/laser_attention/op_kernel
        ${PROJECT_OP_SRC_BASE}/laser_attention/op_host/tiling
    )
    ascendc_compile_definitions(laser_attention_kernel PRIVATE
        -DHAVE_WORKSPACE
        -DHAVE_TILING
    )
    ascendc_compile_options(laser_attention_kernel PRIVATE
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
    )

    foreach(OP ada_block_sparse_attention sparse_block_estimate)
        ascendc_library(${OP}_kernel STATIC
            ${PROJECT_OP_SRC_BASE}/${OP}/op_kernel/${OP}.cpp
        )
        ascendc_include_directories(${OP}_kernel PRIVATE
            ${PROJECT_OP_SRC_BASE}/${OP}/op_kernel
        )
        ascendc_compile_definitions(${OP}_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)
        ascendc_compile_options(${OP}_kernel PRIVATE
            --cce-auto-sync=off -Wno-deprecated-declarations -Werror -fpermissive
        )
        list(APPEND ATTENTION_KERNEL_TARGETS ${OP}_kernel)
    endforeach()
    list(APPEND ATTENTION_KERNEL_TARGETS laser_attention_kernel)
else()
    message(STATUS "Skipping laser_attention, ada_block_sparse_attention and sparse_block_estimate for ${SOC_VERSION}")
endif()

set(MEGA_CHUNK_GDN_KERNEL_TARGETS)
set(MEGA_CHUNK_GDN_KERNEL_TARGETS mega_chunk_gdn_kernel)
ascendc_library(mega_chunk_gdn_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_kernel/mega_kernel.cpp
)
ascendc_include_directories(mega_chunk_gdn_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_kernel
    ${PROJECT_SOURCE_DIR}/third_party/pto-isa/include
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/experiment/runtime
    ${ASCEND_INCLUDE_DIR}/experiment/msprof
)
ascendc_compile_definitions(mega_chunk_gdn_kernel PRIVATE
    -DGDN_KERNEL_NAME=launch_mega_kernel
    -DGDN_D=128
    -DGDN_C=128
)

# create shared library libsgl_kernel_npu.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})
if(ATTENTION_KERNEL_TARGETS)
    target_compile_definitions(${OP_PLUGIN_NAME} PRIVATE SGL_KERNEL_NPU_HAS_A2_A3_ATTENTION)
endif()

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        ${MEGA_CHUNK_GDN_KERNEL_TARGETS}
        ${ATTENTION_KERNEL_TARGETS}
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
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/tiling
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/stub
        ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_host
        ${PROJECT_OP_SRC_BASE}/laser_attention/op_host/tiling
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
