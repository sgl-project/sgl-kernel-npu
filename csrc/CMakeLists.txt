# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/lib)

# set the so name
set(OP_PLUGIN_NAME sgl_kernel_npu)

# --- SOC_VERSION ARCH mapping ------
string(TOLOWER "${SOC_VERSION}" _sgl_soc_lower)
if(NOT DEFINED SGL_KERNEL_ENABLE_A3_ONLY_OPS)
    if(_sgl_soc_lower MATCHES "^ascend950")
        set(SGL_KERNEL_ENABLE_A3_ONLY_OPS OFF)
    else()
        set(SGL_KERNEL_ENABLE_A3_ONLY_OPS ON)
    endif()
endif()
if(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
    add_compile_definitions(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
endif()
message(STATUS "SGL_KERNEL_ENABLE_A3_ONLY_OPS=${SGL_KERNEL_ENABLE_A3_ONLY_OPS} (SOC=${SOC_VERSION})")

# host side files
set(OP_SRCS
    ${PROJECT_OP_SRC_BASE}/pytorch_extensions.cpp
    ${PROJECT_OP_SRC_BASE}/helloworld/op_host/helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_host/cache_loc_assign.cpp
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_host/alloc_extend_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_host/assign_cache.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_host/build_tree.cpp
    ${PROJECT_OP_SRC_BASE}/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp
    ${PROJECT_OP_SRC_BASE}/transfer_state_dim_exchange/op_host/transfer_state_dim_exchange.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/bgmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/bgmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmv_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmv_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmc_expand.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/sgemmc_shrink.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_host/tiling/sgemmc_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/compressor/op_host/compressor.cpp
    ${PROJECT_OP_SRC_BASE}/compressor/op_host/compressor_tiling.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/causal_conv1d_update.cpp
    ${PROJECT_OP_SRC_BASE}/apply_token_bitmask/op_host/apply_token_bitmask.cpp
    ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata.cpp
    ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_host/sparse_attention_score.cpp
    ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_host/tiling/sparse_attention_score_tiling.cpp
    )
# A3-only host sources
if(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_host/mla_preprocess.cpp
        ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_host/batch_matmul_transpose.cpp
        ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_host/tiling/tiling_data.cpp
        ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/lightning_indexer.cpp
        ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_host/tiling/lightning_indexer_tiling.cpp
        ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv.cpp
        ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp
        ${PROJECT_OP_SRC_BASE}/tri_inv/op_host/tri_inv.cpp
        ${PROJECT_OP_SRC_BASE}/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule.cpp
        ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_host/causal_conv1d.cpp
        ${PROJECT_OP_SRC_BASE}/mega_chunk_gdn/op_host/mega_chunk_gdn.cpp
    )
endif()
if(BUILD_CATLASS_MODULE)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_basic.cpp
        ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_op_utils.cpp
    )
    if(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
        list(APPEND OP_SRCS
            ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_gmm_fp8.cpp
            ${PROJECT_OP_SRC_BASE}/catlass/op_host/catlass_matmul_fp8.cpp
        )
    endif()
endif()

# kernel side files without workspace
set(NO_WORKSPACE_KERNEL_SRCS
    ${PROJECT_OP_SRC_BASE}/helloworld/op_kernel/kernel_helloworld.cpp
    ${PROJECT_OP_SRC_BASE}/cache_location_assign/op_kernel/cache_loc_assign_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/assign_cache_op/op_kernel/assign_cache_op.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/bgmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/bgmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmv_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmv_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp
)
# A3-only no-workspace kernels
if(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
    list(APPEND NO_WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
        ${PROJECT_OP_SRC_BASE}/tri_inv/op_kernel/tri_inv_kernel.cpp
        ${PROJECT_OP_SRC_BASE}/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_kernel.cpp
    )
endif()
# catlass fp8 no-workspace kernels: double-gated
if(BUILD_CATLASS_MODULE AND SGL_KERNEL_ENABLE_A3_ONLY_OPS)
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
    ${PROJECT_OP_SRC_BASE}/alloc_extend/op_kernel/alloc_extend_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/build_tree/op_kernel/build_tree_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/compressor/op_kernel/compressor.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel/causal_conv1d_update.cpp
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel/causal_conv1d.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmc_expand_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/lora/op_kernel/sgemmc_shrink_kernel.cpp
    ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_kernel/sparse_attention_score.cpp
)
# A3-only workspace kernels (causal_conv1d kernel is intentionally always-built)
if(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
    list(APPEND WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
        ${PROJECT_OP_SRC_BASE}/lightning_indexer/op_kernel/lightning_indexer_kernel.cpp
    )
endif()

ascendc_library(workspace_kernel STATIC ${WORKSPACE_KERNEL_SRCS})
ascendc_include_directories(workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
    ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_kernel
    ${PROJECT_OP_SRC_BASE}/compressor/op_kernel
    ${PROJECT_OP_SRC_BASE}/utils/kernel
    ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_kernel
)

ascendc_compile_definitions(workspace_kernel PRIVATE
    -DHAVE_WORKSPACE
    -DHAVE_TILING
)

# Build catlass kernel in its own target (no sparse_attention_score include dir),
# so catlass's #include "tla/layout.hpp" resolves to its own copy and avoids
# symbol redefinition (RoundDown/CeilDiv/Max/Min/...) with sparse_attention_score.
set(CATLASS_WORKSPACE_KERNEL_TARGETS)
if(BUILD_CATLASS_MODULE)
    set(CATLASS_WORKSPACE_KERNEL_SRCS
        ${PROJECT_OP_SRC_BASE}/catlass/op_kernel/catlass_matmul_basic_kernel.cpp
    )
    ascendc_library(catlass_workspace_kernel STATIC ${CATLASS_WORKSPACE_KERNEL_SRCS})
    ascendc_include_directories(catlass_workspace_kernel PRIVATE
        ${CATLASS_DIR}/include
        ${PROJECT_OP_SRC_BASE}/utils/kernel
    )
    ascendc_compile_definitions(catlass_workspace_kernel PRIVATE
        -DHAVE_WORKSPACE
        -DHAVE_TILING
    )
    set(CATLASS_WORKSPACE_KERNEL_TARGETS catlass_workspace_kernel)
endif()

# Dedicated A3-only kernel targets. Keep these separate from workspace_kernel:
# their compiler definitions/options apply to the whole AscendC target.
set(A3_ONLY_KERNEL_TARGETS)
if(SGL_KERNEL_ENABLE_A3_ONLY_OPS)
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
    list(APPEND A3_ONLY_KERNEL_TARGETS mega_chunk_gdn_kernel)

    ascendc_library(sparse_attn_sharedkv_kernel STATIC
        ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp
    )
    ascendc_include_directories(sparse_attn_sharedkv_kernel PRIVATE
        ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv/op_kernel
        ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv/op_host
        ${PROJECT_OP_SRC_BASE}/utils/kernel
    )
    ascendc_compile_definitions(sparse_attn_sharedkv_kernel PRIVATE
        -DHAVE_WORKSPACE
        -DHAVE_TILING
    )
    ascendc_compile_options(sparse_attn_sharedkv_kernel PRIVATE
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm
        -cce-vf-remove-membar=false
    )
    list(APPEND A3_ONLY_KERNEL_TARGETS sparse_attn_sharedkv_kernel)
endif()

# create shared library
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        ${A3_ONLY_KERNEL_TARGETS}
        workspace_kernel
        no_workspace_kernel
        ${CATLASS_WORKSPACE_KERNEL_TARGETS}
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

# Enable OpenMP for the pure-host metadata scheduler (parallel per-row cache build).
target_compile_options(${OP_PLUGIN_NAME} PRIVATE -fopenmp)
target_link_libraries(${OP_PLUGIN_NAME} PRIVATE gomp)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_kernel
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/tiling
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host
        ${PROJECT_OP_SRC_BASE}/causal_conv1d_update/op_host/stub
        ${PROJECT_OP_SRC_BASE}/causal_conv1d/op_host
        ${PROJECT_OP_SRC_BASE}/sparse_attn_sharedkv/op_host
        ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_host
        ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_host/tiling
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
