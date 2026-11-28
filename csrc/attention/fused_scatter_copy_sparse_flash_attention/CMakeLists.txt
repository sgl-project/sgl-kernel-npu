sgl_register_operator(
    ID fused_scatter_copy_sparse_flash_attention_split
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglFusedCopySfa
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/fused_scatter_copy_sparse_flash_attention_split_def.cpp
    INFER_SOURCES op_host/fused_scatter_copy_sparse_flash_attention_split_def.cpp
    TILING_SOURCES op_host/fused_scatter_copy_sparse_flash_attention_split_tiling.cpp ../sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp
    KERNEL_SOURCES op_kernel/sgl_fused_copy_sfa.cpp
    KERNEL_INCLUDE_DIRECTORIES . op_kernel op_kernel/sparse_flash_attention_impl
        ../../sparse_flash_attention/op_kernel ../../sparse_flash_attention/op_kernel/common ../../utils/kernel
)

function(sgl_add_copy_sfa_frontend)
    if(TARGET sgl_kernel_npu AND "fused_scatter_copy_sparse_flash_attention_split" IN_LIST cann_operators)
        target_sources(sgl_kernel_npu PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/op_host/fused_copy_sfa.cpp")
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL sgl_add_copy_sfa_frontend)
