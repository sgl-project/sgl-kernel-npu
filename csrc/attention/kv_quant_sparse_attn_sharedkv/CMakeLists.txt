sgl_register_operator(
    ID kv_quant_sparse_attn_sharedkv
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglKvQuantSparseAttnSharedkv
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/kv_quant_sparse_attn_sharedkv_def.cpp
    INFER_SOURCES op_host/kv_quant_sparse_attn_sharedkv_infer.cpp
    TILING_SOURCES
        op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp
        op_host/kv_quant_sparse_attn_sharedkv_check.cpp
        op_host/kv_quant_sparse_attn_sharedkv_check_consistancy.cpp
        op_host/kv_quant_sparse_attn_sharedkv_check_existance.cpp
        op_host/kv_quant_sparse_attn_sharedkv_check_feature.cpp
        op_host/kv_quant_sparse_attn_sharedkv_check_single_para.cpp
    KERNEL_SOURCES op_kernel/kv_quant_sparse_attn_sharedkv.cpp
    # Operator-specific headers live here; util.h is shared with other kernels.
    KERNEL_INCLUDE_DIRECTORIES
        op_kernel ../../utils/kernel
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm
        -cce-vf-remove-membar=false
        -mllvm
        -cce-aicore-hoist-movemask=false
    TORCH_ADAPTER torch_adapter.cpp
)
