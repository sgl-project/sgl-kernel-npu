sgl_register_operator(
    ID fused_quant_scatter_copy_sparse_flash_attention
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglFusedQuantScatterCopySparseFlashAttention
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/fused_quant_copy_sfa_op.cpp
    INFER_SOURCES op_host/fused_quant_copy_sfa_op.cpp
    TILING_SOURCES op_host/fused_quant_copy_sfa_op.cpp
    KERNEL_SOURCES op_kernel/fused_quant_scatter_copy_sparse_flash_attention.cpp
    KERNEL_INCLUDE_DIRECTORIES
        op_kernel
        op_kernel/arch35
        ../../sparse_flash_attention/op_kernel
        ../../sparse_flash_attention/op_kernel/common
        ../../utils/kernel
    KERNEL_COMPILE_DEFINITIONS HAVE_WORKSPACE HAVE_TILING
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm
        -cce-vf-remove-membar=false
        -mllvm
        -cce-aicore-hoist-movemask=false
)
