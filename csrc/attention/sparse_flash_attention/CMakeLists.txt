# Sparse flash attention migrated to the independent CANN framework.
# The legacy sources under csrc/sparse_flash_attention remain unchanged.
sgl_register_operator(
    ID sparse_flash_attention
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseFlashAttention
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/sparse_flash_attention_def.cpp
    INFER_SOURCES op_host/sparse_flash_attention_infer.cpp
    TILING_SOURCES op_host/sparse_flash_attention_tiling.cpp
    KERNEL_SOURCES op_kernel/sparse_flash_attention.cpp
    KERNEL_INCLUDE_DIRECTORIES op_kernel/common ../../utils/kernel
    KERNEL_COMPILE_DEFINITIONS ASCENDC_DUMP=0
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm
        -cce-vf-remove-membar=false
        -mllvm
        -cce-aicore-hoist-movemask=false
)
