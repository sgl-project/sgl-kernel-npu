# Sparse attention with shared KV migrated to the independent CANN framework.
# The legacy sources under csrc/sparse_attn_sharedkv remain unchanged.
sgl_register_operator(
    ID sparse_attn_sharedkv
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseAttnSharedkv
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/sparse_attn_sharedkv_def.cpp
    INFER_SOURCES op_host/sparse_attn_sharedkv_infer.cpp
    TILING_SOURCES op_host/sparse_attn_sharedkv_tiling.cpp
    KERNEL_SOURCES op_kernel/sparse_attn_sharedkv.cpp
)
