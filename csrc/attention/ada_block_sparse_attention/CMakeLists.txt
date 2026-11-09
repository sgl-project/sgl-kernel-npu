# Ada block sparse attention migrated to the independent CANN operator framework.
# arch22 only: the legacy definition registers ascend910b / ascend910_93.
sgl_register_operator(
    ID ada_block_sparse_attention
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglAdaBlockSparseAttention
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/ada_block_sparse_attention_def.cpp
    INFER_SOURCES op_host/ada_block_sparse_attention_infer.cpp
    TILING_SOURCES op_host/ada_block_sparse_attention_tiling.cpp
    KERNEL_SOURCES op_kernel/ada_block_sparse_attention.cpp
    # Legacy device options: manual synchronization and permissive legacy casts.
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -fpermissive
)
