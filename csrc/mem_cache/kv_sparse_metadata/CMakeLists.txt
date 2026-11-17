sgl_register_operator(
    ID kv_sparse_metadata
    EXECUTION_UNIT AICPU
    CANN_OP_TYPE BuildKvSparseMetadata
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    ACLNN_SOURCES op_host/aclnn_build_kv_sparse_metadata.cpp
    ACLNN_HEADERS op_host/aclnn_build_kv_sparse_metadata.h
    KERNEL_SOURCES op_kernel/build_kv_sparse_metadata.cpp
    AICPU_JSON op_kernel/kv_sparse_metadata.json
)
