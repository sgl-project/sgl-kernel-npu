sgl_register_operator(
    ID dcp_sparse_index_remap
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglDcpSparseIndexRemap
    SUPPORTED_ARCHS 2201
    TORCH_ADAPTER torch_adapter.cpp
    OPDEF_SOURCES op_host/dcp_sparse_index_remap.cpp
    INFER_SOURCES op_host/dcp_sparse_index_remap.cpp
    TILING_SOURCES op_host/dcp_sparse_index_remap.cpp
    KERNEL_SOURCES op_kernel/dcp_sparse_index_remap.cpp
    KERNEL_COMPILE_DEFINITIONS SGL_DCP_CANN_PACKAGE
)
