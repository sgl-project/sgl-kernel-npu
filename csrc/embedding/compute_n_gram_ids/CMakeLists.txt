sgl_register_operator(
    ID compute_n_gram_ids
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglComputeNGramIds
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/compute_n_gram_ids.cpp
    INFER_SOURCES op_host/compute_n_gram_ids.cpp
    TILING_SOURCES op_host/compute_n_gram_ids.cpp
    KERNEL_SOURCES op_kernel/compute_n_gram_ids.cpp
)
