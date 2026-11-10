# The legacy operator supports A2/A3 and uses one shared implementation.
sgl_register_operator(
    ID sparse_block_estimate
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseBlockEstimate
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/sparse_block_estimate_def.cpp
    TILING_SOURCES op_host/sparse_block_estimate_tiling.cpp
    INFER_SOURCES op_host/sparse_block_estimate_proto.cpp
    TORCH_ADAPTER torch_adapter.cpp
    KERNEL_SOURCES op_kernel/sparse_block_estimate.cpp
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
)
