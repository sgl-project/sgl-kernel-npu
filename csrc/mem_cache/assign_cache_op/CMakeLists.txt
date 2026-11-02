sgl_register_operator(
    ID assign_cache_op
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglAssignCache
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES op_host/assign_cache_op.cpp
    KERNEL_SOURCES op_kernel/assign_cache_op.cpp
)
