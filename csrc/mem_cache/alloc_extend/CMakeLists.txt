sgl_register_operator(
    ID alloc_extend
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglAllocExtend
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/alloc_extend.cpp
    INFER_SOURCES op_host/alloc_extend.cpp
    TILING_SOURCES op_host/alloc_extend.cpp
    KERNEL_SOURCES op_kernel/alloc_extend.cpp
)
