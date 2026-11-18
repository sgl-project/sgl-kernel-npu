sgl_register_operator(
    ID unidex_copy
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglUnidexCopy
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/unidex_copy.cpp
    INFER_SOURCES op_host/unidex_copy.cpp
    TILING_SOURCES op_host/unidex_copy.cpp
    KERNEL_SOURCES op_kernel/unidex_copy.cpp
)
