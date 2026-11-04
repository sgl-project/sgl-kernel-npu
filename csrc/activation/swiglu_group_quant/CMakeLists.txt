sgl_register_operator(
    ID swiglu_group_quant
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSwigluGroupQuant
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/swiglu_group_quant.cpp
    INFER_SOURCES op_host/swiglu_group_quant.cpp
    TILING_SOURCES op_host/swiglu_group_quant.cpp
    KERNEL_SOURCES op_kernel/swiglu_group_quant.cpp
)
