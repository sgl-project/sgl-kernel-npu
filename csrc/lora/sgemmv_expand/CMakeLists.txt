# LoRA operators migrated to the independent CANN operator framework.
# The legacy lora sources live in csrc/lora/op_host and csrc/lora/op_kernel and
# remain unchanged; only directories listed here are discovered as operators.
sgl_register_operator(
    ID sgemmv_expand
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSgemmvExpand
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/sgemmv_expand.cpp
    INFER_SOURCES op_host/sgemmv_expand.cpp
    TILING_SOURCES op_host/sgemmv_expand.cpp
    KERNEL_SOURCES op_kernel/sgemmv_expand.cpp
)
