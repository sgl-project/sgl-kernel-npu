# LoRA sgemmv_shrink migrated to the independent CANN operator framework.
sgl_register_operator(
    ID sgemmv_shrink
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSgemmvShrink
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/sgemmv_shrink_def.cpp
    INFER_SOURCES op_host/sgemmv_shrink_infer.cpp
    TILING_SOURCES op_host/sgemmv_shrink_tiling.cpp
    KERNEL_SOURCES op_kernel/sgemmv_shrink.cpp
)
