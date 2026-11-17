# LoRA sgemmc_expand migrated to the independent CANN operator framework.
sgl_register_operator(
    ID sgemmc_expand
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSgemmcExpand
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/sgemmc_expand_def.cpp
    INFER_SOURCES op_host/sgemmc_expand_infer.cpp
    TILING_SOURCES op_host/sgemmc_expand_tiling.cpp
    KERNEL_SOURCES op_kernel/sgemmc_expand.cpp
)
