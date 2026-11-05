# The entry preserves the existing internal generic/arch35 dispatch and both run modes.
sgl_register_operator(
    ID causal_conv1d
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglCausalConv1d
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/causal_conv1d_def.cpp
    INFER_SOURCES op_host/causal_conv1d_infer.cpp
    TILING_SOURCES op_host/causal_conv1d_tiling.cpp
    KERNEL_SOURCES op_kernel/causal_conv1d.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
