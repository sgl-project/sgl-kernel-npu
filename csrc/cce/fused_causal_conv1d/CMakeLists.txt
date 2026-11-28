sgl_register_operator(
    ID cce_fused_causal_conv1d
    EXECUTION_UNIT CCE
    SUPPORTED_ARCHS 3510
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_cce_kernel(
    ID cce_fused_causal_conv1d
    ARCH 3510
    VARIANT default
    SOURCE op_kernel/fused_causal_conv1d.cce
)
