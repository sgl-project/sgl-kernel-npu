# Preserve the shared kernel's existing arch22 / arch35 internal dispatch.
sgl_register_operator(
    ID recurrent_gated_delta_rule
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglRecurrentGatedDeltaRule
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/recurrent_gated_delta_rule_def.cpp
    INFER_SOURCES op_host/recurrent_gated_delta_rule_infer.cpp
    TILING_SOURCES op_host/recurrent_gated_delta_rule_tiling.cpp
    KERNEL_SOURCES op_kernel/recurrent_gated_delta_rule.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
