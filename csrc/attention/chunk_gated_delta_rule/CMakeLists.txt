# Keep one native Host contract; select an explicit entry for each generation.
sgl_register_operator(
    ID chunk_gated_delta_rule
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglChunkGatedDeltaRule
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES op_host/chunk_gated_delta_rule.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator_variant(
    ID chunk_gated_delta_rule ARCH 2201
    KERNEL_SOURCES op_kernel/chunk_gated_delta_rule_arch22.cpp
)
sgl_register_operator_variant(
    ID chunk_gated_delta_rule ARCH 3510
    KERNEL_SOURCES op_kernel/chunk_gated_delta_rule_arch35.cpp
)
