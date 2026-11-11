sgl_register_operator(
    ID update_oe_token_table
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglUpdateOeTokenTable
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/update_oe_token_table.cpp
    INFER_SOURCES op_host/update_oe_token_table.cpp
    TILING_SOURCES op_host/update_oe_token_table.cpp
    KERNEL_SOURCES op_kernel/update_oe_token_table.cpp
)
