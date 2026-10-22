sgl_register_operator(
    ID apply_token_bitmask
    EXECUTION_UNIT AICORE
    CALL_MODE ACLNN
    CANN_OP_TYPE SglApplyTokenBitMask
    SUPPORTED_SOCS ${SGL_BITMASK_CONFIRMED_SOCS}
    HOST_SOURCES
        op_host/apply_token_bitmask_def.cpp
        op_host/apply_token_bitmask_tiling.cpp
    KERNEL_SOURCES op_kernel/apply_token_bitmask.cpp
)
