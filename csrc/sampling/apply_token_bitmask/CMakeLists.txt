# Legacy Host and Kernel are common to A2/A3/A5 in cmake/legacy_kernels.cmake.
# This source support scope is independent of the target tested in a given run.
sgl_register_operator(
    ID apply_token_bitmask
    EXECUTION_UNIT AICORE
    CALL_MODE ACLNN
    CANN_OP_TYPE SglApplyTokenBitMask
    TORCH_ADAPTER ../../aclnn_torch_adapter/apply_token_bitmask.cpp
    SUPPORTED_SOCS ascend910b ascend910_93 ascend950
    HOST_SOURCES op_host/apply_token_bitmask.cpp
    KERNEL_SOURCES op_kernel/apply_token_bitmask.cpp
)
