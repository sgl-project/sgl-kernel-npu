# Legacy Host and Kernel are common to A2/A3/A5 in cmake/legacy_kernels.cmake.
# This source support scope is independent of the target tested in a given run.
sgl_register_operator(
    ID apply_token_bitmask
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglApplyTokenBitMask
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES op_host/apply_token_bitmask.cpp
    KERNEL_SOURCES op_kernel/apply_token_bitmask.cpp
)
