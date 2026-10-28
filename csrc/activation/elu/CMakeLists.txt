# ELU: vector (AI Core) element-wise operator, fp16/fp32.
#
# Source paths are relative to this file. Arch IDs follow
# csrc/cmake/architectures.json: 2201 -> ascend910b/ascend910_93 (A2/A3),
# 3510 -> ascend950 (A5).
#
# This declaration is consumed by the CANN build entry (#880). Until that entry
# lands, the same sources are listed in csrc/CMakeLists.txt (OP_SRCS /
# NO_WORKSPACE_KERNEL_SRCS) and the op is registered from
# csrc/pytorch_extensions.cpp, so there is deliberately no TORCH_ADAPTER here
# to avoid registering npu::elu twice.
sgl_register_operator(
    ID elu
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglElu
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES op_host/elu.cpp
    KERNEL_SOURCES op_kernel/kernel_elu.cpp
)
