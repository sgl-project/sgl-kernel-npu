# ELU is the second-rung Ascend C sample after helloworld: a real element-wise
# operator wired through the current CANN build (OpDef, tiling, generated ACLNN
# interface and Torch adapter).
#
# Only arch22 is declared: the Kernel was measured on Ascend910 / CANN 9.1.0 and
# no Ascend950 run has been recorded, so arch35 is not claimed.
sgl_register_operator(
    ID elu
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglElu
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/elu.cpp
    INFER_SOURCES op_host/elu.cpp
    TILING_SOURCES op_host/elu.cpp
    KERNEL_SOURCES op_kernel/kernel_elu.cpp
)
