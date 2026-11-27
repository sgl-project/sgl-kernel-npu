# BF16 hyper-connection epilogue AIV kernels (arch22).
sgl_register_operator(
    ID hyper_conn_grouped_norm
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglHyperConnGroupedNorm
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/hyper_conn_grouped_norm.cpp
    INFER_SOURCES op_host/hyper_conn_grouped_norm.cpp
    TILING_SOURCES op_host/hyper_conn_grouped_norm.cpp
    KERNEL_SOURCES op_kernel/hyper_conn_grouped_norm.cpp
    TORCH_ADAPTER torch_adapter_grouped_norm.cpp
)

sgl_register_operator(
    ID hyper_conn_scaled_silu
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglHyperConnScaledSilu
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/hyper_conn_scaled_silu.cpp
    INFER_SOURCES op_host/hyper_conn_scaled_silu.cpp
    TILING_SOURCES op_host/hyper_conn_scaled_silu.cpp
    KERNEL_SOURCES op_kernel/hyper_conn_scaled_silu.cpp
    TORCH_ADAPTER torch_adapter_scaled_silu.cpp
)

sgl_register_operator(
    ID hyper_conn_branch_mix
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglHyperConnBranchMix
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/hyper_conn_branch_mix.cpp
    INFER_SOURCES op_host/hyper_conn_branch_mix.cpp
    TILING_SOURCES op_host/hyper_conn_branch_mix.cpp
    KERNEL_SOURCES op_kernel/hyper_conn_branch_mix.cpp
    TORCH_ADAPTER torch_adapter_branch_mix.cpp
)

sgl_register_operator(
    ID hyper_conn_branch_combine
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglHyperConnBranchCombine
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/hyper_conn_branch_combine.cpp
    INFER_SOURCES op_host/hyper_conn_branch_combine.cpp
    TILING_SOURCES op_host/hyper_conn_branch_combine.cpp
    KERNEL_SOURCES op_kernel/hyper_conn_branch_combine.cpp
    TORCH_ADAPTER torch_adapter_branch_combine.cpp
)
