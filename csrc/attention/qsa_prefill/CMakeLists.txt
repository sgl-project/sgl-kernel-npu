# D256 QSA prefill physical run preparation and shared-query attention kernels.
# arch22 only: both kernels target __CCE_AICORE__ == 220 (ascend910b / ascend910_93).
sgl_register_operator(
    ID qsa_prefill_prepare
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglQsaPrefillPrepare
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/qsa_prefill_prepare_def.cpp
    INFER_SOURCES op_host/qsa_prefill_prepare_infer.cpp
    TILING_SOURCES op_host/qsa_prefill_prepare_tiling.cpp
    KERNEL_SOURCES op_kernel/qsa_prefill_prepare.cpp
    TORCH_ADAPTER torch_adapter_prepare.cpp
)

sgl_register_operator(
    ID qsa_prefill_runs
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglQsaPrefillRuns
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/qsa_prefill_runs_def.cpp
    INFER_SOURCES op_host/qsa_prefill_runs_infer.cpp
    TILING_SOURCES op_host/qsa_prefill_runs_tiling.cpp
    KERNEL_SOURCES op_kernel/qsa_prefill_runs.cpp
    KERNEL_INCLUDE_DIRECTORIES ../sparse_attention_score/op_kernel
    TORCH_ADAPTER torch_adapter_runs.cpp
)
