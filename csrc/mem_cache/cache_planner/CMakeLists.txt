sgl_register_operator(
    ID cache_planner
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglPlanSparseKvCache
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/cache_planner_def.cpp
    INFER_SOURCES op_host/cache_planner_def.cpp
    TILING_SOURCES op_host/cache_planner_def.cpp
    KERNEL_SOURCES op_kernel/cache_planner.cpp
)
