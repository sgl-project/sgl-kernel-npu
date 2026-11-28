sgl_register_operator(
    ID fused_lightning_indexer_manage
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglFusedLightningIndexerManage
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/fused_lightning_indexer_manage_def.cpp
    INFER_SOURCES op_host/fused_lightning_indexer_manage_def.cpp
    TILING_SOURCES op_host/fused_lightning_indexer_manage_def.cpp
    KERNEL_SOURCES op_kernel/fused_lightning_indexer_manage.cpp
)
