sgl_register_operator(
    ID fused_quant_lightning_indexer_manage_full_visible
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglFusedQuantLightningIndexerManage
    TORCH_ADAPTER op_host/fused_quant_lightning_indexer_manage.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/fused_quant_lightning_indexer_manage_full_visible_def.cpp
    INFER_SOURCES op_host/fused_quant_lightning_indexer_manage_full_visible_def.cpp
    TILING_SOURCES op_host/fused_quant_lightning_indexer_manage_full_visible_def.cpp
    KERNEL_SOURCES op_kernel/sgl_fused_quant_lightning_indexer_manage.cpp
    KERNEL_INCLUDE_DIRECTORIES op_kernel op_kernel/arch35
    KERNEL_COMPILE_DEFINITIONS HAVE_WORKSPACE HAVE_TILING
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
        -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false
)
