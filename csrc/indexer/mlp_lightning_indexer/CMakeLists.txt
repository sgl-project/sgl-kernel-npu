# Official LongCat implementation supports the two arch22 CANN targets.
sgl_register_operator(
    ID mlp_lightning_indexer
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglMlpLightningIndexer
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/lightning_indexer_def.cpp
    INFER_SOURCES op_host/lightning_indexer_proto.cpp
    TILING_SOURCES op_host/lightning_indexer_tiling.cpp
    KERNEL_SOURCES op_kernel/mlp_lightning_indexer.cpp
)
