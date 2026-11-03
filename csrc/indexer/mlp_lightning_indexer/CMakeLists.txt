# Official LongCat implementation supports the two arch22 CANN targets.
sgl_register_operator(
    ID mlp_lightning_indexer
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE MlpLightningIndexer
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    HOST_SOURCES
        op_host/lightning_indexer_def.cpp
        op_host/lightning_indexer_proto.cpp
        op_host/lightning_indexer_tiling.cpp
    KERNEL_SOURCES op_kernel/mlp_lightning_indexer.cpp
)
