# Keep the LongCat compute path for arch22 and the Ascend 950 port.
sgl_register_operator(
    ID mlp_lightning_indexer
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE MlpLightningIndexer
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    HOST_SOURCES
        op_host/lightning_indexer_def.cpp
        op_host/lightning_indexer_proto.cpp
        op_host/lightning_indexer_tiling.cpp
    KERNEL_SOURCES op_kernel/mlp_lightning_indexer.cpp
)
