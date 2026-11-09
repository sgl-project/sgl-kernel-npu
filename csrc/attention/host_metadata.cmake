# Declare host schedulers before the unified request is split into CANN/HOST IDs.

sgl_register_operator(
    ID sparse_attn_sharedkv_metadata_host
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES ../sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata.cpp
    TORCH_ADAPTER sparse_attn_sharedkv_metadata_host/torch_adapter.cpp
)

sgl_register_operator(
    ID kv_quant_sparse_attn_sharedkv_metadata_host
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 3510
    CPP_SOURCES ../kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata.cpp
    TORCH_ADAPTER kv_quant_sparse_attn_sharedkv_metadata_host/torch_adapter.cpp
)
