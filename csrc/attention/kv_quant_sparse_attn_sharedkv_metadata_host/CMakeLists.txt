sgl_register_operator(
    ID kv_quant_sparse_attn_sharedkv_metadata_host
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 3510
    CPP_SOURCES ../../kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
