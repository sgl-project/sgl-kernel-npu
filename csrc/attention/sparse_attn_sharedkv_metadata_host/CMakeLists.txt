sgl_register_operator(
    ID sparse_attn_sharedkv_metadata_host
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES ../../sparse_attn_sharedkv_metadata/op_host/sparse_attn_sharedkv_metadata.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
