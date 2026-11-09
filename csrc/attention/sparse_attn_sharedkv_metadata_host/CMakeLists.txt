# Loaded early for HOST selection and later by attention's directory discovery.
include_guard(GLOBAL)

sgl_register_operator(
    ID sparse_attn_sharedkv_metadata_host
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES op_host/sparse_attn_sharedkv_metadata.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
