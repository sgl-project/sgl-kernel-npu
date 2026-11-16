# LayerNorm bridge to the CANN built-in ACLNN operator. arch22 only: the
# upstream aclnnLayerNormWithImplMode reference lists A2/A3 products.
sgl_register_operator(
    ID layer_norm
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES layer_norm.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
