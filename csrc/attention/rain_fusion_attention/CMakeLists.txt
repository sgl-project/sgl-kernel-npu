# RainFusionAttention bridge to the CANN built-in ACLNN operator.
# arch22 only: aclnnRainFusionAttention supports Atlas A2/A3 and has no 950 path.
sgl_register_operator(
    ID rain_fusion_attention
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201
    CPP_SOURCES rain_fusion_attention.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
