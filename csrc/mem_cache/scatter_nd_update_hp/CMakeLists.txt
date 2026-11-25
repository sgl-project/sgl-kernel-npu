sgl_register_operator(
    ID scatter_nd_update_hp
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglScatterNdUpdateSk
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/scatter_nd_update_hp_def.cpp
    INFER_SOURCES op_host/scatter_nd_update_hp_infer.cpp
    ACLNN_SOURCES
        op_host/op_api/aclnn_scatter_nd_update_hp.cpp
        op_host/op_api/scatter_nd_update_hp.cpp
    ACLNN_HEADERS op_host/op_api/aclnn_scatter_nd_update_hp.h
    ACLNN_LINK_LIBRARIES opapi_math
    TILING_SOURCES op_host/scatter_nd_update_hp_tiling.cpp op_host/scatter_nd_update_hp_tiling_arch35.cpp
    TILING_INCLUDE_DIRECTORIES op_host/compat
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator_variant(
    ID scatter_nd_update_hp ARCH 2201
    KERNEL_SOURCES op_kernel/scatter_nd_update_hp_arch22.cpp
)
sgl_register_operator_variant(
    ID scatter_nd_update_hp ARCH 3510
    KERNEL_SOURCES op_kernel/scatter_nd_update_hp_arch35.cpp
)
