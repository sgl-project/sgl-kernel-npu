# MLA preprocess migrated to the independent CANN operator framework.
# The legacy sources under csrc/mla_preprocess remain unchanged. The shared
# device primitives live in csrc/utils/kernel/device_prims.
sgl_register_operator(
    ID mla_preprocess
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglMlaPreprocess
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/mla_preprocess_def.cpp
    INFER_SOURCES op_host/mla_preprocess_infer.cpp
    TILING_SOURCES op_host/mla_preprocess_tiling.cpp
    KERNEL_SOURCES op_kernel/mla_preprocess.cpp
    KERNEL_INCLUDE_DIRECTORIES ../../utils/kernel
)
