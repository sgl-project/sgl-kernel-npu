# One interface; generation-specific algorithm and entry sources follow #921.
sgl_register_operator(
    ID compressor
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglCompressor
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/compressor_def.cpp
    INFER_SOURCES op_host/compressor_infer.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator_variant(
    ID compressor ARCH 2201
    TILING_SOURCES op_host/arch22/compressor_tiling.cpp op_host/arch22/compressor_tiling_adapter.cpp
    TILING_INCLUDE_DIRECTORIES op_host/compat
    KERNEL_SOURCES op_kernel/compressor_arch22.cpp
    KERNEL_INCLUDE_DIRECTORIES op_kernel op_kernel/arch22
)
sgl_register_operator_variant(
    ID compressor ARCH 3510
    TILING_SOURCES op_host/arch35/compressor_tiling.cpp op_host/arch35/compressor_tiling_adapter.cpp
    TILING_INCLUDE_DIRECTORIES op_host/compat
    KERNEL_SOURCES op_kernel/compressor_arch35.cpp
    KERNEL_INCLUDE_DIRECTORIES op_kernel op_kernel/arch35
)
