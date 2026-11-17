# A5-only fused SiTU + MXFP8 quantization. The legacy A2/A3 path has no
# equivalent operator, so only arch35 is declared.
sgl_register_operator(
    ID situ_mxfp8_quant
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSituMxfp8Quant
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/situ_mxfp8_quant.cpp
    INFER_SOURCES op_host/situ_mxfp8_quant.cpp
    TILING_SOURCES op_host/situ_mxfp8_quant.cpp
    KERNEL_SOURCES op_kernel/situ_mxfp8_quant.cpp
    KERNEL_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/../common/kernel
)
