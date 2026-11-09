sgl_register_operator(
    ID chunk_kda_fwd
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglChunkKdaFwd
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/chunk_kda_fwd_def.cpp
    INFER_SOURCES op_host/chunk_kda_fwd_infer.cpp
    TILING_SOURCES op_host/chunk_kda_fwd_tiling.cpp
    TORCH_ADAPTER torch_adapter.cpp
    KERNEL_SOURCES op_kernel/chunk_kda_fwd.cpp
    KERNEL_INCLUDE_DIRECTORIES op_kernel ../../utils/kernel
    KERNEL_COMPILE_DEFINITIONS HAVE_WORKSPACE HAVE_TILING
    KERNEL_COMPILE_OPTIONS "--cce-auto-sync=off" "-Wno-deprecated-declarations"
)
sgl_register_operator_variant(
    ID chunk_kda_fwd ARCH 3510
    KERNEL_COMPILE_DEFINITIONS HAVE_WORKSPACE HAVE_TILING ENABLE_CV_COMM_VIA_SSBUF=true
)
