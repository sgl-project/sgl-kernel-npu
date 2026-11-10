sgl_register_operator(
    ID kv_compress_epilog
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglKvCompressEpilog
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/kv_compress_epilog.cpp
    INFER_SOURCES op_host/kv_compress_epilog.cpp
    TILING_SOURCES op_host/kv_compress_epilog.cpp
    KERNEL_SOURCES op_kernel/kv_compress_epilog.cpp
)
