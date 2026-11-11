# Same architecture and PTO computation as the legacy direct-launch operator.
sgl_register_operator(
    ID mega_chunk_gdn
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglMegaChunkGdn
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/mega_chunk_gdn_def.cpp
    INFER_SOURCES op_host/mega_chunk_gdn_infer.cpp
    TILING_SOURCES op_host/mega_chunk_gdn_tiling.cpp
    TORCH_ADAPTER torch_adapter.cpp
    KERNEL_SOURCES op_kernel/mega_chunk_gdn.cpp
    KERNEL_INCLUDE_DIRECTORIES op_kernel ../../../third_party/pto-isa/include
    KERNEL_COMPILE_DEFINITIONS GDN_D=128 GDN_C=128
)
