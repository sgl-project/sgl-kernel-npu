# The kernel supports Ascend A2/A3, which use architecture 2201.
sgl_register_operator(
    ID batch_matmul_transpose
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglBatchMatmulTranspose
    SUPPORTED_ARCHS 2201
    # Keep the original Host/tiling/Kernel file layout. Codegen also compiles
    # the scheduling definitions referenced by the combined OpDef/Tiling Host.
    OPDEF_SOURCES op_host/batch_matmul_transpose.cpp op_host/tiling/tiling_data.cpp
    TILING_SOURCES op_host/batch_matmul_transpose.cpp op_host/tiling/tiling_data.cpp
    KERNEL_SOURCES op_kernel/batch_matmul_transpose_kernel.cpp
    # Resolve the relative tiling header in copied sources and expose shared kernel headers.
    KERNEL_INCLUDE_DIRECTORIES op_kernel ../../utils/kernel
    TORCH_ADAPTER torch_adapter.cpp
)
