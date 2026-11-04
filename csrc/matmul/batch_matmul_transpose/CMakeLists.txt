# The legacy kernel targets 910C (ascend910_93), which uses architecture 2201.
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
    # Resolve relative headers even when the SDK compiles a copied source tree.
    KERNEL_INCLUDE_DIRECTORIES op_kernel
    TORCH_ADAPTER torch_adapter.cpp
)
