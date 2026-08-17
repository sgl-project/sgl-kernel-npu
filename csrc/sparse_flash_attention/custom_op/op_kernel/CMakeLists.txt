npu_op_kernel_sources(sparse_flash_attention_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(sparse_flash_attention_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY sparse_flash_attention_kernels
)

npu_op_kernel_options("SparseFlashAttention" ALL OPTIONS
    -DASCENDC_DUMP=0
    --cce-auto-sync=off
    -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false
    -mllvm -cce-aicore-hoist-movemask=false
)
