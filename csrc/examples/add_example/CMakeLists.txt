# CpuKernel sample; computation is common to the supported architectures.
sgl_register_operator(
    ID add_example
    EXECUTION_UNIT AICPU
    CANN_OP_TYPE SglAddExample
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    ACLNN_SOURCES op_host/aclnn_sgl_add_example.cpp
    ACLNN_HEADERS op_host/aclnn_sgl_add_example.h
    KERNEL_SOURCES op_kernel_aicpu/add_example_aicpu.cpp
    AICPU_JSON op_kernel_aicpu/add_example.json
)
