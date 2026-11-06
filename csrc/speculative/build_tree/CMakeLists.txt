# The legacy tree implementation is shared by the declared AI Core architectures.
# Source declarations do not imply that device validation has completed.
sgl_register_operator(
    ID build_tree
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglBuildTree
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/build_tree.cpp
    INFER_SOURCES op_host/build_tree.cpp
    TILING_SOURCES op_host/build_tree.cpp
    KERNEL_SOURCES op_kernel/build_tree.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
