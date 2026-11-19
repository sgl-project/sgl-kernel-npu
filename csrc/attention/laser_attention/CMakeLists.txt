# Legacy-parity experiment: every operator source below is copied verbatim
# from csrc/attentions/csrc/ops/laser_attention. The OpType, tiling macros and
# device Kernel are unchanged; this declaration only wires the legacy files
# into the repository build so the wheel exposes the same operation.
#
# Only arch22 is declared because the repository build targets ascend910b /
# ascend910_93, which is the subset of the legacy definition used here.
sgl_register_operator(
    ID laser_attention
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE LaserAttention
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/laser_attention_def.cpp
    INFER_SOURCES op_host/laser_attention_proto.cpp
    TILING_SOURCES op_host/laser_attention_tiling.cpp
    KERNEL_SOURCES op_kernel/laser_attention.cpp
    # Same device options as the legacy op_host/CMakeLists.txt. -Werror is
    # omitted so a newer SDK diagnostic cannot block the parity build; it does
    # not change generated code.
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -fpermissive
)
