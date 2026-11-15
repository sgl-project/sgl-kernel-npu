# Laser attention migrated to the independent CANN operator framework.
# arch22 only: the legacy definition registers ascend910b / ascend910_93.
sgl_register_operator(
    ID laser_attention
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglLaserAttention
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/laser_attention_def.cpp
    INFER_SOURCES op_host/laser_attention_infer.cpp
    TILING_SOURCES op_host/laser_attention_tiling.cpp
    KERNEL_SOURCES op_kernel/laser_attention.cpp
    # Legacy device options: manual synchronization and permissive legacy casts.
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -fpermissive
)
