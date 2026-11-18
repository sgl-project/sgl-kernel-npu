sgl_register_operator(
    ID slot_map_lookup
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSlotMapLookup
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES op_host/slot_map_lookup.cpp
    INFER_SOURCES op_host/slot_map_lookup.cpp
    TILING_SOURCES op_host/slot_map_lookup.cpp
    KERNEL_SOURCES op_kernel/slot_map_lookup.cpp
)
