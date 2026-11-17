sgl_register_operator(
    ID kv_exchange
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglKvExchange
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/kv_exchange.cpp
    INFER_SOURCES op_host/kv_exchange.cpp
    TILING_SOURCES op_host/kv_exchange.cpp
    KERNEL_SOURCES op_kernel/kv_exchange_kernel.cpp
)
