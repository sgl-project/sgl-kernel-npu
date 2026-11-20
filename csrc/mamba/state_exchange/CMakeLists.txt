sgl_register_operator(
    ID state_exchange
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglStateExchange
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 2201 3510
    OPDEF_SOURCES op_host/state_exchange.cpp
    INFER_SOURCES op_host/state_exchange.cpp
    TILING_SOURCES op_host/state_exchange.cpp
    KERNEL_SOURCES op_kernel/state_exchange_kernel.cpp
)
