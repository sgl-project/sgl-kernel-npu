# Declarations only; the HOST and Torch adapter builds collect selected inputs.
sgl_register_operator(
    ID transfer_kv_dim_exchange
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201 3510
    CPP_SOURCES transfer_kv_dim_exchange.cpp
    HOST_DEPENDENCIES TORCH_NPU
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator(
    ID transfer_mamba_state
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201 3510
    CPP_SOURCES transfer_mamba_state.cpp
    HOST_DEPENDENCIES TORCH_NPU
    TORCH_ADAPTER torch_adapter.cpp
)
foreach(direction IN ITEMS per_layer_direct_pf_lf all_layer_direct_lf_pf)
    sgl_register_operator(
        ID transfer_kv_${direction}
        EXECUTION_UNIT HOST
        SUPPORTED_ARCHS 2201 3510
        CPP_SOURCES transfer_state_dim_exchange.cpp
        HOST_DEPENDENCIES TORCH_NPU
        TORCH_ADAPTER torch_adapter.cpp
    )
endforeach()
