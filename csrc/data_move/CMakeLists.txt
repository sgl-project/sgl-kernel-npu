target_sources(sgl_kernel_npu PRIVATE
    torch_adapter.cpp
    transfer_state_dim_exchange.cpp
    transfer_kv_dim_exchange.cpp
    transfer_mamba_state.cpp
)
