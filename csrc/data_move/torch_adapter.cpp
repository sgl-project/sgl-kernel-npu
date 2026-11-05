#include <ATen/ATen.h>
#include <torch/library.h>

namespace sglang::data_move {
#ifdef SGL_CPP_TRANSFER_KV_DIM_EXCHANGE
void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k, at::Tensor &device_v, at::Tensor &host_v,
                              const at::Tensor &device_indices, const at::Tensor &host_indices, int64_t page_size,
                              int64_t direction, int64_t flags);
#endif
#ifdef SGL_CPP_TRANSFER_MAMBA_STATE
void transfer_mamba_state(at::Tensor &device_buf, at::Tensor &host_buf, const at::Tensor &device_indices,
                          const at::Tensor &host_indices, int64_t direction);
#endif
#ifdef SGL_CPP_TRANSFER_KV_PER_LAYER_DIRECT_PF_LF
void transfer_kv_per_layer_direct_pf_lf(const at::Tensor &src, const at::Tensor &dst, const at::Tensor &src_indices,
                                        const at::Tensor &dst_indices, int64_t layer_id, int64_t flags);
#endif
#ifdef SGL_CPP_TRANSFER_KV_ALL_LAYER_DIRECT_LF_PF
void transfer_kv_all_layer_direct_lf_pf(at::TensorList device_states, at::TensorList host_states,
                                        const at::Tensor &device_indices, const at::Tensor &host_indices,
                                        int64_t flags);
#endif
}  // namespace sglang::data_move

TORCH_LIBRARY_FRAGMENT(npu, m)
{
#ifdef SGL_CPP_TRANSFER_KV_DIM_EXCHANGE
    m.def(
        "transfer_kv_dim_exchange(Tensor device_k, Tensor host_k, "
        "Tensor device_v, Tensor host_v, "
        "Tensor device_indices, Tensor host_indices, int page_size, int direct, int flags) -> ()");
#endif

#ifdef SGL_CPP_TRANSFER_MAMBA_STATE
    m.def(
        "transfer_mamba_state(Tensor device_buf, Tensor host_buf, "
        "Tensor device_indices, Tensor host_indices, int direction) -> ()");
#endif

#ifdef SGL_CPP_TRANSFER_KV_PER_LAYER_DIRECT_PF_LF
    m.def(
        "transfer_state_per_layer_direct_pf_lf(Tensor src, Tensor dst, "
        "Tensor src_indices, Tensor dst_indices, int layer_id, int flags) -> ()");
#endif

#ifdef SGL_CPP_TRANSFER_KV_ALL_LAYER_DIRECT_LF_PF
    m.def(
        "transfer_state_all_layer_direct_lf_pf(Tensor[] device_states, Tensor[] host_states, "
        "Tensor device_indices, Tensor host_indices, int flags) -> ()");
#endif
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
#ifdef SGL_CPP_TRANSFER_KV_DIM_EXCHANGE
    m.impl("transfer_kv_dim_exchange", TORCH_FN(sglang::data_move::transfer_kv_dim_exchange));
#endif

#ifdef SGL_CPP_TRANSFER_MAMBA_STATE
    m.impl("transfer_mamba_state", TORCH_FN(sglang::data_move::transfer_mamba_state));
#endif
#ifdef SGL_CPP_TRANSFER_KV_PER_LAYER_DIRECT_PF_LF
    m.impl("transfer_state_per_layer_direct_pf_lf", TORCH_FN(sglang::data_move::transfer_kv_per_layer_direct_pf_lf));
#endif

#ifdef SGL_CPP_TRANSFER_KV_ALL_LAYER_DIRECT_LF_PF
    m.impl("transfer_state_all_layer_direct_lf_pf", TORCH_FN(sglang::data_move::transfer_kv_all_layer_direct_lf_pf));
#endif
}
