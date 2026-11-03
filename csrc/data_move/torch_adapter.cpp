#include <ATen/ATen.h>
#include <torch/library.h>

namespace sglang::data_move {
void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k, at::Tensor &device_v,
                                       at::Tensor &host_v, const at::Tensor &device_indices,
                                       const at::Tensor &host_indices, int64_t page_size, int64_t direction,
                                       int64_t flags);
void transfer_mamba_state(at::Tensor &device_buf, at::Tensor &host_buf, const at::Tensor &device_indices,
                                   const at::Tensor &host_indices, int64_t direction);
void transfer_kv_per_layer_direct_pf_lf(const at::Tensor &src, const at::Tensor &dst,
                                                    const at::Tensor &src_indices, const at::Tensor &dst_indices,
                                                    int64_t layer_id, int64_t flags);
void transfer_kv_all_layer_direct_lf_pf(at::TensorList device_states, at::TensorList host_states,
                                                    const at::Tensor &device_indices, const at::Tensor &host_indices,
                                                    int64_t flags);
}

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "transfer_kv_dim_exchange(Tensor device_k, Tensor host_k, "
        "Tensor device_v, Tensor host_v, "
        "Tensor device_indices, Tensor host_indices, int page_size, int direct, int flags) -> ()");

    m.def(
        "transfer_mamba_state(Tensor device_buf, Tensor host_buf, "
        "Tensor device_indices, Tensor host_indices, int direction) -> ()");

    m.def(
        "transfer_state_per_layer_direct_pf_lf(Tensor src, Tensor dst, "
        "Tensor src_indices, Tensor dst_indices, int layer_id, int flags) -> ()");

    m.def(
        "transfer_state_all_layer_direct_lf_pf(Tensor[] device_states, Tensor[] host_states, "
        "Tensor device_indices, Tensor host_indices, int flags) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("transfer_kv_dim_exchange", TORCH_FN(sglang::data_move::transfer_kv_dim_exchange));

    m.impl("transfer_mamba_state", TORCH_FN(sglang::data_move::transfer_mamba_state));
    m.impl("transfer_state_per_layer_direct_pf_lf",
           TORCH_FN(sglang::data_move::transfer_kv_per_layer_direct_pf_lf));

    m.impl("transfer_state_all_layer_direct_lf_pf",
           TORCH_FN(sglang::data_move::transfer_kv_all_layer_direct_lf_pf));
}
