#ifdef SGL_CPP_CCE_FUSED_CAUSAL_CONV1D
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <limits>
#include <initializer_list>
#include "op_kernel/fused_causal_conv1d.h"
namespace {
at::Tensor causal_conv1d(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &state,
                         const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &query,
                         const c10::optional<at::Tensor> &slots, const c10::optional<at::Tensor> &initial,
                         int64_t activation, int64_t pad, int64_t mode)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "expected NPU tensors");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be [T,D] or [B,L,D]");
    int64_t dim = x.size(-1), total = x.numel() / std::max<int64_t>(dim, 1);
    TORCH_CHECK(dim > 0 && dim <= 2048 && dim % 128 == 0, "CCE requires D<=2048 divisible by 128");
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({3, dim}), "CCE requires K=3 weights");
    TORCH_CHECK(state.dim() == 3 && state.size(1) >= 2 && state.size(2) == dim, "invalid state shape");
    for (const auto *t : {&x, &weight, &state}) {
        TORCH_CHECK(t->device() == x.device() && t->scalar_type() == at::kBFloat16 && t->is_contiguous(),
                    "expected contiguous BF16 tensors on same NPU");
        TORCH_CHECK(!t->requires_grad(), "forward only");
    }
    const c10::DeviceGuard guard(x.device());
    auto metadata = [&](const c10::optional<at::Tensor> &t) {
        if (!t.has_value() || !t->numel()) return at::Tensor();
        TORCH_CHECK(t->device() == x.device() && t->dim() == 1, "metadata must be 1D on input NPU");
        TORCH_CHECK(t->scalar_type() == at::kInt || t->scalar_type() == at::kLong || t->scalar_type() == at::kBool,
                    "invalid metadata dtype");
        return t->to(at::kLong).contiguous();
    };
    auto q = x.dim() == 2 ? metadata(query) : at::Tensor();
    TORCH_CHECK(x.dim() == 3 || q.defined(), "packed input requires query_start_loc");
    int64_t batch = x.dim() == 3 ? x.size(0) : q.numel() - 1;
    TORCH_CHECK(batch > 0 && total > 0 && total <= std::numeric_limits<int>::max(), "invalid input size");
    auto indices = metadata(slots), flags = metadata(initial);
    for (const auto *t : {&indices, &flags})
        TORCH_CHECK(!t->defined() || t->numel() == batch, "metadata length mismatch");
    TORCH_CHECK(indices.defined() || state.size(0) >= batch, "insufficient state slots");
    auto b = bias.has_value() && bias->numel() ? *bias : at::Tensor();
    if (b.defined())
        TORCH_CHECK(b.device() == x.device() && b.scalar_type() == at::kBFloat16 && b.is_contiguous() &&
                        b.numel() == dim && !b.requires_grad(),
                    "invalid bias");
    auto y = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream(x.get_device());
    for (const auto *t : std::initializer_list<const at::Tensor *>{&x, &weight, &state, &q, &indices, &flags, &b})
        if (t->defined()) c10_npu::NPUCachingAllocator::recordStream(t->storage().data_ptr(), stream);
    auto half = [](const at::Tensor &t) { return t.defined() ? (const uint16_t *)t.data_ptr() : nullptr; };
    auto meta = [](const at::Tensor &t) { return t.defined() ? t.data_ptr<int64_t>() : nullptr; };
    auto status =
        cce_launch_causal_conv1d(half(x), half(weight), (uint16_t *)state.data_ptr(), (uint16_t *)y.data_ptr(), half(b),
                                 meta(q), meta(indices), meta(flags), batch, x.dim() == 3 ? x.size(1) : 0, total, dim,
                                 state.size(1), state.size(0), activation != 0, pad, mode == 1, stream.stream());
    TORCH_CHECK(status == ACL_SUCCESS, "CCE launch failed: ", status);
    return y;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(cce, m)
{
    m.def(
        "causal_conv1d(Tensor x, Tensor weight, Tensor(a!) conv_states, Tensor? bias=None, Tensor? "
        "query_start_loc=None, Tensor? cache_indices=None, Tensor? has_initial_state=None, int activation_mode=0, int "
        "pad_slot_id=-1, int run_mode=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(cce, PrivateUse1, m)
{
    m.impl("causal_conv1d", &causal_conv1d);
}
#endif
