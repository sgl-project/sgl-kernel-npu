#include "aclnn_runtime.h"
#include "aclnn_sgl_qsa_cube_attention.h"
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>

namespace {
at::Tensor qsa_cube_attention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                              const at::Tensor &indices, const at::Tensor &pages, const at::Tensor &requests)
{
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 2 && q.size(0) < 128 && q.size(1) == 3 && q.size(2) == 256,
                "expected short-prefill queries [2..127, 3, 256]");
    TORCH_CHECK(k.dim() == 4 && k.size(0) > 0 && k.size(1) == 128 && k.size(2) == 1 && k.size(3) == 256 &&
                k.sizes() == v.sizes(), "expected K/V [pages, 128, 1, 256]");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0) &&
                (indices.size(1) == 514 || indices.size(1) == 2051), "unexpected QSA index shape");
    TORCH_CHECK(pages.dim() == 2 && pages.size(0) > 0 && pages.size(1) > 0 && pages.size(1) <= 512,
                "expected a page table with at most 512 columns");
    TORCH_CHECK(requests.dim() == 1 && requests.size(0) == q.size(0), "expected one request ID per query");
    c10::DeviceGuard guard(q.device());
    for (const auto &t : {q, k, v}) {
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && t.scalar_type() == at::kBFloat16,
                    "expected contiguous BF16 query, key and value on one device");
    }
    for (const auto &t : {indices, pages, requests}) {
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && t.scalar_type() == at::kInt,
                    "expected contiguous int32 metadata on the query device");
    }
    auto out = at::empty_like(q);
    std::vector<at::Tensor> tensors{q, k, v, indices, pages, requests, out};
    std::vector<std::shared_ptr<aclTensor>> desc;
    for (const auto &t : tensors) {
        desc.push_back(sgl_cann::tensor_descriptor(t, t.scalar_type() == at::kInt ? ACL_INT32 : ACL_BF16));
    }
    static auto query = reinterpret_cast<decltype(&aclnnSglQsaCubeAttentionGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglQsaCubeAttentionGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglQsaCubeAttention)>(
        sgl_cann::op_api_symbol("aclnnSglQsaCubeAttention"));
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(desc[0].get(), desc[1].get(), desc[2].get(), desc[3].get(), desc[4].get(),
                        desc[5].get(), desc[6].get(), &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "QSA Cube workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "QSA workspace size overflow");
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, q.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApi("aclnnSglQsaCubeAttention", [=, tensors = std::move(tensors), desc = std::move(desc)]() -> int {
        // The host task owns the tensor storage and ACL descriptors until it enqueues the kernel.
        (void)tensors;
        (void)desc;
        auto rc = execute(bytes ? workspace.data_ptr() : nullptr, bytes, executor, stream);
        TORCH_CHECK(rc == ACL_SUCCESS, "QSA Cube launch failed: ", aclGetRecentErrMsg());
        return rc;
    });
    return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sgl_qsa_cube_attention(Tensor q, Tensor k, Tensor v, Tensor indices, Tensor pages, Tensor requests) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_qsa_cube_attention", &qsa_cube_attention);
}
