// Framework bridge for the independent LaserAttention CANN operator.
// Validation and output layout follow the legacy attentions plugin; the
// computation remains in the separately copied native implementation.

#include "aclnn_runtime.h"
#include "aclnn_sgl_laser_attention.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

aclDataType TorchToAclDtype(const at::Tensor &tensor)
{
    switch (tensor.scalar_type()) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kByte:
            return ACL_UINT8;
        case at::kFloat:
            return ACL_FLOAT;
        default:
            TORCH_CHECK(false, "laser attention does not support dtype ", tensor.scalar_type());
    }
    return ACL_FLOAT;
}

std::tuple<at::Tensor, at::Tensor> la(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                      const c10::optional<at::Tensor> &atten_mask_opt,
                                      const c10::optional<at::Tensor> &alibi_mask_opt,
                                      const c10::optional<at::Tensor> &drop_mask_opt, double scale_value,
                                      int64_t head_num, std::string input_layout, double keep_prob, int64_t pre_tokens,
                                      int64_t next_tokens, bool is_highPrecision)
{
    TORCH_CHECK(query.dim() == 4, "The first input dimension of la must be 4 but got ", query.dim());
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on an NPU");
    TORCH_CHECK(key.device() == query.device() && value.device() == query.device(),
                "query/key/value must be on the same NPU");
    const bool hasAttenMask = atten_mask_opt.has_value() && atten_mask_opt->defined();
    const bool hasAlibiMask = alibi_mask_opt.has_value() && alibi_mask_opt->defined();
    const bool hasDropMask = drop_mask_opt.has_value() && drop_mask_opt->defined();
    if (hasAttenMask) {
        TORCH_CHECK(atten_mask_opt->device() == query.device(), "atten_mask must be on the same NPU");
    }
    if (hasAlibiMask) {
        TORCH_CHECK(alibi_mask_opt->device() == query.device(), "alibi_mask must be on the same NPU");
    }
    if (hasDropMask) {
        TORCH_CHECK(drop_mask_opt->device() == query.device(), "drop_mask must be on the same NPU");
    }

    const c10::DeviceGuard deviceGuard(query.device());

    // Continuous work buffers keep the legacy ND contract without extra copies.
    auto q = query.contiguous(), k = key.contiguous(), v = value.contiguous();
    at::Tensor attenMask, alibiMask, dropMask;
    if (hasAttenMask) attenMask = atten_mask_opt->contiguous();
    if (hasAlibiMask) alibiMask = alibi_mask_opt->contiguous();
    if (hasDropMask) dropMask = drop_mask_opt->contiguous();

    // The Kernel never writes softmax_log_max_sum; the output stays allocated for
    // signature compatibility with the legacy operator.
    auto softmaxLogMaxSum = at::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
    auto attentionOut = at::empty(q.sizes(), q.options().dtype(at::kFloat));

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglLaserAttentionGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglLaserAttentionGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglLaserAttention)>(sgl_cann::op_api_symbol("aclnnSglLaserAttention"));
    auto qDesc = sgl_cann::tensor_descriptor(q, TorchToAclDtype(q));
    auto kDesc = sgl_cann::tensor_descriptor(k, TorchToAclDtype(k));
    auto vDesc = sgl_cann::tensor_descriptor(v, TorchToAclDtype(v));
    auto attenMaskDesc = hasAttenMask ? sgl_cann::tensor_descriptor(attenMask, TorchToAclDtype(attenMask))
                                      : std::shared_ptr<aclTensor>{};
    auto alibiMaskDesc = hasAlibiMask ? sgl_cann::tensor_descriptor(alibiMask, TorchToAclDtype(alibiMask))
                                      : std::shared_ptr<aclTensor>{};
    auto dropMaskDesc =
        hasDropMask ? sgl_cann::tensor_descriptor(dropMask, TorchToAclDtype(dropMask)) : std::shared_ptr<aclTensor>{};
    auto softmaxDesc = sgl_cann::tensor_descriptor(softmaxLogMaxSum, ACL_FLOAT);
    auto outDesc = sgl_cann::tensor_descriptor(attentionOut, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        queryWorkspace(qDesc.get(), kDesc.get(), vDesc.get(), attenMaskDesc.get(), alibiMaskDesc.get(),
                       dropMaskDesc.get(), scale_value, head_num, input_layout.data(), keep_prob, pre_tokens,
                       next_tokens, is_highPrecision, softmaxDesc.get(), outDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "laser attention workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));

    std::vector<at::Tensor> tensors{q, k, v, softmaxLogMaxSum, attentionOut, workspace};
    if (hasAttenMask) tensors.push_back(attenMask);
    if (hasAlibiMask) tensors.push_back(alibiMask);
    if (hasDropMask) tensors.push_back(dropMask);
    std::vector<std::shared_ptr<aclTensor>> descriptors{qDesc,         kDesc,        vDesc,       attenMaskDesc,
                                                        alibiMaskDesc, dropMaskDesc, softmaxDesc, outDesc};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, input_layout, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "laser attention ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglLaserAttention", launch);
    return std::make_tuple(softmaxLogMaxSum, attentionOut);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "la(Tensor query, Tensor key, Tensor value, Tensor? atten_mask=None, Tensor? alibi_mask=None, "
        "Tensor? drop_mask=None, float scale_value=1.0, int head_num=2, str input_layout='BNSD', "
        "float keep_prob=1.0, int pre_tokens=2147483647, int next_tokens=1, bool is_highPrecision=True) "
        "-> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("la", &la);
}
