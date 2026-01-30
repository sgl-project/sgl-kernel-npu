#include <cstdio>
#include <string>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/apply_top_k_top_p_min_p_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "common_tiling.h"
#include "apply_top_k_top_p_min_p_def.h"
#include "common.h"
#include "aclrtlaunch_apply_top_k_top_p_min_p.h"

namespace sglang::ATKTPMPHost {

using namespace ge_helper;
constexpr uint32_t PADDING_BYTE = 32U;

inline at::Tensor ConstructApplyTopKTopPMinPOutputTensor(const at::Tensor &probs)
{
    for (size_t i = 0; i < probs.sizes().size(); i++) {
        TORCH_CHECK(probs.size(i) > 0,
                    "All values within probs's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", probs.size(i));
    }
    at::Tensor output = at::empty_like(probs);
    return output;
}
}  // namespace sglang::ATKTPMPHost

namespace sglang {
namespace npu_kernel {
HOST_API at::Tensor apply_top_k_top_p_min_p(const at::Tensor &probs, const at::Tensor &k, const at::Tensor &p,
                                            const c10::optional<at::Tensor> &min_p)
{
    using namespace ATKTPMPHost;
    at::Tensor sampledRes = ConstructApplyTopKTopPMinPOutputTensor(probs);

    auto probsType = probs.scalar_type();

    at::Tensor minP = min_p.has_value()
                          ? min_p.value()
                          : at::empty({1}, at::TensorOptions().dtype(probsType).device(probs.options().device()));

    ApplyTopKTopPMinPTilingInfo applyTopKTopPMinPInfo;
    applyTopKTopPMinPInfo.opParamInfo.probs.dtype = SCALAR_TYPE_TO_GE_DATATYPE(probsType);
    applyTopKTopPMinPInfo.opParamInfo.probs.shape = probs.sizes();
    applyTopKTopPMinPInfo.opParamInfo.k.dtype = SCALAR_TYPE_TO_GE_DATATYPE(k.scalar_type());
    applyTopKTopPMinPInfo.opParamInfo.k.shape = k.sizes();
    applyTopKTopPMinPInfo.opParamInfo.p.dtype = SCALAR_TYPE_TO_GE_DATATYPE(p.scalar_type());
    applyTopKTopPMinPInfo.opParamInfo.p.shape = p.sizes();
    if (min_p.has_value()) {
        applyTopKTopPMinPInfo.opParamInfo.minP.dtype = SCALAR_TYPE_TO_GE_DATATYPE(minP.scalar_type());
        applyTopKTopPMinPInfo.opParamInfo.minP.shape = minP.sizes();
    }
    applyTopKTopPMinPInfo.opParamInfo.sampledRes.dtype = SCALAR_TYPE_TO_GE_DATATYPE(sampledRes.scalar_type());
    applyTopKTopPMinPInfo.opParamInfo.sampledRes.shape = sampledRes.sizes();

    ApplyTopKTopPMinPTiling applyTopKTopPMinPTiling(&applyTopKTopPMinPInfo);
    TORCH_CHECK(applyTopKTopPMinPTiling.DoTiling() == ge::GRAPH_SUCCESS, "apply_top_k_top_p_min_p DoTiling failed");

    const auto &tilingData = applyTopKTopPMinPTiling.GetTilingData();

    uint32_t tilingSize = (sizeof(ApplyTopKTopPMinPTiling) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto blockDim = tilingData.coreNum;
    static auto tilingBuffer =
        at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(probs.options().device()));
    aclrtMemcpy(tilingBuffer.data_ptr<uint8_t>(), tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    at::Tensor tilingTensor = at::from_blob(tilingBuffer.data_ptr<uint8_t>(), tilingSize, at::kByte);

    auto workspace = at::empty({applyTopKTopPMinPInfo.workspaceSize},
                               at::TensorOptions().dtype(at::kByte).device(probs.options().device()));
    EXEC_KERNEL_CMD(apply_top_k_top_p_min_p, blockDim, probs, k, p, minP, sampledRes, workspace, tilingTensor);
    return sampledRes;
}
}  // namespace npu_kernel
}  // namespace sglang
