// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;
// dispatch_quant_mode=0: BF16 activations and BF16 weights, no weight scales.
constexpr int64_t DISPATCH_QUANT_MODE_NO_QUANT = 0;

typedef struct {
    const at::Tensor &tensor_;
    aclDataType dtype;
} TensorWrapper;

typedef struct {
    const at::TensorList &tensor_list_;
    aclDataType dtype;
} TensorListWrapper;

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

aclDataType ConvertToAclDataType(const at::ScalarType &data_type)
{
    int64_t dtype_index = static_cast<int64_t>(data_type);
    TORCH_CHECK(dtype_index >= 0 && dtype_index < static_cast<int64_t>(at::ScalarType::NumOptions) + 1,
                "data_type enum value (", dtype_index, ") is out of range: [0, ",
                static_cast<int64_t>(at::ScalarType::NumOptions), "]")
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[dtype_index];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}
// const std::vector<at::Tensor> &weight1
std::tuple<at::Tensor, at::Tensor> npu_zerc_moe(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_ids, const at::Tensor &topk_weights,
    const std::vector<at::Tensor> &weight1, const std::vector<at::Tensor> &weight2, int64_t moe_expert_num,
    int64_t ep_world_size, int64_t ccl_buffer_size, const c10::optional<std::vector<at::Tensor>> &weight_scales1,
    const c10::optional<std::vector<at::Tensor>> &weight_scales2, const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &scales, int64_t max_recv_token_num, int64_t dispatch_quant_mode,
    int64_t combine_quant_mode, std::string comm_alg, int64_t global_bs,
    c10::optional<int64_t> dispatch_quant_out_dtype, c10::optional<int64_t> weight1_type,
    c10::optional<int64_t> weight2_type, const c10::optional<at::Tensor> &timer,
    const c10::optional<std::vector<at::Tensor>> &shared_weight1,
    const c10::optional<std::vector<at::Tensor>> &shared_weight2,
    const c10::optional<std::vector<at::Tensor>> &shared_weight_scales1,
    const c10::optional<std::vector<at::Tensor>> &shared_weight_scales2)
{
    TORCH_CHECK((ep_world_size > 0), "The ep_world_sizes should be greater than 0, current is: ", ep_world_size);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topk_ids.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    TORCH_CHECK(
        ((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) && (topk_ids.scalar_type() == at::kInt),
        "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");

    at::TensorList weight1_ref = weight1;
    at::TensorList weight2_ref = weight2;

    at::TensorList weight_scales1_ref;
    if (weight_scales1.has_value()) {
        weight_scales1_ref = at::TensorList(weight_scales1.value());
    } else {
        weight_scales1_ref = at::TensorList();
    }
    at::TensorList weight_scales2_ref;
    if (weight_scales2.has_value()) {
        weight_scales2_ref = at::TensorList(weight_scales2.value());
    } else {
        weight_scales2_ref = at::TensorList();
    }
    // The generated ACLNN entry requires both DYNAMIC scale lists to be
    // non-null, but BF16 no-quant weights have no scales. One cached
    // placeholder slot per (dtype, device) satisfies that requirement;
    // neither the list length nor the slot shape is ever read - host tiling
    // skips every scale check in no-quant mode (IsNoQuantMode gates both the
    // null and the shape/dtype validation) and the kernel stores nullptr for
    // the scale addresses under UseBf16NoQuant. Replacing the per-call
    // numExperts-wide expansion also removes 2 * numExperts fake
    // aclCreateTensor conversions per call from ConvertTypes.
    if (dispatch_quant_mode == DISPATCH_QUANT_MODE_NO_QUANT) {
        // 每线程缓存的单个 E8M0 占位张量（torch 对象缓存安全；acl 描述符仍由
        // ConvertTypes 每 call 现造）。BF16 合列表协议下保持单元素：kernel 侧
        // UseBf16NoQuant 不解引用 scale，且无 shared scale 追加。
        static thread_local at::Tensor noQuantScalePlaceholder;
        if (!noQuantScalePlaceholder.defined() || noQuantScalePlaceholder.device() != x.device()) {
            noQuantScalePlaceholder = at::empty({0}, x.options().dtype(at::ScalarType::Float8_e8m0fnu));
        }
        if (weight_scales1_ref.empty()) {
            weight_scales1_ref = at::TensorList(noQuantScalePlaceholder);
        }
        if (weight_scales2_ref.empty()) {
            weight_scales2_ref = at::TensorList(noQuantScalePlaceholder);
        }
    }

    auto x_size = x.sizes();
    auto topk_ids_size = topk_ids.sizes();
    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = topk_ids_size[1];

    int64_t local_moe_expert_num = 1;
    local_moe_expert_num = moe_expert_num / ep_world_size;
    at::Tensor expert_token_nums;
    expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kInt));

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg.c_str());

    int64_t dispatch_quant_result_type = dispatch_quant_out_dtype.value_or(28);

    TORCH_CHECK(!scales.has_value(),
                "MegaMoe scales is not supported; this reserved optional slot is used internally by Timer V2");

    at::Tensor y;
    y = at::empty({bs, h}, topk_ids.options().dtype(x.scalar_type()));

    // 合列表协议：共享专家权重作为 weight1/weight2（及其 scale 列表）的追加
    // 元素传入，个数走 shared_expert_num 属性——原型保持 pre-shared-expert 的
    // 输入数，未启用共享专家的调用在 launch 侧与既有版本转换完全一致。
    // python 层接口不变（仍收 4 个 optional 列表），合并在这里完成。
    int64_t shared_expert_num = 0;
    if (shared_weight1.has_value() && !shared_weight1->empty()) {
        TORCH_CHECK(shared_weight1->size() == 1,
                    "l1_weights_shared must carry exactly one [S, 2*I_shared, H] tensor, got ", shared_weight1->size());
        TORCH_CHECK(shared_weight2.has_value() && shared_weight2->size() == 1,
                    "l2_weights_shared must be supplied together with l1_weights_shared "
                    "(one [S, H, I_shared] tensor).");
        shared_expert_num = (*shared_weight1)[0].sizes()[0];
    }
    at::TensorList weight1_launch = weight1_ref;
    at::TensorList weight2_launch = weight2_ref;
    at::TensorList scales1_launch = weight_scales1_ref;
    at::TensorList scales2_launch = weight_scales2_ref;
    std::vector<at::Tensor> merged_weight1;
    std::vector<at::Tensor> merged_weight2;
    std::vector<at::Tensor> merged_scales1;
    std::vector<at::Tensor> merged_scales2;
    if (shared_expert_num > 0) {
        merged_weight1.reserve(weight1.size() + 1);
        merged_weight1.insert(merged_weight1.end(), weight1.begin(), weight1.end());
        merged_weight1.push_back((*shared_weight1)[0]);
        merged_weight2.reserve(weight2.size() + 1);
        merged_weight2.insert(merged_weight2.end(), weight2.begin(), weight2.end());
        merged_weight2.push_back((*shared_weight2)[0]);
        weight1_launch = at::TensorList(merged_weight1);
        weight2_launch = at::TensorList(merged_weight2);
        // MXFP8：共享 scale 追加到既有 scale 列表；BF16 no-quant 无 scale，
        // 保持占位列表（kernel 侧 UseBf16NoQuant 不解引用）。
        const bool hasSharedScales = shared_weight_scales1.has_value() && !shared_weight_scales1->empty();
        if (!weight_scales1_ref.empty() && hasSharedScales) {
            TORCH_CHECK(shared_weight_scales2.has_value() && !shared_weight_scales2->empty(),
                        "l2_weights_shared_sf must be supplied together with l1_weights_shared_sf.");
            merged_scales1.reserve(weight_scales1_ref.size() + 1);
            merged_scales1.insert(merged_scales1.end(), weight_scales1_ref.begin(), weight_scales1_ref.end());
            merged_scales1.push_back((*shared_weight_scales1)[0]);
            merged_scales2.reserve(weight_scales2_ref.size() + 1);
            merged_scales2.insert(merged_scales2.end(), weight_scales2_ref.begin(), weight_scales2_ref.end());
            merged_scales2.push_back((*shared_weight_scales2)[0]);
            scales1_launch = at::TensorList(merged_scales1);
            scales2_launch = at::TensorList(merged_scales2);
        }
    }

    ACLNN_CMD(aclnnZercMoe, context, x, topk_ids, topk_weights, weight1_launch, weight2_launch, scales1_launch,
              scales2_launch, x_active_mask, timer, moe_expert_num, ep_world_size, ccl_buffer_size, max_recv_token_num,
              dispatch_quant_mode, dispatch_quant_result_type, combine_quant_mode, comm_alg_ptr, global_bs,
              shared_expert_num, y, expert_token_nums);

    return std::tie(y, expert_token_nums);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_zerc_moe", &npu_zerc_moe, "npu_zerc_moe");
}

}  // namespace op_api
