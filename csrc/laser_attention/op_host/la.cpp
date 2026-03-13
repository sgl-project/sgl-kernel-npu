/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "la.h"

#include <stdexcept>
#include "defines.h"
#include "common.h"
#include "torch_helper.h"
#include "aclrtlaunch_ascend_laser_attention.h"

#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel {

std::tuple<at::Tensor, at::Tensor> la(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask_opt,
    const c10::optional<at::Tensor> &alibi_mask_opt,
    const c10::optional<at::Tensor> &drop_mask_opt,
    double scale_value, int64_t head_num, std::string input_layout,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool is_highPrecision)
{
    size_t query_dim = query.sizes().size();
    if (query_dim != 4) { // 4 is the first input dimension
        throw std::invalid_argument("The first input dimension of la must be 4 but got " + at::str(query_dim));
    }
    const at::Tensor& atten_mask = c10::value_or_else(atten_mask_opt, [] {return at::Tensor();});
    const at::Tensor& alibi_mask = c10::value_or_else(alibi_mask_opt, [] {return at::Tensor();});
    const at::Tensor& drop_mask = c10::value_or_else(drop_mask_opt, [] {return at::Tensor();});

    at::Tensor softmax_log_max_sum =
        at_npu::native::empty_with_format({query.sizes()[0], query.sizes()[1], query.sizes()[2]},
        query.options(), at_npu::native::get_npu_format(query));

    at::Tensor attention_out =
        at_npu::native::empty_with_format(query.sizes(), query.options(),
        at_npu::native::get_npu_format(query));

    at_npu::native::OpCommand cmd;

    /*cmd.Name("AscendLaserAttention")
            .Input(query, "query")
            .Input(key, "key")
            .Input(value, "value");

    if (atten_mask.defined()) {
        cmd.Input(atten_mask, "atten_mask");
    }

    if (alibi_mask.defined()) {
        cmd.Input(alibi_mask, "alibi_mask");
    }

    if (drop_mask.defined()) {
        cmd.Input(drop_mask, "drop_mask");
    }

    cmd.Output(softmax_log_max_sum, "softmax_log_max_sum")
    .Output(attention_out, "attention_out")
    .Attr("scale_value", static_cast<float>(scale_value))
    .Attr("head_num", head_num)
    .Attr("input_layout", input_layout)
    .Attr("keep_prob", static_cast<float>(keep_prob))
    .Attr("pre_tokens", pre_tokens)
    .Attr("next_tokens", next_tokens)
    .Attr("is_highPrecision", is_highPrecision)
    .Run();*/
    
    
    
    auto tilingBuffer = at::empty({sizeof(AscendLaserAttentionTilingData)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    AscendLaserAttentionTilingData *tilingData = reinterpret_cast<AscendLaserAttentionTilingData *>(tilingBuffer.data_ptr());
    
    tilingData->batchSize = query.size(0);
    tilingData->headNum = head_num;
    tilingData->seqSize = query.size(2);
    tilingData->headDim = query.size(3);

    tilingData->qSeqLength = query.size(2);
    tilingData->kSeqLength = key.size(2);
    tilingData->vSeqLength = value.size(2);
    tilingData->maskSeqLength = 0;
    tilingData->scale = scale_value;
    tilingData->isTriangle = false;
    
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();

    auto col_size = key.size(2);

    int32_t coreNumPerGroup = 1;
    int32_t factor = 2;
    if (col_size <= 8 * 1024 / factor) {    // value is 8 * 1024
        coreNumPerGroup = 1;
    } else if (col_size > 8 * 1024 / factor && col_size <= 16 * 1024 / factor) {    // value is 8?16?1024
        coreNumPerGroup = 2;    // 2 is coreNumPerGroup
    } else if (col_size > 16 * 1024 / factor && col_size <= 32 * 1024 / factor) {    // value is 16?32?1024
        coreNumPerGroup = 4;    // 4 is coreNumPerGroup
    } else {
        if (aicNum == 20) {    // 20 is aicNum
            coreNumPerGroup = 4;    // 4 is coreNumPerGroup
        } else {
            coreNumPerGroup = 8;    // 8 is coreNumPerGroup
        }
    }
    
    int32_t groupNum;
    groupNum = aicNum / coreNumPerGroup;
    
    auto rowSumSize = query.size(0) * head_num * query.size(2) * sizeof(float);
    
    size_t workspace_size = groupNum * 128 * 128 * 32 * 2 * 4 +    // 128?32?2?4 is offset
                    groupNum * 256 * 128 * 8 * 2 * 4 * 2 +    // 256?128?8?2?4 is offset
                    rowSumSize;
    
    void *ptr = 0;
    auto workspace_tensor = at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    
    EXEC_KERNEL_CMD(ascend_laser_attention, groupNum, query, key, value, ptr, ptr, ptr, softmax_log_max_sum, attention_out, workspace_tensor, tilingBuffer);
    
    return std::tuple<at::Tensor, at::Tensor>(softmax_log_max_sum, attention_out);
}

}  // namespace npu_kernel
}  // namespace sglang

