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

#ifndef LA_MINDIE_SD_IMPL_H
#define LA_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

struct AscendLaserAttentionTilingData {
    int32_t batchSize;       // B
    int32_t headNum;         // N
    int32_t seqSize;         // S
    int32_t headDim;         // D
    int32_t coreNumPerGroup; // Y
    int32_t coreGroupNum;    // F

    int32_t qSeqLength;      // qkv???
    int32_t kSeqLength;      // qkv???
    int32_t vSeqLength;      // qkv???
    int32_t maskSeqLength;   // ??
    float scale;             // ??
    float keep_prob;         // ??
    int32_t pre_tokens;      // ??
    int32_t next_tokens;     // ??

    bool isTriangle;        // ?????
    int32_t attenType;       // 0:MHA/1:GQA
    int32_t sparseMode;      // 0:dense/1:sparse
    int32_t headGroupSize;   // N/G
    int32_t windowLen;       // sparse?????
    bool isHighPrecision;    // ???
};

namespace sglang {
namespace npu_kernel {

std::tuple<at::Tensor, at::Tensor> la(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask_opt,
    const c10::optional<at::Tensor> &alibi_mask_opt,
    const c10::optional<at::Tensor> &drop_mask_opt,
    double scale_value, int64_t head_num, std::string input_layout,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool is_highPrecision);

}
}

#endif // LA_MINDIE_SD_IMPL_H
