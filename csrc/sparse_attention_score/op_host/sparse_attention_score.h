/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
 * full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_HOST_H
#define SPARSE_ATTENTION_SCORE_HOST_H

#include <ATen/ATen.h>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

// Native npu_sparse_attention_score: routes the decode/verify main attention through
// the aclnn cube kernel statically linked into libsgl_kernel_npu.so (no PTA plugin,
// no ASCEND_CUSTOM_OPP_PATH). Query layout is TND [T,N,D]; KV is blocked
// [blockNum, blockSize, kvHead, D]; select_idx [kvHead, maxQSeqlen, topK].
// Returns attentionOut [T,N,D] (fp8 input -> fp16 output, else same as query);
// softmaxLse [T,N,1] fp32 is allocated internally for the kernel to write but not
// returned (sglang consumes only attentionOut).
HOST_API at::Tensor sparse_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &select_idx,
    const at::Tensor &block_table, const c10::optional<at::Tensor> &select_num_idx,
    const c10::optional<at::Tensor> &q_dequant_scale, const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale, const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv, int64_t num_key_value_heads, double scale_value,
    int64_t block_size, int64_t top_k, int64_t inner_precise);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SPARSE_ATTENTION_SCORE_HOST_H
