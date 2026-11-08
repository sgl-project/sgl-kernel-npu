/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "key_pool.h"

#include <algorithm>
#include <string_view>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"

namespace {
constexpr std::string_view KEY_POOL_OP_NAME = "aclnnKeyPool";
}

at::Tensor key_pool(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight,
                    const at::Tensor &ape, at::Tensor &state_cache, const at::Tensor &cache_block_table,
                    const at::Tensor &start_pos, const c10::optional<at::Tensor> &norm_weight,
                    const c10::optional<at::Tensor> &norm_bias, const c10::optional<at::Tensor> &cos,
                    const c10::optional<at::Tensor> &sin, const c10::optional<at::Tensor> &cu_seqlens,
                    const c10::optional<at::Tensor> &seqused, int64_t cmp_ratio, double norm_eps, int64_t rotary_mode)
{
    TORCH_CHECK(hidden_states.dim() == 2 || hidden_states.dim() == 3, "key_pool: hidden_states must be TH or BSH");
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16 || hidden_states.scalar_type() == at::kHalf,
                "key_pool: hidden_states must be BF16/FP16");
    TORCH_CHECK(cmp_ratio >= 2 && cmp_ratio <= 128 && (cmp_ratio & (cmp_ratio - 1)) == 0,
                "key_pool: cmp_ratio must be a power of two in [2,128]");
    TORCH_CHECK(norm_eps > 0 && (rotary_mode == 0 || rotary_mode == 1), "key_pool: invalid epsilon/rotary_mode");
    TORCH_CHECK(!cos.has_value() && !sin.has_value() && !seqused.has_value(),
                "key_pool: RoPE and seqused are not supported");
    auto check = [&](const at::Tensor &t, at::ScalarType dtype, const char *name, bool contiguous = true) {
        TORCH_CHECK(t.device() == hidden_states.device() && t.scalar_type() == dtype,
                    "key_pool: invalid device/dtype for ", name);
        TORCH_CHECK(!contiguous || t.is_contiguous(), "key_pool: ", name, " must be contiguous");
        TORCH_CHECK(at_npu::native::get_npu_format(t) == 2, "key_pool: ", name, " must use ND format");
    };
    check(hidden_states, hidden_states.scalar_type(), "hidden_states");
    check(wk, hidden_states.scalar_type(), "wk");
    check(gate_weight, hidden_states.scalar_type(), "gate_weight");
    TORCH_CHECK(wk.dim() == 2 && wk.size(1) == hidden_states.size(-1) && gate_weight.sizes() == wk.sizes(),
                "key_pool: wk/gate_weight must be [D,H]");
    const auto width = wk.size(0);
    check(ape, at::kFloat, "ape");
    TORCH_CHECK(ape.dim() == 2 && ape.size(0) == cmp_ratio && ape.size(1) == width, "key_pool: invalid ape shape");
    check(state_cache, at::kFloat, "state_cache", false);
    TORCH_CHECK(state_cache.dim() == 3 && state_cache.size(2) == 2 * width && state_cache.stride(2) == 1 &&
                    state_cache.stride(1) == 2 * width && state_cache.stride(0) >= state_cache.size(1) * 2 * width,
                "key_pool: invalid state_cache shape/stride");
    check(cache_block_table, at::kInt, "cache_block_table");
    check(start_pos, at::kInt, "start_pos");
    TORCH_CHECK(cache_block_table.dim() == 2 && start_pos.dim() == 1 && start_pos.size(0) == cache_block_table.size(0),
                "key_pool: invalid request metadata shape");
    const auto batch = cache_block_table.size(0);
    TORCH_CHECK(norm_weight.has_value() == norm_bias.has_value(), "key_pool: normalization requires weight and bias");
    if (norm_weight.has_value()) {
        check(*norm_weight, at::kFloat, "norm_weight");
        check(*norm_bias, at::kFloat, "norm_bias");
        TORCH_CHECK(norm_weight->dim() == 1 && norm_bias->dim() == 1 && norm_weight->numel() == width &&
                        norm_bias->numel() == width,
                    "key_pool: invalid norm shape");
    }
    TORCH_CHECK((hidden_states.dim() == 2) == cu_seqlens.has_value(),
                "key_pool: TH requires cu_seqlens; BSH forbids it");
    at::Tensor pooled;
    if (hidden_states.dim() == 2) {
        check(*cu_seqlens, at::kInt, "cu_seqlens");
        TORCH_CHECK(cu_seqlens->dim() == 1 && cu_seqlens->numel() == batch + 1, "key_pool: invalid cu_seqlens shape");
        const auto capacity = std::min(hidden_states.size(0), hidden_states.size(0) / cmp_ratio + batch);
        pooled = at::empty({capacity, width}, hidden_states.options());
    } else {
        TORCH_CHECK(hidden_states.size(0) == batch, "key_pool: inconsistent batch size");
        pooled =
            at::empty({batch, (hidden_states.size(1) + cmp_ratio - 1) / cmp_ratio, width}, hidden_states.options());
    }
    // Device metadata is passed directly, including during NPUGraph replay.
    const auto state_cache_stride = state_cache.stride(0);
    EXEC_NPU_CMD<KEY_POOL_OP_NAME>(hidden_states, wk, gate_weight, ape, state_cache, cache_block_table, start_pos,
                                   norm_weight, norm_bias, cos, sin, cu_seqlens, seqused, cmp_ratio, norm_eps,
                                   rotary_mode, state_cache_stride, pooled);
    return pooled;
}
