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

#ifndef KEY_POOL_IMPL_H
#define KEY_POOL_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

at::Tensor key_pool(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight,
                    const at::Tensor &ape, at::Tensor &state_cache, const at::Tensor &cache_block_table,
                    const at::Tensor &start_pos, const c10::optional<at::Tensor> &norm_weight,
                    const c10::optional<at::Tensor> &norm_bias, const c10::optional<at::Tensor> &cos,
                    const c10::optional<at::Tensor> &sin, const c10::optional<at::Tensor> &cu_seqlens,
                    const c10::optional<at::Tensor> &seqused, int64_t cmp_ratio, double norm_eps, int64_t rotary_mode);

#endif  // KEY_POOL_IMPL_H
