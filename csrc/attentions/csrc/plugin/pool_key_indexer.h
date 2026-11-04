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

#ifndef POOL_KEY_INDEXER_IMPL_H
#define POOL_KEY_INDEXER_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>

at::Tensor pool_key_indexer(const at::Tensor &query, const at::Tensor &pool_key, const at::Tensor &weights,
                            at::IntArrayRef pool_tail_k, const c10::optional<at::IntArrayRef> &actual_seq_q,
                            const c10::optional<at::IntArrayRef> &actual_seq_k,
                            const c10::optional<at::Tensor> &block_table, int64_t topk, int64_t pool_size,
                            std::string layout_q, std::string layout_k, int64_t mask_mode, int64_t quant_mode,
                            bool return_value, int64_t key_stride0);

#endif  // POOL_KEY_INDEXER_IMPL_H
