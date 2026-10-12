/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "pool_key_indexer.h"

#include <cstdint>
#include <string_view>
#include <vector>
#include <torch/library.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"

using namespace at;
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr std::string_view POOL_KEY_INDEXER_OP_NAME = "aclnnPoolKeyIndexer";
}  // namespace

// aclnnPoolKeyIndexer: pool-level TopK over pooled keys, expanded to token
// indices with the incomplete tail pool appended (invalid slots filled -1).
// C prototype (docs/aclnnPoolKeyIndexer.md):
//   GetWorkspaceSize(query, poolKey, weights, poolTailK, actualSeqLengthsQuery,
//       actualSeqLengthsKey, blockTable, qDescale, kDescale, topk, poolSize,
//       layoutQuery, layoutKey, maskMode, quantMode, returnValue, keyStride0,
//       kDescaleStride0, sparseIndicesOut, sparseValuesOut, &workspaceSize, &executor)
// poolTailK / actual_seq_* are aclIntArray (host side): callers must pass host
// values (e.g. derived from seq_lens_cpu), never device tensors.
// quant_mode=-1 (A3): qDescale/kDescale/kDescaleStride0 are unused and passed
// as undefined tensors / -1.
at::Tensor pool_key_indexer(const at::Tensor &query, const at::Tensor &pool_key, const at::Tensor &weights,
                            at::IntArrayRef pool_tail_k, const c10::optional<at::IntArrayRef> &actual_seq_q,
                            const c10::optional<at::IntArrayRef> &actual_seq_k,
                            const c10::optional<at::Tensor> &block_table, int64_t topk, int64_t pool_size,
                            std::string layout_q, std::string layout_k, int64_t mask_mode, int64_t quant_mode,
                            bool return_value, int64_t key_stride0)
{
    TORCH_CHECK(quant_mode == -1, "pool_key_indexer binding only supports quant_mode=-1 (BF16/FP16)");
    const at::Tensor &blockTable = c10::value_or_else(block_table, [] { return at::Tensor(); });
    auto actualSeqQ = actual_seq_q.value_or(at::IntArrayRef{});
    auto actualSeqK = actual_seq_k.value_or(at::IntArrayRef{});

    // sparse_indices: TND -> (T1, topk + pool_size - 1); BSND -> (B, S1, topk + pool_size - 1)
    const int64_t outCols = topk + pool_size - 1;
    at::Tensor sparseIndices;
    at::Tensor sparseValues;
    if (query.dim() == 4) {
        std::vector<int64_t> indiceSize = {query.size(0), query.size(1), outCols};
        sparseIndices = at_npu::native::empty_with_format(indiceSize, query.options().dtype(at::kInt),
                                                          at_npu::native::get_npu_format(query));
        if (return_value) {
            std::vector<int64_t> valueSize = {query.size(0), query.size(1), topk / pool_size};
            sparseValues = at_npu::native::empty_with_format(valueSize, query.options().dtype(at::kFloat),
                                                             at_npu::native::get_npu_format(query));
        }
    } else {
        std::vector<int64_t> indiceSize = {query.size(0), outCols};
        sparseIndices = at_npu::native::empty_with_format(indiceSize, query.options().dtype(at::kInt),
                                                          at_npu::native::get_npu_format(query));
        if (return_value) {
            std::vector<int64_t> valueSize = {query.size(0), topk / pool_size};
            sparseValues = at_npu::native::empty_with_format(valueSize, query.options().dtype(at::kFloat),
                                                             at_npu::native::get_npu_format(query));
        }
    }
    if (!return_value) {
        // returnValue=false requires a (0,) empty tensor for sparseValuesOut.
        std::vector<int64_t> emptySize = {0};
        sparseValues = at_npu::native::empty_with_format(emptySize, query.options().dtype(at::kFloat),
                                                         at_npu::native::get_npu_format(query));
    }

    const char *layoutQPtr = layout_q.c_str();
    const char *layoutKPtr = layout_k.c_str();
    at::Tensor qDescale;  // undefined -> nullptr for optional aclTensor params
    at::Tensor kDescale;
    constexpr int64_t kDescaleStride0Unused = -1;

    EXEC_NPU_CMD<POOL_KEY_INDEXER_OP_NAME>(query, pool_key, weights, pool_tail_k, actualSeqQ, actualSeqK, blockTable,
                                           qDescale, kDescale, topk, pool_size, layoutQPtr, layoutKPtr, mask_mode,
                                           quant_mode, return_value, key_stride0, kDescaleStride0Unused, sparseIndices,
                                           sparseValues);
    return sparseIndices;
}
