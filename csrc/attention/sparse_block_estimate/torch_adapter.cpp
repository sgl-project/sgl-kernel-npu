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

// Torch bridge preserving csrc/attentions/csrc/plugin/sparse_block_estimate.cpp.
// Computation and scheduling remain in the copied legacy CANN implementation.

#include "aclnn_runtime.h"
#include "aclnn_sgl_sparse_block_estimate.h"
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <string>
#include <tuple>
#include <vector>

namespace {
// Match the legacy helper's storage-based descriptors, including views. The
// shared contiguous-buffer helper has a different storage/format convention.
std::shared_ptr<aclTensor> descriptor(const at::Tensor &tensor)
{
    if (!tensor.defined()) return {};
    static constexpr aclDataType types[static_cast<size_t>(at::ScalarType::NumOptions) + 1] = {
        ACL_UINT8,      ACL_INT8,         ACL_INT16,        ACL_INT32,        ACL_INT64,
        ACL_FLOAT16,    ACL_FLOAT,        ACL_DOUBLE,       ACL_DT_UNDEFINED, ACL_COMPLEX64,
        ACL_COMPLEX128, ACL_BOOL,         ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
        ACL_BF16,       ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
    };
    const auto scalarType = static_cast<size_t>(tensor.scalar_type());
    const auto dtype = types[scalarType];
    TORCH_CHECK(dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(tensor.scalar_type())) + " has not been supported");
    const auto itemSize = tensor.itemsize();
    TORCH_CHECK(itemSize != 0, "When ConvertType, tensor item size of cannot be zero.");
    const int64_t storageSize = tensor.storage().nbytes() / itemSize;
    const auto rank = tensor.dim();
    const auto format = rank == 3   ? ACL_FORMAT_NCL
                        : rank == 4 ? ACL_FORMAT_NCHW
                        : rank == 5 ? ACL_FORMAT_NCDHW
                                    : ACL_FORMAT_ND;
    auto *value = aclCreateTensor(tensor.sizes().data(), rank, dtype, tensor.strides().data(), tensor.storage_offset(),
                                  format, &storageSize, 1, const_cast<void *>(tensor.storage().data()));
    return {value, [](aclTensor *p) { aclDestroyTensor(p); }};
}

std::tuple<at::Tensor, at::Tensor> sparse_block_estimate(const at::Tensor &query, const at::Tensor &key,
                                                         c10::OptionalIntArrayRef actual_seq_lengths,
                                                         c10::OptionalIntArrayRef actual_seq_lengths_kv,
                                                         std::string input_layout, int64_t stride, int64_t sparse_size,
                                                         int64_t num_heads, int64_t num_key_value_heads,
                                                         double scale_value, double threshold, bool causal,
                                                         bool keep_sink, bool keep_recent, double row_sparse)
{
    TORCH_CHECK(num_heads != 0, "num_heads must be nonzero.");
    TORCH_CHECK(sparse_size != 0, "sparse_size must be nonzero.");

    // Own list/string data through queued execution. None and [] both create
    // an empty aclIntArray, just as value_or(IntArrayRef{}) did in the plugin.
    auto qLengths = actual_seq_lengths.value_or(at::IntArrayRef{}).vec();
    auto kLengths = actual_seq_lengths_kv.value_or(at::IntArrayRef{}).vec();
    auto layout = std::make_shared<std::string>(input_layout);
    int64_t b, nq, s, d;
    if (input_layout == "BNSD") {
        b = query.size(0);
        nq = query.size(1);
        s = query.size(2);
        d = query.size(3);
    } else if (input_layout == "BSND") {
        b = query.size(0);
        nq = query.size(2);
        s = query.size(1);
        d = query.size(3);
    } else if (input_layout == "BSH") {
        b = query.size(0);
        s = query.size(1);
        d = query.size(2) / num_heads;
        nq = num_heads;
    } else {
        TORCH_CHECK(false, "input_layout only support BNSD, BSND, BSH");
    }
    const int64_t blocks = (s + sparse_size - 1) / sparse_size;
    const int64_t aligned = (blocks + 31) / 32 * 32;
    auto mask = at_npu::native::empty_with_format({b, nq, blocks, aligned}, query.options().dtype(at::kChar),
                                                  at_npu::native::get_npu_format(query));
    auto count = at_npu::native::empty_with_format({b, nq, blocks}, query.options().dtype(at::kInt),
                                                   at_npu::native::get_npu_format(query));

    auto qDesc = descriptor(query), kDesc = descriptor(key);
    auto maskDesc = descriptor(mask), countDesc = descriptor(count);
    auto qArray = std::shared_ptr<aclIntArray>(aclCreateIntArray(qLengths.data(), qLengths.size()), aclDestroyIntArray);
    auto kArray = std::shared_ptr<aclIntArray>(aclCreateIntArray(kLengths.data(), kLengths.size()), aclDestroyIntArray);
    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglSparseBlockEstimateGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseBlockEstimateGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglSparseBlockEstimate)>(
        sgl_cann::op_api_symbol("aclnnSglSparseBlockEstimate"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = queryWorkspace(qDesc.get(), kDesc.get(), qArray.get(), kArray.get(), layout->data(), stride,
                                 sparse_size, num_heads, num_key_value_heads, scale_value, threshold, causal, keep_sink,
                                 keep_recent, row_sparse, maskDesc.get(), countDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "call aclnnSglSparseBlockEstimate failed, detail:", aclGetRecentErrMsg());
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : {query, key, mask, count, workspace})
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [query, key, mask, count, workspace, qDesc, kDesc, maskDesc, countDesc, qArray, kArray, qLengths,
                   kLengths, layout, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "call aclnnSglSparseBlockEstimate failed, detail:", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSparseBlockEstimate", launch);
    return {mask, count};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sparse_block_estimate(Tensor query, Tensor key, int[]? actual_seq_lengths=None, "
        "int[]? actual_seq_lengths_kv=None, str input_layout='BNSD', int stride=8, int sparse_size=128, "
        "int num_heads=1, int num_key_value_heads=1, float scale_value=1, float threshold=1, "
        "bool causal=True, bool keep_sink=True, bool keep_recent=True, float row_sparse=1) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sparse_block_estimate", &sparse_block_estimate);
}
