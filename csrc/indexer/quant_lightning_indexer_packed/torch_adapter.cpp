// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include "aclnn_runtime.h"
#include "aclnn_sgl_quant_lightning_indexer_packed.h"
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <vector>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

// Exported by torch_npu. These preserve workspace/graph lifetime without
// capturing Tensor owners in the task-queue callback.
TORCH_NPU_API void *GetWorkSpaceAddr(uint64_t workspace_size);
TORCH_NPU_API void InitExecSubTheadCtx(aclrtStream acl_stream);

namespace {
at::Tensor quant_lightning_indexer_packed(const at::Tensor &query, const at::Tensor &packed_cache,
                                          const at::Tensor &weights, const at::Tensor &query_scale,
                                          const at::Tensor &seq_q, const at::Tensor &seq_k,
                                          const at::Tensor &block_table, int64_t topk, int64_t sparse_mode) {
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on NPU");
    const c10::OptionalDeviceGuard guard(query.device());
    std::vector<at::Tensor> keep_alive{query, packed_cache, weights, query_scale, seq_q, seq_k, block_table};
    for (const auto &tensor : keep_alive) {
        TORCH_CHECK(tensor.device() == query.device(), "all inputs must be on the same NPU device");
        TORCH_CHECK(tensor.is_contiguous(), "all inputs must be contiguous");
        TORCH_CHECK(at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND, "all inputs must use ND format");
    }
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn, "query must be float8_e4m3fn");
    TORCH_CHECK(query.dim() == 3 && query.size(2) == 128, "query must have TND shape [T, heads, 128]");
    TORCH_CHECK(packed_cache.scalar_type() == at::kByte && packed_cache.dim() == 2,
                "packed_cache must be uint8 [pages, P*132]");
    TORCH_CHECK(packed_cache.size(0) > 0 && packed_cache.size(1) % 132 == 0,
                "packed_cache must have positive pages and width P*132");
    auto page = packed_cache.size(1) / 132;
    TORCH_CHECK(page >= 16 && page <= 1024 && page % 16 == 0, "page size must be a multiple of 16 in [16, 1024]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed_cache.data_ptr()) % 4 == 0, "packed_cache must be FP32 aligned");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be float32");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == query.size(0) &&
                    weights.size(1) == query.size(1) && weights.sizes() == query_scale.sizes(),
                "weights/query_scale must have shape [T, heads]");
    TORCH_CHECK(seq_q.scalar_type() == at::kInt && seq_k.scalar_type() == at::kInt &&
                    block_table.scalar_type() == at::kInt,
                "sequence lengths and block_table must be int32");
    TORCH_CHECK(seq_q.dim() == 1 && seq_k.dim() == 1 && seq_q.numel() == seq_k.numel(),
                "seq_q/seq_k must have shape [batch]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == seq_k.numel(),
                "block_table must have shape [batch, max_pages]");
    TORCH_CHECK(topk > 0 && topk <= 2048, "topk must be in [1, 2048]");
    TORCH_CHECK(sparse_mode == 0 || sparse_mode == 3, "sparse_mode must be 0 or 3");
    const auto heads = query.size(1);
    TORCH_CHECK(heads == 8 || heads == 16 || heads == 24 || heads == 32 || heads == 64,
                "query head count must be 8, 16, 24, 32 or 64");
    TORCH_CHECK(seq_q.numel() > 0, "batch must be positive");
    TORCH_CHECK(block_table.size(1) > 0, "block_table must have positive width");
    constexpr auto maxDim = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(query.size(0) <= maxDim && seq_q.numel() <= maxDim && block_table.size(1) <= maxDim / page,
                "shape exceeds uint32 tiling limits");
    auto output = at::empty({query.size(0), 1, topk}, query.options().dtype(at::kInt));
    if (query.size(0) == 0)
        return output;

    static auto prepare = reinterpret_cast<decltype(&aclnnSglQuantLightningIndexerPackedGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglQuantLightningIndexerPackedGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglQuantLightningIndexerPacked)>(
        sgl_cann::op_api_symbol("aclnnSglQuantLightningIndexerPacked"));
    auto q = sgl_cann::tensor_descriptor(query, ACL_FLOAT8_E4M3FN);
    auto k = sgl_cann::tensor_descriptor(packed_cache, ACL_UINT8);
    auto w = sgl_cann::tensor_descriptor(weights, ACL_BF16);
    auto qs = sgl_cann::tensor_descriptor(query_scale, ACL_FLOAT);
    auto sq = sgl_cann::tensor_descriptor(seq_q, ACL_INT32);
    auto sk = sgl_cann::tensor_descriptor(seq_k, ACL_INT32);
    auto table = sgl_cann::tensor_descriptor(block_table, ACL_INT32);
    auto out = sgl_cann::tensor_descriptor(output, ACL_INT32);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = prepare(q.get(), k.get(), w.get(), qs.get(), sq.get(), sk.get(), table.get(), topk, sparse_mode,
                          out.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Packed QLI workspace query failed: ", status, "; ", aclGetRecentErrMsg());
    auto stream = c10_npu::getCurrentNPUStream();
    auto aclStream = stream.stream(false);
    void *workspace = GetWorkSpaceAddr(workspaceSize);
    keep_alive.push_back(output);
    for (auto &tensor : keep_alive) {
        if (tensor.numel())
            tensor.record_stream(stream);
    }
    auto launch = [q, k, w, qs, sq, sk, table, out, workspace, workspaceSize, executor, aclStream]() -> int {
        InitExecSubTheadCtx(aclStream);
        auto result = execute(workspace, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "Packed QLI execution failed: ", result, "; ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApiV2("SglQuantLightningIndexerPacked", launch);
    return output;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("sgl_quant_lightning_indexer_packed(Tensor query, Tensor packed_cache, Tensor weights, "
          "Tensor query_scale, Tensor seq_q, Tensor seq_k, Tensor block_table, "
          "int topk=2048, int sparse_mode=3) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("sgl_quant_lightning_indexer_packed", &quant_lightning_indexer_packed);
}
