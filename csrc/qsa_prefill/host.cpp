// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "../sparse_attention_score/op_host/tiling/sparse_attention_score_tiling_data.h"
#include "safe_launch.h"
#include "aclrtlaunch_qsa_prefill_runs_kernel.h"
#include "aclrtlaunch_qsa_prefill_prepare_kernel.h"

namespace sglang::npu_kernel {
namespace {
void CheckTensor(const at::Tensor &t, const at::Tensor &q, at::ScalarType dtype)
{
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == dtype && t.is_contiguous(),
                "QSA tensors must have the expected dtype, be contiguous, and share a device");
}
}  // namespace
std::tuple<at::Tensor, at::Tensor> QsaRuns(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                           const at::Tensor &slots, const at::Tensor &runs, const at::Tensor &counts,
                                           double scale)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
    c10_npu::NPUGuard guard(q.device());
    CheckTensor(q, q, at::kBFloat16);
    CheckTensor(k, q, at::kBFloat16);
    CheckTensor(v, q, at::kBFloat16);
    CheckTensor(slots, q, at::kInt);
    CheckTensor(runs, q, at::kInt);
    CheckTensor(counts, q, at::kInt);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 256 && q.size(2) == 256, "packed Q must be [G,256,256]");
    TORCH_CHECK(k.dim() == 3 && k.size(0) > 0 && k.size(1) == 2 && k.size(2) == 256 && v.sizes() == k.sizes(),
                "K,V must be [capacity,2,256]");
    TORCH_CHECK(slots.dim() == 2 && slots.size(1) == 2064 && (slots.size(0) + 15) / 16 == q.size(0),
                "prepared slots must be [R,2064]");
    TORCH_CHECK(runs.sizes() == slots.sizes() && counts.dim() == 1 && counts.numel() == slots.size(0) * 16,
                "prepared metadata shape mismatch");
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "scale must be positive and finite");
    TORCH_CHECK(q.size(0) <= INT32_MAX / 2 && k.size(0) <= INT32_MAX, "tensor extent exceeds int32 indexing");
    auto out = at::zeros(q.sizes(), q.options().dtype(at::kFloat));
    auto lse = at::full({q.size(0), 256}, -INFINITY, q.options().dtype(at::kFloat));
    if (!q.size(0)) return {out, lse};
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(static_cast<uint32_t>(platform->GetSocVersion()) != 4, "QSA prefill requires arch22 hardware");
    const uint32_t cores =
        std::min(static_cast<uint32_t>(q.size(0) * 2), static_cast<uint32_t>(platform->GetCoreNumAic()));
    using Tiling = sglang::SAHost::SATilingData;
    Tiling t{};
    t.batch = 1;
    t.numHeads = 256;
    t.kvHeads = 2;
    t.embeddingSize = 256;
    t.blockSize = 256;
    t.topK = 2064;
    t.maxBlocksPerBatch = k.size(0);
    t.totalQTokens = q.size(0);
    t.totalTaskNum = q.size(0) * 2;
    t.firstBatchTaskNum = 2;
    t.scaleValue = scale;
    t.maxQSeqlen = slots.size(0);
    t.tilingKey = 20002;
    t.groupSize = 128;
    t.mm1OutSize = static_cast<uint64_t>(cores) * 32768 * 4 * 3;
    t.smOnlineOutSize = static_cast<uint64_t>(cores) * 32768 * 2 * 3;
    t.mm2OutSize = t.mm1OutSize;
    t.updateSize = t.mm1OutSize;
    const uint64_t libSize = platform->GetLibApiWorkSpaceSize();
    t.workSpaceSize =
        libSize + ((2064 * 4 + 511) / 512) * 512 + t.mm1OutSize + t.smOnlineOutSize + t.mm2OutSize + t.updateSize;
    t.qBaseTile = 64;
    t.kvBaseTile = 256;
    t.mm1L1TileM = 64;
    t.mm1L1TileN = 256;
    t.mm1L1TileKLeft = 256;
    t.mm1L1TileKRight = 256;
    t.mm2L1TileM = 64;
    t.mm2L1TileN = 256;
    t.mm2L1TileKLeft = 256;
    t.mm2L1TileKRight = 256;
    t.qL1BufNum = 1;
    t.kL1BufNum = 1;
    t.vL1BufNum = 1;
    t.pL1BufNum = 3;
    // Cache immutable tiling by full values and device, never by a hash alone.
    // Graph callers warm the shape first; replay performs no host copy.
    using Key = std::tuple<int, int64_t, int64_t, int64_t, double>;
    static std::map<Key, at::Tensor> cache;
    static std::mutex mutex;
    at::Tensor tiling;
    {
        std::lock_guard<std::mutex> lock(mutex);
        Key key{q.get_device(), q.size(0), k.size(0), slots.size(0), scale};
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto buffer = at::empty({static_cast<int64_t>(sizeof(t))}, q.options().dtype(at::kByte));
            auto status = aclrtMemcpy(buffer.data_ptr(), sizeof(t), &t, sizeof(t), ACL_MEMCPY_HOST_TO_DEVICE);
            TORCH_CHECK(status == ACL_SUCCESS, "QSA tiling copy failed: ", status);
            it = cache.emplace(key, buffer).first;
        }
        tiling = it->second;
    }
    auto workspace = at::zeros({static_cast<int64_t>(t.workSpaceSize)}, q.options().dtype(at::kByte));
    QSA_EXEC_KERNEL_CMD(qsa_prefill_runs_kernel, cores, q, k, v, slots, runs, counts, out, lse, workspace, tiling);
    return {out, lse};
}

std::vector<at::Tensor> QsaPrepare(const at::Tensor &b, const at::Tensor &table, const at::Tensor &req,
                                   const at::Tensor &flags, int64_t length, int64_t base, int64_t capacity)
{
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
    c10_npu::NPUGuard guard(b.device());
    TORCH_CHECK(b.dim() == 2 && b.size(1) == 512 && b.scalar_type() == at::kInt && b.is_contiguous(),
                "contiguous int32 [R,512] blocks required");
    TORCH_CHECK(table.dim() == 2 && table.scalar_type() == at::kInt && table.size(0) > 0 && table.stride(1) == 1 &&
                    table.stride(0) <= INT32_MAX && table.device() == b.device(),
                "int32 unit-stride request table required");
    TORCH_CHECK(req.dim() == 1 && req.numel() == 1 && req.scalar_type() == at::kLong && req.is_contiguous() &&
                    req.device() == b.device(),
                "one int64 request row required");
    TORCH_CHECK(length >= 0 && length <= table.size(1) && length <= INT32_MAX && base >= INT32_MIN &&
                    base + b.size(0) < INT32_MAX && capacity > 0 && capacity <= INT32_MAX,
                "invalid sequence/capacity range");
    TORCH_CHECK(flags.dim() == 1 && flags.numel() == b.size(0) * 16 && flags.scalar_type() == at::kInt &&
                    flags.is_contiguous() && flags.device() == b.device(),
                "contiguous int32 stride16 exact-sharing flags required");
    TORCH_CHECK(b.size(0) <= INT32_MAX / 16, "too many query rows");
    uint32_t rows = b.size(0), blocks = 512, tableStride = table.stride(0), n = length, cap = capacity, outWidth = 2064;
    int32_t start = base;
    auto s = at::empty({rows, outWidth}, b.options()), r = at::empty_like(s),
         c = at::empty({static_cast<int64_t>(rows) * 16}, b.options());
    uint32_t launchBlocks = std::min((rows + 15) / 16, 64u);
    if (rows)
        QSA_EXEC_KERNEL_CMD(qsa_prefill_prepare_kernel, launchBlocks, b, table, req, flags, s, r, c, rows, blocks,
                            tableStride, n, start, cap, outWidth);
    return {s, r, c};
}

}  // namespace sglang::npu_kernel

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "qsa_prefill_prepare(Tensor blocks, Tensor table, Tensor request_row, Tensor flags, int length, int base, int "
        "capacity) -> Tensor[]");
    m.def(
        "qsa_prefill_runs(Tensor q, Tensor k, Tensor v, Tensor slots, Tensor runs, Tensor counts, float scale) -> "
        "(Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("qsa_prefill_prepare", &sglang::npu_kernel::QsaPrepare);
    m.impl("qsa_prefill_runs", &sglang::npu_kernel::QsaRuns);
}
