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

#include "sparse_block_estimate_tiling.h"
#include "sparse_attention_host.h"
#include "aclrtlaunch_sparse_block_estimate.h"
#include "defines.h"

#include <algorithm>
#include <vector>

namespace optiling {
void PromptFlashAttentionSplitNSNew(SparseBlockEstimateTilingData &tiling, uint32_t curCoreNum,
                                    std::vector<int64_t> &actualSeqLengths, std::vector<int64_t> &actualSeqLengthsKV,
                                    int64_t actualSharedPrefixLen, bool useBalanceTiling)
{
    SparseBlockEstimateSeqParams *seqParams = &tiling.sparseBlockEstimateSeqParams;

    uint32_t arrayLen = tiling.get_batchSize();  // batch size
    uint32_t sOuterSize = tiling.get_sOuterFactor() * tiling.get_stride();
    uint32_t sInnerSize = tiling.get_sInnerFactor() * tiling.get_stride();

    std::vector<uint32_t> accumSOuterTilingNums(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sInnerLoopTimes(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sOuterBlockNums(static_cast<size_t>(arrayLen), 0U);

    const size_t tilingElementArrayLen =
        (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);

    int64_t totalBlockWight = 0;
    int totalOuterBlockNum = 0;
    uint32_t preAccumSOuterNum = 0U;
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    uint32_t sInnerPrefixLoopTimes = (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
    bool isSOuterNoTail = true;
    bool isSInnerNoTail = true;
    bool causal = tiling.get_causal();
    for (uint32_t i = 0; i < arrayLen; i++) {
        int seqLen = actualSeqLengths[i];
        int subSeqInnerLen = actualSeqLengthsKV[i];
        sOuterBlockNums[i] = (seqLen + sOuterSize - 1) / sOuterSize;
        sInnerLoopTimes[i] = (subSeqInnerLen + sInnerSize - 1) / sInnerSize + sInnerPrefixLoopTimes;
        accumSOuterTilingNums[i] = (sOuterBlockNums[i] * tiling.get_headNumQ()) + preAccumSOuterNum;
        preAccumSOuterNum = accumSOuterTilingNums[i];

        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[i]);

        if (seqLen % sOuterSize != 0) {
            isSOuterNoTail = false;
        }
        if (subSeqInnerLen % sInnerSize != 0) {
            isSInnerNoTail = false;
        }
        totalOuterBlockNum += sOuterBlockNums[i];
        if (causal) {
            totalBlockWight +=
                (static_cast<int64_t>(sOuterBlockNums[i]) + 1) * static_cast<int64_t>(sOuterBlockNums[i]) / 2;  // div 2
        } else {
            totalBlockWight += static_cast<int64_t>(sOuterBlockNums[i]) * static_cast<int64_t>(sInnerLoopTimes[i]);
        }
    }
    if ((!useBalanceTiling)) {
        accumSOuterTilingNums[0] = 0;
    }

    float coreWightTarget = (float(totalBlockWight * tiling.get_headNumQ()) / float(curCoreNum));

    int curWight = 0;
    int curCore = 0;
    coreSposStart[curCore] = 0;
    coreSidStart[curCore] = 0;
    coreNidStart[curCore] = 0;
    for (uint32_t i = 0; i < tiling.get_headNumQ(); i++) {
        for (uint32_t j = 0; j < arrayLen; j++) {
            for (uint32_t k = 0; k < sOuterBlockNums[j]; k++) {
                int64_t dif = int64_t(coreWightTarget * float(curCore + 1)) - curWight;
                int64_t curWightPlus;
                if (causal) {
                    curWightPlus = k + 1;
                } else {
                    curWightPlus = sInnerLoopTimes[j];
                }
                if ((curWightPlus - dif) > dif) {
                    if (k == 0) {
                        if (j == 0) {
                            coreNidEnd[curCore] = i;
                            coreSidEnd[curCore] = arrayLen;
                            coreSposEnd[curCore] = sOuterBlockNums[arrayLen - 1];
                        } else {
                            coreNidEnd[curCore] = i + 1;
                            coreSidEnd[curCore] = j;
                            coreSposEnd[curCore] = sOuterBlockNums[j - 1];
                        }
                    } else {
                        coreNidEnd[curCore] = i + 1;
                        coreSidEnd[curCore] = j + 1;
                        coreSposEnd[curCore] = k;
                    }
                    curCore += 1;
                    coreNidStart[curCore] = i;
                    coreSidStart[curCore] = j;
                    coreSposStart[curCore] = k;
                }
                curWight += curWightPlus;
            }
        }
    }

    coreNidEnd[curCore] = (tiling.get_headNumQ());
    coreSidEnd[curCore] = arrayLen;
    coreSposEnd[curCore] = sOuterBlockNums[arrayLen - 1];

    // Temporary reuse
    seqParams->set_coreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());

    uint32_t actualCoreNums = curCore + 1;
    tiling.set_actualCoreNums(actualCoreNums);
}

}  // namespace optiling

namespace sglang::npu_kernel {

HOST_API std::tuple<at::Tensor, at::Tensor>
sparse_block_estimate(const at::Tensor &query, const at::Tensor &key, c10::OptionalIntArrayRef actual_seq_lengths,
                      c10::OptionalIntArrayRef actual_seq_lengths_kv, std::string input_layout, int64_t stride,
                      int64_t sparse_size, int64_t num_heads, int64_t num_key_value_heads, double scale_value,
                      double threshold, bool causal, bool keep_sink, bool keep_recent, double row_sparse)
{
    namespace sa = sparse_attention;
    using namespace matmul_tiling;
    auto platform = sa::platform();
    auto qs = sa::check_input(query, input_layout, num_heads);
    auto ks = sa::check_input(key, input_layout, num_key_value_heads);
    sa::check_pair(query, key, qs, ks);
    sa::checked_int(stride, "stride");
    sa::checked_int(sparse_size, "sparse_size");
    TORCH_CHECK(sparse_size >= 128 && sparse_size <= 512 && sparse_size % 128 == 0 && sparse_size % stride == 0,
                "sparse_size must be a multiple of 128 in [128, 512] and divisible by stride");
    TORCH_CHECK(std::isfinite(scale_value) && std::isfinite(threshold) && std::isfinite(row_sparse),
                "scale_value, threshold and row_sparse must be finite");
    auto qLengths = sa::check_lengths(actual_seq_lengths, qs);
    auto kvLengths = sa::check_lengths(actual_seq_lengths_kv, ks);
    auto q = sa::normalize(query, input_layout);
    auto k = sa::normalize(key, input_layout);
    if (input_layout == "BSND") input_layout = "BSH";
    const uint32_t coresAic = platform->GetCoreNumAic();
    const uint32_t coresAiv = platform->GetCoreNumAiv();
    optiling::SparseBlockEstimateTilingData tiling;
    tiling.set_headNumQ(qs.heads);
    tiling.set_headNumKV(ks.heads);
    tiling.set_seqLenQ(qs.sequence);
    tiling.set_seqLenK(ks.sequence);
    tiling.set_dim(qs.dim);
    tiling.set_batchSize(qs.batch);
    tiling.set_actualSeqLengthsSize(qLengths.size());
    tiling.set_actualSeqLengthsKVSize(kvLengths.size());
    tiling.set_sparseSize(sparse_size);
    tiling.set_scaleFactor(scale_value);
    tiling.set_threshold(threshold);
    tiling.set_causal(causal);
    tiling.set_setFirstCol(keep_sink);
    tiling.set_setDiag(keep_recent);
    tiling.set_rowSparse(row_sparse);
    tiling.set_stride(stride);
    // Per-call factors: the old globals made results depend on previous calls.
    int32_t nBase = 1024;
    while (ks.sequence < stride * 2 * nBase && nBase > 512)
        nBase /= 2;
    int64_t tasks = (qs.heads * qs.sequence + coresAiv - 1) / coresAiv;
    tasks = (tasks + sparse_size - 1) / sparse_size;
    int32_t mBase = std::min<int64_t>(128, tasks * sparse_size / stride);
    TORCH_CHECK(mBase > 0, "invalid sparse block estimate tile size");
    tiling.set_sOuterFactor(mBase);
    tiling.set_sInnerFactor(nBase);
    MatmulApiTiling cubeTiling(*platform);
    auto mmType = query.scalar_type() == at::kHalf ? DataType::DT_FLOAT16 : DataType::DT_BF16;
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, mmType, false);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, mmType, true);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND_ALIGN, DataType::DT_FLOAT);
    cubeTiling.SetOrgShape((qs.sequence + stride - 1) / stride, (ks.sequence + stride - 1) / stride,
                           sa::checked_int(qs.dim * stride, "dim * stride"));
    cubeTiling.SetShape(mBase, nBase, qs.dim * stride);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    cubeTiling.SetFixSplit(std::min(mBase, 128), std::min(nBase, 128), 128);
    TORCH_CHECK(cubeTiling.GetTiling(tiling.cubeTilingData) != -1, "SparseBlockEstimate matmul tiling failed");
    tiling.cubeTilingData.set_dbL0C(2);
    tiling.set_coreNumAic(coresAic);
    std::vector<int64_t> actualQ(qs.batch, qs.sequence), actualKV(ks.batch, ks.sequence);
    for (size_t i = 0; i < actualQ.size(); ++i) {
        if (!qLengths.empty()) actualQ[i] = qLengths[qLengths.size() == 1 ? 0 : i];
        if (!kvLengths.empty()) actualKV[i] = kvLengths[kvLengths.size() == 1 ? 0 : i];
    }
    optiling::PromptFlashAttentionSplitNSNew(tiling, coresAiv, actualQ, actualKV, 0, qs.heads == ks.heads);
    TORCH_CHECK(tiling.get_actualCoreNums() <= coresAiv, "invalid SparseBlockEstimate core partition");
    uint64_t tilingKey = 1000000000000000000ULL + (causal ? 1 : 0) + (input_layout == "BSH" ? 10 : 0) +
                         (query.scalar_type() == at::kBFloat16 ? 100 : 0);
    const uint64_t workspaceSize = coresAiv * (static_cast<uint64_t>(mBase) * sizeof(float) * (nBase + 32) * 4 +
                                               static_cast<uint64_t>(mBase) * 2 * 2 * stride * qs.dim) +
                                   platform->GetLibApiWorkSpaceSize();
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(INT64_MAX), "SparseBlockEstimate workspace is too large");
    const int64_t rows = (qs.sequence + sparse_size - 1) / sparse_size;
    const int64_t columns = ((ks.sequence + sparse_size - 1) / sparse_size + 31) / 32 * 32;
    auto mask = at::empty({qs.batch, qs.heads, rows, columns}, q.options().dtype(at::kChar));
    auto count = at::empty({qs.batch, qs.heads, rows}, q.options().dtype(at::kInt));
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));
    auto tilingTensor = sa::device_tiling(tiling);
    auto qLengthsDevice = sa::device_lengths(qLengths);
    auto kvLengthsDevice = sa::device_lengths(kvLengths);
    void *qLengthsPtr = sa::data_or_null(qLengthsDevice);
    void *kvLengthsPtr = sa::data_or_null(kvLengthsDevice);
    EXEC_KERNEL_CMD(sparse_block_estimate, coresAic, q, k, qLengthsPtr, kvLengthsPtr, mask, count, workspace,
                    tilingTensor, tilingKey);
    return {mask, count};
}

}  // namespace sglang::npu_kernel
