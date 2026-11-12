/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"

#include "arch35/qsfa/kv_quant_sparse_flash_attention_vllm_kernel_mla.h"

using namespace AscendC;

namespace {

// A5 BF16 / E4M3, paged MLA with RoPE64, from vLLM-Ascend 37854b83.
template <bool SplitG>
__aicore__ inline void RunPackedQsfa(GM_ADDR query, GM_ADDR packedKv, GM_ADDR sparseIndices, GM_ADDR blockTable,
                                     GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengthsKv, GM_ADDR output,
                                     GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                     const __gm__ SparseFlashAttentionTilingDataMla *tiling)
{
    using Cube =
        BaseApi::QSFAMatmulService<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, false, true, QSFA_LAYOUT::TND,
                                   QSFA_LAYOUT::PA_BSND, QSFATemplateMode::CFA_TEMPLATE_MODE, SplitG, false, false>;
    using CubeDummy = BaseApi::QSFAMatmulServiceDummy<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, false, true,
                                                      QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND,
                                                      QSFATemplateMode::CFA_TEMPLATE_MODE, SplitG, false, false>;
    using Vec =
        BaseApi::QSFAVectorService<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, false, true, QSFA_LAYOUT::TND,
                                   QSFA_LAYOUT::PA_BSND, QSFATemplateMode::CFA_TEMPLATE_MODE, SplitG, false, false>;
    using VecDummy = BaseApi::QSFAVectorServiceDummy<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, false, true,
                                                     QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND,
                                                     QSFATemplateMode::CFA_TEMPLATE_MODE, SplitG, false, false>;
    using CubeBlock = typename std::conditional<g_coreType == AscendC::AIC, Cube, CubeDummy>::type;
    using VecBlock = typename std::conditional<g_coreType == AscendC::AIC, VecDummy, Vec>::type;

    TPipe pipe;
    const SparseFlashAttentionTilingDataMla *localTiling = nullptr;
#if !defined(__DAV_C310_CUBE__)
    SparseFlashAttentionTilingDataMla tilingBuffer;
    auto *dst = reinterpret_cast<uint32_t *>(&tilingBuffer);
    const auto *src = reinterpret_cast<const __gm__ uint32_t *>(tiling);
    static_assert(sizeof(tilingBuffer) % sizeof(uint32_t) == 0);
#pragma unroll
    for (uint32_t i = 0; i < sizeof(tilingBuffer) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    localTiling = &tilingBuffer;
#endif

    BaseApi::KvQuantSparseFlashAttentionVllmMla<CubeBlock, VecBlock> op;
    op.Init(query, packedKv, packedKv, sparseIndices, nullptr, nullptr, blockTable, actualSeqLengthsQ,
            actualSeqLengthsKv, output, softmaxMax, softmaxSum, GetUserWorkspace(workspace), localTiling, &pipe);
    op.Process();
}
}  // namespace

extern "C" __global__ __aicore__ void sgl_kv_quant_sparse_flash_attention(GM_ADDR query, GM_ADDR packedKv,
                                                                          GM_ADDR sparseIndices, GM_ADDR blockTable,
                                                                          GM_ADDR actualSeqLengthsQ,
                                                                          GM_ADDR actualSeqLengthsKv, GM_ADDR output,
                                                                          GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto *data = reinterpret_cast<const __gm__ SparseFlashAttentionTilingDataMla *>(tiling);
    if (data->baseParams.nNumOfQInOneGroup > 64) {
        RunPackedQsfa<true>(query, packedKv, sparseIndices, blockTable, actualSeqLengthsQ, actualSeqLengthsKv, output,
                            softmaxMax, softmaxSum, workspace, data);
    } else {
        RunPackedQsfa<false>(query, packedKv, sparseIndices, blockTable, actualSeqLengthsQ, actualSeqLengthsKv, output,
                             softmaxMax, softmaxSum, workspace, data);
    }
}
