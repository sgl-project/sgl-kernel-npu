// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "sgl_operator_config.h"
#include "../op_kernel/fused_quant_copy_sfa_tiling_data.h"
#include "../op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace optiling {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *query = context->GetInputShape(0);
    const auto *hbm = context->GetInputShape(1);
    auto *hbmOut = context->GetOutputShape(0);
    auto *attentionOut = context->GetOutputShape(1);
    if (!query || !hbm || !hbmOut || !attentionOut) return ge::GRAPH_FAILED;
    *hbmOut = *hbm;
    *attentionOut = *query;
    auto &shape = attentionOut->GetOriginShape();
    if (shape.GetDimNum() != 3) return ge::GRAPH_FAILED;
    shape.SetDim(2, fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(1));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling(gert::TilingContext *context)
{
    const auto *query = context->GetInputShape(0);
    const auto *hbm = context->GetInputShape(1);
    const auto *dram = context->GetInputShape(2);
    const auto *hbmTable = context->GetInputShape(3);
    const auto *dramTable = context->GetInputShape(4);
    const auto *numCacheTokens = context->GetInputShape(10);
    const auto *queryDesc = context->GetInputDesc(0);
    const auto *attrs = context->GetAttrs();
    if (!query || !hbm || !dram || !hbmTable || !dramTable || !numCacheTokens || !queryDesc || !attrs ||
        !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &q = query->GetOriginShape();
    const auto &h = hbm->GetOriginShape();
    const auto &d = dram->GetOriginShape();
    const auto &ht = hbmTable->GetOriginShape();
    const auto &dt = dramTable->GetOriginShape();
    const auto &n = numCacheTokens->GetOriginShape();
    if (q.GetDimNum() != 3 || h.GetDimNum() != 4 || d.GetDimNum() != 4 || ht.GetDimNum() != 2 || dt.GetDimNum() != 2 ||
        n.GetDimNum() != 1 || q.GetDim(0) <= 0 || q.GetDim(1) <= 0 || q.GetDim(2) != 576 || h.GetDim(0) <= 0 ||
        d.GetDim(0) <= 0 || ht.GetDim(0) <= 0 || ht.GetDim(1) <= 0 || dt.GetDim(1) <= 0 ||
        n.GetDim(0) != ht.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const auto *scale = attrs->GetFloat(0);
    if (!scale || q.GetDim(0) > std::numeric_limits<uint32_t>::max() || q.GetDim(1) > UINT32_MAX ||
        h.GetDim(0) > UINT32_MAX || d.GetDim(0) > UINT32_MAX || ht.GetDim(1) > UINT32_MAX ||
        dt.GetDim(1) > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAic();
    if (cores == 0 || (cores & 1U) != 0) return ge::GRAPH_FAILED;

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(FusedQuantScatterCopySparseFlashAttentionTilingData) ||
        !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *td = context->GetTilingData<FusedQuantScatterCopySparseFlashAttentionTilingData>();
    if (!td) return ge::GRAPH_FAILED;
    std::memset(td, 0, sizeof(*td));
    td->baseParams.batchSize = static_cast<uint32_t>(n.GetDim(0));
    td->baseParams.seqSize = static_cast<uint32_t>(ht.GetDim(1)) * fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE;
    td->baseParams.qSeqSize = static_cast<uint32_t>(q.GetDim(0));
    td->baseParams.nNumOfQInOneGroup = static_cast<uint32_t>(q.GetDim(1));
    td->baseParams.sparseBlockCount = fused_quant_scatter_copy_sfa::TOPK_CAPACITY;
    td->baseParams.sparseMode = 3;
    td->baseParams.outputLayout = 1;
    td->baseParams.blockSize = fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE;
    td->baseParams.maxBlockNumPerBatch = static_cast<uint32_t>(ht.GetDim(1));
    td->baseParams.dSizeVInput = fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES;
    td->baseParams.scaleValue = *scale;
    td->singleCoreParams.usedCoreNum = cores;
    td->dramMaxBlocks = static_cast<uint32_t>(dt.GetDim(1));
    td->hbmPhysicalBlocks = static_cast<uint32_t>(h.GetDim(0));
    td->dramPhysicalBlocks = static_cast<uint32_t>(d.GetDim(0));
    const bool bf16 = queryDesc->GetDataType() == ge::DT_BF16;
    const bool fp16 = queryDesc->GetDataType() == ge::DT_FLOAT16;
    if ((!bf16 && !fp16) || !fused_quant_scatter_copy_sfa::IsSupportedHeadCount(td->baseParams.nNumOfQInOneGroup)) {
        return ge::GRAPH_FAILED;
    }
    // This field was reserved in the original fixed 80-byte ABI. The standard
    // CANN entry uses it only to dispatch the same four legacy template bodies.
    td->reserved[0] = (bf16 ? 1U : 0U) | (q.GetDim(1) == 128 ? 2U : 0U);
    raw->SetDataSize(sizeof(*td));
    context->SetBlockDim(cores);
    context->SetScheduleMode(1);
    context->SetTilingKey(0);
    workspace[0] = platform.GetLibApiWorkSpaceSize() +
                   fused_quant_scatter_copy_sfa::UserWorkspaceBytes(cores, static_cast<uint32_t>(q.GetDim(0)));
    return ge::GRAPH_SUCCESS;
}

}  // namespace
}  // namespace optiling

namespace ops {
class SglFusedQuantScatterCopySparseFlashAttention : public OpDef
{
public:
    explicit SglFusedQuantScatterCopySparseFlashAttention(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> queryTypes{ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> byteTypes{ge::DT_INT8, ge::DT_INT8};
        const std::initializer_list<ge::DataType> intTypes{ge::DT_INT32, ge::DT_INT32};
        this->Input("query").ParamType(REQUIRED).DataType(queryTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("hbm_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dram_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char *input : {"hbm_block_table", "dram_block_table", "topk_dst_slots", "topk_src_ids",
                                  "topk_miss_counts", "actual_seq_lengths_query", "actual_seq_lengths_kv",
                                  "num_cache_tokens", "miss_src_ids", "miss_dst_slots", "miss_counts"}) {
            this->Input(input).ParamType(REQUIRED).DataType(intTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Attr("scale_value").Float();
        this->Output("hbm_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attention_out").ParamType(REQUIRED).DataType(queryTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(optiling::InferShape).SetInferDataType(optiling::InferDataType);
        this->AICore().SetTiling(optiling::Tiling);
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglFusedQuantScatterCopySparseFlashAttention);
}  // namespace ops
