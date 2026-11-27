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
    if (attentionOut->GetDimNum() != 3) return ge::GRAPH_FAILED;
    attentionOut->SetDim(2, fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM);
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
    const auto *topkDstSlots = context->GetInputShape(5);
    const auto *topkSrcIds = context->GetInputShape(6);
    const auto *topkMissCounts = context->GetInputShape(7);
    const auto *actualSeqLengthsQuery = context->GetInputShape(8);
    const auto *actualSeqLengthsKv = context->GetInputShape(9);
    const auto *numCacheTokens = context->GetInputShape(10);
    const auto *missSrcIds = context->GetInputShape(11);
    const auto *missDstSlots = context->GetInputShape(12);
    const auto *missCounts = context->GetInputShape(13);
    const auto *queryRope = context->GetOptionalInputShape(14);
    const auto *current = context->GetOptionalInputShape(15);
    const auto *positions = context->GetOptionalInputShape(16);
    const auto *currentValid = context->GetOptionalInputShape(17);
    const auto *requestSlots = context->GetOptionalInputShape(18);
    const auto *reqToToken = context->GetOptionalInputShape(19);
    const auto *residentEnd = context->GetOptionalInputShape(20);
    const auto *queryDesc = context->GetInputDesc(0);
    const auto *attrs = context->GetAttrs();
    if (!query || !hbm || !dram || !hbmTable || !dramTable || !topkDstSlots || !topkSrcIds || !topkMissCounts ||
        !actualSeqLengthsQuery || !actualSeqLengthsKv || !numCacheTokens || !missSrcIds || !missDstSlots ||
        !missCounts || !queryDesc || !attrs || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &q = query->GetOriginShape();
    const auto &h = hbm->GetOriginShape();
    const auto &d = dram->GetOriginShape();
    const auto &ht = hbmTable->GetOriginShape();
    const auto &dt = dramTable->GetOriginShape();
    const auto &topkDst = topkDstSlots->GetOriginShape();
    const auto &topkSrc = topkSrcIds->GetOriginShape();
    const auto &topkMiss = topkMissCounts->GetOriginShape();
    const auto &queryEnds = actualSeqLengthsQuery->GetOriginShape();
    const auto &kvLengths = actualSeqLengthsKv->GetOriginShape();
    const auto &n = numCacheTokens->GetOriginShape();
    const auto &missSrc = missSrcIds->GetOriginShape();
    const auto &missDst = missDstSlots->GetOriginShape();
    const auto &missCountShape = missCounts->GetOriginShape();
    const bool splitQuery = queryRope != nullptr;
    const int64_t queryDim =
        splitQuery ? fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM : fused_quant_scatter_copy_sfa::QUERY_DIM;
    if (q.GetDimNum() != 3 || h.GetDimNum() != 4 || d.GetDimNum() != 4 || ht.GetDimNum() != 2 || dt.GetDimNum() != 2 ||
        n.GetDimNum() != 1 || q.GetDim(0) <= 0 || q.GetDim(1) <= 0 || q.GetDim(2) != queryDim || h.GetDim(0) <= 0 ||
        h.GetDim(1) != fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE ||
        h.GetDim(2) != fused_quant_scatter_copy_sfa::KV_HEAD_COUNT ||
        h.GetDim(3) != fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES || d.GetDim(0) <= 0 ||
        d.GetDim(1) != fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE ||
        d.GetDim(2) != fused_quant_scatter_copy_sfa::KV_HEAD_COUNT ||
        d.GetDim(3) != fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES || ht.GetDim(0) <= 0 || ht.GetDim(1) <= 0 ||
        ht.GetDim(1) > fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST || dt.GetDim(0) != ht.GetDim(0) ||
        dt.GetDim(1) <= 0 || dt.GetDim(1) > fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST ||
        n.GetDim(0) != ht.GetDim(0) || n.GetDim(0) > UINT32_MAX || q.GetDim(0) < n.GetDim(0) ||
        q.GetDim(0) > fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT * n.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const auto validVector = [](const auto &shape, int64_t size) {
        return shape.GetDimNum() == 1 && shape.GetDim(0) == size;
    };
    const auto validTopk = [queryCount = q.GetDim(0)](const auto &shape) {
        return shape.GetDimNum() == 3 && shape.GetDim(0) == queryCount && shape.GetDim(1) == 1 &&
               shape.GetDim(2) == fused_quant_scatter_copy_sfa::TOPK_CAPACITY;
    };
    const auto validMisses = [batchSize = n.GetDim(0), splitQuery](const auto &shape) {
        return splitQuery ? shape.GetDimNum() == 2 && shape.GetDim(0) == batchSize &&
                                shape.GetDim(1) == fused_quant_scatter_copy_sfa::COPY_CAPACITY
                          : shape.GetDimNum() == 3 && shape.GetDim(0) == batchSize && shape.GetDim(1) == 1 &&
                                shape.GetDim(2) == fused_quant_scatter_copy_sfa::COPY_CAPACITY;
    };
    if (!validTopk(topkDst) || !validTopk(topkSrc) || !validVector(topkMiss, q.GetDim(0)) ||
        !validVector(queryEnds, n.GetDim(0)) || !validVector(kvLengths, n.GetDim(0)) ||
        !validVector(missCountShape, n.GetDim(0)) || !validMisses(missSrc) || !validMisses(missDst)) {
        return ge::GRAPH_FAILED;
    }
    const auto *scale = attrs->GetFloat(0);
    const auto *nativeLimPlan = attrs->GetBool(1);
    if (!scale || !nativeLimPlan || (*nativeLimPlan && !splitQuery) ||
        q.GetDim(0) > std::numeric_limits<uint32_t>::max() || q.GetDim(1) > UINT32_MAX || h.GetDim(0) > UINT32_MAX ||
        d.GetDim(0) > UINT32_MAX || ht.GetDim(1) > UINT32_MAX || dt.GetDim(1) > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (splitQuery) {
        const auto &rope = queryRope->GetOriginShape();
        if (rope.GetDimNum() != 3 || rope.GetDim(0) != q.GetDim(0) || rope.GetDim(1) != q.GetDim(1) ||
            rope.GetDim(2) != fused_quant_scatter_copy_sfa::ROPE_DIM) {
            return ge::GRAPH_FAILED;
        }
    }
    const bool stageResident = current != nullptr;
    if ((positions != nullptr) != stageResident || (currentValid != nullptr) != stageResident ||
        (requestSlots != nullptr) != stageResident || (reqToToken != nullptr) != stageResident ||
        (residentEnd != nullptr) != stageResident || (stageResident && !splitQuery)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t requestRows = 0;
    uint32_t requestContext = 0;
    if (stageResident) {
        const auto &currentShape = current->GetOriginShape();
        const auto &positionsShape = positions->GetOriginShape();
        const auto &validShape = currentValid->GetOriginShape();
        const auto &slotsShape = requestSlots->GetOriginShape();
        const auto &tableShape = reqToToken->GetOriginShape();
        const auto &endShape = residentEnd->GetOriginShape();
        if (currentShape.GetDimNum() != 2 || currentShape.GetDim(0) != q.GetDim(0) ||
            currentShape.GetDim(1) != fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES ||
            positionsShape.GetDimNum() != 1 || positionsShape.GetDim(0) != q.GetDim(0) || validShape.GetDimNum() != 1 ||
            validShape.GetDim(0) != q.GetDim(0) || slotsShape.GetDimNum() != 1 || slotsShape.GetDim(0) != n.GetDim(0) ||
            tableShape.GetDimNum() != 2 || tableShape.GetDim(0) <= 1 || tableShape.GetDim(1) <= 0 ||
            tableShape.GetDim(0) > UINT32_MAX || tableShape.GetDim(1) > UINT32_MAX || endShape.GetDimNum() != 1 ||
            endShape.GetDim(0) != tableShape.GetDim(0)) {
            return ge::GRAPH_FAILED;
        }
        requestRows = static_cast<uint32_t>(tableShape.GetDim(0));
        requestContext = static_cast<uint32_t>(tableShape.GetDim(1));
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAic();
    if (cores == 0 || (cores & 1U) != 0) return ge::GRAPH_FAILED;

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(FusedQuantCopySfaResidentTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *td = context->GetTilingData<FusedQuantCopySfaResidentTilingData>();
    if (!td) return ge::GRAPH_FAILED;
    std::memset(td, 0, sizeof(*td));
    auto &attention = td->attention;
    attention.baseParams.batchSize = static_cast<uint32_t>(n.GetDim(0));
    attention.baseParams.seqSize = static_cast<uint32_t>(ht.GetDim(1)) * fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE;
    attention.baseParams.qSeqSize = static_cast<uint32_t>(q.GetDim(0));
    attention.baseParams.nNumOfQInOneGroup = static_cast<uint32_t>(q.GetDim(1));
    attention.baseParams.sparseBlockCount = fused_quant_scatter_copy_sfa::TOPK_CAPACITY;
    attention.baseParams.sparseMode = 3;
    attention.baseParams.outputLayout = 1;
    attention.baseParams.blockSize = fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE;
    attention.baseParams.maxBlockNumPerBatch = static_cast<uint32_t>(ht.GetDim(1));
    attention.baseParams.dSizeVInput = fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES;
    attention.baseParams.scaleValue = *scale;
    attention.singleCoreParams.usedCoreNum = cores;
    attention.dramMaxBlocks = static_cast<uint32_t>(dt.GetDim(1));
    attention.hbmPhysicalBlocks = static_cast<uint32_t>(h.GetDim(0));
    attention.dramPhysicalBlocks = static_cast<uint32_t>(d.GetDim(0));
    td->requestRows = requestRows;
    td->requestContext = requestContext;
    td->nativeLimPlan = *nativeLimPlan ? 1U : 0U;
    const bool bf16 = queryDesc->GetDataType() == ge::DT_BF16;
    const bool fp16 = queryDesc->GetDataType() == ge::DT_FLOAT16;
    if ((!bf16 && !fp16) ||
        !fused_quant_scatter_copy_sfa::IsSupportedHeadCount(attention.baseParams.nNumOfQInOneGroup)) {
        return ge::GRAPH_FAILED;
    }
    // This field was reserved in the original fixed 80-byte ABI. The standard
    // CANN entry uses it only to dispatch the same four legacy template bodies.
    attention.reserved[0] = (bf16 ? 1U : 0U) | (q.GetDim(1) == 128 ? 2U : 0U);
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
        const std::initializer_list<ge::DataType> longTypes{ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::DataType> boolTypes{ge::DT_BOOL, ge::DT_BOOL};
        this->Input("query").ParamType(REQUIRED).DataType(queryTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("hbm_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dram_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char *input : {"hbm_block_table", "dram_block_table", "topk_dst_slots", "topk_src_ids",
                                  "topk_miss_counts", "actual_seq_lengths_query", "actual_seq_lengths_kv",
                                  "num_cache_tokens", "miss_src_ids", "miss_dst_slots", "miss_counts"}) {
            this->Input(input).ParamType(REQUIRED).DataType(intTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("query_rope").ParamType(OPTIONAL).DataType(queryTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("current").ParamType(OPTIONAL).DataType(byteTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("positions").ParamType(OPTIONAL).DataType(longTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("current_valid").ParamType(OPTIONAL).DataType(boolTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char *input : {"request_slots", "req_to_token", "resident_end"}) {
            this->Input(input).ParamType(OPTIONAL).DataType(intTypes).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Attr("scale_value").Float();
        this->Attr("native_lim_plan").AttrType(REQUIRED).Bool();
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
