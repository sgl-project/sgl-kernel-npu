/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Provenance: cann/ops-transformer attention/chunk_gated_delta_rule/ -> sgl-kernel-npu
// csrc/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule.cpp -> attention CANN copy. Original import revision is
// unknown; this copy preserves the local computation.

#include "../op_kernel/chunk_gated_delta_rule_tiling_data.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace optiling {

// Preserve the direct-launch workspace layout and matmul settings.
constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;  // 16 MB reserved system workspace
constexpr uint32_t STAGE_ONE_PARA_NUM = 4;
constexpr uint32_t MASK_NUM = 4;
constexpr int64_t P_NUM = 2;
constexpr int64_t CHUNK_SIZE = 64;

constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;

void ComputeTilingData(int64_t aiCoreNum, int64_t t, int64_t nk, int64_t dk, int64_t nv, int64_t dv, int64_t b,
                       int64_t hasGamma, float scale, bool stateIsFp32, bool isAscend950, bool outputChunkState,
                       ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData &td)
{
    std::memset(&td, 0, sizeof(td));

    int64_t c = CHUNK_SIZE;
    int64_t p = P_NUM;
    td.aiCoreNum = aiCoreNum;
    td.t = t;
    td.nk = nk;
    td.dk = dk;
    td.nv = nv;
    td.dv = dv;
    td.b = b;
    td.hasGamma = hasGamma;
    td.chunkSize = c;
    td.maxGroupLength = p * td.aiCoreNum * c;
    td.stageOneParaNum = STAGE_ONE_PARA_NUM;
    td.scale = scale;
    td.outputChunkState = outputChunkState ? 1 : 0;
    td.stateIsFp32 = stateIsFp32 ? 1 : 0;

    int64_t sizeHigh = 4;
    int64_t sizeLow = 2;
    int64_t s = td.maxGroupLength;

    td.interWorkspaceSz = 0;
    td.interWorkspaceSz += sizeHigh * nv * s;      // gCumExp (FP32)
    td.interWorkspaceSz += sizeLow * nv * s * dk;  // kCumDecay (BF16)
    if (stateIsFp32) {
        td.interWorkspaceSz += sizeHigh * nv * s * dv;  // vInner (FP32, arch35 FP32-state path)
        td.interWorkspaceSz += sizeLow * nv * s * dv;   // vInnerBf16 (BF16, for stage3)
    } else {
        td.interWorkspaceSz += sizeLow * nv * s * dv;  // vInner (BF16)
    }
    td.interWorkspaceSz += sizeLow * nv * s * dk;  // qPrime (BF16)
    td.interWorkspaceSz += sizeLow * nv * s * dv;  // attnInter (BF16, arch22 compat)
    td.interWorkspaceSz += sizeLow * nv * s * dk;  // kg (BF16)
    td.interWorkspaceSz += sizeLow * nv * s * c;   // qkt (BF16)
    if (stateIsFp32) {
        td.interWorkspaceSz += sizeLow * nv * dv * dk;  // stateBf16Wk (BF16, arch35)
    } else if (!isAscend950) {
        td.interWorkspaceSz += sizeHigh * b * nv * dv * dk;  // highState_ (arch22: kernel advances offset)
    }
    td.interWorkspaceSz += sizeHigh * c * c * td.aiCoreNum * MASK_NUM;  // mask (FP32)

    // stage1 temporary workspace
    td.stageWorkspaceSz = sizeLow * c * (2 * c + 3 * dk + dv) * td.stageOneParaNum;
    td.stageWorkspaceSz *= td.aiCoreNum;

    // state strides (contiguous (B, Nv, Dv, Dk) layout assumed under direct-launch)
    td.stateStride1 = static_cast<uint64_t>(dk) * static_cast<uint64_t>(dv);
    td.stateStride0 = static_cast<uint64_t>(nv) * td.stateStride1;
}

ge::graphStatus ComputeMatmulTiling(platform_ascendc::PlatformAscendC *platform, bool stateIsFp32,
                                    ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData &td)
{
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

    // ========== matmulTilingFp32: BF16 A/B transposed, BF16 C ==========
    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetBufferSpace(static_cast<int32_t>(l1Size), static_cast<int32_t>(l0CSize), static_cast<int32_t>(ubSize));
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                true);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16);
    mm.SetBias(false);
    mm.SetDim(1);
    mm.SetShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
    mm.SetOrgShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
    mm.SetFixSplit(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
    if (mm.GetTiling(td.matmulTilingFp32) == -1) {
        return ge::GRAPH_FAILED;
    }
    td.matmulTilingFp32.dbL0C = 1;
    td.matmulTilingFp32.stepKa = 1;
    td.matmulTilingFp32.stepKb = 1;
    td.matmulTilingFp32.depthA1 = 1;
    td.matmulTilingFp32.depthB1 = 1;
    td.matmulTilingFp32.stepM = 1;
    td.matmulTilingFp32.stepN = 1;

    // ========== matmulTilingFp32C: BF16 A/B transposed, FP32 C (arch35 FP32-state path) ==========
    if (stateIsFp32) {
        matmul_tiling::MultiCoreMatmulTiling mmFp32C;
        mmFp32C.SetBufferSpace(static_cast<int32_t>(l1Size), static_cast<int32_t>(l0CSize),
                               static_cast<int32_t>(ubSize));
        mmFp32C.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_BFLOAT16, true);
        mmFp32C.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_BFLOAT16, true);
        mmFp32C.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
        mmFp32C.SetBias(false);
        mmFp32C.SetDim(1);
        mmFp32C.SetShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
        mmFp32C.SetOrgShape(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
        mmFp32C.SetFixSplit(MATMUL_BASE_M, MATMUL_BASE_N, MATMUL_BASE_K);
        if (mmFp32C.GetTiling(td.matmulTilingFp32C) == -1) {
            return ge::GRAPH_FAILED;
        }
        td.matmulTilingFp32C.dbL0C = 1;
        td.matmulTilingFp32C.stepKa = 1;
        td.matmulTilingFp32C.stepKb = 1;
        td.matmulTilingFp32C.depthA1 = 1;
        td.matmulTilingFp32C.depthB1 = 1;
        td.matmulTilingFp32C.stepM = 1;
        td.matmulTilingFp32C.stepN = 1;
    }
    return ge::GRAPH_SUCCESS;
}

// Validate the same working tensors as the isolated legacy Host.
static ge::graphStatus TilingChunkGatedDeltaRule(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    constexpr size_t ranks[] = {3, 3, 3, 2, 4, 1};
    for (size_t i = 0; i < 6; ++i) {
        const auto *shape = context->GetInputShape(i);
        const auto *desc = context->GetInputDesc(i);
        if (!shape || !desc || shape->GetStorageShape().GetDimNum() != ranks[i]) return ge::GRAPH_FAILED;
        const auto dtype = desc->GetDataType();
        if ((i < 4 && dtype != ge::DT_BF16) || (i == 4 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT) ||
            (i == 5 && dtype != ge::DT_INT32))
            return ge::GRAPH_FAILED;
    }
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &k = context->GetInputShape(1)->GetStorageShape();
    const auto &v = context->GetInputShape(2)->GetStorageShape();
    const auto &beta = context->GetInputShape(3)->GetStorageShape();
    const auto &state = context->GetInputShape(4)->GetStorageShape();
    const auto &lengths = context->GetInputShape(5)->GetStorageShape();
    const int64_t t = q.GetDim(0), nk = q.GetDim(1), dk = q.GetDim(2);
    const int64_t nv = v.GetDim(1), dv = v.GetDim(2), b = state.GetDim(0);
    if (t <= 0 || nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || b <= 0 || nk > 64 || nv > 64 || dk > 128 || dv > 128 ||
        nv % nk != 0 || k.GetDim(0) != t || k.GetDim(1) != nk || k.GetDim(2) != dk || v.GetDim(0) != t ||
        beta.GetDim(0) != t || beta.GetDim(1) != nv || state.GetDim(1) != nv || state.GetDim(2) != dv ||
        state.GetDim(3) != dk || lengths.GetDim(0) != b)
        return ge::GRAPH_FAILED;

    const auto *gDesc = context->GetOptionalInputDesc(6);
    const auto *chunkDesc = context->GetOptionalInputDesc(7);
    const bool hasGamma = gDesc != nullptr;
    const bool outputChunkState = chunkDesc != nullptr;
    if (hasGamma) {
        const auto *shape = context->GetOptionalInputShape(6);
        if (!shape || gDesc->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
        const auto &g = shape->GetStorageShape();
        if (g.GetDimNum() != 2 || g.GetDim(0) != t || g.GetDim(1) != nv) return ge::GRAPH_FAILED;
    }
    if (outputChunkState) {
        const auto *shape = context->GetOptionalInputShape(7);
        if (!shape || chunkDesc->GetDataType() != context->GetInputDesc(4)->GetDataType()) return ge::GRAPH_FAILED;
        const auto &chunk = shape->GetStorageShape();
        // Legacy checks the trailing dimensions, not the totalChunks extent.
        if (chunk.GetDimNum() != 4 || chunk.GetDim(1) != nv || chunk.GetDim(2) != dv || chunk.GetDim(3) != dk)
            return ge::GRAPH_FAILED;
    }

    // Platform selection preserves the legacy FP32-state restriction.
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t aiCoreNum = static_cast<int64_t>(platform.GetCoreNumAic());
    const bool isAscend950 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const bool stateIsFp32 = context->GetInputDesc(4)->GetDataType() == ge::DT_FLOAT;
    if (aiCoreNum <= 0 || (stateIsFp32 && !isAscend950)) return ge::GRAPH_FAILED;
    const auto *attrs = context->GetAttrs();
    const auto *scale = attrs ? attrs->GetAttrPointer<float>(0) : nullptr;
    const float scaleValue = scale ? *scale : 1.0f;
    ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData tilingData;
    ComputeTilingData(aiCoreNum, t, nk, dk, nv, dv, b, hasGamma ? 1 : 0, scaleValue, stateIsFp32, isAscend950,
                      outputChunkState, tilingData);
    if (ComputeMatmulTiling(&platform, stateIsFp32, tilingData) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    // Serialize the unchanged raw layout; reserve the legacy 16 MiB system prefix.
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(tilingData) || !workspace) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &tilingData, sizeof(tilingData));
    raw->SetDataSize(sizeof(tilingData));
    context->SetBlockDim(static_cast<uint32_t>(aiCoreNum));
    context->SetTilingKey(0);
    workspace[0] = SYS_WORKSPACE_SIZE + tilingData.interWorkspaceSz + tilingData.stageWorkspaceSz;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

// Output tensors preserve the value/state shapes; the optional result follows chunk_state.
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *value = context->GetInputShape(2);
    const auto *state = context->GetInputShape(4);
    auto *out = context->GetOutputShape(0);
    auto *finalState = context->GetOutputShape(1);
    if (!value || !state || !out || !finalState) return ge::GRAPH_FAILED;
    *out = *value;
    *finalState = *state;
    const auto *chunk = context->GetInputShape(7);
    auto *chunkResult = context->GetOutputShape(2);
    if (chunk) {
        if (!chunkResult) return ge::GRAPH_FAILED;
        *chunkResult = *chunk;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(2));
    context->SetOutputDataType(1, context->GetInputDataType(4));
    const auto chunkType = context->GetInputDataType(7);
    if (chunkType != ge::DT_UNDEFINED) context->SetOutputDataType(2, chunkType);
    return ge::GRAPH_SUCCESS;
}

class SglChunkGatedDeltaRule : public OpDef
{
public:
    explicit SglChunkGatedDeltaRule(const char *name) : OpDef(name)
    {
        // Two dtype combinations differ only in state and its optional chunk buffer.
        for (const auto *input : {"query", "key", "value", "beta"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("initial_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("g")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("chunk_state")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("final_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // The Torch adapter binds this output to the chunk_state working buffer.
        this->Output("chunk_state_result")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0f);
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingChunkGatedDeltaRule);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglChunkGatedDeltaRule);

}  // namespace ops
