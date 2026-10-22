/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule.cpp
 * \brief host-side direct-launch implementation of chunk_gated_delta_rule
 */

#include <cstdint>
#include <cstring>
#include <tuple>

#include "acl/acl.h"

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_chunk_gated_delta_rule.h"

#include "../op_kernel/chunk_gated_delta_rule_tiling_data.h"

namespace sglang {
namespace npu_kernel {

constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;  // 16 MB reserved system workspace
constexpr uint32_t PADDING_BYTE = 32U;
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

void ComputeMatmulTiling(platform_ascendc::PlatformAscendC *platform, bool stateIsFp32,
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
        TORCH_CHECK(false, "chunk_gated_delta_rule: GetTiling failed for matmulTilingFp32");
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
            TORCH_CHECK(false, "chunk_gated_delta_rule: GetTiling failed for matmulTilingFp32C");
        }
        td.matmulTilingFp32C.dbL0C = 1;
        td.matmulTilingFp32C.stepKa = 1;
        td.matmulTilingFp32C.stepKb = 1;
        td.matmulTilingFp32C.depthA1 = 1;
        td.matmulTilingFp32C.depthB1 = 1;
        td.matmulTilingFp32C.stepM = 1;
        td.matmulTilingFp32C.stepN = 1;
    }
}

HOST_API std::tuple<at::Tensor, at::Tensor>
chunk_gated_delta_rule(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                       const c10::optional<at::Tensor> &beta, const c10::optional<at::Tensor> &initial_state,
                       const c10::optional<at::Tensor> &actual_seq_lengths, const c10::optional<double> &scale,
                       const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &chunk_state)
{
    TORCH_CHECK(query.defined() && key.defined() && value.defined(), "query/key/value must be defined");
    TORCH_CHECK(beta.has_value() && beta->defined(), "beta must be provided");
    TORCH_CHECK(initial_state.has_value() && initial_state->defined(), "initial_state must be provided");
    TORCH_CHECK(actual_seq_lengths.has_value() && actual_seq_lengths->defined(), "actual_seq_lengths must be provided");

    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3, "query/key/value must be 3D");
    TORCH_CHECK(beta->dim() == 2, "beta must be 2D (T, Nv)");
    TORCH_CHECK(initial_state->dim() == 4, "initial_state must be 4D (B, Nv, Dv, Dk)");
    TORCH_CHECK(actual_seq_lengths->dim() == 1, "actual_seq_lengths must be 1D (B,)");

    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
                    value.scalar_type() == at::kBFloat16,
                "query/key/value must be bfloat16");
    TORCH_CHECK(beta->scalar_type() == at::kBFloat16, "beta must be bfloat16");
    TORCH_CHECK(initial_state->scalar_type() == at::kBFloat16 || initial_state->scalar_type() == at::kFloat,
                "initial_state must be bfloat16 or float32");
    TORCH_CHECK(actual_seq_lengths->scalar_type() == at::kInt, "actual_seq_lengths must be int32");
    if (g.has_value() && g->defined()) {
        TORCH_CHECK(g->scalar_type() == at::kFloat, "g must be float32");
        TORCH_CHECK(g->dim() == 2, "g must be 2D (T, Nv)");
    }

    int64_t t = query.size(0);
    int64_t nk = query.size(1);
    int64_t dk = query.size(2);
    int64_t nv = value.size(1);
    int64_t dv = value.size(2);
    int64_t b = initial_state->size(0);

    TORCH_CHECK(key.size(0) == t && key.size(1) == nk && key.size(2) == dk, "key shape must match query");
    TORCH_CHECK(value.size(0) == t && value.size(2) == dv, "value shape must match (T, Nv, Dv)");
    TORCH_CHECK(beta->size(0) == t && beta->size(1) == nv, "beta shape must be (T, Nv)");
    TORCH_CHECK(initial_state->size(1) == nv && initial_state->size(2) == dv && initial_state->size(3) == dk,
                "initial_state shape must be (B, Nv, Dv, Dk)");
    TORCH_CHECK(actual_seq_lengths->size(0) == b, "actual_seq_lengths size must equal initial_state dim 0");

    TORCH_CHECK(nk > 0 && nv > 0 && dk > 0 && dv > 0 && t > 0 && b > 0, "dims must be positive");
    TORCH_CHECK(nk <= 64 && nv <= 64, "nk and nv must be <= 64");
    TORCH_CHECK(dk <= 128 && dv <= 128, "dk and dv must be <= 128");
    TORCH_CHECK(nv % nk == 0, "nv must be an integer multiple of nk");
    if (g.has_value() && g->defined()) {
        TORCH_CHECK(g->size(0) == t && g->size(1) == nv, "g shape must be (T, Nv)");
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t aiCoreNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAic());
    if (aiCoreNum <= 0) {
        TORCH_CHECK(false, "invalid AIC core number");
    }

    bool isAscend950 = (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND950);
    bool stateIsFp32 = (initial_state->scalar_type() == at::kFloat);
    if (stateIsFp32) {
        TORCH_CHECK(isAscend950, "FP32 initial_state is only supported on Ascend950 (arch35)");
    }

    bool hasGamma = g.has_value() && g->defined();
    float scaleValue = scale.has_value() ? static_cast<float>(*scale) : 1.0f;

    bool outputChunkState = chunk_state.has_value() && chunk_state->defined();
    if (outputChunkState) {
        TORCH_CHECK(chunk_state->dim() == 4, "chunk_state must be 4D (totalChunks, Nv, Dv, Dk)");
        TORCH_CHECK(chunk_state->size(1) == nv && chunk_state->size(2) == dv && chunk_state->size(3) == dk,
                    "chunk_state shape must match (totalChunks, Nv, Dv, Dk)");
        TORCH_CHECK(chunk_state->scalar_type() == initial_state->scalar_type(),
                    "chunk_state dtype must match initial_state");
        TORCH_CHECK(chunk_state->device() == query.device(), "chunk_state must be on the same device as query");
    }

    at::Tensor query_contig = query.contiguous();
    at::Tensor key_contig = key.contiguous();
    at::Tensor value_contig = value.contiguous();
    at::Tensor beta_contig = beta->contiguous();
    at::Tensor initStateContig = initial_state->contiguous();
    at::Tensor seqLensContig = actual_seq_lengths->contiguous();

    at::Tensor gTensor;
    void *gPtr = nullptr;
    if (hasGamma) {
        gTensor = g->contiguous();
        gPtr = gTensor.data_ptr();
    }

    at::Tensor chunkStateContig;
    void *chunkStatePtr = nullptr;
    if (outputChunkState) {
        chunkStateContig = chunk_state->contiguous();
        chunkStatePtr = chunkStateContig.data_ptr();
    }

    at::Tensor out = at::empty({t, nv, dv}, query.options());
    at::Tensor finalState = at::empty({b, nv, dv, dk}, initial_state->options());

    uint32_t blockDim = static_cast<uint32_t>(aiCoreNum);

    ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData tilingData;
    ComputeTilingData(aiCoreNum, t, nk, dk, nv, dv, b, hasGamma ? 1 : 0, scaleValue, stateIsFp32, isAscend950,
                      outputChunkState, tilingData);
    ComputeMatmulTiling(ascendcPlatform, stateIsFp32, tilingData);

    int64_t totalWorkspace = SYS_WORKSPACE_SIZE + tilingData.interWorkspaceSz + tilingData.stageWorkspaceSz;

    int32_t tilingSize =
        (static_cast<int32_t>(sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData)) + PADDING_BYTE - 1) /
        PADDING_BYTE * PADDING_BYTE;

    auto cpuTiling = at::empty({tilingSize}, at::kByte);
    std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(ChunkGatedDeltaRule::ChunkGatedDeltaRuleTilingData));
    at::Tensor tilingTensor = TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);

    auto workspaceTensor = at::empty({totalWorkspace}, at::TensorOptions().dtype(at::kByte).device(query.device()));

    EXEC_KERNEL_CMD(chunk_gated_delta_rule, blockDim, query_contig, key_contig, value_contig, beta_contig,
                    initStateContig, seqLensContig, gPtr, out, finalState, chunkStatePtr, workspaceTensor,
                    tilingTensor);

    return std::make_tuple(out, finalState);
}

}  // namespace npu_kernel
}  // namespace sglang
