/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d.cpp
 * \brief causal_conv1d host-side implementation
 */

#include <cstring>
#include <limits>
#include <unordered_map>
#include "acl/acl.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "tiling/platform/platform_ascendc.h"
#include "stub/aclrtlaunch_causal_conv1d.h"
#include "defines.h"
#include "torch_helper.h"
#include "common.h"
#include "causal_conv1d.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr int64_t DIM_ALIGN = 16;
constexpr int64_t MAX_DIM_TILE = 4096;
constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MIN_WIDTH = 2;
constexpr int64_t ASCENDC_RESERVED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;

constexpr uint32_t CAUSAL_CONV1D_TPL_RUN_MODE_FN = 0;
constexpr uint32_t CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE = 1;
constexpr uint32_t CAUSAL_CONV1D_TPL_WIDTH_2 = 1;
constexpr uint32_t CAUSAL_CONV1D_TPL_WIDTH_3 = 2;
constexpr uint32_t CAUSAL_CONV1D_TPL_WIDTH_4 = 3;
constexpr uint32_t CAUSAL_CONV1D_TPL_FN_PLAN_INVALID = 0;
constexpr uint32_t CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS = 1;
constexpr uint32_t CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD = 2;

static uint32_t g_causalConv1dCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> g_causalConv1dCaptureMap;

struct CausalConv1dTilingKey {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t batch;
    int64_t inputMode;
    int64_t width;
    int64_t stateLen;
    int64_t numCacheLines;
    int64_t activationMode;
    int64_t padSlotId;
    int64_t runMode;
    int64_t hasBias;
    int64_t hasCacheIndices;
    int64_t hasInitialState;
    int64_t hasNumAccept;
};

struct CausalConv1dTilingKeyHash {
    static inline std::size_t HashCombine(std::size_t seed, std::size_t value)
    {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }

    std::size_t operator()(const CausalConv1dTilingKey &k) const
    {
        std::size_t h = 0;
        h = HashCombine(h, static_cast<std::size_t>(k.dim));
        h = HashCombine(h, static_cast<std::size_t>(k.cuSeqlen));
        h = HashCombine(h, static_cast<std::size_t>(k.seqLen));
        h = HashCombine(h, static_cast<std::size_t>(k.batch));
        h = HashCombine(h, static_cast<std::size_t>(k.inputMode));
        h = HashCombine(h, static_cast<std::size_t>(k.width));
        h = HashCombine(h, static_cast<std::size_t>(k.stateLen));
        h = HashCombine(h, static_cast<std::size_t>(k.numCacheLines));
        h = HashCombine(h, static_cast<std::size_t>(k.activationMode));
        h = HashCombine(h, static_cast<std::size_t>(k.padSlotId));
        h = HashCombine(h, static_cast<std::size_t>(k.runMode));
        h = HashCombine(h, static_cast<std::size_t>(k.hasBias));
        h = HashCombine(h, static_cast<std::size_t>(k.hasCacheIndices));
        h = HashCombine(h, static_cast<std::size_t>(k.hasInitialState));
        h = HashCombine(h, static_cast<std::size_t>(k.hasNumAccept));
        return h;
    }
};

namespace {

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t AlignUp(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

struct UpdateDimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

// Mirror of the GE update-mode tiling policy (ChooseCanonicalUpdateBaseDimChoice):
// pick a baseDim from a fixed candidate set that ideally divides dim and makes the
// batch*baseDimCnt grid land as close as possible to (>=) the AIV core count, so the
// batch-parallel update kernel keeps all cores busy. Falls back to ceil-division when
// no candidate divides dim exactly.
UpdateDimTileChoice ChooseUpdateBaseDimChoice(int64_t batch, int64_t dim, int32_t numCores)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};
    const int64_t coreNum = (numCores > 0) ? static_cast<int64_t>(numCores) : 1;

    auto chooseOnce = [&](bool requireExactDiv) -> UpdateDimTileChoice {
        UpdateDimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        UpdateDimTileChoice bestUnder;

        for (int64_t candBaseDim : candidates) {
            if (candBaseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % candBaseDim != 0)) {
                continue;
            }
            const int64_t baseDimCnt = requireExactDiv ? (dim / candBaseDim) : CeilDiv(dim, candBaseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }
            if (gridSize >= coreNum) {
                const int64_t gap = gridSize - coreNum;
                if (gap < bestOverGap) {
                    bestOver = {candBaseDim, baseDimCnt, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && candBaseDim < bestUnder.baseDim)) {
                bestUnder = {candBaseDim, baseDimCnt, gridSize};
            }
        }
        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    UpdateDimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    return result;
}

void ComputeTilingData(int64_t dim, int64_t cuSeqlen, int64_t seqLen, int64_t batch, int64_t inputMode, int64_t width,
                       int64_t stateLen, int64_t numCacheLines, int64_t activationMode, int64_t padSlotId, bool hasBias,
                       bool hasCacheIndices, bool hasInitialState, bool hasNumAccept, bool isBf16, int32_t numCores,
                       int64_t runMode, CausalConv1dTilingData &td)
{
    (void)padSlotId;
    std::memset(&td, 0, sizeof(td));

    td.dim = dim;
    td.cuSeqlen = cuSeqlen;
    td.seqLen = seqLen;
    td.inputMode = inputMode;
    td.width = width;
    td.stateLen = stateLen;
    td.numCacheLines = numCacheLines;
    td.batch = batch;
    td.activationMode = activationMode;
    td.padSlotId = padSlotId;
    td.hasBias = hasBias ? 1 : 0;
    td.hasCacheIndices = hasCacheIndices ? 1 : 0;
    td.hasInitialStateMode = hasInitialState ? 1 : 0;
    td.hasInitStateWorkspace = hasInitialState ? 1 : 0;
    td.hasNumAcceptedTokens = hasNumAccept ? 1 : 0;

    td.dtypeKey = isBf16 ? 0 : 1;
    td.runModeKey = static_cast<uint32_t>(runMode);
    td.widthKey = (width == 2)   ? CAUSAL_CONV1D_TPL_WIDTH_2
                  : (width == 3) ? CAUSAL_CONV1D_TPL_WIDTH_3
                                 : CAUSAL_CONV1D_TPL_WIDTH_4;

    if (runMode == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE) {
        UpdateDimTileChoice choice = ChooseUpdateBaseDimChoice(batch, dim, numCores);
        if (choice.baseDim <= 0 || choice.baseDimCnt <= 0) {
            choice.baseDim = (dim > 0 && dim <= MAX_DIM_TILE) ? dim : MAX_DIM_TILE;
            choice.baseDimCnt = (choice.baseDim > 0) ? CeilDiv(dim, choice.baseDim) : 1;
            if (choice.baseDimCnt <= 0) {
                choice.baseDimCnt = 1;
            }
        }
        td.baseDim = choice.baseDim;
        td.baseDimCnt = choice.baseDimCnt;
        td.fnPlanKey = CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
        td.tokenBlockSize = 0;
        td.tokenBlockCnt = 0;
    } else if (dim <= MAX_DIM_TILE && numCores > 0) {
        td.baseDim = dim;
        td.baseDimCnt = 1;
        td.fnPlanKey = CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        int64_t tokenCoreBudget = numCores;
        int64_t idealBlockSize = CeilDiv(cuSeqlen, tokenCoreBudget);
        if (idealBlockSize <= 0) {
            idealBlockSize = 1;
        }
        td.tokenBlockSize = idealBlockSize;
        td.tokenBlockCnt = CeilDiv(cuSeqlen, td.tokenBlockSize);
    } else {
        td.baseDim = MAX_DIM_TILE;
        td.baseDimCnt = CeilDiv(dim, td.baseDim);
        td.fnPlanKey = CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD;
        int64_t tokenCoreBudget = (numCores > 0) ? (numCores / td.baseDimCnt) : 1;
        if (tokenCoreBudget <= 0) {
            tokenCoreBudget = 1;
        }
        int64_t idealBlockSize = CeilDiv(cuSeqlen, tokenCoreBudget);
        if (idealBlockSize <= 0) {
            idealBlockSize = 1;
        }
        td.tokenBlockSize = idealBlockSize;
        td.tokenBlockCnt = CeilDiv(cuSeqlen, td.tokenBlockSize);
    }

    td.hasExplicitTokenSeqRanges = 0;
    td.explicitTokenSeqRangeCount = 0;
}

int64_t ComputeWorkspaceSize(int32_t blockDim, int64_t batch, int64_t width, int64_t dim, bool hasInitialState)
{
    if (!hasInitialState) {
        return 0;
    }
    constexpr int64_t kDtypeSize = 2;
    constexpr int64_t kSyncBytesPerBlock = 32;
    int64_t historyCount = (width - 1 > 0) ? width - 1 : 0;
    return ASCENDC_RESERVED_WORKSPACE + static_cast<int64_t>(blockDim) * kSyncBytesPerBlock +
           batch * historyCount * dim * kDtypeSize;
}

}  // namespace

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias,
                                       const at::Tensor &conv_states, const at::Tensor &query_start_loc,
                                       const at::Tensor &cache_indices, const at::Tensor &has_initial_state,
                                       const at::Tensor &num_accepted_tokens, int64_t activation_mode,
                                       int64_t pad_slot_id, int64_t run_mode)
{
    TORCH_CHECK(x.defined(), "x tensor must be defined");
    TORCH_CHECK(weight.defined(), "weight tensor must be defined");
    TORCH_CHECK(conv_states.defined(), "conv_states tensor must be defined");

    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D or 3D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D tensor");

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported");
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match x dtype");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_states.is_contiguous(), "conv_states must be contiguous");

    int64_t dim = (x.dim() == 2) ? x.size(1) : x.size(2);
    int64_t width = weight.size(0);
    TORCH_CHECK(width >= MIN_WIDTH && width <= MAX_WIDTH, "Only support width in [2,4]");

    int64_t inputMode = (x.dim() == 2) ? 0 : 1;
    int64_t seqLen = (inputMode == 1) ? x.size(1) : 0;
    int64_t batch = (inputMode == 1) ? x.size(0) : 0;
    int64_t cuSeqlen = (inputMode == 0) ? x.size(0) : batch * seqLen;

    if (inputMode == 0) {
        int64_t qslSize = query_start_loc.size(0);
        TORCH_CHECK(qslSize >= 2, "query_start_loc must have at least 2 elements");
        batch = qslSize - 1;
    }

    int64_t numCacheLines = conv_states.size(0);
    int64_t stateLen = conv_states.size(1);

    int64_t activationInt = activation_mode;

    bool hasBias = bias.defined() && bias.numel() > 0;
    bool hasCacheIndices = cache_indices.defined() && cache_indices.numel() > 0;
    bool hasInitialState = has_initial_state.defined() && has_initial_state.numel() > 0;
    bool hasNumAccept = num_accepted_tokens.defined() && num_accepted_tokens.numel() > 0;
    bool isBf16 = (dtype == at::kBFloat16);

    at::Tensor y = at::empty_like(x);

    at::Tensor bias_tensor = hasBias ? bias : at::empty({0}, x.options());
    at::Tensor query_start_loc_tensor = (query_start_loc.defined() && query_start_loc.numel() > 0)
                                            ? query_start_loc.to(at::kLong)
                                            : at::empty({0}, x.options().dtype(at::kLong));
    at::Tensor cache_indices_tensor =
        hasCacheIndices ? cache_indices.to(at::kLong) : at::empty({0}, x.options().dtype(at::kLong));
    at::Tensor has_initial_state_tensor =
        hasInitialState ? has_initial_state.to(at::kLong) : at::empty({0}, x.options().dtype(at::kLong));
    at::Tensor num_accepted_tokens_tensor;
    if (hasNumAccept) {
        num_accepted_tokens_tensor = num_accepted_tokens.to(at::kInt);
    } else {
        num_accepted_tokens_tensor = at::empty({0}, x.options().dtype(at::kInt));
    }

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t maxAivCore = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    CausalConv1dTilingData tilingData;
    ComputeTilingData(dim, cuSeqlen, seqLen, batch, inputMode, width, stateLen, numCacheLines, activationInt,
                      pad_slot_id, hasBias, hasCacheIndices, hasInitialState, hasNumAccept, isBf16, maxAivCore,
                      run_mode, tilingData);

    int64_t totalBlocks = (run_mode == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE)
                              ? (tilingData.batch * tilingData.baseDimCnt)
                              : (tilingData.tokenBlockCnt * tilingData.baseDimCnt);
    int32_t blockDim = std::min(maxAivCore, static_cast<int32_t>(totalBlocks));
    if (blockDim <= 0) {
        blockDim = 1;
    }

    int32_t libApiWorkspaceSize = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    int64_t ws = ComputeWorkspaceSize(blockDim, batch, width, dim, hasInitialState);
    int64_t totalWorkspace = std::max(static_cast<int64_t>(libApiWorkspaceSize), ws);
    if (totalWorkspace <= 0) {
        totalWorkspace = libApiWorkspaceSize;
    }

    int32_t tilingSize =
        (static_cast<int32_t>(sizeof(CausalConv1dTilingData)) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;

    CausalConv1dTilingKey key{dim,
                              cuSeqlen,
                              seqLen,
                              batch,
                              inputMode,
                              width,
                              stateLen,
                              numCacheLines,
                              activationInt,
                              pad_slot_id,
                              run_mode,
                              hasBias ? 1 : 0,
                              hasCacheIndices ? 1 : 0,
                              hasInitialState ? 1 : 0,
                              hasNumAccept ? 1 : 0};
    uint64_t hashValue = CausalConv1dTilingKeyHash{}(key);

    static auto globalTilingBuffer = at::empty({tilingSize * static_cast<int64_t>(MAX_CAPTURE_NUM)},
                                               at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(CausalConv1dTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    at::Tensor tilingTensor;
    if (g_causalConv1dCaptureMap.find(hashValue) != g_causalConv1dCaptureMap.end()) {
        tilingTensor =
            at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * g_causalConv1dCaptureMap[hashValue]),
                          tilingSize, at::kByte);
    } else if (g_causalConv1dCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        g_causalConv1dCaptureMap[hashValue] = g_causalConv1dCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, g_causalConv1dCaptureNum * tilingSize, g_causalConv1dCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        g_causalConv1dCaptureNum++;
        tilingTensor =
            at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * g_causalConv1dCaptureMap[hashValue]),
                          tilingSize, at::kByte);
    }

    auto workspaceTensor =
        at::empty({totalWorkspace}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    EXEC_KERNEL_CMD(causal_conv1d, blockDim, x, weight, conv_states, bias_tensor, query_start_loc_tensor,
                    cache_indices_tensor, has_initial_state_tensor, num_accepted_tokens_tensor, y, workspaceTensor,
                    tilingTensor);

    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
