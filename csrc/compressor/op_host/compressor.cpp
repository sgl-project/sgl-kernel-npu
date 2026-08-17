/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file compressor.cpp
 * \brief host wrapper (ge_helper + direct kernel launch) for the Compressor op.
 */
#include <cstdio>
#include <string>
#include <unordered_map>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "defines.h"
#include "torch_helper.h"
#include "compressor_tiling.h"
#include "ge_helper.h"
#include "common_tiling.h"
#include "common.h"
#include "compressor_def.h"
#include "aclrtlaunch_compressor.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t MAX_CAPTURE_NUM = 1024;

namespace {

struct TilingCache {
    at::Tensor buffer;
    std::unordered_map<uint64_t, uint32_t> slots;
    uint32_t nextSlot = 0;
};

bool IsNpuGraphCapturing()
{
    aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model = nullptr;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto status = aclmdlRICaptureGetInfo(stream, &captureStatus, &model);
    TORCH_CHECK(status == ACL_ERROR_NONE, "compressor: failed to query NPU graph capture status, acl error ", status);
    return captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

std::unordered_map<int64_t, TilingCache> &GetTilingCaches()
{
    static std::unordered_map<int64_t, TilingCache> deviceCaches;
    return deviceCaches;
}

}  // namespace

namespace {

using namespace CompressorHost;

struct CompressorShapeInfo {
    bool isBsMerge = false;  // true: TH layout
    int64_t bSize = 0;       // B
    int64_t tSize = 0;       // T
    int64_t sSize = 0;       // S
    int64_t sCompress = 0;   // Sr
    int64_t hiddenSize = 0;  // H
};

// Compute cmp_kv output shape from the input tensors (ported from compressor_proto.cpp).
CompressorShapeInfo ComputeCompressorShape(const at::Tensor &x, const at::Tensor &norm_weight,
                                           const at::Tensor &rope_sin, int64_t cmp_ratio,
                                           const c10::optional<at::Tensor> &cu_seqlens_opt)
{
    CompressorShapeInfo info;
    if (x.dim() == 3) {
        info.isBsMerge = false;
        info.bSize = x.size(0);
        info.sSize = x.size(1);
        info.tSize = info.bSize * info.sSize;
    } else {
        info.isBsMerge = true;
        info.tSize = x.size(0);
    }
    info.hiddenSize = norm_weight.size(0);
    // TH: Sr = min(T, T // cmp_ratio + B); BSH: Sr = ceil(S / cmp_ratio)
    if (info.isBsMerge) {
        int64_t bSize = 1;
        if (cu_seqlens_opt.has_value() && cu_seqlens_opt->numel() >= 2) {
            bSize = cu_seqlens_opt->size(0) - 1;
        }
        info.sCompress = std::min(info.tSize, info.tSize / cmp_ratio + bSize);
    } else {
        info.sCompress = (info.sSize + cmp_ratio - 1) / cmp_ratio;
    }
    return info;
}

}  // namespace

HOST_API at::Tensor compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                               at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                               const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                               const c10::optional<at::Tensor> &state_block_table,
                               const c10::optional<at::Tensor> &cu_seqlens,
                               const c10::optional<at::Tensor> &seqused,
                               const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio,
                               int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode,
                               int64_t state_cache_stride_dim0)
{
    using namespace optiling;
    Compressor compressorOp("compressor");
    auto context = std::make_shared<ge_helper::TilingContext>("compressor");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    // ---- 1) output shape ----
    auto shapeInfo = ComputeCompressorShape(x, norm_weight, rope_sin, cmp_ratio, cu_seqlens);
    at::Tensor cmp_kv;
    if (shapeInfo.isBsMerge) {
        cmp_kv = at::empty({shapeInfo.sCompress, shapeInfo.hiddenSize}, x.options());
    } else {
        cmp_kv = at::empty({shapeInfo.bSize, shapeInfo.sCompress, shapeInfo.hiddenSize}, x.options());
    }

    // ---- 2) fill attrs from args ----
    compressorOp.SetAttrAny("rope_head_dim", static_cast<int32_t>(rope_head_dim));
    compressorOp.SetAttrAny("cmp_ratio", static_cast<int32_t>(cmp_ratio));
    compressorOp.SetAttrAny("coff", static_cast<int32_t>(coff));
    compressorOp.SetAttrAny("norm_eps", static_cast<float>(norm_eps));
    compressorOp.SetAttrAny("rotary_mode", static_cast<int32_t>(rotary_mode));
    compressorOp.SetAttrAny("cache_mode", static_cast<int32_t>(cache_mode));
    // state_cache stride dim0 = dim1*dim2 when not explicitly provided (0)
    int64_t strideDim0 = state_cache_stride_dim0;
    if (strideDim0 == 0 && state_cache.dim() >= 3) {
        strideDim0 = state_cache.size(1) * state_cache.size(2);
    }
    compressorOp.SetAttrAny("state_cache_stride_dim0", static_cast<int32_t>(strideDim0));

    auto xScalarType = x.scalar_type();
    compressorOp.SetToContext(context, xScalarType);

    // ---- 3) register tensors (12 inputs; cmp_kv output; state_cache is in/out) ----
    context->RegisterTensor(x, true);
    context->RegisterTensor(wkv, true);
    context->RegisterTensor(wgate, true);
    context->RegisterTensor(state_cache, true);
    context->RegisterTensor(ape, true);
    context->RegisterTensor(norm_weight, true);
    context->RegisterTensor(rope_sin, true);
    context->RegisterTensor(rope_cos, true);
    context->RegisterTensor(state_block_table, true);
    context->RegisterTensor(cu_seqlens, true);
    context->RegisterTensor(seqused, true);
    context->RegisterTensor(start_pos, true);
    context->RegisterTensor(cmp_kv, false);

    // ---- 4) tiling ----
    CompressorContext compressorContext{};
    if (CompressorTiling::ConvertContext(*context, compressorContext) != ge::GRAPH_SUCCESS) {
        TORCH_CHECK(false, "[compressor] ConvertContext failed");
    }
    CompressorTilingData tilingData{};
    CompressorTiling compressorTiling(&compressorContext);
    if (compressorTiling.RunBigKernelTiling(&tilingData) != ge::GRAPH_SUCCESS) {
        TORCH_CHECK(false, "[compressor] RunBigKernelTiling failed");
    }
    tilingData.tilingKey = compressorContext.tilingKey;

    uint32_t blockDim = compressorContext.blockDim;
    uint64_t tilingKey = compressorContext.tilingKey;
    size_t workspaceSize = context->GetWorkspaceSize();
    // Make sure CalcWorkSpace wrote a value (fall back to context user size)
    if (workspaceSize == 0) {
        workspaceSize = sizeof(CompressorTilingData);
    }

    // ---- 5) copy tiling data to device with graph-capture cache ----
    uint32_t tilingSize = sizeof(CompressorTilingData);
    auto tup = std::make_tuple(tilingData.baseParams.batchSize, tilingData.baseParams.seqSize,
                               tilingData.baseParams.hiddenSize, tilingData.baseParams.headDim,
                               tilingData.baseParams.cmpRatio, tilingData.baseParams.tokenSize,
                               tilingData.baseParams.cgSize, tilingData.baseParams.nSize,
                               tilingData.baseParams.usedCoreNum, tilingData.pageAttentionParams.blockNum,
                               tilingData.pageAttentionParams.blockSize,
                               tilingData.pageAttentionParams.maxBlockNumPerBatch,
                               tilingData.innerSplitParams.mBaseSize, tilingData.innerSplitParams.dBaseSize,
                               tilingData.workspaceParams.mm1KvResSize, tilingData.workspaceParams.mm1ScoreResSize,
                               tilingData.workspaceParams.vec1ResSize, tilingData.workspaceParams.vec1TailCacheSize,
                               tilingData.workspaceParams.dbWorkspaceRatio, tilingData.tilingKey);
    auto hashValue = host_utils::TupleHasher::Hash(tup);

    auto &cache = GetTilingCaches()[x.device().index()];
    if (!cache.buffer.defined()) {
        TORCH_CHECK(!IsNpuGraphCapturing(),
                    "compressor: run one eager warmup before NPU graph capture to initialize the tiling cache");
        cache.buffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                 at::TensorOptions().dtype(at::kByte).device(x.options().device()));
    }
    at::Tensor tilingTensor;
    auto iter = cache.slots.find(hashValue);
    if (iter != cache.slots.end()) {
        // decode replay / graph capture: reuse cached tiling from the device-resident buffer
        tilingTensor = cache.buffer.narrow(0, iter->second * tilingSize, tilingSize);
    } else {
        TORCH_CHECK(cache.nextSlot < MAX_CAPTURE_NUM, "compressor: tiling cache exhausted after ", MAX_CAPTURE_NUM,
                    " unique configurations");
        TORCH_CHECK(!IsNpuGraphCapturing(),
                    "compressor: the current tiling configuration is not cached; run one eager warmup with the same "
                    "tensor shapes, dtypes, optional inputs, and attributes before NPU graph capture");
        const uint32_t slot = cache.nextSlot;
        tilingTensor = cache.buffer.narrow(0, slot * tilingSize, tilingSize);
        auto status = aclrtMemcpy(tilingTensor.data_ptr<uint8_t>(), tilingSize, &tilingData, tilingSize,
                                  ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(status == ACL_ERROR_NONE, "compressor: failed to cache tiling data, acl error ", status);
        cache.slots.emplace(hashValue, slot);
        cache.nextSlot++;
    }

    // ---- 6) workspace ----
    at::Tensor workspace =
        at::empty({(int64_t)workspaceSize}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // ---- 7) dispatch: single kernel entry, dispatch by tilingKey inside kernel ----
    at::Tensor stateBlockTable = state_block_table.has_value()
                                     ? state_block_table.value()
                                     : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(x.options().device()));
    at::Tensor cuSeqlensT = cu_seqlens.has_value()
                                ? cu_seqlens.value()
                                : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(x.options().device()));
    at::Tensor seqUsedT = seqused.has_value()
                              ? seqused.value()
                              : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(x.options().device()));
    at::Tensor startPosT = start_pos.has_value()
                               ? start_pos.value()
                               : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(x.options().device()));

    uint8_t templateId = static_cast<uint8_t>((tilingKey >> 11) & 0x3);
    if (templateId == 1) {
        // EMPTY_X: nothing to compute, return empty cmp_kv
        return cmp_kv;
    }

    EXEC_KERNEL_CMD(compressor, blockDim, x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                    stateBlockTable, cuSeqlensT, seqUsedT, startPosT, cmp_kv, state_cache, workspace, tilingTensor);
    return cmp_kv;
}
}  // namespace npu_kernel
}  // namespace sglang
