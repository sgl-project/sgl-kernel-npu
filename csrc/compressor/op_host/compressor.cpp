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
#include <mutex>
#include <string>
#include <type_traits>
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
    std::unordered_map<std::string, uint32_t> slots;
    uint32_t nextSlot = 0;
};

template <typename T>
void AppendTilingKey(std::string &key, const T &value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    key.append(reinterpret_cast<const char *>(&value), sizeof(T));
}

std::string MakeTilingCacheKey(const optiling::CompressorTilingData &data)
{
    std::string key;
    key.reserve(sizeof(optiling::CompressorTilingData));
    const auto &base = data.baseParams;
    AppendTilingKey(key, base.batchSize);
    AppendTilingKey(key, base.seqSize);
    AppendTilingKey(key, base.hiddenSize);
    AppendTilingKey(key, base.headDim);
    AppendTilingKey(key, base.cmpRatio);
    AppendTilingKey(key, base.tokenSize);
    AppendTilingKey(key, base.csSize);
    AppendTilingKey(key, base.cgSize);
    AppendTilingKey(key, base.nSize);
    AppendTilingKey(key, base.usedCoreNum);
    AppendTilingKey(key, base.ropeHeadDim);
    AppendTilingKey(key, base.normEps);
    AppendTilingKey(key, base.reciprocalD);
    AppendTilingKey(key, base.stateCacheStrideDim0);
    const auto &page = data.pageAttentionParams;
    AppendTilingKey(key, page.blockNum);
    AppendTilingKey(key, page.blockSize);
    AppendTilingKey(key, page.maxBlockNumPerBatch);
    const auto &split = data.innerSplitParams;
    AppendTilingKey(key, split.mBaseSize);
    AppendTilingKey(key, split.dBaseSize);
    const auto &ws = data.workspaceParams;
    AppendTilingKey(key, ws.mm1KvResSize);
    AppendTilingKey(key, ws.mm1ScoreResSize);
    AppendTilingKey(key, ws.vec1ResSize);
    AppendTilingKey(key, ws.vec1TailCacheSize);
    AppendTilingKey(key, ws.dbWorkspaceRatio);
    AppendTilingKey(key, data.tilingKey);
    return key;
}

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

std::mutex &GetTilingCacheMutex()
{
    static std::mutex cacheMutex;
    return cacheMutex;
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
                               const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                               const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio,
                               int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode,
                               int64_t state_cache_stride_dim0)
{
    using namespace optiling;
    TORCH_CHECK(x.device().type() == DEVICE_TYPE, "compressor: x must be an NPU tensor");
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

    // EMPTY_X: nothing to compute, return empty cmp_kv without touching the tiling cache
    uint8_t templateId = static_cast<uint8_t>((tilingKey >> 11) & 0x3);
    if (templateId == 1) {
        return cmp_kv;
    }

    auto key = MakeTilingCacheKey(tilingData);
    at::Tensor tilingTensor;
    if (IsNpuGraphCapturing()) {
        // ── 入图路径：查缓存，必须命中 ──
        std::lock_guard<std::mutex> lock(GetTilingCacheMutex());
        auto &cache = GetTilingCaches()[x.device().index()];
        TORCH_CHECK(cache.buffer.defined(),
                    "compressor: run one eager warmup before NPU graph capture to initialize the tiling cache");
        auto iter = cache.slots.find(key);
        TORCH_CHECK(iter != cache.slots.end(),
                    "compressor: the current tiling configuration is not cached; run one eager warmup ...");
        tilingTensor = cache.buffer.narrow(0, iter->second * tilingSize, tilingSize);
    } else {
        // ── 不入图路径：查缓存 → 填缓存 → 满则退回 memcpy（lightning_indexer 风格）──
        std::lock_guard<std::mutex> lock(GetTilingCacheMutex());
        auto &cache = GetTilingCaches()[x.device().index()];
        if (!cache.buffer.defined()) {
            cache.buffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                     at::TensorOptions().dtype(at::kByte).device(x.options().device()));
        }
        auto iter = cache.slots.find(key);
        if (iter != cache.slots.end()) {
            tilingTensor = cache.buffer.narrow(0, iter->second * tilingSize, tilingSize);  // 命中复用
        } else if (cache.nextSlot >= MAX_CAPTURE_NUM) {
            // 缓存满：退回一次性 memcpy（局部 tensor，避免 static 跨线程/跨设备覆盖）
            at::Tensor t = at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));
            auto status =
                aclrtMemcpy(t.data_ptr<uint8_t>(), tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
            TORCH_CHECK(status == ACL_ERROR_NONE, "compressor: failed to copy tiling data, acl error ", status);
            tilingTensor = t;
        } else {
            // 正常 MISS：填缓存
            const uint32_t slot = cache.nextSlot;
            tilingTensor = cache.buffer.narrow(0, slot * tilingSize, tilingSize);
            auto status = aclrtMemcpy(tilingTensor.data_ptr<uint8_t>(), tilingSize, &tilingData, tilingSize,
                                      ACL_MEMCPY_HOST_TO_DEVICE);
            TORCH_CHECK(status == ACL_ERROR_NONE, "compressor: failed to cache tiling data, acl error ", status);
            cache.slots.emplace(std::move(key), slot);
            cache.nextSlot++;
        }
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

    EXEC_KERNEL_CMD(compressor, blockDim, x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                    stateBlockTable, cuSeqlensT, seqUsedT, startPosT, cmp_kv, state_cache, workspace, tilingTensor);
    return cmp_kv;
}
}  // namespace npu_kernel
}  // namespace sglang
