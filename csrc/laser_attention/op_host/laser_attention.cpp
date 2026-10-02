/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "aclrtlaunch_laser_attention.h"
#include "defines.h"
#include "laser_attention_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kMaxToken = std::numeric_limits<int32_t>::max();
constexpr int64_t kBlockSide = 128;
constexpr int64_t kTilingAlignment = 32;
constexpr int64_t kMaxCachedTilings = 1024;
static_assert(sizeof(LaserAttentionTilingData) == 80, "unexpected Laser Attention tiling ABI");
static_assert(std::is_trivially_copyable<LaserAttentionTilingData>::value,
              "Laser Attention tiling data must be trivially copyable");

struct LaserAttentionLaunchConfig {
    LaserAttentionTilingData tilingData;
    uint32_t blockDim = 0;
    int64_t workspaceSize = 0;
};

struct DeviceTilingCache {
    at::Tensor buffer;
    std::unordered_map<std::string, int64_t> slots;
};

std::mutex gTilingCacheMutex;
std::unordered_map<int64_t, DeviceTilingCache> gTilingCaches;

int32_t checked_int32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int32_t>::max(), name, " is out of int32 range: ", value);
    return static_cast<int32_t>(value);
}

uint64_t checked_mul(uint64_t lhs, uint64_t rhs, const char *name)
{
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs, name, " size overflow");
    return lhs * rhs;
}

uint64_t checked_add(uint64_t lhs, uint64_t rhs, const char *name)
{
    TORCH_CHECK(rhs <= std::numeric_limits<uint64_t>::max() - lhs, name, " size overflow");
    return lhs + rhs;
}

void check_inputs(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, int64_t headNum,
                  const std::string &inputLayout, double scaleValue, double keepProb, int64_t preTokens,
                  int64_t nextTokens)
{
    TORCH_CHECK(query.defined() && key.defined() && value.defined(), "query, key, and value must be defined");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "laser_attn expects query, key, and value in rank-4 BNSD layout");
    TORCH_CHECK(inputLayout == "BNSD", "laser_attn only supports input_layout='BNSD'");

    TORCH_CHECK(query.scalar_type() == at::kHalf, "laser_attn currently supports float16 inputs only");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
                "query, key, and value must have the same dtype");
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
                "query, key, and value must be on the same device");

    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 && query.size(3) > 0,
                "query dimensions must be greater than zero");
    TORCH_CHECK(key.size(0) > 0 && key.size(1) > 0 && key.size(2) > 0 && key.size(3) > 0,
                "key dimensions must be greater than zero");
    TORCH_CHECK(value.size(0) > 0 && value.size(1) > 0 && value.size(2) > 0 && value.size(3) > 0,
                "value dimensions must be greater than zero");

    TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0),
                "query, key, and value batch sizes must match");
    TORCH_CHECK(key.size(1) == value.size(1), "key and value head counts must match");
    TORCH_CHECK(key.size(2) == value.size(2), "key and value sequence lengths must match");
    TORCH_CHECK(query.size(3) == kBlockSide && key.size(3) == kBlockSide && value.size(3) == kBlockSide,
                "laser_attn requires head dimension 128");
    TORCH_CHECK(query.size(1) % key.size(1) == 0,
                "query head count must be divisible by the key/value head count");
    TORCH_CHECK(headNum == query.size(1), "head_num must match query.size(1)");
    TORCH_CHECK(query.size(2) % kBlockSide == 0 && key.size(2) % kBlockSide == 0,
                "query and key/value sequence lengths must be multiples of 128");

    checked_int32(query.size(0), "batch size");
    checked_int32(query.size(1), "head count");
    checked_int32(query.size(2), "query sequence length");
    checked_int32(key.size(2), "key sequence length");
    checked_int32(value.size(2), "value sequence length");
    checked_int32(headNum, "head_num");
    checked_int32(preTokens, "pre_tokens");
    checked_int32(nextTokens, "next_tokens");

    TORCH_CHECK(std::isfinite(scaleValue), "scale_value must be finite");
    TORCH_CHECK(std::isfinite(keepProb) && keepProb > 0.0 && keepProb <= 1.0,
                "keep_prob must be in the interval (0, 1]");
}

void check_optional_mask(const c10::optional<at::Tensor> &mask, const at::Tensor &query, at::ScalarType dtype,
                         const char *name)
{
    if (!mask.has_value() || !mask->defined() || mask->numel() == 0) {
        return;
    }
    TORCH_CHECK(mask->device() == query.device(), name, " must be on the same device as query");
    TORCH_CHECK(mask->scalar_type() == dtype, name, " has an unsupported dtype");
}

LaserAttentionLaunchConfig make_launch_config(const at::Tensor &query, const at::Tensor &key,
                                               const at::Tensor &value, double scaleValue, int64_t headNum,
                                               double keepProb, int64_t preTokens, int64_t nextTokens,
                                               bool isHighPrecision)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get the AscendC platform");

    const uint32_t aicNum = platform->GetCoreNumAic();
    TORCH_CHECK(aicNum > 0 && aicNum <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                "invalid AIC core count: ", aicNum);
    // The 25-AIC platform launches 24 mixed core groups. Keep the scheduler's
    // logical core count equal to the actual launch block count.
    const uint32_t usableAicNum = aicNum == 25 ? 24 : aicNum;

    LaserAttentionLaunchConfig config;
    auto &tiling = config.tilingData;
    tiling.batchSize = checked_int32(query.size(0), "batch size");
    tiling.headNum = checked_int32(headNum, "head_num");
    tiling.seqSize = checked_int32(query.size(2), "query sequence length");
    tiling.headDim = checked_int32(query.size(3), "head dimension");
    tiling.qSeqLength = checked_int32(query.size(2), "query sequence length");
    tiling.kSeqLength = checked_int32(key.size(2), "key sequence length");
    tiling.vSeqLength = checked_int32(value.size(2), "value sequence length");
    tiling.maskSeqLength = 0;
    tiling.scale = static_cast<float>(scaleValue);
    tiling.keepProb = static_cast<float>(keepProb);
    tiling.preTokens = checked_int32(preTokens, "pre_tokens");
    tiling.nextTokens = checked_int32(nextTokens, "next_tokens");
    tiling.isTriangle = 0;
    tiling.attenType = query.size(1) == key.size(1) ? 0 : 1;
    tiling.sparseMode = 0;
    tiling.headGroupSize = checked_int32(query.size(1) / key.size(1), "head group size");
    tiling.windowLen = preTokens == kMaxToken ? 0 : checked_int32(preTokens, "pre_tokens");
    tiling.isHighPrecision = isHighPrecision ? 1 : 0;

    const int32_t colSize = tiling.kSeqLength;
    if (colSize <= 4 * 1024) {
        tiling.coreNumPerGroup = 1;
    } else if (colSize <= 8 * 1024) {
        tiling.coreNumPerGroup = 2;
    } else if (colSize <= 16 * 1024) {
        tiling.coreNumPerGroup = 4;
    } else {
        tiling.coreNumPerGroup = aicNum == 20 ? 4 : 8;
    }
    tiling.coreGroupNum = static_cast<int32_t>(usableAicNum) / tiling.coreNumPerGroup;
    TORCH_CHECK(tiling.coreGroupNum > 0, "not enough AIC cores for laser_attn");

    config.blockDim = static_cast<uint32_t>(tiling.coreGroupNum * tiling.coreNumPerGroup);

    const uint64_t groupNum = checked_mul(static_cast<uint64_t>(tiling.coreGroupNum),
                                          static_cast<uint64_t>(tiling.coreNumPerGroup), "workspace");
    const uint64_t cubeWorkspace = checked_mul(groupNum, 128ULL * 128ULL * 32ULL * 2ULL * 4ULL, "workspace");
    const uint64_t vectorWorkspace =
        checked_mul(groupNum, 256ULL * 128ULL * 8ULL * 2ULL * 4ULL * 2ULL, "workspace");
    const uint64_t rowCount = checked_mul(
        checked_mul(static_cast<uint64_t>(tiling.batchSize), static_cast<uint64_t>(tiling.headNum), "workspace"),
        static_cast<uint64_t>(tiling.seqSize), "workspace");
    const uint64_t rowSumWorkspace = checked_mul(rowCount, sizeof(float), "workspace");
    const uint64_t systemWorkspace = static_cast<uint64_t>(platform->GetLibApiWorkSpaceSize());

    uint64_t workspaceSize = checked_add(cubeWorkspace, vectorWorkspace, "workspace");
    workspaceSize = checked_add(workspaceSize, rowSumWorkspace, "workspace");
    workspaceSize = checked_add(workspaceSize, systemWorkspace, "workspace");
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "laser_attn workspace is too large");
    config.workspaceSize = static_cast<int64_t>(workspaceSize);
    return config;
}

at::Tensor copy_tiling_to_device(const LaserAttentionTilingData &tilingData)
{
    constexpr int64_t tilingSize =
        (static_cast<int64_t>(sizeof(LaserAttentionTilingData)) + kTilingAlignment - 1) / kTilingAlignment *
        kTilingAlignment;
    auto cpuTiling = at::zeros({tilingSize}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(LaserAttentionTilingData));
    return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
}

at::Tensor get_tiling_tensor(const LaserAttentionTilingData &tilingData, const at::Tensor &query)
{
    constexpr int64_t tilingSize =
        (static_cast<int64_t>(sizeof(LaserAttentionTilingData)) + kTilingAlignment - 1) / kTilingAlignment *
        kTilingAlignment;
    const int64_t deviceIndex = query.get_device();
    int currentDevice = 0;
    c10_npu::GetDevice(&currentDevice);
    TORCH_CHECK(deviceIndex == currentDevice, "query must be on the current NPU device");

    std::string key(reinterpret_cast<const char *>(&tilingData), sizeof(LaserAttentionTilingData));
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto streamKey = reinterpret_cast<uintptr_t>(stream);
    key.append(reinterpret_cast<const char *>(&streamKey), sizeof(streamKey));
    std::lock_guard<std::mutex> lock(gTilingCacheMutex);
    auto &cache = gTilingCaches[deviceIndex];
    if (!cache.buffer.defined()) {
        cache.buffer = at::empty({tilingSize * kMaxCachedTilings}, query.options().dtype(at::kByte));
    }

    const auto cached = cache.slots.find(key);
    if (cached != cache.slots.end()) {
        return cache.buffer.narrow(0, cached->second * tilingSize, tilingSize);
    }

    if (static_cast<int64_t>(cache.slots.size()) >= kMaxCachedTilings) {
        return copy_tiling_to_device(tilingData);
    }

    const int64_t slot = static_cast<int64_t>(cache.slots.size());
    auto deviceTiling = copy_tiling_to_device(tilingData);
    auto cachedTiling = cache.buffer.narrow(0, slot * tilingSize, tilingSize);
    cachedTiling.copy_(deviceTiling);
    cache.slots.emplace(key, slot);
    return cachedTiling;
}

at::Tensor prepare_optional_mask(const c10::optional<at::Tensor> &mask)
{
    if (!mask.has_value() || !mask->defined() || mask->numel() == 0) {
        return at::Tensor();
    }
    return mask->contiguous();
}

}  // namespace

HOST_API std::tuple<at::Tensor, at::Tensor> laser_attn(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &attenMaskOpt, const c10::optional<at::Tensor> &alibiMaskOpt,
    const c10::optional<at::Tensor> &dropMaskOpt, double scaleValue, int64_t headNum,
    const std::string &inputLayout, double keepProb, int64_t preTokens, int64_t nextTokens, bool isHighPrecision)
{
    check_inputs(query, key, value, headNum, inputLayout, scaleValue, keepProb, preTokens, nextTokens);
    check_optional_mask(attenMaskOpt, query, at::kHalf, "atten_mask");
    check_optional_mask(alibiMaskOpt, query, at::kHalf, "alibi_mask");
    check_optional_mask(dropMaskOpt, query, at::kByte, "drop_mask");

    const at::Tensor contiguousQuery = query.contiguous();
    const at::Tensor contiguousKey = key.contiguous();
    const at::Tensor contiguousValue = value.contiguous();
    const at::Tensor attenMask = prepare_optional_mask(attenMaskOpt);
    const at::Tensor alibiMask = prepare_optional_mask(alibiMaskOpt);
    const at::Tensor dropMask = prepare_optional_mask(dropMaskOpt);

    void *attenMaskPtr = attenMask.defined() ? attenMask.data_ptr() : nullptr;
    void *alibiMaskPtr = alibiMask.defined() ? alibiMask.data_ptr() : nullptr;
    void *dropMaskPtr = dropMask.defined() ? dropMask.data_ptr() : nullptr;

    auto softmaxLogMaxSum = at::empty({query.size(0), query.size(1), query.size(2)},
                                      query.options().dtype(at::kFloat));
    auto attentionOut = at::empty(query.sizes(), query.options().dtype(at::kFloat));

    const auto config = make_launch_config(query, key, value, scaleValue, headNum, keepProb, preTokens, nextTokens,
                                           isHighPrecision);
    const uint32_t blockDim = config.blockDim;
    auto tilingTensor = get_tiling_tensor(config.tilingData, query);
    auto workspace = at::empty({config.workspaceSize}, query.options().dtype(at::kByte));

    EXEC_KERNEL_CMD(laser_attention, blockDim, contiguousQuery, contiguousKey, contiguousValue, attenMaskPtr,
                    alibiMaskPtr, dropMaskPtr, softmaxLogMaxSum, attentionOut, workspace, tilingTensor);
    return std::make_tuple(softmaxLogMaxSum, attentionOut);
}

}  // namespace sglang::npu_kernel
