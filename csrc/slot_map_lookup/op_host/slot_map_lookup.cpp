// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_slot_map_lookup.h"

#include <limits>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kFixedTopk = 2048;
constexpr uint32_t kMaxContextLenAlign = 8;

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as slot_map");
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max, name, " exceeds uint32 range: ", value);
}

}  // namespace

/*
 * SlotMapLookup reads slot_map[req_indices[b], topk_indices[b, k]] for every
 * query token.
 *
 * Outputs (in-place):
 *   token_on_device[bs, topk]: int32 indicator, 1 for hit and 0 for miss
 *   device_token_pos[bs, topk]: int32 slot position, or -1 for a miss
 */
HOST_API void slot_map_lookup(const at::Tensor &slot_map, const at::Tensor &req_indices, const at::Tensor &topk_indices,
                              at::Tensor &token_on_device, at::Tensor &device_token_pos, int64_t block_dim)
{
    CheckNpuTensor(slot_map, "slot_map");
    CheckNpuTensor(req_indices, "req_indices");
    CheckNpuTensor(topk_indices, "topk_indices");
    CheckNpuTensor(token_on_device, "token_on_device");
    CheckNpuTensor(device_token_pos, "device_token_pos");
    CheckSameDevice(req_indices, slot_map, "req_indices");
    CheckSameDevice(topk_indices, slot_map, "topk_indices");
    CheckSameDevice(token_on_device, slot_map, "token_on_device");
    CheckSameDevice(device_token_pos, slot_map, "device_token_pos");

    TORCH_CHECK(slot_map.is_contiguous(), "slot_map must be contiguous");
    TORCH_CHECK(req_indices.is_contiguous(), "req_indices must be contiguous");
    TORCH_CHECK(topk_indices.is_contiguous(), "topk_indices must be contiguous");
    TORCH_CHECK(token_on_device.is_contiguous(), "token_on_device must be contiguous");
    TORCH_CHECK(device_token_pos.is_contiguous(), "device_token_pos must be contiguous");

    TORCH_CHECK(slot_map.dim() == 2, "slot_map must be 2-D, got ", slot_map.dim());
    TORCH_CHECK(req_indices.dim() == 1, "req_indices must be 1-D, got ", req_indices.dim());
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2-D, got ", topk_indices.dim());
    TORCH_CHECK(token_on_device.dim() == 2, "token_on_device must be 2-D, got ", token_on_device.dim());
    TORCH_CHECK(device_token_pos.dim() == 2, "device_token_pos must be 2-D, got ", device_token_pos.dim());

    TORCH_CHECK(slot_map.scalar_type() == at::kInt, "slot_map must be int32, got ", slot_map.scalar_type());
    TORCH_CHECK(req_indices.scalar_type() == at::kInt, "req_indices must be int32, got ", req_indices.scalar_type());
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt, "topk_indices must be int32, got ", topk_indices.scalar_type());
    TORCH_CHECK(token_on_device.scalar_type() == at::kInt, "token_on_device must be int32, got ",
                token_on_device.scalar_type());
    TORCH_CHECK(device_token_pos.scalar_type() == at::kInt, "device_token_pos must be int32, got ",
                device_token_pos.scalar_type());

    const int64_t size64 = slot_map.size(0);
    const int64_t maxContextLen64 = slot_map.size(1);
    const int64_t bs64 = req_indices.size(0);
    const int64_t topkBs64 = topk_indices.size(0);
    const int64_t topk64 = topk_indices.size(1);
    CheckFitsUint32(size64, "slot_map.size(0)");
    CheckFitsUint32(maxContextLen64, "slot_map.size(1)");
    CheckFitsUint32(bs64, "req_indices.size(0)");
    CheckFitsUint32(topk64, "topk_indices.size(1)");
    TORCH_CHECK(block_dim >= 0, "block_dim must be non-negative, got ", block_dim);
    CheckFitsUint32(block_dim, "block_dim");
    TORCH_CHECK(static_cast<uint64_t>(size64) * static_cast<uint64_t>(maxContextLen64) <= kUint32Max,
                "slot_map is too large for the kernel uint32 row offsets");
    TORCH_CHECK(static_cast<uint64_t>(bs64) * static_cast<uint64_t>(topk64) <= kUint32Max,
                "bs * topk exceeds the kernel uint32 output offsets");

    const uint32_t size = static_cast<uint32_t>(size64);
    const uint32_t maxContextLen = static_cast<uint32_t>(maxContextLen64);
    const uint32_t bs = static_cast<uint32_t>(bs64);
    const uint32_t topkBs = static_cast<uint32_t>(topkBs64);
    const uint32_t topk = static_cast<uint32_t>(topk64);

    TORCH_CHECK(size > 0 && maxContextLen > 0, "slot_map dimensions must be positive, got size=", size,
                " max_context_len=", maxContextLen);
    TORCH_CHECK(bs > 0, "bs must be positive, got ", bs);
    TORCH_CHECK(topk == kFixedTopk, "slot_map_lookup requires topk=", kFixedTopk, ", got ", topk);
    TORCH_CHECK(maxContextLen % kMaxContextLenAlign == 0,
                "slot_map_lookup requires max_context_len to be a multiple of ", kMaxContextLenAlign, ", got ",
                maxContextLen);
    TORCH_CHECK(bs == topkBs, "req_indices dim0 must match topk_indices dim0: ", bs, " vs ", topkBs);
    TORCH_CHECK(
        static_cast<uint32_t>(token_on_device.size(0)) == bs && static_cast<uint32_t>(token_on_device.size(1)) == topk,
        "token_on_device shape must match [bs, topk] = [", bs, ", ", topk, "]");
    TORCH_CHECK(static_cast<uint32_t>(device_token_pos.size(0)) == bs &&
                    static_cast<uint32_t>(device_token_pos.size(1)) == topk,
                "device_token_pos shape must match [bs, topk] = [", bs, ", ", topk, "]");

    const uint32_t effectiveBlockDim = block_dim > 0 ? static_cast<uint32_t>(block_dim) : 8;

    auto npuStream = c10_npu::getCurrentNPUStream();
    slot_map.record_stream(npuStream);
    req_indices.record_stream(npuStream);
    topk_indices.record_stream(npuStream);
    token_on_device.record_stream(npuStream);
    device_token_pos.record_stream(npuStream);

    EXEC_KERNEL_CMD(slot_map_lookup, effectiveBlockDim, slot_map, req_indices, topk_indices, token_on_device,
                    device_token_pos, size, maxContextLen, bs, topk);
}

}  // namespace npu_kernel
}  // namespace sglang
