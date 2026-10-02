/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#pragma once

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
// Load the SDK's GE/GERT definitions before torch_npu, which also ships GE
// headers with the same include guards. Mixing the two sets is not supported.
#include "graph/types.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

namespace sglang::npu_kernel::sparse_attention {

inline int32_t checked_int(int64_t value, const char *name)
{
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<int32_t>::max(), name, " is out of range");
    return static_cast<int32_t>(value);
}

struct InputShape {
    int64_t batch, heads, sequence, dim;
};

inline InputShape check_input(const at::Tensor &x, const std::string &layout, int64_t heads)
{
    TORCH_CHECK(x.device().type() == DEVICE_TYPE, "sparse attention inputs must be NPU tensors");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "sparse attention supports FP16 and BF16");
    checked_int(heads, "head count");
    for (auto size : x.sizes())
        checked_int(size, "input dimension");
    if (layout == "BNSD" || layout == "BSND") {
        TORCH_CHECK(x.dim() == 4, "BNSD and BSND inputs must have rank 4");
        TORCH_CHECK(x.size(layout == "BNSD" ? 1 : 2) == heads, "head count does not match input shape");
        return {x.size(0), heads, x.size(layout == "BNSD" ? 2 : 1), x.size(3)};
    }
    TORCH_CHECK(layout == "BSH" && x.dim() == 3, "supported layouts are BNSD, BSND and BSH");
    TORCH_CHECK(x.size(2) % heads == 0, "hidden size must be divisible by head count");
    return {x.size(0), heads, x.size(1), x.size(2) / heads};
}

inline void check_pair(const at::Tensor &q, const at::Tensor &k, const InputShape &qs, const InputShape &ks)
{
    TORCH_CHECK(q.device() == k.device() && q.scalar_type() == k.scalar_type(),
                "inputs must have the same device and dtype");
    int device = 0;
    c10_npu::GetDevice(&device);
    TORCH_CHECK(q.get_device() == device, "inputs must be on the current NPU device");
    TORCH_CHECK(qs.batch == ks.batch && qs.dim == ks.dim && qs.heads % ks.heads == 0,
                "incompatible query and key/value shapes");
}

inline at::Tensor normalize(const at::Tensor &x, const std::string &layout)
{
    auto result = x.contiguous();
    return layout == "BSND" ? result.view({x.size(0), x.size(1), x.size(2) * x.size(3)}) : result;
}

inline gert::StorageShape storage_shape(at::IntArrayRef sizes)
{
    gert::StorageShape shape;
    for (auto size : sizes) {
        shape.MutableOriginShape().AppendDim(size);
        shape.MutableStorageShape().AppendDim(size);
    }
    return shape;
}

inline at::IntArrayRef check_lengths(c10::OptionalIntArrayRef lengths, const InputShape &shape)
{
    auto values = lengths.value_or(at::IntArrayRef{});
    TORCH_CHECK(values.empty() || values.size() == 1 || values.size() == static_cast<size_t>(shape.batch),
                "actual sequence lengths must have one element or one per batch");
    for (auto value : values) {
        TORCH_CHECK(value > 0 && value <= shape.sequence, "actual sequence length is out of range");
    }
    return values;
}

inline gert::Tensor length_tensor(at::IntArrayRef values)
{
    const int64_t size = static_cast<int64_t>(values.size());
    gert::Tensor tensor(gert::StorageShape({size}, {size}), gert::StorageFormat(), ge::DT_INT64);
    tensor.SetData(gert::TensorData{const_cast<int64_t *>(values.data())});
    return tensor;
}

inline at::Tensor cached_device_copy(const at::Tensor &host)
{
    // Keep the host-to-device copies out of repeated graph captures. Include
    // the stream so that a cached copy is never consumed before it completes.
    static std::mutex mutex;
    static std::unordered_map<int, std::unordered_map<std::string, at::Tensor>> caches;
    int device = 0;
    c10_npu::GetDevice(&device);
    const auto stream = reinterpret_cast<uintptr_t>(c10_npu::getCurrentNPUStream().stream(false));
    std::string key(reinterpret_cast<const char *>(host.data_ptr()), host.nbytes());
    key.append(reinterpret_cast<const char *>(&stream), sizeof(stream));
    const auto dtype = host.scalar_type();
    key.append(reinterpret_cast<const char *>(&dtype), sizeof(dtype));
    std::lock_guard<std::mutex> lock(mutex);
    auto &cache = caches[device];
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    auto result = TorchNpuHelper::CopyTensorHostToDevice(host);
    if (cache.size() < 1024) cache.emplace(std::move(key), result);
    return result;
}

inline at::Tensor device_lengths(at::IntArrayRef values)
{
    if (values.empty()) return at::Tensor();
    auto host = at::tensor(values, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return cached_device_copy(host);
}

inline void *data_or_null(const at::Tensor &tensor)
{
    return tensor.defined() ? tensor.data_ptr() : nullptr;
}

template <typename Tiling>
inline at::Tensor device_tiling(Tiling &tiling)
{
    const int64_t size = (tiling.GetDataSize() + 31) / 32 * 32;
    auto host = at::zeros({size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    tiling.SaveToBuffer(host.data_ptr(), size);
    return cached_device_copy(host);
}

inline platform_ascendc::PlatformAscendC *platform()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get the AscendC platform");
    const auto soc = platform->GetSocVersion();
    TORCH_CHECK(soc == platform_ascendc::SocVersion::ASCEND910B || soc == platform_ascendc::SocVersion::ASCEND910_93,
                "sparse attention is supported only on Ascend A2/A3");
    TORCH_CHECK(platform->GetCoreNumAic() > 0 && platform->GetCoreNumAiv() > 0 && platform->GetCoreNumAiv() <= 64,
                "unsupported sparse attention core count");
    return platform;
}

}  // namespace sglang::npu_kernel::sparse_attention
