// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include <ATen/ATen.h>

namespace sglang::data_move::detail {

// Both kernels compute packed outer pitches from sizes. A payload permutation
// is valid when its bytes remain dense; gaps between slots/layers are not.
inline void validate_packed_buffers(const at::Tensor &device, const at::Tensor &host)
{
    TORCH_CHECK(device.device().type() == c10::DeviceType::PrivateUse1, "device buffer must be on NPU");
    TORCH_CHECK(host.device().is_cpu(), "host buffer must be on CPU");
    TORCH_CHECK(host.is_pinned(), "host buffer must use pinned memory");
    TORCH_CHECK(device.scalar_type() == host.scalar_type(), "device and host buffers must have the same dtype");
    TORCH_CHECK(host.is_contiguous(), "host buffer must be contiguous");

    int64_t elements = 1;
    std::vector<std::pair<int64_t, int64_t>> payload_dims;
    for (int64_t dim = 2; dim < device.dim(); ++dim) {
        elements *= device.size(dim);
        if (device.size(dim) > 1) {
            payload_dims.emplace_back(device.stride(dim), device.size(dim));
        }
    }
    std::sort(payload_dims.begin(), payload_dims.end());
    int64_t expected_stride = 1;
    for (const auto &[stride, size] : payload_dims) {
        TORCH_CHECK(stride == expected_stride, "device slot payload must be physically dense");
        expected_stride *= size;
    }
    TORCH_CHECK(device.size(1) == 1 || device.stride(1) == elements, "device slot pitch must be packed");
    TORCH_CHECK(device.size(0) == 1 || device.stride(0) == device.size(1) * elements,
                "device layer pitch must be packed");
}

// Read only the first token of each page group, matching the existing ABI.
// Validate every pair before submitting any asynchronous copy.
inline std::vector<std::pair<int64_t, int64_t>> validate_transfer_indices(const at::Tensor &device_indices,
                                                                          const at::Tensor &host_indices,
                                                                          int64_t device_slots, int64_t host_slots,
                                                                          int64_t page_size = 1)
{
    TORCH_CHECK(device_indices.dim() == 1 && host_indices.dim() == 1 && device_indices.scalar_type() == at::kLong &&
                    host_indices.scalar_type() == at::kLong,
                "indices must be one-dimensional int64 tensors");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device and host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page size");
    const auto device_cpu = device_indices.cpu().contiguous();
    const auto host_cpu = host_indices.cpu().contiguous();
    const auto *device_data = device_cpu.data_ptr<int64_t>();
    const auto *host_data = host_cpu.data_ptr<int64_t>();
    std::vector<std::pair<int64_t, int64_t>> pairs;
    pairs.reserve(device_cpu.numel() / page_size);
    for (int64_t i = 0; i < device_cpu.numel(); i += page_size) {
        TORCH_CHECK(device_data[i] >= 0 && host_data[i] >= 0, "transfer indices must be non-negative");
        const auto device_slot = device_data[i] / page_size;
        const auto host_slot = host_data[i] / page_size;
        TORCH_CHECK(device_slot < device_slots && host_slot < host_slots,
                    "transfer index exceeds buffer slot/page count");
        pairs.emplace_back(device_slot, host_slot);
    }
    return pairs;
}

inline void check_copy(aclError result)
{
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtMemcpy2dAsync failed: error=", static_cast<int64_t>(result));
}

}  // namespace sglang::data_move::detail
