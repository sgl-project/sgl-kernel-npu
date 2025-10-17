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

namespace sglang {
namespace npu_kernel {

enum class TransferDirection {
    HOST_TO_DEVICE = 1,
    DEVICE_TO_HOST = 2,
};

// @kind: only support 1 or 2, 1 is host to device, 2 is device to host
HOST_API void transfer_kv(
    at::Tensor &device_k, at::Tensor &host_k,
    at::Tensor &device_v, at::Tensor &host_v,
    const at::Tensor &device_indices, const at::Tensor &host_indices, int64_t kind,
    int64_t start_layer_id, int64_t num_layers, int64_t page_size, bool non_blocking)
{
    TORCH_CHECK(device_k.numel() != 0, "device_k must not be empty");
    TORCH_CHECK(host_k.numel() != 0, "host_k must not be empty");
    TORCH_CHECK(device_k.sizes()[0] == host_k.sizes()[1], "the 1st dimension of device_k must be equal to the 2nd dimension of host_k");
    TORCH_CHECK(device_k.sizes()[1] == host_k.sizes()[0], "the 2nd dimension of device_k must be equal to the 1st dimension of host_k");
    TORCH_CHECK(device_k.dim() == host_k.dim(), "the number of dimensions of device_k must be equal to host_k");
    TORCH_CHECK(device_k.dim() > 2, "the number of dimensions of device_k must be greater than 2");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device and host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "device indices size must be divisible by page size");
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(kind == static_cast<int64_t>(TransferDirection::HOST_TO_DEVICE)
                || kind == static_cast<int64_t>(TransferDirection::DEVICE_TO_HOST),
                "kind must be equal to 1(h2d) or 2(d2h)")

    if (device_v.numel() != 0 && host_v.numel() != 0) {
        TORCH_CHECK(device_v.sizes()[0] == host_v.sizes()[1], "the 1st dimension of device_v must be equal to the 2nd dimension of host_v");
        TORCH_CHECK(device_v.sizes()[1] == host_v.sizes()[0], "the 2nd dimension of device_v must be equal to the 1st dimension of host_v");
        TORCH_CHECK(device_v.dim() == host_v.dim(), "the number of dimensions of device_v must be equal to host_v");
        TORCH_CHECK(device_v.dim() > 2, "the number of dimensions of device_v must be greater than 2");
    }

    auto device_indices_cpu = device_indices.cpu();
    auto host_indices_cpu = host_indices.cpu();
    const int64_t num_pages = device_indices.size(0) / page_size;

    for (const auto i : c10::irange(num_pages)) {
        auto device_page_index = device_indices_cpu[i * page_size].item<int64_t>() / page_size;
        auto host_page_index = host_indices_cpu[i * page_size].item<int64_t>() / page_size;
        for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers; ++layer_id) {
            if (kind == 2) {
                // device -> host
                host_k[host_page_index][layer_id].copy_(device_k[layer_id][device_page_index], non_blocking);
            } else {
                // host -> device
                device_k[layer_id][device_page_index].copy_(host_k[host_page_index][layer_id], non_blocking);
            }

            if (device_v.numel() != 0 && host_v.numel() != 0) {
                if (kind == 2) {
                    // device -> host
                    host_v[host_page_index][layer_id].copy_(device_v[layer_id][device_page_index], non_blocking);
                } else {
                    // host -> device
                    device_v[layer_id][device_page_index].copy_(host_v[host_page_index][layer_id], non_blocking);
                }
            }
        }
    }
}

}  // namespace npu_kernel
}  // namespace sglang
