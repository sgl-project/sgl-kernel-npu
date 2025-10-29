// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acl/acl.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

constexpr int64_t KV_TRANS_FLAG_1D = 1 << 0;
constexpr int64_t KV_TRANS_FLAG_2D = 1 << 1;

enum TransferDirection : int64_t {
    H2D = 1,
    D2H = 2,
};

// @direction: only support 1 or 2, 1 is H2D, 2 is D2H
// @flags: only support 2
HOST_API void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k,
                                       at::Tensor &device_v, at::Tensor &host_v,
                                       const at::Tensor &device_indices,
                                       const at::Tensor &host_indices, int64_t page_size,
                                       int64_t direction, int64_t flags)
{
    TORCH_CHECK(device_k.numel() != 0, "device_k must not be empty");
    TORCH_CHECK(host_k.numel() != 0, "host_k must not be empty");
    TORCH_CHECK(device_k.dim() == host_k.dim(), "the number of dimensions of device_k must be equal to host_k");
    TORCH_CHECK(device_k.dim() == 5, "the number of dimensions of device_k must be 5");
    TORCH_CHECK(device_k.sizes()[0] == host_k.sizes()[1], "the layer number of device_k must be equal to host_k");
    TORCH_CHECK(device_k.sizes()[2] == page_size, "the 3rd dimension of device_k must be equal to page size");
    TORCH_CHECK(host_k.sizes()[2] == page_size, "the 3rd dimension of host_k must be equal to page size");
    TORCH_CHECK(page_size > 0, "Page size must be positive");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device and host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "device indices size must be divisible by page size");
    TORCH_CHECK(direction == static_cast<int64_t>(TransferDirection::H2D) ||
                    direction == static_cast<int64_t>(TransferDirection::D2H),
                "direction must be equal to 1(h2d) or 2(d2h)")
    TORCH_CHECK((flags & KV_TRANS_FLAG_2D) == KV_TRANS_FLAG_2D, "now only support 2d(flags=2) copy");

    if (device_v.numel() != 0 && host_v.numel() != 0) {
        TORCH_CHECK(device_v.dim() == host_v.dim(), "the number of dimensions of device_v must be equal to host_v");
        TORCH_CHECK(device_v.dim() == 5, "the number of dimensions of device_v must be 5");
        TORCH_CHECK(device_v.sizes()[0] == host_v.sizes()[1], "the layer number of device_v must be equal to host_v");
        TORCH_CHECK(device_v.sizes()[2] == page_size, "the 3rd dimension of device_v must be equal to page size");
        TORCH_CHECK(host_v.sizes()[2] == page_size, "the 3rd dimension of host_v must be equal to page size");
    }

    auto device_indices_cpu = device_indices.cpu();
    auto host_indices_cpu = host_indices.cpu();
    const int64_t num_pages = device_indices.size(0) / page_size;
    const int64_t device_pages_num = device_k.sizes()[1];
    const int64_t host_pages_num = host_k.sizes()[0];
    const int64_t total_num_layers = device_k.sizes()[0];
    const auto heads_num = device_k.sizes()[3];
    const auto head_dim = device_k.sizes()[4];
    const auto item_size = device_k.element_size();
    const auto device_pitch = device_pages_num * page_size * heads_num * head_dim * item_size;
    const auto host_pitch = page_size * heads_num * head_dim * item_size;
    const auto width = page_size * heads_num * head_dim * item_size;
    const auto height = total_num_layers;
    c10_npu::NPUStream current_stream = c10_npu::getCurrentNPUStream();
    aclrtStream acl_stream = current_stream.stream();

    for (const auto i : c10::irange(num_pages)) {
        auto device_page_index = device_indices_cpu[i * page_size].item<int64_t>() / page_size;
        auto host_page_index = host_indices_cpu[i * page_size].item<int64_t>() / page_size;
        TORCH_CHECK(device_page_index < device_k.sizes()[1],
                    "device_page_index must be less than the 2nd dim of device_k");
        TORCH_CHECK(host_page_index < host_k.sizes()[0], "host_page_index must be less than the 1st dim of host_k");

        void *device_k_ptr = reinterpret_cast<void *>(device_k[0][device_page_index].data_ptr());
        void *host_k_ptr = reinterpret_cast<void *>(host_k[host_page_index][0].data_ptr());
        if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
            aclrtMemcpy2dAsync(host_k_ptr, host_pitch, device_k_ptr, device_pitch, width, height,
                               aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
        } else {
            aclrtMemcpy2dAsync(device_k_ptr, device_pitch, host_k_ptr, host_pitch, width, height,
                               aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
        }

        if (device_v.numel() != 0 && host_v.numel() != 0) {
            void *device_v_ptr = reinterpret_cast<void *>(device_v[0][device_page_index].data_ptr());
            void *host_v_ptr = reinterpret_cast<void *>(host_v[host_page_index][0].data_ptr());
            if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
                aclrtMemcpy2dAsync(host_v_ptr, host_pitch, device_v_ptr, device_pitch, width, height,
                                   aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
            } else {
                aclrtMemcpy2dAsync(device_v_ptr, device_pitch, host_v_ptr, host_pitch, width, height,
                                   aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
            }
        }
    }
    aclrtSynchronizeStream(acl_stream);
}

}  // namespace npu_kernel
}  // namespace sglang
