// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/DeviceGuard.h>

#include "acl/acl.h"
#include "defines.h"
#include "kv_h2d_copy_plan.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

constexpr int64_t KV_TRANS_FLAG_1D = 1 << 0;
constexpr int64_t KV_TRANS_FLAG_2D = 1 << 1;
constexpr int64_t KV_TRANS_FLAG_COALESCE_H2D = 1 << 2;

enum TransferDirection : int64_t {
    H2D = 1,
    D2H = 2,
};

namespace {
std::vector<int64_t> page_ids(const at::Tensor &indices, int64_t page_size, int64_t limit)
{
    const auto first = indices.slice(0, 0, indices.numel(), page_size).to(at::kLong).contiguous();
    const auto *tokens = first.data_ptr<int64_t>();
    std::vector<int64_t> pages;
    pages.reserve(first.numel());
    for (const auto i : c10::irange(first.numel())) {
        TORCH_CHECK(tokens[i] >= 0 && tokens[i] / page_size < limit, "KV page index is out of range");
        pages.push_back(tokens[i] / page_size);
    }
    return pages;
}

void copy_dense_k_h2d(const at::Tensor &device, const at::Tensor &host, const at::Tensor &device_indices,
                      const at::Tensor &host_indices, int64_t page_size, aclrtStream stream)
{
    const auto device_pages = page_ids(device_indices, page_size, device.size(1));
    const auto host_pages = page_ids(host_indices, page_size, host.size(0));
    const size_t width = page_size * device.size(3) * device.size(4) * device.element_size();
    auto *dst = static_cast<uint8_t *>(device.data_ptr());
    auto *src = static_cast<const uint8_t *>(host.data_ptr());
    visit_kv_h2d_copies(device_pages, host_pages, device.size(0), device.size(1), width, [&](const KvH2dCopy &copy) {
        const auto rc = aclrtMemcpy2dAsync(dst + copy.device_offset, copy.device_pitch, src + copy.host_offset,
                                           copy.host_pitch, copy.width, copy.height, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        TORCH_CHECK(rc == ACL_SUCCESS, "KV H2D copy submission failed: ", rc);
    });
}
}  // namespace

// @direction: only support 1 or 2, 1 is H2D, 2 is D2H
// The 2D flag is required; COALESCE_H2D is an optional modifier for K-only transfers.
HOST_API void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k, at::Tensor &device_v,
                                       at::Tensor &host_v, const at::Tensor &device_indices,
                                       const at::Tensor &host_indices, int64_t page_size, int64_t direction,
                                       int64_t flags)
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
    c10_npu::NPUStream current_stream = c10_npu::getCurrentNPUStream();
    aclrtStream acl_stream = current_stream.stream();
    if (direction == TransferDirection::H2D && (flags & KV_TRANS_FLAG_COALESCE_H2D) && device_v.numel() == 0 &&
        host_v.numel() == 0 && device_indices.dim() == 1 && host_indices.dim() == 1 && device_k.is_contiguous() &&
        host_k.is_contiguous() && device_k.scalar_type() == host_k.scalar_type() &&
        device_k.size(3) == host_k.size(3) && device_k.size(4) == host_k.size(4)) {
        copy_dense_k_h2d(device_k, host_k, device_indices_cpu, host_indices_cpu, page_size, acl_stream);
        return;
    }
    const int64_t device_pages_num = device_k.sizes()[1];
    const int64_t host_pages_num = host_k.sizes()[0];
    const int64_t total_num_layers = device_k.sizes()[0];
    const auto heads_num = device_k.sizes()[3];
    const auto item_size = device_k.element_size();
    const auto k_head_dim = device_k.sizes()[4];
    const auto k_device_pitch = device_pages_num * page_size * heads_num * k_head_dim * item_size;
    const auto k_host_pitch = page_size * heads_num * k_head_dim * item_size;
    const auto k_width = page_size * heads_num * k_head_dim * item_size;
    const auto v_head_dim = device_v.numel() == 0 ? 0 : device_v.size(4);
    const auto v_device_pitch = device_pages_num * page_size * heads_num * v_head_dim * item_size;
    const auto v_host_pitch = page_size * heads_num * v_head_dim * item_size;
    const auto v_width = page_size * heads_num * v_head_dim * item_size;
    const auto height = total_num_layers;

    const int64_t num_pages = device_indices.size(0) / page_size;
    for (const auto i : c10::irange(num_pages)) {
        auto device_page_index = device_indices_cpu[i * page_size].item<int64_t>() / page_size;
        auto host_page_index = host_indices_cpu[i * page_size].item<int64_t>() / page_size;
        TORCH_CHECK(device_page_index < device_k.sizes()[1],
                    "device_page_index must be less than the 2nd dim of device_k");
        TORCH_CHECK(host_page_index < host_k.sizes()[0], "host_page_index must be less than the 1st dim of host_k");

        void *device_k_ptr = reinterpret_cast<void *>(device_k[0][device_page_index].data_ptr());
        void *host_k_ptr = reinterpret_cast<void *>(host_k[host_page_index][0].data_ptr());
        if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
            aclrtMemcpy2dAsync(host_k_ptr, k_host_pitch, device_k_ptr, k_device_pitch, k_width, height,
                               aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
        } else {
            aclrtMemcpy2dAsync(device_k_ptr, k_device_pitch, host_k_ptr, k_host_pitch, k_width, height,
                               aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
        }

        if (device_v.numel() != 0 && host_v.numel() != 0) {
            void *device_v_ptr = reinterpret_cast<void *>(device_v[0][device_page_index].data_ptr());
            void *host_v_ptr = reinterpret_cast<void *>(host_v[host_page_index][0].data_ptr());
            if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
                aclrtMemcpy2dAsync(host_v_ptr, v_host_pitch, device_v_ptr, v_device_pitch, v_width, height,
                                   aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
            } else {
                aclrtMemcpy2dAsync(device_v_ptr, v_device_pitch, host_v_ptr, v_host_pitch, v_width, height,
                                   aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
            }
        }
    }
}

// Port of the GLM-5.2 queued H2D implementation measured on Ascend 910B4.
// Tensor handles are captured through submission; the Python wrapper retains them
// through a completion event. Each component uses its actual outer byte strides.
struct QueuedH2dCopy {
    uintptr_t dst, src;
    size_t dpitch, spitch, width, height;
};
struct QueuedH2dRun {
    int64_t h, d, n;
};

HOST_API std::vector<int64_t> transfer_kv_dim_exchange_h2d(std::vector<at::Tensor> devices,
                                                           std::vector<at::Tensor> hosts, at::Tensor host_idx,
                                                           at::Tensor device_idx, int64_t page_size, bool coalesce)
{
    TORCH_CHECK(page_size > 0 && !devices.empty() && devices.size() == hosts.size(), "invalid pools");
    TORCH_CHECK(host_idx.device().is_cpu() && device_idx.device().is_cpu(), "indices must already be on CPU");
    TORCH_CHECK(host_idx.scalar_type() == at::kLong && device_idx.scalar_type() == at::kLong, "indices must be int64");
    TORCH_CHECK(host_idx.dim() == 1 && device_idx.dim() == 1 && host_idx.numel() == device_idx.numel() &&
                    host_idx.numel() % page_size == 0,
                "complete matching token pages required");
    auto hi = host_idx.contiguous();
    auto di = device_idx.contiguous();
    const int64_t *hp = hi.data_ptr<int64_t>();
    const int64_t *dp = di.data_ptr<int64_t>();
    const int64_t pages = hi.numel() / page_size;
    std::vector<QueuedH2dRun> runs;
    int64_t maxh = -1, maxd = -1;
    for (int64_t i = 0; i < pages; ++i) {
        int64_t h = hp[i * page_size], d = dp[i * page_size];
        TORCH_CHECK(h >= 0 && d >= 0 && h % page_size == 0 && d % page_size == 0, "page alignment invalid");
        for (int64_t j = 1; j < page_size; ++j)
            TORCH_CHECK(hp[i * page_size + j] == h + j && dp[i * page_size + j] == d + j,
                        "indices do not describe whole contiguous pages");
        h /= page_size;
        d /= page_size;
        // Keep repeated destinations in input order, as required by the public API.
        maxh = std::max(maxh, h);
        maxd = std::max(maxd, d);
        if (coalesce && !runs.empty() && h == runs.back().h + runs.back().n && d == runs.back().d + runs.back().n)
            ++runs.back().n;
        else
            runs.push_back({h, d, 1});
    }
    at::DeviceGuard guard(devices.front().device());
    std::vector<QueuedH2dCopy> plan;
    int64_t total_bytes = 0, stock_calls = 0;
    for (size_t k = 0; k < devices.size(); ++k) {
        auto &dev = devices[k];
        auto &host = hosts[k];
        TORCH_CHECK(dev.device().type() == c10::DeviceType::PrivateUse1 && dev.device() == devices.front().device() &&
                        host.device().is_cpu(),
                    "pool devices invalid");
        TORCH_CHECK(dev.dim() == 5 && host.dim() == 5 && dev.scalar_type() == host.scalar_type(),
                    "pool shape/dtype invalid");
        TORCH_CHECK(dev.size(0) == host.size(1) && dev.size(2) == page_size && host.size(2) == page_size &&
                        dev.size(3) == host.size(3) && dev.size(4) == host.size(4),
                    "pool dimensions differ");
        TORCH_CHECK(dev.stride(4) == 1 && host.stride(4) == 1 && dev.stride(3) == dev.size(4) &&
                        host.stride(3) == host.size(4) && dev.stride(2) == dev.size(3) * dev.size(4) &&
                        host.stride(2) == host.size(3) * host.size(4),
                    "inner dimensions must be contiguous");
        TORCH_CHECK(maxd < dev.size(1) && maxh < host.size(0), "page index out of bounds");
        if (!pages) continue;
        const int64_t elem = dev.element_size(), layers = dev.size(0);
        const int64_t width = page_size * dev.size(3) * dev.size(4) * elem;
        const int64_t ds0 = dev.stride(0) * elem, ds1 = dev.stride(1) * elem;
        const int64_t hs0 = host.stride(0) * elem, hs1 = host.stride(1) * elem;
        TORCH_CHECK(
            layers > 0 && width > 0 && width <= 5000000 && ds0 >= width && ds1 >= width && hs0 >= width && hs1 >= width,
            "unsupported copy pitches");
        TORCH_CHECK((layers - 1) * ds0 + maxd * ds1 + width <=
                            static_cast<int64_t>(dev.storage().nbytes()) - dev.storage_offset() * elem &&
                        maxh * hs0 + (layers - 1) * hs1 + width <=
                            static_cast<int64_t>(host.storage().nbytes()) - host.storage_offset() * elem,
                    "strided copy exceeds storage");
        const uintptr_t db = reinterpret_cast<uintptr_t>(dev.data_ptr());
        const uintptr_t hb = reinterpret_cast<uintptr_t>(host.data_ptr());
        total_bytes += pages * layers * width;
        stock_calls += pages;
        for (const auto &r : runs) {
            if (coalesce && r.n >= layers) {
                for (int64_t l = 0; l < layers; ++l)
                    plan.push_back({db + l * ds0 + r.d * ds1, hb + r.h * hs0 + l * hs1, static_cast<size_t>(ds1),
                                    static_cast<size_t>(hs0), static_cast<size_t>(width), static_cast<size_t>(r.n)});
            } else {
                for (int64_t j = 0; j < r.n; ++j)
                    plan.push_back({db + (r.d + j) * ds1, hb + (r.h + j) * hs0, static_cast<size_t>(ds0),
                                    static_cast<size_t>(hs1), static_cast<size_t>(width), static_cast<size_t>(layers)});
            }
        }
    }
    for (const auto &c : plan)
        TORCH_CHECK(c.height > 0 && c.height <= 5242880, "copy height invalid");
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Capture owning Tensor handles until the host task is submitted. The
    // Python wrapper additionally retains them until a completion event retires.
    if (!plan.empty())
        at_npu::native::OpCommand::RunOpApi("sgl_kernel_queued_h2d", [plan, devices, hosts, stream]() -> int {
            for (const auto &c : plan) {
                int rc = aclrtMemcpy2dAsync(reinterpret_cast<void *>(c.dst), c.dpitch, reinterpret_cast<void *>(c.src),
                                            c.spitch, c.width, c.height, ACL_MEMCPY_HOST_TO_DEVICE, stream);
                if (rc != ACL_SUCCESS) return rc;
            }
            return 0;
        });
    return {total_bytes, static_cast<int64_t>(plan.size()), stock_calls, static_cast<int64_t>(runs.size()), pages};
}

}  // namespace npu_kernel
}  // namespace sglang
