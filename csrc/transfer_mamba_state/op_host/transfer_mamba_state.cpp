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

enum TransferDirection : int64_t {
    H2D = 1,
    D2H = 2,
};

// Transfer Mamba/SSM state between device (layer-first) and host (page-first)
// with a single slot index per copy (page_size=1).
//
// Device buffer layout: [num_layers, device_size, *state_shape]  (layer-first)
// Host buffer layout:   [host_size, num_layers, 1, *state_shape] (page-first)
//
// For each (device_indices[i], host_indices[i]) pair, copies a 2D block:
//   height = num_layers
//   width  = slot_bytes (product of state_shape dims * element_size)
//
// @direction: 1=H2D (host→device), 2=D2H (device→host)
HOST_API void transfer_mamba_state(at::Tensor &device_buf, at::Tensor &host_buf, const at::Tensor &device_indices,
                                   const at::Tensor &host_indices, int64_t direction)
{
    TORCH_CHECK(device_buf.numel() != 0, "device_buf must not be empty");
    TORCH_CHECK(host_buf.numel() != 0, "host_buf must not be empty");
    TORCH_CHECK(device_buf.dim() >= 2, "device_buf must have at least 2 dims, got %d", device_buf.dim());
    TORCH_CHECK(host_buf.dim() >= 2, "host_buf must have at least 2 dims, got %d", host_buf.dim());
    TORCH_CHECK(device_buf.sizes()[0] == host_buf.sizes()[1],
                "layer count mismatch: device has %lld layers, host has %lld layers", device_buf.sizes()[0],
                host_buf.sizes()[1]);
    TORCH_CHECK(device_buf.dtype() == host_buf.dtype(),
                "device_buf and host_buf must have the same dtype, got %s vs %s", device_buf.dtype().name().data(),
                host_buf.dtype().name().data());
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device and host indices must have the same length");
    TORCH_CHECK(direction == static_cast<int64_t>(TransferDirection::H2D) ||
                    direction == static_cast<int64_t>(TransferDirection::D2H),
                "direction must be 1(H2D) or 2(D2H)");

    const int64_t num_layers = device_buf.sizes()[0];
    const int64_t device_size = device_buf.sizes()[1];

    // Compute slot_bytes = product of all dims after [layer, slot] * element_size
    int64_t slot_bytes = device_buf.element_size();
    for (int64_t i = 2; i < device_buf.dim(); i++) {
        slot_bytes *= device_buf.sizes()[i];
    }

    // Verify host state shape is compatible: host layout is [slot, layer, 1, *state]
    // so host slot_bytes must equal device slot_bytes
    int64_t host_slot_bytes = host_buf.element_size();
    for (int64_t i = 2; i < host_buf.dim(); i++) {
        host_slot_bytes *= host_buf.sizes()[i];
    }
    TORCH_CHECK(host_slot_bytes == slot_bytes, "state shape mismatch: device slot_bytes=%lld, host slot_bytes=%lld",
                slot_bytes, host_slot_bytes);

    // Device stride between layers (layer-first: [layer, slot, ...])
    const int64_t device_pitch = device_size * slot_bytes;
    // Host stride between layers (page-first: [slot, layer, ...])
    const int64_t host_pitch = slot_bytes;
    const int64_t width = slot_bytes;
    const int64_t height = num_layers;

    auto device_indices_cpu = device_indices.cpu();
    auto host_indices_cpu = host_indices.cpu();
    const int64_t num_indices = device_indices.numel();

    c10_npu::NPUStream current_stream = c10_npu::getCurrentNPUStream();
    aclrtStream acl_stream = current_stream.stream();

    char *device_base = reinterpret_cast<char *>(device_buf.data_ptr());
    char *host_base = reinterpret_cast<char *>(host_buf.data_ptr());

    aclrtMemcpyKind kind;
    if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
        kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
    } else {
        kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
    }

    for (int64_t i = 0; i < num_indices; i++) {
        auto device_slot = device_indices_cpu[i].item<int64_t>();
        auto host_slot = host_indices_cpu[i].item<int64_t>();

        TORCH_CHECK(device_slot >= 0 && device_slot < device_size, "device slot index out of range: %lld (size=%lld)",
                    device_slot, device_size);
        TORCH_CHECK(host_slot >= 0 && host_slot < host_buf.sizes()[0], "host slot index out of range: %lld (size=%lld)",
                    host_slot, host_buf.sizes()[0]);

        // Device: layer 0, slot device_slot
        char *device_ptr = device_base + device_slot * slot_bytes;
        // Host: slot host_slot, layer 0
        char *host_ptr = host_base + host_slot * num_layers * slot_bytes;

        if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
            aclrtMemcpy2dAsync(host_ptr, host_pitch, device_ptr, device_pitch, width, height, kind, acl_stream);
        } else {
            aclrtMemcpy2dAsync(device_ptr, device_pitch, host_ptr, host_pitch, width, height, kind, acl_stream);
        }
    }
}

}  // namespace npu_kernel
}  // namespace sglang
