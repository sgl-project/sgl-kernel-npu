// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

constexpr int64_t STATE_TRANS_FLAG_2D = 1 << 1;

enum class StateTransferDirection : int64_t {
    H2D = 1,
    D2H = 2,
};

namespace {

struct StateComponentLayout {
    at::Tensor device;
    at::Tensor host;
    int64_t device_slot_num;
    int64_t host_slot_num;
    size_t slot_bytes;
    size_t device_layer_pitch;
    size_t device_slot_pitch;
    size_t host_slot_pitch;
    size_t host_layer_pitch;
};

int64_t validate_dense_slot_payload(const at::Tensor &device, int64_t component)
{
    // NPU NEXTN stores the temporal state as a transpose of the two innermost
    // dimensions.  The resulting Tensor is not logically contiguous, but every
    // [layer, slot] payload is still one dense physical byte range.  HiCache
    // treats the Host copy as opaque state bytes, so that layout is safe for a
    // byte-exact D2H/H2D round trip.
    std::vector<std::pair<int64_t, int64_t>> payload_dims;
    int64_t slot_elements = 1;
    for (int64_t dim = 2; dim < device.dim(); ++dim) {
        const int64_t size = device.size(dim);
        const int64_t stride = device.stride(dim);
        TORCH_CHECK(size >= 0, "device state component ", component, " has a negative size at dimension ", dim);
        TORCH_CHECK(stride >= 0, "device state component ", component,
                    " has a negative stride at dimension ", dim);
        slot_elements *= size;
        if (size > 1) {
            payload_dims.emplace_back(stride, size);
        }
    }

    std::sort(payload_dims.begin(), payload_dims.end());
    int64_t expected_stride = 1;
    for (const auto &[stride, size] : payload_dims) {
        TORCH_CHECK(stride == expected_stride, "device state component ", component,
                    " slot payload must be physically dense; got payload stride ", stride,
                    " while expecting ", expected_stride);
        expected_stride *= size;
    }
    TORCH_CHECK(expected_stride == slot_elements, "device state component ", component,
                " slot payload span does not match its element count");
    return slot_elements;
}

void check_acl_copy(aclError result, const char *direction, size_t width, size_t height)
{
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtMemcpy2dAsync failed for state ", direction, " transfer: error=",
                static_cast<int64_t>(result), ", width=", width, ", height=", height);
}

StateComponentLayout validate_component(const at::Tensor &device, const at::Tensor &host, int64_t component,
                                        int64_t layer_begin, int64_t layer_count)
{
    TORCH_CHECK(device.defined() && host.defined(), "state component ", component, " must be defined");
    TORCH_CHECK(device.numel() != 0, "device state component ", component, " must not be empty");
    TORCH_CHECK(host.numel() != 0, "host state component ", component, " must not be empty");
    TORCH_CHECK(device.device().type() == c10::DeviceType::PrivateUse1, "device state component ", component,
                " must be on NPU, got ", device.device());
    TORCH_CHECK(host.device().is_cpu(), "host state component ", component, " must be on CPU, got ", host.device());
    TORCH_CHECK(device.scalar_type() == host.scalar_type(), "state component ", component,
                " has different device/host dtypes: ", device.scalar_type(), " vs ", host.scalar_type());
    TORCH_CHECK(device.dim() >= 3, "device state component ", component,
                " must have layout [layers, device_slots, ...], got ", device.dim(), " dimensions");
    TORCH_CHECK(host.dim() == device.dim() + 1, "host state component ", component,
                " must have layout [host_slots, layers, 1, ...]");
    TORCH_CHECK(host.is_contiguous(), "host state component ", component, " must be contiguous");
    TORCH_CHECK(device.size(0) == host.size(1), "state component ", component,
                " has different device/host layer counts");
    TORCH_CHECK(host.size(2) == 1, "host state component ", component, " page dimension must be 1");
    for (int64_t dim = 2; dim < device.dim(); ++dim) {
        TORCH_CHECK(device.size(dim) == host.size(dim + 1), "state component ", component,
                    " trailing shape mismatch at device dimension ", dim);
    }
    TORCH_CHECK(layer_begin >= 0, "layer_begin must be non-negative");
    TORCH_CHECK(layer_count > 0, "layer_count must be positive");
    TORCH_CHECK(layer_begin + layer_count <= device.size(0), "requested layer range [", layer_begin, ", ",
                layer_begin + layer_count, ") exceeds state component ", component, " layer count ", device.size(0));

    const int64_t slot_elements = validate_dense_slot_payload(device, component);
    TORCH_CHECK(device.stride(1) == slot_elements, "device state component ", component,
                " slot payload must be contiguous");
    TORCH_CHECK(device.stride(0) >= device.size(1) * device.stride(1), "device state component ", component,
                " layer pitch overlaps adjacent slots");
    TORCH_CHECK(host.stride(1) == slot_elements, "host state component ", component,
                " layer payload must be contiguous");

    const size_t slot_bytes = static_cast<size_t>(slot_elements) * device.element_size();
    const size_t device_layer_pitch = static_cast<size_t>(device.stride(0)) * device.element_size();
    const size_t device_slot_pitch = static_cast<size_t>(device.stride(1)) * device.element_size();
    const size_t host_slot_pitch = static_cast<size_t>(host.stride(0)) * host.element_size();
    const size_t host_layer_pitch = static_cast<size_t>(host.stride(1)) * host.element_size();
    TORCH_CHECK(slot_bytes <= device_layer_pitch && slot_bytes <= device_slot_pitch &&
                    slot_bytes <= host_slot_pitch && slot_bytes <= host_layer_pitch,
                "invalid state component ", component, " pitch for aclrtMemcpy2dAsync");

    return {
        device,
        host,
        device.size(1),
        host.size(0),
        slot_bytes,
        device_layer_pitch,
        device_slot_pitch,
        host_slot_pitch,
        host_layer_pitch,
    };
}

std::vector<std::pair<int64_t, int64_t>> build_contiguous_runs(const int64_t *device_indices,
                                                                const int64_t *host_indices, int64_t count)
{
    std::vector<std::pair<int64_t, int64_t>> runs;
    int64_t begin = 0;
    while (begin < count) {
        int64_t end = begin + 1;
        while (end < count && device_indices[end] == device_indices[end - 1] + 1 &&
               host_indices[end] == host_indices[end - 1] + 1) {
            ++end;
        }
        runs.emplace_back(begin, end - begin);
        begin = end;
    }
    return runs;
}

}  // namespace

// Submit state-sidecar copies to the caller's current NPU stream.
//
// Device component layout: [layers, device_slots, *state_shape]
// Host component layout:   [host_slots, layers, 1, *state_shape]
HOST_API void transfer_state_dim_exchange(at::TensorList device_states, at::TensorList host_states,
                                          const at::Tensor &device_indices, const at::Tensor &host_indices,
                                          int64_t direction, int64_t layer_begin, int64_t layer_count, int64_t flags)
{
    TORCH_CHECK(device_states.size() != 0, "device_states must not be empty");
    TORCH_CHECK(device_states.size() == host_states.size(),
                "device_states and host_states must contain the same number of components");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(),
                "device and host indices must contain the same number of slots");
    TORCH_CHECK(direction == static_cast<int64_t>(StateTransferDirection::H2D) ||
                    direction == static_cast<int64_t>(StateTransferDirection::D2H),
                "direction must be 1 (H2D) or 2 (D2H)");
    TORCH_CHECK((flags & STATE_TRANS_FLAG_2D) == STATE_TRANS_FLAG_2D,
                "transfer_state_dim_exchange currently requires FAST2D (flags=2)");

    const auto device_indices_cpu = device_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const auto host_indices_cpu = host_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const int64_t index_count = device_indices_cpu.numel();
    if (index_count == 0) {
        return;
    }
    const auto *device_index_data = device_indices_cpu.data_ptr<int64_t>();
    const auto *host_index_data = host_indices_cpu.data_ptr<int64_t>();

    std::vector<StateComponentLayout> components;
    components.reserve(device_states.size());
    for (const auto component : c10::irange(device_states.size())) {
        components.emplace_back(validate_component(device_states[component], host_states[component], component,
                                                   layer_begin, layer_count));
    }

    for (const auto i : c10::irange(index_count)) {
        TORCH_CHECK(device_index_data[i] >= 0, "device index ", device_index_data[i], " must be non-negative");
        TORCH_CHECK(host_index_data[i] >= 0, "host index ", host_index_data[i], " must be non-negative");
        for (const auto &component : components) {
            TORCH_CHECK(device_index_data[i] < component.device_slot_num, "device index ", device_index_data[i],
                        " exceeds component slot count ", component.device_slot_num);
            TORCH_CHECK(host_index_data[i] < component.host_slot_num, "host index ", host_index_data[i],
                        " exceeds component slot count ", component.host_slot_num);
        }
    }

    const auto acl_stream = c10_npu::getCurrentNPUStream().stream();
    if (direction == static_cast<int64_t>(StateTransferDirection::H2D)) {
        const auto runs = build_contiguous_runs(device_index_data, host_index_data, index_count);
        for (const auto &component : components) {
            auto *device_base = static_cast<char *>(component.device.data_ptr());
            auto *host_base = static_cast<char *>(component.host.data_ptr());
            for (const auto layer : c10::irange(layer_begin, layer_begin + layer_count)) {
                for (const auto &[run_begin, run_length] : runs) {
                    const auto device_slot = device_index_data[run_begin];
                    const auto host_slot = host_index_data[run_begin];
                    void *destination =
                        device_base + layer * component.device_layer_pitch + device_slot * component.device_slot_pitch;
                    const void *source =
                        host_base + host_slot * component.host_slot_pitch + layer * component.host_layer_pitch;
                    const auto result = aclrtMemcpy2dAsync(
                        destination, component.device_slot_pitch, source, component.host_slot_pitch,
                        component.slot_bytes, static_cast<size_t>(run_length), ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
                    check_acl_copy(result, "H2D", component.slot_bytes, static_cast<size_t>(run_length));
                }
            }
        }
    } else {
        for (const auto &component : components) {
            auto *device_base = static_cast<char *>(component.device.data_ptr());
            auto *host_base = static_cast<char *>(component.host.data_ptr());
            for (const auto i : c10::irange(index_count)) {
                const auto device_slot = device_index_data[i];
                const auto host_slot = host_index_data[i];
                const void *source = device_base + layer_begin * component.device_layer_pitch +
                                     device_slot * component.device_slot_pitch;
                void *destination =
                    host_base + host_slot * component.host_slot_pitch + layer_begin * component.host_layer_pitch;
                const auto result = aclrtMemcpy2dAsync(
                    destination, component.host_layer_pitch, source, component.device_layer_pitch,
                    component.slot_bytes, static_cast<size_t>(layer_count), ACL_MEMCPY_DEVICE_TO_HOST, acl_stream);
                check_acl_copy(result, "D2H", component.slot_bytes, static_cast<size_t>(layer_count));
            }
        }
    }
}

}  // namespace npu_kernel
}  // namespace sglang
