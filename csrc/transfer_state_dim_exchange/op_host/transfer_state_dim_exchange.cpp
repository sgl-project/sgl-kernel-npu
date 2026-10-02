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

struct ValidatedStateComponents {
    std::vector<StateComponentLayout> layouts;
    int64_t device_slot_limit;
    int64_t host_slot_limit;
};

int64_t validate_dense_slot_payload(const at::Tensor &device, int64_t component)
{
    std::vector<std::pair<int64_t, int64_t>> payload_dims;
    int64_t slot_elements = 1;
    for (int64_t dim = 2; dim < device.dim(); ++dim) {
        const int64_t size = device.size(dim);
        const int64_t stride = device.stride(dim);
        TORCH_CHECK(stride >= 0, "device state component ", component, " has a negative stride at dimension ", dim);
        slot_elements *= size;
        if (size > 1) {
            payload_dims.emplace_back(stride, size);
        }
    }

    std::sort(payload_dims.begin(), payload_dims.end());
    int64_t expected_stride = 1;
    for (const auto &[stride, size] : payload_dims) {
        TORCH_CHECK(stride == expected_stride, "device state component ", component,
                    " slot payload must be physically dense; got payload stride ", stride, " while expecting ",
                    expected_stride);
        expected_stride *= size;
    }
    return slot_elements;
}

int64_t validate_dense_layer_payload(const at::Tensor &device_layer)
{
    std::vector<std::pair<int64_t, int64_t>> payload_dims;
    int64_t slot_elements = 1;
    for (int64_t dim = 1; dim < device_layer.dim(); ++dim) {
        const int64_t size = device_layer.size(dim);
        const int64_t stride = device_layer.stride(dim);
        TORCH_CHECK(stride >= 0, "device layer state has a negative stride at dimension ", dim);
        slot_elements *= size;
        if (size > 1) {
            payload_dims.emplace_back(stride, size);
        }
    }

    std::sort(payload_dims.begin(), payload_dims.end());
    int64_t expected_stride = 1;
    for (const auto &[stride, size] : payload_dims) {
        TORCH_CHECK(stride == expected_stride, "device layer state slot payload must be physically dense; got ",
                    "payload stride ", stride, " while expecting ", expected_stride);
        expected_stride *= size;
    }
    return slot_elements;
}

void check_acl_copy(aclError result, const char *direction, size_t width, size_t height)
{
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtMemcpy2dAsync failed for state ", direction,
                " transfer: error=", static_cast<int64_t>(result), ", width=", width, ", height=", height);
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
    TORCH_CHECK(host.is_pinned(), "host state component ", component, " must use pinned memory");
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
    TORCH_CHECK(slot_bytes <= device_layer_pitch && slot_bytes <= device_slot_pitch && slot_bytes <= host_slot_pitch &&
                    slot_bytes <= host_layer_pitch,
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

ValidatedStateComponents validate_components(at::TensorList device_states, at::TensorList host_states,
                                             int64_t layer_begin, int64_t layer_count)
{
    std::vector<StateComponentLayout> layouts;
    layouts.reserve(device_states.size());
    for (const auto component : c10::irange(device_states.size())) {
        layouts.emplace_back(
            validate_component(device_states[component], host_states[component], component, layer_begin, layer_count));
    }

    int64_t device_slot_limit = layouts.front().device_slot_num;
    int64_t host_slot_limit = layouts.front().host_slot_num;
    for (const auto &layout : layouts) {
        device_slot_limit = std::min(device_slot_limit, layout.device_slot_num);
        host_slot_limit = std::min(host_slot_limit, layout.host_slot_num);
    }
    return {std::move(layouts), device_slot_limit, host_slot_limit};
}

void validate_indices(const int64_t *device_indices, const int64_t *host_indices, int64_t count,
                      int64_t device_slot_limit, int64_t host_slot_limit)
{
    for (const auto i : c10::irange(count)) {
        TORCH_CHECK(device_indices[i] >= 0, "device index ", device_indices[i], " must be non-negative");
        TORCH_CHECK(host_indices[i] >= 0, "host index ", host_indices[i], " must be non-negative");
        TORCH_CHECK(device_indices[i] < device_slot_limit, "device index ", device_indices[i],
                    " exceeds component slot count ", device_slot_limit);
        TORCH_CHECK(host_indices[i] < host_slot_limit, "host index ", host_indices[i], " exceeds component slot count ",
                    host_slot_limit);
    }
}

std::vector<std::pair<int64_t, int64_t>> build_contiguous_runs(const int64_t *device_indices,
                                                               const int64_t *host_indices, int64_t count)
{
    std::vector<std::pair<int64_t, int64_t>> runs;
    runs.reserve(count);
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

void submit_state_all_layer_d2h(at::TensorList device_states, at::TensorList host_states,
                                const at::Tensor &device_indices, const at::Tensor &host_indices, int64_t flags)
{
    TORCH_CHECK(device_states.size() != 0, "device_states must not be empty");
    TORCH_CHECK(device_states.size() == host_states.size(),
                "device_states and host_states must contain the same number of components");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(),
                "device and host indices must contain the same number of slots");
    TORCH_CHECK((flags & STATE_TRANS_FLAG_2D) == STATE_TRANS_FLAG_2D,
                "state direct transfer currently requires FAST2D (flags=2)");

    const auto device_indices_cpu = device_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const auto host_indices_cpu = host_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const int64_t index_count = device_indices_cpu.numel();
    if (index_count == 0) {
        return;
    }
    TORCH_CHECK(device_states[0].dim() >= 2, "device state must have layout [layers, device_slots, ...]");
    const int64_t layer_begin = 0;
    const int64_t layer_count = device_states[0].size(0);
    TORCH_CHECK(layer_count > 0, "device state layer count must be positive");
    const auto *device_index_data = device_indices_cpu.data_ptr<int64_t>();
    const auto *host_index_data = host_indices_cpu.data_ptr<int64_t>();

    auto components = validate_components(device_states, host_states, layer_begin, layer_count);
    validate_indices(device_index_data, host_index_data, index_count, components.device_slot_limit,
                     components.host_slot_limit);

    const auto stream = c10_npu::getCurrentNPUStream().stream();
    for (const auto &component : components.layouts) {
        auto *device_base = static_cast<char *>(component.device.data_ptr());
        auto *host_base = static_cast<char *>(component.host.data_ptr());
        for (const auto i : c10::irange(index_count)) {
            const auto device_slot = device_index_data[i];
            const auto host_slot = host_index_data[i];
            const void *source = device_base + device_slot * component.device_slot_pitch;
            void *destination = host_base + host_slot * component.host_slot_pitch;
            const auto result = aclrtMemcpy2dAsync(destination, component.host_layer_pitch, source,
                                                   component.device_layer_pitch, component.slot_bytes,
                                                   static_cast<size_t>(layer_count), ACL_MEMCPY_DEVICE_TO_HOST, stream);
            check_acl_copy(result, "D2H", component.slot_bytes, static_cast<size_t>(layer_count));
        }
    }
}

void submit_state_per_layer_h2d(const at::Tensor &src, const at::Tensor &dst, const at::Tensor &src_indices,
                                const at::Tensor &dst_indices, int64_t layer_id, int64_t flags)
{
    TORCH_CHECK(src_indices.numel() == dst_indices.numel(),
                "source and destination indices must contain the same number of slots");
    TORCH_CHECK((flags & STATE_TRANS_FLAG_2D) == STATE_TRANS_FLAG_2D,
                "state direct transfer currently requires FAST2D (flags=2)");

    const auto host_indices_cpu = src_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const auto device_indices_cpu = dst_indices.cpu().to(at::kLong).contiguous().reshape({-1});
    const int64_t index_count = host_indices_cpu.numel();
    if (index_count == 0) {
        return;
    }

    TORCH_CHECK(src.defined() && dst.defined(), "source and destination state tensors must be defined");
    TORCH_CHECK(src.numel() != 0, "host state source must not be empty");
    TORCH_CHECK(dst.numel() != 0, "device layer state destination must not be empty");
    TORCH_CHECK(src.device().is_cpu(), "state source must be on CPU, got ", src.device());
    TORCH_CHECK(src.is_pinned(), "state source must use pinned memory");
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "state destination must be on NPU, got ",
                dst.device());
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "state source/destination dtypes differ: ", src.scalar_type(),
                " vs ", dst.scalar_type());
    TORCH_CHECK(dst.dim() >= 2, "device layer state must have layout [device_slots, ...]");
    TORCH_CHECK(src.dim() == dst.dim() + 2,
                "host state must have layout [host_slots, layers, 1, ...] for a Device layer view");
    TORCH_CHECK(src.is_contiguous(), "host state source must be contiguous");
    TORCH_CHECK(layer_id >= 0 && layer_id < src.size(1), "layer_id ", layer_id, " is outside Host state layer count ",
                src.size(1));
    TORCH_CHECK(src.size(2) == 1, "host state page dimension must be 1");
    for (int64_t dim = 1; dim < dst.dim(); ++dim) {
        TORCH_CHECK(dst.size(dim) == src.size(dim + 2), "state trailing shape mismatch at Device layer dimension ",
                    dim);
    }

    const int64_t slot_elements = validate_dense_layer_payload(dst);
    TORCH_CHECK(dst.stride(0) == slot_elements, "device layer state slot payload must be contiguous");
    TORCH_CHECK(src.stride(1) == slot_elements, "host state layer payload must be contiguous");

    const auto *host_index_data = host_indices_cpu.data_ptr<int64_t>();
    const auto *device_index_data = device_indices_cpu.data_ptr<int64_t>();
    validate_indices(device_index_data, host_index_data, index_count, dst.size(0), src.size(0));

    const size_t slot_bytes = static_cast<size_t>(slot_elements) * dst.element_size();
    const size_t device_slot_pitch = static_cast<size_t>(dst.stride(0)) * dst.element_size();
    const size_t host_slot_pitch = static_cast<size_t>(src.stride(0)) * src.element_size();
    const size_t host_layer_pitch = static_cast<size_t>(src.stride(1)) * src.element_size();
    TORCH_CHECK(slot_bytes <= device_slot_pitch && slot_bytes <= host_slot_pitch && slot_bytes <= host_layer_pitch,
                "invalid state pitch for per-layer H2D transfer");

    auto *device_base = static_cast<char *>(dst.data_ptr());
    auto *host_base = static_cast<char *>(src.data_ptr());
    const auto runs = build_contiguous_runs(device_index_data, host_index_data, index_count);
    const auto acl_stream = c10_npu::getCurrentNPUStream().stream();
    for (const auto &[run_begin, run_length] : runs) {
        const auto device_slot = device_index_data[run_begin];
        const auto host_slot = host_index_data[run_begin];
        void *destination = device_base + device_slot * device_slot_pitch;
        const void *source = host_base + host_slot * host_slot_pitch + static_cast<size_t>(layer_id) * host_layer_pitch;
        const auto result = aclrtMemcpy2dAsync(destination, device_slot_pitch, source, host_slot_pitch, slot_bytes,
                                               static_cast<size_t>(run_length), ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
        check_acl_copy(result, "H2D", slot_bytes, static_cast<size_t>(run_length));
    }
}

}  // namespace

HOST_API void transfer_state_per_layer_direct_pf_lf(const at::Tensor &src, const at::Tensor &dst,
                                                    const at::Tensor &src_indices, const at::Tensor &dst_indices,
                                                    int64_t layer_id, int64_t flags)
{
    submit_state_per_layer_h2d(src, dst, src_indices, dst_indices, layer_id, flags);
}

HOST_API void transfer_state_all_layer_direct_lf_pf(at::TensorList device_states, at::TensorList host_states,
                                                    const at::Tensor &device_indices, const at::Tensor &host_indices,
                                                    int64_t flags)
{
    submit_state_all_layer_d2h(device_states, host_states, device_indices, host_indices, flags);
}

}  // namespace npu_kernel
}  // namespace sglang
