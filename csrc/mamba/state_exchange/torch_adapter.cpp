// BSD 3-Clause License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

#include "aclnn_runtime.h"
#include "aclnn_sgl_state_exchange.h"
#include "op_kernel/state_exchange_tiling.h"

#include <array>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <torch/library.h>
#include <vector>

namespace {

using optiling::STATE_EXCHANGE_ATTR_SIZE;
using optiling::STATE_EXCHANGE_MAX_ENTRIES;
using optiling::STATE_EXCHANGE_META_STRIDE;

constexpr int64_t kDirectionH2D = 1;
constexpr int64_t kDirectionD2H = 2;

// Zero leaves the choice to the tiling function, which reads the AIV core count.
int64_t ResolveBlockDim()
{
    if (const char *env = std::getenv("SGLANG_STATE_EXCHANGE_BLOCK_DIM")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed > 0) {
            return static_cast<int64_t>(parsed);
        }
    }
    return 0;
}

// First participating AIV core index. The default (-1) selects the native
// acc_offload occupancy: only the last 2 AIV cores copy, leaving the rest of
// the cores free for concurrent compute on the main stream. Explicit values
// (e.g. 0 for all cores) override for perf sweeps.
int64_t ResolveStartDim()
{
    if (const char *env = std::getenv("SGLANG_STATE_EXCHANGE_START_DIM")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed >= 0) {
            return static_cast<int64_t>(parsed);
        }
    }
    return -1;
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

// The kernel walks (slot, layer) payload rows addressed as
// base + layer*layer_pitch + slot*slot_pitch and copies `width` contiguous
// bytes per row, so only the trailing payload dims must be packed; slot and
// layer dims may be strided views (e.g. layer-narrowed draft host pools).
bool TrailingDimsPacked(const at::Tensor &t, int64_t from)
{
    int64_t expect = 1;
    for (int64_t i = t.dim() - 1; i >= from; --i) {
        if (t.stride(i) != expect) {
            return false;
        }
        expect *= t.size(i);
    }
    return true;
}

// Mirrors the Mamba host pool contract: device payload [layers, slots,
// *payload], host payload [slots, layers, 1, *payload] (both may be
// layer/slot-narrowed views; base pointers and strides carry the offsets).
// Pool slot counts deliberately differ in production (the host pool is sized
// by --hicache-ratio) and are NOT cross-checked against the indices; the
// kernel only dereferences slots listed in the indices, whose bounds the
// Python operator layer validates.
void CheckComponentPair(const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t which)
{
    TORCH_CHECK(device_buf.scalar_type() == host_buf.scalar_type(), "component ", which,
                ": device and host dtypes must match, got ", device_buf.scalar_type(), " vs ", host_buf.scalar_type());
    TORCH_CHECK(device_buf.dim() >= 3 && host_buf.dim() == device_buf.dim() + 1, "component ", which,
                ": buffers must be device (layers, slots, *payload) and host (slots, layers, 1, *payload), got "
                "device ",
                device_buf.dim(), "-D / host ", host_buf.dim(), "-D");
    TORCH_CHECK(host_buf.size(1) == device_buf.size(0), "component ", which,
                ": host layer count (dim 1) must match device layer count (dim 0), got ", host_buf.size(1), " vs ",
                device_buf.size(0));
    TORCH_CHECK(host_buf.size(2) == 1, "component ", which, ": host dim 2 must be the mamba singleton dim");
    TORCH_CHECK(device_buf.size(1) > 0, "component ", which, ": device slot count (dim 1) must be positive");
    TORCH_CHECK(device_buf.size(0) > 0, "component ", which, ": device layer count (dim 0) must be positive");

    const int64_t esz = device_buf.element_size();
    int64_t width = esz;
    for (int64_t i = 2; i < device_buf.dim(); ++i) {
        width *= device_buf.size(i);
    }
    int64_t hostWidth = host_buf.element_size();
    for (int64_t i = 3; i < host_buf.dim(); ++i) {
        hostWidth *= host_buf.size(i);
    }
    TORCH_CHECK(width > 0 && width == hostWidth, "component ", which,
                ": device and host packed payload widths must be positive and match, got ", width, " vs ", hostWidth);
    TORCH_CHECK(TrailingDimsPacked(device_buf, 2) && TrailingDimsPacked(host_buf, 3), "component ", which,
                ": trailing payload dims must be packed (contiguous within a (slot, layer) row)");
    TORCH_CHECK(device_buf.stride(0) * esz >= width && device_buf.stride(1) * esz >= width, "component ", which,
                ": device layer/slot pitches must not overlap adjacent rows");
    TORCH_CHECK(host_buf.stride(0) * esz >= width && host_buf.stride(1) * esz >= width, "component ", which,
                ": host slot/layer pitches must not overlap adjacent rows");
}

// Component views carry their layer offsets in their base addresses; the
// whole view (all its layers) is transferred.
void FillEntry(int64_t *fields, const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t host_dva)
{
    const int64_t esz = device_buf.element_size();
    fields[0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr()));
    fields[1] = host_dva;
    fields[2] = device_buf.stride(1) * esz;  // device_slot_pitch
    fields[3] = device_buf.stride(0) * esz;  // device_layer_pitch
    fields[4] = host_buf.stride(0) * esz;    // host_slot_pitch
    fields[5] = host_buf.stride(1) * esz;    // host_layer_pitch
    // The kernel copies this many contiguous bytes per (slot, layer) row:
    // the packed extent of the trailing payload dims.
    int64_t width = esz;
    for (int64_t i = 2; i < device_buf.dim(); ++i) {
        width *= device_buf.size(i);
    }
    fields[6] = width;
    fields[7] = device_buf.size(0);  // layer_count
}

void state_exchange_submit(const at::Tensor &device_indices, const at::Tensor &host_indices,
                           at::TensorList device_components, at::TensorList host_components, at::IntArrayRef host_dvas,
                           int64_t direction)
{
    TORCH_CHECK(direction == kDirectionH2D || direction == kDirectionD2H, "direction must be H2D (1) or D2H (2), got ",
                direction);
    TORCH_CHECK(!device_components.empty() && device_components.size() == host_components.size(),
                "device_components and host_components must be non-empty lists of the same length");
    TORCH_CHECK(device_components.size() <= STATE_EXCHANGE_MAX_ENTRIES, "at most ", STATE_EXCHANGE_MAX_ENTRIES,
                " entries are supported, got ", device_components.size());
    TORCH_CHECK(host_dvas.size() == device_components.size(), "host_dvas must match the component count, got ",
                host_dvas.size(), " vs ", device_components.size());

    CheckNpuTensor(device_indices, "device_indices");
    CheckNpuTensor(host_indices, "host_indices");
    TORCH_CHECK(device_indices.scalar_type() == at::kLong && host_indices.scalar_type() == at::kLong,
                "device and host indices must have dtype int64");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device/host indices must have the same length");
    TORCH_CHECK(device_indices.device() == host_indices.device(), "indices must share the same NPU");
    TORCH_CHECK(device_indices.dim() == 1 && host_indices.dim() == 1 && device_indices.is_contiguous() &&
                    host_indices.is_contiguous(),
                "indices must be contiguous 1-D tensors");
    if (device_indices.numel() == 0) {
        // Nothing to transfer.
        return;
    }

    const c10::DeviceGuard deviceGuard(device_indices.device());
    auto fields = std::make_shared<std::array<int64_t, STATE_EXCHANGE_ATTR_SIZE>>();
    (*fields)[optiling::NUM_ENTRIES] = static_cast<int64_t>(device_components.size());
    (*fields)[optiling::DIRECTION] = direction;
    (*fields)[optiling::START_DIM] = ResolveStartDim();
    (*fields)[optiling::BLOCK_DIM] = ResolveBlockDim();
    for (int64_t c = 0; c < static_cast<int64_t>(device_components.size()); ++c) {
        const at::Tensor &device_buf = device_components[c];
        const at::Tensor &host_buf = host_components[c];
        TORCH_CHECK(device_buf.device() == device_indices.device(), "Device components must share the index NPU");
        TORCH_CHECK(host_buf.device().type() == at::kCPU, "component ", c,
                    ": host components must be CPU (device-mapped) tensors");
        CheckComponentPair(device_buf, host_buf, c);
        TORCH_CHECK(host_dvas[static_cast<size_t>(c)] != 0, "component ", c, ": host DVA must be non-zero");
        FillEntry(fields->data() + optiling::ATTR_HEADER_SIZE + c * STATE_EXCHANGE_META_STRIDE, device_buf, host_buf,
                  host_dvas[c]);
    }

    auto deviceDesc = sgl_cann::tensor_descriptor(device_indices, ACL_INT64);
    auto hostDesc = sgl_cann::tensor_descriptor(host_indices, ACL_INT64);
    auto transfer = std::shared_ptr<aclIntArray>(aclCreateIntArray(fields->data(), fields->size()), aclDestroyIntArray);
    TORCH_CHECK(transfer, "aclCreateIntArray failed: ", aclGetRecentErrMsg());
    static auto query = reinterpret_cast<decltype(&aclnnSglStateExchangeGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglStateExchangeGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglStateExchange)>(sgl_cann::op_api_symbol("aclnnSglStateExchange"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(deviceDesc.get(), hostDesc.get(), transfer.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "State exchange workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    at::Tensor workspace;
    auto stream = c10_npu::getCurrentNPUStream();
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, device_indices.options().dtype(at::kByte));
    }
    auto aclStream = stream.stream(false);
    // Keep the Tensor and ACLNN descriptor owners alive through queued launch.
    auto launch = [device_indices, host_indices, components = device_components.vec(), fields, transfer, deviceDesc,
                   hostDesc, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "State exchange ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglStateExchange", launch);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "state_exchange("
        "Tensor device_indices, Tensor host_indices, "
        "Tensor[] device_components, Tensor[] host_components, "
        "int[] host_dvas, int direction) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("state_exchange", &state_exchange_submit);
}
