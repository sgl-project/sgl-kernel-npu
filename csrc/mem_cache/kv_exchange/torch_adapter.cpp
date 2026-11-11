// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aclnn_runtime.h"
#include "aclnn_sgl_kv_exchange.h"
#include "op_kernel/kv_exchange_tiling.h"

#include <array>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <torch/library.h>
#include <vector>

namespace {

using optiling::KV_EXCHANGE_ATTR_SIZE;
using optiling::KV_EXCHANGE_MAX_COMPONENTS;
using optiling::KV_EXCHANGE_META_STRIDE;

constexpr int64_t kDirectionH2D = 1;
constexpr int64_t kDirectionD2H = 2;
constexpr int64_t kHostLayoutPageFirst = 0;
constexpr int64_t kHostLayoutLayerFirst = 1;

// Zero leaves the choice to the tiling function, which reads the AIV core count.
int64_t ResolveBlockDim()
{
    if (const char *env = std::getenv("SGLANG_KV_EXCHANGE_BLOCK_DIM")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed > 0) {
            return static_cast<int64_t>(parsed);
        }
    }
    return 0;
}

// First participating AIV core index. The default (-1) selects the native
// acc_offload kv_exchange occupancy: only the last 2 AIV cores copy, leaving
// the rest of the cores free for concurrent compute on the main stream, which
// is what lets the per-layer H2D overlap attention. Explicit values (e.g. 0
// for all cores) override for perf sweeps.
int64_t ResolveStartDim()
{
    if (const char *env = std::getenv("SGLANG_KV_EXCHANGE_START_DIM")) {
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

// The kernel walks (page, layer) blocks addressed as
// base + layer*layer_pitch + page*page_pitch and copies `width` contiguous
// bytes from each block, so only the trailing (page_size, 1, width) dims must
// be packed; page/layer dims may be strided views (e.g. transposed host
// layer-first pools).
bool TrailingDimsPacked(const at::Tensor &t)
{
    int64_t expect = 1;
    for (int64_t i = t.dim() - 1; i >= 2; --i) {
        if (t.stride(i) != expect) {
            return false;
        }
        expect *= t.size(i);
    }
    return true;
}

// Mirrors the Python _check_component contract: matching dtypes, page shapes
// and packed trailing dims. Pool page counts deliberately differ in
// production (the host pool is sized by --hicache-ratio) and are NOT checked;
// the kernel only dereferences pages listed in the token indices.
void CheckComponentPair(const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t page_size, int64_t which)
{
    TORCH_CHECK(device_buf.scalar_type() == host_buf.scalar_type(), "component ", which,
                ": device and host dtypes must match, got ", device_buf.scalar_type(), " vs ", host_buf.scalar_type());
    // 4-D (layers, pages, page_size, 1) is accepted alongside the canonical
    // 5-D form: the fp32 Indexer scale caches carry no trailing width dim, so
    // the caller can hand them over without padding a singleton (width
    // resolves to 1 below, matching the padded form).
    TORCH_CHECK((device_buf.dim() == 5 || device_buf.dim() == 4) && (host_buf.dim() == 5 || host_buf.dim() == 4),
                "component ", which,
                ": buffers must be 4-D or 5-D -- device (layers, pages, page_size, [1,] width), host "
                "(pages, layers, page_size, [1,] width), got device ",
                device_buf.dim(), "-D / host ", host_buf.dim(), "-D");
    TORCH_CHECK(device_buf.size(2) == page_size, "component ", which, ": device dim(2) must equal page_size");
    TORCH_CHECK(host_buf.size(2) == page_size, "component ", which, ": host dim(2) must equal page_size");
    TORCH_CHECK(device_buf.size(3) == host_buf.size(3), "component ", which,
                ": device and host singleton dims (dim 3) must match");
    TORCH_CHECK((device_buf.dim() == host_buf.dim()) &&
                    (device_buf.dim() == 4 ||
                     device_buf.size(4) * device_buf.element_size() == host_buf.size(4) * host_buf.element_size()),
                "component ", which, ": device and host packed row widths must match");
    TORCH_CHECK(TrailingDimsPacked(device_buf) && TrailingDimsPacked(host_buf), "component ", which,
                ": trailing (page_size, 1, width) dims must be packed (contiguous within a "
                "(page, layer) block)");
}

// Component views carry their layer offsets in their base addresses.
void FillComponent(int64_t *fields, const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t host_dva)
{
    const int64_t esz = device_buf.element_size();
    fields[0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr()));
    fields[1] = host_dva;
    fields[2] = device_buf.stride(0) * esz;  // device_layer_pitch
    fields[3] = device_buf.stride(1) * esz;  // device_page_pitch
    fields[4] = host_buf.stride(0) * esz;    // host_page_pitch
    fields[5] = host_buf.stride(1) * esz;    // host_layer_pitch
    // The kernel copies this many contiguous bytes per (page, layer) block:
    // the packed extent of the trailing dims (prod(shape[2:]) * itemsize).
    int64_t width = esz;
    for (int64_t i = 2; i < host_buf.dim(); ++i) {
        width *= host_buf.size(i);
    }
    fields[6] = width;
    fields[7] = 0;
    fields[8] = device_buf.size(0);
}

void kv_exchange_submit(const at::Tensor &device_indices, const at::Tensor &host_indices,
                        at::TensorList device_components, at::TensorList host_components, at::IntArrayRef host_dvas,
                        int64_t page_size, int64_t direction, int64_t host_layout_mode)
{
    TORCH_CHECK(direction == kDirectionH2D || direction == kDirectionD2H, "direction must be H2D (1) or D2H (2), got ",
                direction);
    TORCH_CHECK(host_layout_mode == kHostLayoutPageFirst || host_layout_mode == kHostLayoutLayerFirst,
                "host_layout_mode must be page_first (0) or layer_first (1), got ", host_layout_mode);
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);
    TORCH_CHECK(!device_components.empty() && device_components.size() == host_components.size(),
                "device_components and host_components must be non-empty lists of the same length");
    TORCH_CHECK(device_components.size() <= KV_EXCHANGE_MAX_COMPONENTS, "at most ", KV_EXCHANGE_MAX_COMPONENTS,
                " components are supported, got ", device_components.size());
    TORCH_CHECK(host_dvas.size() == device_components.size(), "host_dvas must match the component count, got ",
                host_dvas.size(), " vs ", device_components.size());

    CheckNpuTensor(device_indices, "device_indices");
    CheckNpuTensor(host_indices, "host_indices");
    TORCH_CHECK(device_indices.scalar_type() == at::kLong && host_indices.scalar_type() == at::kLong,
                "device and host indices must have dtype int64");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device/host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");

    TORCH_CHECK(device_indices.device() == host_indices.device(), "indices must share the same NPU");
    TORCH_CHECK(device_indices.dim() == 1 && host_indices.dim() == 1 && device_indices.is_contiguous() &&
                    host_indices.is_contiguous(),
                "indices must be contiguous 1-D tensors");

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        // Nothing to transfer.
        return;
    }

    const c10::DeviceGuard deviceGuard(device_indices.device());
    auto fields = std::make_shared<std::array<int64_t, KV_EXCHANGE_ATTR_SIZE>>();
    (*fields)[optiling::NUM_COMPONENTS] = static_cast<int64_t>(device_components.size());
    (*fields)[optiling::PAGE_SIZE] = page_size;
    (*fields)[optiling::DIRECTION] = direction;
    (*fields)[optiling::HOST_LAYOUT] = host_layout_mode;
    (*fields)[optiling::START_DIM] = ResolveStartDim();
    (*fields)[optiling::BLOCK_DIM] = ResolveBlockDim();
    for (int64_t c = 0; c < static_cast<int64_t>(device_components.size()); ++c) {
        const at::Tensor &device_buf = device_components[c];
        const at::Tensor &host_buf = host_components[c];
        TORCH_CHECK(device_buf.device() == device_indices.device(), "Device components must share the index NPU");
        TORCH_CHECK(host_buf.device().type() == at::kCPU, "component ", c,
                    ": host components must be CPU (MemFabric-mapped) tensors");
        CheckComponentPair(device_buf, host_buf, page_size, c);
        TORCH_CHECK(host_dvas[static_cast<size_t>(c)] != 0, "component ", c, ": MF host DVA must be non-zero");
        // Layer offsets are already folded into the component views; the
        // whole view (all its layers) is transferred.
        FillComponent(fields->data() + optiling::ATTR_HEADER_SIZE + c * KV_EXCHANGE_META_STRIDE, device_buf, host_buf,
                      host_dvas[c]);
    }

    auto deviceDesc = sgl_cann::tensor_descriptor(device_indices, ACL_INT64);
    auto hostDesc = sgl_cann::tensor_descriptor(host_indices, ACL_INT64);
    auto transfer = std::shared_ptr<aclIntArray>(aclCreateIntArray(fields->data(), fields->size()), aclDestroyIntArray);
    TORCH_CHECK(transfer, "aclCreateIntArray failed: ", aclGetRecentErrMsg());
    static auto query = reinterpret_cast<decltype(&aclnnSglKvExchangeGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglKvExchangeGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglKvExchange)>(sgl_cann::op_api_symbol("aclnnSglKvExchange"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(deviceDesc.get(), hostDesc.get(), transfer.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "KV exchange workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    at::Tensor workspace;
    auto stream = c10_npu::getCurrentNPUStream();
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, device_indices.options().dtype(at::kByte));
        workspace.record_stream(stream);
    }
    device_indices.record_stream(stream);
    host_indices.record_stream(stream);
    for (const auto &tensor : device_components) {
        tensor.record_stream(stream);
    }
    auto aclStream = stream.stream(false);
    // Keep the Tensor and ACLNN descriptor owners alive through queued launch.
    auto launch = [device_indices, host_indices, components = device_components.vec(), fields, transfer, deviceDesc,
                   hostDesc, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "KV exchange ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglKvExchange", launch);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "kv_exchange("
        "Tensor device_indices, Tensor host_indices, "
        "Tensor[] device_components, Tensor[] host_components, "
        "int[] host_dvas, int page_size, int direction, int host_layout) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_exchange", &kv_exchange_submit);
}
