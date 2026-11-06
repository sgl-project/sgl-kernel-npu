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

#include "kv_exchange.h"

#include "torch_helper.h"

#include "aclrtlaunch_kv_exchange.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t kMaxComponents = 4;

constexpr int64_t kDirectionH2D = 1;
constexpr int64_t kDirectionD2H = 2;
constexpr int64_t kHostLayoutPageFirst = 0;
constexpr int64_t kHostLayoutLayerFirst = 1;

// AIV core count differs per SoC family (arch35/Ascend950: 64, arch22: 48);
// SGLANG_KV_EXCHANGE_BLOCK_DIM overrides for perf sweeps.
int64_t ResolveBlockDim()
{
    if (const char *env = std::getenv("SGLANG_KV_EXCHANGE_BLOCK_DIM")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed > 0) {
            return static_cast<int64_t>(parsed);
        }
    }
#ifdef SGL_KERNEL_ARCH_35
    return 64;
#else
    return 48;
#endif
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

// Transfer-description layout (see op_kernel/kv_exchange_kernel.cpp): the
// production kv_exchange kernel takes the description as flat launch
// parameters -- 8 header scalars plus 4 x 9 component scalars in the same
// field order as the legacy device-resident meta slots.
//
// The component views arrive with their layer offsets already folded into
// their base addresses (device view slicing / host DVA of the sliced view),
// so the layer range starts at 0 and the device kernel's addressing format
// stays untouched. Components with layer_lo >= layer_hi (or width <= 0)
// transfer nothing.
struct ComponentParams {
    int64_t fields[9];
};

#define KV_COMPONENT_ARGS(c)                                                                   \
    comps[c].fields[0], comps[c].fields[1], comps[c].fields[2], comps[c].fields[3],            \
        comps[c].fields[4], comps[c].fields[5], comps[c].fields[6], comps[c].fields[7],         \
        comps[c].fields[8]

void FillComponent(ComponentParams &cp, const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t host_dva,
                   int64_t layer_lo, int64_t layer_hi)
{
    const int64_t esz = device_buf.element_size();
    cp.fields[0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr()));
    cp.fields[1] = host_dva;
    cp.fields[2] = device_buf.stride(0) * esz;  // device_layer_pitch
    cp.fields[3] = device_buf.stride(1) * esz;  // device_page_pitch
    cp.fields[4] = host_buf.stride(0) * esz;    // host_page_pitch
    cp.fields[5] = host_buf.stride(1) * esz;    // host_layer_pitch
    // The kernel copies this many contiguous bytes per (page, layer) block:
    // the packed extent of the trailing dims (prod(shape[2:]) * itemsize).
    int64_t width = esz;
    for (int64_t i = 2; i < host_buf.dim(); ++i) {
        width *= host_buf.size(i);
    }
    cp.fields[6] = width;
    cp.fields[7] = layer_lo;
    cp.fields[8] = layer_hi;
}

// Index uploads are the caller's (python) job; only the async-use lifetime
// is recorded here, on the stream the kernel launches on.
void RecordStreamLifetime(at::TensorList device_components, const at::Tensor &device_indices,
                          const at::Tensor &host_indices)
{
    const auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &t : device_components) {
        t.record_stream(npuStream);
    }
    device_indices.record_stream(npuStream);
    host_indices.record_stream(npuStream);
}

}  // namespace

/*
 * One AIV kv_exchange submission from the split-KV component list.
 *
 * The Python layer (implementation/memfabric/mem_cache/kv_transfer.py)
 * selects the paired component views, uploads the token indices and resolves
 * each host view's MF DVA; this adapter packs the same component/pitch/
 * width/layer description into the kernel launch parameter block, so the
 * description never round-trips through device memory: no device-resident
 * meta, no pinned host staging and no separate H2D meta upload.
 */
HOST_API void kv_exchange_submit(const at::Tensor &device_indices, const at::Tensor &host_indices,
                                 at::TensorList device_components, at::TensorList host_components,
                                 const std::vector<int64_t> &host_dvas, int64_t page_size, int64_t direction,
                                 int64_t host_layout_mode)
{
    TORCH_CHECK(direction == kDirectionH2D || direction == kDirectionD2H, "direction must be H2D (1) or D2H (2), got ",
                direction);
    TORCH_CHECK(host_layout_mode == kHostLayoutPageFirst || host_layout_mode == kHostLayoutLayerFirst,
                "host_layout_mode must be page_first (0) or layer_first (1), got ", host_layout_mode);
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);
    TORCH_CHECK(!device_components.empty() && device_components.size() == host_components.size(),
                "device_components and host_components must be non-empty lists of the same length");
    TORCH_CHECK(device_components.size() <= kMaxComponents, "at most ", kMaxComponents,
                " components are supported, got ", device_components.size());
    TORCH_CHECK(host_dvas.size() == device_components.size(), "host_dvas must match the component count, got ",
                host_dvas.size(), " vs ", device_components.size());

    CheckNpuTensor(device_indices, "device_indices");
    CheckNpuTensor(host_indices, "host_indices");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device/host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        // Nothing to transfer.
        return;
    }

    ComponentParams comps[kMaxComponents] = {};
    for (int64_t c = 0; c < static_cast<int64_t>(device_components.size()); ++c) {
        const at::Tensor &device_buf = device_components[c];
        const at::Tensor &host_buf = host_components[c];
        CheckNpuTensor(device_buf, "device_components");
        TORCH_CHECK(host_buf.device().type() == at::kCPU, "component ", c,
                    ": host components must be CPU (MemFabric-mapped) tensors");
        CheckComponentPair(device_buf, host_buf, page_size, c);
        TORCH_CHECK(host_dvas[static_cast<size_t>(c)] != 0, "component ", c, ": MF host DVA must be non-zero");
        // Layer offsets are already folded into the component views; the
        // whole view (all its layers) is transferred.
        FillComponent(comps[c], device_buf, host_buf, host_dvas[static_cast<size_t>(c)], 0, device_buf.size(0));
    }

    // start_dim: negative (the default) reserves the last 2 AIV cores to
    // match the native acc_offload kv_exchange occupancy; SGLANG_KV_
    // EXCHANGE_START_DIM=0 re-engages all launched cores for perf sweeps.
    const int64_t start_dim = ResolveStartDim();
    // EXEC_KERNEL_CMD requires plain identifier arguments (blockdim goes into
    // a lambda capture list, and the rest must bind to ConvertTypes' lvalue
    // references), so hoist every runtime value into named locals.
    const int64_t num_components = static_cast<int64_t>(device_components.size());
    const int64_t num_pages_local = num_pages;
    const int64_t page_size_local = page_size;
    const int64_t direction_local = direction;
    const int64_t host_layout_local = host_layout_mode;
    const uint32_t block_dim_u32 = ResolveBlockDim();
    EXEC_KERNEL_CMD(kv_exchange, block_dim_u32, device_indices, host_indices, num_components, num_pages_local,
                    page_size_local, direction_local, host_layout_local, start_dim, KV_COMPONENT_ARGS(0),
                    KV_COMPONENT_ARGS(1), KV_COMPONENT_ARGS(2), KV_COMPONENT_ARGS(3));
    RecordStreamLifetime(device_components, device_indices, host_indices);
}

}  // namespace npu_kernel
}  // namespace sglang
