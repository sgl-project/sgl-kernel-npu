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
#include <cstdint>
#include <limits>

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_elementwise_add_fp16.h"
#include "aclrtlaunch_elementwise_add_fp32.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel {

namespace {

/* UB budget that is not available to user tensors (system/stack reserve). */
constexpr uint64_t UB_RESERVED_BYTES = 16384;
/* 512 elements is >= 32 B for fp16/fp32, i.e. the vector-unit alignment. */
constexpr uint64_t VEC_ALIGN_ELEMS = 512;
/* Upper bound on one tile: keeps the tail waste small for short inputs. */
constexpr uint64_t TILE_ELEMS_CAP = 8192;
/* inQueueX(2) + inQueueY(2) + outQueueZ(2) tiles live in the UB at once. */
constexpr uint64_t IN_FLIGHT_TILES = 6;

/*
 * Pick the per-tile element count from the actual UB capacity of the running
 * device rather than hard-coding it. Keeping this on the host is what lets the
 * same kernel binary run on parts with different UB sizes.
 */
uint32_t ComputeTileLength(int64_t elemSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t usableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    uint64_t bytesPerUnit = VEC_ALIGN_ELEMS * static_cast<uint64_t>(elemSize) * IN_FLIGHT_TILES;
    uint64_t units = (bytesPerUnit > 0) ? (usableUb / bytesPerUnit) : 0;
    if (units == 0) {
        units = 1;  /* minimum legal tile: one alignment unit */
    }
    uint64_t tileElems = std::min<uint64_t>(units * VEC_ALIGN_ELEMS, TILE_ELEMS_CAP);
    return static_cast<uint32_t>(tileElems);
}

int64_t CoreNum()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    return coreNum < 1 ? 1 : coreNum;
}

}  // namespace

HOST_API at::Tensor elementwise_add(const at::Tensor &x, const at::Tensor &y)
{
    /* --- 1. validate every assumption before touching the device --- */
    TORCH_CHECK(x.dim() >= 1, "elementwise_add: x must be at least 1D, got ", x.dim(), "D tensor");
    TORCH_CHECK(y.dim() == x.dim(), "elementwise_add: x and y must have the same rank, got ", x.dim(), " and ",
                y.dim());
    TORCH_CHECK(x.sizes() == y.sizes(), "elementwise_add: x and y must have the same shape");
    TORCH_CHECK(x.is_contiguous(), "elementwise_add: x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "elementwise_add: y must be contiguous");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "elementwise_add: x and y must share one dtype");

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
                "elementwise_add: only float16 and float32 are supported, got ", dtype);

    const int64_t numel = x.numel();
    if (numel == 0) {
        return at::empty_like(x);
    }
    TORCH_CHECK(numel <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "elementwise_add: input is too large (", numel, " elements)");

    /* --- 2. tiling: one full tile per loop iteration, one block per slice --- */
    const uint32_t tileLength = ComputeTileLength(static_cast<int64_t>(x.element_size()));
    const int64_t numTiles = (numel + tileLength - 1) / tileLength;
    const int64_t blockDim = std::min(CoreNum(), std::max<int64_t>(1, numTiles));
    const int64_t blockSpan = blockDim * static_cast<int64_t>(tileLength);

    /*
     * Pad the element count up to blockDim * k * tileLength. The kernel then
     * always processes whole tiles: no tail branch, no DataCopyPad, and the
     * same code path for aligned and unaligned inputs. Most LLM activations
     * are already aligned, so the extra copy is the exception, not the rule.
     */
    const int64_t paddedLen = ((numel + blockSpan - 1) / blockSpan) * blockSpan;
    TORCH_CHECK(paddedLen <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "elementwise_add: input is too large after alignment (", numel, " elements)");
    const bool needsPadding = (paddedLen != numel);

    at::Tensor xWork = x;
    at::Tensor yWork = y;
    at::Tensor zWork;
    if (needsPadding) {
        /* Flatten first: copying a multi-dimensional tensor into a 1-D buffer
         * would otherwise hit a broadcasting error. */
        xWork = at::zeros({paddedLen}, x.options());
        yWork = at::zeros({paddedLen}, y.options());
        xWork.narrow(0, 0, numel).copy_(x.reshape({numel}));
        yWork.narrow(0, 0, numel).copy_(y.reshape({numel}));
        zWork = at::empty({paddedLen}, x.options());
    } else {
        zWork = at::empty_like(x);
    }

    /* Host temporaries must outlive the asynchronous kernel. */
    auto npuStream = c10_npu::getCurrentNPUStream();
    if (needsPadding) {
        xWork.record_stream(npuStream);
        yWork.record_stream(npuStream);
    }
    zWork.record_stream(npuStream);

    /* --- 3. launch --- */
    /* EXEC_KERNEL_CMD takes non-const lvalue references, so bind the flat
     * views to named objects instead of passing temporaries. */
    at::Tensor xFlat = xWork.view({-1});
    at::Tensor yFlat = yWork.view({-1});
    at::Tensor zFlat = zWork.view({-1});
    const uint32_t totalLength = static_cast<uint32_t>(paddedLen);
    const uint32_t blockDimU32 = static_cast<uint32_t>(blockDim);

    if (dtype == at::kHalf) {
        EXEC_KERNEL_CMD(elementwise_add_fp16, blockDimU32, xFlat, yFlat, zFlat, totalLength, tileLength);
    } else {
        EXEC_KERNEL_CMD(elementwise_add_fp32, blockDimU32, xFlat, yFlat, zFlat, totalLength, tileLength);
    }

    /* --- 4. trim the padding and restore the caller's shape --- */
    if (needsPadding) {
        return zWork.narrow(0, 0, numel).view(x.sizes());
    }
    return zWork;
}

}  // namespace npu_kernel
}  // namespace sglang
