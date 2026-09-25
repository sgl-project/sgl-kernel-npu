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
#include <cmath>
#include <cstdint>
#include <limits>

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_elu_fp16.h"
#include "aclrtlaunch_elu_fp32.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel {

namespace {

/*
 * Choose a per-tile element count that:
 *   - is a multiple of the 32B vector unit for both fp16 and fp32;
 *   - keeps the total UB footprint (double-buffered input/output queues plus
 *     the two scratch buffers: 2 + 2 + 1 + 1 = 6 tiles in flight) inside the
 *     usable Unified Buffer;
 *   - is capped so that tiny inputs do not force excessive zero padding.
 */
uint32_t ComputeTileLength(int64_t elemSize)
{
    constexpr uint64_t UB_RESERVED_BYTES = 16384;
    constexpr uint64_t VEC_ALIGN_ELEMS = 512;
    constexpr uint64_t TILE_ELEMS_CAP = 4096;
    constexpr uint64_t BUFFERS_PER_TILE = 6;  // inX(2) + outY(2) + tmpA(1) + tmpB(1)

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t usableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    uint64_t bytesPerUnit = VEC_ALIGN_ELEMS * static_cast<uint64_t>(elemSize) * BUFFERS_PER_TILE;
    uint64_t alignUnits = (bytesPerUnit > 0) ? (usableUb / bytesPerUnit) : 0;
    if (alignUnits == 0) {
        alignUnits = 1;
    }

    uint64_t tileElems = alignUnits * VEC_ALIGN_ELEMS;
    if (tileElems > TILE_ELEMS_CAP) {
        tileElems = TILE_ELEMS_CAP;
    }
    return static_cast<uint32_t>(tileElems);
}

}  // namespace

HOST_API at::Tensor elu(const at::Tensor &x, double alpha)
{
    /* --- input validation --- */
    TORCH_CHECK(x.dim() >= 1, "elu: input must be at least 1D, got ", x.dim(), "D tensor");
    TORCH_CHECK(x.is_contiguous(), "elu: input must be contiguous");

    at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat, "elu: only float16 and float32 are supported, got ", dtype);

    float alphaFloat = static_cast<float>(alpha);
    TORCH_CHECK(std::isfinite(alphaFloat), "elu: alpha must be finite");

    int64_t numel = x.numel();
    if (numel == 0) {
        return at::empty_like(x);
    }

    /* --- tiling: blockDim and tileLength --- */
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    if (coreNum < 1) {
        coreNum = 1;
    }

    uint32_t tileLength = ComputeTileLength(static_cast<int64_t>(x.element_size()));

    /* Launch one block per tile up to the number of AIV cores. */
    int64_t numTiles = (numel + tileLength - 1) / tileLength;
    int64_t blockDim = std::min(coreNum, std::max<int64_t>(1, numTiles));
    int64_t perBlockElems = blockDim * static_cast<int64_t>(tileLength);

    /*
     * Align the total element count to `blockDim * tileLength` so that every
     * block owns exactly `blockLength / tileLength` full tiles. The kernel is
     * therefore free of in-kernel tail handling. Alignment is achieved by
     * padding the (rare) misaligned inputs with zeros on the host and by
     * returning a narrowed view afterwards. LLM activations are almost always
     * already aligned, in which case no padding copy happens at all.
     */
    int64_t paddedLen = ((numel + perBlockElems - 1) / perBlockElems) * perBlockElems;
    TORCH_CHECK(paddedLen <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()), "elu: input is too large (",
                numel, " elements)");
    bool needsPadding = (paddedLen != numel);

    at::Tensor xWork = x;
    at::Tensor yWork;
    if (needsPadding) {
        /* The kernel treats the input as one linear [numel] vector. Flatten the
         * (possibly multi-dimensional) input first so copying into the 1-D
         * zero-padded buffer does not hit a broadcast error, and keep the
         * flattened view (contiguous input => zero-copy) to feed the kernel. */
        at::Tensor xFlat = x.reshape({numel});
        xWork = at::zeros({paddedLen}, x.options());
        xWork.narrow(0, 0, numel).copy_(xFlat);
        yWork = at::empty({paddedLen}, x.options());
    } else {
        yWork = at::empty_like(x);
    }

    /* Keep temporaries alive until the asynchronous kernel finishes. */
    auto npuStream = c10_npu::getCurrentNPUStream();
    if (needsPadding) {
        xWork.record_stream(npuStream);
    }
    yWork.record_stream(npuStream);

    /* --- launch the kernel --- */
    uint32_t totalLength = static_cast<uint32_t>(paddedLen);
    uint32_t blockDimU32 = static_cast<uint32_t>(blockDim);
    if (dtype == at::kHalf) {
        EXEC_KERNEL_CMD(elu_fp16, blockDimU32, xWork, yWork, totalLength, tileLength, alphaFloat);
    } else {
        EXEC_KERNEL_CMD(elu_fp32, blockDimU32, xWork, yWork, totalLength, tileLength, alphaFloat);
    }

    if (needsPadding) {
        /* Trim the zero-padding tail, then restore the input's original shape.
         * The element count is unchanged and the narrowed view is contiguous,
         * so the reshape to x.sizes() always succeeds. */
        return yWork.narrow(0, 0, numel).view(x.sizes());
    }
    return yWork;
}

}  // namespace npu_kernel
}  // namespace sglang
