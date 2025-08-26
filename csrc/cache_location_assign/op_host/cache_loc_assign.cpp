// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdexcept>
#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/cache_loc_assign.h"
#include "aclrtlaunch_cache_loc_assign.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t BLK_SIZE_ALIN_FOR_INT64 = 4;
constexpr uint32_t BLK_SIZE_ALIN_FOR_INT32 = 8;
constexpr uint32_t MAX_STEP = 5;

uint64_t alinInt64Count(uint64_t count)
{
    return (count + BLK_SIZE_ALIN_FOR_INT64 - 1) / BLK_SIZE_ALIN_FOR_INT64 * BLK_SIZE_ALIN_FOR_INT64;
}

uint64_t alinInt32Count(uint64_t count)
{
    return (count + BLK_SIZE_ALIN_FOR_INT32 - 1) / BLK_SIZE_ALIN_FOR_INT32 * BLK_SIZE_ALIN_FOR_INT32;
}

at::Tensor getTiling(const at::Tensor &reqPoolIndices, uint64_t rowSize, uint64_t poolSize, uint32_t &blockDim)
{
    auto batchSize = reqPoolIndices.sizes()[0];
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    blockDim = ascendcPlatform->GetCoreNumAiv();

    auto tilingBuffer =
        at::empty({sizeof(AssignCacheTillingData)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    AssignCacheTillingData *tillingData = reinterpret_cast<AssignCacheTillingData *>(tilingBuffer.data_ptr());
    tillingData->vcoreNum = blockDim;
    tillingData->poolSize = poolSize;
    tillingData->batchSize = batchSize;
    tillingData->rowNumNoTail = batchSize / (tillingData->vcoreNum);
    tillingData->tailNum = batchSize % (tillingData->vcoreNum);
    tillingData->rowSize = rowSize;

    if (reqPoolIndices.options().dtype() == at::kInt) {
        tillingData->key = 1;
        tillingData->reqInxBufferCount = alinInt32Count(batchSize);
        tillingData->reqInxBufferSize = tillingData->reqInxBufferCount * sizeof(int32_t);
    } else if (reqPoolIndices.options().dtype() == at::kLong) {
        tillingData->key = 2;
        tillingData->reqInxBufferCount = alinInt64Count(batchSize);
        tillingData->reqInxBufferSize = tillingData->reqInxBufferCount * sizeof(int64_t);
    }

    tillingData->tokenCountAlignInt32 = alinInt32Count(MAX_STEP);
    tillingData->tokenColAlignInt32 = tillingData->tokenCountAlignInt32 * sizeof(int32_t);

    tillingData->offsetCountAlignInt64 = alinInt64Count(batchSize);
    tillingData->offsetColAlignInt64 = tillingData->offsetCountAlignInt64 * sizeof(int64_t);

    tillingData->cacheLocSize = batchSize * MAX_STEP;
    tillingData->cacheLocCountAlignInt32 = alinInt32Count(tillingData->cacheLocSize);
    tillingData->cacheLocAlignInt32 = tillingData->cacheLocCountAlignInt32 * sizeof(int32_t);

    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t ubBufferSizeToUse = tillingData->tokenColAlignInt32 + 3 * tillingData->offsetColAlignInt64 +
                                 + 3 * batchSize * sizeof(int32_t) + tillingData->cacheLocAlignInt32;
    if (ubBufferSizeToUse > ubSize) {
        throw std::invalid_argument("Batch size is too large, buffer is not enough to do calculate");
    }

    auto tilingTensor = TorchNpuHepler::CopyTensorHostToDevice(tilingBuffer);
    return tilingTensor;
}

HOST_API at::Tensor cache_loc_assign(const at::Tensor &reqPoolIndices, const at::Tensor &tokenPool,
    const at::Tensor &startOffset, const at::Tensor &endOffset, const at::Tensor &outCacheLoc)
{
    auto reqIdxType = reqPoolIndices.options().dtype();
    if ((reqIdxType != at::kInt && reqIdxType != at::kLong) || tokenPool.options().dtype() != at::kInt ||
        startOffset.options().dtype() != at::kLong || endOffset.options().dtype() != at::kLong ||
        outCacheLoc.options().dtype() != at::kInt) {
        throw std::invalid_argument("Only support inputTensor combo1: int64, int32, int64, int64, int32; combo2: "
                                    "int32, int32, int64, int64, int32");
    }

    uint64_t poolSize = tokenPool.sizes()[0];
    uint64_t rowSize = tokenPool.sizes()[1];
    uint32_t blockDim;
    at::Tensor tilingTensor = getTiling(reqPoolIndices, rowSize, poolSize, blockDim);

    /* lauch the kernal function via torch */
    EXEC_KERNEL_CMD(
        cache_loc_assign, blockDim, reqPoolIndices, tokenPool, startOffset, endOffset, outCacheLoc, tilingTensor);
    return tokenPool;
}

}  // namespace npu_kernel
}  // namespace sglang