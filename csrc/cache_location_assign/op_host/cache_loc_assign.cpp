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
#include "tiling/assign_cache_tiling.h"
#include "aclrtlaunch_cache_loc_assign.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t BLK_SIZE = 32;
constexpr uint32_t BLK_SIZE_ALIN_FOR_INT32 = 8;
constexpr uint32_t MAX_STEP = 5;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

uint64_t alinIn64Bytes(uint64_t length)
{
    return (length * sizeof(int64_t) + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE;
}

uint64_t alinIn32Bytes(uint64_t length)
{
    return (length * sizeof(int32_t) + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE;
}

uint64_t alinIn32Count(uint64_t count)
{
    return (count + BLK_SIZE_ALIN_FOR_INT32 - 1) / BLK_SIZE_ALIN_FOR_INT32 * BLK_SIZE_ALIN_FOR_INT32;
}

at::Tensor getTiling(
    uint64_t batchSize, uint64_t rowSize, uint64_t &blockDim, uint64_t &workspaceSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t maxAIVCore = static_cast<uint64_t>(ascendcPlatform->GetCoreNumAiv());
    blockDim = maxAIVCore;
    uint64_t userWorkspaceSize =  DEFAULT_SYNCALL_NEED_SIZE * blockDim;
    uint64_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    workspaceSize = userWorkspaceSize + systemWorkspaceSize;

    auto tilingBuffer =
        at::empty({sizeof(AssignCacheTillingData)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    AssignCacheTillingData *tillingData = reinterpret_cast<AssignCacheTillingData *>(tilingBuffer.data_ptr());
    tillingData->vcoreNum = blockDim;
    tillingData->workspaceSize = workspaceSize;
    tillingData->rowNumNoTail = batchSize / (tillingData->vcoreNum);
    tillingData->tailNum = batchSize % (tillingData->vcoreNum);
    tillingData->rowSize = rowSize;

    tillingData->tokenCountAlignInt32 = alinIn32Count(MAX_STEP);
    tillingData->tokenColAlignInt32 = tillingData->tokenCountAlignInt32 * sizeof(int32_t);

    tillingData->cacheLocSize = batchSize * MAX_STEP;
    tillingData->cacheLocCountAlignIn32 = alinIn32Count(tillingData->cacheLocSize);
    tillingData->cacheLocAlignIn32 = tillingData->cacheLocCountAlignIn32 * sizeof(int32_t);

    tillingData->cacheLocIdxSize = batchSize + 1;
    tillingData->cacheLocIdxCountAlignIn32 = alinIn32Count(tillingData->cacheLocIdxSize);
    tillingData->cacheLocIdxAlignIn32 = tillingData->cacheLocIdxCountAlignIn32* sizeof(int32_t);

    uint64_t maxRowNumPerCore = (batchSize + tillingData->vcoreNum - 1) / tillingData->vcoreNum;
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t maxRowSize4OffSet = alinIn64Bytes(maxRowNumPerCore);
    uint64_t ubBufferSizeToUse = tillingData->tokenColAlignInt32 + 2 * maxRowSize4OffSet +
                                 tillingData->cacheLocAlignIn32 + tillingData->cacheLocIdxAlignIn32;
    if (ubBufferSizeToUse > ubSize) {
        throw std::invalid_argument("Batch size is too large, buffer is not enough to do calculate");
    }

    auto tilingTensor = TorchNpuHepler::CopyTensorHostToDevice(tilingBuffer);
    return tilingTensor;
}

HOST_API at::Tensor cache_loc_assign(const at::Tensor &tokenPool, const at::Tensor &startOffset,
    const at::Tensor &endOffset, const at::Tensor &outCacheLoc, const at::Tensor &outCacheLocIdx)
{
    if (tokenPool.options().dtype() != at::kInt || startOffset.options().dtype() != at::kLong ||
        endOffset.options().dtype() != at::kLong || outCacheLoc.options().dtype() != at::kInt ||
        outCacheLocIdx.options().dtype() != at::kInt) {
        throw std::invalid_argument("Only support inputTensor combo: int32, int64, int64, int32, int32");
    }
    uint64_t rowSize = tokenPool.sizes()[1];
    uint64_t batchSize = tokenPool.sizes()[0];
    uint64_t blockDim;
    uint64_t workspaceSize;
    at::Tensor tilingTensor = getTiling(batchSize, rowSize, blockDim, workspaceSize);

    auto workspace_tensor =
        at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(tokenPool.options().device()));

    /* lauch the kernal function via torch */
    EXEC_KERNEL_CMD(cache_loc_assign,
        blockDim,
        tokenPool,
        startOffset,
        endOffset,
        outCacheLoc,
        outCacheLocIdx,
        workspace_tensor,
        tilingTensor);
    return tokenPool;
}

}  // namespace npu_kernel
}  // namespace sglang
