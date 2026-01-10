// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <memory>
#include "acl/acl.h"
#include "defines.h"
#include "shmem_api.h"
#include "all_reduce_tiling.h"
#include "torch_helper.h"
#include "aclrtlaunch_AllReduce.h"
#include "../../include/zccl.h"

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t BIG_DATA_SIZE = 2 * 1024 * 1024;
constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;
constexpr uint32_t MAGIC = 1024;

namespace sglang {
namespace zccl {

inline size_t GetSizeFromTypeEnum(ZCCLDataType dtype)
{
    switch (dtype) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        default:
            break;
    }
}

template<typename T>
std::shared_ptr<T> MakeAclDevicePtr(size_t size, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST) {
    if (size == 0) {
        return nullptr;
    }
    void* rawPtr = nullptr;
    auto res = aclrtMalloc(&rawPtr, size, policy);
    if (res != 0) {
        return nullptr;  // 分配失败
    }
    return std::shared_ptr<T>(
        static_cast<T*>(rawPtr),
        [](T* p) { if (p) aclrtFree(p); }
    );
}

template<typename T>
std::shared_ptr<T> MakeAclHostPtr(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* rawPtr = nullptr;
    auto res = aclrtMallocHost(&rawPtr, size);
    if (res != 0) {
        return nullptr;  // 分配失败
    }
    return std::shared_ptr<T>(
        static_cast<T*>(rawPtr),
        [](T* p) { if (p) aclrtFreeHost(p); }
    );
}

template<typename T>
std::shared_ptr<T> MakeSharedMemPtr(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* rawPtr = shmem_malloc(size);
    if (rawPtr == nullptr) {
        return nullptr;  // 分配失败
    }
	aclrtMemset(rawPtr, size, 0, size);
    return std::shared_ptr<T>(
        static_cast<T*>(rawPtr),
        [](T* p) { if (p) shmem_free(p); }
    );
}

void SetTilingConfig(uint64_t elements, int teamId, bool isSmallData, uint32_t blockDim,
					std::shared_ptr<AllReduceTilingData> tilingconfig)
{
	// 根据通信组数对核进行进行分组
    uint32_t rankSize = shmem_team_n_pes(teamId);
	uint32_t coreGroupNum = isSmallData ? blockDim : blockDim / 2;
	tilingconfig->coreNumPerRank_ = blockDim / coreGroupNum;
	// 计算每个通信组处理的数据
    tilingconfig->eleNumPerRank_ = elements / rankSize;
	// 计算组内每个核处理的数据, 如果不能对齐, former 处理大块数据,
    tilingconfig->formerNum_ = tilingconfig->eleNumPerRank_ % tilingconfig->coreNumPerRank_;
    tilingconfig->tailNum_ = tilingconfig->coreNumPerRank_ - tilingconfig->formerNum_;
    tilingconfig->tailLength_ = tilingconfig->eleNumPerRank_ / tilingconfig->coreNumPerRank_;
    tilingconfig->formerLength_ =
		(tilingconfig->eleNumPerRank_ - tilingconfig->tailLength_ * tilingconfig->tailNum_) / tilingconfig->formerNum_;
}

// 在host上申请内存，然后进行tilling，再把配置下发到device上
extern "C" HOST_API int ZcclAllReduce(uint8_t *input, uint8_t *output,
    size_t inputNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp)
{
	// 申请host 和 device 上的 tiling 块内存
    auto deviceTilingConfig = MakeAclDevicePtr<AllReduceTilingData>(sizeof(AllReduceTilingData));
	if (deviceTilingConfig == nullptr) {
		return -1;
	}
    auto hostTilingConfig = MakeAclHostPtr<AllReduceTilingData>(sizeof(AllReduceTilingData));
	if (hostTilingConfig == nullptr) {
		return -1;
	}
	// 配置 tiling
	bool isSmallData = true;
	uint32_t blockDim = 0;
    size_t typeSize = getSizeFromTypeEnum(dataType);
    if (inputNumel * typeSize < BIG_DATA_SIZE) {
        blockDim = BLOCK_NUM_SMALL_DATA;
    } else {
        blockDim = BLOCK_NUM_LARGE_DATA;
		isSmallData = false;
    }
    SetTilingConfig(inputNumel, teamId, isSmallData, blockDim, hostTilingConfig);
	// tiling 从 host 下发到 device
    aclrtMemcpy(deviceTilingConfig.get(), sizeof(AllReduceTilingData),
        hostTilingConfig.get(), sizeof(AllReduceTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
	// 申请 shared mem
    uint64_t fftsAddr = shmemx_get_ffts_config();
    size_t gvaSize = blockDim * SYNC_FLAG_INTERVAL * sizeof(int) + GVA_BUFF_MAX_SIZE;
    auto gva = MakeSharedMemPtr<void>(gvaSize);
	if (gva == nullptr) {
		return -1;
	}
    int dataTypeSize = static_cast<int>(dataType);
    ACLRT_LAUNCH_KERNEL(AllReduce)(blockDim, stream, input, output, gva.get(), inputNumel,
        dataTypeSize, teamId, fftsAddr, MAGIC, deviceTilingConfig.get(), reduceOp);
    return 0;
}

}  // namespace zccl
}  // namespace sglang
