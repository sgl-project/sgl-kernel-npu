#include <iostream>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_data.h"
#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_assign_cache_op.h"

namespace sglang {
namespace npu_kernel {
using namespace custom_assign;

#define OP_CHECK(expression, error_msg, action)                                                                \
    do {                                                                                                       \
        if (!expression) {                                                                                     \
            std::cerr << "[ERROR] " << (error_msg) << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; \
            action;                                                                                            \
        }                                                                                                      \
    } while (0)

HOST_API at::Tensor GetTilingTensor(CustomAssignTilingData &tilingData, size_t tilingSize)
{
    auto buffer = at::empty({static_cast<int64_t>(tilingSize)}, at::kByte);
    tilingData.SetToBuffer(buffer.data_ptr<uint8_t>(), tilingSize);
    auto tilingTensor = TorchNpuHelper::CopyTensorHostToDevice(buffer);
    return tilingTensor;
}

HOST_API size_t GetElementByteSize(const at::Tensor &tensor)
{
    at::ScalarType dtype = tensor.scalar_type();
    return at::elementSize(dtype);
}

HOST_API bool assign_cache_op(at::Tensor &dstTensor, const at::Tensor &srcTensor, const at::Tensor &dstStartIdx,
                              const at::Tensor &dstEndIdx, const at::Tensor &srcStartIdx, const at::Tensor &srcEndIdx)
{
    auto dstShape = dstTensor.sizes(), dstStartShape = dstStartIdx.sizes(), dstEndShape = dstEndIdx.sizes();
    auto srcShape = srcTensor.sizes(), srcStartShape = srcStartIdx.sizes(), srcEndShape = srcEndIdx.sizes();
    OP_CHECK(dstShape[0] == srcShape[0] && dstStartShape[0] == srcStartShape[0] && dstEndShape[0] == srcEndShape[0],
             "batch size is not same between srcTensor and dstTensor", return false);
    OP_CHECK(dstShape[0] == dstStartShape[0] && dstShape[0] == dstEndShape[0],
             "batch size is not same between srcTensor and dstTensor", return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDim = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t eleBytes = GetElementByteSize(dstTensor);
    uint32_t syncWorkspaceSize = blockDim * 32 + blockDim * 32 + 32;
    struct CustomAssignTilingData tilingData = {.batchSize = static_cast<uint32_t>(dstShape[0]),
                                                .tokenPoolLength = static_cast<uint32_t>(dstShape[1]),
                                                .typeBytes = eleBytes,
                                                .syncWorkspaceSize = syncWorkspaceSize,
                                                .ubSize = static_cast<uint32_t>(ubSize)};
    at::Tensor tiling = GetTilingTensor(tilingData, sizeof(tilingData));

    auto sync = at::zeros({syncWorkspaceSize, 1}, at::kByte);
    auto syncDevice = TorchNpuHelper::CopyTensorHostToDevice(sync);
    EXEC_KERNEL_CMD(assign_cache_op, blockDim, dstTensor, srcTensor, dstStartIdx, dstEndIdx, srcStartIdx, srcEndIdx,
                    syncDevice, tiling);
    return true;
}
}  // namespace npu_kernel

}  // namespace sglang
