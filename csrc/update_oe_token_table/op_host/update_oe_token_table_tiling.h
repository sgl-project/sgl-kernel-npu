#ifndef SGLANG_UPDATE_OE_TOKEN_TABLE_TILING_H
#define SGLANG_UPDATE_OE_TOKEN_TABLE_TILING_H

#include <cstdint>

namespace sglang::npu_kernel {

#pragma pack(push, 1)
struct UpdateOeTokenTableTilingData {
    uint32_t usedCoreNum = 0U;
    uint32_t blockFactor = 0U;
    uint32_t tailBlockFactor = 0U;
    uint32_t ubFactor = 0U;
    uint32_t batchSize = 0U;
    uint32_t maxContextLen = 0U;
    uint32_t ignoreTokenNum = 0U;
};
#pragma pack(pop)

}  // namespace sglang::npu_kernel

#endif  // SGLANG_UPDATE_OE_TOKEN_TABLE_TILING_H
