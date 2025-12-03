#ifndef LIGHTNING_TILING_DATA_H
#define LIGHTNING_TILING_DATA_H
#include <cstdint>

namespace sglang {
namespace LIHost {

// -----------算子TilingData定义---------------
#pragma pack(push, 1)
struct LITilingData {
    uint32_t bSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t gSize = 0U;
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    uint32_t sparseCount = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t blockSize = 0U;
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t sparseMode = 0U;
    uint32_t tilingKey = 0U;
};
#pragma pack(pop)
}  // namespace LIHost
}  // namespace sglang
#endif
