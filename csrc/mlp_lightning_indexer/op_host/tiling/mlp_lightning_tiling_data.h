#ifndef MLP_LIGHTNING_TILING_DATA_H
#define MLP_LIGHTNING_TILING_DATA_H

#include <cstdint>

namespace sglang::MlpLIHost {

#pragma pack(push, 1)
struct LITilingData {
    uint32_t bSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t gSize = 0U;
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    uint32_t sparseCount = 0U;
    uint32_t blockLen = 1U;
    uint32_t qBlockLen = 1U;
    uint32_t initNum = 0U;
    uint32_t localNum = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t blockSize = 0U;
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t sparseMode = 0U;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int8_t returnValue = 0;
    uint32_t tilingKey = 0U;
};
#pragma pack(pop)

}  // namespace sglang::MlpLIHost

#endif  // MLP_LIGHTNING_TILING_DATA_H
