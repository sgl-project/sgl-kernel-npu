#ifndef APPLY_TOP_K_TOP_P_MIN_P_TILING_DATA_H
#define APPLY_TOP_K_TOP_P_MIN_P_TILING_DATA_H
#include <cstdint>

namespace sglang {
namespace ATKTPMPHost {

// -----------算子TilingData定义---------------
#pragma pack(push, 1)
struct ApplyTopKTopPMinPTilingData {
    int64_t batchSize = 0;
    int64_t vocabSize = 0;
    int64_t batchPerCore = 0;
    int64_t batchTailCore = 0;
    int64_t ubSize = 0;
    int64_t coreNum = 0;
    int64_t loopDataNum = 0;
    int64_t tilingKey = 0;
};
#pragma pack(pop)
}  // namespace ATKTPMPHost
}  // namespace sglang
#endif
