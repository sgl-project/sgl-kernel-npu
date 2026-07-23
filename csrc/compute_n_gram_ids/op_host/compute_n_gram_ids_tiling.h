#ifndef SGLANG_COMPUTE_N_GRAM_IDS_TILING_H
#define SGLANG_COMPUTE_N_GRAM_IDS_TILING_H

#include <cstdint>

namespace sglang::npu_kernel {

#pragma pack(push, 1)
struct ComputeNGramIdsTilingData {
    uint32_t coreNum = 0U;
    uint32_t batchSize = 0U;
    uint32_t totalTask = 0U;
    uint32_t oeN = 0U;
    uint32_t oeK = 0U;
    uint32_t maxContextLen = 0U;
};
#pragma pack(pop)

}  // namespace sglang::npu_kernel

#endif  // SGLANG_COMPUTE_N_GRAM_IDS_TILING_H
