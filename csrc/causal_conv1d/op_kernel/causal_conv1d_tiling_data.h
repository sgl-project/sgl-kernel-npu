/*!
 * \file causal_conv1d_tiling_data.h
 * \brief tiling data struct
 */

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>

namespace sglang {
namespace npu_kernel {

struct CausalConv1dTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t inputMode;

    int64_t width;

    int64_t stateLen;
    int64_t numCacheLines;

    int64_t batch;

    int64_t activationMode;
    int64_t padSlotId;
    int64_t hasBias;

    int64_t dimTileSize;
    int64_t blocksPerSeq;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_TILING_DATA_H_
