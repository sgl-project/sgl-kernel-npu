/*!
 * \file causal_conv1d_update_tilingdata.h
 * \brief tiling data struct
 */

#ifndef CAUSAL_CONV1D_UPDATE_TILING_DATA_H_
#define CAUSAL_CONV1D_UPDATE_TILING_DATA_H_

#include <cstdint>

namespace sglang {
namespace npu_kernel {

struct CausalConv1dUpdateTilingData {
    // used core num
    int64_t numCore;

    // batch per core
    int64_t blockFactor;
    int64_t blockTailFactor;
    // token per loop
    // int64_t baseN;

    // x [batch, seqlen, dim]
    // weight [width, dim]
    int64_t batch;
    int64_t seqLen;
    int64_t dim;
    int64_t width;
    int64_t stateLen;

    int64_t hasIndices;
    int64_t hasBias;
    int64_t hasNumAccept;
    int64_t hasQueryLoc;
    int64_t activationMode;
    int64_t padSlotId;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_UPDATE_TILING_DATA_H_
