#pragma once
#include <cstdint>
namespace optiling {
struct CachePlannerTilingData {
    uint32_t realReqCount, maxContextLen, sourceStride, cacheCapacity, batchSize;
};
}  // namespace optiling
