#pragma once

#include <cstdint>

namespace optiling {

struct SlotMapLookupTilingData {
    uint32_t size;
    uint32_t maxContextLen;
    uint32_t batchSize;
    uint32_t topk;
};

}  // namespace optiling
