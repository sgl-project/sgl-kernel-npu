#pragma once

#include <cstdint>

namespace optiling {

struct SglDcpSparseIndexRemapTilingData {
    uint32_t rows;
    int32_t page_shift;
    int32_t dcp_shift;
    int32_t dcp_rank;
};

}  // namespace optiling
