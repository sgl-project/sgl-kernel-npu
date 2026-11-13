#pragma once

#include <cstdint>

namespace optiling {

struct SglDcpVerifyPageTablesTilingData {
    uint32_t rows;
    uint32_t slot_rows;
    uint32_t slot_cols;
    uint32_t ordinary_cols;
    uint32_t local_cols;
    uint32_t active_cols;
    uint32_t query_len;
    uint32_t page_size;
    uint32_t dcp_size;
    uint32_t index_bytes;
};

}  // namespace optiling
