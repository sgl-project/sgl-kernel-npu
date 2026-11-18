#pragma once

#include <cstdint>

namespace optiling {

enum UnidexCopyAttr {
    SRC_ADDR,
    DST_ADDR,
    SRC_ROWS,
    DST_ROWS,
    BLOCK_BYTES,
    MAX_COPY,
    BLOCK_DIM,
    UNIDEX_COPY_ATTR_COUNT,
};

struct UnidexCopyTilingData {
    uint64_t srcAddr;
    uint64_t dstAddr;
    uint32_t srcRows;
    uint32_t dstRows;
    uint32_t blockBytes;
    uint32_t maxCopy;
};

}  // namespace optiling
