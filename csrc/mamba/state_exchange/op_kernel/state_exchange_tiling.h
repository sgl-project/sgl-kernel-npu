// BSD 3-Clause License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

#pragma once

#include <cstdint>

namespace optiling {
constexpr int32_t STATE_EXCHANGE_MAX_ENTRIES = 16;
constexpr int32_t STATE_EXCHANGE_META_STRIDE = 8;

enum StateExchangeAttrField {
    NUM_ENTRIES,
    DIRECTION,
    START_DIM,
    BLOCK_DIM,
    ATTR_HEADER_SIZE,
};
constexpr int32_t STATE_EXCHANGE_ATTR_SIZE = ATTR_HEADER_SIZE + STATE_EXCHANGE_MAX_ENTRIES * STATE_EXCHANGE_META_STRIDE;

struct SglStateExchangeTilingData {
    int64_t numEntries;
    int64_t indicesLen;
    int64_t direction;
    int64_t startDim;
    int64_t entries[STATE_EXCHANGE_MAX_ENTRIES][STATE_EXCHANGE_META_STRIDE];
};
}  // namespace optiling
