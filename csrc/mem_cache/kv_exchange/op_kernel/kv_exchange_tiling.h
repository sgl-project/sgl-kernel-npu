// BSD 3-Clause License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

#pragma once

#include <cstdint>

namespace optiling {
constexpr int32_t KV_EXCHANGE_MAX_COMPONENTS = 4;
constexpr int32_t KV_EXCHANGE_META_STRIDE = 9;

enum KvExchangeAttrField {
    NUM_COMPONENTS,
    PAGE_SIZE,
    DIRECTION,
    HOST_LAYOUT,
    START_DIM,
    BLOCK_DIM,
    ATTR_HEADER_SIZE,
};
constexpr int32_t KV_EXCHANGE_ATTR_SIZE = ATTR_HEADER_SIZE + KV_EXCHANGE_MAX_COMPONENTS * KV_EXCHANGE_META_STRIDE;

struct SglKvExchangeTilingData {
    int64_t numComponents;
    int64_t numPages;
    int64_t pageSize;
    int64_t direction;
    int64_t hostLayout;
    int64_t startDim;
    int64_t components[KV_EXCHANGE_MAX_COMPONENTS][KV_EXCHANGE_META_STRIDE];
};
}  // namespace optiling
