#ifndef LIM_CACHE_BUDGET_H
#define LIM_CACHE_BUDGET_H

#include <cstdint>
#include "kernel_operator.h"
#include "lim_payload_layout.h"
#include "lim_request_limits.h"

namespace lim_cache_budget {
constexpr uint32_t TOPK = lim_request_limits::TOPK;
constexpr uint32_t BLOCK = lim_request_limits::BLOCK_SIZE;
// Slot payloads reserve two values for misses.
constexpr uint32_t MAX_CACHE_TOKENS = lim_payload_layout::PENDING_MISS_SLOT / BLOCK * BLOCK;

__aicore__ inline bool Valid(uint32_t budget, uint32_t finalLength, uint32_t routes)
{
    return routes != 0U && budget >= routes * TOPK && budget % BLOCK == 0U && budget <= MAX_CACHE_TOKENS &&
           budget <= finalLength;
}
}  // namespace lim_cache_budget

#endif
