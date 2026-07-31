#ifndef TILING_ARGS_H
#define TILING_ARGS_H
#include <cstdint>

namespace Moe {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 4U * 1024UL * 1024UL;
// A5 multi-round Combine has two 4 MiB token-state ping-pong slots.
constexpr uint64_t A5_MULTI_ROUND_COMBINE_STATE_WIN_OFFSET = 8U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 102U * 1024UL * 1024UL;
}  // namespace Moe
#endif  // TILING_ARGS_H
