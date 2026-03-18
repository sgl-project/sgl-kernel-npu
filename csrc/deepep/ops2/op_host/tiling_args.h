#ifndef TILING_ARGS_H
#define TILING_ARGS_H
#include <cstdint>

namespace Moe {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 8U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 404U * 1024UL * 1024UL;
constexpr int64_t STATE_SIZE = 2 * 1024 * 1024;
}  // namespace Moe
#endif  // TILING_ARGS_H
