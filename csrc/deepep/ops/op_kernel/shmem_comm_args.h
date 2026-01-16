#ifndef SHMEM_COMM_ARGS_H
#define SHMEM_COMM_ARGS_H
#include <cstdint>

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#include "kernel_operator.h"

namespace ShmemMoe {
constexpr int CAM_MAX_RANK_SIZE = 384;  // Maximum number of NPU cards supported by the communication library
constexpr int PING_PONG_SIZE = 2;
constexpr int UB_ALIGN = 32;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t ADDR_OFFSET = 32U;

constexpr uint64_t WIN_MAGIC_OFFSET = 100UL * 1024UL;
constexpr uint64_t HALF_WIN_STATE_OFFSET = 180 * 1024UL;  // notify(60KB) + dispatch(60KB) + combine(60KB)
constexpr uint64_t NOTIFY_META_SIZE = 60 * 1024UL;  // notify(60KB)
constexpr uint64_t DISPATCH_META_SIZE = 60 * 1024UL;  // dispatch(60KB)

enum Op : int { COPYONLY = -1, ADD = 0, MUL = 1, MAX = 2, MIN = 3 };

constexpr uint64_t CYCLE_TO_TIME = 50;  // cycle num is converted into a fixed base unit of time, set at 50
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000000UL;

constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr static int32_t UB_HEAD_OFFSET = 96;
constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN;

}  // namespace ShmemMoe
#endif  // SHMEM_COMM_ARGS_H
