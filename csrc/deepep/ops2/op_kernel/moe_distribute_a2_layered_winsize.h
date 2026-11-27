/*!
 * \file moe_distribute_a2_layered_winsize.h
 * \brief Single source of truth for the fixed HCCL window layout of the A2
 *        multi-server layered (hierarchy) dispatch/combine kernels.
 *
 * The layered kernels hard-code the window layout below (RDMA double buffer +
 * IPC offset + NOTIFY region). The host tiling functions need the very same
 * constants to validate HCCL_BUFFSIZE before launch: the device-side asserts
 * are compiled out in release builds, and the unsigned window arithmetic on an
 * undersized window wraps around into huge offsets that end in out-of-window
 * MTE writes ("invalid GM address").
 *
 * Keep the host checks, the kernel layout and python/deep_ep/doc/A2_DEEPEP.md
 * in sync through this header. tests/python/deepep/test_a2_layered_winsize.py
 * (CPU-only) locks the consistency.
 */

#ifndef MOE_DISTRIBUTE_A2_LAYERED_WINSIZE_H
#define MOE_DISTRIBUTE_A2_LAYERED_WINSIZE_H

#include <cstdint>

namespace MoeDistributeA2Layered {
/* RDMA data region (double buffer) at the head of the window. */
constexpr uint64_t RDMA_DATA_SIZE = 800UL * 1024UL * 1024UL;
/* Reserved area (flags/counters/magic) at the base of the IPC share region. */
constexpr uint64_t IPC_DATA_OFFSET = 4UL * 1024UL * 1024UL;
/* Notify/status region at the tail of the window. */
constexpr uint64_t NOTIFY_DATA_SIZE = 400UL * 1024UL * 1024UL;
/* Window bytes consumed before a single token can be staged (1204MB). */
constexpr uint64_t FIXED_WIN_OVERHEAD = RDMA_DATA_SIZE + IPC_DATA_OFFSET + NOTIFY_DATA_SIZE;

/* Per token the IPC expert slots additionally hold expert ids, weights, quant
 * scales and an arrival flag (EXTRA_TOKEN_INFO_NUM entries of k uint32 each;
 * the dispatch kernel pads k up to K_ALIGN_ENTRIES int32 entries). */
constexpr uint64_t EXTRA_TOKEN_INFO_NUM = 4UL;
constexpr uint64_t TOKEN_INFO_UNIT_SIZE = 4UL; /* sizeof(uint32_t) */
/* Upper bound of sizeof(XType)/sizeof(ExpandXType) (bfloat16/float16). */
constexpr uint64_t TOKEN_DTYPE_MAX_SIZE = 2UL;
/* alignK_ = RoundUp(k, BITS32_PER_BLOCK) in the dispatch kernel. */
constexpr uint64_t K_ALIGN_ENTRIES = 8UL;

constexpr uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1UL) / align * align;
}

/* Minimal window the layered dispatch kernel can lay out without overflowing.
 * Mirrors winSizeMin in moe_distribute_dispatch_v2_layered_custom.h with
 * sizeof(XType) replaced by its upper bound TOKEN_DTYPE_MAX_SIZE, so the host
 * check is always at least as strict as the kernel assert. */
constexpr uint64_t CalcDispatchWinSizeMin(uint64_t moeExpertNum, uint64_t bs, uint64_t h, uint64_t k)
{
    return moeExpertNum * bs *
               (h * TOKEN_DTYPE_MAX_SIZE + EXTRA_TOKEN_INFO_NUM * AlignUp(k, K_ALIGN_ENTRIES) * TOKEN_INFO_UNIT_SIZE) +
           FIXED_WIN_OVERHEAD;
}

/* Same for the layered combine kernel: mirrors winSizeMin in
 * moe_distribute_combine_v2_layered_custom.h (raw k, no alignment). */
constexpr uint64_t CalcCombineWinSizeMin(uint64_t moeExpertNum, uint64_t bs, uint64_t h, uint64_t k)
{
    return moeExpertNum * bs * (h * TOKEN_DTYPE_MAX_SIZE + EXTRA_TOKEN_INFO_NUM * k * TOKEN_INFO_UNIT_SIZE) +
           FIXED_WIN_OVERHEAD;
}
}  // namespace MoeDistributeA2Layered

#endif  // MOE_DISTRIBUTE_A2_LAYERED_WINSIZE_H
