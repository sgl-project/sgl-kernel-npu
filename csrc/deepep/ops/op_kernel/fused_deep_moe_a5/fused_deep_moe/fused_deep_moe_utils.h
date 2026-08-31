#ifndef FUSED_DEEP_MOE_UTILS_H
#define FUSED_DEEP_MOE_UTILS_H

#include <kernel_operator.h>
#include "../fused_deep_moe_a5_base.h"

namespace CVSoftSync {
constexpr uint32_t SOFT_SYNC_SPACE_SIZE = 512;
}

template <typename T>
__aicore__ inline T FlushAndGetValue(AscendC::GlobalTensor<T> &globalTensor, uint64_t index)
{
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        globalTensor[index]);
    __asm__ __volatile__("");
    T value = globalTensor.GetValue(index);
    return value;
}

template <typename T>
__aicore__ inline void SetValueAndFlush(AscendC::GlobalTensor<T> &globalTensor, uint64_t index, T value)
{
    globalTensor.SetValue(index, value);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        globalTensor[index]);
    __asm__ __volatile__("");
}

template <typename T>
__aicore__ inline T FlushAndSpinValue(AscendC::GlobalTensor<T> &globalTensor, uint64_t index)
{
    T value = FlushAndGetValue(globalTensor, index);
    if (value == 0) {
        SetValueAndFlush(globalTensor, index, 1);
    } else {
        SetValueAndFlush(globalTensor, index, 0);
    }
    return value;
}

__aicore__ inline void EncreaseSyncFlag(__gm__ int32_t *flagAddr, int32_t idx)
{
    // flag++, like set flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(flagAddr + idx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t));
    int32_t value = FlushAndGetValue<int32_t>(global, 0);
    SetValueAndFlush<int32_t>(global, 0, value + 1);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CheckSyncFlag(__gm__ int32_t *flagAddr, int32_t idx, uint32_t target)
{
    //  check flag, like wait flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(flagAddr + idx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t));
    while (true) {
        int32_t value = FlushAndGetValue<int32_t>(global, 0);
        if (value >= target) {
            break;
        }
        SPIN_WAIT_CYCLES();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// -----------------------------------------------------------------------------
// L2 cross-expert weight prefetch helpers.
// Pull the next expert's GEMM weights from GM into the chip-shared L2 cache by
// issuing chunked GM->UB reads with CACHE_MODE_NORMAL (read-allocate) during
// otherwise idle spin windows. Data written to the UB pad is never consumed.
// Synchronization notes:
//   * All chunks are DataCopy instructions on the MTE2 pipe of the issuing
//     core; a single MTE pipe executes in order and the pad region has no
//     consumer, so no SetFlag/WaitFlag pairing is required for pad reuse.
//   * The first chunk is only issued from inside CheckSyncFlagWithL2Prefetch,
//     which begins with PipeBarrier<PIPE_ALL> (drains earlier users of the UB
//     region, e.g. dispatch buffers). In-flight chunks are drained by the
//     existing PipeBarrier<PIPE_ALL> at the end of each kernel stage.
// -----------------------------------------------------------------------------
namespace L2Prefetch {
constexpr uint32_t UB_PAD_OFFSET = 160 * 1024;  // UB landing pad, above epilogue usage (max 128KB)
constexpr uint32_t CHUNK_BYTES = 32 * 1024;     // MTE2-friendly chunk size (>= 20KB per copy)
constexpr uint32_t SPIN_INTERVAL = 4;           // issue at most one chunk per N spin polls
}  // namespace L2Prefetch

struct L2PrefetchCtx {
    AscendC::GlobalTensor<uint8_t> srcWeight;  // next expert B weights (stripe slice)
    AscendC::GlobalTensor<uint8_t> srcScale;   // next expert MxScaleB (fetched once by stripe 0)
    AscendC::LocalTensor<uint8_t> pad;         // UB landing pad, data discarded
    uint32_t weightBytes{0};
    uint32_t scaleBytes{0};
    uint32_t cursor{0};
    bool active{false};
};

// Slice [stripeId] of the weight region into ctx; scale region is only fetched by stripe 0.
__aicore__ inline void L2PrefetchInit(L2PrefetchCtx &ctx, AscendC::LocalTensor<uint8_t> &pad, __gm__ uint8_t *weightGm,
                                      uint32_t weightBytes, __gm__ uint8_t *scaleGm, uint32_t scaleBytes,
                                      uint32_t stripeId, uint32_t stripeNum)
{
    ctx.pad = pad;
    ctx.cursor = 0;
    ctx.active = false;
    ctx.weightBytes = 0;
    ctx.scaleBytes = 0;
    uint32_t stripeBytes = (((weightBytes + stripeNum - 1) / stripeNum) + 31U) & ~31U;  // 32B aligned slice
    uint32_t stripeStart = stripeBytes * stripeId;
    if (stripeStart < weightBytes) {
        uint32_t remain = weightBytes - stripeStart;
        ctx.weightBytes = remain < stripeBytes ? remain : stripeBytes;
        ctx.srcWeight.SetGlobalBuffer(weightGm + stripeStart);
#if defined(FUSED_DEEP_MOE_ENABLE_L2_READ_AHEAD)
        // Requires a target CANN providing AscendC::CacheMode::CACHE_MODE_READ_AHEAD; verify on board first
        ctx.srcWeight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_READ_AHEAD);
#else
        ctx.srcWeight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
#endif
        ctx.active = ctx.weightBytes > 0;
    }
    if (scaleBytes > 0 && stripeId == 0) {  // scale is tiny relative to weights, single-stripe fetch
        ctx.scaleBytes = scaleBytes;
        ctx.srcScale.SetGlobalBuffer(scaleGm);
        ctx.srcScale.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        ctx.active = true;
    }
}

__aicore__ inline void L2PrefetchDeactivate(L2PrefetchCtx &ctx)
{
    ctx.active = false;
}

// Issue at most one chunk; non-blocking, safe to call from spin loops.
__aicore__ inline void L2PrefetchStep(L2PrefetchCtx &ctx)
{
    if (!ctx.active) {
        return;
    }
    if (ctx.cursor < ctx.weightBytes) {
        uint32_t remain = ctx.weightBytes - ctx.cursor;
        uint32_t chunk = (remain < L2Prefetch::CHUNK_BYTES ? remain : L2Prefetch::CHUNK_BYTES) & ~31U;
        if (chunk == 0) {
            ctx.active = false;
            return;
        }
        AscendC::DataCopy(ctx.pad, ctx.srcWeight[ctx.cursor], chunk);
        ctx.cursor += chunk;
        return;
    }
    uint32_t scaleCursor = ctx.cursor - ctx.weightBytes;
    if (scaleCursor < ctx.scaleBytes) {
        uint32_t remain = ctx.scaleBytes - scaleCursor;
        uint32_t chunk = (remain < L2Prefetch::CHUNK_BYTES ? remain : L2Prefetch::CHUNK_BYTES) & ~31U;
        if (chunk == 0) {
            ctx.active = false;
            return;
        }
        AscendC::DataCopy(ctx.pad, ctx.srcScale[scaleCursor], chunk);
        ctx.cursor += chunk;
        return;
    }
    ctx.active = false;
}

// Same wait semantics as CheckSyncFlag, plus best-effort L2 prefetch between spins.
__aicore__ inline void CheckSyncFlagWithL2Prefetch(__gm__ int32_t *flagAddr, int32_t idx, uint32_t target,
                                                   L2PrefetchCtx *pfCtx)
{
    //  check flag, like wait flag, with L2 prefetch during idle spins
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(flagAddr + idx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t));
    uint32_t spinCnt = 0;
    while (true) {
        int32_t value = FlushAndGetValue<int32_t>(global, 0);
        if (value >= target) {
            break;
        }
        if (pfCtx != nullptr && pfCtx->active && (spinCnt % L2Prefetch::SPIN_INTERVAL) == 0) {
            L2PrefetchStep(*pfCtx);
        }
        SPIN_WAIT_CYCLES();
        ++spinCnt;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif  // FUSED_DEEP_MOE_UTILS_H
