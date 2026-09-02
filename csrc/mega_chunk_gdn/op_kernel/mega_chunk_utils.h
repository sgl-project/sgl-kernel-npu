// mega_chunk_utils.h — small device-side helpers shared by the GDN mega-kernel stages.

#pragma once
#include <pto/pto-inst.hpp>

namespace mega_chunk {

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
    constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllA2A3()
{
#if __CCE_AICORE__ == 220
    // These constants are already defined on A5
    constexpr uint16_t SYNC_AIV_FLAG = 12;
    constexpr uint16_t SYNC_AIC_FLAG = 11;
    constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
    constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly) {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
#if defined(__DAV_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
#endif
}

// ── A5 ── mirrors include/pto/npu/a5/SyncAll.hpp:101-111
// Requires: getFFTSMsg from <pto/npu/a5/SyncAll.hpp> (A5 defines its own)
AICORE inline void SyncAllMixA5()
{
#if __CCE_AICORE__ != 220
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    // 1. Wait for both paired AIVs: one flag ID per subblock (base, base + SYNC_FLAG_ID_MAX).
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
    // 2. Global all-AIC barrier.
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    // 3. Release both paired AIVs.
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
    // Set on MTE3 so prior stores drain; wait on PIPE_S.
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
#endif
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllMegaKernel()
{
#if __CCE_AICORE__ == 220
    SyncAllA2A3<isAIVOnly>();
#else
    static_assert(isAIVOnly == false, "No support for AIV only global sync");
    SyncAllMixA5();
#endif
}

template <pipe_t Pipe, uint8_t VEC_NUM = 2>
AICORE inline void SetCrossFlag(int32_t flag)
{
    ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}

template <pipe_t Pipe>
AICORE inline void SignalBothVecOnA5(uint16_t flag)
{
    // A5: the flag offset is 16 on new core.
    constexpr uint16_t VEC_FLAG_OFFSET = 16;

    set_intra_block(Pipe, flag);
    set_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void WaitBothVecOnA5(uint16_t flag)
{
    // A5: the flag offset is 16 on new core.
    constexpr uint16_t VEC_FLAG_OFFSET = 16;

    wait_intra_block(Pipe, flag);
    wait_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

/**
 * @brief Returns the outer matrix layout based on the target architecture and
 * matrix orientation.
 *
 * On DAV C310 targets, the layout depends on whether the matrix is "left-sided"
 * (L0A). DAV C310: L0A is NZ, L0B is ZN. Older: L0A is ZZ, L0B is ZN.
 *
 * Link:
 * https://pto-isa.github.io/docs/isa/cube/nz-fractal-layout/#per-buffer-nz-layouts
 *
 * @param is_left Whether the matrix is on the left side (L0A) or not (L0B).
 * @return The appropriate @c BLayout for the target architecture.
 */
constexpr inline pto::BLayout GetOuterLayout(bool is_left)
{
#ifdef __DAV_C310__
    return is_left ? pto::BLayout::ColMajor : pto::BLayout::RowMajor;
#else
    return pto::BLayout::RowMajor;
#endif
}

/**
 * @brief Pipe in-core barrier for vector core that is a no-op for A5.
 *
 */
AICORE inline void PipeBarrierVec()
{
#if __CCE_AICORE__ == 220
    pipe_barrier(PIPE_V);
#endif
}

}  // namespace mega_chunk
