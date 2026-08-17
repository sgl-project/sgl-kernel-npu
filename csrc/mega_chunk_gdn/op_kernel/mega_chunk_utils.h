// mega_chunk_utils.h — small device-side helpers shared by the GDN mega-kernel stages.

#pragma once
#include <pto/pto-inst.hpp>

namespace mega_chunk {

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
