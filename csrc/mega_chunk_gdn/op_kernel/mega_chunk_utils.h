// mega_chunk_utils.h — small device-side helpers shared by the GDN mega-kernel stages.

#pragma once
#include <pto/pto-inst.hpp>

namespace mega_chunk {

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
