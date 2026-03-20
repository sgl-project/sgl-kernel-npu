#pragma once

#include <cstdint>

namespace sglang {

namespace npu_kernel {

/**
 * @brief `tri_inv_col_sweep` kernel tiling parameter structure.
 */
struct TriInvColumnSweepTiling {
    /// @brief Number of blocks.
    uint32_t num_blocks;
    /// @brief Total number of input elements.
    uint32_t num_elems;
    /// @brief Input matrix size.
    uint32_t matrix_size;
};

}  // namespace npu_kernel
}  // namespace sglang
