// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Block count and column tile selection for the hyper-connection epilogue.
 *
 * Block dispatch on Ascend910_9382 is linear in the block count and not free:
 * an empty vector kernel measures 1.48 us at 4 blocks and 5.37 us at 48, about
 * 0.088 us per block. These kernels run roughly ten vector passes per element
 * (two casts in, the sigmoid expansion, the product, the branch reduction, and
 * the output cast), so they are vector-ALU bound rather than bandwidth bound,
 * and at decode row counts the best block count sits well below the 48
 * available vector cores. A model fitted to the trivial kernel did not
 * transfer, so the tables below are a measured sweep rather than a formula; see
 * csrc/hyper_conn/README.md for the measurement setup.
 *
 * The 2048-row column is a prefill chunk: dispatch cost is irrelevant next to
 * the work, so the optimum saturates every core. Extrapolating from the
 * 64-row entry instead made the combine kernel 1.6x slower than the eager
 * chain it replaces, which is why the table has to cover both phases.
 */

#ifndef SGL_KERNEL_NPU_HYPER_CONN_TILING_H
#define SGL_KERNEL_NPU_HYPER_CONN_TILING_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../op_kernel/hyper_conn_limits.h"

namespace sglang {
namespace npu_kernel {
namespace hyper_conn {

// UB buffers are sized from the tile, so the per-branch width is capped.
constexpr uint32_t MAX_WIDTH = 4096;
// Unified Buffer is 192 KB; leave room for the TPipe bookkeeping.
constexpr uint32_t TUNED_UB_BUDGET = 176 * 1024;
// Smallest column tile worth a strided MTE2 burst.
constexpr uint32_t MIN_COLUMN_TILE = 16;

enum class EpilogueKernel : size_t {
    GROUPED_NORM = 0,
    SCALED_SILU = 1,
    BRANCH_MIX = 2,
    BRANCH_COMBINE = 3,
};

constexpr size_t TUNED_KERNELS = 4;
constexpr size_t TUNED_WIDTH_COUNT = 2;
constexpr size_t TUNED_ROW_COUNT = 11;

// Per-branch widths that were swept. Other widths use the nearest column.
constexpr uint32_t TUNED_WIDTHS[TUNED_WIDTH_COUNT] = {2048, 2560};
// Row counts that were swept. Decode runs on fixed npu-graph batch buckets, so
// the common cases are tabulated exactly and the rest take the next larger.
constexpr uint32_t TUNED_ROWS[TUNED_ROW_COUNT] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 2048};

// Measured optimum block counts, [kernel][width][rows].
constexpr uint32_t TUNED_BLOCKS[TUNED_KERNELS][TUNED_WIDTH_COUNT][TUNED_ROW_COUNT] = {
    // GROUPED_NORM
    {{4, 6, 8, 12, 24, 24, 24, 32, 32, 48, 48}, {4, 8, 12, 12, 20, 24, 24, 32, 32, 48, 48}},
    // SCALED_SILU
    {{1, 2, 4, 4, 6, 8, 8, 12, 12, 48, 48}, {1, 2, 4, 6, 6, 8, 8, 12, 12, 48, 48}},
    // BRANCH_MIX
    {{4, 8, 8, 16, 20, 20, 24, 32, 40, 48, 48}, {4, 8, 8, 16, 20, 32, 24, 40, 40, 48, 48}},
    // BRANCH_COMBINE
    {{2, 4, 8, 12, 20, 20, 24, 24, 24, 48, 48}, {2, 4, 8, 16, 20, 32, 24, 24, 24, 48, 48}},
};

// Measured optimum column tiles, [width][rows]. The mix and combine kernels
// share this one ladder: they read the same [rows, hc * width] layout with the
// same strided branch copy. The ladder is the mix optimum; it is within the
// sweep noise of the combine optimum everywhere except one and two rows, where
// the combine sweep preferred the full width.
constexpr uint32_t TUNED_COLUMN_TILES[TUNED_WIDTH_COUNT][TUNED_ROW_COUNT] = {
    {512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
    {640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280},
};

/* UB bytes the mix kernel needs for one task (shared conservatively by combine).
 *
 * Deliberately conservative in two ways:
 * 1. It still counts the bf16 round scratch buffer that the
 *    fp32-throughout rewrite removed, so it overestimates `branch_mix` by
 *    2 * branches * tile bytes (actual usage is 24 * branches * tile + 4 * tile:
 *    two double-buffered bf16 input tiles = 8 B and four fp32 scratch buffers =
 *    16 B per (hc * tile) element, plus a double-buffered bf16 output tile =
 *    4 B per tile element). Keeping 26 ensures SelectColumnTile returns the
 *    same tile the measured TUNED_COLUMN_TILES sweep was taken at.
 * 2. `branch_combine` actually needs 16 * branches * tile + 8 * tile + 224
 *    bytes (for example ~136 KB at hc = 8, tile = 1024, whereas sharing this
 *    formula caps hc = 8, width = 4096 to tile = 512). Re-measure on device
 *    before relaxing either bound.
 */
inline uint32_t ColumnTileUbBytes(uint32_t branches, uint32_t tile)
{
    return 26 * branches * tile + 4 * tile;
}

inline size_t SelectWidthIndex(uint32_t width)
{
    size_t best = 0;
    uint32_t bestDistance = UINT32_MAX;
    for (size_t index = 0; index < TUNED_WIDTH_COUNT; ++index) {
        const uint32_t tuned = TUNED_WIDTHS[index];
        const uint32_t distance = width > tuned ? width - tuned : tuned - width;
        if (distance < bestDistance) {
            bestDistance = distance;
            best = index;
        }
    }
    return best;
}

inline size_t SelectRowIndex(uint32_t rows)
{
    for (size_t index = 0; index < TUNED_ROW_COUNT; ++index) {
        if (TUNED_ROWS[index] >= rows) {
            return index;
        }
    }
    return TUNED_ROW_COUNT - 1;
}

inline uint32_t TunedBlocks(EpilogueKernel kernel, uint32_t width, uint32_t rows)
{
    return TUNED_BLOCKS[static_cast<size_t>(kernel)][SelectWidthIndex(width)][SelectRowIndex(rows)];
}

/* Column tile for the mix and combine task split.
 *
 * The kernels require the tile to divide `width` and to be a multiple of 16
 * elements, so a strided branch copy lands contiguously in UB and no tile is
 * partial. The tuned value is used when it satisfies both that and the UB
 * budget; otherwise the largest divisor that fits. Returns 0 when no tile
 * fits, which the caller rejects.
 */
inline uint32_t SelectColumnTile(uint32_t width, uint32_t branches, uint32_t rows)
{
    const uint32_t tuned = TUNED_COLUMN_TILES[SelectWidthIndex(width)][SelectRowIndex(rows)];
    if (tuned != 0 && width % tuned == 0 && ColumnTileUbBytes(branches, tuned) <= TUNED_UB_BUDGET) {
        return tuned;
    }
    uint32_t selected = 0;
    for (uint32_t candidate = MIN_COLUMN_TILE; candidate <= width; candidate += MIN_COLUMN_TILE) {
        if (width % candidate == 0 && ColumnTileUbBytes(branches, candidate) <= TUNED_UB_BUDGET) {
            selected = candidate;
        }
    }
    return selected;
}

/* Clamp a tuned block count to the work available and the cores present.
 *
 * The sweep ran on 48 vector cores; a device with fewer cannot exceed its own
 * count, and a block per task beyond the task count only adds dispatch cost.
 */
inline uint32_t LaunchBlocks(uint32_t tuned, uint64_t tasks, int64_t coreNum)
{
    uint64_t blocks = tuned;
    blocks = std::min<uint64_t>(blocks, tasks);
    if (coreNum > 0) {
        blocks = std::min<uint64_t>(blocks, static_cast<uint64_t>(coreNum));
    }
    return static_cast<uint32_t>(blocks == 0 ? 1 : blocks);
}

}  // namespace hyper_conn
}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_HYPER_CONN_TILING_H
