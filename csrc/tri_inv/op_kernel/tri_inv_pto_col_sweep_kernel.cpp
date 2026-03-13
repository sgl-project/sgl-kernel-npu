// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../op_host/tiling_tri_inv.h"

#define MEMORY_BASE

#include "pto/pto-inst.hpp"
#include "kernel_operator.h"

using namespace pto;

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

/**
 * @brief Runs triangular matrix inverse on input buffer.
 *
 * @tparam T Input data type (fp16 or fp32).
 * @tparam S Matrix size. Supports 16, 32, 64, 128.

 * @param vec_in Pointer to input buffer in global memory.
 * @param vec_out Pointer to output buffer in global memory.
 * @param total_length Input tensor length, i.e., numel().
 */
template <typename T, unsigned S>
AICORE void runTTriInv(__gm__ T *vec_in, __gm__ T *vec_out, uint32_t total_length)
{
    set_mask_norm();
    set_vector_mask(-1, -1);

    constexpr uint32_t tile_len = S * S;
    const uint32_t matrix_in_size = tile_len * sizeof(T);
    const uint32_t b_size = S * sizeof(T);
    const uint32_t diff_size = S * sizeof(T);

    // UB zero address
    constexpr unsigned UB_ZERO_ADDR = 0x0;

    // define GlobalData on global memory with shape and stride
    using ShapeDim5 = pto::Shape<1, 1, 1, S, S>;
    using StrideDim5 = pto::Stride<1, 1, 1, S, 1>;
    using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

    GlobalData global_in(vec_in);
    TASSIGN(global_in, vec_in + block_idx * tile_len);

    GlobalData global_out(vec_out);
    TASSIGN(global_out, vec_out + block_idx * tile_len);

    // define TileData on UB buffer with static shape and dynamic mask
    using TileData = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, -1, -1>;
    using TileVecData = Tile<TileType::Vec, T, 1, S, BLayout::RowMajor, -1, -1>;

    // Define all tiles
    TileData matrix_in(S, S);
    TASSIGN(matrix_in, UB_ZERO_ADDR);

    TileVecData b(1, S);
    TASSIGN(b, UB_ZERO_ADDR + matrix_in_size);

    TileData inv_matrix_out(S, S);
    const uint32_t out_start_ub_addr = matrix_in_size + b_size + diff_size;
    TASSIGN(inv_matrix_out, out_start_ub_addr);

    // Set output to all zeros.
    TEXPANDS(inv_matrix_out, static_cast<T>(0));

    // synchronization operations between hardware pipelines
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // load data from global memory to UB buffer
    TLOAD(matrix_in, global_in);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TileVecData x(1, S);
    // TODO (anastasios) only first k elements must be updated
    TileVecData diff(1, S);
    TileVecData A_k(1, S);

    // For every output column j-th
    for (int32_t j = 0; j < S; j++) {
        // Column sweep on each column.

        // `b` vector is  j-th standard vector (e_j).
        TEXPANDS(b, static_cast<T>(0));
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        b.SetValue(j, static_cast<T>(1));

        // Solve A x = e_j for vector x
        // Must be offset by UB address
        TASSIGN(x, out_start_ub_addr + j * S * sizeof(T));
        TASSIGN(diff, UB_ZERO_ADDR + matrix_in_size + b_size);

        for (int32_t k = S - 1; k >= 0; k--) {
            TASSIGN(A_k, k * S * sizeof(T));

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            // x[k] = b[k] / A[k, k]
            const T alpha = b.GetValue(k);
            x.SetValue(k, alpha);

            if (k > 0) {
                // b[:k] -= A[:k, k] * x[k]
                TEXPANDS(diff, static_cast<T>(0));
                TMULS(diff, A_k, alpha);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

                TSUB(b, b, diff);
            }
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(global_out, inv_matrix_out);
}

/**
 * @brief Run the `tri_inv_col_sweep` kernel.
 *
 * @tparam T Input data type. Supports fp16/half.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] matrix_size Matrix size to invert.
 */
template <typename T>
AICORE void run_tri_inv_col_sweep(__gm__ T *vec_in, __gm__ T *vec_out, uint32_t vec_len, uint32_t matrix_size)
{
    if (matrix_size == 16) {
        runTTriInv<T, 16>(vec_in, vec_out, vec_len);
    } else if (matrix_size == 32) {
        runTTriInv<T, 32>(vec_in, vec_out, vec_len);
    } else if (matrix_size == 64) {
        runTTriInv<T, 64>(vec_in, vec_out, vec_len);
    } else if (matrix_size == 128) {
        runTTriInv<T, 128>(vec_in, vec_out, vec_len);
    }
}

#elif (__CHECK_FEATURE_AT_PRECOMPILE) || (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

template <typename T>
AICORE void run_tri_inv_col_sweep(__gm__ T *vec_in, __gm__ T *vec_out, uint32_t vec_len, uint32_t matrix_size)
{}

#endif

/**
 * @brief Copies tiling structure from global memory to registers.
 *
 * @tparam TilingT Structure representing kernel tiling parameters.
 * @param [in] tiling Pointer to the structure allocated in registers.
 * @param [in] tiling_global Pointer to the structure in global memory.
 */
template <typename TilingT>
AICORE void GetTilingData(TilingT *const tiling, __gm__ void *tiling_global)
{
    uint32_t *const tiling_32b = reinterpret_cast<uint32_t *>(tiling);
    const __gm__ uint32_t *const tiling_global_32b = reinterpret_cast<__gm__ uint32_t *>(tiling_global);

    for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); i++) {
        tiling_32b[i] = tiling_global_32b[i];
    }
}

/**
 * @brief Run the `tri_inv_col_sweep` kernel on dtype fp16/half.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
// clang-format off
extern "C" __global__ AICORE void tri_inv_pto_col_sweep_fp16(__gm__ void* vec_in, __gm__ void* vec_out,
                                                         __gm__ void* tiling_gm)
{
    // clang-format on
    sglang::npu_kernel::TriInvColumnSweepTiling tiling;
    GetTilingData(&tiling, tiling_gm);
    run_tri_inv_col_sweep<half>((__gm__ half *)vec_in, (__gm__ half *)vec_out, tiling.num_elems, tiling.matrix_size);
}

/**
 * @brief Run the `tri_inv_col_sweep` kernel on dtype float32.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
// clang-format off
extern "C" __global__ AICORE void tri_inv_pto_col_sweep_fp32(__gm__ void* vec_in, __gm__ void* vec_out,
                                                         __gm__ void* tiling_gm)
{
// clang-format off
    sglang::npu_kernel::TriInvColumnSweepTiling tiling;
    GetTilingData(&tiling, tiling_gm);
    run_tri_inv_col_sweep<float>((__gm__ float *)vec_in, (__gm__ float *)vec_out, tiling.num_elems, tiling.matrix_size);
}
