// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_tri_inv.h"

#include "../op_host/tiling_tri_inv.h"

/**
 * @brief Run the `tri_inv_col_sweep` kernel on dtype fp16/half.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void tri_inv_col_sweep_fp16(GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR tiling_gm)
{
    sglang::npu_kernel::TriInvColumnSweepTiling tiling;
    sglang::npu_kernel::GetTilingData(&tiling, tiling_gm);
    sglang::npu_kernel::run_tri_inv_col_sweep<half>(vec_in, vec_out, tiling.num_elems, tiling.matrix_size);
}

/**
 * @brief Run the `tri_inv_col_sweep` kernel on dtype float32.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void tri_inv_col_sweep_fp32(GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR tiling_gm)
{
    sglang::npu_kernel::TriInvColumnSweepTiling tiling;
    sglang::npu_kernel::GetTilingData(&tiling, tiling_gm);
    sglang::npu_kernel::run_tri_inv_col_sweep<float>(vec_in, vec_out, tiling.num_elems, tiling.matrix_size);
}
