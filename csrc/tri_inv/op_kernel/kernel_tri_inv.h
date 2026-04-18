// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * @file kernel_tri_inv.h
 * @brief Kernel implementing a Vector matrix inverse kernel operation.
 */

#pragma once

#include "kernel_operator.h"

namespace sglang {

namespace npu_kernel {
/**
 * @brief Returns the matrix inverse of an upper triangular square matrix of
 * size `matrix_size`. The matrix has ones on the main diagonal.
 *
 * The column sweep algorithm is used for the linear system Ax=e_j where e_j is
 * the standard vector.
 *
 * @tparam T Input data type. Supports only `half` and `float32`.
 *
 */
template <typename T>
class KernelTriInvColumnSweep
{
    constexpr static uint32_t BUFFER_NUM = 1;

public:
    /**
     * @brief Class constructor.
     *
     * @param [in] vec_len Total length of input tensor.
     * @param [in] matrix_size Input square matrix size.
     */
    __aicore__ inline KernelTriInvColumnSweep(uint32_t vec_len, uint32_t matrix_size)
        : vec_len_(vec_len), matrix_size_(matrix_size), tile_len_(matrix_size * matrix_size)
    {}

    /**
     * @brief Initialize global and local memory structures.
     *
     * @param [in] vec_in Pointer to the input vector in global memory.
     * @param [in] vec_out Pointer to the output vector in global memory.
     */
    __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out)
    {
        global_in_.SetGlobalBuffer((__gm__ T *)vec_in, vec_len_);
        global_out_.SetGlobalBuffer((__gm__ T *)vec_out, vec_len_);

        pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
        pipe_.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(T));
        pipe_.InitBuffer(b_buf_, matrix_size_ * sizeof(T));
    }

    /**
     * @brief Run the kernel.
     */
    __aicore__ inline void Process()
    {
        using namespace AscendC;
        const uint32_t global_offset = AscendC::GetBlockIdx() * tile_len_;

        const AscendC::LocalTensor<T> tile_in_lt = in_q_.AllocTensor<T>();
        AscendC::DataCopy(tile_in_lt, global_in_[global_offset], tile_len_);
        in_q_.EnQue(tile_in_lt);

        InvertMatrix();

        AscendC::LocalTensor<T> tile_out_lt = out_q_.DeQue<T>();
        AscendC::DataCopy(global_out_[global_offset], tile_out_lt, tile_len_);
        out_q_.FreeTensor(tile_out_lt);
    }

private:
    __aicore__ inline void InvertMatrix()
    {
        using namespace AscendC;

        const int32_t n_rows = matrix_size_;
        const int32_t n_cols = matrix_size_;

        LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
        const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();

        // Left-hand side Ax=b.
        LocalTensor<T> b = b_buf_.Get<T>();

        Duplicate(vec_out_lt, static_cast<T>(0), tile_len_);

        // For every output column j-th
        for (int32_t j = 0; j < n_cols; j++) {
            // Column sweep on each column.

            // `b` vector is e_j standard vector.
            Duplicate(b, static_cast<T>(0), matrix_size_);
            b.SetValue(j, static_cast<T>(1));

            // Ax=b
            LocalTensor<T> x = vec_out_lt[j * n_rows];
            for (int32_t k = n_rows - 1; k >= 0; k--) {
                const LocalTensor<T> A_k = vec_in_lt[k * n_rows];

                // x[k] = b[k] / A[k, k]
                x.SetValue(k, b.GetValue(k));

                if (k > 0) {
                    // b[:k] -= A[:k, k] * x[k]
                    const float x_k = -static_cast<float>(x.GetValue(k));
                    AscendC::Axpy<T>(b, A_k, static_cast<T>(x_k), k);
                }
            }
        }

        out_q_.EnQue<T>(vec_out_lt);
        in_q_.FreeTensor<T>(vec_in_lt);
    }

    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> in_q_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> out_q_;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> b_buf_;

    AscendC::GlobalTensor<T> global_in_;
    AscendC::GlobalTensor<T> global_out_;

    const uint32_t vec_len_;
    const uint32_t matrix_size_;
    const uint32_t tile_len_;
};

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
__aicore__ inline void run_tri_inv_col_sweep(GM_ADDR vec_in, GM_ADDR vec_out, uint32_t vec_len, uint32_t matrix_size)
{
    if ASCEND_IS_AIV {
        KernelTriInvColumnSweep<T> op(vec_len, matrix_size);
        op.Init(vec_in, vec_out);
        op.Process();
    }
}

/**
 * @brief Copies tiling structure from global memory to registers.
 *
 * @tparam TilingT Structure representing kernel tiling parameters.
 * @param [in] tiling Pointer to the structure allocated in registers.
 * @param [in] tiling_global Pointer to the structure in global memory.
 */
template <typename TilingT>
__aicore__ inline void GetTilingData(TilingT *const tiling, GM_ADDR tiling_global)
{
    uint32_t *const tiling_32b = reinterpret_cast<uint32_t *>(tiling);
    const __gm__ uint32_t *const tiling_global_32b = reinterpret_cast<__gm__ uint32_t *>(tiling_global);

    for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); i++) {
        tiling_32b[i] = tiling_global_32b[i];
    }
}

}  // namespace npu_kernel
}  // namespace sglang
