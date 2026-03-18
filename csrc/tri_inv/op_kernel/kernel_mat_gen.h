/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_mat_gen.h
 * @brief Kernel implementing an AIV matrix generator that generates the matrix
 * formulation of column sweep.
 */
#pragma once

namespace sglang {

namespace npu_kernel {
/**
 * @brief Returns a sequence of matrices that encode the column-sweep steps in
 * matrix notation. On the first iteration, it returns the identity matrix.
 *
 * @tparam Input data type. Support fp16/half.
 *
 * See discussion on Section 3.2.1 of [1], in particular Equations 3.8 and 3.9
 * on page 54.
 *
 * \code{.python}
  def aiv_matrix_gen(A: npt.ArrayLike):
    n = A.shape[0]
    I_n = np.eye(n, dtype=A.dtype)

    # Your transformation A = 2I_n - A
    A = 2 * I_n - A

    for k in reversed(range(n)):
      M = I_n.copy()
      M[:, k] = A[:, k]
      yield M

  * \endcode
  *
  *  [1] Parallelism in Matrix Computations.E.Gallopoulos, B.Philippe and
 A.H.Sameh.
  * Hard cover(ISBN : 978 - 94 - 017 - 7187 - 0),
  * Soft cover(ISBN : 978 - 94 - 024 - 0317 - 6),
  * Electronic(ISBN : 978 - 94 - 017 - 7188 - 7)
  **/
template <typename T = half>
class KernelMatGen
{
public:
    /**
     * @brief Class constructor.
     *
     * @param [in] matrix_size Input square matrix size.
     * @param [in] circular_buffer_len Length of workspace circular buffer to
     * overcome GM memory consistency issues.
     */
    __aicore__ inline KernelMatGen(uint32_t matrix_size, uint32_t circular_buffer_len)
        : matrix_size_(matrix_size),
          tile_len_(matrix_size * matrix_size),
          aic_id_(AscendC::GetBlockIdx() / AscendC::GetTaskRation()),
          global_in_offset_(aic_id_ * tile_len_),
          ws_circular_buffer_len_(circular_buffer_len),
          global_out_offset_(aic_id_ * tile_len_ * ws_circular_buffer_len_)
    {}

    /**
     * @brief Initialize global and local memory structures.
     *
     * @param [in] vec_in Pointer to the input vector in global memory.
     * @param [in] vec_out Pointer to the output vector in global memory.
     */
    __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out)
    {
        const uint32_t vec_len = AscendC::GetBlockNum() * tile_len_;
        global_in_.SetGlobalBuffer((__gm__ T *)vec_in, vec_len);
        global_out_.SetGlobalBuffer((__gm__ T *)vec_out, vec_len * ws_circular_buffer_len_);

        pipe_.InitBuffer(in_q_, 1, tile_len_ * sizeof(T));
        pipe_.InitBuffer(out_q_, 1, tile_len_ * sizeof(T));
        pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(T));
    }

    /**
     * @brief Run the kernel.
     *
     */
    __aicore__ inline void Process()
    {
        // Read input matrix into work_buf_.
        const AscendC::LocalTensor<T> in_lt = in_q_.template AllocTensor<T>();
        DataCopy(in_lt, global_in_[global_in_offset_], in_lt.GetSize());
        in_q_.EnQue(in_lt);

        ReadInputMatrixInUB();

        // AIV-0 writes identity matrix for AIC
        if (AscendC::GetSubBlockIdx() == 0) {
            EnQueueIdentityMatrix();
            AscendC::LocalTensor<T> out_lt = out_q_.template DeQue<T>();
            DataCopy(global_out_[global_out_offset_], out_lt, out_lt.GetSize());
            out_q_.FreeTensor(out_lt);
        }

        //  Sync with all AIVs in group, to write the matrix.
        SyncGroup();

        // First matrix is identity (just wait one more round)
        SyncGroup();

        const AscendC::LocalTensor<T> work_lt = work_buf_.Get<T>();
        uint32_t circular_buf_idx = 1;

        // Matrix column sweep algorithm requires `matrix_size_` iterations.
        for (int32_t col_index = matrix_size_ - 2; col_index >= 0; col_index--) {
            // AIV-0: writes the  (col_index + 1)-th column of the identity matrix and
            // writes the "column-sweep" column of matrix `M`.
            if (AscendC::GetSubBlockIdx() == 0) {
                const AscendC::LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();
                AscendC::Duplicate(vec_out_lt, static_cast<T>(0), matrix_size_ * matrix_size_);
                AscendC::PipeBarrier<PIPE_ALL>();

                // Set one on the main diagonal
                for (uint32_t i = 0; i < matrix_size_; i++) {
                    vec_out_lt.SetValue(i * matrix_size_ + i, static_cast<T>(1));
                }

                AscendC::PipeBarrier<PIPE_ALL>();
                // Write the (col_index)-th column of matrix M.
                const uint32_t col_offset = col_index * matrix_size_;
                DataCopy(vec_out_lt[col_offset], work_lt[col_offset], matrix_size_);
                AscendC::PipeBarrier<PIPE_ALL>();
                out_q_.EnQue<T>(vec_out_lt);

                AscendC::LocalTensor<T> out_lt = out_q_.template DeQue<T>();
                DataCopy(global_out_[global_out_offset_ + circular_buf_idx * tile_len_], out_lt, out_lt.GetSize());
                out_q_.FreeTensor(out_lt);
                circular_buf_idx = (circular_buf_idx + 1) % ws_circular_buffer_len_;
            }

            // Sync with all AIVs in group, to write the matrix.
            SyncGroup();
        }
    }

private:
    /**
     * @brief Read (and transform) the input triangular matrix A into the
     * `work_buf_`. The transformation is `2*I_n - A`.
     */
    __aicore__ inline void ReadInputMatrixInUB()
    {
        AscendC::LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
        AscendC::LocalTensor<T> work_lt = work_buf_.Get<T>();
        Muls(work_lt, vec_in_lt, static_cast<T>(-1), vec_in_lt.GetSize());
        for (uint32_t i = 0; i < matrix_size_; i++) {
            work_lt.SetValue(i * matrix_size_ + i, 1);
        }
        in_q_.FreeTensor<T>(vec_in_lt);
    }

    /**
     * @brief EnQue identity matrix on output queue.
     *
     */
    __aicore__ inline void EnQueueIdentityMatrix()
    {
        const AscendC::LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();
        AscendC::Duplicate(vec_out_lt, static_cast<T>(0), matrix_size_ * matrix_size_);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Set one on the main diagonal
        for (uint32_t i = 0; i < matrix_size_; i++) {
            vec_out_lt.SetValue(i * matrix_size_ + i, static_cast<T>(1));
        }

        out_q_.EnQue<T>(vec_out_lt);
    }

    /**
     * @brief Returns a synchronization config.
     *
     * @param [in] mode Synchronization mode.
     * @param [in] flag_id Flag to use for synchronization.
     * @return Synchronization config.
     */
    __aicore__ inline int GetSyncConf(int mode, int flag_id)
    {
        return 1 | (mode << 4) | (flag_id << 8);
    }

    /**
     * @brief Synchronize cube and vector cores within a single group.
     *
     */
    __aicore__ inline void SyncGroup()
    {
        const int mode = 2;

        const int AIV_SET_FLAG_ID = 11;
        const int AIC_SET_FLAG_ID = 12;
        ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
        wait_flag_dev(AIC_SET_FLAG_ID);
        return;
    }

    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> in_q_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_q_;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> work_buf_;

    AscendC::GlobalTensor<T> global_in_;
    AscendC::GlobalTensor<T> global_out_;

    const uint32_t matrix_size_;
    const uint32_t tile_len_;
    const uint32_t aic_id_;
    const uint32_t global_in_offset_;
    const uint32_t ws_circular_buffer_len_;
    const uint32_t global_out_offset_;
};
}  // namespace npu_kernel
}  // namespace sglang