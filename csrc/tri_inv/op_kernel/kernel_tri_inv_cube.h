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
#include "kernel_mat_gen.h"

namespace sglang {

namespace npu_kernel {
/**
 * @brief Returns the matrix inverse of an upper triangular square matrix of
 * size `matrix_size`. The matrix has ones on the main diagonal.
 *
 * The column sweep algorithm is used for the linear system Ax=e_j where e_j is
 * the standard vector.
 *
 * The kernel assumes \f$ matrix_size_ + 1 \f$ synchronizations with the AIVs.
 *
 * @tparam InputT Input data type. Default value fp16/half.
 *
 */
template <typename InputT>
class KernelTriInvCubeColSweep
{
    using OutputT = float;

public:
    /**
     * @brief Class constructor.
     *
     * @param [in] vec_len Total length of input tensor.
     * @param [in] matrix_size Input square matrix size.
     * @param [in] circular_buffer_len Length of workspace circular buffer to
     * overcome GM memory consistency issues.
     */
    __aicore__ inline KernelTriInvCubeColSweep(uint32_t vec_len, uint32_t matrix_size, uint32_t circular_buffer_len)
        : vec_len_(vec_len),
          matrix_size_(matrix_size),
          ws_circular_buffer_len_(circular_buffer_len),
          tile_len_(matrix_size * matrix_size),
          global_in_offset_(AscendC::GetBlockIdx() * tile_len_ * ws_circular_buffer_len_),
          global_out_offset_(AscendC::GetBlockIdx() * tile_len_)
    {}

    /**
     * @brief Initialize global and local memory structures.
     *
     * @param [in] matrix_stream_in Pointer where the input matrices will be
     * written by AIVs in global memory.
     * @param [in] inv_matrix_out Pointer where the output matrix inverses
     * will be written.
     */
    __aicore__ inline void Init(GM_ADDR matrix_stream_in, GM_ADDR inv_matrix_out)
    {
        global_A_.SetGlobalBuffer((__gm__ InputT *)matrix_stream_in, vec_len_ * ws_circular_buffer_len_);
        global_C_.SetGlobalBuffer((__gm__ OutputT *)inv_matrix_out, vec_len_);

        pipe_.InitBuffer(a1_q_, 1, tile_len_ * sizeof(InputT));
        pipe_.InitBuffer(a2_q_, 1, tile_len_ * sizeof(InputT));
        pipe_.InitBuffer(b1_q_, 1, tile_len_ * sizeof(InputT));
        pipe_.InitBuffer(b2_q_, 1, tile_len_ * sizeof(InputT));
        pipe_.InitBuffer(co1_q_, 1, tile_len_ * sizeof(OutputT));
    }

    /**
     * @brief Run kernel.
     *
     */
    __aicore__ inline void Process()
    {
        // On the first iteration, the AIVs will send the identity matrix to AIC
        SyncGroup();
        LoadIdentityMatrixinL0C();

        // Read again the identity matrix from AIV core.
        uint32_t circular_buf_idx = 0;

        // Matrix column sweep algorithm requires `matrix_size_` iterations.
        for (uint32_t iter = 0; iter < matrix_size_; iter++) {
            // Sync with all AIVs in group, to write the matrix.
            SyncGroup();

            // Load next matrix A and perform C = A @ C
            LoadMatrixAintoL0A(circular_buf_idx);
            MultiplyAWithC();
            circular_buf_idx = (circular_buf_idx + 1) % ws_circular_buffer_len_;
        }

        // Write L0C matrix to global memory
        CopyCL0ToGlobal();
    }

private:
    /**
     * @brief Loads matrix from global memory into L0A (`a1_q_` queue).
     */
    __aicore__ inline void LoadMatrixAintoL0A(uint32_t iter)
    {
        // Load matrix from global_A_ into L1
        CopyGmToL1A(a1_q_, global_A_[global_in_offset_ + iter * tile_len_], m_blocks_, k_blocks_);
        CopyL1ToL0A(a2_q_, a1_q_, m_blocks_, k_blocks_);
    }

    /**
     * @brief Copy a tensor from global memory to the queue in A L1 memory.
     *
     * The data layout is transformed from ND to NZ.
     *
     * @param [in] q Destination queue. The position must be A1.
     * @param [in] global Source global tensor.
     * @param [in] fractals_h Number of fractal patterns in the height dimension of
     * the input matrix.
     * @param [in] fractals_w Number of fractal patterns in the width dimension of
     * the input matrix.
     */
    __aicore__ inline void CopyGmToL1A(AscendC::TQue<AscendC::QuePosition::A1, 1> &q,
                                       const AscendC::GlobalTensor<InputT> &global, uint16_t fractals_h,
                                       uint16_t fractals_w)
    {
        const AscendC::LocalTensor<InputT> lt = q.template AllocTensor<InputT>();
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = m_blocks_ * 16;
        params.dValue = k_blocks_ * 16;
        params.srcDValue = params.dValue;
        params.dstNzC0Stride = params.nValue;
        params.dstNzNStride = 1;
        // params.dstNzMatrixStride = 1; // This should have no effect when ndNum=1;
        DataCopy(lt, global, params);
        q.EnQue(lt);
    }

    /**
     * @brief Copy a tensor from global memory to the queue in B L1 memory.
     *
     * The data layout is transformed from ND to NZ.
     *
     * @param [in] q Destination queue. The position must be B1.
     * @param [in] global Source global tensor.
     * @param [in] fractals_h Number of fractal patterns in the height dimension of
     * the input matrix.
     * @param [in] fractals_w Number of fractal patterns in the width dimension of
     * the input matrix.
     */
    __aicore__ inline void CopyGmToL1B(AscendC::TQue<AscendC::QuePosition::B1, 1> &q,
                                       const AscendC::GlobalTensor<InputT> &global, uint16_t fractals_h,
                                       uint16_t fractals_w)
    {
        const AscendC::LocalTensor<InputT> lt = q.template AllocTensor<InputT>();
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = fractals_h * 16;
        params.dValue = fractals_w * 16;
        params.srcDValue = params.dValue;
        params.dstNzC0Stride = params.nValue;
        params.dstNzNStride = 1;
        // params.dstNzMatrixStride = 1; // This should have no effect when ndNum=1;
        DataCopy(lt, global, params);
        q.EnQue(lt);
    }

    /**
     * @brief Copy a tensor from a given A L1 to A L0 queue.
     *
     * The function performs an NZ to ZZ layout transformation.
     *
     * @param [in] l0_q Destination queue. The position must be A2.
     * @param [in] l1_q Source queue. The position must be A1.
     * @param [in] fractals_h Number of fractal patterns in the height dimension of
     * the input matrix.
     * @param [in] fractals_w Number of fractal patterns in the width dimension of
     * the input matrix.
     */
    __aicore__ inline void CopyL1ToL0A(AscendC::TQue<AscendC::QuePosition::A2, 1> &l0_q,
                                       AscendC::TQue<AscendC::QuePosition::A1, 1> &l1_q, uint16_t fractals_h,
                                       uint16_t fractals_w)
    {
        const AscendC::LocalTensor<InputT> l0_lt = l0_q.template AllocTensor<InputT>();
        AscendC::LocalTensor<InputT> l1_lt = l1_q.template DeQue<InputT>();

        int src_offset = 0;
        int dst_offset = 0;

        for (uint16_t i = 0; i < fractals_h; ++i) {
            AscendC::LoadData2dParams params;
            params.repeatTimes = fractals_w;
            params.srcStride = fractals_h;
            params.ifTranspose = false;

            LoadData(l0_lt[dst_offset], l1_lt[src_offset], params);

            src_offset += 16 * 16;
            dst_offset += fractals_w * 16 * 16;
        }

        l1_q.FreeTensor(l1_lt);
        l0_q.EnQue(l0_lt);
    }

    /**
     * @brief Copy a tensor from a given B L1 to B L0 queue.
     *
     * The function performs an NZ to ZN layout transformation.
     *
     * @param [in] l0_q Destination queue. The position must be B2.
     * @param [in] l1_q Source queue. The position must be B1.
     * @param [in] fractals_h Number of fractal patterns in the height dimension of
     * the input matrix.
     * @param [in] fractals_w Number of fractal patterns in the width dimension of
     * the input matrix.
     */
    __aicore__ inline void CopyL1ToL0B(AscendC::TQue<AscendC::QuePosition::B2, 1> &l0_q,
                                       AscendC::TQue<AscendC::QuePosition::B1, 1> &l1_q, uint16_t fractals_h,
                                       uint16_t fractals_w)
    {
        AscendC::LocalTensor<InputT> l1_lt = l1_q.template DeQue<InputT>();
        const AscendC::LocalTensor<InputT> l0_lt = l0_q.template AllocTensor<InputT>();
        constexpr bool transpose = true;
        int src_offset = 0;
        int dst_offset = 0;

        for (uint16_t i = 0; i < fractals_h; ++i) {
            AscendC::LoadData2dParams params;
            params.repeatTimes = fractals_w;
            params.srcStride = fractals_h;
            params.ifTranspose = transpose;

            LoadData(l0_lt[dst_offset], l1_lt[src_offset], params);

            src_offset += 16 * 16;
            dst_offset += fractals_w * 16 * 16;
        }

        l1_q.FreeTensor(l1_lt);
        l0_q.EnQue(l0_lt);
    }

    /**
     * @brief Copy a tensor from CO1 queue to B1 queue.
     *
     * @param [in] dst_q Destination queue. The position must be B1.
     * @param [in] src_q Source queue. The position must be CO1.
     * @param [in] height Height of the matrix.
     * @param [in] width Width of the matrix.
     */
    __aicore__ inline void CopyC01ToB1(AscendC::TQue<AscendC::QuePosition::B1, 1> &dst_q,
                                       AscendC::TQue<AscendC::QuePosition::CO1, 1> &src_q, uint32_t height,
                                       uint32_t width)
    {
        AscendC::LocalTensor<OutputT> src_lt = src_q.template DeQue<OutputT>();
        const AscendC::LocalTensor<InputT> dst_lt = dst_q.template AllocTensor<InputT>();

        AscendC::FixpipeParamsV220 params;
        params.nSize = width;
        params.mSize = height;
        params.srcStride = height;
        params.dstStride = width;
        params.ndNum = 1;
        params.quantPre = QuantMode_t::F322F16;

        AscendC::Fixpipe<InputT, OutputT, AscendC::CFG_NZ>(dst_lt, src_lt, params);
        src_q.FreeTensor(src_lt);
        dst_q.EnQue(dst_lt);
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
        ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
        wait_flag_dev(AIV_SET_FLAG_ID);

        return;
    }

    /**
     * @brief Perform matrix multiplication in Cube unit, like C = A @ C
     *
     * Assumes that the matrices A and C are enqueued.
     */
    __aicore__ inline void MultiplyAWithC()
    {
        AscendC::LocalTensor<InputT> a2_lt = a2_q_.DeQue<InputT>();

        // Load C matrix from L0C into L0B.
        CopyC01ToB1(b1_q_, co1_q_, M_, N_);
        CopyL1ToL0B(b2_q_, b1_q_, k_blocks_, n_blocks_);
        AscendC::LocalTensor<InputT> b2_lt = b2_q_.DeQue<InputT>();

        AscendC::LocalTensor<OutputT> c1_lt = co1_q_.AllocTensor<OutputT>();

        Mmad(c1_lt, a2_lt, b2_lt, {M_, N_, K_, false /* accumulate_c */, 0, false, false, false});

        co1_q_.EnQue<OutputT>(c1_lt);
        a2_q_.FreeTensor(a2_lt);
        b2_q_.FreeTensor(b2_lt);
    }

    /**
     * @brief Loads the identity matrix from global memory to L0C (`co1_q_`
     * queue).
     */
    __aicore__ inline void LoadIdentityMatrixinL0C()
    {
        LoadIdentityMatrixinL0A();
        LoadIdentityMatrixinL0B();
        AscendC::LocalTensor<InputT> a2_lt = a2_q_.DeQue<InputT>();
        AscendC::LocalTensor<InputT> b2_lt = b2_q_.DeQue<InputT>();
        AscendC::LocalTensor<OutputT> c1_lt = co1_q_.AllocTensor<OutputT>();

        Mmad(c1_lt, a2_lt, b2_lt, {M_, N_, K_, false, 0, false, false, false});

        co1_q_.EnQue<OutputT>(c1_lt);
        a2_q_.FreeTensor(a2_lt);
        b2_q_.FreeTensor(b2_lt);
    }

    /**
     * @brief Loads the identity matrix from global memory to L0A (`a1_q_`
     * queue).
     */
    __aicore__ inline void LoadIdentityMatrixinL0A()
    {
        CopyGmToL1A(a1_q_, global_A_[global_in_offset_], m_blocks_, k_blocks_);
        CopyL1ToL0A(a2_q_, a1_q_, m_blocks_, k_blocks_);
    }

    /**
     * @brief Loads the identity matrix from global memory to L0B (`b1_q_`
     * queue).
     */
    __aicore__ inline void LoadIdentityMatrixinL0B()
    {
        // Here, we "abuse" the 'global_A_' pointer
        CopyGmToL1B(b1_q_, global_A_[global_in_offset_], k_blocks_, n_blocks_);

        // Plain copy from L1 to L0B, because the layout is already correct.
        AscendC::LocalTensor<InputT> src = b1_q_.template DeQue<InputT>();
        const AscendC::LocalTensor<InputT> dst = b2_q_.template AllocTensor<InputT>();

        AscendC::LoadData2dParams params;
        params.repeatTimes = n_blocks_ * k_blocks_;
        params.srcStride = 1;
        params.ifTranspose = false;

        LoadData(dst, src, params);

        b1_q_.FreeTensor(src);
        b2_q_.EnQue(dst);
    }

    /**
     * @brief Copy a tensor from CO1 queue to global memory.
     *
     */
    __aicore__ inline void CopyCL0ToGlobal()
    {
        constexpr uint16_t fractal_size = 16;

        AscendC::LocalTensor<OutputT> lt = co1_q_.template DeQue<OutputT>();

        AscendC::FixpipeParams<OutputT> params;
        params.cburstNum = M_;
        params.burstLen = N_ * fractal_size * sizeof(OutputT) / 32;
        params.dstStride = M_;

        AscendC::Nz2NdParams nz2nd_params;
        nz2nd_params.nz2ndEn = true;
        nz2nd_params.originalNSize = M_;
        params.nz2ndParams = nz2nd_params;

        AscendC::Fixpipe(global_C_[global_out_offset_], lt, params);

        co1_q_.FreeTensor(lt);
    }

    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::QuePosition::A1, 1> a1_q_;
    AscendC::TQue<AscendC::QuePosition::A2, 1> a2_q_;
    AscendC::TQue<AscendC::QuePosition::B1, 1> b1_q_;
    AscendC::TQue<AscendC::QuePosition::B2, 1> b2_q_;

    AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_q_;

    AscendC::GlobalTensor<InputT> global_A_;
    AscendC::GlobalTensor<OutputT> global_C_;

    const uint32_t vec_len_;
    const uint32_t matrix_size_;
    const uint32_t ws_circular_buffer_len_;
    const uint32_t tile_len_;
    const uint32_t global_in_offset_;
    const uint32_t global_out_offset_;

    constexpr static uint32_t M_CUBE_BLOCK_SIZE = 16;
    constexpr static uint32_t N_CUBE_BLOCK_SIZE = 16;
    constexpr static uint32_t K_CUBE_BLOCK_SIZE = 16;

    const uint16_t M_ = matrix_size_;
    const uint16_t K_ = matrix_size_;
    const uint16_t N_ = matrix_size_;

    const uint32_t n_blocks_ = N_ / N_CUBE_BLOCK_SIZE;
    const uint32_t k_blocks_ = K_ / K_CUBE_BLOCK_SIZE;
    const uint32_t m_blocks_ = M_ / M_CUBE_BLOCK_SIZE;
};

/**
 * @brief Run the `tri_inv_cube_col_sweep` kernel.
 *
 * @tparam InputT Input data type. Supports fp16/half.
 *
 * @param [in] matrix_stream_in Pointer where the input matrices will be
 * written by AIVs in global memory.
 * @param [in] inv_matrix_out Pointer where the output matrix inverses
 * will be written.
 * @param [in] matrix_size Input square matrix size to invert.
 */
template <typename InputT>
__aicore__ inline void run_tri_inv_cube_col_sweep(GM_ADDR matrix_stream_in, GM_ADDR inv_matrix_out, GM_ADDR workspace,
                                                  uint32_t vec_len, uint32_t matrix_size,
                                                  uint32_t ws_circular_buffer_len)
{
    if ASCEND_IS_AIV {
        KernelMatGen<InputT> op(matrix_size, ws_circular_buffer_len);
        op.Init(matrix_stream_in, workspace);
        op.Process();
    }

    if ASCEND_IS_AIC {
        KernelTriInvCubeColSweep<InputT> op(vec_len, matrix_size, ws_circular_buffer_len);
        op.Init(workspace, inv_matrix_out);
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
