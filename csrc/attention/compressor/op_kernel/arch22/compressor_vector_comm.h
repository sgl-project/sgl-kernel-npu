/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_vector_comm.h
 * \brief common components for various vector operations
 */

#ifndef COMPRESSOR_VECTOR_COMM_H
#define COMPRESSOR_VECTOR_COMM_H

#include "compressor_comm.h"
namespace Compressor {

struct MatRpeatParam {
    uint32_t row;
    uint32_t col;
    uint32_t dtypeMask;
    uint32_t loopTimes;
    uint32_t colRemain;
    uint8_t repeatStride;
};

struct RmsNormParam {
    float reciprocal;
    float epsilon;
    uint32_t row;
    uint32_t col;
};

/**
 * @brief ColumnSum sums the matrix by column
 * @param dstLocal output tensor [1, col]; may share the same space as shareTmpUb
 * @param srcLocal input tensor [row, col]
 * @param shareTmpUb temporary buffer; internal space required is [ceil(row / 2) * col * sizeof(float)]
 * @param row number of rows
 * @param col number of columns
 */
__aicore__ inline void ColumnSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                 const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    // when the number of rows is 1, directly copy srcLocal to dstLocal
    if (unlikely(row == 1)) {
        DataCopy(dstLocal, srcLocal, row * col);
        PipeBarrier<PIPE_V>();
        return;
    }
    for (uint32_t mask = MAX_R << 1; mask > 1; mask >>= 1) {
        if (row & mask) {
            // sum the input in halves and put into the temporary space
            Add(shareTmpUb, srcLocal, srcLocal[mask * col / 2],
                mask * col / 2);  // 2: perform computation on the matrix by column
            PipeBarrier<PIPE_V>();
            // add the remainder to the first half
            if (unlikely(row > mask)) {
                if ((row - mask) > (mask >> 1)) {
                    Add(shareTmpUb, shareTmpUb, srcLocal[mask * col],
                        mask * col / 2);  // 2: perform computation on the matrix by column
                    PipeBarrier<PIPE_V>();
                    Add(shareTmpUb, shareTmpUb, srcLocal[(mask + (mask >> 1)) * col], (row - mask - (mask >> 1)) * col);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(shareTmpUb, shareTmpUb, srcLocal[mask * col], (row - mask) * col);
                    PipeBarrier<PIPE_V>();
                }
            }
            // each time add the second half rows to the first half
            for (uint32_t i = mask >> 2; i > 1; i >>= 1) {
                Add(shareTmpUb, shareTmpUb, shareTmpUb[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            if (mask == 2) {  // 2: last matrix operation
                DataCopy(dstLocal, shareTmpUb, col);
            } else {
                Add(dstLocal, shareTmpUb, shareTmpUb[col], col);
            }
            PipeBarrier<PIPE_V>();
            break;
        }
    }
}

/**
 * @brief ColumnMax computes the maximum value of the matrix by column
 * @param dstLocal output tensor [1, col]; may share the same space as shareTmpUb
 * @param srcLocal input tensor [row, col]
 * @param shareTmpUb temporary buffer; internal space required is [ceil(row / 2) * col * sizeof(float)]
 * @param row number of rows
 * @param col number of columns
 */
__aicore__ inline void ColumnMax(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                 const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    // when the number of rows is 1, directly copy srcLocal to dstLocal
    if (unlikely(row == 1)) {
        DataCopy(dstLocal, srcLocal, row * col);
        PipeBarrier<PIPE_V>();
        return;
    }
    for (uint32_t mask = MAX_R << 1; mask > 1; mask >>= 1) {
        if (row & mask) {
            // take the max of the input in halves and put into the temporary space
            Max(shareTmpUb, srcLocal, srcLocal[mask * col / 2],
                mask * col / 2);  // 2: perform computation on the matrix by column
            PipeBarrier<PIPE_V>();
            // take the max of the remainder and the first half, then add to the first half
            if (unlikely(row > mask)) {
                if ((row - mask) > (mask >> 1)) {
                    Max(shareTmpUb, shareTmpUb, srcLocal[mask * col],
                        mask * col / 2);  // 2: perform computation on the matrix by column
                    PipeBarrier<PIPE_V>();
                    Max(shareTmpUb, shareTmpUb, srcLocal[(mask + (mask >> 1)) * col], (row - mask - (mask >> 1)) * col);
                    PipeBarrier<PIPE_V>();
                } else {
                    Max(shareTmpUb, shareTmpUb, srcLocal[mask * col], (row - mask) * col);
                    PipeBarrier<PIPE_V>();
                }
            }
            // each time take the max of the second half rows and the first half, then add to the first half
            for (uint32_t i = mask >> 2; i > 1; i >>= 1) {
                Max(shareTmpUb, shareTmpUb, shareTmpUb[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            if (mask == 2) {  // 2: last matrix operation
                DataCopy(dstLocal, shareTmpUb, col);
            } else {
                Max(dstLocal, shareTmpUb, shareTmpUb[col], col);
            }
            PipeBarrier<PIPE_V>();
            break;
        }
    }
}

/**
 * @brief MatSubVec subtracts a vector from the matrix row by row
 * @param dstLocal output tensor [row, col]
 * @param src0Local input tensor [row, col]
 * @param src1Local input tensor [1, col]
 * @param repeatParam describes the layout of the data to process, including
            row number of rows
            col number of columns
            dtypeMask number of elements participating in computation per iteration
            loopTimes number of loop iterations
            colRemain remaining number of columns
            repeatStride loop stride (actual column length in memory)
 */
__aicore__ inline void MatSubVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief MatDivVec divides the matrix row by row by a vector
 * @param dstLocal output tensor [row, col]
 * @param src0Local input tensor [row, col]
 * @param src1Local input tensor [1, col]
 * @param repeatParam describes the layout of the data to process, including
            row number of rows
            col number of columns
            dtypeMask number of elements participating in computation per iteration
            loopTimes number of loop iterations
            colRemain remaining number of columns
            repeatStride loop stride (actual column length in memory)
 */
__aicore__ inline void MatDivVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief MatMulVec multiplies the matrix row by row by a vector
 * @param dstLocal output tensor [row, col]
 * @param src0Local input tensor [row, col]
 * @param src1Local input tensor [1, col]
 * @param repeatParam describes the layout of the data to process, including
            row number of rows
            col number of columns
            dtypeMask number of elements participating in computation per iteration
            loopTimes number of loop iterations
            colRemain remaining number of columns
            repeatStride loop stride (actual column length in memory)
 */
__aicore__ inline void MatMulVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief RowSum sums each row of the matrix
 * @param dstLocal output tensor [1, row]
 * @param srcLocal input tensor [row, col]
 * @param shareTmpUb temporary buffer; internal space required is [row, col]; may share the same space as srcLocal
 * @param repeatParam describes the layout of the data to process, including
            row number of rows
            col number of columns
            dtypeMask number of elements participating in computation per iteration
            loopTimes number of loop iterations
            colRemain remaining number of columns
            repeatStride loop stride (actual column length in memory)
 */
__aicore__ inline void RowSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                              const LocalTensor<float> &shareTmpUb, const MatRpeatParam &repeatParam)
{
    uint32_t blockCount = repeatParam.loopTimes;
    if (blockCount > 0 && repeatParam.colRemain > 0) {
        Add(shareTmpUb, srcLocal, srcLocal[blockCount * repeatParam.dtypeMask], repeatParam.colRemain, repeatParam.row,
            {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount >> 1; loopCount > 0; loopCount = blockCount >> 1) {
        blockCount = (blockCount + 1) >> 1;
        for (uint32_t i = 0; i < loopCount; i++) {
            Add(shareTmpUb[i * repeatParam.dtypeMask], srcLocal[i * repeatParam.dtypeMask],
                srcLocal[(i + blockCount) * repeatParam.dtypeMask], repeatParam.dtypeMask, repeatParam.row,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstLocal, shareTmpUb,
                   (repeatParam.col < repeatParam.dtypeMask) ? repeatParam.col : repeatParam.dtypeMask, repeatParam.row,
                   1, 1, repeatParam.repeatStride);
}

/**
 * @brief RowDivs divides each row of the matrix by the corresponding element
 * @param dstLocal output tensor [row, col]
 * @param src0Local input tensor [row, col]
 * @param src1Local input tensor [row, 1]; must be expanded into one datablock (actual memory must be [row,
 FP32_BLOCK_ELEMENT_NUM])
 * @param repeatParam describes the layout of the data to process, including
            row number of rows
            col number of columns
            dtypeMask number of elements participating in computation per iteration
            loopTimes number of loop iterations
            colRemain remaining number of columns
            repeatStride loop stride (actual column length in memory)
 */
__aicore__ inline void RowDivs(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                               const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 1});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 1});
        }
    }
}

/**
 * @brief RowMuls multiplies each row of the matrix by the same element
 * @param dstLocal output tensor [row, col]
 * @param src0Local input tensor [row, col]
 * @param src1Local input tensor [row, 1]; must be expanded into one datablock (actual memory must be [row,
 FP32_BLOCK_ELEMENT_NUM])
 * @param repeatParam describes the layout of the data to process, including
            row number of rows
            col number of columns
            dtypeMask number of elements participating in computation per iteration
            loopTimes number of loop iterations
            colRemain remaining number of columns
            repeatStride loop stride (actual column length in memory)
 */
__aicore__ inline void RowMuls(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                               const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 1});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 1});
        }
    }
}

}  // namespace Compressor
#endif  // COMPRESSOR_VECTOR_COMM_H
