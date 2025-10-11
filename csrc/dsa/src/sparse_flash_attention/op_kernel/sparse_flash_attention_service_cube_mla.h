/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_service_cube_mla.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum;             //一般为kv的head num，对应n2
    uint32_t headDim;             //mla下rope为64，nope为512, 对应d
    uint32_t maxblockNumPerBatch; //block table 每一行的最大个数
    uint32_t actHeadDim;          //实际拷贝col大小,考虑到N切块   s*d, 对应d
    uint32_t copyRowNum;          //总共要拷贝的行数
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
};

// 场景：query、queryRope、key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor,
                                        uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;       //nd矩阵的行数
    // T为int4场景下，dValue = col / 2，srcDValue = colStride / 2
    nd2nzPara.dValue = col;          //nd矩阵的列数
    nd2nzPara.srcDValue = colStride; //同一nd矩阵相邻行起始地址间的偏移
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

/*
    适用PA数据从GM拷贝到L1，支持ND、NZ数据；
    PA的layout分 BNBD（blockNum,N,blockSize,D） BBH（blockNum,blockSize,N*D
    BSH\BSND\TND 为BBH
    shape.copyRowNumAlign 需要16字节对齐，如拷贝k矩阵，一次拷贝128*512，遇到尾块 10*512 需对齐到16*512
*/
template <typename T, SFA_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,  //l1
                                  GlobalTensor<T> &srcTensor, //gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,       // blockSize, headNum, headDim                           
                                  const Position &startPos)   // bacthIdx nIdx curSeqIdx
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;   // 获取block table上的索引
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;    // 获取在单个块上超出的行数
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        // 计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt;  //一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;  //一个block未拷满
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim ;   //PA的偏移

        uint64_t dStride = shape.headDim;
        if constexpr (SRC_LAYOUT == SFA_LAYOUT::BSND || SRC_LAYOUT == SFA_LAYOUT::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) +
                      reaminRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) +
                      reaminRowCnt * shape.headDim + startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);                     
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename SFAT> class SFAMatmulService {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SFAT::queryType;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using MM_OUT_T = T;
