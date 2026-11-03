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
 * \file lightning_indexer_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_LIGHTNING_INDEXER_SERVICE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../quant_lightning_indexer_common.h"

namespace QLIKernel {
using namespace QLICommon;
template <typename QLIT>
class QLIMatmul {
public:
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;
    using QK_T = typename QLIT::qkType;

    __aicore__ inline QLIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm, const GlobalTensor<K_T>& keyGm,
                                               const GlobalTensor<Q_T>& queryGm);
    __aicore__ inline void InitParams(const ConstInfo& constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLICommon::RunInfo& qliCubeRunInfo);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t QLI_KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QLI_QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t QLI_M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;
    static constexpr uint32_t QLI_FIX_M_EVENT = EVENT_ID2;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 256;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t FP8_BLOCK_CUBE = 32;
    static constexpr FixpipeConfig QLI_CFG_ROW_MAJOR_UB = {
        CO2Layout::ROW_MAJOR,
        true}; // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t qliS1GmOffset, uint64_t qliS2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const QLICommon::RunInfo& qliCubeRunInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                      const QLICommon::RunInfo& qliCubeRunInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const QLICommon::RunInfo& qliCubeRunInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const QLICommon::RunInfo& qliCubeRunInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset,
                                      const QLICommon::RunInfo& qliCubeRunInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t qliS2GmOffset,
                                    const QLICommon::RunInfo& qliCubeRunInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t qliS2GmOffset,
                                         const QLICommon::RunInfo& qliCubeRunInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;

    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;

    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;
    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<QK_T> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<QK_T> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t qliQueryMte2BufferIndex = 0;
    uint64_t qliQueryMte1BufferIndex = 0;
    uint64_t l0BufIdx_ = 0;

    bool isKeyCacheValid_ = false; // L1中是否有可复用的数据
    uint64_t keyGmStart_ = 0;      // L1数据对应的GM S2偏移
    uint64_t keyLoadedSize_ = 0;   // L1中实际加载的S2元素数量
    uint64_t s2BasicBlock_ = 128;
    uint64_t keyBufferOffset_ = 16384; // 128*128

    ConstInfo qliCubeConstInfo;

private:
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
};

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitParams(const ConstInfo& constInfo)
{
    qliCubeConstInfo = constInfo;
    s2BasicBlock_ = (qliCubeConstInfo.tSize <= 256) ? 256 : 128;
    keyBufferOffset_ = s2BasicBlock_ * D_BASIC_BLOCK; // 128*128
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(qliCubeConstInfo.mBaseSize, 2) * qliCubeConstInfo.s2BaseSize *
                                 sizeof(float)); // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<QK_T>();
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm,
                                                            const GlobalTensor<K_T>& keyGm,
                                                            const GlobalTensor<Q_T>& queryGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeMm1(const QLICommon::RunInfo& qliCubeRunInfo)
{
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT +
                                                                      qliCubeRunInfo.loop % 2);
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
        QLICommon::ConstInfo::CROSS_VC_EVENT + qliCubeRunInfo.loop % 2 + QLICommon::ConstInfo::AIV0_AIV1_OFFSET);
    uint64_t qliS2GmBaseOffset = qliCubeRunInfo.s2Idx * qliCubeConstInfo.s2BaseSize;
    uint64_t s1gProcessSize = qliCubeRunInfo.actMBaseSize;
    uint64_t s2ProcessSize = qliCubeRunInfo.actualSingleProcessSInnerSize;
    if (s2BasicBlock_ == 128) {
        for (uint64_t qliS2GmOffset = 0; qliS2GmOffset < s2ProcessSize; qliS2GmOffset += s2BasicBlock_) {
            WaitFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            uint64_t s2L1RealSize =
                qliS2GmOffset + s2BasicBlock_ > s2ProcessSize ? s2ProcessSize - qliS2GmOffset : s2BasicBlock_;
            if (PAGE_ATTENTION) {
                KeyNd2NzForPA(s2L1RealSize, qliS2GmBaseOffset + qliS2GmOffset, qliCubeRunInfo);
            } else {
                KeyNd2Nz(s2L1RealSize, qliS2GmOffset, qliCubeRunInfo);
            }

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t qliS1GmOffset = 0; qliS1GmOffset < s1gProcessSize;
                 qliS1GmOffset += qliCubeConstInfo.mBaseSize) {
                uint64_t s1gL1RealSize = qliS1GmOffset + qliCubeConstInfo.mBaseSize > s1gProcessSize ?
                                             s1gProcessSize - qliS1GmOffset :
                                             qliCubeConstInfo.mBaseSize;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * qliCubeConstInfo.gSize);
                if (qliCubeRunInfo.isFirstS2InnerLoop && qliS2GmOffset == 0) {
                    qliQueryMte2BufferIndex++;
                    qliQueryMte1BufferIndex = qliQueryMte2BufferIndex;
                    WaitFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + qliQueryMte2BufferIndex % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, qliS1GmOffset, qliCubeRunInfo);
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    qliQueryMte1BufferIndex =
                        qliQueryMte2BufferIndex -
                        (CeilDiv(s1gProcessSize, qliCubeConstInfo.mBaseSize) - 1 - (qliS1GmOffset > 0));
                }
                for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gOffset = 0; s1gOffset < s1gL1SizeAlign2G;
                         s1gOffset += qliCubeConstInfo.mBaseSize) {
                        WaitFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize = s1gOffset + qliCubeConstInfo.mBaseSize > s1gL1SizeAlign2G ?
                                                     s1gL1SizeAlign2G - s1gOffset :
                                                     qliCubeConstInfo.mBaseSize;
                        LoadQueryToL0a(qliS1GmOffset, s1gOffset, s1gL1SizeAlign2G, s1gL0RealSize, qliCubeRunInfo);
                        LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, qliCubeRunInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                        WaitFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, qliCubeRunInfo);

                        SetFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                        Fixp(qliS1GmOffset + s1gOffset, qliS2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize,
                             qliCubeRunInfo);
                        SetFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        l0BufIdx_++;
                    }
                }
                if (qliS2GmOffset + s2BasicBlock_ >= s2ProcessSize && qliCubeRunInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + qliQueryMte1BufferIndex % QUERY_BUF_NUM);
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }
    } else if (s2BasicBlock_ == 256) {
        // 第一个s2循环 keycache置为false
        if (qliCubeRunInfo.isFirstS2InnerLoop) {
            isKeyCacheValid_ = false;
        }
        for (uint64_t qliS2GmOffset = 0; qliS2GmOffset < s2ProcessSize; qliS2GmOffset += s2BasicBlock_) {
            // 缓存命中不需要进行key的搬运
            bool qliKeyCacheHit = isKeyCacheValid_ && (qliS2GmBaseOffset >= keyGmStart_) &&
                                  (qliS2GmBaseOffset + s2ProcessSize <= keyGmStart_ + keyLoadedSize_);
            if (!qliKeyCacheHit) {
                WaitFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                // 缓存未命中，需要从GM搬到L1 min（256, 剩余s2）
                uint64_t s2TotalRemainNum = qliCubeRunInfo.actS2Size - qliS2GmBaseOffset;
                uint64_t qliS2L1LoadSize = (s2TotalRemainNum < s2BasicBlock_) ? s2TotalRemainNum : s2BasicBlock_;
                if (PAGE_ATTENTION) {
                    KeyNd2NzForPA(qliS2L1LoadSize, qliS2GmBaseOffset + qliS2GmOffset, qliCubeRunInfo);
                } else {
                    KeyNd2Nz(qliS2L1LoadSize, qliS2GmOffset, qliCubeRunInfo);
                }

                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

                isKeyCacheValid_ = true;
                keyGmStart_ = qliS2GmBaseOffset;
                keyLoadedSize_ = qliS2L1LoadSize;
            }
            uint64_t l1S2Offset = qliS2GmBaseOffset - keyGmStart_;
            uint64_t l1TotalSize = keyLoadedSize_;
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t qliS1GmOffset = 0; qliS1GmOffset < s1gProcessSize;
                 qliS1GmOffset += qliCubeConstInfo.mBaseSize) {
                uint64_t s1gL1RealSize = qliS1GmOffset + qliCubeConstInfo.mBaseSize > s1gProcessSize ?
                                             s1gProcessSize - qliS1GmOffset :
                                             qliCubeConstInfo.mBaseSize;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * qliCubeConstInfo.gSize);
                if (qliCubeRunInfo.isFirstS2InnerLoop && qliS2GmOffset == 0) {
                    qliQueryMte2BufferIndex++;
                    qliQueryMte1BufferIndex = qliQueryMte2BufferIndex;
                    WaitFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + qliQueryMte2BufferIndex % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, qliS1GmOffset, qliCubeRunInfo);
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    qliQueryMte1BufferIndex =
                        qliQueryMte2BufferIndex -
                        (CeilDiv(s1gProcessSize, qliCubeConstInfo.mBaseSize) - 1 - (qliS1GmOffset > 0));
                }
                uint64_t s2Boundry = l1S2Offset + s2ProcessSize;
                for (uint64_t s2L1Offset = l1S2Offset; s2L1Offset < s2Boundry; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize = s2L1Offset + S2_BASIC_BLOCK_L0 > l1S2Offset + s2ProcessSize ?
                                                l1S2Offset + s2ProcessSize - s2L1Offset :
                                                S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gOffset = 0; s1gOffset < s1gL1SizeAlign2G;
                         s1gOffset += qliCubeConstInfo.mBaseSize) {
                        WaitFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize = s1gOffset + qliCubeConstInfo.mBaseSize > s1gL1SizeAlign2G ?
                                                     s1gL1SizeAlign2G - s1gOffset :
                                                     qliCubeConstInfo.mBaseSize;
                        LoadQueryToL0a(qliS1GmOffset, s1gOffset, s1gL1SizeAlign2G, s1gL0RealSize, qliCubeRunInfo);
                        LoadKeyToL0b(s2L1Offset, l1TotalSize, s2L0RealSize, qliCubeRunInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                        WaitFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, qliCubeRunInfo);

                        SetFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                        Fixp(qliS1GmOffset + s1gOffset, (s2L1Offset - l1S2Offset), s1gL0RealSize, s2L0RealSize,
                             qliCubeRunInfo);
                        SetFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        l0BufIdx_++;
                    }
                }
                if (qliS2GmOffset + s2BasicBlock_ >= s2ProcessSize && qliCubeRunInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + qliQueryMte1BufferIndex % QUERY_BUF_NUM);
                }
            }
            bool qliL1FullyUsed = (qliS2GmBaseOffset + s2ProcessSize >= keyGmStart_ + keyLoadedSize_);
            if (qliL1FullyUsed || qliCubeRunInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                keyL1BufIdx_++;
                isKeyCacheValid_ = false;
            }
        }
    }

    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_CV_EVENT +
                                                                     qliCubeRunInfo.loop % 2);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
        QLICommon::ConstInfo::CROSS_CV_EVENT + qliCubeRunInfo.loop % 2 + QLICommon::ConstInfo::AIV0_AIV1_OFFSET);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t qliS2GmOffset,
                                                 const QLICommon::RunInfo& qliCubeRunInfo)
{
    Nd2NzParams qliNd2nzPara;
    qliNd2nzPara.ndNum = 1;
    qliNd2nzPara.nValue = s2L1RealSize; // 行数
    qliNd2nzPara.dValue = qliCubeConstInfo.headDim;
    qliNd2nzPara.srcDValue = qliCubeConstInfo.headDim;
    qliNd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    qliNd2nzPara.dstNzNStride = 1;
    qliNd2nzPara.srcNdMatrixStride = 0;
    qliNd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_],
             keyGm_[qliCubeRunInfo.tensorKeyOffset + qliS2GmOffset * qliCubeConstInfo.headDim], qliNd2nzPara);
}

// blkNum, blkSize, N2, D
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t qliS2GmOffset,
                                                      const QLICommon::RunInfo& qliCubeRunInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + qliS2GmOffset) / qliCubeConstInfo.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + qliS2GmOffset) % qliCubeConstInfo.kCacheBlockSize;
        uint64_t keyGmOffset =
            blkTableGm_.GetValue(qliCubeRunInfo.bIdx * qliCubeConstInfo.maxBlockNumPerBatch + s2BlkId) *
                qliCubeConstInfo.keyStride0 +
            s2BlkOffset * qliCubeConstInfo.headDim;

        uint64_t qliS2Mte2Size = s2L1RealSize - s2L1Offset;
        qliS2Mte2Size = s2BlkOffset + qliS2Mte2Size >= qliCubeConstInfo.kCacheBlockSize ?
                            qliCubeConstInfo.kCacheBlockSize - s2BlkOffset :
                            qliS2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = qliS2Mte2Size; // 行数
        nd2nzPara.dValue = qliCubeConstInfo.headDim;
        nd2nzPara.srcDValue = qliCubeConstInfo.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_ + s2L1Offset * FP8_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += qliS2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t qliS1GmOffset,
                                                   const QLICommon::RunInfo& qliCubeRunInfo)
{
    uint64_t dstNzC0Stride = CeilAlign(s1gL1RealSize, 2 * qliCubeConstInfo.gSize);
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = qliCubeConstInfo.headDim;
    nd2nzPara.srcDValue = qliCubeConstInfo.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(dstNzC0Stride, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(qliQueryMte2BufferIndex % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[qliCubeRunInfo.tensorQueryOffset + qliS1GmOffset * qliCubeConstInfo.headDim], nd2nzPara);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadQueryToL0a(uint64_t qliS1GmOffset, uint64_t s1gL1Offset,
                                                       uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                                       const QLICommon::RunInfo& qliCubeRunInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(qliCubeConstInfo.headDim, FP8_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             queryL1_[(qliQueryMte1BufferIndex % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                     const QLICommon::RunInfo& qliCubeRunInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(qliCubeConstInfo.headDim, FP8_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_], loadData2DParamsV2);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                   const QLICommon::RunInfo& qliCubeRunInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = qliCubeConstInfo.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::Fixp(uint64_t qliS1GmOffset, uint64_t qliS2GmOffset, uint64_t s1gL0RealSize,
                                             uint64_t s2L0RealSize, const QLICommon::RunInfo& qliCubeRunInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);

    // s1gL0RealSize：2*gSize(128)对齐, 最大256
    // s2L0RealSize <= S2_BASIC_BLOCK_L0, 未约束
    uint32_t qliNSize = (s2L0RealSize + 7) >> 3 << 3; // 32B对齐
    uint32_t qliMSize = (s1gL0RealSize + 1) >> 1 << 1;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // 固定参数
    fixpipeParams.mSize = qliMSize;
    fixpipeParams.srcStride = qliMSize;                             // 已16对齐
    fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(float); // 落到同一个bank
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数

    // nSize已保证N方向32B对齐
    if (qliNSize <= (256 / sizeof(float))) {
        // N方向小于一个bank(256B), 只需搬一个ND块, 且不用补齐
        fixpipeParams.nSize = qliNSize;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
    } else {
        // N方向在(256B, 512B]范围， 直接按512B搬, 注意此时不能开unitflag
        fixpipeParams.nSize = S2_BASIC_BLOCK_L0 / 2; // 分2个ND搬, S2_BASIC_BLOCK_L0不为128会有问题
        fixpipeParams.params.ndNum = 2;
        fixpipeParams.params.srcNdStride = ((fixpipeParams.mSize + 15) / 16) * fixpipeParams.nSize;
        fixpipeParams.params.dstNdStride = qliCubeConstInfo.s2BaseSize * qliCubeConstInfo.mBaseSize / 2;
    }
    Fixpipe<QK_T, QK_T, QLI_CFG_ROW_MAJOR_UB>(mm1ResUB_[(qliCubeRunInfo.loop % 2) * qliCubeConstInfo.s2BaseSize / 2],
                                              cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + 1);

    SetFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + 1);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(QLI_KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QLI_QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(QLI_M_MTE1_EVENT + 1);

    WaitFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(QLI_FIX_M_EVENT + 1);
}
} // namespace QLIKernel
#endif
