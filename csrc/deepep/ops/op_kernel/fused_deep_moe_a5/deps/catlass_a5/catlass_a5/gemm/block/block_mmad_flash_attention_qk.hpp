/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief matmul implementation for single q&k^t base tile
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_HPP
#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"

////////////////////////////////////////////////////////////////////
#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr int32_t NUM1 = 1;
constexpr int32_t NUM4 = 4;
constexpr int32_t NUM64 = 64;
constexpr int32_t NUM512 = 512;
constexpr int32_t NUM576 = 576;
constexpr int32_t BASIC_BLOCK_SIZE = 256;
constexpr int32_t Q_BLK = 256;
constexpr int32_t MAX_STACK_LEN = 512;
constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;

struct PFAKernelParams {
    // Data members
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline PFAKernelParams() {}

    __aicore__ inline PFAKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
            GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
            actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_) {}
};

enum class Format
{
    TND = 0,
    BSND = 1
};

enum class CacheMode 
{
    normalCache = 0,
    pagedCache = 1,
};

enum class PageShape 
{
    BnBsND = 0,
    BnNBsD = 1,
    normalShape = 2,
};

enum class MaskCategory 
{
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SWA = 4,
};

enum class CacheLayout : uint8_t
{
    nd = 0,
    nz = 1,
};

#endif
////////////////////////////////////////////////////////////////////
namespace Catlass::Gemm::Block {
struct BlockMmadQKTileHelper {
    uint32_t qkL1TileM;
    uint32_t qkL1TileN;
    uint32_t qkL1TileKLeft;
    uint32_t qkL1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;

    __aicore__ inline
    BlockMmadQKTileHelper() {}

    __aicore__ inline
    BlockMmadQKTileHelper(
        uint32_t m,
        uint32_t n,
        uint32_t kl,
        uint32_t kr,
        uint32_t pbn,
        uint32_t vbn) :
        qkL1TileM(m),
        qkL1TileN(n),
        qkL1TileKLeft(kl),
        qkL1TileKRight(kr),
        qL1BufNum(pbn),
        kL1BufNum(vbn) {}
};

template <
    class ArchTag_,
    bool PAGED_CACHE_FLAG_,
    bool NZ_LAYOUT_FLAG_,
    bool PA_BNNBSD_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadFlashAttentionQK<ArchTag_, PAGED_CACHE_FLAG_, NZ_LAYOUT_FLAG_, PA_BNNBSD_FLAG_, false>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_>
{
public:
    using DispatchPolicy = MmadFlashAttentionQK<ArchTag_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / (L0_STAGES * 2);

    static constexpr uint32_t MAX_L1_STAGES = 3; // 编译期常量，为静态L1Tensor数组开辟准备。取一个buffer份数的极大值
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV
    static constexpr uint32_t STRIDE_LIMIT = 65535;

    __aicore__ inline
    BlockMmadTla(Arch::Resource<ArchTag> &resource, BlockMmadQKTileHelper &BlockMmadQKTileHelper)
    {
        l1ATileM = BlockMmadQKTileHelper.qkL1TileM;
        l1BTileN = BlockMmadQKTileHelper.qkL1TileN;
        l1ATileK = BlockMmadQKTileHelper.qkL1TileKLeft;
        l1BTileK = BlockMmadQKTileHelper.qkL1TileKRight;
        l1ABufNum = BlockMmadQKTileHelper.qL1BufNum;
        l1BBufNum = BlockMmadQKTileHelper.kL1BufNum;

        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1ATileM * l1ATileK * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
                L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    // Destructor
    __aicore__ inline
    ~BlockMmadTla() {}

    template <class TensorA>
    __aicore__ inline
    void loadQGM(TensorA &gATensor, GemmCoord actualOriShape)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});
        auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        auto gATensorTlaTile = GetTile(gATensor,
            tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }
    
    __aicore__ inline
    uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }
    
    template <class TensorB, class L1BTile>
    __aicore__ inline
    void CopyGmToL1BwithNZ(TensorB& gBTile, L1BTile& l1BTile, uint32_t blockIdx, uint32_t blockOffset, uint32_t len,
                uint32_t embed, uint32_t blockSize, uint32_t kvHeads, uint32_t curBaseTileSize) {
        const uint32_t blockCount = embed / 16;
        const uint32_t blockLen = len;

        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = blockSize - blockLen;
        repeatParams.dstStride = RoundUp(curBaseTileSize, 16) - blockLen;

        auto dstOffset = l1BTile.layout()(l1BTile.coord());
        auto srcOffset = (blockIdx * blockSize) * kvHeads * embed + blockOffset * 16;
        AscendC::DataCopy(l1BTile.data()[dstOffset], gBTile.data()[srcOffset], repeatParams);
    }

    template <class TensorB, class TensorC>
    __aicore__ inline
    void operator()(TensorB &gBTensor, TensorC &ubCTensor,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    GemmCoord actualOriShape, uint32_t blockSize,
                    uint32_t kvSTileIdx, uint32_t kvSeqlenTriDown, uint32_t kvHeads,
                    uint32_t kvNumTokens, uint32_t kvSBaseTile, uint32_t isShrink, 
                    uint32_t globalWindowSize, uint32_t localWindowSize,
                    Arch::CrossCoreFlag qkReadyFlag,
                    uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
        CopyL0CToDst copyL0CToDstSub0;
        CopyL0CToDst copyL0CToDstSub1;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t curBaseTileSize = actualOriShape[1];

        uint32_t l1BBufId = kvSTileIdx % l1BBufNum;
        uint32_t l1BEventId = l1BBufId + 1;

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});

        // P full base tile already on L1
        uint32_t mL0LoopNum = CeilDiv(rowNum, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(embed, L0_TILE_K);

        // while splitting the base tile S to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
            uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (curBaseTileSize - nL0Itr * L0_TILE_N) : L0_TILE_N;
            uint32_t nLoopCounter = GetCurLoopCounter(nL0Itr, nL0LoopNum, nL0Itr);
            // l0C nbuffer chunked only in n loop
            uint32_t l0CLoopCounter = kvSTileIdx;
            uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
            uint32_t l0CEventId = l0CBufId;
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
            auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(embed, l0TileNAct);
            auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
            auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(embed, l0TileNAct));
            if constexpr (PAGED_CACHE_FLAG_) {
                if constexpr (!PA_BNNBSD_FLAG_) {
                    uint32_t blockTableIdx = kvSTileIdx * 128 / blockSize;
                    uint32_t blockOffset = kvSTileIdx * 128 % blockSize;
                    auto blockIdx = gBlockTable.GetValue(blockTableIdx);
                    auto gBTensorTlaTile = GetTile(gBTensor,
                        tla::MakeCoord(0, blockIdx * blockSize + blockOffset), tla::MakeShape(embed, l0TileNAct));
                    copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
                } else {
                    if (isShrink == 1) {
                        uint32_t globalRemain = 0;
                        if (kvSTileIdx == 0){
                            globalRemain = globalWindowSize;
                            uint32_t blockTableIdx = kvSTileIdx * 128 / blockSize;
                            uint32_t blockOffset = kvSTileIdx * 128 % blockSize;
                            auto blockIdx = gBlockTable.GetValue(blockTableIdx);
                            auto gBTensorTlaTile = GetTile(gBTensor,
                                tla::MakeCoord(0, (blockIdx * blockSize) * kvHeads + blockOffset), tla::MakeShape(embed, globalRemain));
                            auto l1BTensorTlaTile0 = GetTile(l1BTensorTlaTile,
                                tla::MakeCoord(0, 0), tla::MakeShape(embed, globalRemain));
                            if constexpr (NZ_LAYOUT_FLAG_) {
                                CopyGmToL1BwithNZ(gBTensorTlaTile, l1BTensorTlaTile0, blockIdx, blockOffset, globalRemain,
                                    embed, blockSize, kvHeads, l0TileNAct);
                            } else {
                                copyGmToL1B(l1BTensorTlaTile0, gBTensorTlaTile);
                            }
                        }
                        uint32_t curKvSeqlenStart = kvSeqlenTriDown - rowNum + 1 - localWindowSize +
                                                    kvSTileIdx * 128 - globalWindowSize * !(kvSTileIdx == 0);
                        uint32_t blockTableIdx = curKvSeqlenStart / blockSize;
                        uint32_t blockOffset = curKvSeqlenStart % blockSize;
                        uint32_t preRemain = (blockSize - blockOffset);
                        preRemain = min(128 - globalRemain, preRemain);
                        preRemain = min(l0TileNAct, preRemain);
                        if (preRemain > 0) {
                            auto blockIdx = gBlockTable.GetValue(blockTableIdx);
                            auto gBTensorTlaTile = GetTile(gBTensor,
                                tla::MakeCoord(0, (blockIdx * blockSize) * kvHeads + blockOffset), tla::MakeShape(embed, preRemain));
                            auto l1BTensorTlaTile1 = GetTile(l1BTensorTlaTile,
                                tla::MakeCoord(0, globalRemain), tla::MakeShape(embed, preRemain));
                            if constexpr (NZ_LAYOUT_FLAG_) {
                                CopyGmToL1BwithNZ(gBTensorTlaTile, l1BTensorTlaTile1, blockIdx, blockOffset, preRemain,
                                    embed, blockSize, kvHeads, l0TileNAct);
                            } else {
                                copyGmToL1B(l1BTensorTlaTile1, gBTensorTlaTile);
                            }
                            
                        }
                        if (preRemain + globalRemain < 128 && preRemain + globalRemain < l0TileNAct) {
                            uint32_t lastRemain = l0TileNAct - preRemain - globalRemain;
                            blockTableIdx += 1;
                            blockOffset = 0;
                            auto blockIdx = gBlockTable.GetValue(blockTableIdx);
                            auto gBTensorTlaTile = GetTile(gBTensor,
                                tla::MakeCoord(0, (blockIdx * blockSize) * kvHeads + blockOffset), tla::MakeShape(embed, lastRemain));
                            auto l1BTensorTlaTile2 = GetTile(l1BTensorTlaTile,
                                tla::MakeCoord(0, preRemain + globalRemain), tla::MakeShape(embed, lastRemain));
                            if constexpr (NZ_LAYOUT_FLAG_) {
                                CopyGmToL1BwithNZ(gBTensorTlaTile, l1BTensorTlaTile2, blockIdx, blockOffset, lastRemain,
                                    embed, blockSize, kvHeads, l0TileNAct);
                            } else {
                                copyGmToL1B(l1BTensorTlaTile2, gBTensorTlaTile);
                            }
                            
                        }
                    } else {
                        uint32_t blockTableIdx = kvSTileIdx * 128 / blockSize;
                        uint32_t blockOffset = kvSTileIdx * 128 % blockSize;
                        auto blockIdx = gBlockTable.GetValue(blockTableIdx);
                        auto gBTensorTlaTile = GetTile(gBTensor,
                            tla::MakeCoord(0, (blockIdx * blockSize) * kvHeads + blockOffset), tla::MakeShape(embed, l0TileNAct));
                        if constexpr (NZ_LAYOUT_FLAG_) {
                            CopyGmToL1BwithNZ(gBTensorTlaTile, l1BTensorTlaTile, blockIdx, blockOffset, l0TileNAct,
                                    embed, blockSize, kvHeads, l0TileNAct);
                        } else {
                            copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
                        }
                    } 
                }
            } else {
                if constexpr (NZ_LAYOUT_FLAG_) {
                    auto gBTensorTlaTile = GetTile(gBTensor,
                        tla::MakeCoord(0, kvSTileIdx * 128), tla::MakeShape(embed, l0TileNAct));

                    const uint32_t blockCount = embed / 16; // headdim / 16
                    const uint32_t blockLen = l0TileNAct;
                    const uint32_t srcStride = kvNumTokens - blockLen;

                    AscendC::DataCopyParams repeatParams;

                    if (srcStride <= STRIDE_LIMIT) {
                        repeatParams.blockCount = blockCount;
                        repeatParams.blockLen = blockLen;
                        repeatParams.srcStride = srcStride; // num tokens - blockLen
                        repeatParams.dstStride = RoundUp(blockLen, 16) - curBaseTileSize;
                        auto dstOffset = 0;
                        auto srcOffset = kvSTileIdx * 128 * 16;

                        AscendC::DataCopy(l1BTensorTlaTile.data()[dstOffset], gBTensorTlaTile.data()[srcOffset], repeatParams);
                    } else {
                        repeatParams.blockCount = 1;
                        repeatParams.blockLen = blockLen;
                        repeatParams.srcStride = 0;
                        repeatParams.dstStride = 0;
                        for (uint32_t i = 0; i < blockCount; ++i) {
                            auto dstOffset = i * RoundUp(blockLen, 16) * 16;
                            auto srcOffset = kvSTileIdx * 128 * 16 + kvNumTokens * i * 16;
                            AscendC::DataCopy(l1BTensorTlaTile.data()[dstOffset], gBTensorTlaTile.data()[srcOffset], repeatParams);
                        }
                    }
                } else {
                    auto gBTensorTlaTile = GetTile(gBTensor,
                        tla::MakeCoord(0, kvSTileIdx * 128), tla::MakeShape(embed, l0TileNAct));
                    copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                // different m chunks will be concated in the same piece of l0C buffer
                auto l0CTensorTlaTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0LoopNum, kL0Itr);
 	                uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(nLoopCounter, kL0LoopNum, kL0Itr);
                    uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                        (embed - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;
                    // when L0B buffers wouldn't be reused across the k loop
                    // redundant L0B load caused by m loop can be avoided
                    auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, 0), tla::MakeShape(l0TileKAct, l0TileNAct));
                    auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                    
                    auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K), tla::MakeShape(l0TileMAct, l0TileKAct));
                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    if ((mL0Itr == 0)  && (kL0Itr == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
                    }
                    copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if ((mL0Itr == mL0LoopNum - 1)  && (kL0Itr == kL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                    }

                    bool initMmad = (kL0Itr == 0);
                    uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(
                        l0CTensorTlaTile,
                        l0ATensorTla,
                        l0BTensorTla,
                        l0TileMAligned,
                        l0TileNAct,
                        l0TileKAct,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }
            // fixpipe
            if (nL0Itr == 0) {
                // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
                WaitCrossCoreSync<4, PIPE_FIX>(qkReadyFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            // 需要kernel传输ubCTensor的时候确保其shape的m，n是满足32B（8个32位元素）对齐的
            // rounded up by 8 and splited in half to each AIV
            // valid rows in AIV0: [0, mFixPAligned8 / 2 - 1]
            // valid rows in AIV1: [mFixPAligned8 / 2, rowNum - 1]
            if constexpr (std::is_same_v<ElementC, half>) {
                uint32_t mFixPAligned8 = RoundUp(rowNum, 8);
                uint32_t mPerSubCore = mFixPAligned8 / 2;
                uint32_t nFixPAligned16 = RoundUp(l0TileNAct, 16);
                auto ubCTensorTlaTile = GetTile(ubCTensor,
                    tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mPerSubCore, nFixPAligned16));
                auto l0CTensorTlaTileSub0 = GetTile(l0CTensorTla,
                        tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, l0TileNAct));
                auto l0CTensorTlaTileSub1 = GetTile(l0CTensorTla,
                        tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, l0TileNAct));
                copyL0CToDstSub0(ubCTensorTlaTile, l0CTensorTlaTileSub0, false, 0);
                copyL0CToDstSub1(ubCTensorTlaTile, l0CTensorTlaTileSub1, true, 0);
            }   else {
                uint32_t mFixPAligned8 = RoundUp(rowNum, 8);
                uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);

                auto ubCTensorTlaTile = GetTile(ubCTensor,
                    tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mFixPAligned8, nFixPAligned8));
                                    copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(qkReadyFlag);
    }

protected:
    // Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};
////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Block
#endif  // GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_HPP