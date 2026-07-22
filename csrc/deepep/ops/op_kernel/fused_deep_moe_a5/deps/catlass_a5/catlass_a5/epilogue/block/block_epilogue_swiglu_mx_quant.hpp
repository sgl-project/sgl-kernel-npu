/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SWIGLU_QUANT_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SWIGLU_QUANT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/epilogue/tile/tile_swiglu_mx_quant.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Epilogue::Block {

template <class L0TileShape_, class DataTypeAct_, class DataTypeGate_, class DataTypeGluRes_, class DataTypeQuantOut_,
          class DataTypeQuantScale_>
class BlockEpilogue<BlockEpilogueSwigluMxQuant, L0TileShape_, DataTypeAct_, DataTypeGate_, DataTypeGluRes_,
                    DataTypeQuantOut_, DataTypeQuantScale_>
{
public:
    using DispatchPolicy = BlockEpilogueSwigluMxQuant;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ActType = DataTypeAct_;
    using GateType = DataTypeGate_;
    using GluResType = DataTypeGluRes_;
    using QuantOutType = DataTypeQuantOut_;
    using QuantScaleType = DataTypeQuantScale_;
    using CalcType = float;

    constexpr static uint32_t MX_BLOCK_SIZE = 32;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint8_t AIC_SYNC_AIV_MODE = 4;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    ;
    constexpr static uint32_t UB_ALIGN_SIZE = 32;
    constexpr static uint32_t UB_SUB_BANK_NUM = 2;
    constexpr static uint32_t UB_TWO_BANK_ELEMS_B32 = 128;
    constexpr static uint32_t QBMM_BUFFER_NUM = 2;
    constexpr static int64_t PER_BLOCK_SIZE = 128LL;
    constexpr static uint32_t N_HALF = 2;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});

    constexpr static uint16_t EMAX = std::is_same_v<QuantOutType, float8_e4m3_t>   ? 0x0400
                                     : std::is_same_v<QuantOutType, float8_e5m2_t> ? 0x0780
                                                                                   : 0;

    struct Arguments {
        GM_ADDR quantOutGmAddr{nullptr};
        GM_ADDR quantScaleGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
    };

    using Params = Arguments;

    CATLASS_DEVICE BlockEpilogue() {}

    CATLASS_DEVICE ~BlockEpilogue() {}

    CATLASS_DEVICE void Init(const Params *params)
    {
        if ASCEND_IS_AIC {
            return;
        }

        params_ = params;
        subBlockIdx_ = AscendC::GetSubBlockIdx();

        constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        constexpr uint32_t scaleElems = elems / MX_BLOCK_SIZE;
        constexpr uint32_t mmResSize = elems * UB_SUB_BANK_NUM * sizeof(ActType);
        constexpr uint32_t scaleBlockBufBytes = L0_TILE_M * UB_ALIGN_SIZE;

        uint32_t offset = mmResSize;

        quantOutputUb_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, offset, elems);
        offset += elems * sizeof(int8_t);

        quantScaleOutputUb_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, offset, scaleElems);
        offset += scaleElems * sizeof(int8_t);

        gluResUb_ = AscendC::LocalTensor<GluResType>(AscendC::TPosition::VECCALC, offset, elems);
        offset += elems * sizeof(GluResType);

        maxExpUb_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, offset, scaleElems);
        offset += scaleElems * sizeof(uint16_t);

        halfScaleUb_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, offset, scaleElems);
        offset += scaleElems * sizeof(uint16_t);

        quantScaleBlockOutputUb_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, offset, scaleBlockBufBytes);
    }

    CATLASS_DEVICE void UpdateGlobalAddr(AscendC::GlobalTensor<QuantOutType> &gmQ,
                                         AscendC::GlobalTensor<QuantScaleType> &gmQScale)
    {
        if ASCEND_IS_AIV {
            qGlobal_ = gmQ.template ReinterpretCast<int8_t>();
            qScaleGlobal_ = gmQScale.template ReinterpretCast<int8_t>();
        }
    }

    template <class TensorUbPing, class TensorUbPong>
    CATLASS_DEVICE void operator()(const GemmCoord &resShape, int64_t totalMOffset, const GemmCoord &blockCoord,
                                   const TensorUbPing &mmResPing, const TensorUbPong &mmResPong, uint32_t l1TileM,
                                   uint32_t l1TileN)
    {
        uint32_t singleM = resShape.m();
        uint32_t singleN = resShape.n();

        uint32_t halfSingleM = CeilDiv(singleM, AscendC::GetTaskRation());
        uint32_t singleMInVec = subBlockIdx_ == 1 ? singleM - halfSingleM : halfSingleM;
        if (singleMInVec == 0) {
            return;
        }
        uint32_t mOffset = subBlockIdx_ * halfSingleM;

        __ubuf__ ActType *actData = reinterpret_cast<__ubuf__ ActType *>(mmResPing.GetPhyAddr());
        __ubuf__ GateType *gateData = reinterpret_cast<__ubuf__ GateType *>(mmResPong.GetPhyAddr());

        auto gluResAddr = reinterpret_cast<__ubuf__ GluResType *>(gluResUb_.GetPhyAddr());
        auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpUb_.GetPhyAddr());
        auto halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleUb_.GetPhyAddr());
        auto quantOutAddr = reinterpret_cast<__ubuf__ int8_t *>(quantOutputUb_.GetPhyAddr());
        auto quantScaleAddr = reinterpret_cast<__ubuf__ QuantScaleType *>(quantScaleOutputUb_.GetPhyAddr());

        Tile::TileSwigluAndMxQuant<ArchTag, ActType, GateType, CalcType, GluResType, QuantOutType, QuantScaleType, EMAX>
            swigluQuantTile;

        swigluQuantTile(quantOutAddr, quantScaleAddr, gluResAddr, maxExpAddr, halfScaleAddr, actData, gateData,
                        static_cast<uint16_t>(singleMInVec), singleN, singleN);

        TransMxScaleLayout(quantScaleOutputUb_, quantScaleBlockOutputUb_, singleMInVec, singleN);

        int64_t yOffset =
            (totalMOffset + mOffset + blockCoord.m() * l1TileM) * params_->baseN + blockCoord.n() * l1TileN;
        uint32_t baseNHalf = params_->baseN;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        CopyQuantOutputFromUb2Gm(singleMInVec, yOffset, singleN, baseNHalf);
        CopyQuantScaleFromUb2Gm(singleMInVec, yOffset, singleN, baseNHalf);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

private:
    CATLASS_DEVICE void TransMxScaleLayout(AscendC::LocalTensor<int8_t> &srcUb, AscendC::LocalTensor<int8_t> &dstUb,
                                           uint16_t mSize, uint32_t nSize)
    {
        uint32_t blockScaleN = CeilDiv(nSize, MX_BLOCK_SIZE);
        __ubuf__ int8_t *srcAddr = (__ubuf__ int8_t *)srcUb.GetPhyAddr();
        __ubuf__ int8_t *dstAddr = (__ubuf__ int8_t *)dstUb.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg maskScaleN;
            for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
                uint32_t elemNum = blockScaleN;
                maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
                AscendC::MicroAPI::RegTensor<int8_t> vreg0;
                AscendC::MicroAPI::UnalignReg u0;
                __ubuf__ int8_t *rowSrc = srcAddr + mIdx * blockScaleN;
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, rowSrc);
                AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, rowSrc);
                __ubuf__ int8_t *dstUb = dstAddr + mIdx * UB_ALIGN_SIZE;
                AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0,
                                                                                                maskScaleN);
            }
        }
    }

    CATLASS_DEVICE void CopyQuantOutputFromUb2Gm(uint16_t singleMInVec, int64_t yOffset, uint32_t singleN,
                                                 uint32_t baseNHalf)
    {
        uint32_t blockLen = singleN * sizeof(int8_t);
        uint32_t dstStride = (baseNHalf - singleN) * sizeof(int8_t);

        AscendC::DataCopyExtParams copyParams{singleMInVec, blockLen, 0, dstStride, 0};
        AscendC::DataCopyPad(qGlobal_[yOffset], quantOutputUb_, copyParams);
    }

    CATLASS_DEVICE void CopyQuantScaleFromUb2Gm(uint16_t singleMInVec, int64_t yOffset, uint32_t singleN,
                                                uint32_t baseNHalf)
    {
        uint32_t blockScaleN = CeilDiv(singleN, MX_BLOCK_SIZE);
        uint32_t blockLen = blockScaleN * sizeof(int8_t);
        uint32_t scaleN = CeilDiv(baseNHalf, MX_BLOCK_SIZE);
        uint32_t dstStride = (scaleN - blockScaleN) * sizeof(int8_t);

        int64_t yScaleOffset = yOffset / MX_BLOCK_SIZE;

        AscendC::DataCopyExtParams copyParams{singleMInVec, blockLen, 0, dstStride, 0};
        AscendC::DataCopyPad(qScaleGlobal_[yScaleOffset], quantScaleBlockOutputUb_, copyParams);
    }

    const Params *params_{nullptr};
    uint32_t subBlockIdx_{0};

    AscendC::GlobalTensor<int8_t> qGlobal_;
    AscendC::GlobalTensor<int8_t> qScaleGlobal_;

    AscendC::LocalTensor<GluResType> gluResUb_;
    AscendC::LocalTensor<uint16_t> maxExpUb_;
    AscendC::LocalTensor<uint16_t> halfScaleUb_;
    AscendC::LocalTensor<int8_t> quantOutputUb_;
    AscendC::LocalTensor<int8_t> quantScaleOutputUb_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutputUb_;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SWIGLU_QUANT_HPP
