/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ASCEND950
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ASCEND950

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Block {

template <class L1TileShape_, class OType_, class OTmpType_>
class BlockEpilogue<EpilogueAscend950FARescaleO, L1TileShape_, OType_, OTmpType_>
{
public:
    using DispatchPolicy = EpilogueAscend950FARescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementO = typename OType_::Element;
    using ElementOTmp = typename OTmpType_::Element;
    using LayoutTagO = typename OType_::Layout;
    using LayoutTagOTmp = typename OTmpType_::Layout;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t S1_BASE_SIZE = tla::get<0>(L1TileShape{});
    static constexpr uint32_t D_BASE_SIZE = tla::get<2>(L1TileShape{});
    static constexpr uint32_t VEC2_UB_SIZE = S1_BASE_SIZE / 2 * D_BASE_SIZE * sizeof(ElementOTmp);

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t &ubBufAddrStart)
    {
        vf2OutUb = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubBufAddrStart);
        ubBufAddrStart += VEC2_UB_SIZE;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst &attenOutGm, const AscendC::LocalTensor<ElementOTmp> &expMaxUb,
                                   const AscendC::LocalTensor<ElementOTmp> &sumUb, TensorSrc &bmm2Res, bool isFirstLoop,
                                   bool isLastUpdate, uint64_t MM2_RES_INTRA_EVENT)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
        uint32_t m = tla::get<0>(bmm2Res.shape());
        uint32_t n = tla::get<1>(bmm2Res.shape());
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementOTmp));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;

        __ubuf__ float *vec2ResUbAddr = (__ubuf__ ElementOTmp *)vf2OutUb.GetPhyAddr();
        __ubuf__ float *bmm2UbAddr = (__ubuf__ ElementOTmp *)bmm2Res.data().GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ ElementOTmp *)expMaxUb.GetPhyAddr();
        __ubuf__ float *sumUbAddr = (__ubuf__ ElementOTmp *)sumUb.GetPhyAddr();

        if (isFirstLoop) {
            DataCopy(vf2OutUb, bmm2Res.data(), m * n);
        } else if (!isLastUpdate) {
            FlashUpdateNew<ElementOTmp, D_BASE_SIZE>(vec2ResUbAddr, bmm2UbAddr, expMaxUbAddr, m, nLoops, tailN);
        } else {
            FlashUpdateLastNew<ElementOTmp, D_BASE_SIZE>(vec2ResUbAddr, bmm2UbAddr, expMaxUbAddr, sumUbAddr, m, nLoops,
                                                         tailN);
        }
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_RES_INTRA_EVENT);

        if (isFirstLoop && isLastUpdate) {
            LastDivNew<ElementOTmp, D_BASE_SIZE>(vec2ResUbAddr, bmm2UbAddr, sumUbAddr, m, nLoops, tailN);
        }
        if (isLastUpdate) {
            AscendC::LocalTensor<ElementO> attenOut;
            attenOut.SetAddr(vf2OutUb.address_);
            AscendC::Cast(attenOut, vf2OutUb, AscendC::RoundMode::CAST_ROUND, m * D_BASE_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOVMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOVMTE3);
            auto layoutUb = tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(D_BASE_SIZE, tla::Int<1>{}));
            auto attenOutUb = tla::MakeTensor(attenOut, layoutUb, Arch::PositionUB{});
            using CopyUbToGmO = Tile::CopyUb2GmTla<ArchTag, decltype(attenOutUb), TensorDst>;
            CopyUbToGmO copyUbToGmO;
            copyUbToGmO(attenOutGm, attenOutUb);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
    }

private:
    AscendC::LocalTensor<ElementOTmp> vf2OutUb;
    static constexpr int32_t SYNC_MODE = 4;
    static constexpr uint16_t FLOAT_REP_SIZE = 64;
    static constexpr int32_t eventOVMTE3 = 3;
    static constexpr int32_t eventOMTE3V = 3;

    template <class T, uint16_t DBaseSize>
    __simd_vf__ inline void FlashUpdateNew(__ubuf__ T *updateUb, __ubuf__ T *curUb, __ubuf__ T *expMaxUb, uint16_t m,
                                           uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> expMaxVreg;
        RegTensor<float> preSrcVreg;
        RegTensor<float> curSrcVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> addVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(expMaxVreg, expMaxUb + i);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(preSrcVreg, updateUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                LoadAlign(curSrcVreg, curUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                Mul(mulVreg, expMaxVreg, preSrcVreg, pregFull);
                Add(addVreg, mulVreg, curSrcVreg, pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(updateUb + i * DBaseSize + j * FLOAT_REP_SIZE, addVreg,
                                                        pregFull);
            }
            LoadAlign(preSrcVreg, updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            LoadAlign(curSrcVreg, curUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            Mul(mulVreg, expMaxVreg, preSrcVreg, pregTailN);
            Add(addVreg, mulVreg, curSrcVreg, pregTailN);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, addVreg,
                                                    pregTailN);
        }
    }
    template <class T, uint16_t DBaseSize>
    __simd_vf__ inline void FlashUpdateLastNew(__ubuf__ T *updateUb, __ubuf__ T *curUb, __ubuf__ T *expMaxUb,
                                               __ubuf__ T *expSumUb, uint16_t m, uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> expMaxVreg;
        RegTensor<float> preSrcVreg;
        RegTensor<float> curSrcVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> addVreg;
        RegTensor<float> divDstVreg;
        RegTensor<float> expSumVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(expMaxVreg, expMaxUb + i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(expSumVreg, expSumUb + i);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(preSrcVreg, updateUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                LoadAlign(curSrcVreg, curUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                Mul(mulVreg, expMaxVreg, preSrcVreg, pregFull);
                Add(addVreg, mulVreg, curSrcVreg, pregFull);
                Div(divDstVreg, addVreg, expSumVreg, pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(updateUb + i * DBaseSize + j * FLOAT_REP_SIZE, divDstVreg,
                                                        pregFull);
            }
            LoadAlign(preSrcVreg, updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            LoadAlign(curSrcVreg, curUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            Mul(mulVreg, expMaxVreg, preSrcVreg, pregTailN);
            Add(addVreg, mulVreg, curSrcVreg, pregTailN);
            Div(divDstVreg, addVreg, expSumVreg, pregTailN);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, divDstVreg,
                                                    pregTailN);
        }
    }
    template <class T, uint16_t DBaseSize>
    __simd_vf__ inline void LastDivNew(__ubuf__ T *updateUb, __ubuf__ T *curUb, __ubuf__ T *expSumUb, uint16_t m,
                                       uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> curSrcVreg;
        RegTensor<float> divDstVreg;
        RegTensor<float> expSumVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(expSumVreg, expSumUb + i);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(curSrcVreg, curUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                Div(divDstVreg, curSrcVreg, expSumVreg, pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(updateUb + i * DBaseSize + j * FLOAT_REP_SIZE, divDstVreg,
                                                        pregFull);
            }
            LoadAlign(curSrcVreg, curUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            Div(divDstVreg, curSrcVreg, expSumVreg, pregTailN);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, divDstVreg,
                                                    pregTailN);
        }
    }
};
}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ASCEND950
