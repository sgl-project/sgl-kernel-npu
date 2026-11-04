/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should have received a copy of the License with this source code repository.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file block_mmad_bf16_moe.h
 * \brief mega_moe 私有 BF16 BlockMmad (no-quant GMM1)
 *
 * BlockMmadMxMoe (block_mmad_moe.h) 的 BF16 对应版本，供 dispatch_quant_mode=0
 * 的 GMM1 使用。与 MX 版本的差异与 dispatch_ffn_combine -> dispatch_ffn_combine_bf16
 * 的适配方式一致：
 *   1. 去掉 MX scale 的 GM->L1/L1->L0 搬运，MMAD 使用默认 trait (BF16 x BF16)。
 *   2. BF16 元素是 FP8 的两倍宽：L1 仍放 2 个 256x256 A/B buffer
 *      (A=B=128KiB, 每半区恰为 256KiB)，L0 的 K tile 由 128 缩到 64
 *      (256x64x2B=32KiB, 与 L0A/L0B 半区一致)。
 *   3. K 方向无需 64 对齐补零。
 *   4. 保留 Compute()/CopyOut() 分级接口与 K 方向 issue/consume 流水，
 *      接口签名与 BlockMmadMxMoe 相同 (scale/bias 形参保留但不使用)，
 *      GroupMatmulSwigluQuant 的调用点无需区分 dtype。
 *   5. 不支持 resident-A prefix：BF16 下 L1 两个半区已被 A/B 占满。
 */

#ifndef BLOCK_MMAD_BF16_MOE_H
#define BLOCK_MMAD_BF16_MOE_H

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"
#include "block_mmad_moe.h"

namespace Blaze {
namespace Gemm {
namespace Block {
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_>
class BlockMmadBf16Moe
{
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;

    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t L1_BUFFER_NUM = 2;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>>;

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curKL1 = 0;
    };

    // Same field layout as BlockMmadMxMoe::L1Params so Gmm1Context can build
    // either block with one designated initializer; scaleKL1 is unused here.
    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmadBf16Moe()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true);  // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmadBf16Moe()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false);  // false means row first when fixpipe_l0c2out
    }

    // Resident-A prefix does not fit beside two BF16 A/B buffers; the
    // template argument is accepted for interface parity and ignored.
    template <bool EnableResidentPrefix = false>
    __aicore__ inline void Init(const ProblemShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        (void)l1Params.scaleKL1;
        (void)l1Params.l1BufNum;
        splitKNum_ = splitKNum;
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        enableL0cPingPong_ = dbL0C;
        uint64_t aL1OneBuffer = baseM_ * kL1_ * sizeof(AType);
        for (int32_t bufferId = 0; bufferId < L1_BUFFER_NUM; bufferId++) {
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset;
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer;
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA>
    __aicore__ inline void IssueAInL1(TensorA const &gmA, TileL1L0Param &tileL1L0Param, uint64_t l1BufId,
                                      uint64_t iter0)
    {
        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, iter0 * kL1_),
                                  AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curKL1));
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    }

    template <typename TensorB>
    __aicore__ inline void IssueBInL1(TensorB const &gmB, TileL1L0Param &tileL1L0Param, uint64_t l1BufId,
                                      uint64_t iter0)
    {
        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curKL1, tileL1L0Param.curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(iter0 * kL1_, 0),
                                  AscendC::Te::MakeShape(tileL1L0Param.curKL1, tileL1L0Param.curN));
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
    }

    __aicore__ inline void SetKTileParam(TileL1L0Param &tileL1L0Param, uint64_t iter0)
    {
        tileL1L0Param.curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void IssueKTile(TensorA const &gmA, TensorB const &gmB, TileL1L0Param &tileL1L0Param,
                                      uint64_t iter0, uint64_t l1BufId)
    {
        SetKTileParam(tileL1L0Param, iter0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        IssueAInL1(gmA, tileL1L0Param, l1BufId, iter0);
        IssueBInL1(gmB, tileL1L0Param, l1BufId, iter0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    }

    template <typename TensorL0C>
    __aicore__ inline void ConsumeKTile(TileL1L0Param &tileL1L0Param, TensorL0C &tensorL0C, uint64_t iter0,
                                        uint64_t l1BufId, uint64_t splitKIdx)
    {
        SetKTileParam(tileL1L0Param, iter0);
        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curKL1, tileL1L0Param.curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        uint64_t kL0Iter = CeilDiv(tileL1L0Param.curKL1, baseK_);
        for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curKL0 = (iter1 + 1 == kL0Iter) ? (tileL1L0Param.curKL1 - iter1 * baseK_) : baseK_;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            uint16_t mte1WaitMFlag = static_cast<uint16_t>((l0PingPong_ & 0x1) + M_MTE1_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            // A L1->L0
            auto layoutAL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(
                    tileL1L0Param.curM, curKL0);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, iter1 * baseK_),
                                                  AscendC::Te::MakeShape(tileL1L0Param.curM, curKL0));
            auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
            AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

            // B L1->L0
            auto layoutBL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(
                    curKL0, tileL1L0Param.curN);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(iter1 * baseK_, 0),
                                                  AscendC::Te::MakeShape(curKL0, tileL1L0Param.curN));
            auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
            AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            AscendC::Te::MmadParams params;
            params.m = static_cast<uint16_t>(tileL1L0Param.curM);
            params.k = static_cast<uint16_t>(curKL0);
            params.n = static_cast<uint16_t>(tileL1L0Param.curN);
            params.unitFlag = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKIdx == splitKNum_ - 1))
                                  ? FINAL_ACCUMULATION
                                  : NON_FINAL_ACCUMULATION;
            params.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && splitKIdx == 0 && !isBias_);
            AscendC::Te::Mmad(AscendC::Te::MmadAtom<
                                  AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitDefault>>{}
                                  .with(params),
                              tensorL0C, tensorAL0, tensorBL0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    }

    // 分级接口第一步: GM->L1 load + MMAD (K循环), 不写输出。
    // 流水顺序与 BlockMmadMxMoe 一致: issue(0), issue(1), consume(0), issue(2), consume(1), ...
    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias>
    __aicore__ inline auto Compute(TensorA const &gmA, TensorB const &gmB, TensorScaleA const &gmScaleA,
                                   TensorScaleB const &gmScaleB, TensorBias const &gmBias,
                                   BlockShape const &singleShape, uint64_t splitKIdx = 0)
    {
        (void)gmScaleA;
        (void)gmScaleB;
        (void)gmBias;
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);
        uint64_t abBase = abL1LoopCnt_;
        if (kL1Iter_ != 0) {
            IssueKTile(gmA, gmB, tileL1L0Param, 0, abBase & 1);
            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                if (iter0 + 1 < kL1Iter_) {
                    uint64_t nextIter = iter0 + 1;
                    IssueKTile(gmA, gmB, tileL1L0Param, nextIter, (abBase + nextIter) & 1);
                }
                ConsumeKTile(tileL1L0Param, tensorL0C, iter0, (abBase + iter0) & 1, splitKIdx);
            }
        }
        abL1LoopCnt_ = abBase + kL1Iter_;
        return tensorL0C;
    }

    template <bool EnableResidentPrefix, typename TensorA, typename TensorB, typename TensorScaleA,
              typename TensorScaleB, typename TensorBias>
    __aicore__ inline auto ComputeMaybeWithResidentAPrefix(TensorA const &gmA, TensorB const &gmB,
                                                           TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                                           TensorBias const &gmBias, BlockShape const &singleShape,
                                                           bool reuseResident, uint64_t splitKIdx = 0)
    {
        (void)reuseResident;
        return Compute(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, splitKIdx);
    }

    // 分级接口第二步: L0C->UB/GM fixpipe 写出 (与 BlockMmadMxMoe::CopyOut 相同)。
    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyOut(TensorC const &gmC, TensorL0C &tensorL0C, uint64_t splitKIdx = 0)
    {
        if (splitKIdx == splitKNum_ - 1) {
            if constexpr (AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>) {
                auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
                AscendC::Te::Copy(CopyL0C2UB.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            } else {
                auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                AscendC::Te::Copy(CopyL0C2GM.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            }
            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    constexpr static uint16_t M_MTE1_FLAG_0 = 4;
    constexpr static uint16_t M_MTE1_FLAG_1 = 5;

    uint64_t k_{0};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
};

// MegaMoe instantiates BlockMmadMxMoe from its dispatch element type. BF16
// activations/weights (dispatch_quant_mode=0) route to the scale-free block,
// so the GMM1 context and GroupMatmulSwigluQuant stay dtype-agnostic.
template <class DispatchPolicy_, class LayoutA_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_>
class BlockMmadMxMoe<DispatchPolicy_, bfloat16_t, LayoutA_, bfloat16_t, LayoutB_, CType_, LayoutC_, BiasType_,
                     LayoutBias_>
    : public BlockMmadBf16Moe<bfloat16_t, LayoutA_, bfloat16_t, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_>
{};
#endif
}  // namespace Block
}  // namespace Gemm
}  // namespace Blaze

#endif  // BLOCK_MMAD_BF16_MOE_H
