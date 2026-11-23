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
 * \file block_mmad_moe.h
 * \brief mega_moe 私有 BlockMmad (MX量化矩阵乘分级实现)
 *
 * Fork 来源: ops-tensor b24535e30a6f7a3909c53c8b09713a9c0d08e54b 的
 *   include/blaze/gemm/block/block_mmad_qbmm_mx.h (类 BlockMmad 的
 *   MatmulWithScaleMx 特化), 类名改为 BlockMmadMxMoe。
 *
 * 本类从上述固定版本派生，并仅作 MegaMoe GMM1 所需的专用化：
 *   1. 拆出 Compute()/CopyOut() 分级接口: Compute = GM->L1 load + MMAD(K循环),
 *      不碰输出; CopyOut = L0C->UB/GM fixpipe。调用侧可在两者之间插入跨核同步
 *      (GMM1 GM中转的ready投票), 同步语义留在 mega_moe 调用点, 本类保持纯 GEMM 组件。
 *   2. Compute 按值返回 L0C tensor, CopyOut 按参接收。刻意不引入 tensorL0C_
 *      成员: opt0820 的成员化实现在 fast 用例实测 +25.7% 劣化, 根因是 K 循环内
 *      对成员的 store/reload 破坏了上游寄存器驻留的 codegen。
 *   3. K 方向使用显式 prologue/steady-state/drain 流水: 下一 K tile 的
 *      GM->L1 MTE2 在当前 tile 的 L1->L0/MMAD 之前发射, 从而让两条流水重叠。
 *      每次 Compute 都完整 drain, 不把悬空状态带过 SwiGLU 两个 weight half。
 *   4. 将 L1 布局固定为 MegaMoe GMM1 的 2 个 A/B buffer，每 2 个
 *      K tile 共用一组 scale；不保留上游其他 full-load/缓冲数路径。
 *   5. operator() 保留并等价于 Compute()+CopyOut() 串联, 原语义完全不变。
 *
 * 本类只服务 MegaMoe GMM1 的固定 L1 配置: 2 个 A/B buffer, 每 2 个 K tile
 * 共用一组 scale。它不是上游 BlockMmad 的通用替代品。
 *
 * 注意: 本文件刻意保持在上游同名空间 Blaze::Gemm::Block 内, 保证 body 中
 * 裸引用的常量/函数 (CeilDiv、MXFP_*、INPUT_BUFFER_FLAG_* 等) 名字解析与上游一致。
 * 升级 ops-tensor 版本时需重新对照上游做人工合并。
 */

#ifndef BLOCK_MMAD_MOE_H
#define BLOCK_MMAD_MOE_H

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"

namespace Blaze {
namespace Gemm {
namespace Block {
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <class DispatchPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmadMxMoe
{
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    static_assert(DispatchPolicy::fullLoadMode == 0, "BlockMmadMxMoe K pipeline only supports non-full-load GMM1");
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t k_;
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;

    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    constexpr static uint64_t SCALE_C0 = 2;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;
    constexpr static uint64_t L1_BUFFER_NUM = 2;
    constexpr static uint64_t K_TILES_PER_SCALE = 2;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;

    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmAKL1 = 0;
        uint64_t curGmBKL1 = 0;
        uint64_t curPadAKL1 = 0;  // pad to 64 align
        uint64_t curPadBKL1 = 0;  // pad to 64 align
    };

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    template <typename TSA, typename TSB, typename TA, typename TB, typename TBias>
    struct L1InputBundle {
        TSA scaleA;
        TSB scaleB;
        TA a;
        TB b;
        TBias bias;
    };

    __aicore__ inline BlockMmadMxMoe()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true);  // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmadMxMoe()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false);  // false means row first when fixpipe_l0c2out
    }

public:
    template <bool EnableResidentPrefix = false>
    __aicore__ inline void Init(const ProblemShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        (void)l1Params.scaleKL1;
        (void)l1Params.l1BufNum;
        scaleKL1_ = kL1_ * K_TILES_PER_SCALE;
        splitKNum_ = splitKNum;
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        enableL0cPingPong_ = dbL0C;
        residentPrefixEnabled_ = EnableResidentPrefix;
        residentSavedColumns_ = 0U;
        residentMinSavedBytes_ = 0U;
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        uint64_t bL1OneBuffer = (baseN_ * kL1_) >> sizeShift;
        uint64_t scaleBL1OneBuffer = baseN_ * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        uint64_t aL1OneBuffer = (baseM_ * Blaze::Gemm::CeilAlign(kL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        uint64_t scaleAL1OneBuffer = baseM_ * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        for (int32_t bufferId = 0; bufferId < L1_BUFFER_NUM; bufferId++) {
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset;
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer;
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer;

            // With bias disabled, each 256-K ping/pong partition leaves one
            // 256x256 FP8 payload (64 KiB) after the normal A/B/MX-scale
            // windows.  Keep one A strip in each half; MX scale A continues to
            // use the ordinary two-K scale window.  Bias would overlap this
            // tail, so it always selects the original pipeline.
            if constexpr (EnableResidentPrefix) {
                residentAOffset_[bufferId] = l1BufferBiasOffset_[bufferId];
                uint64_t partitionEnd = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId + 1U);
                if (isBias_ || residentAOffset_[bufferId] + aL1OneBuffer > partitionEnd) {
                    residentPrefixEnabled_ = false;
                }
            }
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
        if constexpr (EnableResidentPrefix) {
            // Keep every multiply in the runtime benefit gate at uint32.
            // These bounds are far above the production 256xK tile, but make
            // the device compiler independent of a 128-bit integer runtime.
            constexpr uint64_t MAX_UINT32_VALUE = 0xFFFFFFFFULL;
            if (k_ > MAX_UINT32_VALUE || kL1_ > MAX_UINT32_VALUE / L1_BUFFER_NUM || baseM_ > MAX_UINT32_VALUE ||
                aL1OneBuffer > MAX_UINT32_VALUE) {
                residentPrefixEnabled_ = false;
            } else {
                uint32_t maxResidentColumns = static_cast<uint32_t>(kL1_) * static_cast<uint32_t>(L1_BUFFER_NUM);
                uint32_t kColumns = static_cast<uint32_t>(k_);
                residentSavedColumns_ = kColumns < maxResidentColumns ? kColumns : maxResidentColumns;
                uint32_t baseM = static_cast<uint32_t>(baseM_);
                if (residentSavedColumns_ == 0U || baseM == 0U || baseM > 0xFFFFFFFFU / residentSavedColumns_) {
                    residentPrefixEnabled_ = false;
                } else {
                    residentMinSavedBytes_ = static_cast<uint32_t>(aL1OneBuffer);
                }
            }
        }
    }

    __aicore__ inline bool CanReuseActivationPrefix(BlockShape const &singleShape) const
    {
        if (!residentPrefixEnabled_ || kL1Iter_ == 0U) {
            return false;
        }
        uint64_t actualMValue = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        if (actualMValue > baseM_ || actualMValue > 0xFFFFFFFFULL) {
            return false;
        }
        uint32_t actualM = static_cast<uint32_t>(actualMValue);
        uint32_t savedBytes = actualM * residentSavedColumns_;
        if constexpr (IsFp4<AType>()) {
            savedBytes >>= 1U;
        }
        return savedBytes >= residentMinSavedBytes_;
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void IssueScalesInL1(TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                           TileL1L0Param &tileL1L0Param, uint64_t scaleL1BufId, uint64_t iter0)
    {
        if ((iter0 & 1) != 0) {
            return;
        }
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            tileL1L0Param.curM, Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        uint64_t kL1Offset = iter0 * kL1_;
        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        auto gmBlockScaleA = gmScaleA.Slice(
            AscendC::Te::MakeCoord(0, iter0 * Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
            AscendC::Te::MakeShape(tileL1L0Param.curM,
                                   Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

        auto gmBlockScaleB = gmScaleB.Slice(
            AscendC::Te::MakeCoord(iter0 * Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
            AscendC::Te::MakeShape(Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                                   tileL1L0Param.curN));
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
    }

    template <typename TensorA>
    __aicore__ inline void IssueAInL1(TensorA const &gmA, TileL1L0Param &tileL1L0Param, uint64_t l1BufId,
                                      uint64_t iter0)
    {
        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadAKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, iter0 * kL1_),
                                  AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
        Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    }

    template <typename TensorB>
    __aicore__ inline void IssueBInL1(TensorB const &gmB, TileL1L0Param &tileL1L0Param, uint64_t l1BufId,
                                      uint64_t iter0)
    {
        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadBKL1, tileL1L0Param.curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(iter0 * kL1_, 0),
                                  AscendC::Te::MakeShape(tileL1L0Param.curGmBKL1, tileL1L0Param.curN));
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
    }

    template <typename TensorA>
    __aicore__ inline void IssueResidentAInL1(TensorA const &gmA, TileL1L0Param &tileL1L0Param, uint64_t residentSlot,
                                              uint64_t iter0)
    {
        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadAKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(residentAOffset_[residentSlot]), layoutAL1);
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, iter0 * kL1_),
                                  AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    }

    __aicore__ inline void SetKTileParam(TileL1L0Param &tileL1L0Param, uint64_t iter0)
    {
        tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
        tileL1L0Param.curPadBKL1 = Blaze::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE);
        tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
        tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1;
    }

    __aicore__ inline auto MakeL1Views(TileL1L0Param &tileL1L0Param, uint64_t l1BufId, uint64_t scaleL1BufId)
    {
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            tileL1L0Param.curM, Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);
        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadAKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);

        auto layoutBiasL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>{}(
            1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(l1BufferBiasOffset_[scaleL1BufId]),
            layoutBiasL1);
        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadBKL1, tileL1L0Param.curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
        return L1InputBundle<decltype(tensorScaleAL1), decltype(tensorScaleBL1), decltype(tensorAL1),
                             decltype(tensorBL1), decltype(tensorBiasL1)>{tensorScaleAL1, tensorScaleBL1, tensorAL1,
                                                                          tensorBL1, tensorBiasL1};
    }

    __aicore__ inline auto MakeResidentAL1Views(TileL1L0Param &tileL1L0Param, uint64_t residentSlot, uint64_t l1BufId,
                                                uint64_t scaleL1BufId)
    {
        auto inputs = MakeL1Views(tileL1L0Param, l1BufId, scaleL1BufId);
        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadAKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(residentAOffset_[residentSlot]), layoutAL1);
        return L1InputBundle<decltype(inputs.scaleA), decltype(inputs.scaleB), decltype(tensorAL1), decltype(inputs.b),
                             decltype(inputs.bias)>{inputs.scaleA, inputs.scaleB, tensorAL1, inputs.b, inputs.bias};
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
              typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, uint64_t iter0, TensorScaleAL1 &tensorScaleAL1,
                                   TensorScaleBL1 &tensorScaleBL1, TensorAL1 &tensorAL1, TensorBL1 &tensorBL1,
                                   TensorBiasL1 &tensorBiasL1, TensorL0C &tensorL0C, uint64_t splitKIdx,
                                   uint64_t biasBufId)
    {
        // 从scaleKL1中切出kL1_对应的部分
        auto scaleKL1Len = Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto coordScaleKL1 = (iter0 & 1) * scaleKL1Len;
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(AscendC::Te::MakeCoord(coordScaleKL1, 0),
                                                        AscendC::Te::MakeShape(scaleKL1Len, tileL1L0Param.curN));
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(AscendC::Te::MakeCoord(0, coordScaleKL1),
                                                        AscendC::Te::MakeShape(tileL1L0Param.curM, scaleKL1Len));

        uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            auto curKL0 = (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadBKL1)
                              ? (tileL1L0Param.curPadBKL1 - iter1 * baseK_)
                              : baseK_;
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            uint16_t mte1WaitMFlag = (l0PingPong_ & 0x1) + M_MTE1_FLAG_0;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            // A, ScaleA L1->L0
            auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                tileL1L0Param.curM, curKL0);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, iter1 * baseK_),
                                                  AscendC::Te::MakeShape(tileL1L0Param.curM, curKL0));
            auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

            auto CopyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
            auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, fp8_e8m0_t>(l0Offset >> 4), layoutScaleAL0);
            AscendC::Te::Copy(
                CopyL12L0ScaleA, tensorScaleAL0,
                tensorBlockScaleAL1.Slice(
                    AscendC::Te::MakeCoord(
                        0, iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                    AscendC::Te::MakeShape(tileL1L0Param.curM,
                                           Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE)));

            // bias L1->BT
            auto layoutBt = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(baseN_ * biasBufId * sizeof(float)),
                layoutBt);
            if (NeedBias(iter0, iter1, splitKIdx)) {
                auto CopyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                AscendC::Te::Copy(CopyL12BT, tensorBt, tensorBiasL1);
            }

            // B, scaleB L1->L0
            auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                curKL0, tileL1L0Param.curN);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(iter1 * baseK_, 0),
                                                  AscendC::Te::MakeShape(curKL0, tileL1L0Param.curN));
            auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

            auto CopyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
            auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, fp8_e8m0_t>(l0Offset >> 4), layoutScaleBL0);
            AscendC::Te::Copy(
                CopyL12L0ScaleB, tensorScaleBL0,
                tensorBlockScaleBL1.Slice(
                    AscendC::Te::MakeCoord(
                        iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
                    AscendC::Te::MakeShape(Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                                           tileL1L0Param.curN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            Mmad(tileL1L0Param, iter0, iter1, kL0Iter, curKL0, tensorL0C, tensorAL0, tensorBL0, tensorBt, splitKIdx);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias>
    __aicore__ inline void IssueKTile(TensorA const &gmA, TensorB const &gmB, TensorScaleA const &gmScaleA,
                                      TensorScaleB const &gmScaleB, TensorBias const &gmBias,
                                      TileL1L0Param &tileL1L0Param, uint64_t iter0, uint64_t l1BufId,
                                      uint64_t scaleL1BufId, uint64_t splitKIdx)
    {
        SetKTileParam(tileL1L0Param, iter0);
        IssueScalesInL1(gmScaleA, gmScaleB, tileL1L0Param, scaleL1BufId, iter0);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        IssueAInL1(gmA, tileL1L0Param, l1BufId, iter0);

        if (isBias_ && iter0 == 0 && splitKIdx == 0) {
            auto layoutBiasL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>{}(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(l1BufferBiasOffset_[scaleL1BufId]),
                layoutBiasL1);
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
        }

        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadBKL1, tileL1L0Param.curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(iter0 * kL1_, 0),
                                  AscendC::Te::MakeShape(tileL1L0Param.curGmBKL1, tileL1L0Param.curN));
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    }

    template <typename TensorL0C>
    __aicore__ inline void ConsumeKTile(TileL1L0Param &tileL1L0Param, TensorL0C &tensorL0C, uint64_t iter0,
                                        uint64_t l1BufId, uint64_t scaleL1BufId, uint64_t splitKIdx)
    {
        SetKTileParam(tileL1L0Param, iter0);
        auto inputs = MakeL1Views(tileL1L0Param, l1BufId, scaleL1BufId);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        Iterate(tileL1L0Param, iter0, inputs.scaleA, inputs.scaleB, inputs.a, inputs.b, inputs.bias, tensorL0C,
                splitKIdx, scaleL1BufId);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

        if ((iter0 & 1) != 0 || iter0 + 1 == kL1Iter_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
        }
    }

    template <bool FillResident, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void IssueResidentPrefixKTile(TensorA const &gmA, TensorB const &gmB,
                                                    TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                                    TileL1L0Param &tileL1L0Param, uint64_t iter0, uint64_t residentSlot,
                                                    uint64_t l1BufId, uint64_t scaleL1BufId)
    {
        uint16_t residentEvent = static_cast<uint16_t>(INPUT_BUFFER_FLAG_2 + residentSlot);
        SetKTileParam(tileL1L0Param, iter0);
        IssueScalesInL1(gmScaleA, gmScaleB, tileL1L0Param, scaleL1BufId, iter0);

        if constexpr (FillResident) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(residentEvent);
            IssueResidentAInL1(gmA, tileL1L0Param, residentSlot, iter0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        IssueBInL1(gmB, tileL1L0Param, l1BufId, iter0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    }

    template <bool FirstUse, typename TensorL0C>
    __aicore__ inline void ConsumeResidentPrefixKTile(TileL1L0Param &tileL1L0Param, TensorL0C &tensorL0C,
                                                      uint64_t iter0, uint64_t residentSlot, uint64_t l1BufId,
                                                      uint64_t scaleL1BufId, uint64_t splitKIdx)
    {
        uint16_t residentEvent = static_cast<uint16_t>(INPUT_BUFFER_FLAG_2 + residentSlot);
        SetKTileParam(tileL1L0Param, iter0);
        auto inputs = MakeResidentAL1Views(tileL1L0Param, residentSlot, l1BufId, scaleL1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        Iterate(tileL1L0Param, iter0, inputs.scaleA, inputs.scaleB, inputs.a, inputs.b, inputs.bias, tensorL0C,
                splitKIdx, scaleL1BufId);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        if ((iter0 & 1U) != 0U || iter0 + 1U == kL1Iter_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
        }
        if constexpr (!FirstUse) {
            // The second weight half is the terminal consumer of this slot.
            // Release it only after all L1->L0 reads have completed, so the
            // next M/N tile may safely install a different activation strip.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(residentEvent);
        }
    }

    template <bool ReuseResident, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
              typename TensorBias>
    __aicore__ inline void IssueMaybeResidentKTile(TensorA const &gmA, TensorB const &gmB, TensorScaleA const &gmScaleA,
                                                   TensorScaleB const &gmScaleB, TensorBias const &gmBias,
                                                   TileL1L0Param &tileL1L0Param, uint64_t iter0, uint64_t residentCount,
                                                   uint64_t abBase, uint64_t scaleBase, uint64_t splitKIdx)
    {
        uint64_t l1BufId = (abBase + iter0) & 1U;
        uint64_t scaleL1BufId = (scaleBase + (iter0 >> 1U)) & 1U;
        if (iter0 < residentCount) {
            IssueResidentPrefixKTile<!ReuseResident>(gmA, gmB, gmScaleA, gmScaleB, tileL1L0Param, iter0, iter0, l1BufId,
                                                     scaleL1BufId);
            return;
        }
        IssueKTile(gmA, gmB, gmScaleA, gmScaleB, gmBias, tileL1L0Param, iter0, l1BufId, scaleL1BufId, splitKIdx);
    }

    template <bool ReuseResident, typename TensorL0C>
    __aicore__ inline void ConsumeMaybeResidentKTile(TileL1L0Param &tileL1L0Param, TensorL0C &tensorL0C, uint64_t iter0,
                                                     uint64_t residentCount, uint64_t abBase, uint64_t scaleBase,
                                                     uint64_t splitKIdx)
    {
        uint64_t l1BufId = (abBase + iter0) & 1U;
        uint64_t scaleL1BufId = (scaleBase + (iter0 >> 1U)) & 1U;
        if (iter0 < residentCount) {
            ConsumeResidentPrefixKTile<!ReuseResident>(tileL1L0Param, tensorL0C, iter0, iter0, l1BufId, scaleL1BufId,
                                                       splitKIdx);
            return;
        }
        ConsumeKTile(tileL1L0Param, tensorL0C, iter0, l1BufId, scaleL1BufId, splitKIdx);
    }

    // 分级接口第一步: GM->L1 load + MMAD (K循环), 不写输出。
    // L0C tensor 按值返回给调用侧, 由调用侧转交 CopyOut; 不落成员(见文件头注释2)。
    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias>
    __aicore__ inline auto Compute(TensorA const &gmA, TensorB const &gmB, TensorScaleA const &gmScaleA,
                                   TensorScaleB const &gmScaleB, TensorBias const &gmBias,
                                   BlockShape const &singleShape, uint64_t splitKIdx = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);
        uint64_t abBase = abL1LoopCnt_;
        uint64_t scaleBase = scaleLoopCnt_;
        if (kL1Iter_ != 0) {
            // Prologue + steady state. Buffer ids use only bit operations;
            // K_TILES_PER_SCALE and L1_BUFFER_NUM are both fixed at two.
            IssueKTile(gmA, gmB, gmScaleA, gmScaleB, gmBias, tileL1L0Param, 0, abBase & 1, scaleBase & 1, splitKIdx);
            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                if (iter0 + 1 < kL1Iter_) {
                    uint64_t nextIter = iter0 + 1;
                    IssueKTile(gmA, gmB, gmScaleA, gmScaleB, gmBias, tileL1L0Param, nextIter, (abBase + nextIter) & 1,
                               (scaleBase + (nextIter >> 1)) & 1, splitKIdx);
                }
                ConsumeKTile(tileL1L0Param, tensorL0C, iter0, (abBase + iter0) & 1, (scaleBase + (iter0 >> 1)) & 1,
                             splitKIdx);
            }
        }
        abL1LoopCnt_ = abBase + kL1Iter_;
        scaleLoopCnt_ = scaleBase + ((kL1Iter_ + 1) >> 1);
        return tensorL0C;
    }

    // Preserve the first two A K=256 strips across the two SwiGLU weight
    // halves.  M/N tiling and L0C accumulation order are unchanged: half 0
    // fills the two spare L1 payload slots, then half 1 reuses and releases
    // them.  K strips beyond the resident prefix keep the original two-stage
    // GM->L1/MMAD pipeline.  This is intentionally a prefix rather than a full
    // resident-A policy because H=2048/3200 does not fit in the remaining L1.
    template <bool ReuseResident, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
              typename TensorBias>
    __aicore__ inline auto ComputeWithResidentAPrefix(TensorA const &gmA, TensorB const &gmB,
                                                      TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                                      TensorBias const &gmBias, BlockShape const &singleShape,
                                                      uint64_t splitKIdx = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1U) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);

        uint64_t abBase = abL1LoopCnt_;
        uint64_t scaleBase = scaleLoopCnt_;
        uint64_t residentCount = kL1Iter_ < L1_BUFFER_NUM ? kL1Iter_ : L1_BUFFER_NUM;
        if (kL1Iter_ != 0U) {
            // Keep the original prologue/steady-state pipeline exactly:
            // issue(0), issue(1), consume(0), issue(2), consume(1), ... .
            // Resident A changes only the address and skips the second-half
            // GM copy; it must not serialize MTE2 behind MMAD.
            IssueMaybeResidentKTile<ReuseResident>(gmA, gmB, gmScaleA, gmScaleB, gmBias, tileL1L0Param, 0U,
                                                   residentCount, abBase, scaleBase, splitKIdx);
            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                if (iter0 + 1U < kL1Iter_) {
                    IssueMaybeResidentKTile<ReuseResident>(gmA, gmB, gmScaleA, gmScaleB, gmBias, tileL1L0Param,
                                                           iter0 + 1U, residentCount, abBase, scaleBase, splitKIdx);
                }
                ConsumeMaybeResidentKTile<ReuseResident>(tileL1L0Param, tensorL0C, iter0, residentCount, abBase,
                                                         scaleBase, splitKIdx);
            }
        }
        abL1LoopCnt_ = abBase + kL1Iter_;
        scaleLoopCnt_ = scaleBase + ((kL1Iter_ + 1U) >> 1U);
        return tensorL0C;
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias>
    __aicore__ inline auto ComputeWithResidentAPrefixRuntime(TensorA const &gmA, TensorB const &gmB,
                                                             TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                                             TensorBias const &gmBias, BlockShape const &singleShape,
                                                             bool reuseResident, uint64_t splitKIdx = 0)
    {
        if (reuseResident) {
            return ComputeWithResidentAPrefix<true>(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, splitKIdx);
        }
        return ComputeWithResidentAPrefix<false>(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, splitKIdx);
    }

    template <bool EnableResidentPrefix, typename TensorA, typename TensorB, typename TensorScaleA,
              typename TensorScaleB, typename TensorBias>
    __aicore__ inline auto ComputeMaybeWithResidentAPrefix(TensorA const &gmA, TensorB const &gmB,
                                                           TensorScaleA const &gmScaleA, TensorScaleB const &gmScaleB,
                                                           TensorBias const &gmBias, BlockShape const &singleShape,
                                                           bool reuseResident, uint64_t splitKIdx = 0)
    {
        if constexpr (!EnableResidentPrefix) {
            return Compute(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, splitKIdx);
        } else {
            if (CanReuseActivationPrefix(singleShape)) {
                return ComputeWithResidentAPrefixRuntime(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape,
                                                         reuseResident, splitKIdx);
            }
            return Compute(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, splitKIdx);
        }
    }

    // 分级接口第二步: L0C->UB/GM fixpipe 写出。调用侧可在 Compute 与本调用之间
    // 插入跨核同步 (GMM1 GM中转: 在此处之前投票, FIX_S排干的只是早已完成的
    // 上个tile的fixpipe, 不再停等本tile)。
    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyOut(TensorC const &gmC, TensorL0C &tensorL0C, uint64_t splitKIdx = 0)
    {
        if (splitKIdx == splitKNum_ - 1) {
            if constexpr (AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>) {
                // C L0C->UB
                auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
                AscendC::Te::Copy(CopyL0C2UB.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            } else {
                // C L0C->GM
                auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                AscendC::Te::Copy(CopyL0C2GM.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            }
            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(TensorA const &gmA, TensorB const &gmB, TensorScaleA const &gmScaleA,
                                      TensorScaleB const &gmScaleB, TensorBias const &gmBias, TensorC const &gmC,
                                      BlockShape const &singleShape, uint64_t splitKIdx = 0)
    {
        auto tensorL0C = Compute(gmA, gmB, gmScaleA, gmScaleB, gmBias, singleShape, splitKIdx);
        CopyOut(gmC, tensorL0C, splitKIdx);
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1, uint64_t kIter2)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0 && kIter2 == 0;
    }

    template <typename TensorL0CType, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(TileL1L0Param &tileL1L0Param, uint64_t iter0, uint64_t iter1, uint64_t kL0Iter,
                                uint64_t curKL0, TensorL0CType &tensorL0C, TensorAL0 const &tensorAL0,
                                TensorBL0 const &tensorBL0, TensorBT const &tensorBt, uint64_t splitKIdx)
    {
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.k = static_cast<uint16_t>(Blaze::Gemm::CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.unitFlag = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKIdx == splitKNum_ - 1))
                              ? FINAL_ACCUMULATION
                              : NON_FINAL_ACCUMULATION;
        params.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && splitKIdx == 0 && !isBias_);
        if (NeedBias(iter0, iter1, splitKIdx)) {
            AscendC::Te::Mmad(AscendC::Te::MmadAtom<
                                  AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                                  .with(params),
                              tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            AscendC::Te::Mmad(AscendC::Te::MmadAtom<
                                  AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                                  .with(params),
                              tensorL0C, tensorAL0, tensorBL0);
        }
    }

    constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
    constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;
    constexpr static uint16_t M_MTE1_FLAG_0 = 4;
    constexpr static uint16_t M_MTE1_FLAG_1 = 5;

    uint64_t l1BufferAOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBOffset_[L1_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleAOffset_[SCALE_BUFFER_NUM] = {0UL};
    uint64_t l1BufferScaleBOffset_[SCALE_BUFFER_NUM] = {0UL};
    uint64_t l1BufferBiasOffset_[SCALE_BUFFER_NUM] = {0UL};
    uint64_t residentAOffset_[L1_BUFFER_NUM] = {0UL};
    bool residentPrefixEnabled_{false};
    uint32_t residentSavedColumns_{0U};
    uint32_t residentMinSavedBytes_{0U};
};
#endif
}  // namespace Block
}  // namespace Gemm
}  // namespace Blaze

#endif  // BLOCK_MMAD_MOE_H
