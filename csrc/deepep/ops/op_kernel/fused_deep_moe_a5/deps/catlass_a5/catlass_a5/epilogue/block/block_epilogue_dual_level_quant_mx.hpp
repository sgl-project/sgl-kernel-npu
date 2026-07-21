/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_DUAL_LEVEL_QUANT_MX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_DUAL_LEVEL_QUANT_MX_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/epilogue/tile/tile_copy_dual_level_quant_mx.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/layout/matrix.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

namespace Catlass::Epilogue::Block {

/**
 * @brief BlockQuantDualLevelMx
 *
 * 与 BlockEpilogue<EpilogueAscend950DualLevelQuantMx, ...> 的关系:
 *   - 完全复用 microapi 计算 (ComputeDualLevelQuantPerRow / Level0+xTmp / Level1+reciprocal / Fp4Packed / FP4FromHalf)
 *   - 完全复用 UB 布局
 *   - **不再区分 A 与 B,也不感知 MN tile,仅处理 "一个 RowMajor 矩阵的一个 (rows × K_sub) tile"**
 *   - 不做 sub-tile 内部 swizzle / subblockIdx 分发(由 kernel 层的 QuantAllScheduler 处理)
 *
 * 调用语义(由新 kernel `DualLevelQuantMxBatchedMatmulTla` 的 AIV path 驱动):
 *   每个 AIV worker 从 QuantAllScheduler 拿一个 task,kernel 算出 batch+tile 的 GM 基址 + tile layout,
 *   传给 QuantizeTilePerRow 处理。A 与 B 的差异完全由 kernel 端选择不同 base 解决。
 *
 * 模板参数:
 *   DispatchPolicy_   - EpilogueAscend950DualLevelQuantMx<UB_STAGES>   (复用)
 *   SubTileShape_     - MatrixShape<M_sub, K_sub>; K_sub 必须 % LEVEL0_BLOCK_SIZE == 0
 *   InputType_        - GemmType<ElementInput, layout::RowMajor>       (fp16 | bf16)
 *   OutputType_       - GemmType<float4_e2m1x2_t, layout::RowMajor>    (FP4 packed)
 *   Scale1Type_       - GemmType<float, layout::RowMajor>
 *   Scale2Type_       - GemmType<float8_e8m0_t, layout::RowMajor>
 *   TileCopy_         - TileCopyDualLevelQuantMx<...>
 */
template <class DispatchPolicy_, class SubTileShape_, class InputType_, class OutputType_, class Scale1Type_,
          class Scale2Type_, class TileCopy_>
class BlockQuantDualLevelMx
{
public:
    // -----------------------------------------------------------------------
    // Type aliases
    // -----------------------------------------------------------------------
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;

    using SubTileShape = SubTileShape_;

    using ElementInput = typename InputType_::Element;
    using LayoutInput = typename InputType_::Layout;
    using ElementOutput = typename OutputType_::Element;
    using LayoutOutput = typename OutputType_::Layout;
    using ElementScale1 = typename Scale1Type_::Element;
    using LayoutScale1 = typename Scale1Type_::Layout;
    using ElementScale2 = typename Scale2Type_::Element;
    using LayoutScale2 = typename Scale2Type_::Layout;

    using TileCopy = TileCopy_;
    using CopyGmToUbInput = typename TileCopy_::CopyGmToUbInput;
    using CopyUbToGmOutput =
        Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Catlass::Gemm::GemmType<uint8_t, layout::RowMajor>>;
    using CopyUbToGmScale1 = typename TileCopy_::CopyUbToGmScale1;
    using CopyUbToGmScale2 = typename TileCopy_::CopyUbToGmScale2;

    // -----------------------------------------------------------------------
    // Constants  (与 BlockEpilogue<DualLevelQuantMx, ...> 完全一致,不要漂移)
    // -----------------------------------------------------------------------
    static constexpr uint32_t LEVEL0_BLOCK_SIZE = 512;
    static constexpr uint32_t LEVEL1_BLOCK_SIZE = 32;
    static constexpr float FP4_E2M1_MAX = 6.0f;
    static constexpr int32_t FP4_E2M1_EMAX = 2;

    // *** scheduler 必须从这里派生,不可在 kernel 里写死 (P2-A) ***
    static constexpr uint32_t SUB_TILE_M = SubTileShape::ROW;
    static constexpr uint32_t SUB_TILE_K = SubTileShape::COLUMN;
    static constexpr uint32_t SUB_TILE_COUNT = SubTileShape::COUNT;

    static_assert(SUB_TILE_K % LEVEL0_BLOCK_SIZE == 0,
                  "SubTileShape::COLUMN (K_sub) must be a multiple of LEVEL0_BLOCK_SIZE (512).");
    static_assert(SUB_TILE_M > 0 && SUB_TILE_K > 0, "SubTileShape must be positive.");

    static constexpr uint32_t L0_BLOCKS_PER_SUBTILE = SUB_TILE_K / LEVEL0_BLOCK_SIZE;
    static constexpr uint32_t L1_BLOCKS_PER_SUBTILE = SUB_TILE_K / LEVEL1_BLOCK_SIZE;
    static constexpr uint32_t L1_BLOCKS_PER_L0 = LEVEL0_BLOCK_SIZE / LEVEL1_BLOCK_SIZE;  // 16

    // MicroAPI vector lane counts
    static constexpr uint32_t VL_HALF = 128;
    static constexpr uint32_t VL_FLOAT = 64;
    static constexpr uint32_t UB_BLK = 32;

    // MicroAPI bit-manipulation constants.
    static constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    static constexpr int16_t SHR_NUM_BF16 = 7;
    static constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    static constexpr uint32_t FP4_E2M1_MAX_RECIPROCAL = 0x3e2aaaab;
    static constexpr uint16_t INF_FOR_BF16_CONST = 0x7f80;
    static constexpr uint16_t INF_FOR_FP16_CONST = 0x7c00;
    static constexpr uint32_t INVALID_FOR_FP32 = 0x00800000;
    static constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
    static constexpr int32_t FP32_BIAS_VAL = 127;
    static constexpr int32_t FP32_BIAS_NEG_VAL = -127;
    static constexpr int32_t NEG_ONE_I32 = -1;
    static constexpr int32_t NEG_ZERO_I32 = static_cast<int32_t>(0x80000000);
    static constexpr int16_t SHR_NUM_FP32 = 23;

#define CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(name, layout, sat, round)                  \
    static constexpr MAPI::CastTrait name = {MAPI::RegLayout::layout, MAPI::SatMode::sat, \
                                             MAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::round}

    // -----------------------------------------------------------------------
    // UB budget (单缓冲, A/B 串行复用,单 kernel 路径不需要多 stage)
    // -----------------------------------------------------------------------
    static constexpr size_t UB_INPUT_BYTES = SUB_TILE_COUNT * sizeof(ElementInput);
    static constexpr size_t UB_FP4OUT_BYTES = SUB_TILE_COUNT / 2;
    static constexpr size_t UB_SCALE1_BYTES = SUB_TILE_M * RoundUp<8>(L0_BLOCKS_PER_SUBTILE) * sizeof(float);
    static constexpr size_t UB_SCALE2_BYTES = SUB_TILE_M * RoundUp<32>(L1_BLOCKS_PER_SUBTILE) * sizeof(uint8_t);
    static constexpr size_t UB_SCRATCH_BYTES = 4096;
    static constexpr size_t UB_RECIPROCAL_BYTES = 64;
    static constexpr size_t UB_PER_STAGE =
        UB_INPUT_BYTES + UB_FP4OUT_BYTES + UB_SCALE1_BYTES + UB_SCALE2_BYTES + UB_RECIPROCAL_BYTES;

    static_assert(UB_STAGES * UB_PER_STAGE + UB_SCRATCH_BYTES <= ArchTag::UB_SIZE,
                  "Sub-tile shape too large for UB; reduce SUB_TILE_M / SUB_TILE_K or UB_STAGES.");

    // -----------------------------------------------------------------------
    // Construction / lifecycle
    //
    // Prime the flag so the first GM->UB transfer does not need to wait.
    // 由于 block 不区分 A/B,只使用 EVENT_ID0 通道。
    // -----------------------------------------------------------------------
    CATLASS_DEVICE
    BlockQuantDualLevelMx(Arch::Resource<ArchTag> &resource)
    {
        AllocateUbBuffers(resource);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockQuantDualLevelMx()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    // -----------------------------------------------------------------------
    // QuantizeTilePerRow:处理 "一个 RowMajor 矩阵的一个 (rows × actualK) tile"
    //
    // 调用方(kernel)负责:
    //   - 把 GM tensor 偏移到 batch + tile 起点
    //   - 提供 tile 级 layout (其 stride(0) 给出 GM 的全行 stride)
    //   - 提供 actualTileShape (实际有效 rows / actualK,用于 K-tail 处理)
    //
    // 本函数实现:
    //   1. UB Duplicate 清零 → 保证 K-tail padding 区域为 0
    //   2. DataCopyPad GM→UB,UB 行 stride 固定 SUB_TILE_K
    //   3. ComputeDualLevelQuantPerRow (rows, SUB_TILE_K) 全量量化
    //   4. UB→GM 三路: FP4 packed bytes / scale1 (float) / scale2 (fp8_e8m0)
    //
    // 同步: per-call 自闭环,MTE3_V flag 在 call 之间手递 (上一 call 末 set,本 call 头 wait)
    // -----------------------------------------------------------------------
    template <class LayoutInputTile, class LayoutOutputTile, class LayoutScale1Tile, class LayoutScale2Tile>
    CATLASS_DEVICE void QuantizeTilePerRow(
        AscendC::GlobalTensor<ElementInput> const &gmInputTile, LayoutInputTile const &layoutInputTileGm,
        AscendC::GlobalTensor<uint8_t> const &gmOutputFp4ByteTile, LayoutOutputTile const &layoutOutputFp4ByteTileGm,
        AscendC::GlobalTensor<ElementScale1> const &gmScale1Tile, LayoutScale1Tile const &layoutScale1TileGm,
        AscendC::GlobalTensor<ElementScale2> const &gmScale2Tile, LayoutScale2Tile const &layoutScale2TileGm,
        MatrixCoord const &actualTileShape)
    {
        // ---- UB 清零仅 K-tail 需要; full tile 会被 GM->UB 完全覆盖 ----
        if (actualTileShape.column() < SUB_TILE_K) {
            AscendC::Duplicate(ubInput, static_cast<ElementInput>(0), SUB_TILE_COUNT);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        // ---- GM → UB: DataCopyPad,显式 rightPadding 把每行末尾补到 32B 边界 ----
        // K-tail fix: DataCopyPad pads each row to a 32B boundary.
        constexpr uint32_t kEleNumPerBlk = BYTE_PER_BLK / sizeof(ElementInput);
        uint32_t rightPadding = (kEleNumPerBlk - actualTileShape.column() % kEleNumPerBlk) % kEleNumPerBlk;
        AscendC::DataCopyExtParams dataCopyParams(
            actualTileShape.row(), actualTileShape.column() * sizeof(ElementInput),
            (layoutInputTileGm.stride(0) - actualTileShape.column()) * sizeof(ElementInput),
            (SUB_TILE_K - actualTileShape.column() - rightPadding) / kEleNumPerBlk, 0);
        AscendC::DataCopyPadExtParams<ElementInput> padParams(true, 0, rightPadding, static_cast<ElementInput>(0));
        AscendC::DataCopyPad(ubInput, gmInputTile, dataCopyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // ---- 量化计算 (per-row K-axis) ----
        // cols 固定传 SUB_TILE_K,microapi 始终按完整 512 元素处理 (K-tail 由 padding 0 覆盖)
        ComputeDualLevelQuantPerRow(ubInput, ubFp4Out, ubScale1, ubScale2, actualTileShape.row(), SUB_TILE_K);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        // ---- UB → GM 三路写出 ----
        // OutputFp4: GM shape 按 actual K,UB source stride 固定 SUB_TILE_K/2 (K-tail 支持)
        uint32_t packedCols = CeilDiv(actualTileShape.column(), static_cast<uint32_t>(2));
        auto layoutOutputUb = layout::RowMajor(actualTileShape.row(), packedCols,
                                               static_cast<layout::RowMajor::LongIndex>(SUB_TILE_K / 2));
        copyUbToGmOutput(gmOutputFp4ByteTile, ubFp4Out, layoutOutputFp4ByteTileGm, layoutOutputUb);

        // Scale1: float, RowMajor [rows, ceil(actualK/512)]
        MatrixCoord scale1TileShape{actualTileShape.row(), CeilDiv<LEVEL0_BLOCK_SIZE>(actualTileShape.column())};
        auto layoutScale1Ub = layout::RowMajor::template MakeLayoutInUb<ElementScale1>(scale1TileShape);
        copyUbToGmScale1(gmScale1Tile, ubScale1, layoutScale1TileGm, layoutScale1Ub);

        // Scale2: fp8_e8m0, RowMajor [rows, round_up(ceil(actualK/32), 2)].
        uint32_t scale2Cols = RoundUp<2>(CeilDiv<LEVEL1_BLOCK_SIZE>(actualTileShape.column()));
        MatrixCoord scale2TileShape{actualTileShape.row(), scale2Cols};
        auto layoutScale2Ub = layout::RowMajor::template MakeLayoutInUb<ElementScale2>(scale2TileShape);
        auto ubScale2Typed = ubScale2.template ReinterpretCast<ElementScale2>();
        copyUbToGmScale2(gmScale2Tile, ubScale2Typed, layoutScale2TileGm, layoutScale2Ub);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    // -----------------------------------------------------------------------
    // FP4 packing helpers (kernel 端可能用到,公开)
    //
    // Helpers for computing packed FP4 byte offsets through layout APIs.
    // -----------------------------------------------------------------------
    template <class LayoutOutput>
    CATLASS_DEVICE static int64_t GetPackedFp4ByteOffset(LayoutOutput const &layoutOutput, MatrixCoord const &offset)
    {
        return layoutOutput.GetOffset(offset) / 2;
    }

    template <class LayoutOutput>
    CATLASS_DEVICE static layout::RowMajor MakePackedFp4ByteGmLayout(LayoutOutput const &layoutOutput,
                                                                     MatrixCoord const &logicalShape)
    {
        uint32_t packedCols = CeilDiv(logicalShape.column(), 2);
        return layout::RowMajor(logicalShape.row(), packedCols, layoutOutput.stride(0) / 2);
    }

private:
    // -----------------------------------------------------------------------
    // UB allocation: single-buffer layout for the quant path.
    //   ubInput | ubFp4Out | ubScale1 | ubScale2 | ubReciprocal | (scratch)
    // -----------------------------------------------------------------------
    CATLASS_DEVICE
    void AllocateUbBuffers(Arch::Resource<ArchTag> &resource)
    {
        size_t offset = 0;
        ubInput = resource.ubBuf.template GetBufferByByte<ElementInput>(offset);
        offset += UB_INPUT_BYTES;
        ubFp4Out = resource.ubBuf.template GetBufferByByte<uint8_t>(offset);
        offset += UB_FP4OUT_BYTES;
        ubScale1 = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += UB_SCALE1_BYTES;
        ubScale2 = resource.ubBuf.template GetBufferByByte<uint8_t>(offset);
        offset += UB_SCALE2_BYTES;
        ubReciprocal = resource.ubBuf.template GetBufferByByte<uint16_t>(offset);
    }

    // =======================================================================
    // MicroAPI helpers for the dual-level quantization flow.
    // =======================================================================

    // -----------------------------------------------------------------------
    // ComputeDualLevelQuantPerRow: 对 UB 中 (rows × cols) 按行做 K 轴二级动态量化
    // -----------------------------------------------------------------------
    CATLASS_DEVICE
    void ComputeDualLevelQuantPerRow(AscendC::LocalTensor<ElementInput> ubIn, AscendC::LocalTensor<uint8_t> ubFp4,
                                     AscendC::LocalTensor<float> ubS1, AscendC::LocalTensor<uint8_t> ubS2,
                                     uint32_t rows, uint32_t cols)
    {
        uint32_t l0Blocks = cols / LEVEL0_BLOCK_SIZE;
        uint32_t l1Blocks = cols / LEVEL1_BLOCK_SIZE;

        uint32_t inStride = cols;
        uint32_t fp4Stride = cols / 2;
        uint32_t s1Stride = RoundUp<8>(l0Blocks);
        uint32_t s2Stride = RoundUp<32>(l1Blocks);

        auto *inBase = (__ubuf__ ElementInput *)ubIn.GetPhyAddr();
        auto *fp4Base = (__ubuf__ uint8_t *)ubFp4.GetPhyAddr();
        auto *s1Base = (__ubuf__ float *)ubS1.GetPhyAddr();
        auto *s2Base = (__ubuf__ uint8_t *)ubS2.GetPhyAddr();
        auto *recipBase = (__ubuf__ uint16_t *)ubReciprocal.GetPhyAddr();

        for (uint32_t row = 0; row < rows; ++row) {
            auto *rowIn = inBase + row * inStride;
            auto *rowFp4 = fp4Base + row * fp4Stride;
            auto *rowS1 = s1Base + row * s1Stride;
            auto *rowS2 = s2Base + row * s2Stride;

            for (uint32_t l0 = 0; l0 < l0Blocks; ++l0) {
                auto *x = rowIn + l0 * LEVEL0_BLOCK_SIZE;
                auto *y = rowFp4 + l0 * (LEVEL0_BLOCK_SIZE / 2);
                auto *s1 = rowS1 + l0;
                auto *s2 = rowS2 + l0 * L1_BLOCKS_PER_L0;

                ComputeLevel0AndXTmp(x, s1);
                ComputeLevel1ScaleAndReciprocal(x, s2, recipBase);
                ComputeFp4Packed(x, y, recipBase);
            }
        }
    }

    // -----------------------------------------------------------------------
    // ComputeLevel0AndXTmp (verbatim copy)
    // -----------------------------------------------------------------------
    __simd_vf__ inline void ComputeLevel0AndXTmp(__ubuf__ ElementInput *xAddr, __ubuf__ float *s1Addr)
    {
        namespace MAPI = AscendC::MicroAPI;
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastXToFp32Zero, ZERO, UNKNOWN, UNKNOWN);

        MAPI::RegTensor<ElementInput> x0, x1, x2, x3;
        MAPI::RegTensor<uint16_t> absX0, absX1, absX2, absX3;
        MAPI::RegTensor<float> level0Scale;
        MAPI::RegTensor<float> x0Z, x0O, x1Z, x1O, x2Z, x2O, x3Z, x3O;
        MAPI::RegTensor<uint32_t> yMaxExpReg, invalidDataReg;
        MAPI::RegTensor<uint16_t> absForXReg, infForXReg, zeroReg16;
        MAPI::MaskReg infMask, invalidDataMask;
        MAPI::MaskReg maskAll16 = MAPI::CreateMask<uint16_t, MAPI::MaskPattern::ALL>();
        MAPI::MaskReg maskAll32 = MAPI::CreateMask<float, MAPI::MaskPattern::ALL>();
        MAPI::UnalignRegForStore ureg;

        MAPI::Duplicate(yMaxExpReg, FP4_E2M1_MAX_RECIPROCAL);
        MAPI::Duplicate(absForXReg, ABS_FOR_UINT16);
        MAPI::Duplicate(invalidDataReg, INVALID_FOR_FP32);
        MAPI::Duplicate(zeroReg16, static_cast<uint16_t>(0));
        if constexpr (std::is_same_v<ElementInput, half>) {
            MAPI::Duplicate(infForXReg, INF_FOR_FP16_CONST);
        } else {
            MAPI::Duplicate(infForXReg, INF_FOR_BF16_CONST);
        }

        auto *loadAddr = xAddr;
        MAPI::DataCopy<ElementInput, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_NORM>(x0, loadAddr,
                                                                                                     VL_HALF);
        MAPI::DataCopy<ElementInput, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_NORM>(x1, loadAddr,
                                                                                                     VL_HALF);
        MAPI::DataCopy<ElementInput, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_NORM>(x2, loadAddr,
                                                                                                     VL_HALF);
        MAPI::DataCopy<ElementInput, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_NORM>(x3, loadAddr,
                                                                                                     VL_HALF);

        MAPI::And(absX0, (MAPI::RegTensor<uint16_t> &)x0, absForXReg, maskAll16);
        MAPI::And(absX1, (MAPI::RegTensor<uint16_t> &)x1, absForXReg, maskAll16);
        MAPI::And(absX2, (MAPI::RegTensor<uint16_t> &)x2, absForXReg, maskAll16);
        MAPI::And(absX3, (MAPI::RegTensor<uint16_t> &)x3, absForXReg, maskAll16);

        MAPI::Compare<uint16_t, AscendC::CMPMODE::GE>(infMask, absX0, infForXReg, maskAll16);
        MAPI::Select<uint16_t>(absX0, zeroReg16, absX0, infMask);
        MAPI::Compare<uint16_t, AscendC::CMPMODE::GE>(infMask, absX1, infForXReg, maskAll16);
        MAPI::Select<uint16_t>(absX1, zeroReg16, absX1, infMask);
        MAPI::Compare<uint16_t, AscendC::CMPMODE::GE>(infMask, absX2, infForXReg, maskAll16);
        MAPI::Select<uint16_t>(absX2, zeroReg16, absX2, infMask);
        MAPI::Compare<uint16_t, AscendC::CMPMODE::GE>(infMask, absX3, infForXReg, maskAll16);
        MAPI::Select<uint16_t>(absX3, zeroReg16, absX3, infMask);

        MAPI::Max(absX0, absX0, absX1, maskAll16);
        MAPI::Max(absX2, absX2, absX3, maskAll16);
        MAPI::Max(absX0, absX0, absX2, maskAll16);
        MAPI::ReduceMax(absX0, absX0, maskAll16);

        MAPI::Cast<float, ElementInput, kCastXToFp32Zero>(level0Scale, (MAPI::RegTensor<ElementInput> &)absX0,
                                                          maskAll16);
        MAPI::Mul(level0Scale, level0Scale, (MAPI::RegTensor<float> &)yMaxExpReg, maskAll32);

        MAPI::Compare<uint32_t, AscendC::CMPMODE::LT>(invalidDataMask, (MAPI::RegTensor<uint32_t> &)level0Scale,
                                                      invalidDataReg, maskAll32);
        MAPI::Select<float>(level0Scale, (MAPI::RegTensor<float> &)zeroReg16, level0Scale, invalidDataMask);

        MAPI::StoreUnAlign<float, MAPI::PostLiteral::POST_MODE_UPDATE>(s1Addr, level0Scale, ureg,
                                                                       static_cast<uint32_t>(1));
        MAPI::StoreUnAlignPost<float, MAPI::PostLiteral::POST_MODE_UPDATE>(s1Addr, ureg, static_cast<int32_t>(0));

        MAPI::Duplicate(level0Scale, level0Scale, maskAll32);

        CalcXTmp(xAddr, level0Scale, x0, x0Z, x0O);
        CalcXTmp(xAddr + VL_HALF, level0Scale, x1, x1Z, x1O);
        CalcXTmp(xAddr + 2 * VL_HALF, level0Scale, x2, x2Z, x2O);
        CalcXTmp(xAddr + 3 * VL_HALF, level0Scale, x3, x3Z, x3O);
    }

    // -----------------------------------------------------------------------
    // CalcXTmp (verbatim copy)
    // -----------------------------------------------------------------------
    template <class RegTensorFloat, class RegTensorInput>
    __simd_callee__ inline void CalcXTmp(__ubuf__ ElementInput *xTmpAddr, RegTensorFloat level0ScaleReg,
                                         RegTensorInput xReg, RegTensorFloat &xZeroFP32, RegTensorFloat &xOneFP32)
    {
        namespace MAPI = AscendC::MicroAPI;
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastXToFp32Zero, ZERO, UNKNOWN, UNKNOWN);
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastXToFp32One, ONE, UNKNOWN, UNKNOWN);
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastFp32ToBF16, ZERO, NO_SAT, CAST_RINT);

        MAPI::RegTensor<ElementInput> xZero, xOne, zero;
        MAPI::MaskReg zeroMask;
        MAPI::MaskReg maskAll16 = MAPI::CreateMask<ElementInput, MAPI::MaskPattern::ALL>();
        MAPI::MaskReg maskAll32 = MAPI::CreateMask<uint32_t, MAPI::MaskPattern::ALL>();
        MAPI::Duplicate(zero, static_cast<ElementInput>(0));

        MAPI::Compare<ElementInput, AscendC::CMPMODE::EQ>(zeroMask, xReg, zero, maskAll16);

        MAPI::Cast<float, ElementInput, kCastXToFp32Zero>(xZeroFP32, xReg, maskAll16);
        MAPI::Cast<float, ElementInput, kCastXToFp32One>(xOneFP32, xReg, maskAll16);

        MAPI::Div(xZeroFP32, xZeroFP32, level0ScaleReg, maskAll32);
        MAPI::Div(xOneFP32, xOneFP32, level0ScaleReg, maskAll32);

        MAPI::Cast<ElementInput, float, kCastFp32ToBF16>(xZero, xZeroFP32, maskAll32);
        MAPI::Cast<ElementInput, float, kCastFp32ToBF16>(xOne, xOneFP32, maskAll32);

        MAPI::Pack<uint16_t, uint32_t, MAPI::HighLowPart::LOWEST>((MAPI::RegTensor<uint16_t> &)xZero,
                                                                  (MAPI::RegTensor<uint32_t> &)xZero);
        MAPI::Pack<uint16_t, uint32_t, MAPI::HighLowPart::LOWEST>((MAPI::RegTensor<uint16_t> &)xOne,
                                                                  (MAPI::RegTensor<uint32_t> &)xOne);

        MAPI::Interleave(xZero, xOne, xZero, xOne);
        MAPI::Select<ElementInput>(xZero, xReg, xZero, zeroMask);

        MAPI::DataCopy(xTmpAddr, xZero, maskAll16);
    }

    // -----------------------------------------------------------------------
    // ComputeLevel1ScaleAndReciprocal (verbatim copy)
    // -----------------------------------------------------------------------
    __simd_vf__ inline void ComputeLevel1ScaleAndReciprocal(__ubuf__ ElementInput *xTmpAddr, __ubuf__ uint8_t *s2Addr,
                                                            __ubuf__ uint16_t *recipAddr)
    {
        namespace MAPI = AscendC::MicroAPI;
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastHalfToBF16, UNKNOWN, UNKNOWN, CAST_TRUNC);

        MAPI::RegTensor<ElementInput> xTmp0, xTmp1;
        MAPI::RegTensor<bfloat16_t> xTmp0BF16, xTmp1BF16;
        MAPI::RegTensor<uint16_t> xTmp0Exp, xTmp1Exp;
        MAPI::RegTensor<uint16_t> xTmp0ExpFP16, xTmp1ExpFP16;
        MAPI::RegTensor<uint16_t> expMaskBF16Reg, expMaskFP16Reg;
        MAPI::RegTensor<uint16_t> yMaxExpReg, nanE8M0Reg, biasE8M0Reg;
        MAPI::RegTensor<uint16_t> zeroReg, nanBF16Reg, specialExpReg;
        MAPI::RegTensor<uint16_t> mxScale1B16, reversedShareExp1;
        MAPI::RegTensor<uint8_t> mxScale1B8;

        MAPI::MaskReg infMask, zeroMask, invalidDataMask;
        MAPI::MaskReg infNanMask0, infNanMask1;
        MAPI::MaskReg maskAll16 = MAPI::CreateMask<uint16_t, MAPI::MaskPattern::ALL>();
        MAPI::MaskReg maskReduceB16 = MAPI::CreateMask<uint8_t, MAPI::MaskPattern::VL16>();

        MAPI::Duplicate(expMaskBF16Reg, INF_FOR_BF16_CONST);
        MAPI::Duplicate(expMaskFP16Reg, INF_FOR_FP16_CONST);
        MAPI::Duplicate(yMaxExpReg, FP4_E2M1_BF16_MAX_EXP);
        MAPI::Duplicate(nanE8M0Reg, NAN_FOR_FP8_E8M0);
        MAPI::Duplicate(biasE8M0Reg, BF16_EXP_BIAS);
        MAPI::Duplicate(zeroReg, static_cast<uint16_t>(0));
        MAPI::Duplicate(nanBF16Reg, NAN_CUSTOMIZATION);
        MAPI::Duplicate(specialExpReg, SPECIAL_EXP_THRESHOLD);

        auto *xLoad = xTmpAddr;
        auto *s2Write = s2Addr;
        auto *recipWrite = recipAddr;

        for (uint16_t i = 0; i < 2; ++i) {
            MAPI::DataCopy<ElementInput, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_DINTLV_B16>(
                xTmp0, xTmp1, xLoad, VL_HALF * 2);

            if constexpr (std::is_same_v<ElementInput, half>) {
                MAPI::And(xTmp0ExpFP16, (MAPI::RegTensor<uint16_t> &)xTmp0, expMaskFP16Reg, maskAll16);
                MAPI::And(xTmp1ExpFP16, (MAPI::RegTensor<uint16_t> &)xTmp1, expMaskFP16Reg, maskAll16);
                MAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(infNanMask0, xTmp0ExpFP16, expMaskFP16Reg, maskAll16);
                MAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(infNanMask1, xTmp1ExpFP16, expMaskFP16Reg, maskAll16);
                MAPI::Cast<bfloat16_t, ElementInput, kCastHalfToBF16>(xTmp0BF16, xTmp0, maskAll16);
                MAPI::Cast<bfloat16_t, ElementInput, kCastHalfToBF16>(xTmp1BF16, xTmp1, maskAll16);
                MAPI::And(xTmp0Exp, (MAPI::RegTensor<uint16_t> &)xTmp0BF16, expMaskBF16Reg, maskAll16);
                MAPI::And(xTmp1Exp, (MAPI::RegTensor<uint16_t> &)xTmp1BF16, expMaskBF16Reg, maskAll16);
                MAPI::Select<uint16_t>(xTmp0Exp, xTmp0Exp, expMaskBF16Reg, infNanMask0);
                MAPI::Select<uint16_t>(xTmp1Exp, xTmp1Exp, expMaskBF16Reg, infNanMask1);
            } else {
                MAPI::And(xTmp0Exp, (MAPI::RegTensor<uint16_t> &)xTmp0, expMaskBF16Reg, maskAll16);
                MAPI::And(xTmp1Exp, (MAPI::RegTensor<uint16_t> &)xTmp1, expMaskBF16Reg, maskAll16);
            }

            MAPI::Max(xTmp0Exp, xTmp1Exp, xTmp0Exp, maskAll16);
            MAPI::ReduceMaxWithDataBlock(xTmp0Exp, xTmp0Exp, maskAll16);

            MAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(infMask, xTmp0Exp, expMaskBF16Reg, maskAll16);
            MAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, xTmp0Exp, zeroReg, maskAll16);
            MAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, xTmp0Exp, yMaxExpReg, maskAll16);
            MAPI::Select<uint16_t>(xTmp0Exp, yMaxExpReg, xTmp0Exp, invalidDataMask);
            MAPI::Sub(xTmp0Exp, xTmp0Exp, yMaxExpReg, maskAll16);
            MAPI::ShiftRights(mxScale1B16, xTmp0Exp, SHR_NUM_BF16, maskAll16);
            MAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0Reg, infMask);
            MAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, zeroReg, zeroMask);

            MAPI::Pack<uint8_t, uint16_t, MAPI::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            MAPI::UnalignRegForStore ureg;
            MAPI::StoreUnAlign<uint8_t, MAPI::PostLiteral::POST_MODE_UPDATE>(s2Write, mxScale1B8, ureg,
                                                                             static_cast<uint32_t>(8));
            MAPI::StoreUnAlignPost<uint8_t, MAPI::PostLiteral::POST_MODE_UPDATE>(s2Write, ureg,
                                                                                 static_cast<int32_t>(0));

            MAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(invalidDataMask, xTmp0Exp, biasE8M0Reg, maskAll16);
            MAPI::Sub(reversedShareExp1, biasE8M0Reg, xTmp0Exp, maskAll16);
            MAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16Reg, infMask);
            MAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zeroReg, zeroMask);
            MAPI::Select<uint16_t>(reversedShareExp1, specialExpReg, reversedShareExp1, invalidDataMask);
            MAPI::DataCopy<uint16_t, MAPI::PostLiteral::POST_MODE_UPDATE>(recipWrite, reversedShareExp1,
                                                                          UB_BLK / sizeof(uint16_t), maskReduceB16);
        }
    }

    // -----------------------------------------------------------------------
    // ComputeFp4Packed (verbatim copy)
    // -----------------------------------------------------------------------
    __simd_vf__ inline void ComputeFp4Packed(__ubuf__ ElementInput *xTmpAddr, __ubuf__ uint8_t *yAddr,
                                             __ubuf__ uint16_t *recipAddr)
    {
        namespace MAPI = AscendC::MicroAPI;
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastXToFp32Zero, ZERO, UNKNOWN, UNKNOWN);
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastXToFp32One, ONE, UNKNOWN, UNKNOWN);
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastBF16ToFp4, ZERO, SAT, CAST_RINT);
        CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT(kCastFp32ToBF16, ZERO, NO_SAT, CAST_RINT);

        MAPI::RegTensor<ElementInput> xTmp0, xTmp1;
        MAPI::RegTensor<uint16_t> scaleForMulFP16;
        MAPI::RegTensor<float> scaleForMulZeroFP32;
        MAPI::RegTensor<fp4x2_e2m1_t> y0FP4, y1FP4;

        MAPI::RegTensor<float> xTmp0ZeroFP32, xTmp0OneFP32;
        MAPI::RegTensor<float> xTmp1ZeroFP32, xTmp1OneFP32;
        MAPI::RegTensor<bfloat16_t> xTmp0ZeroBF16, xTmp0OneBF16;
        MAPI::RegTensor<bfloat16_t> xTmp1ZeroBF16, xTmp1OneBF16;

        MAPI::MaskReg dataMaskB8 = MAPI::CreateMask<uint8_t>();
        MAPI::MaskReg dataMaskB16 = MAPI::CreateMask<uint16_t>();
        MAPI::MaskReg dataMaskB32 = MAPI::CreateMask<uint32_t>();

        auto *xLoad = xTmpAddr;
        auto *yWrite = yAddr;
        auto *recipRead = recipAddr;

        for (uint16_t i = 0; i < 2; ++i) {
            MAPI::DataCopy<uint16_t, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, recipRead, UB_BLK / sizeof(uint16_t));

            MAPI::DataCopy<ElementInput, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::LoadDist::DIST_DINTLV_B16>(
                xTmp0, xTmp1, xLoad, VL_HALF * 2);

            if constexpr (std::is_same_v<ElementInput, half>) {
                MAPI::Cast<float, bfloat16_t, kCastXToFp32Zero>(
                    scaleForMulZeroFP32, (MAPI::RegTensor<bfloat16_t> &)scaleForMulFP16, dataMaskB16);

                MAPI::Cast<float, ElementInput, kCastXToFp32Zero>(xTmp0ZeroFP32, xTmp0, dataMaskB16);
                MAPI::Cast<float, ElementInput, kCastXToFp32One>(xTmp0OneFP32, xTmp0, dataMaskB16);
                MAPI::Mul(xTmp0ZeroFP32, scaleForMulZeroFP32, xTmp0ZeroFP32, dataMaskB32);
                MAPI::Mul(xTmp0OneFP32, scaleForMulZeroFP32, xTmp0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(xTmp0ZeroFP32);
                ComputeFP4FromHalf(xTmp0OneFP32);
                MAPI::Cast<bfloat16_t, float, kCastFp32ToBF16>(xTmp0ZeroBF16, xTmp0ZeroFP32, dataMaskB32);
                MAPI::Cast<bfloat16_t, float, kCastFp32ToBF16>(xTmp0OneBF16, xTmp0OneFP32, dataMaskB32);
                MAPI::Pack<uint16_t, uint32_t, MAPI::HighLowPart::LOWEST>((MAPI::RegTensor<uint16_t> &)xTmp0ZeroBF16,
                                                                          (MAPI::RegTensor<uint32_t> &)xTmp0ZeroBF16);
                MAPI::Pack<uint16_t, uint32_t, MAPI::HighLowPart::LOWEST>((MAPI::RegTensor<uint16_t> &)xTmp0OneBF16,
                                                                          (MAPI::RegTensor<uint32_t> &)xTmp0OneBF16);
                MAPI::Interleave(xTmp0ZeroBF16, xTmp0OneBF16, xTmp0ZeroBF16, xTmp0OneBF16);

                MAPI::Cast<float, ElementInput, kCastXToFp32Zero>(xTmp1ZeroFP32, xTmp1, dataMaskB16);
                MAPI::Cast<float, ElementInput, kCastXToFp32One>(xTmp1OneFP32, xTmp1, dataMaskB16);
                MAPI::Mul(xTmp1ZeroFP32, scaleForMulZeroFP32, xTmp1ZeroFP32, dataMaskB32);
                MAPI::Mul(xTmp1OneFP32, scaleForMulZeroFP32, xTmp1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(xTmp1ZeroFP32);
                ComputeFP4FromHalf(xTmp1OneFP32);
                MAPI::Cast<bfloat16_t, float, kCastFp32ToBF16>(xTmp1ZeroBF16, xTmp1ZeroFP32, dataMaskB32);
                MAPI::Cast<bfloat16_t, float, kCastFp32ToBF16>(xTmp1OneBF16, xTmp1OneFP32, dataMaskB32);
                MAPI::Pack<uint16_t, uint32_t, MAPI::HighLowPart::LOWEST>((MAPI::RegTensor<uint16_t> &)xTmp1ZeroBF16,
                                                                          (MAPI::RegTensor<uint32_t> &)xTmp1ZeroBF16);
                MAPI::Pack<uint16_t, uint32_t, MAPI::HighLowPart::LOWEST>((MAPI::RegTensor<uint16_t> &)xTmp1OneBF16,
                                                                          (MAPI::RegTensor<uint32_t> &)xTmp1OneBF16);
                MAPI::Interleave(xTmp1ZeroBF16, xTmp1OneBF16, xTmp1ZeroBF16, xTmp1OneBF16);

                MAPI::Interleave(xTmp0ZeroBF16, xTmp1ZeroBF16, xTmp0ZeroBF16, xTmp1ZeroBF16);
                MAPI::Cast<fp4x2_e2m1_t, bfloat16_t, kCastBF16ToFp4>(y0FP4, xTmp0ZeroBF16, dataMaskB16);
                MAPI::Cast<fp4x2_e2m1_t, bfloat16_t, kCastBF16ToFp4>(y1FP4, xTmp1ZeroBF16, dataMaskB16);
            } else {
                MAPI::Mul(xTmp0, (MAPI::RegTensor<bfloat16_t> &)scaleForMulFP16, xTmp0, dataMaskB16);
                MAPI::Mul(xTmp1, (MAPI::RegTensor<bfloat16_t> &)scaleForMulFP16, xTmp1, dataMaskB16);
                MAPI::Interleave(xTmp0, xTmp1, xTmp0, xTmp1);
                MAPI::Cast<fp4x2_e2m1_t, bfloat16_t, kCastBF16ToFp4>(y0FP4, xTmp0, dataMaskB16);
                MAPI::Cast<fp4x2_e2m1_t, bfloat16_t, kCastBF16ToFp4>(y1FP4, xTmp1, dataMaskB16);
            }

            MAPI::DataCopy<uint8_t, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::StoreDist::DIST_PACK4_B32>(
                yWrite, (MAPI::RegTensor<uint8_t> &)y0FP4, 64, dataMaskB8);
            MAPI::DataCopy<uint8_t, MAPI::PostLiteral::POST_MODE_UPDATE, MAPI::StoreDist::DIST_PACK4_B32>(
                yWrite, (MAPI::RegTensor<uint8_t> &)y1FP4, 64, dataMaskB8);
        }
    }

    // -----------------------------------------------------------------------
    // ComputeFP4FromHalf (verbatim copy)
    // -----------------------------------------------------------------------
    template <class RegTensorFloat>
    __simd_callee__ inline void ComputeFP4FromHalf(RegTensorFloat &Reg)
    {
        namespace MAPI = AscendC::MicroAPI;

        MAPI::MaskReg pregAll32 = MAPI::CreateMask<uint32_t, MAPI::MaskPattern::ALL>();
        MAPI::MaskReg zeroMask, specialMask, negInfMask;

        MAPI::RegTensor<int32_t> negZero, maxExpFP32, exp0FP32, exp1FP32;
        MAPI::Duplicate(negZero, NEG_ZERO_I32);

        MAPI::Compare<int32_t, AscendC::CMPMODE::EQ>(negInfMask, (MAPI::RegTensor<int32_t> &)Reg, negZero, pregAll32);

        MAPI::Duplicate(maxExpFP32, static_cast<int32_t>(MAX_EXP_FOR_FP32));
        MAPI::And(exp0FP32, (MAPI::RegTensor<int32_t> &)Reg, maxExpFP32, pregAll32);
        MAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FP32, pregAll32);
        MAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG_VAL, pregAll32);
        MAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        MAPI::Adds(exp0FP32, exp0FP32, NEG_ONE_I32, pregAll32);
        MAPI::Muls(exp1FP32, exp0FP32, NEG_ONE_I32, pregAll32);
        MAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS_VAL, pregAll32);
        MAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FP32, pregAll32);

        MAPI::Mul(Reg, Reg, (MAPI::RegTensor<float> &)exp1FP32, pregAll32);
        MAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_VAL, pregAll32);
        MAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FP32, pregAll32);
        MAPI::CompareScalar<float, AscendC::CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MAPI::Truncate<float, AscendC::RoundMode::CAST_RINT>(Reg, Reg, pregAll32);
        MAPI::Mul(Reg, Reg, (MAPI::RegTensor<float> &)exp0FP32, pregAll32);

        MAPI::CompareScalar<float, AscendC::CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
        MAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
        MAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
        MAPI::Select<int32_t>((MAPI::RegTensor<int32_t> &)Reg, negZero, (MAPI::RegTensor<int32_t> &)Reg, zeroMask);
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    AscendC::LocalTensor<ElementInput> ubInput;
    AscendC::LocalTensor<uint8_t> ubFp4Out;
    AscendC::LocalTensor<float> ubScale1;
    AscendC::LocalTensor<uint8_t> ubScale2;
    AscendC::LocalTensor<uint16_t> ubReciprocal;

    CopyGmToUbInput copyGmToUbInput;
    CopyUbToGmOutput copyUbToGmOutput;
    CopyUbToGmScale1 copyUbToGmScale1;
    CopyUbToGmScale2 copyUbToGmScale2;
};

}  // namespace Catlass::Epilogue::Block

#undef CATLASS_DUAL_LEVEL_QUANT_MX_CAST_TRAIT

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_DUAL_LEVEL_QUANT_MX_HPP
