/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "sgl_operator_config.h"
#include "../op_kernel/swiglu_group_quant_tiling_data.h"
#include <algorithm>
#include <cstring>

namespace optiling {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t STATIC_QUANT = 1;
constexpr int64_t MX_QUANT = 2;
constexpr int64_t FP8_QUANT = 3;
constexpr int64_t CACHE_LINE_SIZE = 128;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 0;
constexpr int64_t ACTIVATE_SPLIT = 2;
constexpr int64_t FP8_DIM_ALIGN = 256;

static int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
static int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
static int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

struct SwigluGroupQuantInputs {
    int64_t bs = 0;  // product of x's dims except the last
    int64_t d = 0;   // x's last dim
    bool hasTopkWeight = false;
    int64_t g = 0;  // element count of group_index
    bool hasGroupIndex = false;
    int64_t quantMode = STATIC_QUANT;
    int64_t roundScale = 0;
    int64_t ue8m0Scale = 0;
    int64_t outputOrigin = 0;
    int64_t groupListType = GROUP_LIST_TYPE_COUNT;
    float clampValue = 0.0f;
    int64_t hasClampValue = 0;
};

//! Computes the row / d / group tiling factors, then the launch geometry.
class SwigluGroupQuantTiling
{
public:
    explicit SwigluGroupQuantTiling(const SwigluGroupQuantInputs &inputs, platform_ascendc::PlatformAscendC &platform)
        : inputs_(inputs), platform_(platform)
    {}

    bool DoOpTiling(SwigluGroupQuantTilingData &out, uint32_t &tilingKey)
    {
        if (!GetPlatformInfo()) {
            return false;
        }
        if (!GetShapeAttrsInfoInner()) {
            return false;
        }
        if (!CalcOpTiling()) {
            return false;
        }
        SetTilingData();

        tilingKey = tilingKey_;
        usedCoreNums_ = inputs_.hasGroupIndex ? coreNum_ : usedCoreNums_;
        out = tilingData_;
        return true;
    }

    int64_t UsedCoreNums() const
    {
        return usedCoreNums_;
    }

private:
    bool GetPlatformInfo()
    {
        coreNum_ = static_cast<int64_t>(platform_.GetCoreNumAiv());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
        uint64_t ubSizePlatForm = 0;
        platform_.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
        return true;
    }

    void GetAttr()
    {
        quantMode_ = inputs_.quantMode;
        splitFactor_ = quantMode_ == MX_QUANT ? PER_MX_FP16 : PER_BLOCK_FP16;
        roundScale_ = inputs_.roundScale;
        ue8m0Scale_ = inputs_.ue8m0Scale;
        outputOrigin_ = inputs_.outputOrigin;
        groupListType_ = inputs_.groupListType;
        clampValue_ = inputs_.clampValue;
        hasClampValue_ = inputs_.hasClampValue;
    }

    bool GetShapeAttrsInfoInner()
    {
        bs_ = inputs_.bs;
        d_ = inputs_.d;
        if (d_ % ACTIVATE_SPLIT != 0) {
            return false;
        }
        hasTopkWeight_ = inputs_.hasTopkWeight;
        hasGroupIndex_ = inputs_.hasGroupIndex;
        g_ = inputs_.g;

        GetAttr();

        if (quantMode_ != STATIC_QUANT && quantMode_ != MX_QUANT && quantMode_ != FP8_QUANT) {
            return false;
        }
        if (quantMode_ == FP8_QUANT && d_ % FP8_DIM_ALIGN != 0) {
            return false;
        }
        if (groupListType_ != GROUP_LIST_TYPE_COUNT) {
            return false;
        }

        splitD_ = d_ / ACTIVATE_SPLIT;
        scaleCol_ = CeilDiv(splitD_, splitFactor_);
        return true;
    }

    void CalcGroupIndexTiling()
    {
        if (hasGroupIndex_ && groupListType_ == GROUP_LIST_TYPE_COUNT) {
            gFactor_ = g_;
            int64_t groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / static_cast<int64_t>(sizeof(int64_t))) *
                                     DOUBLE_BUFFER * static_cast<int64_t>(sizeof(int64_t));
            int64_t groupIndexSumSize = BLOCK_SIZE;
            if (groupIndexSize + groupIndexSumSize <= ubSize_) {
                gLoop_ = 1;
                tailGFactor_ = gFactor_;
            } else {
                int64_t base = 2;
                while (1) {
                    gFactor_ = CeilDiv(g_, base);
                    groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / static_cast<int64_t>(sizeof(int64_t))) *
                                     DOUBLE_BUFFER * static_cast<int64_t>(sizeof(int64_t));
                    if (groupIndexSize + groupIndexSumSize < ubSize_) {
                        break;
                    }
                    base++;
                }
                if (gFactor_ > CACHE_LINE_SIZE / static_cast<int64_t>(sizeof(int64_t))) {
                    gFactor_ = DownAlign(gFactor_, CACHE_LINE_SIZE / static_cast<int64_t>(sizeof(int64_t)));
                }
                gLoop_ = CeilDiv(g_, gFactor_);
                tailGFactor_ = g_ % gFactor_ == 0 ? gFactor_ : g_ % gFactor_;
            }
        }
    }

    //! Shared head of the three mode-specific tilings: split bs_ across cores, start with one row.
    void CalcRowTilingHead(int64_t &rowOnceLoop)
    {
        rowOfFormerBlock_ = CeilDiv(bs_, coreNum_);
        usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), coreNum_);
        rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

        const int64_t minRowPerCore = 1;
        rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
        rowFactor_ = rowOnceLoop;
    }

    void CalcRowTilingTail()
    {
        rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
        rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
        tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
        tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
    }

    int64_t CalcColumnUbSize(int64_t rows, int64_t columns) const
    {
        const int64_t localScaleCols = CeilDiv(columns, splitFactor_);
        const int64_t inputSize = rows * RoundUp(columns, 16) * 2 * DOUBLE_BUFFER;
        const int64_t outputSize = rows * RoundUp(columns, 32) * DOUBLE_BUFFER;
        int64_t totalSize = 2 * inputSize + outputSize;

        if (quantMode_ == MX_QUANT) {
            const int64_t swigluSize = rows * RoundUp(columns, 16) * 2;
            const int64_t exponentSize = rows * RoundUp(localScaleCols, 16) * 2;
            const int64_t scaleSize = rows * RoundUp(localScaleCols, 32) * DOUBLE_BUFFER;
            totalSize += swigluSize + 2 * exponentSize + scaleSize;
        } else {
            const bool scaleE8m0 = quantMode_ == FP8_QUANT && ue8m0Scale_;
            totalSize += scaleE8m0 ? RoundUp(rows * localScaleCols, 32) * DOUBLE_BUFFER
                                   : RoundUp(rows * localScaleCols, 8) * 4 * DOUBLE_BUFFER;
        }
        if (hasTopkWeight_) {
            totalSize += RoundUp(rows, 8) * 4 * DOUBLE_BUFFER;
        }
        if (quantMode_ == FP8_QUANT && outputOrigin_) {
            totalSize += rows * RoundUp(columns, 16) * 2 * DOUBLE_BUFFER;
        }
        return totalSize;
    }

    bool CalcColumnTiling(int64_t rows)
    {
        const int64_t alignment = quantMode_ == MX_QUANT ? 2 * PER_MX_FP16 : PER_BLOCK_FP16;
        int64_t bestFactor = 0;
        for (int64_t candidate = alignment; candidate <= splitD_; candidate += alignment) {
            if (CalcColumnUbSize(rows, candidate) > ubSize_) {
                break;
            }
            bestFactor = candidate;
        }
        if (bestFactor == 0) {
            return false;
        }
        dFactor_ = bestFactor;
        dLoop_ = CeilDiv(splitD_, dFactor_);
        tailDFactor_ = splitD_ - (dLoop_ - 1) * dFactor_;
        return true;
    }

    bool CalcMxQuantOpTiling()
    {
        int64_t rowOnceLoop = 0;
        CalcRowTilingHead(rowOnceLoop);

        int64_t x0Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        int64_t x1Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        int64_t swigluSize = rowOnceLoop * RoundUp(splitD_, 16) * 2;
        int64_t maxExpSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
        int64_t invScaleSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
        int64_t ySize = rowOnceLoop * RoundUp(splitD_, 32) * 1 * DOUBLE_BUFFER;
        int64_t scaleSize = rowOnceLoop * RoundUp(scaleCol_, 32) * 1 * DOUBLE_BUFFER;

        int64_t totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;

        int64_t topkWeightSize = RoundUp(rowOnceLoop, 8) * 4 * DOUBLE_BUFFER;
        totalSize = hasTopkWeight_ ? totalSize + topkWeightSize : totalSize;

        if (totalSize <= ubSize_) {
            // row and d both fit in UB at once
            dLoop_ = 1;
            dFactor_ = splitD_;
            tailDFactor_ = dFactor_;
        } else if (!CalcColumnTiling(rowOnceLoop)) {
            return false;
        }

        // d fits whole: try to pull in more rows
        if (dFactor_ == splitD_) {
            while (rowFactor_ <= rowOfFormerBlock_) {
                x0Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                x1Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                swigluSize = rowFactor_ * RoundUp(dFactor_, 16) * 2;
                maxExpSize = rowFactor_ * RoundUp(scaleCol_, 16) * 2;
                invScaleSize = rowFactor_ * RoundUp(scaleCol_, 16) * 2;
                ySize = rowFactor_ * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
                scaleSize = rowFactor_ * RoundUp(scaleCol_, 32) * 1 * DOUBLE_BUFFER;
                totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;
                if (hasTopkWeight_) {
                    topkWeightSize = RoundUp(rowFactor_, 8) * 4 * DOUBLE_BUFFER;
                    totalSize += topkWeightSize;
                }
                if (totalSize > ubSize_) {
                    rowFactor_ = rowFactor_ - 1;
                    break;
                }
                rowFactor_ = rowFactor_ + 1;
            }
            rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
        }

        CalcRowTilingTail();
        return true;
    }

    bool CalcGroupQuantOpTiling()
    {
        int64_t rowOnceLoop = 0;
        CalcRowTilingHead(rowOnceLoop);

        int64_t x0Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        int64_t x1Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        int64_t ySize = rowOnceLoop * RoundUp(splitD_, 32) * 1 * DOUBLE_BUFFER;
        int64_t scaleSize = RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;

        int64_t totalSize = x0Size + x1Size + ySize + scaleSize;

        int64_t topkWeightSize = RoundUp(rowOnceLoop, 8) * 4 * DOUBLE_BUFFER;
        totalSize = hasTopkWeight_ ? totalSize + topkWeightSize : totalSize;

        if (totalSize <= ubSize_) {
            // row and d both fit in UB at once
            dLoop_ = 1;
            dFactor_ = splitD_;
            tailDFactor_ = dFactor_;
        } else if (!CalcColumnTiling(rowOnceLoop)) {
            return false;
        }

        // d fits whole: try to pull in more rows
        if (dFactor_ == splitD_) {
            while (rowFactor_ <= rowOfFormerBlock_) {
                x0Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                x1Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                ySize = rowFactor_ * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
                scaleSize = RoundUp(rowFactor_ * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
                totalSize = x0Size + x1Size + ySize + scaleSize;
                if (hasTopkWeight_) {
                    topkWeightSize = RoundUp(rowFactor_, 8) * 4 * DOUBLE_BUFFER;
                    totalSize += topkWeightSize;
                }
                if (totalSize > ubSize_) {
                    rowFactor_ = rowFactor_ - 1;
                    break;
                }
                rowFactor_ = rowFactor_ + 1;
            }
            rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
        }

        CalcRowTilingTail();
        return true;
    }

    bool CalcFp8QuantOpTiling()
    {
        int64_t rowOnceLoop = 0;
        CalcRowTilingHead(rowOnceLoop);

        int64_t x0Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        int64_t x1Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        int64_t ySize = rowOnceLoop * RoundUp(splitD_, 32) * 1 * DOUBLE_BUFFER;
        int64_t scaleSize = ue8m0Scale_ ? RoundUp(rowOnceLoop * scaleCol_, 32) * 1 * DOUBLE_BUFFER
                                        : RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;

        int64_t totalSize = x0Size + x1Size + ySize + scaleSize;

        int64_t topkWeightSize = RoundUp(rowOnceLoop, 8) * 4 * DOUBLE_BUFFER;
        totalSize = hasTopkWeight_ ? totalSize + topkWeightSize : totalSize;

        int64_t yOriginSize = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
        totalSize = outputOrigin_ ? totalSize + yOriginSize : totalSize;

        if (totalSize <= ubSize_) {
            // row and d both fit in UB at once
            dLoop_ = 1;
            dFactor_ = splitD_;
            tailDFactor_ = dFactor_;
        } else if (!CalcColumnTiling(rowOnceLoop)) {
            return false;
        }

        // d fits whole: try to pull in more rows
        if (dFactor_ == splitD_) {
            while (rowFactor_ <= rowOfFormerBlock_) {
                x0Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                x1Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                ySize = rowFactor_ * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
                scaleSize = RoundUp(rowFactor_ * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
                totalSize = x0Size + x1Size + ySize + scaleSize;
                if (hasTopkWeight_) {
                    topkWeightSize = RoundUp(rowFactor_, 8) * 4 * DOUBLE_BUFFER;
                    totalSize += topkWeightSize;
                }
                if (outputOrigin_) {
                    yOriginSize = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                    totalSize += yOriginSize;
                }
                if (totalSize > ubSize_) {
                    rowFactor_ = rowFactor_ - 1;
                    break;
                }
                rowFactor_ = rowFactor_ + 1;
            }
            rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
        }

        CalcRowTilingTail();
        return true;
    }

    bool CalcOpTiling()
    {
        CalcGroupIndexTiling();
        if (quantMode_ == STATIC_QUANT) {
            return CalcGroupQuantOpTiling();
        } else if (quantMode_ == MX_QUANT) {
            return CalcMxQuantOpTiling();
        }
        return CalcFp8QuantOpTiling();
    }

    void SetTilingData()
    {
        tilingData_.bs = bs_;
        tilingData_.d = d_;
        tilingData_.splitD = splitD_;
        tilingData_.scaleCol = scaleCol_;
        tilingData_.rowOfFormerBlock = rowOfFormerBlock_;
        tilingData_.rowOfTailBlock = rowOfTailBlock_;
        tilingData_.rowLoopOfFormerBlock = rowLoopOfFormerBlock_;
        tilingData_.rowLoopOfTailBlock = rowLoopOfTailBlock_;
        tilingData_.rowFactor = rowFactor_;
        tilingData_.tailRowFactorOfFormerBlock = tailRowFactorOfFormerBlock_;
        tilingData_.tailRowFactorOfTailBlock = tailRowFactorOfTailBlock_;
        tilingData_.dLoop = dLoop_;
        tilingData_.dFactor = dFactor_;
        tilingData_.tailDFactor = tailDFactor_;
        tilingData_.roundScale = roundScale_;
        tilingData_.ue8m0Scale = ue8m0Scale_;
        tilingData_.outputOrigin = outputOrigin_;
        tilingData_.clampValue = clampValue_;
        tilingData_.hasClampValue = hasClampValue_;
        tilingData_.g = g_;
        tilingData_.ubSize = ubSize_;
        tilingData_.gLoop = gLoop_;
        tilingData_.gFactor = gFactor_;
        tilingData_.tailGFactor = tailGFactor_;
        tilingData_.groupListType = groupListType_;
        tilingData_.coreNum = coreNum_;

        // Tail of the tiling: map the quant mode onto the kernel's tiling key.
        if (quantMode_ == STATIC_QUANT) {
            tilingKey_ = SWIGLU_GROUP_QUANT_TILING_KEY_GROUP_QUANT;
        } else if (quantMode_ == MX_QUANT) {
            tilingKey_ = SWIGLU_GROUP_QUANT_TILING_KEY_MX_QUANT;
        } else {
            tilingKey_ = outputOrigin_ ? SWIGLU_GROUP_QUANT_TILING_KEY_FP8_QUANT_YORIGIN
                                       : SWIGLU_GROUP_QUANT_TILING_KEY_FP8_QUANT;
        }
    }

    const SwigluGroupQuantInputs &inputs_;
    platform_ascendc::PlatformAscendC &platform_;
    SwigluGroupQuantTilingData tilingData_{};
    uint32_t tilingKey_ = 0;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t usedCoreNums_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t splitD_ = 0;
    int64_t scaleCol_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
    int64_t quantMode_ = STATIC_QUANT;
    int64_t splitFactor_ = PER_BLOCK_FP16;
    int64_t roundScale_ = 0;
    int64_t ue8m0Scale_ = 0;
    int64_t outputOrigin_ = 0;
    float clampValue_ = 0.0f;
    int64_t hasClampValue_ = 0;
    int64_t g_ = 0;
    int64_t gLoop_ = 0;
    int64_t gFactor_ = 0;
    int64_t tailGFactor_ = 0;
    int64_t groupListType_ = GROUP_LIST_TYPE_COUNT;
    bool hasTopkWeight_ = false;
    bool hasGroupIndex_ = false;
};

static ge::graphStatus TilingSwigluGroupQuant(gert::TilingContext *ctx)
{
    const auto *shape = ctx->GetInputShape(0);
    if (!shape || !ctx->GetPlatformInfo() || !ctx->GetAttrs()) return ge::GRAPH_FAILED;
    const auto &x = shape->GetStorageShape();
    if (x.GetDimNum() == 0) return ge::GRAPH_FAILED;
    const auto *attrs = ctx->GetAttrs();
    SwigluGroupQuantInputs inputs;
    inputs.d = x.GetDim(x.GetDimNum() - 1);
    inputs.bs = 1;
    for (size_t i = 0; i + 1 < x.GetDimNum(); ++i)
        inputs.bs *= x.GetDim(i);
    inputs.hasTopkWeight = ctx->GetOptionalInputShape(1) != nullptr;
    const auto *group = ctx->GetOptionalInputShape(2);
    inputs.hasGroupIndex = group != nullptr;
    inputs.g = group ? group->GetStorageShape().GetShapeSize() : 0;
    inputs.quantMode = *attrs->GetInt(0);
    inputs.roundScale = *attrs->GetBool(1);
    inputs.ue8m0Scale = *attrs->GetBool(2);
    inputs.outputOrigin = *attrs->GetBool(3);
    inputs.groupListType = *attrs->GetInt(4);
    inputs.clampValue = *attrs->GetFloat(5);
    inputs.hasClampValue = *attrs->GetBool(7);
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    SwigluGroupQuantTiling tiling(inputs, platform);
    SwigluGroupQuantTilingData data{};
    uint32_t key = 0;
    if (!tiling.DoOpTiling(data, key)) return ge::GRAPH_FAILED;
    data.tilingKey = key;
    if (ctx->GetInputDesc(0)->GetDataType() == ge::DT_BF16) data.dtype |= SWIGLU_GROUP_QUANT_DTYPE_X_BF16;
    if (*attrs->GetBool(6)) data.dtype |= SWIGLU_GROUP_QUANT_DTYPE_Y_E5M2;
    if (inputs.quantMode == MX_QUANT || (inputs.quantMode == FP8_QUANT && inputs.ue8m0Scale))
        data.dtype |= SWIGLU_GROUP_QUANT_DTYPE_SCALE_E8M0;
    auto *raw = ctx->GetRawTilingData();
    if (!raw || raw->GetCapacity() < sizeof(data)) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    ctx->SetBlockDim(tiling.UsedCoreNums());
    ctx->SetTilingKey(0);
    auto *workspace = ctx->GetWorkspaceSizes(1);
    if (!workspace) return ge::GRAPH_FAILED;
    workspace[0] = platform.GetLibApiWorkSpaceSize() + 32;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeSwigluGroupQuant(gert::InferShapeContext *ctx)
{
    const auto &x = *ctx->GetInputShape(0);
    const auto last = x.GetDimNum() - 1;
    const auto width = x.GetDim(last) / 2;
    auto &y = *ctx->GetOutputShape(0);
    y = x;
    y.SetDim(last, width);
    *ctx->GetOutputShape(2) = y;
    auto &scale = *ctx->GetOutputShape(1);
    scale = y;
    if (*ctx->GetAttrs()->GetInt(0) == MX_QUANT) {
        scale.SetDim(last, CeilDiv(width, 64));
        scale.AppendDim(2);
    } else {
        scale.SetDim(last, CeilDiv(width, 128));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwigluGroupQuant(gert::InferDataTypeContext *ctx)
{
    const auto *attrs = ctx->GetAttrs();
    auto mode = *attrs->GetInt(0);
    ctx->SetOutputDataType(0, *attrs->GetBool(6) ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN);
    ctx->SetOutputDataType(
        1, mode == MX_QUANT || (mode == FP8_QUANT && *attrs->GetBool(2)) ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT);
    ctx->SetOutputDataType(2, ctx->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class SglSwigluGroupQuant : public OpDef
{
public:
    explicit SglSwigluGroupQuant(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::Format> formatND = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format(formatND)
            .UnknownShapeFormat(formatND);
        this->Input("topk_weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format(formatND)
            .UnknownShapeFormat(formatND);
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64})
            .Format(formatND)
            .UnknownShapeFormat(formatND);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})
            .Format(formatND)
            .UnknownShapeFormat(formatND);
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0})
            .Format(formatND)
            .UnknownShapeFormat(formatND);
        this->Output("y_origin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format(formatND)
            .UnknownShapeFormat(formatND);
        this->Attr("quant_mode").AttrType(REQUIRED).Int();
        this->Attr("round_scale").AttrType(REQUIRED).Bool();
        this->Attr("ue8m0_scale").AttrType(REQUIRED).Bool();
        this->Attr("output_origin").AttrType(REQUIRED).Bool();
        this->Attr("group_list_type").AttrType(REQUIRED).Int();
        this->Attr("clamp_value").AttrType(REQUIRED).Float();
        this->Attr("y_e5m2").AttrType(REQUIRED).Bool();
        this->Attr("has_clamp_value").AttrType(REQUIRED).Bool();
        this->SetInferShape(optiling::InferShapeSwigluGroupQuant)
            .SetInferDataType(optiling::InferDataTypeSwigluGroupQuant);
        this->AICore().SetTiling(optiling::TilingSwigluGroupQuant);
        for (auto config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglSwigluGroupQuant);
}  // namespace ops
