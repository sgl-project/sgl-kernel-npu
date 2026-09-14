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
 * \file swiglu_group_quant.cpp
 * \brief Host-side tiling + launch for swiglu_group_quant (A5-only).
 *
 * Each quant mode has its own tiling routine below; all three share a head that splits the batch
 * across cores. The result is written into a packed struct that is memcpy'd to the device, and the
 * kernel entry dispatches on the tiling key and dtype that struct carries.
 *
 * Two notes on the parameter surface:
 *   - `group_size` and `dst_type` do not participate in the tiling math. They are accepted for
 *     caller compatibility and are otherwise inert.
 *   - The UB-fitting loops take the form `while (totalSize < ubSize_)`, which only terminates early
 *     when the first probe already fits.
 */

#include <cstring>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/swiglu_group_quant_tiling_data.h"
#include "defines.h"
#include "torch_helper.h"
#include "common_tiling.h"
#include "common.h"
#include "aclrtlaunch_swiglu_group_quant.h"

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr int64_t WORKSPACE_SIZE = 32;
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
constexpr int64_t FP8_MAX_SCALE_COL_DIVISOR = 128;

// Graph mode tiling cache, mirroring kv_compress_epilog: the tiling buffer must be stable across a
// captured graph, so each distinct tiling lands in a fixed slot of a persistent device buffer
// instead of being re-uploaded (a host->device copy inside capture is not replayable).
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
uint32_t actualCaptureNum = 0;
std::unordered_map<uint64_t, uint32_t> captureMap;

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y)
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
    explicit SwigluGroupQuantTiling(const SwigluGroupQuantInputs &inputs) : inputs_(inputs) {}

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
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        coreNum_ = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
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
        } else {
            dFactor_ = splitD_;
            int64_t base = 1;
            while (totalSize < ubSize_) {
                dFactor_ = base * splitFactor_;
                scaleCol_ = CeilDiv(dFactor_, splitFactor_);
                x0Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                x1Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                swigluSize = rowOnceLoop * RoundUp(dFactor_, 16) * 2;
                maxExpSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
                invScaleSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
                ySize = rowOnceLoop * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
                scaleSize = rowOnceLoop * RoundUp(scaleCol_, 32) * 1 * DOUBLE_BUFFER;
                totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;
                if (hasTopkWeight_) {
                    totalSize += topkWeightSize;
                }
                base++;
            }
            dFactor_ = (base - 1) * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            dLoop_ = CeilDiv(splitD_, dFactor_);
            tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
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
        } else {
            dFactor_ = splitD_;
            int64_t base = 1;
            while (totalSize < ubSize_) {
                dFactor_ = base * splitFactor_;
                scaleCol_ = CeilDiv(dFactor_, splitFactor_);
                x0Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                x1Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                ySize = rowOnceLoop * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
                scaleSize = RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
                totalSize = x0Size + x1Size + ySize + scaleSize;
                if (hasTopkWeight_) {
                    totalSize += topkWeightSize;
                }
                base++;
            }
            dFactor_ = (base - 1) * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            dLoop_ = CeilDiv(splitD_, dFactor_);
            tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
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
        } else {
            dFactor_ = splitD_;
            int64_t base = 1;
            while (totalSize < ubSize_) {
                dFactor_ = base * splitFactor_;
                scaleCol_ = CeilDiv(dFactor_, splitFactor_);
                x0Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                x1Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                ySize = rowOnceLoop * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
                scaleSize = RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
                totalSize = x0Size + x1Size + ySize + scaleSize;
                if (hasTopkWeight_) {
                    totalSize += topkWeightSize;
                }
                if (outputOrigin_) {
                    yOriginSize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                    totalSize += yOriginSize;
                }
                base++;
            }
            dFactor_ = (base - 1) * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            dLoop_ = CeilDiv(splitD_, dFactor_);
            tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
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

//! Allocates y / scale / y_origin with the shape and dtype each quant mode calls for.
std::tuple<at::Tensor, at::Tensor, at::Tensor> ConstructOutputs(const at::Tensor &x, at::ScalarType dst_type,
                                                                int64_t quant_mode, bool ue8m0_scale)
{
    constexpr int64_t SWIGLU_FACTOR = 2;
    constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;

    std::vector<int64_t> y_size(x.sizes().begin(), x.sizes().end());
    int64_t y_last_dim = y_size.back() / SWIGLU_FACTOR;
    y_size.back() = y_last_dim;

    const auto y_dtype = dst_type == at::kFloat8_e5m2 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));

    std::vector<int64_t> scale_size(y_size.begin(), y_size.end());
    if (quant_mode == MX_QUANT) {
        int64_t scale_last_dim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_last_dim = (scale_last_dim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_size.back() = scale_last_dim;
        scale_size.push_back(MX_SCALE_ALIGN_FACTOR);
    } else {
        scale_size.back() = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
    }

    auto scale_type = at::kFloat;
    if (quant_mode == MX_QUANT || (quant_mode == FP8_QUANT && ue8m0_scale)) {
        scale_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale = at::empty(scale_size, x.options().dtype(scale_type));
    at::Tensor y_origin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::make_tuple(y, scale, y_origin);
}

}  // namespace

HOST_API std::tuple<at::Tensor, at::Tensor, at::Tensor> swiglu_group_quant(
    const at::Tensor &x, const c10::optional<at::Tensor> &topk_weight, const c10::optional<at::Tensor> &group_index,
    c10::optional<at::ScalarType> dst_type, int64_t quant_mode, int64_t group_size, bool round_scale, bool ue8m0_scale,
    bool output_origin, int64_t group_list_type, double clamp_value)
{
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension, but got ", x.dim());
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x should be FLOAT16 or BFLOAT16, but got ", x.scalar_type());
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(quant_mode == STATIC_QUANT || quant_mode == MX_QUANT || quant_mode == FP8_QUANT,
                "Unsupported quant mode, only support ", STATIC_QUANT, " (group), ", MX_QUANT, " (mx) or ", FP8_QUANT,
                " (fp8), but got ", quant_mode);

    const at::ScalarType out_dtype = dst_type.value_or(at::kFloat8_e4m3fn);
    TORCH_CHECK(out_dtype == at::kFloat8_e4m3fn || out_dtype == at::kFloat8_e5m2,
                "dst_type must be Float8_e4m3fn or Float8_e5m2, but got ", out_dtype);

    const int64_t d = x.size(-1);
    if (quant_mode == STATIC_QUANT || quant_mode == FP8_QUANT) {
        TORCH_CHECK(d % FP8_DIM_ALIGN == 0, "In group/fp8 quant, the last dim of x should be divisible by ",
                    FP8_DIM_ALIGN, ", actual ", d, ".");
    } else {
        TORCH_CHECK(d % FP8_MAX_SCALE_COL_DIVISOR == 0, "In mx quant, the last dim of x should be divisible by ",
                    FP8_MAX_SCALE_COL_DIVISOR, ", actual ", d, ".");
    }
    TORCH_CHECK(d % ACTIVATE_SPLIT == 0, "In swiglu, the last dim of x should be divisible by ", ACTIVATE_SPLIT,
                ", actual ", d, ".");

    // Optional inputs are detected by the kernel as a null GM pointer (see the `!= nullptr` tests in
    // swiglu_group_quant_perf.h Init), so an absent optional must be passed as a real nullptr rather
    // than an empty tensor's data_ptr. The host's tiling must agree with the kernel on presence, so
    // both sides use "present and non-empty" here.
    const bool has_topk_weight = topk_weight.has_value() && topk_weight->defined() && topk_weight->numel() > 0;
    const bool has_group_index = group_index.has_value() && group_index->defined() && group_index->numel() > 0;

    SwigluGroupQuantInputs inputs;
    inputs.bs = 1;
    for (int64_t i = 0; i + 1 < x.dim(); ++i) {
        inputs.bs *= x.size(i);
    }
    inputs.d = d;
    inputs.hasTopkWeight = has_topk_weight;
    inputs.hasGroupIndex = has_group_index;
    inputs.g = has_group_index ? group_index->numel() : 0;
    inputs.quantMode = quant_mode;
    inputs.roundScale = round_scale ? 1 : 0;
    inputs.ue8m0Scale = ue8m0_scale ? 1 : 0;
    inputs.outputOrigin = output_origin ? 1 : 0;
    inputs.groupListType = group_list_type;
    inputs.clampValue = static_cast<float>(clamp_value);
    inputs.hasClampValue = clamp_value != 0.0 ? 1 : 0;

    auto outputs = ConstructOutputs(x, out_dtype, quant_mode, ue8m0_scale);
    at::Tensor y = std::get<0>(outputs);
    at::Tensor scale = std::get<1>(outputs);
    at::Tensor y_origin = std::get<2>(outputs);

    // An empty batch has nothing to quantize, and the row tiling below would divide by a zero row
    // count. Returning the empty outputs is both correct and a no-op for the kernel.
    if (inputs.bs == 0) {
        return outputs;
    }

    SwigluGroupQuantTilingData tilingData{};
    uint32_t tilingKey = 0;
    SwigluGroupQuantTiling tiling(inputs);
    TORCH_CHECK(tiling.DoOpTiling(tilingData, tilingKey), "swiglu_group_quant: tiling failed for x of shape ",
                x.sizes(), ", quant_mode=", quant_mode);
    tilingData.tilingKey = tilingKey;

    // dtype code bits, mirroring the dispatch in the kernel entry.
    int32_t dtypeCode = 0;
    if (x.scalar_type() == at::kBFloat16) {
        dtypeCode |= SWIGLU_GROUP_QUANT_DTYPE_X_BF16;
    }
    if (y.scalar_type() == at::kFloat8_e5m2) {
        dtypeCode |= SWIGLU_GROUP_QUANT_DTYPE_Y_E5M2;
    }
    if (scale.scalar_type() == at::kFloat8_e8m0fnu) {
        dtypeCode |= SWIGLU_GROUP_QUANT_DTYPE_SCALE_E8M0;
    }
    tilingData.dtype = dtypeCode;

    // Absent optionals must reach the kernel as null, and only the presence the tiling agreed on.
    void *topk_weight_ptr = has_topk_weight ? topk_weight->data_ptr() : nullptr;
    void *group_index_ptr = has_group_index ? group_index->data_ptr() : nullptr;

    const uint32_t tilingSize = (sizeof(SwigluGroupQuantTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    at::Tensor tilingTensor;

    auto tup = std::make_tuple(inputs.bs, inputs.d, tilingData.splitD, tilingData.scaleCol, tilingData.rowOfFormerBlock,
                               tilingData.rowOfTailBlock, tilingData.rowLoopOfFormerBlock,
                               tilingData.rowLoopOfTailBlock, tilingData.rowFactor,
                               tilingData.tailRowFactorOfFormerBlock, tilingData.tailRowFactorOfTailBlock,
                               tilingData.dLoop, tilingData.dFactor, tilingData.tailDFactor, tilingData.roundScale,
                               tilingData.ue8m0Scale, tilingData.outputOrigin, tilingData.hasClampValue, tilingData.g,
                               tilingData.ubSize, tilingData.gLoop, tilingData.gFactor, tilingData.tailGFactor,
                               tilingData.groupListType, tilingData.coreNum, tilingData.dtype, tilingKey);
    uint64_t hashValue = host_utils::TupleHasher::Hash(tup);

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tilingData, sizeof(SwigluGroupQuantTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    static auto globalTilingBuffer = at::empty({static_cast<int64_t>(tilingSize) * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(x.device()));

    if (captureMap.find(hashValue) != captureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        captureMap[hashValue] = actualCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer
            .slice(0, static_cast<int64_t>(actualCaptureNum) * tilingSize,
                   static_cast<int64_t>(actualCaptureNum) * tilingSize + tilingSize)
            .copy_(deviceTiling);
        actualCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    auto workspace_tensor = at::empty({WORKSPACE_SIZE}, at::TensorOptions().dtype(at::kByte).device(x.device()));

    // The kernel reads GetBlockNum() for its used-core count when there is no group index, so the
    // launch dim must be coreNum when a group index is present (each core walks the whole group
    // list), else the reduced used-core count.
    const int64_t blockDim = inputs.hasGroupIndex ? tilingData.coreNum : tiling.UsedCoreNums();
    EXEC_KERNEL_CMD(swiglu_group_quant, blockDim, x, topk_weight_ptr, group_index_ptr, y, scale, y_origin,
                    workspace_tensor, tilingTensor);

    return std::make_tuple(y, scale, y_origin);
}

}  // namespace npu_kernel
}  // namespace sglang
