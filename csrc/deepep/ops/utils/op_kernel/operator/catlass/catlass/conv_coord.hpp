/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_COORD_HPP
#define CATLASS_CONV_COORD_HPP

#include "catlass/coord.hpp"

namespace Catlass {

/// Shape of conv3d operation
struct Conv3dParams {
public:
    typedef uint32_t Index;
    static constexpr uint32_t N0 = 16;
    using Fmap6HDShape = Coord<6, Index>;   // {batch, di, cin1, hi, wi, cin0}
    using FilterFracZ3DShape = Coord<7, Index>; // {kd, cin1, kh, kw, n1, n0, cin0}
    using Out6HDShape = Coord<6, Index>;   // {batch, do, cout1, ho, wo, cout0}
    using Strides = Coord<3, Index>;
    using Pads = Coord<3, Index>;
    using Dilations = Coord<3, Index>;
private:
    Fmap6HDShape fmap6HDShape_;
    FilterFracZ3DShape filterFracZ3DShape_;
    Out6HDShape out6HDShape_;
    Strides strides_;
    Pads pads_;
    Dilations dilations_;
    Index cout_;
public:
    CATLASS_HOST_DEVICE
    Conv3dParams(
        Index BATCH = 1,
        Index Di = 1,
        Index Cin1 = 1,
        Index Hi = 1,
        Index Wi = 1,
        Index C0 = 16,
        Index Kd = 1,
        Index Kh = 1,
        Index Kw = 1,
        Index N1 = 1,
        Index Do = 1,
        Index Ho = 1,
        Index Wo = 1,
        Index Cout1 = 1,
        Index Cout = 1,
        Index padHead = 0,
        Index padTop = 0,
        Index padLeft = 0,
        Index strideD = 1,
        Index strideH = 1,
        Index strideW = 1,
        Index dilationD = 1,
        Index dilationH = 1,
        Index dilationW = 1
    )
    : fmap6HDShape_(MakeCoord(BATCH, Di, Cin1, Hi, Wi, C0)),
      filterFracZ3DShape_(MakeCoord(Kd, Cin1, Kh, Kw, N1, N0, C0)),
      out6HDShape_(MakeCoord(BATCH, Do, Cout1, Ho, Wo, C0)),
      cout_(Cout),
      pads_(MakeCoord(padHead, padTop, padLeft)),
      strides_(MakeCoord(strideD, strideH, strideW)),
      dilations_(MakeCoord(dilationD, dilationH, dilationW))
    {}

    CATLASS_HOST_DEVICE
    static Conv3dParams MakeConvCoord(
        const uint32_t* fmapShape,
        const uint32_t* filterShape,
        const uint32_t* paddings,
        const uint32_t* strides,
        const uint32_t* dilations)
    {
        return Conv3dParams(
            fmapShape[0],
            fmapShape[1],
            fmapShape[2],
            fmapShape[3],
            fmapShape[4],
            fmapShape[5],
            filterShape[0],
            filterShape[1],
            filterShape[2],
            CeilDiv(filterShape[3], N0),
            (fmapShape[1] + paddings[0] * 2 - dilations[0] * (filterShape[0] - 1) - 1) / strides[0] + 1,  //Do
            (fmapShape[3] + paddings[1] * 2 - dilations[1] * (filterShape[1] - 1) - 1) / strides[1] + 1,  //Ho
            (fmapShape[4] + paddings[2] * 2 - dilations[2] * (filterShape[2] - 1) - 1) / strides[2] + 1,  //Wo
            CeilDiv(filterShape[3], fmapShape[5]),
            filterShape[3],
            paddings[0],
            paddings[1],
            paddings[2],
            strides[0],
            strides[1],
            strides[2],
            dilations[0],
            dilations[1],
            dilations[2]
        );
    }
    

    //fmapShape
    CATLASS_HOST_DEVICE
    Index const &batch() const { return fmap6HDShape_[0]; }
    CATLASS_HOST_DEVICE
    Index const &cin1() const { return fmap6HDShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &di() const { return fmap6HDShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const &hi() const { return fmap6HDShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &wi() const { return fmap6HDShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &cin0() const { return fmap6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const hiwi() const { return fmap6HDShape_[3] * fmap6HDShape_[4]; }

    //filterShape
    CATLASS_HOST_DEVICE
    Index const &kd() const { return filterFracZ3DShape_[0]; }
    CATLASS_HOST_DEVICE
    Index const &kh() const { return filterFracZ3DShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &kw() const { return filterFracZ3DShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const khkw() const
    { 
        return filterFracZ3DShape_[2] * filterFracZ3DShape_[3];
    }
    CATLASS_HOST_DEVICE
    Index const kdc1khkw() const{
        return filterFracZ3DShape_[0] * filterFracZ3DShape_[1] * filterFracZ3DShape_[2] * filterFracZ3DShape_[3];
    }
    CATLASS_HOST_DEVICE
    Index const &n1() const { return filterFracZ3DShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &n0() const { return filterFracZ3DShape_[5]; }

    //outShape
    CATLASS_HOST_DEVICE
    Index const &dout() const { return out6HDShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const &ho() const { return out6HDShape_[3]; }
    CATLASS_HOST_DEVICE
    Index const &wo() const { return out6HDShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const &cout1() const { return out6HDShape_[2]; }
    CATLASS_HOST_DEVICE
    Index const &cout0() const { return out6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const &cout() const { return cout_; }

    /// paddings
    CATLASS_HOST_DEVICE
    Index const &padhead() const { return pads_[0]; }
    CATLASS_HOST_DEVICE
    Index const &padtail() const { return pads_[0]; }
    CATLASS_HOST_DEVICE
    Index const &padtop() const { return pads_[1]; }
    CATLASS_HOST_DEVICE
    Index const &padbottom() const { return pads_[1]; }
    CATLASS_HOST_DEVICE
    Index const &padleft() const { return pads_[2]; }
    CATLASS_HOST_DEVICE
    Index const &padright() const { return pads_[2]; }

    /// strideSize
    CATLASS_HOST_DEVICE
    Index const &sD() const { return strides_[0]; }
    CATLASS_HOST_DEVICE
    Index const &sH() const { return strides_[1]; }
    CATLASS_HOST_DEVICE
    Index const &sW() const { return strides_[2]; }

    /// dilationSize
    CATLASS_HOST_DEVICE
    Index const &dD() const { return dilations_[0]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelD() const { return 1 + (filterFracZ3DShape_[0] - 1) * dilations_[0]; }
    CATLASS_HOST_DEVICE
    Index const &dH() const { return dilations_[1]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelH() const { return 1 + (filterFracZ3DShape_[2] - 1) * dilations_[1]; }
    CATLASS_HOST_DEVICE
    Index const &dW() const { return dilations_[2]; }
    CATLASS_HOST_DEVICE
    Index const dilatedKernelW() const { return 1 + (filterFracZ3DShape_[3] - 1) * dilations_[2]; }
    
    ///// used in block
    CATLASS_HOST_DEVICE
    Index const howo() const { return out6HDShape_[3] * out6HDShape_[4]; }
    CATLASS_HOST_DEVICE
    Index const alignCout() const { return out6HDShape_[2] * out6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const wicin0() const { return fmap6HDShape_[4] * fmap6HDShape_[5]; }
    CATLASS_HOST_DEVICE
    Index const khkwcin0() const { return filterFracZ3DShape_[2] * filterFracZ3DShape_[3] * filterFracZ3DShape_[6]; }
    CATLASS_HOST_DEVICE
    Index const alignCinKhKwKd() const
    { 
        return filterFracZ3DShape_[0] * filterFracZ3DShape_[1] * filterFracZ3DShape_[2] * filterFracZ3DShape_[3] * filterFracZ3DShape_[6];
    }
    CATLASS_HOST_DEVICE
    Index const kdcin1() const { return filterFracZ3DShape_[0] * filterFracZ3DShape_[1]; }
    CATLASS_HOST_DEVICE
    Index const fmapOneBatchSize() const
    {
        return fmap6HDShape_[1] * fmap6HDShape_[2] * fmap6HDShape_[3] * fmap6HDShape_[4] * fmap6HDShape_[5];
    }
    CATLASS_HOST_DEVICE
    Index const outputOneBatchSize() const
    {
        return out6HDShape_[1] * out6HDShape_[2] * out6HDShape_[3] * out6HDShape_[4] * out6HDShape_[5];
    }
};

template <
    uint32_t noCnt_ = 1,
    uint32_t doCnt_ = 1,
    uint32_t co1Cnt_ = 1,
    uint32_t howoCnt_ = 1
>
struct ConvCoreShape {
    static uint32_t const noCnt = noCnt_;
    static uint32_t const doCnt = doCnt_;
    static uint32_t const co1Cnt = co1Cnt_;
    static uint32_t const howoCnt = howoCnt_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<4> ToCoord()
    {
        return MakeCoord(noCnt, doCnt, co1Cnt, howoCnt);
    }
};

template<
    uint32_t mAL1_ = 1,
    uint32_t Kd_ = 1,
    uint32_t Ci1_ = 1
>
struct ConvFmapL1Shape {
    static uint32_t constexpr mAL1 = mAL1_;
    static uint32_t constexpr Kd = Kd_;
    static uint32_t constexpr Ci1 = Ci1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(mAL1, Kd, Ci1);
    }
};

template<
    uint32_t Kd_ = 1,
    uint32_t Ci1_ = 1,
    uint32_t nBL1_ = 1
>
struct ConvFilterL1Shape {
    static uint32_t constexpr Kd = Kd_;
    static uint32_t constexpr Ci1 = Ci1_;
    static uint32_t constexpr nBL1 = nBL1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(Kd, Ci1, nBL1);
    }
};

template <
    uint32_t mL0_ = 1,
    uint32_t kL0_ = 1,
    uint32_t nL0_ = 1
>
struct ConvL0Shape {
    static uint32_t constexpr mL0 = mL0_;
    static uint32_t constexpr kL0 = kL0_;
    static uint32_t constexpr nL0 = nL0_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(mL0, kL0, nL0);
    }
};

struct Conv3d6HdCoord : public Coord<4, uint32_t> {
    using Index = uint32_t;

    using Base = Coord<4, Index>;

    static constexpr int N_INDEX = 0;
    static constexpr int D_INDEX = 1;
    static constexpr int C1_INDEX = 2;
    static constexpr int HW_INDEX = 3;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv3d6HdCoord() {}

    CATLASS_HOST_DEVICE
    Conv3d6HdCoord(Coord<4, Index> const &coord) : Base(coord) {}

    CATLASS_HOST_DEVICE
    Conv3d6HdCoord(Index n, Index d, Index c1, Index hw)
        : Base(MakeCoord(n, d, c1, hw)) {}

    CATLASS_HOST_DEVICE
    Index const& n() const
    {
        return this->At(N_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& n()
    {
        return this->At(N_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& d() const
    {
        return this->At(D_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& d()
    {
        return this->At(D_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& c1() const
    {
        return this->At(C1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& c1()
    {
        return this->At(C1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& hw() const
    {
        return this->At(HW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& hw()
    {
        return this->At(HW_INDEX);
    }
};

struct Conv3dFracZ3dCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;

    using Base = Coord<2, Index>;

    static constexpr int KDC1KHKW_INDEX = 0;
    static constexpr int N1_INDEX = 1;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv3dFracZ3dCoord() {}

    CATLASS_HOST_DEVICE
    Conv3dFracZ3dCoord(Index kdc1khkw, Index n1)
        : Base(MakeCoord(kdc1khkw, n1)) {}

    CATLASS_HOST_DEVICE
    Index const& kdc1khkw() const
    {
        return this->At(KDC1KHKW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& kdc1khkw()
    {
        return this->At(KDC1KHKW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const& n1() const
    {
        return this->At(N1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index& n1()
    {
        return this->At(N1_INDEX);
    }
};
}

#endif  // CATLASS_CONV_COORD_HPP