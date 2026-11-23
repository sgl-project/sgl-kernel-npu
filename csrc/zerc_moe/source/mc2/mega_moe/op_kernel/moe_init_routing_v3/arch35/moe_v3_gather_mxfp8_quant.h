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
 * \file moe_v3_gather_mxfp8_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE
#define MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

// MX量化一个块的元素数量为32个，即1个scale对应32个x的元素
constexpr int64_t MX_BLOCK_SIZE = 32LL;
// 一个VL放多少fp32->fp8的元素个数，即按fp32算的一个VL能放几个元素
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
// fp16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
// bf16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
// bfloat16的nan值（与inf值不同）
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
// 对于x的目标类型为e5m2的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
// 对于x的目标类型为e4m3的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
// e8m0的inf/nan值（按定义应该是nan值）
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
// e8m0的极小值，用于写0
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
// bf16指数位的偏移位数
constexpr int16_t BF16_EXP_SHR_BITS = 7;
// 用于计算halfScale=BF16_EXP_INVSUB-sharedExp，得到的halfScale就是1/realScale，可以用于量化xQuant=x*halfScale
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;

template <typename T, typename U>
__simd_vf__ inline void vfComputeMaxExp(__ubuf__ T *xAddr, __ubuf__ uint16_t *maxExpOutAddr, uint32_t xElemNum,
                                        uint16_t vfLoopNum, uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::Reg;

    // 用于把float16转为bfloat16，不涉及宽度变化
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};

    // 0存奇数位元素，1存偶数位元素
    RegTensor<T> x0, x1;
    RegTensor<bfloat16_t> x0BF16, x1BF16;
    RegTensor<uint16_t> exp0, exp1, exp0FP16, exp1FP16, maxExp;
    // 存储FP16/BF16的指数位为1的mask
    RegTensor<uint16_t> emaskFP16, emaskBF16;
    Duplicate(emaskFP16, FP16_EMASK_AND_INF_VAL);
    Duplicate(emaskBF16, BF16_EMASK_AND_INF_VAL);
    // 2字节Reg的MaskALL
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg mask0, mask1, mask0FP16NanInf, mask1FP16NanInf;
    // 非对齐搬出至UB用
    UnalignReg uReg;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        mask0 = UpdateMask<T>(xElemNum);
        mask1 = UpdateMask<T>(xElemNum);
        MaskDeInterleave<T>(mask0, mask1, mask0, mask1);

        // 双搬入，奇数位放x0、偶数位放x1
        DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);

        if constexpr (IsSameType<T, half>::value) {
            // 单独提取fp16中inf/nan的元素
            And(exp0FP16, (RegTensor<uint16_t> &)x0, emaskFP16, mask0);
            And(exp1FP16, (RegTensor<uint16_t> &)x1, emaskFP16, mask1);
            Compare<uint16_t, CMPMODE::EQ>(mask0FP16NanInf, exp0FP16, emaskFP16, mask0);
            Compare<uint16_t, CMPMODE::EQ>(mask1FP16NanInf, exp1FP16, emaskFP16, mask1);

            // fp16要先转成bf16
            Cast<bfloat16_t, T, traitFP16ToBF16>(x0BF16, x0, mask0);
            Cast<bfloat16_t, T, traitFP16ToBF16>(x1BF16, x1, mask1);
            // 用BF16_EMASK_AND_INF_VAL提取BF16的指数位
            And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1BF16, emaskBF16, mask1);

            // exp[expFP16==nan/inf]=nan/inf
            Select(exp0, emaskBF16, exp0, mask0FP16NanInf);
            Select(exp1, emaskBF16, exp1, mask1FP16NanInf);
        } else {
            // 用BF16_EMASK_AND_INF_VAL提取BF16的指数位
            And(exp0, (RegTensor<uint16_t> &)x0, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1, emaskBF16, mask1);
        }
        // 先计算奇数位、偶数位之间的最大值，即每2个元素的最大值，放在maxExp
        // 由于mask0和mask1相同或是比mask1多一个元素，使用mask0保证不漏元素
        Max(maxExp, exp0, exp1, mask0);
        // 再每32B（对于T来说就是每16个元素）取一个最大值，此时maskExp里放的是每2个元素的最大值，且末尾数据为0，因此用MaskALL
        ReduceMaxWithDataBlock(maxExp, maxExp, maskAllB16);
        // 非对齐搬出，每次写出numVRegBlocks个元素（ReduceMaxWithDataBlock后放在maxExp头部）
        DataCopyUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExpOutAddr, maxExp, uReg, numVRegBlocks);
    }
    // 非对齐搬出收尾，尾部多出的位置写0
    DataCopyUnAlignPost(maxExpOutAddr, uReg, 0);
}

template <typename T, typename U>
__simd_vf__ inline void vfComputeScale(__ubuf__ uint16_t *maxExpInAddr, __ubuf__ uint16_t *mxScaleOutAddr,
                                       __ubuf__ uint16_t *invScaleOutAddr, uint32_t scaleElemNum,
                                       uint32_t validScaleElemNum, uint16_t vfLoopNum, uint32_t vlForT,
                                       uint16_t expLowerBoundValue)
{
    using namespace AscendC::Reg;

    RegTensor<uint16_t> maxExp, sharedExp, mxScale, invScale;
    // infBF16存放bf16指数的inf值
    RegTensor<uint16_t> infBF16;
    Duplicate(infBF16, BF16_EMASK_AND_INF_VAL);
    // zeroB16存放B16长度的0值
    RegTensor<uint16_t> zeroB16;
    Duplicate(zeroB16, 0);
    // expLowerBound存放对于maxExp来说过小的值
    RegTensor<uint16_t> expLowerBound;
    Duplicate(expLowerBound, expLowerBoundValue);
    // nanForE8M0以16位的低8位存放float8_e8m0的nan值
    RegTensor<uint16_t> nanForE8M0;
    Duplicate(nanForE8M0, FP8_E8M0_NAN_VAL);
    // invSub用于计算invSCale=BF16_EXP_INVSUB-sharedExp，得到的invScale==1/realScale，可用于量化x
    RegTensor<uint16_t> invSub;
    Duplicate(invSub, BF16_EXP_INVSUB);
    // nanBF16存放bf16的nan值
    RegTensor<uint16_t> nanBF16;
    Duplicate(nanBF16, BF16_NAN_VAL);
    // specialMinE8M0存放float8_e8m0的特殊极小值
    RegTensor<uint16_t> specialMinE8M0;
    Duplicate(specialMinE8M0, FP8_E8M0_SPECIAL_MIN);

    // 循环用mask
    MaskReg maskLoop, maskValid;
    // 存储compare后的结果用的mask
    MaskReg maskInfBF16, maskZero, maskLowExp, maskSpecialMin;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskLoop = UpdateMask<uint16_t>(scaleElemNum);
        maskValid = UpdateMask<uint16_t>(validScaleElemNum);
        // 拷入vfComputeMaxExp算好的maxExp
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpInAddr, vlForT);

        // 1.计算并拷出mxScale（float8_e8m0）

        // maskLowExp提取maxExp过小的位置
        Compare<uint16_t, CMPMODE::LT>(maskLowExp, maxExp, expLowerBound, maskValid);
        // maxExp[<expLowerBound]=expLowerBound
        Select<uint16_t>(maxExp, expLowerBound, maxExp, maskLowExp);

        // sharedExp=maxExp-expLowerBound
        Sub(sharedExp, maxExp, expLowerBound, maskValid);
        // mxScale=sharedExp>>BF16_EXP_SHR_BITS) 即把sharedExp存储的指数位右移到低8位，以便存放在float8_e8m0中
        ShiftRights(mxScale, sharedExp, BF16_EXP_SHR_BITS, maskValid);  // 右移，幂运算

        // mxScale[maxExp==infBF16]=nanE8M0
        Compare<uint16_t, CMPMODE::EQ>(maskInfBF16, maxExp, infBF16, maskValid);
        Select<uint16_t>(mxScale, nanForE8M0, mxScale, maskInfBF16);
        // mxScale[maxExp==0]=0
        Compare<uint16_t, CMPMODE::EQ>(maskZero, maxExp, zeroB16, maskValid);
        Select<uint16_t>(mxScale, zeroB16, mxScale, maskZero);

        // 拷出算好的mxScale，即为算子输出expandedScale的值。
        // 对于每个uint16的mxScale，都只拷出其低8位，也就是以uint8的长度拷出，实际上存储的就是float8_e8m0类型的二进制值，因此POST_MODE_UPDATE的数目为vlForT/2。
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(mxScaleOutAddr, mxScale, vlForT / 2,
                                                                                    maskLoop);

        // 2.计算并拷出invScale（bfloat16）

        // invScale=invSub-sharedExp
        Sub<uint16_t>(invScale, invSub, sharedExp, maskValid);
        // invScale[maxExp==InfBF16]=nanBF16
        Select<uint16_t>(invScale, nanBF16, invScale, maskInfBF16);
        // invScale[maxExp==0]=0
        Select<uint16_t>(invScale, zeroB16, invScale, maskZero);
        // invScale[(invScale=invSub-sharedExp)==0]=FP8_E8M0_SPECIAL_MIN
        Compare<uint16_t, CMPMODE::EQ>(maskSpecialMin, invSub, sharedExp, maskValid);
        Select<uint16_t>(invScale, specialMinE8M0, invScale, maskSpecialMin);

        // 拷出算好的invScale，用于在后续量化xQuant=invScale*x，以uint16的长度拷出，实际存储的为bfloat16类型的二进制值
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(invScaleOutAddr, invScale, vlForT, maskLoop);
    }
}

template <typename T, typename U>
__simd_vf__ inline void vfComputeData(__ubuf__ T *xAddr, __ubuf__ uint16_t *invScaleInAddr,
                                      __ubuf__ int8_t *xQuantOutAddr, uint32_t xElemNum, uint16_t vfLoopNum,
                                      uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::Reg;

    // fp16转bf16，不涉及宽度变化，精度截断
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};
    // b16转b32，奇数位的2字节->4字节，元素个数减半
    static constexpr CastTrait traitB16ToB32Layout0 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::UNKNOWN};
    // b16转b32，偶数位的2字节->4字节，元素个数减半
    static constexpr CastTrait traitB16ToB32Layout1 = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::UNKNOWN};
    // b32转b8，每个元素4字节->1字节，放在原来4字节的低位，超出b8的值写b8max，精度截断
    static constexpr CastTrait traitB32ToB8Layout0 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                      RoundMode::CAST_RINT};

    RegTensor<uint16_t> invScale;
    RegTensor<float> invScaleFP32;
    RegTensor<T> x0, x1;
    RegTensor<float> x0FP32Layout0, x0FP32Layout1, x1FP32Layout0, x1FP32Layout1;
    RegTensor<U> xQuant0, xQuant1, xQuant2, xQuant3;

    MaskReg maskXQuant0B32, maskXQuant1B32, maskXQuant2B32, maskXQuant3B32;
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg maskAllB32 = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskXQuant0B32 = UpdateMask<float>(xElemNum);
        maskXQuant1B32 = UpdateMask<float>(xElemNum);
        maskXQuant2B32 = UpdateMask<float>(xElemNum);
        maskXQuant3B32 = UpdateMask<float>(xElemNum);

        // 拷入待量化的x：x0存放偶数位的值，x1存放奇数位的值，双搬共拷入vlForT*2个元素
        DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);
        // 拷入invScale，类型为uint16（计算时翻译成bfloat16），其中每个元素广播至32B，这里就是每个元素重复16次，共256B/32B=8个元素，也就是numVRegBlocks的值
        // 应该是加载(256/32=8)B的元素
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(invScale, invScaleInAddr,
                                                                                  numVRegBlocks);
        if constexpr (IsSameType<T, half>::value) {
            // 对于T为float16，先把x和invScale转成float32，再计算x*invScale后转为float8对应类型

            // 0.invScaleFP32存放invScale转为FP32后的值，invScale每个元素重复16次->invScaleFP32每个元素重复8次
            Cast<float, bfloat16_t, traitB16ToB32Layout0>(invScaleFP32, (RegTensor<bfloat16_t> &)invScale, maskAllB16);

            // 1.对奇数位元素x0进行量化：
            // x0FP32Layout0放x0的奇数位转为fp32的值，x0FP32Layout1放x0的偶数位转fp32的值
            Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
            // x0FP32*=invScaleFP32，这里每8个x0FP32Layout0、x0FP32Layout1对应一个invScaleFP32，加起来共16个x0FP32对应1个invScale
            Mul(x0FP32Layout0, x0FP32Layout0, invScaleFP32, maskAllB32);
            Mul(x0FP32Layout1, x0FP32Layout1, invScaleFP32, maskAllB32);
            // 把x0FP32Layout0、x0FP32Layout1由奇偶交织恢复为连续存储
            Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);

            // 2.对偶数位元素x1进行量化，与x0的做法一致：
            Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
            Mul(x1FP32Layout0, x1FP32Layout0, invScaleFP32, maskAllB32);
            Mul(x1FP32Layout1, x1FP32Layout1, invScaleFP32, maskAllB32);
            Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);

            // 3.把量化后的x0FP32Layout[i]、x1FP32Layout0[i]恢复顺序：
            // 按x的原始顺序依次为：x0FP32Layout0、x1FP32Layout0、x0FP32Layout1、x1FP32Layout1
            Interleave(x0FP32Layout0, x1FP32Layout0, x0FP32Layout0, x1FP32Layout0);
            Interleave(x0FP32Layout1, x1FP32Layout1, x0FP32Layout1, x1FP32Layout1);

            // 4.挨个cast回目标fp8类型
            Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant1, x1FP32Layout0, maskXQuant1B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant2, x0FP32Layout1, maskXQuant2B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
        } else {
            // 对于T为bfloat16，直接计算x*invScale，然后将结果先转为float32，再转为float8对应类型

            // 1.乘以invScale
            // x[i]*= invScale，这里invScale每元素重复16次，刚好对于x0、x1就是每32元素一个invScale
            Mul(x0, x0, (RegTensor<T> &)invScale, maskAllB16);
            Mul(x1, x1, (RegTensor<T> &)invScale, maskAllB16);
            // 恢复x0、x1为原始顺序
            Interleave(x0, x1, x0, x1);

            // 2.x0部分转为FP32，再转为FP8目标类型：
            // x0FP32Layout0存x0转为fp32的奇数位元素，x0FP32Layout1存x0转为fp32的偶数位元素，各64个
            Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
            // 恢复x0FP32Layout0、x0FP32Layout1的顺序
            Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);
            // 转为FP8对应类型，每个4字节x->1字节xQuant，但个数不增加，还是各64个
            Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant1, x0FP32Layout1, maskXQuant1B32);

            // 3.x1部分转为FP32，再转为FP8目标类型：
            Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
            Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);
            Cast<U, float, traitB32ToB8Layout0>(xQuant2, x1FP32Layout0, maskXQuant2B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
        }

        // 拷出xQuant到xOutAddr。OUT_ELE_NUM_ONE_BLK=64，即一个VL下有64个float32->fp8的元素，一次拷出64个元素
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant0B32);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant1, OUT_ELE_NUM_ONE_BLK, maskXQuant1B32);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant2, OUT_ELE_NUM_ONE_BLK, maskXQuant2B32);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant3, OUT_ELE_NUM_ONE_BLK, maskXQuant3B32);
    }
}

template <typename T, typename U>
class MoeGatherOutMxfp8Quant
{
public:
    __aicore__ inline MoeGatherOutMxfp8Quant(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR expandedRowIdxGatherAddr,
                                GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void CopyInGatherIdx(int64_t globalTokenIdx);
    __aicore__ inline void ProcessOneToken(int64_t globalTokenIdx);
    __aicore__ inline void CopyIn(int64_t srcTokenIdx);
    __aicore__ inline void Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum);
    __aicore__ inline void CopyOut(int64_t dstIdx);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> gatherIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXAndScaleOutGm_;
    GlobalTensor<int32_t> expandedRowIdxGatherGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t validScaleCols_;  // 一个token实际有意义的scale有多少个元素（列）
    // 一个token的scale有多少个元素（列），即CeilAlign(CeliDiv(h,32),2)，在actualScaleCols_为奇数时，scaleCols_=validScaleCols_+1
    int64_t scaleCols_;
    int64_t n_;
    int64_t k_;
    int64_t perCoreTokens_;
    int64_t coreTokens_;
    int64_t perLoopCols_;
    int64_t perLoopScaleCols_;
    LocalTensor<uint8_t> outLocal_;

    // 会赋值为计算maxExp时，maxExp对应不同的目标fp8类型U的下限
    uint16_t lowerBoundOfB16MaxExp_ = 0;

    // 一个RegTensor的长度Bytes
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    // ub的一个块大小，通常是32B
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    // 一个RegTensor能塞下多少T(B16)类型的元素
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    // 一个RegTensor有几个UbBlock，即ReduceWithBlock后的元素个数（256B/32B=8）
    // 这里如果ubBlockSize_获取的值为0，应该让kernel挂掉
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_;  // 8个元素
    // 每个ubBlock对应多少T类型的元素
    const uint32_t numElemPerUbBlock_ = ubBlockSize_ / sizeof(T);
};

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                          GM_ADDR expandedRowIdxGatherAddr, GM_ADDR expandedRowIdxAddr,
                                                          GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                          const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                          TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXAndScaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    expandedRowIdxGatherGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxGatherAddr);

    pipe_->InitBuffer(gatherIdxInQueue_, 2, AlignBytes(k_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, 2, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xQuantOutQueue_, 2, AlignBytes(perLoopCols_ + perLoopScaleCols_, sizeof(int8_t)));
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
    }
}

template <typename T, typename U>
__aicore__ inline void
MoeGatherOutMxfp8Quant<T, U>::InitKernelTiling(const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2);
    n_ = tilingData->n;
    k_ = tilingData->k;

    // core split by token count n
    perCoreTokens_ = Ceil(n_, tilingData->coreNum);
    needCoreNum_ = Ceil(n_, perCoreTokens_);
    int64_t lastCoreTokens = n_ - (needCoreNum_ - 1) * perCoreTokens_;

    if (blockIdx_ == needCoreNum_ - 1) {
        coreTokens_ = lastCoreTokens;
    } else {
        coreTokens_ = perCoreTokens_;
    }

    // cols: colLoops_==1, directly use perLoopCols
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE;
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t tokenBase = perCoreTokens_ * blockIdx_;
        for (int64_t t = 0; t < coreTokens_; t++) {
            ProcessOneToken(tokenBase + t);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyInGatherIdx(int64_t globalTokenIdx)
{
    LocalTensor<int32_t> gatherIdxLocal = gatherIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(gatherIdxLocal, expandedRowIdxGatherGm_[globalTokenIdx * k_], dataCopyParams, dataCopyPadParams);
    gatherIdxInQueue_.EnQue<int32_t>(gatherIdxLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::ProcessOneToken(int64_t globalTokenIdx)
{
    // 批量搬入 k 个 gather idx
    // CopyInGatherIdx(globalTokenIdx);
    // LocalTensor<int32_t> gatherIdxLocal = gatherIdxInQueue_.DeQue<int32_t>();
    // SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    // CopyIn -> Compute（一次量化）
    CopyIn(globalTokenIdx);
    Compute(perLoopCols_, perLoopScaleCols_, perLoopScaleCols_);

    // DeQue 一次量化结果，写出 k 份（不插 MTE3 等待）
    outLocal_ = xQuantOutQueue_.DeQue<uint8_t>();
    for (int64_t jj = 0; jj < k_; jj++) {
        // int64_t dstIdx = gatherIdxLocal.GetValue(jj);
        // CopyOut(dstIdx);
        CopyOut(expandedRowIdxGatherGm_(globalTokenIdx * k_ + jj));
    }
    xQuantOutQueue_.FreeTensor(outLocal_);

    // gatherIdxInQueue_.FreeTensor(gatherIdxLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyIn(int64_t srcTokenIdx)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t loopColsTail = perLoopCols_ % MX_BLOCK_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > numElemPerUbBlock_) {
            padParams.rightPadding = numElemPerUbBlock_ - (loopColsTail - numElemPerUbBlock_);
        } else {
            padParams.rightPadding = numElemPerUbBlock_;
        }
    }
    DataCopyPad(inLocal, xInGm_[srcTokenIdx * cols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::Compute(uint32_t xElemNum, uint32_t scaleElemNum,
                                                             uint32_t validScaleElemNum)
{
    // deque input
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    auto xLocalAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());
    // alloc outputs
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    auto xQuantLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(xQuantLocal.GetPhyAddr());
    // scale data is placed right after quant data in the same UB buffer
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(xQuantLocalAddr + perLoopCols_);
    // get tmp buffers
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    LocalTensor<uint16_t> invScaleLocal = invScaleBuffer_.Get<uint16_t>();
    auto invScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(invScaleLocal.GetPhyAddr());

    // 按每次双搬vlForHalfNumber_*2来计算loopNum
    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ * 2 - 1) / (vlForB16_ * 2);
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    VF_CALL<vfComputeMaxExp<T, U>>(xLocalAddr, maxExpLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                   numUbBlocksPerVReg_);
    VF_CALL<vfComputeScale<T, U>>(maxExpLocalAddr, mxScaleLocalAddr, invScaleLocalAddr, scaleElemNum, validScaleElemNum,
                                  vfLoopNumForScale, vlForB16_, lowerBoundOfB16MaxExp_);
    VF_CALL<vfComputeData<T, U>>(xLocalAddr, invScaleLocalAddr, xQuantLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                 numUbBlocksPerVReg_);

    // free input
    xInQueue_.FreeTensor(xLocal);
    // Enqueue outputs.
    xQuantOutQueue_.EnQue(xQuantLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U>::CopyOut(int64_t dstIdx)
{
    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(perLoopCols_ + perLoopScaleCols_), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXAndScaleOutGm_[dstIdx * (cols_ + scaleCols_)], outLocal_, copyOutParams);
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE
