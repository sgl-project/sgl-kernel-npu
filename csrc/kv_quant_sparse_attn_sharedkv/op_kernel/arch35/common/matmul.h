









/*!
 * \file matmul.h
 * \brief
 */
#ifndef MATMUL_H
#define MATMUL_H
#include "buffers_policy.h"
using namespace AscendC;
namespace fa_base_matmul {

constexpr uint32_t UNITFLAG_DISABLE = 0;
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16;
static constexpr uint32_t INT4_ONE_FRACTAL_ELEMENT = 64;
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16;
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32;
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
static constexpr uint32_t MMAD_MN_SIZE_10 = 10;
static constexpr uint8_t LOAD3D_STRIDE_W = 1;
static constexpr uint8_t LOAD3D_STRIDE_H = 1;
static constexpr uint8_t LOAD3D_FILTER_W = 1;
static constexpr uint8_t LOAD3D_FILTER_H = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_W = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_H = 1;
static constexpr uint32_t K_STEP_ALIGN_BASE = 2;
static constexpr uint32_t M_STEP_ALIGN_BASE = 2;

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true;
    uint32_t unitFlag = 0;



    uint32_t realM = 0;
};

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size) {
    if constexpr (IsSameType<T, float>::value) {
        return ((size + 7) >> 3 << 3) >> 3;
    } else if constexpr ((IsSameType<T, fp8_e5m2_t>::value ||
                          IsSameType<T, fp8_e4m3fn_t>::value ||
                          IsSameType<T, hifloat8_t>::value ||
                          IsSameType<T, int8_t>::value)) {
        return ((size + 31) >> 5 << 5) >> 5;
    } else {
        return ((size + 15) >> 4 << 4) >> 4;
    }
}

template <typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA;
    loadData2DParamsA.mStartPosition = 0;
    loadData2DParamsA.kStartPosition = 0;
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose;
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) / 16;
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize);
    } else {
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) / 16;
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize);
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsA.ifTranspose) {
            loadData2DParamsA.kStep = CeilAlign(loadData2DParamsA.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((kSplitSize + 63) >> 6 << 6) / 16 : ((mSplitSize + 31) >> 5 << 5) / 16;
    } else {
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) / 16 : loadData2DParamsA.mStep;
    }
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) / 16;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) / 16 : loadData2DParamsA.mStep;
    LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
}


template <typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB;
    loadData2DParamsB.mStartPosition = 0;
    loadData2DParamsB.kStartPosition = 0;
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose;
    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) / 16;
        loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize);
    } else {
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) / 16;
        loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize);
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.kStep = CeilAlign(loadData2DParamsB.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) / 16;
        } else {
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) / 16;
        }
    } else {
        loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose ? (((mmParam.singleK + 15) >> 4 << 4) / 16) : (((mmParam.singleN + 15 ) >> 4 << 4) / 16);
    }
    loadData2DParamsB.dstStride = loadData2DParamsB.ifTranspose ? (nSplitSize + 15) / 16 : loadData2DParamsB.mStep;
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsB.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], loadData2DParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
        }
    } else {
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    }
}
#else
static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize,
                                     uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = mSplitSize / LOAD3D_L1W_SIZE;
        loadData3DParams.l1W = LOAD3D_L1W_SIZE;
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255;

        loadData3DParams.mExtension = mSplitSize;
        loadData3DParams.kExtension = kSplitSize;
        loadData3DParams.mStartPt = 0;
        loadData3DParams.kStartPt = 0;
        loadData3DParams.strideW = 1;
        loadData3DParams.strideH = 1;
        loadData3DParams.filterW = 1;
        loadData3DParams.filterSizeW = false;
        loadData3DParams.filterH = 1;
        loadData3DParams.filterSizeH = false;
        loadData3DParams.dilationFilterW = 1;
        loadData3DParams.dilationFilterH = 1;
        loadData3DParams.enTranspose = 0;
        loadData3DParams.fMatrixCtrl = 0;
        loadData3DParams.channelSize = kSplitSize;
        LoadData<T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams);
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = (kSplitSize / ONE_FRACTAL_H_ELEMENT) * (mmParam.singleM /
                                       (ONE_FRACTAL_W_BYTE / sizeof(T)));
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = true;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParams);
    }
}


__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize,
                                     uint32_t nSplitSize)
{
    if constexpr (BL == ABLayout::KN) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = kSplitSize / LOAD3D_L1W_SIZE;
        loadData3DParams.l1W = LOAD3D_L1W_SIZE;
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255;

        loadData3DParams.mExtension = kSplitSize;
        loadData3DParams.kExtension = nSplitSize;
        loadData3DParams.mStartPt = 0;
        loadData3DParams.kStartPt = 0;
        loadData3DParams.strideW = LOAD3D_STRIDE_W;
        loadData3DParams.strideH = LOAD3D_STRIDE_H;
        loadData3DParams.filterW = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW = false;
        loadData3DParams.filterH = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH = false;
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W;
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H;
        loadData3DParams.enTranspose = 1;
        loadData3DParams.fMatrixCtrl = 0;
        loadData3DParams.channelSize = nSplitSize;
        LoadData<T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams);
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = (nSplitSize + (ONE_FRACTAL_H_ELEMENT - 1)) / ONE_FRACTAL_H_ELEMENT *
                                       (kSplitSize / (ONE_FRACTAL_W_BYTE / sizeof(T)));
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParams);
    }
}

#endif

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulFull(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param)
{
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
    LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    l0aBuffer.Set<HardEvent::MTE1_M>();

    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
    LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    l0bBuffer.Set<HardEvent::MTE1_M>();

    l0aBuffer.Wait<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.Set<HardEvent::M_MTE1>();
    l0bBuffer.Set<HardEvent::M_MTE1>();
}


template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulK(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float32_t>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        l0aBuffer.Set<HardEvent::MTE1_M>();

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        l0bBuffer.Set<HardEvent::MTE1_M>();

        l0aBuffer.Wait<HardEvent::MTE1_M>();
        l0bBuffer.Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.Set<HardEvent::M_MTE1>();
        l0bBuffer.Set<HardEvent::M_MTE1>();
    }
}


template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulN(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              struct MMParam &param)
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value) {
        L1Boffset = ((param.singleK + 31) >> 5 << 5) * baseN;
    }
#endif
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
    LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    l0aBuffer.Set<HardEvent::MTE1_M>();
    l0aBuffer.Wait<HardEvent::MTE1_M>();

    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);
        l0bBuffer.Set<HardEvent::MTE1_M>();

        l0bBuffer.Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        l0bBuffer.Set<HardEvent::M_MTE1>();
    }
    l0aBuffer.Set<HardEvent::M_MTE1>();
}


template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKM(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              BuffersPolicyDB<BufferType::L0A> &aL0BuffsDb,
                              BuffersPolicyDB<BufferType::L0B> &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              struct MMParam &param)
{
    uint32_t mLoops = (param.singleM + baseM - 1) / baseN;
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t mSplitSize = (mLoops == 1) ? param.singleM : baseM;
    uint32_t mplitTailSize = (param.singleM % baseM) ? (param.singleM % baseM) : mSplitSize;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitTailSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
    uint64_t L1Boffset = kSplitSize * param.singleN;
    uint64_t L0Coffset = mSplitSize * param.singleN;

    for (uint32_t k = 0; k < kLoops; k++) {
        kSplitSize = (k == (kLoops - 1)) ? kSplitTailSize : kSplitSize;
        for (uint32_t m = 0; m < mLoops; m++){
            mSplitSize = (m == (mLoops - 1)) ? mplitTailSize : mSplitSize;
            Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
            l0aBuffer.Wait<HardEvent::M_MTE1>();
            LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
            LoadDataToL0A(L0ATensor, aL1Tensor, param,
                k * param.singleM * kSplitSize + m * kSplitSize * mSplitSize,
                kSplitSize, mSplitSize);
            l0aBuffer.Set<HardEvent::MTE1_M>();
            l0aBuffer.Wait<HardEvent::MTE1_M>();

            Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
            l0bBuffer.Wait<HardEvent::M_MTE1>();
            LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSize, param.singleN);
            l0bBuffer.Set<HardEvent::MTE1_M>();
            l0bBuffer.Wait<HardEvent::MTE1_M>();

            MmadParams mmadParams;
            mmadParams.m = mSplitSize;
            mmadParams.n = param.singleN;
            mmadParams.k = kSplitSize;
            mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
            mmadParams.cmatrixSource = false;
            Mmad(cL0Tensor[m * L0Coffset], L0ATensor, L0BTensor, mmadParams);

            l0aBuffer.Set<HardEvent::M_MTE1>();
            l0bBuffer.Set<HardEvent::M_MTE1>();
        }
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulBase(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param)
{
    if ((param.singleK + baseK - 1) / baseK > 1) {
        MatmulK<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else if ((param.singleN + baseN - 1) / baseN > 1) {
        MatmulN<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else {
        MatmulFull<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKPP(const LocalTensor<A> &aL1Tensor,
                                 const LocalTensor<B> &bL1Tensor,
                                 BuffersPolicyDB<BufferType::L0A> &aL0BuffsDb,
                                 BuffersPolicyDB<BufferType::L0B> &bL0BuffsDb,
                                 const LocalTensor<C> &cL0Tensor,
                                 const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
    uint64_t L1Aoffset = AlignUp(param.singleM, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    uint64_t L1Boffset = AlignUp(param.singleN, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    for (uint32_t k = 0; k < kLoops; k++) {
        if (k == kLoops - 1) {
            kSplitSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
            kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
        }
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, kSplitSizeAlign, param.singleM);
        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSizeAlign, param.singleN);

        l0aBuffer.Set<HardEvent::MTE1_M>();
        l0aBuffer.Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        mmadParams.n = param.singleN;
        mmadParams.k = kSplitSize;
        if (mmadParams.m == 1) {
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = (param.isOutKFisrt == true) && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
    #if (__CCE_AICORE__ != 310) && (!(defined __DAV_310R6__))
        if ((mmadParams.m / FP16_ONE_FRACTAL_ELEMENT) * (mmadParams.n / FP16_ONE_FRACTAL_ELEMENT) < MMAD_MN_SIZE_10) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    #endif
        l0aBuffer.Set<HardEvent::M_MTE1>();
    }
}
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                    uint32_t rowSize, uint32_t kSplitSize, uint32_t mSplitSize)
{
    uint32_t blockElementCnt = ONE_FRACTAL_W_BYTE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        blockElementCnt = INT4_ONE_FRACTAL_ELEMENT;
    }
    if constexpr (AL == ABLayout::MK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = kSplitSize / blockElementCnt - 1;
        loadData2DParams.repeatTimes = mSplitSize / ONE_FRACTAL_H_ELEMENT;
        uint32_t loopTimes = kSplitSize / blockElementCnt;
        uint64_t l1Offset = rowSize * blockElementCnt;
        uint64_t l0Offset = ONE_FRACTAL_H_ELEMENT * blockElementCnt;
        for(uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadData(aL0Tensor[loop * l0Offset], aL1Tensor[loop * l1Offset], loadData2DParams);
        }
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2dTransposeParams loadData2dTransposeParams;
        loadData2dTransposeParams.startIndex = 0;
        loadData2dTransposeParams.srcStride = 1;
        loadData2dTransposeParams.dstFracGap = (kSplitSize + blockElementCnt -1) / blockElementCnt;
        loadData2dTransposeParams.dstGap = mSplitSize / ONE_FRACTAL_H_ELEMENT - 1;
        if(rowSize == kSplitSize) {
            loadData2dTransposeParams.repeatTimes = (kSplitSize + blockElementCnt - 1) / blockElementCnt;
            uint32_t loopTimes = mSplitSize / blockElementCnt;
            uint64_t l1Offset = rowSize * blockElementCnt;
            uint64_t l0Offset = kSplitSize * blockElementCnt;
            for(uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadDataWithTranspose(aL0Tensor[loop * l0Offset], aL1Tensor[loop * l1Offset], loadData2dTransposeParams);
            }
        } else {
            loadData2dTransposeParams.repeatTimes = ((kSplitSize + blockElementCnt - 1) / blockElementCnt) * (mSplitSize / blockElementCnt);
            LoadDataWithTranspose(aL0Tensor, aL1Tensor, loadData2dTransposeParams);
        }
    }
}

template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                     uint32_t rowSize, uint32_t kSplitSize, uint32_t nSplitSize)
{
    uint32_t blockElementCnt = ONE_FRACTAL_W_BYTE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        blockElementCnt = INT4_ONE_FRACTAL_ELEMENT;
    }
    if constexpr (BL == ABLayout::KN) {
        LoadData2dTransposeParams loadData2dTransposeParams;

        loadData2dTransposeParams.startIndex = 0;
        loadData2dTransposeParams.srcStride = 1;
        loadData2dTransposeParams.dstFracGap = 0;
        loadData2dTransposeParams.dstGap = nSplitSize / ONE_FRACTAL_H_ELEMENT - 1;
        loadData2dTransposeParams.repeatTimes = (kSplitSize + blockElementCnt - 1) / blockElementCnt;

        uint32_t loopTimes = nSplitSize / blockElementCnt;
        uint64_t l1Offset = rowSize * blockElementCnt;
        uint64_t l0Offset = blockElementCnt * blockElementCnt;
        for(uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadDataWithTranspose(bL0Tensor[loop * l0Offset], bL1Tensor[loop * l1Offset], loadData2dTransposeParams);
        }
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        if(rowSize == kSplitSize) {
            loadData2DParams.repeatTimes = ((nSplitSize + ONE_FRACTAL_H_ELEMENT - 1) / ONE_FRACTAL_H_ELEMENT) * (kSplitSize / blockElementCnt);
            LoadData(bL0Tensor, bL1Tensor, loadData2DParams);
        } else {
            loadData2DParams.repeatTimes = (nSplitSize + ONE_FRACTAL_H_ELEMENT - 1) / ONE_FRACTAL_H_ELEMENT;
            uint32_t loopTimes = kSplitSize / blockElementCnt;
            uint64_t l1Offset = nSplitSize * blockElementCnt;
            uint64_t l0Offset = rowSize * blockElementCnt;
            for (uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadData(bL0Tensor[loop * l0Offset], bL1Tensor[loop * l1Offset], loadData2DParams);
            }
        }
    }
}
}
#endif
