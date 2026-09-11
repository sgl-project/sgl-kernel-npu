









/*!
 * \file CopyInL1.h
 * \brief
 */
#ifndef COPYINL1_H
#define COPYINL1_H

enum class KVLAYOUT
{
    BNBD, // [blockNums, headNum, blockSize, headDim]
    BBH, // [blockNums, blockSize, headNum * headDim]
    NZ // [blockNums, headNum, d1, blockSize, d0], d1 = headDim / d0, d0 = 32 (block byte) / sizeof(KV_T)
};

struct CopyParam{
    uint32_t width;
    uint32_t height;
    uint32_t orgWidth;
};

struct PAShape{
    uint32_t blockNum;
    uint32_t blockSize;
    uint32_t headNum;
    uint32_t headDim;
    uint32_t maxblockNumPerBatch;
    uint32_t actHeadDim;
    uint32_t copyRowNum;
    uint32_t copyRowNumAlign;
};

struct Position{
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Offset;
    uint32_t dIdx;
};

template<typename L1Type>
__aicore__ inline void GmCopyInToL1(LocalTensor<L1Type>& L1Tensor, GlobalTensor<L1Type>& GmTensor, const CopyParam& mmCopyParam)
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;
    gm2L1Nd2NzParams.nValue = mmCopyParam.height;
    gm2L1Nd2NzParams.dValue = mmCopyParam.width;
    gm2L1Nd2NzParams.srcNdMatrixStride = 0;
    gm2L1Nd2NzParams.srcDValue = mmCopyParam.orgWidth;
    gm2L1Nd2NzParams.dstNzC0Stride = BaseApi::Align16Func(gm2L1Nd2NzParams.nValue);
    DataCopy(L1Tensor, GmTensor, gm2L1Nd2NzParams);
}



template<typename L1Type>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col, // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;
    nd2nzPara.dValue = col;
    nd2nzPara.srcDValue = colStride;
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template<typename L1Type>
__aicore__ inline void DataCopyGmNZToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct,
                                        uint32_t dstRowStride,
                                        uint32_t srcRowStride,
                                        uint32_t col) // D
{

    uint32_t blockElementCnt = 32U / sizeof(L1Type);
    if constexpr (IsSameType<L1Type, int4b_t>::value) {
        blockElementCnt = 64U;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = col / blockElementCnt;
    intriParams.blockLen = rowAct;
    intriParams.dstStride = dstRowStride;
    intriParams.srcStride = srcRowStride;
    DataCopy(l1Tensor, gmTensor, intriParams);
}

template<typename L1Type>
__aicore__ inline void GmCopyInToL1HasRopePA(LocalTensor<L1Type>& nopeTensor, LocalTensor<L1Type>& ropeTensor,
                                GlobalTensor<L1Type>& nopeGmTensor, GlobalTensor<L1Type>& ropeGmTensor,
                                GlobalTensor<int32_t>& blockTableGm, KVLAYOUT KvLayout,
                                const PAShape &shape,
                                const PAShape &ropeShape,
                                const Position &startPos)
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Offset;
    uint32_t blockElementCnt = 32U / sizeof(L1Type);

    while(copyFinishRowCnt < shape.copyRowNum){
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint64_t remainRowCnt = curS2Idx % shape.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum){
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim;
        uint64_t keyRopeOffset = idInBlockTable * ropeShape.blockSize * ropeShape.headNum * ropeShape.headDim;
        if (KvLayout == KVLAYOUT::NZ) {
            offset += static_cast<uint64_t>(startPos.n2Idx * shape.blockSize * shape.headDim) + remainRowCnt * blockElementCnt + startPos.dIdx * shape.blockSize;
            keyRopeOffset += static_cast<uint64_t>(startPos.n2Idx * ropeShape.blockSize * ropeShape.headDim) + remainRowCnt * blockElementCnt + startPos.dIdx * ropeShape.blockSize;
            LocalTensor<L1Type> tmpNopeDstTensor = nopeTensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = nopeGmTensor[offset];
            DataCopyGmNZToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, (shape.copyRowNumAlign - copyRowCnt), (shape.blockSize - copyRowCnt), shape.actHeadDim);

            LocalTensor<L1Type> tmpRopeDstTensor = ropeTensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpRopeSrcTensor = ropeGmTensor[keyRopeOffset];
            DataCopyGmNZToL1(tmpRopeDstTensor, tmpRopeSrcTensor, copyRowCnt, (ropeShape.copyRowNumAlign - copyRowCnt), (ropeShape.blockSize - copyRowCnt), ropeShape.actHeadDim);
        } else {
            uint64_t dStride = shape.headDim;
            uint64_t dRopeStride = ropeShape.headDim;
            if (KvLayout == KVLAYOUT::BBH) {
                offset += static_cast<uint64_t>(startPos.n2Idx * shape.headDim) + remainRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
                keyRopeOffset += static_cast<uint64_t>(startPos.n2Idx * ropeShape.headDim) + remainRowCnt * ropeShape.headDim * ropeShape.headNum;
                dStride = shape.headDim * shape.headNum;
                dRopeStride = ropeShape.headDim * ropeShape.headNum;
            } else{
                offset += static_cast<uint64_t>(startPos.n2Idx * shape.headDim * shape.blockSize) + remainRowCnt * shape.headDim + startPos.dIdx;
                keyRopeOffset += static_cast<uint64_t>(startPos.n2Idx * ropeShape.headDim * ropeShape.blockSize) + remainRowCnt * ropeShape.headDim;
            }

            uint32_t dValue = shape.actHeadDim;
            uint32_t srcDValue = dStride;
            uint32_t dRopeValue = ropeShape.actHeadDim;
            uint32_t srcRopeDValue = dRopeStride;
            LocalTensor<L1Type> tmpNopeDstTensor = nopeTensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = nopeGmTensor[offset];
            DataCopyGmNDToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);

            LocalTensor<L1Type> tmpRopeDstTensor = ropeTensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpRopeSrcTensor = ropeGmTensor[keyRopeOffset];
            DataCopyGmNDToL1(tmpRopeDstTensor, tmpRopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dRopeValue, srcRopeDValue);
        }
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template<typename L1Type>
__aicore__ inline void GmCopyInToL1PA(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                GlobalTensor<int32_t>& blockTableGm, KVLAYOUT KvLayout,
                                const PAShape &shape, const Position &startPos)
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Offset;
    uint32_t blockElementCnt = 32U / sizeof(L1Type);
    while(copyFinishRowCnt < shape.copyRowNum){
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint64_t remainRowCnt = curS2Idx % shape.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum){
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim;
        if (KvLayout == KVLAYOUT::NZ) {
            offset += static_cast<uint64_t>(startPos.n2Idx * shape.blockSize * shape.headDim) + remainRowCnt * blockElementCnt + startPos.dIdx * shape.blockSize;

            LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
            DataCopyGmNZToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, (shape.copyRowNumAlign - copyRowCnt), (shape.blockSize - copyRowCnt), shape.actHeadDim);
        } else {
            uint64_t dStride = shape.headDim;
            if (KvLayout == KVLAYOUT::BBH) {
                offset += static_cast<uint64_t>(startPos.n2Idx * shape.headDim) + remainRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
                dStride = shape.headDim * shape.headNum;
            } else {
                offset += static_cast<uint64_t>(startPos.n2Idx * shape.headDim * shape.blockSize) + remainRowCnt * shape.headDim + startPos.dIdx;
            }

            uint32_t dValue = shape.actHeadDim;
            uint32_t srcDValue = dStride;
            LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
            DataCopyGmNDToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        }
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue;
    gm2L1Nd2NzParams.srcNdMatrixStride = 0;
    gm2L1Nd2NzParams.srcDValue = srcDValue;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
        IsSameType<INPUT_T, hifloat8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 31) >> 5 << 5;
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1;
    gm2L1Nd2NzParams.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2NzGS1Merge(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t ndNum, uint32_t nValue, uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = ndNum;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue;
    gm2L1Nd2NzParams.srcNdMatrixStride = srcNdMatrixStride;
    gm2L1Nd2NzParams.srcDValue = srcDValue;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
        IsSameType<INPUT_T, hifloat8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 31) >> 5 << 5;
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 15) >> 4 << 4;
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 15) >> 4 << 4;
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1;
    gm2L1Nd2NzParams.dstNzMatrixStride = nValue * 32 / sizeof(INPUT_T);
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
#endif
