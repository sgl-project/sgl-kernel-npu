#include "kernel_operator.h"
namespace custom_assign {

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TYPEBYPE_ID = 2;

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

template <typename T>
class AssignCacheOp
{
public:
    __aicore__ inline AssignCacheOp(){};
    __aicore__ inline void Init(__gm__ uint8_t *dstPtr, __gm__ uint8_t *srcPtr, __gm__ uint8_t *dstStartIdxPtr,
                                __gm__ uint8_t *dstEndIdxPtr, __gm__ uint8_t *srcStartIdxPtr,
                                __gm__ uint8_t *srcEndIdxPtr, __gm__ uint8_t *sync, __gm__ uint8_t *tilingPtr);
    __aicore__ inline void CopyElement(AscendC::LocalTensor<T> &dstTensor, AscendC::LocalTensor<T> &srcTensor,
                                       uint32_t bytes);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(__gm__ uint8_t *tilingPtr);

private:
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> dstGM_;
    AscendC::GlobalTensor<T> srcGM_;

    AscendC::GlobalTensor<int64_t> dstStartIdxGm_;
    AscendC::GlobalTensor<int64_t> dstEndIdxGm_;
    AscendC::GlobalTensor<int64_t> srcStartIdxGm_;
    AscendC::GlobalTensor<int64_t> srcEndIdxGm_;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf2_;
    AscendC::LocalTensor<T> tmpTensor1_;
    AscendC::LocalTensor<T> tmpTensor2_;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> vecOut_;
    AscendC::GlobalTensor<int32_t> syncGm_;

    uint32_t batchSize_;
    uint32_t tokenPoolLength_;
    uint32_t syncWorkspaceSize_;
    uint32_t ubSize_;
    uint32_t ubUsedBufSize_;
};

template <typename T>
__aicore__ inline void AssignCacheOp<T>::Init(__gm__ uint8_t *dstPtr, __gm__ uint8_t *srcPtr,
                                              __gm__ uint8_t *dstStartIdxPtr, __gm__ uint8_t *dstEndIdxPtr,
                                              __gm__ uint8_t *srcStartIdxPtr, __gm__ uint8_t *srcEndIdxPtr,
                                              __gm__ uint8_t *sync, __gm__ uint8_t *tilingPtr)
{
    this->ParseTilingData(tilingPtr);
    ubUsedBufSize_ = ubSize_ >> 2;  // make sure not overflow

    dstGM_.SetGlobalBuffer((__gm__ T *)dstPtr);
    srcGM_.SetGlobalBuffer((__gm__ T *)srcPtr);
    dstStartIdxGm_.SetGlobalBuffer((__gm__ int64_t *)dstStartIdxPtr);
    dstEndIdxGm_.SetGlobalBuffer((__gm__ int64_t *)dstEndIdxPtr);
    srcStartIdxGm_.SetGlobalBuffer((__gm__ int64_t *)srcStartIdxPtr);
    srcEndIdxGm_.SetGlobalBuffer((__gm__ int64_t *)srcEndIdxPtr);

    pipe_.InitBuffer(tmpBuf1_, ubUsedBufSize_);
    pipe_.InitBuffer(tmpBuf2_, ubUsedBufSize_);
    tmpTensor1_ = tmpBuf1_.Get<T>();
    tmpTensor2_ = tmpBuf2_.Get<T>();

    syncGm_.SetGlobalBuffer((__gm__ int32_t *)(sync), syncWorkspaceSize_);
    pipe_.InitBuffer(vecOut_, 1, 8 * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void AssignCacheOp<T>::CopyElement(AscendC::LocalTensor<T> &dstTensor,
                                                     AscendC::LocalTensor<T> &srcTensor, uint32_t bytes)
{
    for (uint32_t i = 0; i < bytes / sizeof(T); i++) {
        T tmp = srcTensor.GetValue(i);
        dstTensor.SetValue(i, tmp);
    }
}

template <typename T>
__aicore__ inline void AssignCacheOp<T>::Process()
{
    int32_t vecIdx = AscendC::GetBlockIdx();   // current vector core id
    int32_t coreNum = AscendC::GetBlockNum();  // total vector core number

    for (uint32_t batchId = vecIdx; batchId < batchSize_; batchId += coreNum) {
        uint32_t srcStartIdx = srcStartIdxGm_.GetValue(batchId);
        uint32_t srcEndIdx = srcEndIdxGm_.GetValue(batchId);
        uint32_t dstStartIdx = dstStartIdxGm_.GetValue(batchId);
        uint32_t dstEndIdx = dstEndIdxGm_.GetValue(batchId);
        uint32_t totalBytes = (srcEndIdx - srcStartIdx) * sizeof(T);
        uint32_t ubLoopNum = (totalBytes + ubUsedBufSize_ - 1) / ubUsedBufSize_;
        uint32_t tailBytes = totalBytes % ubUsedBufSize_;
        if (totalBytes > 0 && tailBytes == 0) {
            tailBytes = ubUsedBufSize_;
        }
        uint32_t loopOffset = 0;
        uint32_t ubDataNum = ubUsedBufSize_ / sizeof(T);

        // load src data & dst data from gm to ub, then copy to gm of dst data address
        for (uint32_t loopId = 0; loopId < ubLoopNum; loopId++) {
            uint16_t copyLen = ubDataNum * sizeof(T) / BLOCK_SIZE;
            AscendC::DataCopyParams copyParams = {1, copyLen, 0, 0};
            copyParams.blockLen = (loopId == ubLoopNum - 1) ? (tailBytes + BLOCK_SIZE - 1) / BLOCK_SIZE : copyLen;
            DataCopy(tmpTensor1_, srcGM_[srcStartIdx + loopOffset], copyParams);
            SET_FLAG(MTE2, MTE3, EVENT_ID1);
            WAIT_FLAG(MTE2, MTE3, EVENT_ID1);

            if (loopId == ubLoopNum - 1) {
                DataCopy(tmpTensor2_, dstGM_[batchId * tokenPoolLength_ + dstStartIdx + loopOffset], copyParams);
                SET_FLAG(MTE2, MTE3, EVENT_ID2);
                WAIT_FLAG(MTE2, MTE3, EVENT_ID2);
                CopyElement(tmpTensor2_, tmpTensor1_, tailBytes);
                if (batchId > 0) {
                    auto syncBuf = vecOut_.AllocTensor<int32_t>();
                    // wait for last kernel copy data from UB to GM
                    AscendC::IBWait(syncGm_, syncBuf, (vecIdx == 0) ? coreNum - 1 : vecIdx - 1, 0);
                    vecOut_.FreeTensor(syncBuf);
                }
                DataCopy(dstGM_[batchId * tokenPoolLength_ + dstStartIdx + loopOffset], tmpTensor2_, copyParams);
                SET_FLAG(MTE3, MTE2, EVENT_ID1);
                WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
                if (batchId != batchSize_ - 1) {
                    auto syncBuf = vecOut_.AllocTensor<int32_t>();
                    // publish event current kernel has finished copying date from UB to GM
                    AscendC::IBSet(syncGm_, syncBuf, vecIdx, 0);
                    vecOut_.FreeTensor(syncBuf);
                }
                break;
            }

            DataCopy(dstGM_[batchId * tokenPoolLength_ + dstStartIdx + loopOffset], tmpTensor1_, copyParams);
            SET_FLAG(MTE3, MTE2, EVENT_ID1);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
            loopOffset += ubDataNum;
        }
    }
}

template <typename T>
__aicore__ inline void AssignCacheOp<T>::ParseTilingData(__gm__ uint8_t *tilingPtr)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingPtr);

    int64_t locId = 0;
    batchSize_ = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
    locId++;
    tokenPoolLength_ = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
    // jump typeBytes field
    locId += 2;
    syncWorkspaceSize_ = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
    locId++;
    ubSize_ = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
}
}  // namespace custom_assign

extern "C" __global__ __aicore__ void assign_cache_op(GM_ADDR dstPtr, GM_ADDR srcPtr, GM_ADDR dstStartIdxPtr,
                                                      GM_ADDR dstEndIdxPtr, GM_ADDR srcStartIdxPtr,
                                                      GM_ADDR srcEndIdxPtr, GM_ADDR sync, GM_ADDR tilingPtr)
{
    uint32_t typeByte = ((__gm__ uint32_t *)tilingPtr)[custom_assign::TYPEBYPE_ID];
    if ASCEND_IS_AIV {
        if (typeByte == 1) {
            custom_assign::AssignCacheOp<int8_t> op;
            op.Init(dstPtr, srcPtr, dstStartIdxPtr, dstEndIdxPtr, srcStartIdxPtr, srcEndIdxPtr, sync, tilingPtr);
            op.Process();
        } else if (typeByte == 2) {
            custom_assign::AssignCacheOp<int16_t> op;
            op.Init(dstPtr, srcPtr, dstStartIdxPtr, dstEndIdxPtr, srcStartIdxPtr, srcEndIdxPtr, sync, tilingPtr);
            op.Process();
        } else if (typeByte == 4) {
            custom_assign::AssignCacheOp<int32_t> op;
            op.Init(dstPtr, srcPtr, dstStartIdxPtr, dstEndIdxPtr, srcStartIdxPtr, srcEndIdxPtr, sync, tilingPtr);
            op.Process();
        } else if (typeByte == 8) {
            custom_assign::AssignCacheOp<int64_t> op;
            op.Init(dstPtr, srcPtr, dstStartIdxPtr, dstEndIdxPtr, srcStartIdxPtr, srcEndIdxPtr, sync, tilingPtr);
            op.Process();
        }
    }
}
