









/*!
 * \file buffers_policy.h

 */
#ifndef BUFFERS_POLICY_H
#define BUFFERS_POLICY_H

#include "buffer_manager.h"
#include "buffer.h"
#define NUM_2 2
#define NUM_3 3
#define NUM_4 4

namespace fa_base_matmul {
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicySingleBuffer {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        buffer_ = bufferManager.template AllocBuffer<syncType>(size);
        buffer_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager){
        buffer_.UnInit();
        bufferManager.FreeBuffer(buffer_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get(){
        return buffer_;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetPre(){
        return Get();
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused(){
        return Get();
    }
private:
    Buffer<bufferType, syncType> buffer_;
};


template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        ping_ = bufferManager.template AllocBuffer<syncType>(size);
        pong_ = bufferManager.template AllocBuffer<syncType>(size);

        ping_.Init();
        pong_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager){
        ping_.UnInit();
        pong_.UnInit();

        bufferManager.FreeBuffer(ping_);
        bufferManager.FreeBuffer(pong_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_) { // 1
            flag1_ = 0;
            return ping_;
        } else { // 0
            flag1_ = 1;
            return pong_;
        }
    }


    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_) { // 0->1
            return pong_;
        } else { // 1->0
            return ping_;
        }
    }


    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) {
            flag2_ = 1;
            return pong_;
        } else {
            flag2_ = 0;
            return ping_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused(bool isNextS2IdxNoChange) {
        if (isNextS2IdxNoChange) {
            if (flag2_ == 0) {
                return pong_;
            } else {
                return ping_;
            }
        } else {
            return GetReused();
        }
    }

private:
    Buffer<bufferType, syncType> ping_;
    Buffer<bufferType, syncType> pong_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};


template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicy3buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_ == 0) {
            flag1_ = 1;
            return a_;
        } else if (flag1_ == 1) {
            flag1_ = NUM_2;
            return b_;
        } else {
            flag1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetVec() { // mixcore architecture
        if (flag1_vec1_ == 0) {
            flag1_vec1_ = 1;
            return a_;
        } else if (flag1_vec1_ == 1) {
            flag1_vec1_ = NUM_2;
            return b_;
        } else {
            flag1_vec1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetCube() { // mixcore architecture
        if (flag1_bmm2_ == 0) {
            flag1_bmm2_ = 1;
            return a_;
        } else if (flag1_bmm2_ == 1) {
            flag1_bmm2_ = NUM_2;
            return b_;
        } else {
            flag1_bmm2_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) {
            return a_;
        } else {
            return b_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) {
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1){
            flag2_ = NUM_2;
            return b_;
        } else {
            flag2_ = 0;
            return c_;
        }
    }
private:
    Buffer<bufferType, syncType> a_;
    Buffer<bufferType, syncType> b_;
    Buffer<bufferType, syncType> c_;
    uint32_t flag1_ = 0;
    uint32_t flag1_vec1_ = 0;
    uint32_t flag1_bmm2_ = 0;
    uint32_t flag2_ = 0;
};


template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class BuffersPolicy4buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        a_ = bufferManager.template AllocBuffer<syncType>(size);
        b_ = bufferManager.template AllocBuffer<syncType>(size);
        c_ = bufferManager.template AllocBuffer<syncType>(size);
        d_ = bufferManager.template AllocBuffer<syncType>(size);

        a_.Init();
        b_.Init();
        c_.Init();
        d_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
        d_.UnInit();

        bufferManager.FreeBuffer(a_);
        bufferManager.FreeBuffer(b_);
        bufferManager.FreeBuffer(c_);
        bufferManager.FreeBuffer(d_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get(uint32_t id) {
        uint32_t flag = id % 4;
        if (flag == 0) {
            return a_;
        } else if (flag == 1) {
            return b_;
        } else if (flag == 2) { // 2:c_
            return c_;
        } else {
            return d_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        auto& buffer = Get(head_);
        head_++;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        auto& buffer = Get(used_);
        used_ = (used_ - tail_ + 1) % (head_ - tail_) + tail_;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType> &GetFree() {
        if (tail_ == used_) {
            used_++;
        }
        auto& buffer = Get(tail_);
        tail_++;
        return buffer;
    }
private:
    Buffer<bufferType, syncType> a_;
    Buffer<bufferType, syncType> b_;
    Buffer<bufferType, syncType> c_;
    Buffer<bufferType, syncType> d_;
    uint32_t tail_ = 0;
    uint32_t head_ = 0;
    uint32_t used_ = 0;
};

template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Matrix2x2BufferPolicy { // 4buffer

// MracBuffer:memory address with row first, alloc/use/free with column first
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size) {
        bufferM0k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM0k1_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k1_ = bufferManager.template AllocBuffer<syncType>(size);

        bufferM0k0_.Init();
        bufferM0k1_.Init();
        bufferM1k0_.Init();
        bufferM1k1_.Init();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType> &bufferManager) {
        bufferM0k0_.UnInit();
        bufferM0k1_.UnInit();
        bufferM1k0_.UnInit();
        bufferM1k1_.UnInit();

        bufferManager.FreeBuffer(bufferM0k0_);
        bufferManager.FreeBuffer(bufferM0k1_);
        bufferManager.FreeBuffer(bufferM1k0_);
        bufferManager.FreeBuffer(bufferM1k1_);
    }

    __aicore__ inline void SetMExtent(int32_t mExtent) {
        aIdx_ = -1;
        amIdx_ = (amIdx_ + mSize_ - 1) % mSize_;
        akIdx_ = 0;

        uIdx_ = -1;
        umIdx_ = (umIdx_ + mSize_ - 1) % mSize_;
        ukIdx_ = 0;

        fIdx_ = -1;
        fmIdx_ = (fmIdx_ + mSize_ - 1) % mSize_;
        fkIdx_ = 0;

        mExtent_ = mExtent;
    }

    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++;
        return GetBuffer(aIdx_, amIdx_, akIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++;
        return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++;
        return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType> &PeekNextK() {
        return PeekBuffer(amIdx_, (1 - akIdx_));
    }
private:
    __aicore__ inline Buffer<bufferType, syncType> &GetBuffer(int32_t xIdx, int32_t &mIdx, int32_t &kIdx) {

        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if (mIdx == 1 && kIdx == 0) {
            return bufferM1k0_;
        } else {
            return bufferM1k1_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType> &PeekBuffer(int32_t mIdx, int32_t kIdx) {

        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if ((mIdx == 1) && (kIdx == 0)) {
            return bufferM1k0_;
        } else { // mIdx == 1 && kIdx == 1
            return bufferM1k1_;
        }
    }

    Buffer<bufferType, syncType> bufferM0k0_;
    Buffer<bufferType, syncType> bufferM0k1_;
    Buffer<bufferType, syncType> bufferM1k0_;
    Buffer<bufferType, syncType> bufferM1k1_;
    int32_t mSize_ = 2;
    int32_t kSize_ = 2;

    // Alloc
    int32_t aIdx_ = -1;
    int32_t amIdx_ = 0;
    int32_t akIdx_ = 0;
    // Reuse
    int32_t uIdx_ = -1;
    int32_t umIdx_ = 0;
    int32_t ukIdx_ = 0;

    // Free
    int32_t fIdx_ = -1;
    int32_t fmIdx_ = 0;
    int32_t fkIdx_ = 0;

    int32_t mExtent_ = 0;
};
}
#endif
