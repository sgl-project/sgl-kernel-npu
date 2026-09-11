









/*!
 * \file buffer_manager.h

 */
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include "buffer.h"

// L1  TPosition::A1
// L0A TPosition::A2
// L0B TPosition::B2
// L0C TPosition::CO1
// UB  TPosition::VECIN
namespace fa_base_matmul {
template<BufferType bufferType>
class BufferManager {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t size) {
        static_assert(bufferType != BufferType::GM, "GM should use workspace.");
        TBuf<BufferInfo<bufferType>::Position> tbuf;
        pipe->InitBuffer(tbuf, size);
        mem_ = tbuf.template Get<uint8_t>();
    }

    __aicore__ inline void Init(__gm__ uint8_t* workspace) {
        static_assert(bufferType == BufferType::GM, "BufferType should be GM.");
        mem_.SetGlobalBuffer((__gm__ uint8_t*)workspace);
    }

    template<SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline Buffer<bufferType, syncType> AllocBuffer(uint32_t size) {
        TensorType temp = mem_[offset_];
        offset_ += size;
        return Buffer<bufferType, syncType>(temp, size);
    }

    template<SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline void FreeBuffer(Buffer<bufferType, syncType> &buffer){
    }
private:
    uint32_t offset_ = 0;
    TensorType mem_;
};
}
#endif
