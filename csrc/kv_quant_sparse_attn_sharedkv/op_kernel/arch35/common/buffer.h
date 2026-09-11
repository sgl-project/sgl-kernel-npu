









/*!
 * \file buffer.h

 */
#ifndef BUFFER_H
#define BUFFER_H
#include<type_traits>
#include"lib/matmul_intf.h"
#include"kernel_event.h"
#include"kernel_common.h"
#include"kernel_tpipe.h"
using namespace AscendC;
namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 16)


#define AIV0_AIV1_OFFSET 16

enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
    UB = 4,
    GM = 5,
};

enum class SyncType {
    NO_SYNC,
    INNER_CORE_SYNC,
    CROSS_CORE_SYNC_FORWARD,
    CROSS_CORE_SYNC_BOTH,
    CROSS_CORE_SYNC_BACKWARD,
};

constexpr uint32_t INVALID_CROSS_CORE_EVENT_ID = 16;
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;

template<BufferType Type>
struct BufferInfo{

    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE2_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::M_FIX;
        } else if constexpr (Type == BufferType::GM) {
            return HardEvent::MTE2_S;
        }
    }

    __aicore__ const static constexpr HardEvent ProdWaitConsStatus() {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE1_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::FIX_M;
        } else if constexpr (Type == BufferType::GM) {
            return HardEvent::S_MTE2;
        }
    }

    __aicore__ const static constexpr TPosition GetTPosition() {
        if constexpr (Type == BufferType::L1) {
            return TPosition::A1;
        } else if constexpr (Type == BufferType::L0A) {
            return TPosition::A2;
        } else if constexpr (Type == BufferType::L0B) {
            return TPosition::B2;
        } else if constexpr (Type == BufferType::L0C) {
            return TPosition::CO1;
        } else if constexpr (Type == BufferType::UB) {
            return TPosition::VECIN;
        } else if constexpr (Type == BufferType::GM) {
            return TPosition::GM;
        }
    }

    static constexpr HardEvent EventP2C = ConsWaitProdStatus();
    static constexpr TPosition Position = GetTPosition();
};






template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Buffer {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;

    template <typename T>
    using TargetTensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<T>, LocalTensor<T>>;
public:
    __aicore__ inline Buffer() {}
    __aicore__ inline Buffer(TensorType tensor, uint32_t size) {
        tensor_ = tensor;
        size_ = size;
        if constexpr (syncType == SyncType::CROSS_CORE_SYNC_FORWARD) {
            id0_ = MAKE_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = MAKE_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
            id0_ = MAKE_ID;
            id1_ = MAKE_ID;
        } else {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        }
    }

    __aicore__ inline void Init() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>();
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
                SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    __aicore__ inline void UnInit() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(p2cEventId_);
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Wait() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    WaitFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_);
                } else {
                    WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                }
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Set() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    SetFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_);
                } else {
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                }
            }
        }
    }

    __aicore__ inline void WaitCrossCore() {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id0_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id0_);
            }
        } else if constexpr (bufferType == BufferType::L1) {

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                }
            }
        }
    }

    __aicore__ inline void SetCrossCore() {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {

            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {

            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id1_);
            }
        } else if constexpr (bufferType == BufferType::L1) {

            if ASCEND_IS_AIC {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_);
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_ + AIV0_AIV1_OFFSET);
                }
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
            }
        }
    }

    template<typename T>
    __aicore__ inline TargetTensorType<T> GetTensor() {
        return tensor_.template ReinterpretCast<T>();
    }

    template<typename T>
    __aicore__ inline TargetTensorType<T> GetTensor(uint64_t startindex) {
        TargetTensorType<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    TensorType tensor_;
    uint32_t size_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
    uint32_t id0_;
    uint32_t id1_;
};
}
#endif
