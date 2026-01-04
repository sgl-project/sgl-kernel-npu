/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * ZBCCL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ZBCCL_REF_H
#define ZBCCL_REF_H

#include <atomic>
#include <cstdint>

namespace zbccl {

class ZReferable
{
public:
    ZReferable() = default;
    virtual ~ZReferable() = default;

    inline void IncreaseRef()
    {
        refCount_.fetch_add(1, std::memory_order_relaxed);
    }

    inline void DecreaseRef()
    {
        // delete itself if reference count equal to 0
        if (refCount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

protected:
    std::atomic<int32_t> refCount_{0};
};

template <typename T>
class ZRef
{
public:
    // constructor
    ZRef() noexcept = default;

    // fix: can't be explicit
    ZRef(T *newObj) noexcept
    {
        // if new obj is not null, increase reference count and assign to mObj
        // else nothing need to do as mObj is nullptr by default
        if (newObj != nullptr) {
            newObj->IncreaseRef();
            mObj = newObj;
        }
    }

    ZRef(const ZRef<T> &other) noexcept
    {
        // if other's obj is not null, increase reference count and assign to mObj
        // else nothing need to do as mObj is nullptr by default
        if (other.mObj != nullptr) {
            other.mObj->IncreaseRef();
            mObj = other.mObj;
        }
    }

    ZRef(ZRef<T> &&other) noexcept : mObj(std::exchange(other.mObj, nullptr))
    {
        // move constructor
        // since this mObj is null, just exchange
    }

    // de-constructor
    ~ZRef()
    {
        if (mObj != nullptr) {
            mObj->DecreaseRef();
        }
    }

    // operator =
    inline ZRef<T> &operator=(T *newObj)
    {
        this->Set(newObj);
        return *this;
    }

    inline ZRef<T> &operator=(const ZRef<T> &other)
    {
        if (this != &other) {
            this->Set(other.mObj);
        }
        return *this;
    }

    ZRef<T> &operator=(ZRef<T> &&other) noexcept
    {
        if (this != &other) {
            auto tmp = mObj;
            mObj = std::__exchange(other.mObj, nullptr);
            if (tmp != nullptr) {
                tmp->DecreaseRef();
            }
        }
        return *this;
    }

    // equal operator
    inline bool operator==(const ZRef<T> &other) const
    {
        return mObj == other.mObj;
    }

    inline bool operator==(T *other) const
    {
        return mObj == other;
    }

    inline bool operator!=(const ZRef<T> &other) const
    {
        return mObj != other.mObj;
    }

    inline bool operator!=(T *other) const
    {
        return mObj != other;
    }

    // get operator and set
    inline T *operator->() const
    {
        return mObj;
    }

    inline T *Get() const
    {
        return mObj;
    }

    inline void Set(T *newObj)
    {
        if (newObj == mObj) {
            return;
        }

        if (newObj != nullptr) {
            newObj->IncreaseRef();
        }

        if (mObj != nullptr) {
            mObj->DecreaseRef();
        }

        mObj = newObj;
    }

private:
    T *mObj = nullptr;
};

template <typename C, typename... ARGS>
inline ZRef<C> ZMakeRef(ARGS... args)
{
    return new (std::nothrow) C(args...);
}

}  // namespace zbccl
#endif  // ZBCCL_REF_H
