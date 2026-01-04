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
#ifndef ZBCCL_SMA_BLOCK_POOL_H
#define ZBCCL_SMA_BLOCK_POOL_H

#include "zbccl_sma.h"
#include "zbccl_sma_config.h"

namespace zbccl {
namespace sma {
namespace device {
/**
 * @brief Type of block type
 */
enum DeviceBlockType {
    BT_SMALL,
    BT_BIG,
};

class DeviceBlock;
class DeviceSegment;
class DeviceBlockPool;

using DeviceBlockPtr = ZRef<DeviceBlock>;
using DeviceSegmentPtr = ZRef<DeviceSegment>;
using DeviceBlockPoolPtr = ZRef<DeviceBlockPool>;

using StreamSet = ska::flat_hash_set<aclrtStream>;

/**
 * @brief DeviceBlock
 */
class DeviceBlock : public ZReferable
{
public:
    DeviceBlock(int32_t device, aclrtStream stream, size_t size, const DeviceBlockPoolPtr &pool, uintptr_t ptr)
        : deviceId_(device), stream_(stream), size_(size), requested_size_(0), pool_(pool), ptr_(ptr)
    {}

    DeviceBlock(int32_t device, aclrtStream stream, size_t size) : deviceId_(device), stream_(stream), size_(size) {}

    ~DeviceBlock() override = default;

private:
    int32_t deviceId_{-1};              // device Id
    aclrtStream stream_{nullptr};       // allocation stream
    StreamSet stream_uses_;             // streams on which the block was used
    size_t size_{0};                    // block size in bytes
    size_t requested_size_{0};          // memory originally requested
    DeviceBlockPoolPtr pool_{nullptr};  // owning memory pool
    uintptr_t ptr_{0};                  // memory address
    DeviceBlockPtr prev_{nullptr};      // prev block if split from a larger allocation
    DeviceBlockPtr next_{nullptr};      // next block if split from a larger allocation
    int32_t event_count_{0};            // number of outstanding CUDA events
    bool allocated_{false};             // in-use flag
    int64_t gc_count_base_{0};          // get_free_blocks_call_count when DeviceBlock is inserted

    friend struct DeviceBlockCompareBySize;
    friend struct DeviceAllocParams;
};

/**
 * @brief Comparator function of device block
 */
struct DeviceBlockCompareBySize {
    bool operator()(const DeviceBlockPtr &a, const DeviceBlockPtr &b) const
    {
        if (a->stream_ != b->stream_) {
            return (uintptr_t)a->stream_ < (uintptr_t)b->stream_;
        }

        if (a->size_ != b->size_) {
            return a->size_ < b->size_;
        }

        return a->ptr_ < b->ptr_;
    }
};

class DeviceSegment : public ZReferable
{
public:
};

class DeviceBlockPool : public ZReferable
{
public:
private:
};

struct DeviceAllocParams {
    DeviceAllocParams(int32_t device, size_t size, aclrtStream stream, DeviceBlockPoolPtr &pool, size_t alloc_size)
        : searchKey(device, stream, size), pool(pool), allocSize(alloc_size)
    {}

    int32_t device() const;
    aclrtStream stream() const;
    size_t size() const;

    DeviceBlock searchKey;
    DeviceBlockPoolPtr pool;
    size_t allocSize;
    DeviceBlockPtr block{nullptr};
    // StatTypes stat_types = {false};
    ZResult result{Z_OK};
};

inline int32_t DeviceAllocParams::device() const
{
    return searchKey.deviceId_;
}

inline aclrtStream DeviceAllocParams::stream() const
{
    return searchKey.stream_;
}

inline size_t DeviceAllocParams::size() const
{
    return searchKey.size_;
}

}  // namespace device
}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_BLOCK_POOL_H
