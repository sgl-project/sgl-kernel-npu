// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.
#ifndef SGL_KERNEL_NPU_SHORT_RESIDENT_COPY_H
#define SGL_KERNEL_NPU_SHORT_RESIDENT_COPY_H

#include "kernel_operator.h"

// This stage is called only by AIV tasks. All tasks must subsequently reach
// the same MIX barrier before attention reads the destination.
namespace ShortResidentCopy {
using namespace AscendC;
constexpr int64_t FEATURES = 576;
constexpr uint32_t BYTES = FEATURES * sizeof(uint16_t);

template <typename T>
class Copy
{
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR current, GM_ADDR host, GM_ADDR cache, GM_ADDR table, GM_ADDR slots,
                                GM_ADDR positions, GM_ADDR valid, GM_ADDR qEnds, GM_ADDR hbmTable, GM_ADDR lengths,
                                GM_ADDR cacheTokens, GM_ADDR oldBegin, GM_ADDR oldEnd, uint32_t batch, uint32_t rows,
                                uint32_t context, uint32_t capacity, uint32_t sourceRows, uint32_t currentRows,
                                uint32_t hbmPages, uint32_t cores, bool allocateQueue)
    {
        current_.SetGlobalBuffer((__gm__ T *)current);
        host_.SetGlobalBuffer((__gm__ T *)host);
        cache_.SetGlobalBuffer((__gm__ T *)cache);
        table_.SetGlobalBuffer((__gm__ int32_t *)table);
        slots_.SetGlobalBuffer((__gm__ int32_t *)slots);
        positions_.SetGlobalBuffer((__gm__ int64_t *)positions);
        valid_.SetGlobalBuffer((__gm__ uint8_t *)valid);
        qEnds_.SetGlobalBuffer((__gm__ int32_t *)qEnds);
        hbmTable_.SetGlobalBuffer((__gm__ int32_t *)hbmTable);
        lengths_.SetGlobalBuffer((__gm__ int32_t *)lengths);
        cacheTokens_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        oldBegin_.SetGlobalBuffer((__gm__ int32_t *)oldBegin);
        oldEnd_.SetGlobalBuffer((__gm__ int32_t *)oldEnd);
        batch_ = batch;
        rows_ = rows;
        context_ = context;
        capacity_ = capacity;
        sourceRows_ = sourceRows;
        currentRows_ = currentRows;
        hbmPages_ = hbmPages;
        core_ = GetBlockIdx();
        cores_ = cores;
        if (allocateQueue && core_ < cores_) pipe->InitBuffer(queue_, 2, BYTES);
    }

    __aicore__ inline void Process()
    {
        if (core_ >= cores_) return;
        for (uint32_t b = 0; b < batch_; ++b) {
            const int32_t len = lengths_.GetValue(b);
            if (len <= 0 || cacheTokens_.GetValue(b) != 0) continue;
            const int32_t req = slots_.GetValue(b);
            const int32_t startQ = b == 0 ? 0 : qEnds_.GetValue(b - 1);
            const int32_t endQ = qEnds_.GetValue(b);
            ASSERT_MSG(req > 0 && req < rows_ && startQ >= 0 && endQ > startQ && endQ <= currentRows_ &&
                           len < capacity_ && len <= context_,
                       "invalid active short request metadata");
            if (req <= 0 || req >= rows_ || startQ < 0 || endQ <= startQ || endQ > currentRows_ || len >= capacity_ ||
                len > context_)
                continue;
            const int64_t first = positions_.GetValue(startQ);
            ASSERT_MSG(first >= 0 && first + endQ - startQ == len, "invalid short query range");
            if (first < 0 || first + endQ - startQ != len) continue;
            for (int32_t q = startQ; q < endQ; ++q) {
                ASSERT_MSG(valid_.GetValue(q) != 0 && positions_.GetValue(q) == first + q - startQ,
                           "active current KV rows must be valid and consecutive");
            }
            int32_t oldEnd = oldBegin_.GetValue(req) == 0 ? oldEnd_.GetValue(req) : 0;
            if (oldEnd < 0) oldEnd = 0;
            if (oldEnd > first) oldEnd = static_cast<int32_t>(first);
            for (int64_t pos = oldEnd + core_; pos < first; pos += cores_) {
                const int32_t physical = table_.GetValue(static_cast<int64_t>(req) * context_ + pos);
                ASSERT_MSG(physical >= 0 && physical < sourceRows_, "short history source out of bounds");
                if (physical < 0 || physical >= sourceRows_) continue;
                const int64_t destination = Destination(b, pos);
                if (destination >= 0) Transfer(host_, static_cast<int64_t>(physical) * FEATURES, destination);
            }
            for (int32_t q = startQ + core_; q < endQ; q += cores_) {
                if (valid_.GetValue(q) == 0 || positions_.GetValue(q) != first + q - startQ) continue;
                const int64_t destination = Destination(b, first + q - startQ);
                if (destination >= 0) Transfer(current_, static_cast<int64_t>(q) * FEATURES, destination);
            }
        }
    }

    __aicore__ inline void ValidateCurrent()
    {
        if (core_ >= cores_) return;
        for (uint32_t b = core_; b < batch_; b += cores_) {
            if (lengths_.GetValue(b) <= 0) continue;
            const int32_t req = slots_.GetValue(b);
            const int32_t start = b == 0 ? 0 : qEnds_.GetValue(b - 1);
            const int32_t end = qEnds_.GetValue(b);
            ASSERT_MSG(req > 0 && req < rows_ && start >= 0 && end > start && end <= currentRows_ && end - start <= 16,
                       "invalid active current request metadata");
            if (req <= 0 || req >= rows_ || start < 0 || end <= start || end > currentRows_) continue;
            const int64_t first = positions_.GetValue(start);
            ASSERT_MSG(first >= 0 && first + end - start <= context_, "current positions exceed context");
            for (int32_t q = start; q < end; ++q) {
                ASSERT_MSG(valid_.GetValue(q) != 0 && positions_.GetValue(q) == first + q - start,
                           "active current KV rows must be valid and consecutive");
            }
        }
    }

    // Call only after every copying task has crossed the barrier. Each row is
    // committed by one AIV task, and the next launch observes stream ordering.
    __aicore__ inline void Commit()
    {
        if (core_ >= cores_) return;
        for (uint32_t b = core_; b < batch_; b += cores_) {
            const int32_t req = slots_.GetValue(b);
            const int32_t len = lengths_.GetValue(b);
            if (req <= 0 || req >= rows_ || len <= 0) continue;
            if (cacheTokens_.GetValue(b) != 0) {
                oldBegin_.SetValue(req, -1);
                oldEnd_.SetValue(req, -1);
            } else {
                const int32_t startQ = b == 0 ? 0 : qEnds_.GetValue(b - 1);
                const int32_t endQ = qEnds_.GetValue(b);
                bool ok = len < capacity_ && len <= context_ && startQ >= 0 && endQ > startQ && endQ <= currentRows_;
                int64_t first = -1;
                if (ok) {
                    first = positions_.GetValue(startQ);
                    ok = first >= 0 && first + endQ - startQ == len;
                }
                if (ok) {
                    for (int32_t q = startQ; q < endQ; ++q) {
                        if (valid_.GetValue(q) == 0 || positions_.GetValue(q) != first + q - startQ) ok = false;
                    }
                    int32_t begin = oldBegin_.GetValue(req) == 0 ? oldEnd_.GetValue(req) : 0;
                    if (begin < 0) begin = 0;
                    if (begin > first) begin = static_cast<int32_t>(first);
                    for (int32_t pos = begin; pos < first; ++pos) {
                        const int32_t physical = table_.GetValue(static_cast<int64_t>(req) * context_ + pos);
                        if (physical < 0 || physical >= sourceRows_) ok = false;
                    }
                }
                oldBegin_.SetValue(req, ok ? 0 : -1);
                oldEnd_.SetValue(req, ok ? len : -1);
            }
        }
    }

private:
    __aicore__ inline int64_t Destination(uint32_t batch, int64_t pos)
    {
        const int64_t page = pos / 128;
        ASSERT_MSG(page >= 0 && page < capacity_ / 128, "short resident logical destination out of bounds");
        if (page < 0 || page >= capacity_ / 128) return -1;
        const int32_t physical = hbmTable_.GetValue(static_cast<int64_t>(batch) * (capacity_ / 128) + page);
        ASSERT_MSG(physical >= 0 && physical < hbmPages_, "short resident destination out of bounds");
        if (physical < 0 || physical >= hbmPages_) return -1;
        return (static_cast<int64_t>(physical) * 128 + pos % 128) * FEATURES;
    }
    __aicore__ inline void Transfer(GlobalTensor<T> &src, int64_t srcOffset, int64_t dstOffset)
    {
        auto local = queue_.AllocTensor<T>();
        DataCopyExtParams params{1, BYTES, 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, src[srcOffset], params, pad);
        queue_.EnQue(local);
        local = queue_.DeQue<T>();
        DataCopyPad(cache_[dstOffset], local, params);
        queue_.FreeTensor(local);
    }
    GlobalTensor<T> current_, host_, cache_;
    GlobalTensor<int32_t> table_, slots_, qEnds_, hbmTable_, lengths_, cacheTokens_, oldBegin_, oldEnd_;
    GlobalTensor<int64_t> positions_;
    GlobalTensor<uint8_t> valid_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> queue_;
    uint32_t batch_, rows_, context_, capacity_, sourceRows_, currentRows_, hbmPages_, core_, cores_;
};
}  // namespace ShortResidentCopy
#endif
