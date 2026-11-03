#include "kernel_operator.h"

constexpr uint32_t QUEUE_DEPTH = 2;

class RestoreShortHistory
{
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR table, GM_ADDR requests, GM_ADDR lengths,
                                GM_ADDR first, GM_ADDR oldBegin, GM_ADDR oldEnd, GM_ADDR nextBegin, GM_ADDR nextEnd,
                                GM_ADDR nextFirst, uint32_t batch, uint32_t rows, uint32_t tableRows, uint32_t context,
                                uint32_t tableStride, uint32_t stride, uint32_t srcRows, uint32_t bytes,
                                uint32_t capacity, uint32_t reqStride, uint32_t lenStride, uint32_t firstStride,
                                uint32_t types, AscendC::TPipe *pipe)
    {
        this->batch = batch;
        this->rows = rows;
        this->tableRows = tableRows;
        this->context = context;
        this->tableStride = tableStride;
        this->stride = stride;
        this->srcRows = srcRows;
        this->bytes = bytes;
        this->capacity = capacity;
        this->reqStride = reqStride;
        this->lenStride = lenStride;
        this->firstStride = firstStride;
        this->types = types;
        srcGm.SetGlobalBuffer((__gm__ uint8_t *)src, srcRows * bytes);
        dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst, rows * stride * bytes);
        table32.SetGlobalBuffer((__gm__ int32_t *)table);
        table64.SetGlobalBuffer((__gm__ int64_t *)table);
        req32.SetGlobalBuffer((__gm__ int32_t *)requests);
        req64.SetGlobalBuffer((__gm__ int64_t *)requests);
        len32.SetGlobalBuffer((__gm__ int32_t *)lengths);
        len64.SetGlobalBuffer((__gm__ int64_t *)lengths);
        first32.SetGlobalBuffer((__gm__ int32_t *)first);
        first64.SetGlobalBuffer((__gm__ int64_t *)first);
        oldBeginGm.SetGlobalBuffer((__gm__ int32_t *)oldBegin, rows);
        oldEndGm.SetGlobalBuffer((__gm__ int32_t *)oldEnd, rows);
        nextBeginGm.SetGlobalBuffer((__gm__ int32_t *)nextBegin, rows);
        nextEndGm.SetGlobalBuffer((__gm__ int32_t *)nextEnd, rows);
        nextFirstGm.SetGlobalBuffer((__gm__ int32_t *)nextFirst, rows);
        pipe->InitBuffer(queue, QUEUE_DEPTH, (bytes + 31U) & ~31U);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        for (uint32_t i = 0; i < batch; ++i) {
            const int64_t row = (types & 1U) ? req64.GetValue(i * reqStride) : req32.GetValue(i * reqStride);
            const int64_t length = (types & 2U) ? len64.GetValue(i * lenStride) : len32.GetValue(i * lenStride);
            const int64_t first = (types & 4U) ? first64.GetValue(i * firstStride) : first32.GetValue(i * firstStride);
            if (row <= 0 || row >= rows || row >= tableRows) continue;
            if (length <= 0 || length > context || length > 2147483647LL || first < 0 || first > length ||
                first > 2147483647LL) {
                if (core == 0) {
                    nextBeginGm.SetValue(row, -1);
                    nextEndGm.SetValue(row, -1);
                    nextFirstGm.SetValue(row, -1);
                }
                continue;
            }
            if (core == 0) {
                nextBeginGm.SetValue(row, length < capacity ? 0 : -1);
                nextEndGm.SetValue(row, static_cast<int32_t>(length));
                nextFirstGm.SetValue(row, static_cast<int32_t>(first));
            }
            if (length >= capacity) continue;
            int64_t start = oldBeginGm.GetValue(row) == 0 ? oldEndGm.GetValue(row) : 0;
            if (start < 0) start = 0;
            if (start > first) start = first;
            int64_t end = first;
            if (end > stride) end = stride;
            if (end > context) end = context;
            uint32_t offsets[QUEUE_DEPTH] = {0, 0};
            uint32_t head = 0, tail = 0, queued = 0;
            for (int64_t pos = start + core; pos < end; pos += cores) {
                uint64_t tableOffset = static_cast<uint64_t>(row) * tableStride + pos;
                int64_t physical = (types & 8U) ? table64.GetValue(tableOffset) : table32.GetValue(tableOffset);
                if (physical < 0 || physical >= srcRows) continue;
                if (queued == QUEUE_DEPTH) {
                    CopyOut(offsets[head]);
                    head = (head + 1) % QUEUE_DEPTH;
                    --queued;
                }
                CopyIn(static_cast<uint32_t>(physical) * bytes);
                offsets[tail] = (static_cast<uint32_t>(row) * stride + static_cast<uint32_t>(pos)) * bytes;
                tail = (tail + 1) % QUEUE_DEPTH;
                ++queued;
            }
            while (queued) {
                CopyOut(offsets[head]);
                head = (head + 1) % QUEUE_DEPTH;
                --queued;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset)
    {
        auto local = queue.AllocTensor<uint8_t>();
        AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(local, srcGm[offset], params, pad);
        queue.EnQue(local);
    }
    __aicore__ inline void CopyOut(uint32_t offset)
    {
        auto local = queue.DeQue<uint8_t>();
        AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(dstGm[offset], local, params);
        queue.FreeTensor(local);
    }
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, QUEUE_DEPTH> queue;
    AscendC::GlobalTensor<uint8_t> srcGm, dstGm;
    AscendC::GlobalTensor<int32_t> table32, req32, len32, first32, oldBeginGm, oldEndGm, nextBeginGm, nextEndGm,
        nextFirstGm;
    AscendC::GlobalTensor<int64_t> table64, req64, len64, first64;
    uint32_t batch, rows, tableRows, context, tableStride, stride, srcRows, bytes, capacity, reqStride, lenStride,
        firstStride, types;
};

extern "C" __global__ __aicore__ void restore_short_kv_history(
    GM_ADDR src, GM_ADDR dst, GM_ADDR table, GM_ADDR requests, GM_ADDR lengths, GM_ADDR first, GM_ADDR oldBegin,
    GM_ADDR oldEnd, GM_ADDR nextBegin, GM_ADDR nextEnd, GM_ADDR nextFirst, uint32_t batch, uint32_t rows,
    uint32_t tableRows, uint32_t context, uint32_t tableStride, uint32_t stride, uint32_t srcRows, uint32_t bytes,
    uint32_t capacity, uint32_t reqStride, uint32_t lenStride, uint32_t firstStride, uint32_t types)
{
    AscendC::TPipe pipe;
    RestoreShortHistory kernel;
    kernel.Init(src, dst, table, requests, lengths, first, oldBegin, oldEnd, nextBegin, nextEnd, nextFirst, batch, rows,
                tableRows, context, tableStride, stride, srcRows, bytes, capacity, reqStride, lenStride, firstStride,
                types, &pipe);
    kernel.Process();
}
