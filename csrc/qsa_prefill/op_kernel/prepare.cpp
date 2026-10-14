#include "kernel_operator.h"
using namespace AscendC;

class BlockRuns
{
public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR t, GM_ADDR req, GM_ADDR flags, GM_ADDR s, GM_ADDR r, GM_ADDR c,
                                uint32_t rows_, uint32_t blocks_, uint32_t tableStride, uint32_t length_, int32_t base_,
                                uint32_t cap_, uint32_t outWidth_)
    {
        rows = rows_;
        blocks = blocks_;
        length = length_;
        base = base_;
        cap = cap_;
        outWidth = outWidth_;
        pipe.InitBuffer(bb, 2048 * 4);
        pipe.InitBuffer(tb, 4096 * 4);
        pipe.InitBuffer(sb, 4096 * 4);
        pipe.InitBuffer(rb, 4096 * 4);
        pipe.InitBuffer(cb, 16 * 4);
        pipe.InitBuffer(fb, 4096 * 4);
        pipe.InitBuffer(eb, 4096 * 4);
        pipe.InitBuffer(mb, 512);
        bi = bb.Get<int32_t>();
        ti = tb.Get<int32_t>();
        ss = sb.Get<int32_t>();
        rr = rb.Get<int32_t>();
        cc = cb.Get<int32_t>();
        gb.SetGlobalBuffer((__gm__ int32_t *)b);
        gt.SetGlobalBuffer((__gm__ int32_t *)t);
        gf.SetGlobalBuffer((__gm__ int32_t *)flags);
        gs.SetGlobalBuffer((__gm__ int32_t *)s);
        gr.SetGlobalBuffer((__gm__ int32_t *)r);
        gc.SetGlobalBuffer((__gm__ int32_t *)c);
        GlobalTensor<int64_t> rq;
        rq.SetGlobalBuffer((__gm__ int64_t *)req);
        tableOffset = static_cast<uint64_t>(rq.GetValue(0)) * tableStride;
    }
    __aicore__ inline bool IsLinear(LocalTensor<int32_t> data, int32_t first, uint32_t n)
    {
        uint32_t aligned = n / 64 * 64;
        auto actual = fb.Get<float>();
        auto expected = eb.Get<float>();
        auto mask = mb.Get<uint8_t>();
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Cast(actual, data, RoundMode::CAST_NONE, aligned);
        CreateVecIndex(expected, static_cast<float>(first), aligned);
        PipeBarrier<PIPE_V>();
        Sub(actual, actual, expected, aligned);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, actual, 0.0f, CMPMODE::EQ, aligned);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        auto wide = mask.ReinterpretCast<uint64_t>();
        for (uint32_t z = 0; z < aligned / 64; ++z)
            if (wide.GetValue(z) != ~uint64_t(0)) return false;
        for (uint32_t z = aligned; z < n; ++z)
            if (data.GetValue(z) != first + static_cast<int32_t>(z)) return false;
        return true;
    }
    __aicore__ inline void Consume(int32_t start, uint32_t n)
    {
        if (!n) return;
        DataCopyExtParams cp{1, n * 4u, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(ti, gt[tableOffset + static_cast<uint32_t>(start)], cp, pad);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        // A generic fast path: verify every slot, never infer continuity from endpoints.
        int32_t first = ti.GetValue(0);
        bool contiguous = n >= 64 && first >= 0 && cap <= 16777216u && static_cast<uint64_t>(first) + n <= cap;
        if (contiguous) contiguous = IsLinear(ti, first, n);
        if (contiguous) {
            uint32_t consumed = 0;
            while (consumed < n) {
                int32_t x = first + consumed;
                if (!runSize || x != previous + 1 || count % 256 == 0) {
                    if (runSize) rr.SetValue(runBegin, runSize);
                    runBegin = count;
                    runSize = 0;
                    ss.SetValue(runBegin, x);
                }
                uint32_t take = min(n - consumed, 256 - count % 256);
                runSize += take;
                count += take;
                consumed += take;
                previous = first + consumed - 1;
            }
        } else
            for (uint32_t j = 0; j < n; ++j) {
                int32_t x = ti.GetValue(j);
                if (x < 0 || static_cast<uint32_t>(x) >= cap) continue;
                if (!runSize || x != previous + 1 || count % 256 == 0) {
                    if (runSize) rr.SetValue(runBegin, runSize);
                    runBegin = count;
                    runSize = 0;
                    ss.SetValue(runBegin, x);
                }
                ++runSize;
                ++count;
                previous = x;
            }
        SetFlag<HardEvent::S_MTE2>(0);
        WaitFlag<HardEvent::S_MTE2>(0);
    }
    __aicore__ inline void Process()
    {
        for (uint32_t group = GetBlockIdx(); group < (rows + 15) / 16; group += GetBlockNum()) {
            const uint32_t begin = group * 16, queries = min(16u, rows - begin);
            DataCopy(cc, gf[begin * 16], 16);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
            for (uint32_t sub = 0; sub < queries;) {
                const bool shared16 = cc.GetValue(1) != 0;
                const bool shared8 = sub % 8 == 0 && sub + 7 < queries && cc.GetValue(2 + sub / 8) != 0;
                const bool shared4 = sub % 4 == 0 && sub + 3 < queries && cc.GetValue(4 + sub / 4) != 0;
                const bool shared2 = sub % 2 == 0 && sub + 1 < queries && cc.GetValue(8 + sub / 2) != 0;
                const uint32_t step = shared16 ? 16 : (shared8 ? 8 : (shared4 ? 4 : (shared2 ? 2 : 1)));
                const uint32_t row = begin + sub;
                DataCopyExtParams cp{1, blocks * 4u, 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                DataCopyPad(bi, gb[static_cast<uint64_t>(row) * blocks], cp, pad);
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                count = 0;
                runBegin = 0;
                runSize = 0;
                previous = -2;
                int32_t logicalStart = 0;
                uint32_t logicalSize = 0;
                int32_t firstBlock = bi.GetValue(0);
                if (firstBlock >= 0 && static_cast<int64_t>(firstBlock) + blocks <= 16777216 &&
                    (static_cast<int64_t>(firstBlock) + blocks) * 4 <= length && IsLinear(bi, firstBlock, blocks)) {
                    logicalStart = firstBlock * 4;
                    logicalSize = blocks * 4;
                } else
                    for (uint32_t j = 0; j < blocks; ++j) {
                        int32_t b = bi.GetValue(j);
                        if (b < 0 || static_cast<int64_t>(b) * 4 >= length) continue;
                        int32_t first = b * 4;
                        uint32_t n = min(4u, length - static_cast<uint32_t>(first));
                        if (logicalSize && first != logicalStart + static_cast<int32_t>(logicalSize)) {
                            Consume(logicalStart, logicalSize);
                            logicalSize = 0;
                        }
                        if (!logicalSize) logicalStart = first;
                        logicalSize += n;
                    }
                Consume(logicalStart, logicalSize);
                if (runSize) rr.SetValue(runBegin, runSize);
                cc.SetValue(0, count);
                SetFlag<HardEvent::S_MTE3>(0);
                WaitFlag<HardEvent::S_MTE3>(0);
                DataCopy(gs[static_cast<uint64_t>(row) * outWidth], ss, outWidth);
                DataCopy(gr[static_cast<uint64_t>(row) * outWidth], rr, outWidth);
                // Counts remain available for every original query. All exact-sharing
                // flags at the group leader are preserved; unused row metadata is ignored.
                for (uint32_t j = 0; j < step; ++j)
                    DataCopy(gc[(row + j) * 16], cc, 16);
                SetFlag<HardEvent::MTE3_S>(0);
                WaitFlag<HardEvent::MTE3_S>(0);
                sub += step;
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> bb, tb, sb, rb, cb, fb, eb, mb;
    LocalTensor<int32_t> bi, ti, ss, rr, cc;
    GlobalTensor<int32_t> gb, gt, gf, gs, gr, gc;
    uint32_t rows, blocks, length, cap, outWidth, count, runBegin, runSize;
    int32_t base, previous;
    uint64_t tableOffset;
};
extern "C" __global__ __aicore__ void qsa_prefill_prepare_kernel(GM_ADDR b, GM_ADDR t, GM_ADDR req, GM_ADDR flags,
                                                                 GM_ADDR s, GM_ADDR r, GM_ADDR c, uint32_t rows,
                                                                 uint32_t blocks, uint32_t tableStride, uint32_t length,
                                                                 int32_t base, uint32_t cap, uint32_t outWidth)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BlockRuns op;
    op.Init(b, t, req, flags, s, r, c, rows, blocks, tableStride, length, base, cap, outWidth);
    op.Process();
}
