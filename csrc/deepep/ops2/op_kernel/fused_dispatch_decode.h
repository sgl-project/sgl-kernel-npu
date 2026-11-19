#ifndef FUSED_DISPATCH_DECODE_H
#define FUSED_DISPATCH_DECODE_H

#include "kernel_operator.h"
#include "moe_distribute_base.h"
#include "comm_args.h"
#include "notify_dispatch_tiling.h"

namespace FusedDispatchDecodeImpl {
using namespace AscendC;

// EP8, H4096, Top8. All ranks must use the same token count in [1,4].
// Use Notify's reserved ping-pong window, leaving Combine's storage untouched.
class FusedDispatchDecode {
    static constexpr uint32_t RANKS = 8, EXPERTS = 36, TOPK = 8, HIDDEN = 4096;
    static constexpr uint32_t TOKEN_OFFSET = 4096, TOKEN_BYTES = HIDDEN + 32;
    TPipe pipe_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    TBuf<> input_, fp32_, abs_, token_, ids_, flag_, meta_;
    GM_ADDR windows_[RANKS];
    GM_ADDR x_;
    GM_ADDR idsIn_;
    GM_ADDR values_;
    GM_ADDR scales_;
    GM_ADDR ends_;
    GM_ADDR source_;
    GM_ADDR counts_;
    GM_ADDR total_;
    uint32_t rank_, block_, tokens_;
    uint64_t epoch_;

    template <HardEvent Event>
    __aicore__ inline void Fence()
    {
        int id = static_cast<int>(pipe_.FetchEventID(Event));
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }

    template <typename T>
    __aicore__ inline void Store(GM_ADDR address, LocalTensor<T> src, uint32_t n)
    {
        GlobalTensor<T> dst;
        dst.SetGlobalBuffer((__gm__ T *)address);
        DataCopyPad(dst, src, {1U, static_cast<uint32_t>(n * sizeof(T)), 0U, 0U, 0U});
    }

    __aicore__ inline void Rendezvous(uint64_t phase)
    {
        // Call only after every local core has completed its data transfers.
        // Read-completion acknowledgements prevent a fast rank from reusing a
        // ping-pong slot while a slow receiver is still reading its payload.
        if (block_ == 0) {
            auto flag = flag_.Get<uint64_t>();
            uint64_t expected = epoch_ * 2 + phase;
            flag(0) = expected;
            Fence<HardEvent::S_MTE3>();
            for (uint32_t r = 0; r < RANKS; ++r)
                Store(windows_[r] + rank_ * 32, flag, 1);
            Fence<HardEvent::MTE3_MTE2>();
            GlobalTensor<uint64_t> ready;
            ready.SetGlobalBuffer((__gm__ uint64_t *)windows_[rank_]);
            bool done;
            do {
                DataCopy(flag, ready, RANKS * 4);
                Fence<HardEvent::MTE2_S>();
                done = true;
                for (uint32_t r = 0; r < RANKS; ++r)
                    if (flag(r * 4) < expected) done = false;
                Fence<HardEvent::S_MTE2>();
            } while (!done);
        }
        SyncAll<true>();
    }

    __aicore__ inline void StageToken()
    {
        if (block_ == 0) {
            GlobalTensor<int32_t> inputIds;
            inputIds.SetGlobalBuffer((__gm__ int32_t *)idsIn_);
            auto ids = ids_.Get<int32_t>();
            DataCopy(ids, inputIds, tokens_ * TOPK);
            Fence<HardEvent::MTE2_MTE3>();
            Store(windows_[rank_] + Moe::IPC_DATA_OFFSET, ids, tokens_ * TOPK);
            Fence<HardEvent::MTE3_S>();
        }
        if (block_ >= tokens_) return;
        GlobalTensor<bfloat16_t> x;
        x.SetGlobalBuffer((__gm__ bfloat16_t *)x_ + block_ * HIDDEN);
        auto in = input_.Get<bfloat16_t>();
        auto fp = fp32_.Get<float>();
        auto magnitude = abs_.Get<float>();
        auto output = token_.Get<int8_t>();
        DataCopy(in, x, HIDDEN);
        Fence<HardEvent::MTE2_V>();
        Cast(fp, in, RoundMode::CAST_NONE, HIDDEN);
        PipeBarrier<PIPE_V>();
        Abs(magnitude, fp, HIDDEN);
        PipeBarrier<PIPE_V>();
        // Match CamMoeDispatchNormal's quantization and rounding exactly.
        Max(magnitude, magnitude[64], magnitude, 64, HIDDEN / 64 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
        WholeReduceMax(magnitude, magnitude, 64, 1, 8, 1, 8);
        Fence<HardEvent::V_S>();
        float multiplier = 127.0f / (magnitude(0) + 1e-12f);
        Fence<HardEvent::S_V>();
        Muls(fp, fp, multiplier, HIDDEN);
        PipeBarrier<PIPE_V>();
        auto integers = fp.ReinterpretCast<int32_t>();
        auto halves = fp.ReinterpretCast<half>();
        Cast(integers, fp, RoundMode::CAST_RINT, HIDDEN);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.0f);
        PipeBarrier<PIPE_V>();
        Cast(halves, integers, RoundMode::CAST_ROUND, HIDDEN);
        PipeBarrier<PIPE_V>();
        Cast(output, halves, RoundMode::CAST_TRUNC, HIDDEN);
        Fence<HardEvent::V_S>();
        output.ReinterpretCast<float>()(HIDDEN / sizeof(float)) = 1.0f / multiplier;
        Fence<HardEvent::S_MTE3>();
        Store(windows_[rank_] + Moe::IPC_DATA_OFFSET + TOKEN_OFFSET + block_ * TOKEN_BYTES,
              output, TOKEN_BYTES);
        Fence<HardEvent::MTE3_S>();
    }

    __aicore__ inline void ReceiveExpert()
    {
        if (block_ >= EXPERTS) return;
        auto ids = ids_.Get<int32_t>();
        uint32_t idsPerRank = tokens_ * TOPK;
        for (uint32_t r = 0; r < RANKS; ++r) {
            GlobalTensor<int32_t> remoteIds;
            remoteIds.SetGlobalBuffer((__gm__ int32_t *)(windows_[r] + Moe::IPC_DATA_OFFSET));
            DataCopy(ids[r * idsPerRank], remoteIds, idsPerRank);
        }
        Fence<HardEvent::MTE2_S>();
        int32_t expert = rank_ * EXPERTS + block_;
        int32_t firstExpert = rank_ * EXPERTS;
        int32_t row = 0;
        for (uint32_t i = 0; i < RANKS * idsPerRank; ++i) {
            int32_t id = ids(i);
            if (id >= firstExpert && id < expert) ++row;
        }
        auto metadata = meta_.Get<int32_t>();
        auto triple = metadata;
        auto cumulative = metadata[8];
        auto groupEnd = metadata[16].ReinterpretCast<int64_t>();
        auto total = metadata[24];
        auto payload = token_.Get<int8_t>();
        for (uint32_t r = 0; r < RANKS; ++r) {
            for (uint32_t i = 0; i < idsPerRank; ++i) {
                if (ids(r * idsPerRank + i) != expert) continue;
                GlobalTensor<int8_t> remoteToken;
                remoteToken.SetGlobalBuffer((__gm__ int8_t *)(windows_[r] + Moe::IPC_DATA_OFFSET +
                                            TOKEN_OFFSET + (i / TOPK) * TOKEN_BYTES));
                DataCopy(payload, remoteToken, TOKEN_BYTES);
                Fence<HardEvent::MTE2_MTE3>();
                Store(values_ + row * HIDDEN, payload, HIDDEN);
                Store(scales_ + row * sizeof(float), payload.ReinterpretCast<float>()[HIDDEN / sizeof(float)], 1);
                triple(0) = r;
                triple(1) = i / TOPK;
                triple(2) = i % TOPK;
                Fence<HardEvent::S_MTE3>();
                Store(source_ + row * 3 * sizeof(int32_t), triple, 3);
                Fence<HardEvent::MTE3_S>();
                Fence<HardEvent::MTE3_MTE2>();
                ++row;
            }
            cumulative(r) = row;
        }
        groupEnd(0) = row;
        total(0) = row;
        Fence<HardEvent::S_MTE3>();
        Store(counts_ + block_ * RANKS * sizeof(int32_t), cumulative, RANKS);
        Store(ends_ + block_ * sizeof(int64_t), groupEnd, 1);
        if (block_ == EXPERTS - 1) Store(total_, total, 1);
        Fence<HardEvent::MTE3_S>();
    }

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR ids, GM_ADDR values, GM_ADDR scales,
                               GM_ADDR ends, GM_ADDR source, GM_ADDR counts, GM_ADDR total, GM_ADDR tiling)
    {
        REGISTER_TILING_DEFAULT(NotifyDispatchTilingData);
        GET_TILING_DATA_WITH_STRUCT(NotifyDispatchTilingData, data, tiling);
        rank_ = data.notifyDispatchInfo.rankId;
        tokens_ = data.notifyDispatchInfo.numTokens;
        block_ = GetBlockIdx();
        x_ = x; idsIn_ = ids; values_ = values; scales_ = scales;
        ends_ = ends; source_ = source; counts_ = counts; total_ = total;
        hccl_.InitV2(GetHcclContext<HCCL_GROUP_ID_0>(), &data);
        hccl_.SetCcTilingV2(offsetof(NotifyDispatchTilingData, mc2CcTiling1));
        uint64_t winSize = data.notifyDispatchInfo.totalWinSize;
        GlobalTensor<uint64_t> counter;
        counter.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsInAddr(rank_) + winSize -
                                Moe::STATE_SIZE * 3 + Moe::STATE_WIN_OFFSET));
        auto index = block_ * Moe::UB_ALIGN_SIZE;
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counter[index]);
        epoch_ = counter(index);
        if (epoch_ == 0) epoch_ = 1;
        counter(index) = epoch_ + 1;
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counter[index]);
        for (uint32_t r = 0; r < RANKS; ++r) {
            windows_[r] = hccl_.GetWindowsInAddr(r) + ((winSize - 4 * Moe::STATE_SIZE) / 2) * (epoch_ % 2);
#ifdef OPT_RANK_OFFSET
            windows_[r] += r * OPT_RANK_OFFSET;
#endif
        }
        pipe_.InitBuffer(input_, HIDDEN * sizeof(bfloat16_t));
        pipe_.InitBuffer(fp32_, HIDDEN * sizeof(float));
        pipe_.InitBuffer(abs_, HIDDEN * sizeof(float));
        pipe_.InitBuffer(token_, TOKEN_BYTES);
        pipe_.InitBuffer(ids_, RANKS * tokens_ * TOPK * sizeof(int32_t));
        pipe_.InitBuffer(flag_, RANKS * 32);
        pipe_.InitBuffer(meta_, 128);
    }

    __aicore__ inline void Process()
    {
        StageToken();
        SyncAll<true>();
        Rendezvous(1);
        ReceiveExpert();
        SyncAll<true>();
        Rendezvous(2);
        hccl_.Finalize();
    }
};
} // namespace FusedDispatchDecodeImpl
#endif
