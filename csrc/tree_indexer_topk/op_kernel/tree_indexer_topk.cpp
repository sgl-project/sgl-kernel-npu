// SPDX-License-Identifier: BSD-3-Clause
#include "kernel_operator.h"
#include "../../lightning_indexer/op_kernel/lightning_indexer_common.h"
#include "../../lightning_indexer/op_kernel/lightning_indexer_vector.h"

namespace {
using namespace AscendC;
constexpr uint32_t TILE = 1024;

// The sort determines only a score threshold. A separate ascending-index
// compaction handles ties, independent of Sort32/MrgSort tie ordering.
class TreeIndexerTopK {
public:
    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR output, uint32_t rows, uint32_t slots, uint32_t k)
    {
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
        rows_ = rows;
        slots_ = slots;
        k_ = k;
        keep_ = (k + 31) / 32 * 32;
        // Fixed UB footprint < 48 KiB, independent of history length.
        pipe_.InitBuffer(sourceBuf_, 2 * TILE * sizeof(float));
        pipe_.InitBuffer(sortBuf_, 2 * TILE * sizeof(float));
        pipe_.InitBuffer(winnerBuf_, 2 * TILE * sizeof(float));
        pipe_.InitBuffer(mergeBuf_, 4 * TILE * sizeof(float));
        pipe_.InitBuffer(compactBuf_, TILE * sizeof(int32_t));
        pipe_.InitBuffer(gtBuf_, TILE / 8);
        pipe_.InitBuffer(eqBuf_, TILE / 8);
        source_ = sourceBuf_.Get<float>();
        sorted_ = sortBuf_.Get<float>();
        winners_ = winnerBuf_.Get<float>();
        merged_ = mergeBuf_.Get<float>();
        compact_ = compactBuf_.Get<int32_t>();
        gt_ = gtBuf_.Get<uint8_t>();
        eq_ = eqBuf_.Get<uint8_t>();
        // Bit-cast instead of converting the integer value to float.
        union { uint32_t bits; float value; } inf;
        inf.bits = 0xff800000U;
        negInf_ = inf.value;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = GetBlockIdx(); row < rows_; row += GetBlockNum()) {
            RunRow(row);
        }
    }

private:
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        event_t id = static_cast<event_t>(pipe_.FetchEventID(E));
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }

    __aicore__ inline uint32_t PopCount(uint32_t x)
    {
        x -= (x >> 1) & 0x55555555U;
        x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
        x = (x + (x >> 4)) & 0x0f0f0f0fU;
        return (x * 0x01010101U) >> 24;
    }

    __aicore__ inline void Load(uint32_t row, uint32_t begin)
    {
        uint32_t count = slots_ - begin < TILE ? slots_ - begin : TILE;
        Duplicate(source_, negInf_, TILE);
        Sync<HardEvent::V_MTE2>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((8 - count % 8) % 8), negInf_};
        DataCopyPad(source_, scores_[static_cast<uint64_t>(row) * slots_ + begin], copy, pad);
        Sync<HardEvent::MTE2_V>();
        // Match the Triton validity predicate: NaN and -inf are absent.
        CompareScalar(gt_, source_, negInf_, CMPMODE::GT, TILE);
        PipeBarrier<PIPE_V>();
        Select(source_, gt_, source_, negInf_, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE);
        CreateVecIndex(source_[TILE].ReinterpretCast<int32_t>(), static_cast<int32_t>(begin), TILE);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunRow(uint32_t row)
    {
        // Streaming top-keep. Keeping ceil(k/32)*32 also handles small k and
        // the native merge instruction's 32-element list alignment.
        for (uint32_t begin = 0; begin < slots_; begin += TILE) {
            Load(row, begin);
            sglang::npu_kernel::LIServiceVec::SortAll(source_, sorted_, TILE);
            PipeBarrier<PIPE_V>();
            if (begin == 0) {
                DataCopy(winners_, source_, 2 * keep_);
            } else {
                MrgSort4Info info;
                info.elementLengths[0] = keep_;
                info.elementLengths[1] = keep_;
                info.elementLengths[2] = keep_;
                info.elementLengths[3] = keep_;
                info.ifExhaustedSuspension = false;
                info.validBit = 0b0011;
                info.repeatTimes = 1;
                MrgSortSrcList<float> lists;
                lists.src1 = winners_;
                lists.src2 = source_;
                lists.src3 = winners_;
                lists.src4 = source_;
                MrgSort<float>(merged_, lists, info);
                PipeBarrier<PIPE_V>();
                DataCopy(winners_, merged_, 2 * keep_);
            }
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_S>();
        const float threshold = winners_.GetValue(2 * (k_ - 1));

        // Count strict winners before admitting any threshold ties.
        uint32_t greater = 0;
        for (uint32_t begin = 0; begin < slots_; begin += TILE) {
            Load(row, begin);
            CompareScalar(gt_, source_, threshold, CMPMODE::GT, TILE);
            Sync<HardEvent::V_S>();
            auto words = gt_.ReinterpretCast<uint32_t>();
            for (uint32_t w = 0; w < TILE / 32; ++w) greater += PopCount(words.GetValue(w));
        }
        uint32_t tiesLeft = k_ - greater;
        uint32_t written = 0;
        for (uint32_t begin = 0; begin < slots_; begin += TILE) {
            Load(row, begin);
            CompareScalar(gt_, source_, threshold, CMPMODE::GT, TILE);
            CompareScalar(eq_, source_, threshold, CMPMODE::EQ, TILE);
            Sync<HardEvent::V_S>();
            auto gtWords = gt_.ReinterpretCast<uint32_t>();
            auto eqWords = eq_.ReinterpretCast<uint32_t>();
            for (uint32_t w = 0; w < TILE / 32; ++w) {
                uint32_t ties = threshold > negInf_ ? eqWords.GetValue(w) : 0;
                uint32_t count = PopCount(ties);
                if (count > tiesLeft) {
                    uint32_t chosen = 0;
                    for (uint32_t n = 0; n < tiesLeft; ++n) {
                        uint32_t bit = ties & (0U - ties);
                        chosen |= bit;
                        ties &= ties - 1;
                    }
                    ties = chosen;
                    count = tiesLeft;
                }
                tiesLeft -= count;
                gtWords.SetValue(w, gtWords.GetValue(w) | ties);
            }
            Sync<HardEvent::S_V>();
            uint64_t selected = 0;
            GatherMask(compact_, source_[TILE].ReinterpretCast<int32_t>(), gt_.ReinterpretCast<uint32_t>(),
                       true, TILE, {1, 1, 0, 0}, selected);
            Sync<HardEvent::V_MTE3>();
            if (selected > 0) {
                DataCopyExtParams copy{1, static_cast<uint32_t>(selected * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(output_[static_cast<uint64_t>(row) * k_ + written], compact_, copy);
            }
            Sync<HardEvent::MTE3_V>();
            written += static_cast<uint32_t>(selected);
        }
        if (written < k_) {
            Duplicate(compact_, static_cast<int32_t>(-1), TILE);
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams copy{1, static_cast<uint32_t>((k_ - written) * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(output_[static_cast<uint64_t>(row) * k_ + written], compact_, copy);
            Sync<HardEvent::MTE3_V>();
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> sourceBuf_, sortBuf_, winnerBuf_, mergeBuf_, compactBuf_, gtBuf_, eqBuf_;
    LocalTensor<float> source_, sorted_, winners_, merged_;
    LocalTensor<int32_t> compact_;
    LocalTensor<uint8_t> gt_, eq_;
    GlobalTensor<float> scores_;
    GlobalTensor<int32_t> output_;
    uint32_t rows_, slots_, k_, keep_;
    float negInf_;
};
}  // namespace

extern "C" __global__ __aicore__ void tree_indexer_topk(
    GM_ADDR scores, GM_ADDR output, uint32_t rows, uint32_t slots, uint32_t k)
{
    TreeIndexerTopK op;
    op.Init(scores, output, rows, slots, k);
    op.Process();
}
