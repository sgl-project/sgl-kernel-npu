#include "kernel_operator.h"

constexpr uint32_t TOPK_TILE_LEN = 64;
#define SLOT_LINE_SHIFT 3
#define SLOT_LINE_ELEMS (1U << SLOT_LINE_SHIFT)
#define SLOT_LINE_MASK (SLOT_LINE_ELEMS - 1U)
#define SLOT_LINES_PER_TILE (TOPK_TILE_LEN + 1U)
#define SENTINEL_LINE_OFFSET (TOPK_TILE_LEN * SLOT_LINE_ELEMS)
#define SENTINEL_LINE_OFFSET_BYTES (SENTINEL_LINE_OFFSET * sizeof(int32_t))
#define MTE_BATCH_LEN 8
#define AIV_PIPELINE_DEPTH 2

class KernelSlotMapLookup
{
public:
    __aicore__ inline KernelSlotMapLookup() {}

    __aicore__ inline void Init(GM_ADDR slot_map, GM_ADDR req_indices, GM_ADDR topk_indices, GM_ADDR token_on_device,
                                GM_ADDR device_token_pos, uint32_t size, uint32_t max_context_len, uint32_t bs,
                                uint32_t topk, AscendC::TPipe *pipe)
    {
        this->size = size;
        this->max_context_len = max_context_len;
        this->bs = bs;
        this->topk = topk;

        slotMapGm.SetGlobalBuffer((__gm__ int32_t *)slot_map, static_cast<uint64_t>(size) * max_context_len);
        reqIndicesGm.SetGlobalBuffer((__gm__ int32_t *)req_indices, bs);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices, static_cast<uint64_t>(bs) * topk);
        tokenOnDeviceGm.SetGlobalBuffer((__gm__ int32_t *)token_on_device, static_cast<uint64_t>(bs) * topk);
        deviceTokenPosGm.SetGlobalBuffer((__gm__ int32_t *)device_token_pos, static_cast<uint64_t>(bs) * topk);

        pipe->InitBuffer(topkIdxBuf, AIV_PIPELINE_DEPTH * TOPK_TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(posResultBuf, AIV_PIPELINE_DEPTH * TOPK_TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(tokenResultBuf, AIV_PIPELINE_DEPTH * TOPK_TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(slotLineBuf, AIV_PIPELINE_DEPTH * SLOT_LINES_PER_TILE * SLOT_LINE_ELEMS * sizeof(int32_t));
        pipe->InitBuffer(mteOffsetBuf, AIV_PIPELINE_DEPTH * TOPK_TILE_LEN * sizeof(uint32_t));
    }

    // Pipeline: overlap current tile's MTE2 with previous tile's V+MTE3.
    //
    //   iter 0:  MTE2[0]
    //   iter 1:  MTE2[1]  ||  V[0] + MTE3[0]
    //   iter 2:  MTE2[0]  ||  V[1] + MTE3[1]
    //   ...
    //   drain:                 V[last] + MTE3[last]
    __aicore__ inline void Process()
    {
        const uint32_t workerNum = AscendC::GetBlockNum();
        const uint32_t workerIdx = AscendC::GetBlockIdx();

        const uint32_t tilesPerBatch = (topk + TOPK_TILE_LEN - 1) / TOPK_TILE_LEN;
        const uint32_t taskNum = bs * tilesPerBatch;
        uint32_t taskBegin = 0;
        uint32_t taskEnd = 0;
        SplitRange(0, taskNum, workerNum, workerIdx, taskBegin, taskEnd);

        const uint32_t totalTasks = taskEnd - taskBegin;
        if (totalTasks == 0) {
            return;
        }

        AscendC::LocalTensor<int32_t> topkBaseLocal = topkIdxBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> posBaseLocal = posResultBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> tokenBaseLocal = tokenResultBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> slotLineBaseLocal = slotLineBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> mteOffsetBaseLocal = mteOffsetBuf.Get<uint32_t>();

        for (uint32_t buf = 0; buf < AIV_PIPELINE_DEPTH; ++buf) {
            const uint32_t sentinelBase = buf * SLOT_LINES_PER_TILE * SLOT_LINE_ELEMS + SENTINEL_LINE_OFFSET;
            AscendC::Duplicate(slotLineBaseLocal[sentinelBase], static_cast<int32_t>(-1), SLOT_LINE_ELEMS);
        }

        uint32_t slotTileLen[AIV_PIPELINE_DEPTH];
        uint32_t slotGmOffset[AIV_PIPELINE_DEPTH];
        bool slotSkipV[AIV_PIPELINE_DEPTH];

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);

        for (uint32_t iter = 0; iter < totalTasks + 1; ++iter) {
            const uint32_t curBuf = iter % AIV_PIPELINE_DEPTH;
            const uint32_t prevBuf = (iter + AIV_PIPELINE_DEPTH - 1) % AIV_PIPELINE_DEPTH;

            // ========== Previous tile: UB -> Vector -> UB -> MTE3 -> GM ==========
            if (iter > 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(prevBuf);

                AscendC::LocalTensor<int32_t> prevPosLocal = posBaseLocal[prevBuf * TOPK_TILE_LEN];
                AscendC::LocalTensor<int32_t> prevTokenLocal = tokenBaseLocal[prevBuf * TOPK_TILE_LEN];
                AscendC::LocalTensor<int32_t> prevSlotLineLocal =
                    slotLineBaseLocal[prevBuf * SLOT_LINES_PER_TILE * SLOT_LINE_ELEMS];
                AscendC::LocalTensor<uint32_t> prevMteOffsetLocal = mteOffsetBaseLocal[prevBuf * TOPK_TILE_LEN];

                // V: Gather slot_id from UB lines, compute tokenOnDevice = min(slot_id+1, 1)
                if (!slotSkipV[prevBuf]) {
                    for (uint32_t g = 0; g < slotTileLen[prevBuf]; g += MTE_BATCH_LEN) {
                        AscendC::Gather(prevPosLocal[g], prevSlotLineLocal, prevMteOffsetLocal[g],
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(MTE_BATCH_LEN));
                    }
                    AscendC::Adds(prevTokenLocal, prevPosLocal, static_cast<int32_t>(1), TOPK_TILE_LEN);
                    AscendC::Mins(prevTokenLocal, prevTokenLocal, static_cast<int32_t>(1), TOPK_TILE_LEN);
                }

                // MTE3: UB -> GM (tokenOnDevice + deviceTokenPos)
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(prevBuf);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(prevBuf);
                AscendC::DataCopy(tokenOnDeviceGm[slotGmOffset[prevBuf]], prevTokenLocal, slotTileLen[prevBuf]);
                AscendC::DataCopy(deviceTokenPosGm[slotGmOffset[prevBuf]], prevPosLocal, slotTileLen[prevBuf]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(prevBuf);
            }

            if (iter >= totalTasks) {
                break;
            }

            // ========== Current tile: GM -> MTE2 -> UB ==========
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(curBuf);

            const uint32_t task = taskBegin + iter;
            const uint32_t b = task / tilesPerBatch;
            const uint32_t tileIdx = task % tilesPerBatch;
            const uint32_t kBegin = tileIdx * TOPK_TILE_LEN;
            uint32_t kEnd = kBegin + TOPK_TILE_LEN;
            if (kEnd > topk) {
                kEnd = topk;
            }
            const uint32_t tileLen = kEnd - kBegin;
            const uint32_t gmOffset = b * topk + kBegin;

            slotTileLen[curBuf] = tileLen;
            slotGmOffset[curBuf] = gmOffset;

            AscendC::LocalTensor<int32_t> topkLocal = topkBaseLocal[curBuf * TOPK_TILE_LEN];
            AscendC::LocalTensor<int32_t> posLocal = posBaseLocal[curBuf * TOPK_TILE_LEN];
            AscendC::LocalTensor<int32_t> tokenLocal = tokenBaseLocal[curBuf * TOPK_TILE_LEN];
            AscendC::LocalTensor<int32_t> slotLineLocal =
                slotLineBaseLocal[curBuf * SLOT_LINES_PER_TILE * SLOT_LINE_ELEMS];
            AscendC::LocalTensor<uint32_t> mteOffsetLocal = mteOffsetBaseLocal[curBuf * TOPK_TILE_LEN];

            // Invalid req_id: fill defaults directly in UB, skip V stage
            const int32_t req_id = reqIndicesGm.GetValue(b);
            if (req_id < 0 || static_cast<uint32_t>(req_id) >= size) {
                AscendC::Duplicate(tokenLocal, static_cast<int32_t>(0), TOPK_TILE_LEN);
                AscendC::Duplicate(posLocal, static_cast<int32_t>(-1), TOPK_TILE_LEN);
                slotSkipV[curBuf] = true;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(curBuf);
                continue;
            }
            slotSkipV[curBuf] = false;

            // MTE2: topkIndicesGm -> UB
            AscendC::DataCopy(topkLocal, topkIndicesGm[gmOffset], tileLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(curBuf);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(curBuf);

            // MTE2: slotMapGm -> UB (64x 32B point lookups, one 8-elem line per index)
            const uint32_t slotRowOffset = static_cast<uint32_t>(req_id) * max_context_len;
            for (uint32_t i = 0; i < tileLen; ++i) {
                const int32_t topkIdx = topkLocal.GetValue(i);
                const uint32_t slotLineBase = i << SLOT_LINE_SHIFT;
                if (topkIdx < 0 || static_cast<uint32_t>(topkIdx) >= max_context_len) {
                    mteOffsetLocal.SetValue(i, SENTINEL_LINE_OFFSET_BYTES);
                } else {
                    const uint32_t topkIdxU = static_cast<uint32_t>(topkIdx);
                    const uint32_t lineBase = topkIdxU & ~SLOT_LINE_MASK;
                    const uint32_t lineOffset = topkIdxU & SLOT_LINE_MASK;
                    mteOffsetLocal.SetValue(i, (slotLineBase + lineOffset) * sizeof(int32_t));
                    AscendC::DataCopy(slotLineLocal[slotLineBase], slotMapGm[slotRowOffset + lineBase],
                                      SLOT_LINE_ELEMS);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(curBuf);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    }

private:
    __aicore__ inline void SplitRange(uint32_t rangeBegin, uint32_t rangeEnd, uint32_t workerNum, uint32_t workerIdx,
                                      uint32_t &taskBegin, uint32_t &taskEnd) const
    {
        const uint32_t rangeLen = rangeEnd - rangeBegin;
        const uint32_t baseLen = rangeLen / workerNum;
        const uint32_t extra = rangeLen - baseLen * workerNum;
        const uint32_t prefixExtra = workerIdx < extra ? workerIdx : extra;
        taskBegin = rangeBegin + workerIdx * baseLen + prefixExtra;
        taskEnd = taskBegin + baseLen + (workerIdx < extra ? 1 : 0);
    }

    AscendC::GlobalTensor<int32_t> slotMapGm;
    AscendC::GlobalTensor<int32_t> reqIndicesGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int32_t> tokenOnDeviceGm;
    AscendC::GlobalTensor<int32_t> deviceTokenPosGm;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> topkIdxBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> posResultBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tokenResultBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> slotLineBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> mteOffsetBuf;

    uint32_t size = 0;
    uint32_t max_context_len = 0;
    uint32_t bs = 0;
    uint32_t topk = 0;
};

extern "C" __global__ __aicore__ void slot_map_lookup(GM_ADDR slot_map, GM_ADDR req_indices, GM_ADDR topk_indices,
                                                      GM_ADDR token_on_device, GM_ADDR device_token_pos, uint32_t size,
                                                      uint32_t max_context_len, uint32_t bs, uint32_t topk)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelSlotMapLookup kernel;
    AscendC::TPipe pipe;
    kernel.Init(slot_map, req_indices, topk_indices, token_on_device, device_token_pos, size, max_context_len, bs, topk,
                &pipe);
    kernel.Process();
}
