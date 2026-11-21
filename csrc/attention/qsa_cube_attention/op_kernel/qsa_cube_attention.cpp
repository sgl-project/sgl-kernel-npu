// Complete long-row QSA: one AIC and two AIVs, four bounded GM slots per group.
#include "kernel_operator.h"
#include "lib/matmul_intf.h" // CANN's generated MIX wrapper calls ClearWorkspace.
using namespace AscendC;

namespace {
constexpr uint32_t D = 256, T = 64, M = 16, SLOT_BYTES = 288256;
constexpr uint32_t TOPK = 2051;
constexpr int32_t FP32_EXACT_INTEGER_MAX = 1 << 24;
constexpr uint32_t PAGE_TABLE_UB_STRIDE = 176;
constexpr uint32_t DENSE_M = 128;
constexpr uint16_t DENSE_META_READY = 12, DENSE_QK_READY = 13;
constexpr uint32_t KV_ROWS = 2 * T;
constexpr float NEG = -1.0e20f, LN2 = 0.6931471805599453f;
constexpr float SCALE = 0.0625f * 1.4426950408889634f;
// No high-level Matmul or SyncAll: these cross-core counters are ours.
constexpr uint16_t TILE_READY = 0, QK_READY = 1, P_READY = 2, PV_READY = 3, ROW_DONE = 4;
constexpr uint16_t P_NEXT_READY = 5, PV_NEXT_READY = 6, TILE_NEXT_READY = 7;
constexpr uint16_t V_ODD_READY = 8, V_ODD_NEXT_READY = 9;
constexpr uint16_t V_EVEN_READY = 10, V_EVEN_NEXT_READY = 11;

template<HardEvent E> __aicore__ inline void Fence()
{
    const auto e = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(e);
    WaitFlag<E>(e);
}

// AIC and both AIV lanes must agree on the same request interval.
__aicore__ inline uint64_t RequestEnd(GM_ADDR t2r, uint64_t row, uint64_t end, int32_t &request)
{
    GlobalTensor<int32_t> requests;
    requests.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(t2r));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(requests[(row / 16) * 16]);
    request = requests.GetValue(row);
    uint64_t next = row + 1;
    for (; next < end; ++next) {
        if ((next & 15) == 0)
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(requests[next]);
        if (requests.GetValue(next) != request) break;
    }
    return next;
}

template<uint32_t H, bool DENSE = false> struct Slot {
    static constexpr bool SPARSE_PAIRED = H == 12 && !DENSE;
    static constexpr uint32_t AIV_ROWS = H == 12 ? 1 : 6 / H;
    GlobalTensor<float> score, delta;
    GlobalTensor<bfloat16_t> p, packedK, packedV;
    __aicore__ inline void Init(GM_ADDR ws, uint32_t group, uint32_t parity = 0, uint32_t lane = 0)
    {
        GM_ADDR base = ws + (group * 4 + parity) * SLOT_BYTES;
        // Sparse H12 writes 24 score rows, so its score/P/delta regions must
        // stay disjoint. Shared QK and H3/H6 retain their original offsets.
        constexpr uint32_t scoreLaneBytes = SPARSE_PAIRED ? 12288 : 6144;
        constexpr uint32_t pOffset = SPARSE_PAIRED ? 24576 : 12288;
        constexpr uint32_t deltaOffset = SPARSE_PAIRED ? 27648 : (H == 12 ? 15360 : 13824);
        constexpr uint32_t kOffset = SPARSE_PAIRED ? 52224 : 26112;
        constexpr uint32_t vOffset = SPARSE_PAIRED ? 117760 : 157184;
        score.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + (group * 4 + (parity & 2)) * SLOT_BYTES + lane * scoreLaneBytes));
        p.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base + pOffset + lane * (H == 12 ? 1536 : 768)));
        delta.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + deltaOffset + lane * (H == 12 ? 12288 : 6144)));
        packedK.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base + kOffset + lane * AIV_ROWS * T * D * 2));
        packedV.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(base + vOffset + lane * AIV_ROWS * T * D * 2));
    }
};

template<uint32_t H, bool DENSE = false> class Cube {
    static constexpr uint32_t QUERY_ROWS = H == 12 ? 2 : 12 / H;
    static constexpr uint32_t QK_PASSES = H == 12 && !DENSE ? 2 : 1;
    static constexpr uint32_t QK_COLUMNS = H == 12 && !DENSE ? KV_ROWS : 2 * KV_ROWS;
    static constexpr uint32_t QK_SLICE_ELEMENTS = QK_COLUMNS * (D / 4);
    static constexpr uint32_t DENSE_QUERIES = 120 / H;
    static constexpr uint32_t DENSE_V_BANKS = H == 3 ? 2 : 4;
    static constexpr uint32_t DENSE_V_STRIDE = (QUERY_ROWS / 2) * KV_ROWS * D;
    static constexpr bool DENSE_OUTPUT_PINGPONG = H == 12 && DENSE;
    static constexpr uint32_t DENSE_K_ROWS = DENSE_OUTPUT_PINGPONG ? 64 : 128;
public:
    __aicore__ inline void Run(GM_ADDR q, GM_ADDR k, GM_ADDR table, GM_ADDR t2r, GM_ADDR ws,
                              const SglQsaCubeAttentionTilingData &t, TPipe &pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
        denseK_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        denseTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
        const uint32_t group = GetBlockIdx();
        GlobalTensor<int32_t> pairMask;
        pairMask.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            ws + t.usedCoreNum * 4 * SLOT_BYTES + group * 256));
        slot_.Init(ws, group);
        pipe.InitBuffer(l1Buf_, (H == 12 && !DENSE ? 408 : 400) * 1024);
        pipe.InitBuffer(aBuf_, (dense_ ? 64 : (H == 12 ? 20 : 12)) * 1024);
        pipe.InitBuffer(bBuf_, 64 * 1024);
        pipe.InitBuffer(cBuf_, (dense_ ? (DENSE_OUTPUT_PINGPONG ? 128 : 64) : 16) * 1024);
        if constexpr (DENSE_OUTPUT_PINGPONG) {
            for (uint32_t i = 0; i < 2; ++i) {
                denseOutputFree_[i] = pipe.AllocEventID<HardEvent::FIX_M>();
                denseInputFree_[i] = pipe.AllocEventID<HardEvent::M_MTE1>();
            }
        }
        q1_ = l1Buf_.Get<bfloat16_t>(); p1_ = q1_[QK_PASSES * M * D];
        k1_ = p1_[4096];
        qa_ = aBuf_.Get<bfloat16_t>(); pa_ = qa_[QK_PASSES * M * D];
        b_ = bBuf_.Get<bfloat16_t>(); c_ = cBuf_.Get<float>();
        const uint32_t groupRows = dense_ ? DENSE_QUERIES : QUERY_ROWS;
        for (uint64_t groupBase = uint64_t(group) * groupRows; groupBase < t.rows;
             groupBase += uint64_t(t.usedCoreNum) * groupRows) {
            const uint64_t groupEnd = groupBase + groupRows < t.rows ? groupBase + groupRows : t.rows;
            uint64_t end = dense_ ? groupBase : groupEnd;
            for (uint64_t row = groupBase; row < groupEnd;
                 row += end - row < QUERY_ROWS ? end - row : QUERY_ROWS) {
                if constexpr (dense_) {
                    if (row == end) {
                        int32_t request = 0;
                        end = t.numRequests > 1 ? RequestEnd(t2r, row, groupEnd, request) : groupEnd;
                        const uint32_t req = request >= 0 && request < static_cast<int32_t>(t.numRequests) ? request : 0;
                        denseTableOffset_ = uint64_t(req) * t.pageTableWidth;
                        cachedPage_ = ~uint32_t(0);
                        DenseQK(row, end, ws, group, t);
                    }
                    DensePVRow(pairMask, ws, group, t);
                    continue;
                }
                Zero(q1_, QK_PASSES * M, D); Zero(p1_, M, 2 * KV_ROWS);
                PipeBarrier<PIPE_MTE2>();
                const uint32_t validRows = t.rows - row < QUERY_ROWS ? t.rows - row : QUERY_ROWS;
                // Sparse H12 keeps two independently padded M16 matrices.
                // This preserves the existing L0A layout and K64 recurrence.
                if constexpr (H == 12) {
                    DataCopy(q1_, q_[row * H * D], Nz(H, D, M, D));
                    if (validRows == 2)
                        DataCopy(q1_[M * D], q_[(row + 1) * H * D], Nz(H, D, M, D));
                } else {
                    DataCopy(q1_, q_[row * H * D], Nz(validRows * H, D, M, D));
                }
                Fence<HardEvent::MTE2_MTE1>();
                for (uint32_t pass = 0; pass < QK_PASSES; ++pass)
                    Load(qa_[pass * M * D], q1_[pass * M * D], D / 16, 1, false);
                Fence<HardEvent::MTE1_M>();
                // Two N64 tiles share QK submission; softmax/PV remain N64 and ordered.
                for (uint32_t batch = 0; batch < t.numTiles; batch += 2) {
                    const uint32_t bank = (batch / 2) % 3;
                    auto k0 = k1_[bank * 2 * KV_ROWS * D];
                    auto k1 = k1_[((bank + 1) % 3) * 2 * KV_ROWS * D];
                    auto v0 = k1_[((bank + 2) % 3) * 2 * KV_ROWS * D];
                    if (batch == 0) {
                        CrossCoreWaitFlag<2>(TILE_READY);
                        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(pairMask);
                        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(pairMask[32]);
                        activePairs_ = uint32_t(pairMask.GetValue(0)) | uint32_t(pairMask.GetValue(32));
                        LoadK(k0, k1, ws, group, batch, t.numTiles);
                    }
                    Fence<HardEvent::MTE2_MTE1>();
                    auto vCopy = Nz(KV_ROWS, D, KV_ROWS, D);
                    vCopy.ndNum = QUERY_ROWS / 2; vCopy.srcNdMatrixStride = KV_ROWS * D;
                    vCopy.dstNzMatrixStride = KV_ROWS * D;
                    slot_.Init(ws, group, batch & 3);
                    CrossCoreWaitFlag<2>((batch & 2) ? V_EVEN_NEXT_READY : V_EVEN_READY);
                    if (Active(batch)) DataCopy(v0, slot_.packedV, vCopy);
                    for (uint32_t pair = 0; pair < QUERY_ROWS / 2 && Active(batch); ++pair) {
                        auto kSource = pair == 0 ? k0 : k1;
                        for (uint32_t pass = 0; pass < QK_PASSES; ++pass) {
                        auto queryK = kSource[H == 12 ? pass * KV_ROWS * D : 0];
                        Load(b_, queryK, QK_SLICE_ELEMENTS / 256, 1, false);
                        for (uint32_t part = 0; part < 4; ++part) {
                            Fence<HardEvent::MTE1_M>();
                            Fence<HardEvent::M_MTE1>();
                            if (part + 1 < 4)
                                Load(b_[((part + 1) & 1) * QK_SLICE_ELEMENTS],
                                     queryK[(part + 1) * QK_SLICE_ELEMENTS], QK_SLICE_ELEMENTS / 256, 1, false);
                            MmadParams mm{};
                            mm.m = M; mm.n = QK_COLUMNS; mm.k = D / 4; mm.cmatrixInitVal = part == 0;
                            Mmad(c_, qa_[pass * M * D + part * M * (D / 4)], b_[(part & 1) * QK_SLICE_ELEMENTS], mm);
                        }
                        Fence<HardEvent::M_FIX>();
                        Fence<HardEvent::M_MTE1>();
                        Store(slot_.score[(pass * H + pair * 2 * H) * 2 * KV_ROWS], QK_COLUMNS,
                              c_[pair * 2 * H * 16], H == 12 ? H : 2 * H);
                        Fence<HardEvent::FIX_M>();
                        }
                    }
                    CrossCoreSetFlag<2, PIPE_FIX>(QK_READY);
                    Fence<HardEvent::MTE1_MTE2>();
                    for (uint32_t half = 0; half < 2 && batch + half < t.numTiles; ++half) {
                        slot_.Init(ws, group, (batch + half) & 3);
                        CrossCoreWaitFlag<2>(half == 0 ? P_READY : P_NEXT_READY);
                        auto pCopy = Nz(H, T, M, T);
                        pCopy.ndNum = QUERY_ROWS; pCopy.srcNdMatrixStride = H * T;
                        pCopy.dstNzMatrixStride = H == 12 ? T * M : T * M + H * 16;
                        if (Active(batch)) DataCopy(p1_, slot_.p, pCopy);
                        Fence<HardEvent::MTE2_MTE1>();
                        if (half == 0 && batch + 2 < t.numTiles) {
                            // Give current P's L1 transfer priority over future K.
                            CrossCoreWaitFlag<2>(((batch + 2) & 2) ? TILE_NEXT_READY : TILE_READY);
                            Slot<H, DENSE> next;
                            next.Init(ws, group, (batch + 2) & 3);
                            if (Active(batch + 2)) LoadKTile(k1, next, 0);
                        }
                        if (half == 0 && batch + 1 < t.numTiles) {
                            Slot<H, DENSE> next;
                            next.Init(ws, group, (batch + 1) & 3);
                            // QK has released K's L1 buffer. Hide the next V load
                            // behind this tile's PV instead of waiting until PV ends.
                            CrossCoreWaitFlag<2>((batch & 2) ? V_ODD_NEXT_READY : V_ODD_READY);
                            if (Active(batch)) DataCopy(k0, next.packedV, vCopy);
                        }
                        if (half == 1 && batch + 2 < t.numTiles) {
                            LoadK(k1, v0, ws, group, batch + 2, t.numTiles, true);
                        }
                        if (Active(batch)) {
                        auto pvV = half == 0 ? v0 : k0;
                        ComputePV(pvV);
                        }
                        CrossCoreSetFlag<2, PIPE_FIX>(half == 0 ? PV_READY : PV_NEXT_READY);
                        Fence<HardEvent::FIX_M>();
                        Fence<HardEvent::M_MTE1>();
                        Fence<HardEvent::MTE1_MTE2>();
                    }
                }
                CrossCoreWaitFlag<2>(ROW_DONE);
            }
        }
        if constexpr (DENSE_OUTPUT_PINGPONG) {
            // The final MTE1->MTE2 edge also follows both QK input drains.
            Fence<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < 2; ++i)
                pipe.ReleaseEventID<HardEvent::M_MTE1>(denseInputFree_[i]);
        }
    }
private:
    __aicore__ inline void ComputePV(LocalTensor<bfloat16_t> pvV, uint32_t pOffset = 0)
    {
        LoadPV(pvV, 0, pOffset);
        for (uint32_t part = 0; part < QUERY_ROWS; ++part) {
            Fence<HardEvent::MTE1_M>();
            Fence<HardEvent::M_MTE1>();
            if (part + 1 < QUERY_ROWS) LoadPV(pvV, part + 1, pOffset);
            MmadParams mm{};
            mm.m = M; mm.n = D; mm.k = T; mm.cmatrixInitVal = H == 12 || part == 0;
            if constexpr (DENSE_OUTPUT_PINGPONG) {
                // Shared QK has drained both L0C banks before entering PV.
                auto output = c_[part * DENSE_M * 128];
                Mmad(output, pa_[(part & 1) * T * M], b_[(part & 1) * T * D], mm);
                Fence<HardEvent::M_FIX>();
                Store(slot_.delta[part * H * D], D, output, H);
            } else {
                Mmad(c_, pa_[(part & 1) * T * M], b_[(part & 1) * T * D], mm);
                if constexpr (H == 12) {
                    Fence<HardEvent::M_FIX>();
                    Store(slot_.delta[part * H * D], D, c_, H);
                    Fence<HardEvent::FIX_M>();
                }
            }
        }
        Fence<HardEvent::M_FIX>();
        Fence<HardEvent::M_MTE1>();
        if constexpr (H != 12) Store(slot_.delta, D, c_, QUERY_ROWS * H);
    }
    __aicore__ inline void DenseLoadV(uint32_t tile, GM_ADDR ws, uint32_t group)
    {
        Slot<H, DENSE> source;
        source.Init(ws, group, tile & 3);
        auto copy = Nz(KV_ROWS, D, KV_ROWS, D);
        copy.ndNum = QUERY_ROWS / 2; copy.srcNdMatrixStride = KV_ROWS * D;
        copy.dstNzMatrixStride = KV_ROWS * D;
        if (Active(tile)) DataCopy(k1_[(tile % DENSE_V_BANKS) * DENSE_V_STRIDE], source.packedV, copy);

    }
    __aicore__ inline void DensePVRow(GlobalTensor<int32_t> pairMask, GM_ADDR ws,
                                    uint32_t group, const SglQsaCubeAttentionTilingData &t)
    {
        CrossCoreWaitFlag<2>(V_EVEN_READY);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(pairMask);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(pairMask[32]);
        activePairs_ = uint32_t(pairMask.GetValue(0)) | uint32_t(pairMask.GetValue(32));
        CrossCoreSetFlag<2, PIPE_MTE2>(QK_READY);
        Zero(p1_, M, 2 * KV_ROWS);
        PipeBarrier<PIPE_MTE2>();
        DenseLoadV(0, ws, group);
        if constexpr (H == 12) {
            slot_.Init(ws, group, 0);
            CrossCoreWaitFlag<2>(P_READY);
            auto copy = Nz(H, T, M, T);
            copy.ndNum = QUERY_ROWS; copy.srcNdMatrixStride = H * T;
            copy.dstNzMatrixStride = T * M;
            if (Active(0)) DataCopy(p1_, slot_.p, copy);
        }
        for (uint32_t tile = 0; tile < t.numTiles; ++tile) {
            slot_.Init(ws, group, tile & 3);
            if constexpr (H != 12) {
                CrossCoreWaitFlag<2>((tile & 1) ? P_NEXT_READY : P_READY);
                // P-ready follows current V and P writes in the same MTE3 pipe.
                if (tile != 0) DenseLoadV(tile, ws, group);
                auto copy = Nz(H, T, M, T);
                copy.ndNum = QUERY_ROWS; copy.srcNdMatrixStride = H * T;
                copy.dstNzMatrixStride = T * M + H * 16;
                if (Active(tile)) DataCopy(p1_, slot_.p, copy);
            }
            Fence<HardEvent::MTE2_MTE1>();
            // Current V bank stays live through the unchanged PV reader.
            if (Active(tile)) {
                if constexpr (H == 12)
                    ComputePV(k1_[(tile % DENSE_V_BANKS) * DENSE_V_STRIDE], (tile & 1) * QUERY_ROWS * T * M);
                else
                    ComputePV(k1_[(tile % DENSE_V_BANKS) * DENSE_V_STRIDE]);
            }
            CrossCoreSetFlag<2, PIPE_FIX>((tile & 1) ? PV_NEXT_READY : PV_READY);
            if constexpr (H == 12) {
                if (tile + 1 < t.numTiles) {
                    // Use the other P bank: events targeting M or FIX do not
                    // make Scalar wait for current MTE1 readers. The existing
                    // loop-end MTE1->MTE2 wait protects reuse two tiles later.
                    const uint32_t next = tile + 1;
                    CrossCoreWaitFlag<2>((next & 1) ? P_NEXT_READY : P_READY);
                    DenseLoadV(next, ws, group);
                    Slot<H, DENSE> input;
                    input.Init(ws, group, next & 3);
                    auto copy = Nz(H, T, M, T);
                    copy.ndNum = QUERY_ROWS; copy.srcNdMatrixStride = H * T;
                    copy.dstNzMatrixStride = T * M;
                    if (Active(next)) DataCopy(p1_[(next & 1) * QUERY_ROWS * T * M], input.p, copy);
                }
            }
            Fence<HardEvent::FIX_M>();
            Fence<HardEvent::M_MTE1>();
            Fence<HardEvent::MTE1_MTE2>();
        }
        CrossCoreWaitFlag<2>(ROW_DONE);
    }
    __aicore__ inline void DenseQK(uint64_t row, uint64_t end, GM_ADDR ws, uint32_t group,
                                   const SglQsaCubeAttentionTilingData &t)
    {
        GlobalTensor<int32_t> meta;
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            ws + uint64_t(t.usedCoreNum) * (4 * SLOT_BYTES + 256) + group * 256));
        CrossCoreWaitFlag<2>(DENSE_META_READY);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(meta);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(meta[32]);
        const int32_t a = meta.GetValue(0), b = meta.GetValue(32);
        const int32_t high = a > b ? a : b;
        const uint32_t span = high < 0 ? 128 : ((uint32_t(high) + 128) / 128) * 128;
        const uint32_t capacity = t.pageTableWidth * t.pageTokens;
        GlobalTensor<float> score;
        score.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
            ws + uint64_t(t.usedCoreNum) * (4 * SLOT_BYTES + 512) +
            uint64_t(group) * DENSE_M * capacity * sizeof(float)));
        const uint32_t queries = end - row;
        Zero(q1_, DENSE_M, D);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(q1_, q_[row * H * D], Nz(queries * H, D, DENSE_M, D));
        Fence<HardEvent::MTE2_MTE1>();
        // Q^T is B: NZ order directly matches its four K64 L0B slices.
        Load(b_, q1_, DENSE_M * D / 256, 1, false);
        Fence<HardEvent::MTE1_M>();
        if (high >= 0) {
            auto first = q1_[DENSE_M * D];
            auto second = first[128 * D];
            if constexpr (DENSE_OUTPUT_PINGPONG) {
                for (uint32_t i = 0; i < 2; ++i) {
                    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(denseOutputFree_[i]));
                    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(denseInputFree_[i]));
                }
            }
            DenseLoadK(first, 0, t);
            for (uint32_t col = 0; col < span; col += 128) {
                #pragma unroll
                for (uint32_t pass = 0; pass < 128 / DENSE_K_ROWS; ++pass) {
                    const uint32_t column = col + pass * DENSE_K_ROWS;
                    const uint32_t outputBank = DENSE_OUTPUT_PINGPONG ? pass : (col / 128) & 1;
                    auto current = outputBank == 0 ? first : second;
                    auto next = outputBank == 0 ? second : first;
                    Fence<HardEvent::MTE2_MTE1>();
                    // Two 32KiB L0A banks overlap K64-token loading with MMAD.
                    // Each token still uses a complete K256 dot product.
                    auto input = qa_[DENSE_OUTPUT_PINGPONG ? outputBank * DENSE_K_ROWS * D : 0];
                    if constexpr (DENSE_OUTPUT_PINGPONG)
                        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(denseInputFree_[outputBank]));
                    Load(input, current, DENSE_K_ROWS * D / 256, 1, false);
                    if (column + DENSE_K_ROWS < span) DenseLoadK(next, column + DENSE_K_ROWS, t);
                    Fence<HardEvent::MTE1_M>();
                    if constexpr (!DENSE_OUTPUT_PINGPONG) Fence<HardEvent::M_MTE1>();
                    auto output = c_;
                    if constexpr (DENSE_OUTPUT_PINGPONG) {
                        WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(denseOutputFree_[outputBank]));
                        output = c_[outputBank * DENSE_M * 128];
                    }
                    MmadParams mm{};
                    mm.m = DENSE_K_ROWS; mm.n = 128; mm.k = D; mm.cmatrixInitVal = true;
                    Mmad(output, input, b_, mm);
                    if constexpr (DENSE_OUTPUT_PINGPONG)
                        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(denseInputFree_[outputBank]));
                    Fence<HardEvent::M_FIX>();
                    if constexpr (!DENSE_OUTPUT_PINGPONG) Fence<HardEvent::M_MTE1>();
                    FixpipeParamsV220 copy{};
                    copy.nSize = DENSE_M; copy.mSize = DENSE_K_ROWS; copy.srcStride = DENSE_K_ROWS;
                    // Keep NZ in GM: four tokens of one head block are contiguous.
                    copy.dstStride = DENSE_K_ROWS * 16 * sizeof(float) / 32;
                    copy.quantPre = QuantMode_t::NoQuant; copy.ndNum = 1;
                    Fixpipe<float, float, CFG_NZ>(score[uint64_t(column) * DENSE_M], output, copy);
                    if constexpr (DENSE_OUTPUT_PINGPONG)
                        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(denseOutputFree_[outputBank]));
                    else
                        Fence<HardEvent::FIX_M>();
                    Fence<HardEvent::MTE1_MTE2>();
                }
            }
            if constexpr (DENSE_OUTPUT_PINGPONG)
                for (uint32_t i = 0; i < 2; ++i)
                    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(denseInputFree_[i]));
            // Both score exports must finish before publishing QK and before
            // the following PV phase reuses the first L0C bank.
            if constexpr (DENSE_OUTPUT_PINGPONG)
                for (uint32_t i = 0; i < 2; ++i)
                    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(denseOutputFree_[i]));
        }
        CrossCoreSetFlag<2, PIPE_FIX>(DENSE_QK_READY);
        // Dense phase has released L1/L0 before the original sparse PV phase.
        Fence<HardEvent::MTE1_MTE2>();
    }
    __aicore__ inline void DenseLoadK(LocalTensor<bfloat16_t> dst, uint32_t col,
                                      const SglQsaCubeAttentionTilingData &t)
    {
        const uint32_t page = col / t.pageTokens;
        if (page != cachedPage_) {
            const uint64_t index = denseTableOffset_ + page;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(denseTable_[(index / 16) * 16]);
            cachedPhysical_ = denseTable_.GetValue(index);
            cachedPage_ = page;
        }
        if (cachedPhysical_ >= 0 && cachedPhysical_ < static_cast<int32_t>(t.numCacheBlocks)) {
            auto copy = Nz(16, D, 16, D);
            copy.ndNum = DENSE_K_ROWS / 16;
            copy.srcNdMatrixStride = 16 * D;
            copy.dstNzMatrixStride = 16 * D;
            DataCopy(dst,
                     denseK_[(uint64_t(cachedPhysical_) * t.pageTokens + col % t.pageTokens) * D], copy);
        } else {
            Zero(dst, DENSE_K_ROWS, D);
        }
    }
    __aicore__ inline void LoadPV(LocalTensor<bfloat16_t> source, uint32_t part, uint32_t pOffset = 0)
    {
        const uint32_t buffer = part & 1, pair = part / 2;
        Load(pa_[buffer * T * M], p1_[pOffset + part * T * M], T / 16, 1, false);
        for (uint32_t kb = 0; kb < T / 16; ++kb)
            Load(b_[buffer * T * D + kb * (D / 16) * 256],
                 source[pair * KV_ROWS * D + (buffer * (T / 16) + kb) * 256],
                 D / 16, KV_ROWS / 16, true);
    }
    __aicore__ inline void LoadKTile(LocalTensor<bfloat16_t> dst, Slot<H, DENSE> &source, uint32_t half)
    {
        if constexpr (H == 12) {
            // Each query's two N64 tiles form one N128 matrix. No QK
            // products with the other query's sparse K are needed.
            const auto copy = Nz(T, D, KV_ROWS, D);
            for (uint32_t query = 0; query < 2; ++query)
                DataCopy(dst[query * KV_ROWS * D + half * T * 16],
                         source.packedK[query * T * D], copy);
        } else {
            DataCopy(dst[half * KV_ROWS * 16], source.packedK, Nz(KV_ROWS, D, 2 * KV_ROWS, D));
        }
    }
    __aicore__ inline void LoadK(LocalTensor<bfloat16_t> first, LocalTensor<bfloat16_t> second,
                                  GM_ADDR ws, uint32_t group, uint32_t batch, uint32_t tiles, bool firstLoaded = false)
    {
        if (dense_ || !Active(batch)) return;
        auto copy = Nz(KV_ROWS, D, 2 * KV_ROWS, D);
        for (uint32_t half = 0; half < 2; ++half) {
            Slot<H, DENSE> source;
            source.Init(ws, group, (batch + (batch + half < tiles ? half : 0)) & 3);
            if (!firstLoaded || half != 0)
                LoadKTile(first, source, half);
            if constexpr (QUERY_ROWS == 4)
                DataCopy(second[half * KV_ROWS * 16], source.packedK[KV_ROWS * D], copy);
        }
    }
    __aicore__ inline Nd2NzParams Nz(uint16_t n, uint32_t d, uint16_t pad, uint32_t stride)
    {
        Nd2NzParams p{};
        p.ndNum = 1; p.nValue = n; p.dValue = d; p.srcDValue = stride;
        p.dstNzC0Stride = pad; p.dstNzNStride = 1;
        return p;
    }
    __aicore__ inline void Zero(LocalTensor<bfloat16_t> x, uint16_t n, uint16_t d)
    {
        InitConstValueParams<bfloat16_t> p{};
        p.repeatTimes = n; p.blockNum = d / 16; p.initValue = static_cast<bfloat16_t>(0);
        InitConstValue(x, p);
    }
    __aicore__ inline void Load(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                               uint8_t count, uint16_t stride, bool transpose)
    {
        LoadData2DParams p{};
        p.repeatTimes = count; p.srcStride = stride; p.ifTranspose = transpose;
        LoadData(dst, src, p);
    }
    __aicore__ inline void Store(GlobalTensor<float> dst, uint16_t n, LocalTensor<float> src, uint16_t rows = H)
    {
        FixpipeParamsV220 p{};
        p.nSize = n; p.mSize = rows; p.srcStride = M; p.dstStride = n;
        p.quantPre = QuantMode_t::NoQuant; p.ndNum = 1;
        Fixpipe<float, float, CFG_ROW_MAJOR>(dst, src, p);
    }
    __aicore__ inline bool Active(uint32_t tile) const { return (activePairs_ & (1U << (tile / 2))) != 0; }
    uint32_t activePairs_ = 0;
    Slot<H, DENSE> slot_;
    static constexpr bool dense_ = DENSE;
    uint32_t cachedPage_ = ~uint32_t(0);
    uint64_t denseTableOffset_ = 0;
    int32_t denseOutputFree_[2];
    int32_t cachedPhysical_ = -1;
    GlobalTensor<int32_t> denseTable_;
    GlobalTensor<bfloat16_t> q_, denseK_;
    TBuf<TPosition::A1> l1Buf_;
    TBuf<TPosition::A2> aBuf_;
    TBuf<TPosition::B2> bBuf_;
    TBuf<TPosition::CO1> cBuf_;
    LocalTensor<bfloat16_t> q1_, p1_, k1_, qa_, pa_, b_;
    LocalTensor<float> c_;
    int32_t denseInputFree_[2];
};

template<uint32_t H, bool DENSE = false> class Vector {
    static constexpr uint32_t AIV_ROWS = H == 12 ? 1 : 6 / H;
    static constexpr uint32_t QUERY_ROWS = 2 * AIV_ROWS;
    static constexpr uint32_t DENSE_QUERIES = 120 / H;
    static constexpr bool PREFETCH_V = H == 12 && DENSE;
    static constexpr uint32_t DENSE_K_ROWS = H == 12 ? 64 : 128;
    static constexpr uint32_t PAGE_STRIDE = H == 3 ? 512 : PAGE_TABLE_UB_STRIDE;
    static constexpr uint32_t EXTRA_PAGE_INTS = 2 * (PAGE_STRIDE - PAGE_TABLE_UB_STRIDE);
    static constexpr uint32_t CALC_BYTES = 22368 * (H == 12 ? 2 : 1);
public:
    __aicore__ inline void Run(GM_ADDR k, GM_ADDR v, GM_ADDR idx, GM_ADDR table, GM_ADDR t2r, GM_ADDR out,
                              GM_ADDR ws, const SglQsaCubeAttentionTilingData &t, TPipe &pipe)
    {
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idx));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
        t2r_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(t2r));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        const uint32_t sub = GetSubBlockIdx(), group = GetBlockIdx() / GetSubBlockNum();
        GlobalTensor<int32_t> pairMask;
        pairMask.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            ws + t.usedCoreNum * 4 * SLOT_BYTES + group * 256));
        constexpr uint32_t heads = AIV_ROWS * H;
        // Full-cache element offsets fit signed int32 only below this bound.
        pageTokens_ = t.pageTokens;
        linearAddresses_ = static_cast<uint64_t>(t.numCacheBlocks) * pageTokens_ * D <= (uint64_t(1) << 31);
        slot_.Init(ws, group);
        // Each AIV owns six original states or twelve heads of one paired query.
        // Dense H12 keeps 16 KiB scratch, two 32 KiB V banks and 16 KiB
        // constant compact-expansion patterns (177696 B total UB).
        // Original H3/H6 UB remains 188096 B, below the 184 KiB boundary.
        constexpr uint32_t states = H == 12 ? 2 : 1;
        pipe.InitBuffer(metaBuf_, 20128 + EXTRA_PAGE_INTS * sizeof(int32_t));
        pipe.InitBuffer(calcBuf_, CALC_BYTES + (PREFETCH_V ? 16384 : 0));
        pipe.InitBuffer(kvBuf_, H == 12 ? (DENSE ? 80 : 64) * 1024 : 4 * T * D * 2);
        pipe.InitBuffer(groupBuf_, 6336);
        pipe.InitBuffer(addressBuf_, 8192);
        physicalGroups_ = addressBuf_.Get<int32_t>();
        groupPageOffsets_ = physicalGroups_[1024];
        groupOffsets_ = groupBuf_.Get<uint32_t>();
        groupFirst_ = groupBuf_.Get<int32_t>()[512];
        groupOther_ = groupBuf_.Get<int32_t>()[1024];
        groupMasks_ = groupBuf_.Get<uint8_t>()[6144];
        groupTempMask_ = groupBuf_.Get<uint8_t>()[6272];
        CreateVecIndex(groupBuf_.Get<int32_t>(), int32_t(0), 512);
        PipeBarrier<PIPE_V>();
        ShiftLeft(groupOffsets_, groupOffsets_, uint32_t(4), 512);
        if constexpr (PREFETCH_V) {
            copyDone_[0] = pipe.AllocEventID<HardEvent::MTE3_MTE2>();
            copyDone_[1] = pipe.AllocEventID<HardEvent::MTE3_S>();
            pCopyDone_ = pipe.AllocEventID<HardEvent::MTE3_V>();
        } else {
            for (uint32_t i = 0; i < 2; ++i) copyDone_[i] = pipe.AllocEventID<HardEvent::MTE3_S>();
        }
        kHalfDone_ = pipe.AllocEventID<HardEvent::MTE3_S>();
        packedK_ = kvBuf_.Get<bfloat16_t>();
        packedV_ = packedK_[H == 12 ? (DENSE ? 8192 : T * D) : 2 * T * D];
        auto meta = metaBuf_.Get<int32_t>();
        page_ = meta; ids_ = meta[352 + EXTRA_PAGE_INTS]; validBase_ = meta[4512 + EXTRA_PAGE_INTS]; req_ = meta[5024 + EXTRA_PAGE_INTS];
        if (dense_) {
            // Shared QK skips K packing. Its first 8 KiB holds metadata/score
            // scratch; keep offsets and four alpha slots beyond that region.
            // H3 needs both query-index buffers, so neither can host this state.
            denseTokenOffsets_ = packedK_.ReinterpretCast<uint32_t>()[2048];
            denseAlpha_ = packedK_.ReinterpretCast<float>()[2112];
            denseScoreOffsets_ = packedK_.ReinterpretCast<uint32_t>()[2560];
            for (uint32_t i = 0; i < T; ++i)
                denseTokenOffsets_.SetValue(i, ((i / 4) * 128 + (i % 4) * 16) * sizeof(float));
        }
        auto calc = calcBuf_.Get<float>();
        acc_ = calc; delta_ = calc[1536 * states]; score_ = calc[3072 * states]; prob_ = calc[3456 * states];
        mask_ = calc[3840 * states]; reduce_ = calc[3968 * states]; temp_ = calc[4016 * states];
        maximum_ = calc[5400 * states]; nextMax_ = calc[5448 * states]; normalizer_ = calc[5496 * states]; alpha_ = calc[5544 * states];
        probBf_ = calcBuf_.Get<bfloat16_t>()[8832 * states];
        outBf_ = calcBuf_.Get<bfloat16_t>()[9216 * states];
        maskBits_ = calcBuf_.Get<uint8_t>()[21504 * states];
        if constexpr (PREFETCH_V) {
            if (t.topk == 514) {
                auto offsets = calcBuf_.Get<int32_t>()[CALC_BYTES / sizeof(int32_t)];
                auto lanes = offsets[2048];
                CreateVecIndex(offsets, int32_t(0), 2048);
                PipeBarrier<PIPE_V>();
                CreateVecIndex(lanes, int32_t(0), 2048);
                PipeBarrier<PIPE_V>();
                ShiftRight(offsets, offsets, int32_t(2), 2048);
                PipeBarrier<PIPE_V>();
                ShiftLeft(offsets, offsets, int32_t(2), 2048);
                PipeBarrier<PIPE_V>();
                Sub(lanes, lanes, offsets, 2048);
                PipeBarrier<PIPE_V>();
            }
        }
        // Both AIVs participate even when the last group's second query is absent.
        const uint32_t groupRows = dense_ ? DENSE_QUERIES : QUERY_ROWS;
        for (uint64_t groupBase = uint64_t(group) * groupRows; groupBase < t.rows;
             groupBase += uint64_t(t.usedCoreNum) * groupRows) {
            const uint64_t groupEnd = groupBase + groupRows < t.rows ? groupBase + groupRows : t.rows;
            uint64_t end = dense_ ? groupBase : groupEnd;
            for (uint64_t baseRow = groupBase; baseRow < groupEnd;
                 baseRow += end - baseRow < QUERY_ROWS ? end - baseRow : QUERY_ROWS) {
                if constexpr (dense_) {
                    if (baseRow == end) {
                        int32_t request = 0;
                        end = t.numRequests > 1 ? RequestEnd(t2r, baseRow, groupEnd, request) : groupEnd;
                        DenseMetadata(baseRow, end, request, ws, group, sub, t);
                    }
                }
                const uint64_t row = baseRow + AIV_ROWS * sub;
                Duplicate(maximum_, NEG, heads * 8);
                Duplicate(normalizer_, 0.0f, heads * 8);

                Duplicate(outBf_.ReinterpretCast<float>(), NEG, 8);
                int32_t request[2] = {-1, -1};
                for (uint32_t r = 0; r < AIV_ROWS; ++r) {
                    fullGroups_[r] = false;
                    if (row + r < end) {
                        CopyInts(req_, t2r_[row + r], 1);
                        if (t.topk == 514) {
                            CopyInts(groupFirst_, indices_[(row + r) * 514], 512);
                            CopyInts(ids_[r * 2080 + 2064], indices_[(row + r) * 514 + 512], 2);
                        } else {
                            CopyInts(ids_[r * 2080], indices_[(row + r) * TOPK], TOPK);
                        }
                        Fence<HardEvent::MTE2_S>();
                        request[r] = req_.GetValue(0);
                        if (request[r] >= 0 && request[r] < static_cast<int32_t>(t.numRequests)) {
                            CopyInts(page_[r * PAGE_STRIDE], table_[static_cast<uint64_t>(request[r]) * t.pageTableWidth], t.pageTableWidth);
                            Fence<HardEvent::MTE2_S>();
                        }
                        Fence<HardEvent::MTE2_V>();
                        if (t.topk == 514) ExpandGroups(r);
                        if (request[r] >= 0 && request[r] < static_cast<int32_t>(t.numRequests))
                            BuildGroupMask(r, t);
                    }
                }
                uint32_t ownPairs = 0;
                for (uint32_t r = 0; r < AIV_ROWS; ++r) {
                    if (request[r] < 0 || request[r] >= static_cast<int32_t>(t.numRequests)) continue;
                    uint64_t allGroups = ~uint64_t(0);
                    for (uint32_t j = 0; j < 8; ++j)
                        allGroups &= groupMasks_.ReinterpretCast<uint64_t>().GetValue(r * 8 + j);
                    if (allGroups == ~uint64_t(0)) {
                        ownPairs |= 0xffffU;
                    } else {
                    // Prologue scratch only: accumulator/delta are initialized afterwards.
                    Cast(acc_, ids_[r * 2080], RoundMode::CAST_NONE, 2048);
                    PipeBarrier<PIPE_V>();
                    WholeReduceMax(temp_, acc_, T, 32, 8, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                    Fence<HardEvent::V_S>();
                    for (uint32_t pair = 0; pair < 16; ++pair)
                        if (temp_.GetValue(pair * 16) >= 0 || temp_.GetValue(pair * 16 + 8) >= 0)
                            ownPairs |= 1U << pair;
                    }
                    for (uint32_t i = 2048; i < TOPK; ++i)
                        if (ids_.GetValue(r * 2080 + i) >= 0) ownPairs |= 1U << 16;
                }
                req_.SetValue(0, static_cast<int32_t>(ownPairs));
                Fence<HardEvent::S_MTE3>();
                DataCopy(pairMask[sub * 32], req_, 8);
                // The first pair's normal TILE/QK handshake also publishes masks.
                // Before QK_READY, both first pairs are conservatively prepared.
                activePairs_ = 0x1ffffU;
                Fence<HardEvent::V_S>();
                Duplicate(acc_, 0.0f, heads * D);
                int32_t prefetchedTile = -1;
                for (uint32_t tile = 0; tile < t.numTiles; ++tile) {
                    slot_.Init(ws, group, tile & 3, sub);
                    if constexpr (dense_) {
                        if constexpr (PREFETCH_V) {
                            if (prefetchCopyPending_) {
                                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(copyDone_[0]));
                                prefetchCopyPending_ = false;
                            }
                        }
                        // Retire old P/delta before reusing the four-slot ring.
                        if constexpr (!PREFETCH_V)
                            if (tile >= 4) ConsumeDenseDelta(tile - 4, ws, group, sub);
                        slot_.Init(ws, group, tile & 3, sub);
                        if (!PREFETCH_V || prefetchedTile != static_cast<int32_t>(tile)) {
                            if constexpr (PREFETCH_V) {
                                // No V was prefetched for this tile. The last
                                // P copy follows every earlier V source reader,
                                // including when fallback zero-fills bank zero.
                                if (pCopyPending_) {
                                    WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(pCopyDone_));
                                    pCopyPending_ = false;
                                }
                            }
                            PrepareTile(tile, request, t);
                        }
                        if (tile == 0) {
                            CrossCoreWaitFlag<2>(QK_READY);
                            DataCopy(req_, pairMask[(1 - sub) * 32], 8);
                            Fence<HardEvent::MTE2_S>();
                            activePairs_ = ownPairs | uint32_t(req_.GetValue(0));
                        }
                    } else {
                    if ((tile & 1) == 0) {
                        if (tile == 0) {
                            PrepareTile(tile, request, t);
                            slot_.Init(ws, group, 1, sub);
                            PrepareTile(tile + 1, request, t);
                            Fence<HardEvent::V_MTE3>();
                            CrossCoreSetFlag<2, PIPE_MTE3>(TILE_READY);
                        }
                        // Preserve the baseline's gather/QK overlap with two tile pairs.
                        for (uint32_t next = tile + 2; next < tile + 4 && next < t.numTiles; ++next) {
                            slot_.Init(ws, group, next & 3, sub);
                            PrepareTile(next, request, t, true);
                        }
                        if (tile + 2 < t.numTiles) {
                            Fence<HardEvent::V_MTE3>();
                            CrossCoreSetFlag<2, PIPE_MTE3>(((tile + 2) & 2) ? TILE_NEXT_READY : TILE_READY);
                        }
                        CrossCoreWaitFlag<2>(QK_READY);
                        if (tile == 0) {
                            DataCopy(req_, pairMask[(1 - sub) * 32], 8);
                            Fence<HardEvent::MTE2_S>();
                            activePairs_ = ownPairs | uint32_t(req_.GetValue(0));
                        }
                        slot_.Init(ws, group, tile & 3, sub);
                    }
                    }
                    bool nextV = false;
                    // Successful shared prefetch filled this tile's valid
                    // bank with ones after checking every group and request.
                    const bool fullMask = PREFETCH_V && prefetchedTile == static_cast<int32_t>(tile);
                    if (Active(tile)) {
                    valid_ = validBase_[(tile & 3) * 2 * T];
                    Fence<HardEvent::S_V>();
                    if (!fullMask) {
                        Cast(mask_, valid_, RoundMode::CAST_NONE, 2 * T);
                        PipeBarrier<PIPE_V>();
                        for (uint32_t r = 0; r < AIV_ROWS; ++r)
                            CompareScalar(maskBits_[r * 32], mask_[r * T], 0.0f, CMPMODE::GT, T);
                        PipeBarrier<PIPE_V>();
                    }
                    if (dense_) {
                        for (uint32_t r = 0; r < AIV_ROWS; ++r)
                            DenseScores(row + r, r, tile, request[r], t);
                    } else {
                    // Extract this N64 tile from the paired-tile 6x256 score block.
                    constexpr uint32_t scoreColumns = H == 12 ? KV_ROWS : 2 * KV_ROWS;
                    const DataCopyParams scoreCopy{H, T * sizeof(float) / 32, (scoreColumns - T) * sizeof(float) / 32, 0};
                    for (uint32_t r = 0; r < AIV_ROWS; ++r)
                        DataCopy(score_[r * H * T], slot_.score[r * H * 2 * KV_ROWS +
                            (H == 12 ? (tile & 1) * T :
                             ((sub * AIV_ROWS + r) % 2) * T + (tile & 1) * KV_ROWS)], scoreCopy);
                    }
                    Fence<HardEvent::MTE2_V>();
                    if constexpr (PREFETCH_V) {
                        // Submit future V only after the current score DMA is
                        // consumed. It uses a separate UB bank, not the P path.
                        nextV = tile + 1 < t.numTiles && CanPrefetchV(tile + 1, request[0], t);
                        if (nextV) BeginPrefetchV(tile + 1);
                    }
                    Muls(score_, score_, SCALE, heads * T);
                    PipeBarrier<PIPE_V>();
                    if (!fullMask) {
                        for (uint32_t r = 0; r < AIV_ROWS; ++r)
                            Select(score_[r * H * T], maskBits_[r * 32], score_[r * H * T],
                                   outBf_.ReinterpretCast<float>(), SELMODE::VSEL_CMPMASK_SPR, uint64_t(T), H,
                                   {1, 1, 0, 8, 8, 0});
                        PipeBarrier<PIPE_V>();
                    }
                    WholeReduceMax(reduce_, score_, T, heads, 8, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                    PipeBarrier<PIPE_V>();
                    Brcb(temp_, reduce_, heads, {1, 8});
                    PipeBarrier<PIPE_V>();
                    Max(nextMax_, maximum_, temp_, uint64_t(8), heads, {1, 1, 1, 1, 1, 8});
                    PipeBarrier<PIPE_V>();
                    Sub(alpha_, maximum_, nextMax_, heads * 8);
                    Adds(maximum_, nextMax_, 0.0f, heads * 8);
                    PipeBarrier<PIPE_V>();
                    Muls(alpha_, alpha_, LN2, heads * 8);
                    Sub(prob_, score_, maximum_, uint64_t(T), heads, {1, 1, 0, 8, 8, 1});
                    PipeBarrier<PIPE_V>();
                    Exp(alpha_, alpha_, heads * 8);
                    Muls(prob_, prob_, LN2, heads * T);
                    PipeBarrier<PIPE_V>();
                    Exp(prob_, prob_, heads * T);
                    PipeBarrier<PIPE_V>();
                    if (fullMask) {
                        Muls(prob_, prob_, 1.0f, heads * T);
                    } else {
                        for (uint32_t h = 0; h < heads; ++h)
                            Mul(prob_[h * T], prob_[h * T], mask_[(h / H) * T], T);
                    }
                    PipeBarrier<PIPE_V>();
                    if constexpr (dense_) {
                        Adds(denseAlpha_[(tile & 3) * heads * 8], alpha_, 0.0f, heads * 8);
                        PipeBarrier<PIPE_V>();
                    }
                    // Protect the previous P DMA only at BF16 UB reuse.
                    if constexpr (PREFETCH_V) {
                        // Only the preceding P copy reads probBf_. A later V
                        // export uses a separate UB bank and need not gate Cast.
                        if (pCopyPending_)
                            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(pCopyDone_));
                    } else {
                        Fence<HardEvent::MTE3_V>();
                    }
                    Cast(probBf_, prob_, RoundMode::CAST_RINT, heads * T);
                    PipeBarrier<PIPE_V>();
                    // Publish converted P before independent normalizer/acc work.
                    Fence<HardEvent::V_MTE3>();
                    DataCopy(slot_.p, probBf_, heads * T);
                    if constexpr (PREFETCH_V) {
                        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(pCopyDone_));
                        pCopyPending_ = true;
                        // P(t) follows V(t). Before producing V(t+2), its
                        // parity bank's V(t) source reader has therefore retired.
                        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(copyDone_[0]));
                        prefetchCopyPending_ = true;
                    }
                    CrossCoreSetFlag<2, PIPE_MTE3>((tile & 1) == 0 ? P_READY : P_NEXT_READY);
                    // FP32 sum is needed only after P publication.
                    WholeReduceSum(reduce_, prob_, T, heads, 8, 1, 8);
                    PipeBarrier<PIPE_V>();
                    Brcb(temp_, reduce_, heads, {1, 8});
                    Mul(normalizer_, normalizer_, alpha_, heads * 8);
                    PipeBarrier<PIPE_V>();
                    Add(normalizer_, normalizer_, temp_, uint64_t(8), heads, {1, 1, 1, 1, 1, 8});
                    // Preserve accumulator order before consuming the PV delta.
                    if (!dense_ && (tile & 1) == 0)
                        for (uint32_t h = 0; h < heads; ++h)
                            Mul(acc_[h * D], acc_[h * D], alpha_[h * 8], uint64_t(T), D / T,
                                {1, 1, 0, 8, 8, 0});
                    } else {
                        CrossCoreSetFlag<2, PIPE_MTE3>((tile & 1) == 0 ? P_READY : P_NEXT_READY);
                    }
                    if constexpr (dense_) {
                        if constexpr (PREFETCH_V) {
                            // Release the old V reader before reusing its GM
                            // slot, then overlap V export with delta recovery.
                            if (tile >= 3)
                                CrossCoreWaitFlag<2>(((tile - 3) & 1) ? PV_NEXT_READY : PV_READY);
                            if (nextV) {
                                PublishPrefetchV(tile + 1, ws, group, sub);
                                prefetchedTile = static_cast<int32_t>(tile + 1);
                            }
                            if (tile >= 3) AccumulateDenseDelta(tile - 3, ws, group, sub);
                        }
                        Fence<HardEvent::V_MTE2>();
                        continue;
                    }
                    // Both future K tiles stay early. Submit their V only after
                    // the current pair of P tiles; readiness is tracked separately.
                    if ((tile & 1) != 0 && deferredVMask_ != 0) {
                        for (uint32_t half = 0; half < 2; ++half) {
                            if ((deferredVMask_ & (1U << half)) == 0) continue;
                            const uint32_t next = tile + 1 + half;
                            slot_.Init(ws, group, next & 3, sub);
                            PrepareDeferredV(next);
                        }
                        slot_.Init(ws, group, tile & 3, sub);
                    }
                    if ((tile & 1) == 0 && tile + 1 < t.numTiles) {
                        Fence<HardEvent::V_MTE2>();
                        continue;
                    }
                    if (!Active(tile)) {
                        CrossCoreWaitFlag<2>(PV_READY);
                        if ((tile & 1) != 0) CrossCoreWaitFlag<2>(PV_NEXT_READY);
                        continue;
                    }
                    CrossCoreWaitFlag<2>(PV_READY);
                    if ((tile & 1) != 0) {
                        slot_.Init(ws, group, (tile - 1) & 3, sub);
                        DataCopy(delta_, slot_.delta, heads * D);
                        Fence<HardEvent::MTE2_V>();
                        Add(acc_, acc_, delta_, heads * D);
                        PipeBarrier<PIPE_V>();
                        for (uint32_t h = 0; h < heads; ++h)
                            Mul(acc_[h * D], acc_[h * D], alpha_[h * 8], uint64_t(T), D / T,
                                {1, 1, 0, 8, 8, 0});
                        Fence<HardEvent::V_MTE2>();
                        slot_.Init(ws, group, tile & 3, sub);
                        CrossCoreWaitFlag<2>(PV_NEXT_READY);
                    }
                    DataCopy(delta_, slot_.delta, heads * D);
                    Fence<HardEvent::MTE2_V>();
                    Add(acc_, acc_, delta_, heads * D);
                    PipeBarrier<PIPE_V>();
                    // Close vector readers before reusing local tile buffers.
                    Fence<HardEvent::V_MTE2>();
                }
                if constexpr (dense_) {
                    constexpr uint32_t remaining = PREFETCH_V ? 3 : 4;
                    const uint32_t first = t.numTiles > remaining ? t.numTiles - remaining : 0;
                    for (uint32_t tile = first; tile < t.numTiles; ++tile)
                        ConsumeDenseDelta(tile, ws, group, sub);
                }
                Fence<HardEvent::V_S>();
                for (uint32_t h = 0; h < heads; ++h) {
                    const float normalizer = normalizer_.GetValue(h * 8);
                    const float inv = normalizer > 0 ? 1.0f / normalizer : 0.0f;
                    Muls(acc_[h * D], acc_[h * D], inv, D);
                }
                PipeBarrier<PIPE_V>();
                Cast(outBf_, acc_, RoundMode::CAST_RINT, heads * D);
                Fence<HardEvent::V_MTE3>();
                for (uint32_t r = 0; r < AIV_ROWS; ++r)
                    if (row + r < end) DataCopy(out_[(row + r) * H * D], outBf_[r * H * D], H * D);
                CrossCoreSetFlag<2, PIPE_MTE3>(ROW_DONE);
                Fence<HardEvent::MTE3_V>();
                Fence<HardEvent::MTE3_MTE2>();
            }
        }
        if constexpr (PREFETCH_V) {
            if (prefetchCopyPending_)
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(copyDone_[0]));
            if (pCopyPending_)
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(pCopyDone_));
            Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::V_S>();
            pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(copyDone_[0]);
            pipe.ReleaseEventID<HardEvent::MTE3_V>(pCopyDone_);
            if (copyPending_)
                WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
            pipe.ReleaseEventID<HardEvent::MTE3_S>(copyDone_[1]);
        } else {
            for (uint32_t i = 0; i < 2; ++i) {
                if (copyPending_ && (!dense_ || i == 1))
                    WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[i]));
                pipe.ReleaseEventID<HardEvent::MTE3_S>(copyDone_[i]);
            }
        }
        if (kHalfPending_) WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(kHalfDone_));
        pipe.ReleaseEventID<HardEvent::MTE3_S>(kHalfDone_);
    }
private:
    __aicore__ inline bool CanPrefetchV(uint32_t tile, int32_t request,
                                       const SglQsaCubeAttentionTilingData &t)
    {
        if (tile >= 32 || !Active(tile) || request < 0 || request >= static_cast<int32_t>(t.numRequests))
            return false;
        const uint32_t bits = groupMasks_.ReinterpretCast<uint32_t>().GetValue(tile / 2) >> ((tile & 1) * 16);
        return (bits & 0xffffU) == 0xffffU;
    }
    __aicore__ inline void BeginPrefetchV(uint32_t tile)
    {
        auto valid = validBase_[(tile & 3) * 2 * T];
        Duplicate(valid, int32_t(1), 2 * T);
        // Keep code/address live ranges bounded, unlike the two-query prototype.
        #pragma unroll 1
        for (uint32_t g = 0; g < 16; ++g)
            DataCopy(packedV_[((tile & 1) * T + g * 4) * D],
                     v_[GroupAddress(tile * T / 4 + g)], 4 * D);
    }
    __aicore__ inline void PublishPrefetchV(uint32_t tile, GM_ADDR ws, uint32_t group, uint32_t sub)
    {
        Slot<H, DENSE> next;
        next.Init(ws, group, tile & 3, sub);
        Fence<HardEvent::MTE2_MTE3>();
        Fence<HardEvent::S_MTE3>();
        DataCopy(next.packedV, packedV_[(tile & 1) * T * D], T * D);
    }
    __aicore__ inline void ConsumeDenseDelta(uint32_t tile, GM_ADDR ws, uint32_t group, uint32_t sub)
    {
        CrossCoreWaitFlag<2>((tile & 1) ? PV_NEXT_READY : PV_READY);
        AccumulateDenseDelta(tile, ws, group, sub);
    }
    __aicore__ inline void AccumulateDenseDelta(uint32_t tile, GM_ADDR ws, uint32_t group, uint32_t sub)
    {
        constexpr uint32_t heads = AIV_ROWS * H;
        if (!Active(tile)) return;
        slot_.Init(ws, group, tile & 3, sub);
        DataCopy(delta_, slot_.delta, heads * D);
        Fence<HardEvent::MTE2_V>();
        for (uint32_t h = 0; h < heads; ++h)
            Mul(acc_[h * D], acc_[h * D], denseAlpha_[(tile & 3) * heads * 8 + h * 8],
                uint64_t(T), D / T, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Add(acc_, acc_, delta_, heads * D);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void DenseMetadata(uint64_t row, uint64_t end, int32_t request,
                                         GM_ADDR ws, uint32_t group, uint32_t sub,
                                         const SglQsaCubeAttentionTilingData &t)
    {
        denseQueryBase_ = row;
        denseQueryEnd_ = end;
        denseRequest_ = request;
        const uint32_t capacity = t.pageTableWidth * t.pageTokens;
        GlobalTensor<int32_t> meta;
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            ws + uint64_t(t.usedCoreNum) * (4 * SLOT_BYTES + 256) + group * 256));
        denseScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
            ws + uint64_t(t.usedCoreNum) * (4 * SLOT_BYTES + 512) +
            uint64_t(group) * DENSE_M * capacity * sizeof(float)));
        const uint32_t queries = end - row;
        CopyInts(page_, t2r_[row], queries);
        Fence<HardEvent::MTE2_S>();
        int32_t high = -1;
        for (uint32_t q = sub; q < queries; q += 2) {
            if (request < 0 || request >= static_cast<int32_t>(t.numRequests) || page_.GetValue(q) != request) continue;
            if (t.topk == 514) {
                const int32_t value = CompactHigh(row + q, capacity);
                high = value > high ? value : high;
                continue;
            }
            CopyInts(ids_, indices_[(row + q) * TOPK], TOPK);
            Fence<HardEvent::MTE2_V>();
            auto values = packedK_.ReinterpretCast<float>();
            Cast(values, ids_, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            Mins(values, values, static_cast<float>(static_cast<int32_t>(capacity - 1)), 2048);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(temp_, values, T, 32, 8, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            Fence<HardEvent::V_S>();
            for (uint32_t i = 0; i < 32; ++i) {
                const int32_t value = static_cast<int32_t>(temp_.GetValue(i * 8));
                high = value > high ? value : high;
            }
            for (uint32_t i = 2048; i < TOPK; ++i) {
                int32_t value = ids_.GetValue(i);
                if (value >= static_cast<int32_t>(capacity)) value = capacity - 1;
                high = value > high ? value : high;
            }
        }
        req_.SetValue(0, high);
        Fence<HardEvent::S_MTE3>();
        DataCopy(meta[sub * 32], req_, 8);
        CrossCoreSetFlag<2, PIPE_MTE3>(DENSE_META_READY);
        CrossCoreWaitFlag<2>(DENSE_QK_READY);
        DataCopy(req_, meta[(1 - sub) * 32], 8);
        Fence<HardEvent::MTE2_S>();
        const int32_t other = req_.GetValue(0);
        high = other > high ? other : high;
        denseStride_ = high < 0 ? 128 : ((uint32_t(high) + 128) / 128) * 128;
    }
    __aicore__ inline int32_t CompactHigh(uint64_t row, uint32_t capacity)
    {
        CopyInts(groupFirst_, indices_[row * 514], 512);
        CopyInts(ids_[2064], indices_[row * 514 + 512], 2);
        Fence<HardEvent::MTE2_S>();
        const int32_t position = ids_.GetValue(2064), seqLen = ids_.GetValue(2065);
        const int32_t visible = (position + 1) / 4;
        const int32_t available = visible < seqLen / 4 ? visible : seqLen / 4;
        const int32_t complete = available < 512 ? available : 512;
        Fence<HardEvent::MTE2_V>();
        auto values = packedK_.ReinterpretCast<float>();
        ShiftLeft(groupOther_, groupFirst_, int32_t(2), 512);
        PipeBarrier<PIPE_V>();
        Cast(values, groupOther_, RoundMode::CAST_NONE, 512);
        PipeBarrier<PIPE_V>();
        CompareScalar(groupMasks_, values, 0.0f, CMPMODE::GE, 512);
        CompareScalar(groupTempMask_, values, float(seqLen), CMPMODE::LT, 512);
        PipeBarrier<PIPE_V>();
        And(groupMasks_.ReinterpretCast<uint16_t>(), groupMasks_.ReinterpretCast<uint16_t>(),
            groupTempMask_.ReinterpretCast<uint16_t>(), 32);
        Cast(acc_, groupOffsets_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, 512);
        Adds(values, values, 3.0f, 512);
        PipeBarrier<PIPE_V>();
        Muls(acc_, acc_, 1.0f / 16.0f, 512);
        if (complete < 512) Mins(values, values, float(seqLen - 1), 512);
        PipeBarrier<PIPE_V>();
        CompareScalar(groupTempMask_, acc_, float(complete), CMPMODE::LT, 512);
        PipeBarrier<PIPE_V>();
        And(groupMasks_.ReinterpretCast<uint16_t>(), groupMasks_.ReinterpretCast<uint16_t>(),
            groupTempMask_.ReinterpretCast<uint16_t>(), 32);
        Duplicate(acc_, -1.0f, 512);
        PipeBarrier<PIPE_V>();
        Select(values, groupMasks_, values, acc_, SELMODE::VSEL_TENSOR_TENSOR_MODE, 512);
        PipeBarrier<PIPE_V>();
        Mins(values, values, static_cast<float>(static_cast<int32_t>(capacity - 1)), 512);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(temp_, values, T, 8, 8, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        Fence<HardEvent::V_S>();
        int32_t high = -1;
        for (uint32_t i = 0; i < 8; ++i) {
            const int32_t value = static_cast<int32_t>(temp_.GetValue(i * 8));
            high = value > high ? value : high;
        }
        const int32_t tailStart = visible * 4, tailCount = position + 1 - tailStart;
        if (tailCount > 0 && tailStart >= 0 && tailStart < seqLen) {
            int32_t last = tailStart + tailCount - 1;
            if (last >= seqLen) last = seqLen - 1;
            if (last >= static_cast<int32_t>(capacity)) last = capacity - 1;
            high = last > high ? last : high;
        }
        return high;
    }
    __aicore__ inline void DenseScores(uint64_t row, uint32_t r, uint32_t tile, int32_t request,
                                       const SglQsaCubeAttentionTilingData &t)
    {
        if (row >= denseQueryEnd_ || request < 0 || request >= static_cast<int32_t>(t.numRequests) || request != denseRequest_) {
            Duplicate(score_[r * H * T], 0.0f, H * T);
            PipeBarrier<PIPE_V>();
            return;
        }
        auto scratch = packedK_.ReinterpretCast<float>();
        const uint32_t base = tile * T;
        const uint32_t head = (row - denseQueryBase_) * H;
        const uint32_t alignedHead = head & ~uint32_t(15), headOffset = head & 15;
        const uint16_t headBlocks = (headOffset + H + 15) / 16;
        bool full = false;
        if (tile < 32) {
            const uint32_t groups = groupMasks_.ReinterpretCast<uint32_t>().GetValue(r * 16 + tile / 2) >> ((tile & 1) * 16);
            full = (groups & 0xffffU) == 0xffffU;
        }
        if (full) {
            Fence<HardEvent::V_MTE2>();
            const DataCopyParams copy{headBlocks, 8, (DENSE_K_ROWS - 4) * 16 * sizeof(float) / 32, 0};
            if (t.topk == 514) {
                // Compact groups begin at 4*g; both panel sizes are multiples of4.
                for (uint32_t g = 0; g < T / 4; ++g) {
                    const uint32_t source = denseScoreOffsets_.GetValue(r * 512 + base / 4 + g);
                    DataCopy(scratch[g * 128], denseScore_[uint64_t(source) + alignedHead * DENSE_K_ROWS], copy);
                }
            } else {
            for (uint32_t g = 0; g < T / 4; ++g) {
                const uint32_t source = denseScoreOffsets_.GetValue(r * 512 + base / 4 + g);
                if (((source / 16) & (DENSE_K_ROWS - 1)) <= DENSE_K_ROWS - 4) {
                    DataCopy(scratch[g * 128], denseScore_[uint64_t(source) + alignedHead * DENSE_K_ROWS], copy);
                } else {
                    const int32_t token = ids_.GetValue(r * 2080 + base + g * 4);
                    const DataCopyParams single{headBlocks, 2, (DENSE_K_ROWS - 1) * 16 * sizeof(float) / 32, 6};
                    for (uint32_t j = 0; j < 4; ++j)
                        DataCopy(scratch[g * 128 + j * 16],
                                 denseScore_[DenseScoreAddress(token + j, alignedHead)], single);
                }
            }
            }
        } else {
            Duplicate(scratch, 0.0f, T * 32);
            Fence<HardEvent::V_MTE2>();
            const DataCopyParams copy{headBlocks, 2, (DENSE_K_ROWS - 1) * 16 * sizeof(float) / 32, 6};
            for (uint32_t i = 0; i < T && base + i < TOPK; ++i) {
                const int32_t token = ids_.GetValue(r * 2080 + base + i);
                if (valid_.GetValue(r * T + i) != 0 && token >= 0 && token < static_cast<int32_t>(denseStride_)) {
                    DataCopy(scratch[(i / 4) * 128 + (i % 4) * 16],
                             denseScore_[DenseScoreAddress(token, alignedHead)], copy);
                }
            }
        }
        Fence<HardEvent::MTE2_V>();
        for (uint32_t h = 0; h < H; ++h) {
            const uint32_t lane = headOffset + h;
            Gather(score_[(r * H + h) * T], scratch, denseTokenOffsets_,
                   ((lane / 16) * 64 + lane % 16) * sizeof(float), T);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint64_t DenseScoreAddress(uint32_t token, uint32_t alignedHead)
    {
        return uint64_t(token / DENSE_K_ROWS) * DENSE_K_ROWS * DENSE_M +
               alignedHead * DENSE_K_ROWS + (token % DENSE_K_ROWS) * 16;
    }
    __aicore__ inline void CopyInts(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, uint32_t n)
    {
        DataCopyExtParams p{1, n * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, p, pad);
    }
    __aicore__ inline void ExpandGroups(uint32_t r)
    {
        // The E3 result stays in UB. Per-row writable scratch is dead until
        // attention; shared H12 copies its immutable patterns from calc's tail.
        auto ids = ids_[r * 2080];
        const int32_t position = ids.GetValue(2064);
        const int32_t seqLen = ids.GetValue(2065);
        const int32_t visible = (position + 1) / 4;
        const int32_t available = visible < seqLen / 4 ? visible : seqLen / 4;
        const int32_t complete = available < 512 ? available : 512;
        const int32_t tailStart = visible * 4;
        const int32_t remainder = position + 1 - tailStart;
        const int32_t tailCount = remainder < 3 ? remainder : 3;
        auto offsets = acc_.ReinterpretCast<int32_t>();
        auto lanes = acc_[2048].ReinterpretCast<int32_t>();
        auto valid = calcBuf_.Get<uint8_t>()[16384];
        auto part = valid[256];
        // A complete group in this range expands to four exact FP32 integers.
        // Validate 512 group IDs once instead of masking/converting 2048 tokens.
        fullGroups_[r] = complete == 512 && position < FP32_EXACT_INTEGER_MAX &&
                         seqLen <= FP32_EXACT_INTEGER_MAX;
        if (fullGroups_[r]) {
            Cast(acc_, groupFirst_, RoundMode::CAST_NONE, 512);
            PipeBarrier<PIPE_V>();
            CompareScalar(groupMasks_[r * 64], acc_, 0.0f, CMPMODE::GE, 512);
            CompareScalar(groupTempMask_, acc_, float(seqLen / 4), CMPMODE::LT, 512);
            PipeBarrier<PIPE_V>();
            And(groupMasks_[r * 64].ReinterpretCast<uint16_t>(),
                groupMasks_[r * 64].ReinterpretCast<uint16_t>(),
                groupTempMask_.ReinterpretCast<uint16_t>(), 32);
        }
        // CreateVecIndex first writes one block through Scalar, then uses V.
        // Protect that Scalar write when recycling a vector scratch region.
        Fence<HardEvent::V_S>();
        if (fullGroups_[r]) {
            uint64_t bits = ~uint64_t(0);
            for (uint32_t i = 0; i < 8; ++i)
                bits &= groupMasks_.ReinterpretCast<uint64_t>().GetValue(r * 8 + i);
            fullGroups_[r] = bits == ~uint64_t(0);
        }
        if constexpr (PREFETCH_V) {
            Adds(offsets, calcBuf_.Get<int32_t>()[CALC_BYTES / sizeof(int32_t)], int32_t(0), 2048);
        } else {
            CreateVecIndex(offsets, int32_t(0), 2048);
            PipeBarrier<PIPE_V>();
            ShiftRight(offsets, offsets, int32_t(2), 2048);
            PipeBarrier<PIPE_V>();
            ShiftLeft(offsets, offsets, int32_t(2), 2048);
        }
        PipeBarrier<PIPE_V>();
        Gather(ids, groupFirst_, offsets.ReinterpretCast<uint32_t>(), uint32_t(0), 2048);
        PipeBarrier<PIPE_V>();
        if (!fullGroups_[r]) {
            Cast(lanes.ReinterpretCast<float>(), ids, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();
            CompareScalar(valid, lanes.ReinterpretCast<float>(), 0.0f, CMPMODE::GE, 2048);
            PipeBarrier<PIPE_V>();
        }
        ShiftLeft(ids, ids, int32_t(2), 2048);
        Fence<HardEvent::V_S>();
        if constexpr (PREFETCH_V) {
            Adds(lanes, calcBuf_.Get<int32_t>()[CALC_BYTES / sizeof(int32_t) + 2048], int32_t(0), 2048);
        } else {
            CreateVecIndex(lanes, int32_t(0), 2048);
            PipeBarrier<PIPE_V>();
            Sub(lanes, lanes, offsets, 2048);
        }
        PipeBarrier<PIPE_V>();
        Add(ids, ids, lanes, 2048);
        PipeBarrier<PIPE_V>();
        if (!fullGroups_[r]) {
            // Match E3's full-body group validity and partial-body token validity.
            if (complete == 512) {
                Sub(lanes, ids, lanes, 2048);
                PipeBarrier<PIPE_V>();
                Cast(offsets.ReinterpretCast<float>(), lanes, RoundMode::CAST_NONE, 2048);
            } else {
                Cast(offsets.ReinterpretCast<float>(), ids, RoundMode::CAST_NONE, 2048);
            }
            PipeBarrier<PIPE_V>();
            CompareScalar(part, offsets.ReinterpretCast<float>(), float(seqLen), CMPMODE::LT, 2048);
            PipeBarrier<PIPE_V>();
            And(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(), part.ReinterpretCast<uint16_t>(), 128);
            if (complete < 512) {
                Fence<HardEvent::V_S>();
                CreateVecIndex(lanes, int32_t(0), 2048);
                PipeBarrier<PIPE_V>();
                Cast(offsets.ReinterpretCast<float>(), lanes, RoundMode::CAST_NONE, 2048);
                PipeBarrier<PIPE_V>();
                CompareScalar(part, offsets.ReinterpretCast<float>(), float(complete * 4), CMPMODE::LT, 2048);
                PipeBarrier<PIPE_V>();
                And(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(), part.ReinterpretCast<uint16_t>(), 128);
            }
            Cast(offsets.ReinterpretCast<float>(), ids, RoundMode::CAST_NONE, 2048);
            Duplicate(lanes.ReinterpretCast<float>(), -1.0f, 2048);
            PipeBarrier<PIPE_V>();
            Select(offsets.ReinterpretCast<float>(), valid, offsets.ReinterpretCast<float>(),
                   lanes.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 2048);
            PipeBarrier<PIPE_V>();
            Cast(ids, offsets.ReinterpretCast<float>(), RoundMode::CAST_RINT, 2048);
        }
        Fence<HardEvent::V_S>();
        for (uint32_t j = 0; j < 3; ++j) ids.SetValue(2048 + j, -1);
        for (int32_t j = 0; j < tailCount; ++j) {
            const int32_t token = tailStart + j;
            ids.SetValue(complete * 4 + j, token >= 0 && token < seqLen ? token : -1);
        }
        Fence<HardEvent::S_V>();
    }
    __aicore__ inline void BuildGroupMask(uint32_t r, const SglQsaCubeAttentionTilingData &t)
    {
        Gather(groupFirst_, ids_[r * 2080], groupOffsets_, uint32_t(0), 512);
        PipeBarrier<PIPE_V>();
        // The full compact path already set every group bit and guarantees
        // group4 order. Page/cache validity is still checked below.
        if (!fullGroups_[r]) {
            for (uint32_t j = 1; j < 4; ++j) {
                Gather(groupOther_, ids_[r * 2080], groupOffsets_, j * 4, 512);
                PipeBarrier<PIPE_V>();
                Sub(groupOther_, groupOther_, groupFirst_, 512);
                PipeBarrier<PIPE_V>();
                CompareScalar(groupTempMask_, groupOther_, int32_t(j), CMPMODE::EQ, 512);
                PipeBarrier<PIPE_V>();
                if (j == 1) {
                    DataCopy(groupMasks_[r * 64], groupTempMask_, 64);
                } else {
                    And(groupMasks_[r * 64].ReinterpretCast<uint16_t>(),
                        groupMasks_[r * 64].ReinterpretCast<uint16_t>(),
                        groupTempMask_.ReinterpretCast<uint16_t>(), 32);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        // Validate group4 addresses once per row; keep invalid/non-contiguous
        // tiles on the original per-index path.
        // A3 int32 CompareScalar only implements EQ. Cast for range checks.
        auto pageFloat = groupOther_.ReinterpretCast<float>();
        auto offsets = groupPageOffsets_[r * 512];
        Cast(pageFloat, groupFirst_, RoundMode::CAST_NONE, 512);
        PipeBarrier<PIPE_V>();
        CompareScalar(groupTempMask_, pageFloat, 0.0f, CMPMODE::GE, 512);
        MergeGroupMask(r);
        CompareScalar(groupTempMask_, pageFloat, static_cast<float>(static_cast<int32_t>(t.pageTableWidth * pageTokens_)), CMPMODE::LT, 512);
        MergeGroupMask(r);
        Maxs(pageFloat, pageFloat, 0.0f, 512);
        PipeBarrier<PIPE_V>();
        Mins(pageFloat, pageFloat, static_cast<float>(static_cast<int32_t>(t.pageTableWidth * pageTokens_ - 1)), 512);
        PipeBarrier<PIPE_V>();
        Muls(pageFloat, pageFloat, 1.0f / static_cast<int32_t>(pageTokens_), 512);
        PipeBarrier<PIPE_V>();
        Cast(offsets, pageFloat, RoundMode::CAST_FLOOR, 512);
        PipeBarrier<PIPE_V>();
        ShiftLeft(groupOther_.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), uint32_t(2), 512);
        PipeBarrier<PIPE_V>();
        Gather(physicalGroups_[r * 512], page_[r * PAGE_STRIDE], groupOther_.ReinterpretCast<uint32_t>(), uint32_t(0), 512);
        Muls(offsets, offsets, int32_t(pageTokens_), 512);
        PipeBarrier<PIPE_V>();
        Sub(offsets, groupFirst_, offsets, 512);
        PipeBarrier<PIPE_V>();
        Cast(pageFloat, offsets, RoundMode::CAST_NONE, 512);
        PipeBarrier<PIPE_V>();
        CompareScalar(groupTempMask_, pageFloat, static_cast<float>(static_cast<int32_t>(pageTokens_ - 4)), CMPMODE::LE, 512);
        MergeGroupMask(r);
        Cast(pageFloat, physicalGroups_[r * 512], RoundMode::CAST_NONE, 512);
        PipeBarrier<PIPE_V>();
        CompareScalar(groupTempMask_, pageFloat, 0.0f, CMPMODE::GE, 512);
        MergeGroupMask(r);
        CompareScalar(groupTempMask_, pageFloat, static_cast<float>(static_cast<int32_t>(t.numCacheBlocks)), CMPMODE::LT, 512);
        MergeGroupMask(r);
        if (linearAddresses_) {
            // Page stride is 3 << 16 (TP8) or 3 << 17 (TP4); keep integer addresses exact.
            auto physical = physicalGroups_[r * 512];
            auto offset = groupPageOffsets_[r * 512];
            ShiftLeft(groupOther_.ReinterpretCast<uint32_t>(), physical.ReinterpretCast<uint32_t>(), uint32_t(1), 512);
            ShiftLeft(offset.ReinterpretCast<uint32_t>(), offset.ReinterpretCast<uint32_t>(), uint32_t(8), 512);
            PipeBarrier<PIPE_V>();
            if (pageTokens_ != 128) Add(physical, physical, groupOther_, 512);
            PipeBarrier<PIPE_V>();
            ShiftLeft(physical.ReinterpretCast<uint32_t>(), physical.ReinterpretCast<uint32_t>(), uint32_t(pageTokens_ == 128 ? 15 : (pageTokens_ == 768 ? 16 : 17)), 512);
            PipeBarrier<PIPE_V>();
            Add(physical, physical, offset, 512);
        }
        if constexpr (dense_) {
            // Cache the token-panel part once per row. The query's head block
            // is added at submission; invalid groups never consume this table.
            auto scoreOffset = denseScoreOffsets_[r * 512];
            auto panel = groupOther_.ReinterpretCast<uint32_t>();
            ShiftRight(panel, groupFirst_.ReinterpretCast<uint32_t>(), uint32_t(H == 12 ? 6 : 7), 512);
            PipeBarrier<PIPE_V>();
            ShiftLeft(panel, panel, uint32_t(H == 12 ? 6 : 7), 512);
            PipeBarrier<PIPE_V>();
            Sub(scoreOffset.ReinterpretCast<int32_t>(), groupFirst_, panel.ReinterpretCast<int32_t>(), 512);
            PipeBarrier<PIPE_V>();
            ShiftLeft(scoreOffset, scoreOffset, uint32_t(4), 512);
            ShiftLeft(panel, panel, uint32_t(7), 512);
            PipeBarrier<PIPE_V>();
            Add(scoreOffset.ReinterpretCast<int32_t>(), scoreOffset.ReinterpretCast<int32_t>(),
                panel.ReinterpretCast<int32_t>(), 512);
        }
        Fence<HardEvent::V_S>();
    }
    __aicore__ inline void MergeGroupMask(uint32_t r)
    {
        PipeBarrier<PIPE_V>();
        And(groupMasks_[r * 64].ReinterpretCast<uint16_t>(),
            groupMasks_[r * 64].ReinterpretCast<uint16_t>(),
            groupTempMask_.ReinterpretCast<uint16_t>(), 32);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline int64_t GroupAddress(uint32_t index)
    {
        const int64_t physical = physicalGroups_.GetValue(index);
        if (linearAddresses_) return physical;
        return (physical * pageTokens_ + groupPageOffsets_.GetValue(index)) * D;
    }
    __aicore__ inline void PrepareTile(uint32_t tile, const int32_t *request,
                                      const SglQsaCubeAttentionTilingData &t, bool deferV = false)
    {
        if (!Active(tile)) {
            if (!dense_ || tile == 0) CrossCoreSetFlag<2, PIPE_MTE3>((tile & 1) != 0 ?
                ((tile & 2) ? V_ODD_NEXT_READY : V_ODD_READY) :
                ((tile & 2) ? V_EVEN_NEXT_READY : V_EVEN_READY));
            return;
        }
        valid_ = validBase_[(tile & 3) * 2 * T];
        const uint32_t base = tile * T;
        const uint32_t cols = TOPK - base < T ? TOPK - base : T;
        Duplicate(valid_, int32_t(1), 2 * T);
        Fence<HardEvent::V_MTE2>();
        Fence<HardEvent::V_S>();
        const bool requestsValid = request[0] >= 0 && request[0] < static_cast<int32_t>(t.numRequests) &&
            (AIV_ROWS == 1 || (request[1] >= 0 && request[1] < static_cast<int32_t>(t.numRequests)));
        if (tile < 32 && requestsValid) {
            const uint32_t shift = (tile & 1) * 16;
            const uint32_t groups0 = groupMasks_.ReinterpretCast<uint32_t>().GetValue(tile / 2) >> shift;
            const uint32_t groups1 = AIV_ROWS == 1 ? groups0 :
                groupMasks_.ReinterpretCast<uint32_t>().GetValue(16 + tile / 2) >> shift;
            if (((groups0 & groups1) & 0xffffU) == 0xffffU) {
                int64_t address[32];
                if constexpr (!dense_ || AIV_ROWS == 1) {
                    #pragma unroll
                    for (uint32_t r = 0; r < AIV_ROWS; ++r) {
                        #pragma unroll
                        for (uint32_t g = 0; g < 16; ++g) {
                            const uint32_t index = r * 512 + base / 4 + g;
                            address[r * 16 + g] = GroupAddress(index);
                        }
                    }
                }
                // Two N32-per-query halves reuse the existing 64 KiB K UB.
                // Keep GM's two complete N64 query rows unchanged.
                const DataCopyParams copyHalf{AIV_ROWS, (T / 2) * D * 2 / 32, 0, (T / 2) * D * 2 / 32};
                for (uint32_t half = 0; half < 2 && !dense_; ++half) {
                    const auto done = static_cast<event_t>(half == 0 ? copyDone_[0] : kHalfDone_);
                    if (half == 0 ? copyPending_ : kHalfPending_) WaitFlag<HardEvent::MTE3_S>(done);
                    #pragma unroll
                    for (uint32_t r = 0; r < AIV_ROWS; ++r) {
                        #pragma unroll
                        for (uint32_t g = 0; g < 8; ++g)
                            DataCopy(packedK_[(half * (H == 12 ? T / 2 : T) + r * (T / 2) + g * 4) * D],
                                     k_[address[r * 16 + half * 8 + g]], 4 * D);
                    }
                    Fence<HardEvent::MTE2_MTE3>();
                    Fence<HardEvent::S_MTE3>();
                    DataCopy(slot_.packedK[half * (T / 2) * D],
                             packedK_[half * (H == 12 ? T / 2 : T) * D], copyHalf);
                    SetFlag<HardEvent::MTE3_S>(done);
                    if (half == 1) kHalfPending_ = true;
                }
                if (deferV) {
                    deferredVMask_ |= 1U << (tile & 1);
                    return;
                }
                if (copyPending_) WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
                if constexpr (dense_ && AIV_ROWS > 1) {
                    // Two owned queries would keep 32 addresses live. Submit
                    // each shared V address directly instead.
                    #pragma unroll 1
                    for (uint32_t g = 0; g < AIV_ROWS * 16; ++g) {
                        const uint32_t index = (g / 16) * 512 + base / 4 + (g & 15);
                        DataCopy(packedV_[g * 4 * D], v_[GroupAddress(index)], 4 * D);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t g = 0; g < AIV_ROWS * 16; ++g)
                        DataCopy(packedV_[g * 4 * D], v_[address[g]], 4 * D);
                }
                Fence<HardEvent::MTE2_MTE3>();
                Fence<HardEvent::S_MTE3>();
                DataCopy(slot_.packedV, packedV_, AIV_ROWS * T * D);
                SetFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
                copyPending_ = true;
                if (!dense_ || tile == 0) CrossCoreSetFlag<2, PIPE_MTE3>((tile & 1) != 0 ?
                    ((tile & 2) ? V_ODD_NEXT_READY : V_ODD_READY) :
                    ((tile & 2) ? V_EVEN_NEXT_READY : V_EVEN_READY));
                return;
            }
        }
        if (copyPending_) {
            if (!dense_) WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[0]));
            WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
        }
        if (kHalfPending_) WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(kHalfDone_));
        for (uint32_t r = 0; r < AIV_ROWS; ++r) {
            const bool requestValid = request[r] >= 0 && request[r] < static_cast<int32_t>(t.numRequests);
            const uint32_t groups = tile < 32 && requestValid ?
                groupMasks_.ReinterpretCast<uint32_t>().GetValue(r * 16 + tile / 2) >> ((tile & 1) * 16) : 0;
            if ((groups & 0xffffU) == 0xffffU) {
                int64_t address[16];
                #pragma unroll
                for (uint32_t g = 0; g < 16; ++g) {
                    const uint32_t index = r * 512 + base / 4 + g;
                    address[g] = GroupAddress(index);
                }
                #pragma unroll
                for (uint32_t g = 0; g < 16; ++g) {
                    if (!dense_) DataCopy(packedK_[(r * T + 4 * g) * D], k_[address[g]], 4 * D);
                    DataCopy(packedV_[(r * T + 4 * g) * D], v_[address[g]], 4 * D);
                }
                continue;
            }
            for (uint32_t i = 0; i < T;) {
                int64_t offset = -1;
                uint32_t count = 1;
                int32_t token = -1;
                if (i < cols && requestValid) {
                    token = ids_.GetValue(r * 2080 + base + i);
                    if (token >= 0 && token / pageTokens_ < t.pageTableWidth) {
                        const uint32_t pageOffset = token % pageTokens_;
                        if ((i & 3) == 0 && (groups & (1U << (i / 4))) != 0 &&
                            pageOffset + 3 < pageTokens_) count = 4;
                        const int32_t physical = page_.GetValue(r * PAGE_STRIDE + token / pageTokens_);
                        if (physical >= 0 && physical < static_cast<int32_t>(t.numCacheBlocks))
                            offset = (static_cast<int64_t>(physical) * pageTokens_ + pageOffset) * D;
                    }
                }
                if (offset >= 0) {
                    if (!dense_) DataCopy(packedK_[(r * T + i) * D], k_[offset], count * D);
                    DataCopy(packedV_[(r * T + i) * D], v_[offset], count * D);
                } else {
                    if (!requestValid || i >= cols) count = T - i;
                    else if (token < 0) {
                        // Coalesce negative-index runs; values and masks stay zero.
                        while (i + count < cols && ids_.GetValue(r * 2080 + base + i + count) < 0)
                            ++count;
                    }
                    // Valid lanes are fully overwritten by DMA; zero only holes.
                    if (!dense_) Duplicate(packedK_.ReinterpretCast<uint16_t>()[(r * T + i) * D], uint16_t(0), count * D);
                    Duplicate(packedV_.ReinterpretCast<uint16_t>()[(r * T + i) * D], uint16_t(0), count * D);
                    for (uint32_t j = 0; j < count; ++j) valid_.SetValue(r * T + i + j, 0);
                }
                i += count;
            }
        }
        Fence<HardEvent::V_MTE3>();
        Fence<HardEvent::MTE2_MTE3>();
        Fence<HardEvent::S_MTE3>();
        if (!dense_) {
            DataCopy(slot_.packedK, packedK_, AIV_ROWS * T * D);
            SetFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[0]));
            SetFlag<HardEvent::MTE3_S>(static_cast<event_t>(kHalfDone_));
            kHalfPending_ = true;
        }
        DataCopy(slot_.packedV, packedV_, AIV_ROWS * T * D);
        SetFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
        copyPending_ = true;
        if (!dense_ || tile == 0) CrossCoreSetFlag<2, PIPE_MTE3>((tile & 1) != 0 ?
            ((tile & 2) ? V_ODD_NEXT_READY : V_ODD_READY) :
            ((tile & 2) ? V_EVEN_NEXT_READY : V_EVEN_READY));
    }
    __aicore__ inline void PrepareDeferredV(uint32_t tile)
    {
        int64_t address[32];
        #pragma unroll
        for (uint32_t r = 0; r < AIV_ROWS; ++r) {
            #pragma unroll
            for (uint32_t g = 0; g < 16; ++g) {
                const uint32_t index = r * 512 + tile * T / 4 + g;
                address[r * 16 + g] = GroupAddress(index);
            }
        }
        WaitFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
        #pragma unroll
        for (uint32_t g = 0; g < AIV_ROWS * 16; ++g)
            DataCopy(packedV_[g * 4 * D], v_[address[g]], 4 * D);
        Fence<HardEvent::MTE2_MTE3>();
        Fence<HardEvent::S_MTE3>();
        DataCopy(slot_.packedV, packedV_, AIV_ROWS * T * D);
        SetFlag<HardEvent::MTE3_S>(static_cast<event_t>(copyDone_[1]));
        CrossCoreSetFlag<2, PIPE_MTE3>((tile & 1) != 0 ?
            ((tile & 2) ? V_ODD_NEXT_READY : V_ODD_READY) :
            ((tile & 2) ? V_EVEN_NEXT_READY : V_EVEN_READY));
        deferredVMask_ &= ~(1U << (tile & 1));
    }
    static constexpr bool dense_ = DENSE;
    uint64_t denseQueryBase_ = 0;
    uint64_t denseQueryEnd_ = 0;
    int32_t denseRequest_ = -1;
    uint32_t denseStride_ = 0;
    GlobalTensor<float> denseScore_;
    LocalTensor<uint32_t> denseTokenOffsets_, denseScoreOffsets_;
    LocalTensor<float> denseAlpha_;
    uint32_t pageTokens_ = 768;
    bool linearAddresses_ = false;
    bool fullGroups_[2] = {false, false};
    uint32_t deferredVMask_ = 0;
    TEventID kHalfDone_;
    bool kHalfPending_ = false;
    TEventID copyDone_[2];
    bool copyPending_ = false;
    __aicore__ inline bool Active(uint32_t tile) const { return (activePairs_ & (1U << (tile / 2))) != 0; }
    uint32_t activePairs_ = 0;
    Slot<H, DENSE> slot_;
    GlobalTensor<int32_t> indices_, table_, t2r_;
    GlobalTensor<bfloat16_t> out_, k_, v_;
    TBuf<TPosition::VECCALC> metaBuf_, calcBuf_, kvBuf_, groupBuf_, addressBuf_;
    LocalTensor<int32_t> page_, ids_, valid_, validBase_, req_;
    LocalTensor<float> acc_, delta_, score_, prob_, mask_, reduce_, temp_, alpha_, maximum_, nextMax_, normalizer_;
    LocalTensor<bfloat16_t> probBf_, outBf_, packedK_, packedV_;
    bool prefetchCopyPending_ = false;
    TEventID pCopyDone_;
    bool pCopyPending_ = false;
    LocalTensor<uint8_t> maskBits_, groupMasks_, groupTempMask_;
    LocalTensor<uint32_t> groupOffsets_;
    LocalTensor<int32_t> groupFirst_, groupOther_, physicalGroups_, groupPageOffsets_;
};
} // namespace

extern "C" __global__ __aicore__ void sgl_qsa_cube_attention(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR indices, GM_ADDR table,
    GM_ADDR tokenToReq, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(t, tiling);
    TPipe pipe;
    GM_ADDR scratch = GetUserWorkspace(workspace);
    if ASCEND_IS_AIC {
        if (TILING_KEY_IS(25)) {
            Cube<12, true> op;
            op.Run(q, k, table, tokenToReq, scratch, t, pipe);
        } else if (TILING_KEY_IS(24)) {
            Cube<12> op;
            op.Run(q, k, table, tokenToReq, scratch, t, pipe);
        } else if (TILING_KEY_IS(13)) {
            Cube<6, true> op;
            op.Run(q, k, table, tokenToReq, scratch, t, pipe);
        } else if (TILING_KEY_IS(12)) {
            Cube<6> op;
            op.Run(q, k, table, tokenToReq, scratch, t, pipe);
        } else if (TILING_KEY_IS(7)) {
            Cube<3, true> op;
            op.Run(q, k, table, tokenToReq, scratch, t, pipe);
        } else if (TILING_KEY_IS(6)) {
            Cube<3> op;
            op.Run(q, k, table, tokenToReq, scratch, t, pipe);
        }
    } else {
        if (TILING_KEY_IS(25)) {
            Vector<12, true> op;
            op.Run(k, v, indices, table, tokenToReq, out, scratch, t, pipe);
        } else if (TILING_KEY_IS(24)) {
            Vector<12> op;
            op.Run(k, v, indices, table, tokenToReq, out, scratch, t, pipe);
        } else if (TILING_KEY_IS(13)) {
            Vector<6, true> op;
            op.Run(k, v, indices, table, tokenToReq, out, scratch, t, pipe);
        } else if (TILING_KEY_IS(12)) {
            Vector<6> op;
            op.Run(k, v, indices, table, tokenToReq, out, scratch, t, pipe);
        } else if (TILING_KEY_IS(7)) {
            Vector<3, true> op;
            op.Run(k, v, indices, table, tokenToReq, out, scratch, t, pipe);
        } else if (TILING_KEY_IS(6)) {
            Vector<3> op;
            op.Run(k, v, indices, table, tokenToReq, out, scratch, t, pipe);
        }
    }
}
