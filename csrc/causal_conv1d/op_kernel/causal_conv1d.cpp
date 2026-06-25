// causal_conv1d.cpp — PTO-ISA depthwise causal conv1d + bias + (opt) SiLU.
//
// Drop-in for sgl's AscendC causal_conv1d, written on the PTO tile ISA (like
// csrc/mega_chunk_gdn). Same op:
//   y[t,c] = act( bias[c] + sum_{k=0..K-1} W[k,c]*xext[t+k,c] ),
//   xext = [history(K-1 rows, from conv_states if has_initial_state else 0), x]
// Per channel a K-tap depthwise filter. The filter width K is a RUNTIME argument;
// the compile-time template parameter is the accumulator-ring size RS (a power of
// two, K <= RS). The host routes a request to the RS = roundUpToPow2(K) variant, so
// any width in [2, 64] is served by the six compiled RS variants {2,4,8,16,32,64}.
// fp16/bf16 I/O, fp32 accumulate. Weights/bias enter native and are cast to fp32 on device.
//
// Work grid (in-kernel, uniform task striding): batch x blocksPerSeq x lchunks.
//   blocksPerSeq = ceil(dim/col_w) (channel tiles), lchunks splits the L axis so
//   all cores stay busy even at small batch. Causal halo = K-1 replayed rows per
//   chunk. State writeback is a SEPARATE launch (causal_conv1d_wb_*) to avoid a race
//   between chunk-0's history read and the tail chunk's state write.
//
// Scalars (qsl/cidx/hinit) are read by direct __gm__ pointer indexing (same as
// mega_kernel's cu_seqlens). NOTE codegen needs "type* name", not "type *name".

#include <pto/pto-inst.hpp>

// clang-format off
// The AscendC launch codegen parses the expanded __global__ signature and needs
// "type* name" (pointer glued to the type); PointerAlignment: Right would turn
// it into "type *name" and the generated launch stub loses the parameter names.
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format on

using namespace pto;

namespace cc1d {

// RS (compile-time, power of two) sizes the accumulator ring and the entire UB
// layout; K (runtime, <= RS) only drives loop bounds, so one RS variant serves
// every width with roundUpToPow2(width) == RS. MAX_W is the compile-time per-RS
// channel-tile capacity. Ascend 910B2 AIV UB = 192 KiB; the static_assert in
// convChunk checks the chosen (RS, MAX_W) layout fits.
constexpr uint32_t UB_BYTES_PER_CORE = 192u * 1024u;

template <typename TileT>
AICORE inline void applySiluToTile(TileT &dst, TileT &src, TileT &tmp)
{
    using T = typename TileT::DType;
    TMULS(tmp, src, (T)-1);
    pipe_barrier(PIPE_V);
    TEXP(tmp, tmp);
    pipe_barrier(PIPE_V);
    TADDS(tmp, tmp, (T)1);
    pipe_barrier(PIPE_V);
    TDIV(dst, src, tmp);
}

// One conv chunk: outputs [l0,l1) for channels [c0,c0+lanes) of a sequence whose
// tokens start at element row `start` (token index). history from convStates.
template <typename IoElemType, uint32_t RS, uint32_t MAX_W>
AICORE inline void convChunk(__gm__ IoElemType *x, __gm__ IoElemType *y, __gm__ IoElemType *wgt, __gm__ IoElemType *bia,
                             __gm__ IoElemType *convStates, uint32_t dim, uint32_t stateLen, uint32_t K, uint32_t start,
                             uint32_t len, uint32_t cacheIdx, bool hasInit, uint32_t c0, int32_t lanes, uint32_t l0,
                             uint32_t l1, uint32_t activation, uint32_t hasBias)
{
    using GlobalShape = pto::Shape<1, 1, 1, 1, DYNAMIC>;
    using GlobalStride = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalIoTensor = pto::GlobalTensor<IoElemType, GlobalShape, GlobalStride>;
    using IoTile = Tile<TileType::Vec, IoElemType, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;
    using AccumTile = Tile<TileType::Vec, float, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;

    constexpr uint32_t accumTileBytes = MAX_W * sizeof(float);
    constexpr uint32_t ioTileBytes = MAX_W * sizeof(IoElemType);

    // UB byte offsets, all compile-time on RS: the layout is sized for the worst
    // case K == RS so no offset depends on the runtime K. fp32 region: RS weights
    // (weight k at k*accumTileBytes) | bias | RS accumulators | RS-1 temps | xin_f.
    // Then the I/O region: 4 ioTileBytes-sized tiles (input load double-buffered:
    // xin_h[0] | out0 | out1 | xin_h[1]).
    constexpr uint32_t ubBiasOffset = RS * accumTileBytes;
    constexpr uint32_t ubAccumRingBase = (RS + 1u) * accumTileBytes;
    constexpr uint32_t ubProductBase = (2u * RS + 1u) * accumTileBytes;  // temp k at ubProductBase+(k-1)*accumTileBytes
    constexpr uint32_t ubInputFp32 = (3u * RS) * accumTileBytes;
    constexpr uint32_t ubIoBase = (3u * RS + 1u) * accumTileBytes;  // I/O region base
    static_assert(ubIoBase + 4u * ioTileBytes <= UB_BYTES_PER_CORE,
                  "conv1d UB exceeds UB_BYTES_PER_CORE: lower RS/MAX_W or raise it");
    // NOTE: keep these `const`, NOT `constexpr`. These arrays are indexed by a
    // runtime value in the hot loop; making them constexpr makes the ascendc/cce
    // compiler emit ~5x slower device code (measured 951us vs 186us at B8/L512/
    // d6144). (Harmless under the bisheng JIT path, but not here.)
    const uint32_t ubOutputOffset[2] = {ubIoBase + ioTileBytes, ubIoBase + 2u * ioTileBytes};
    const uint32_t ubInputOffset[2] = {ubIoBase, ubIoBase + 3u * ioTileBytes};

    // weights/bias arrive native (fp16/bf16); cast them to the resident fp32 tiles on
    // device. The accumulator/temp/xin_f region is idle until the input loop, so use it
    // as scratch: stage all K(+bias) native tiles there (the loads pipeline on MTE2 just
    // like a plain load), one MTE2->V barrier, then cast each (the TCVTs pipeline on V).
    // EVENT_ID3 here is the same load barrier the original used and is reused below.
    constexpr uint32_t ubStageBase = ubAccumRingBase;  // accumulators + temps + xin_f are free here
    static_assert((RS + 1u) * ioTileBytes <= 2u * RS * accumTileBytes,
                  "conv1d: native weight/bias staging does not fit the scratch region");
    // The PREVIOUS task's output phase reads this same region with V (TCVT(outT, acc));
    // drain V before our staging MTE2 overwrites it -- otherwise a cross-task WAR
    // corrupts that task's output. Self-contained on ID0 (clean here; reused by the loop).
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (uint32_t k = 0; k < K; ++k) {
        GlobalIoTensor wG(wgt + (uint64_t)k * dim + c0, {lanes});
        IoTile wStage(lanes);
        TASSIGN(wStage, ubStageBase + k * ioTileBytes);
        TLOAD(wStage, wG);
    }
    if (hasBias) {
        GlobalIoTensor bG(bia + c0, {lanes});
        IoTile bStage(lanes);
        TASSIGN(bStage, ubStageBase + K * ioTileBytes);
        TLOAD(bStage, bG);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);  // all native tiles staged before any cast
    for (uint32_t k = 0; k < K; ++k) {
        IoTile wStage(lanes);
        AccumTile wT(lanes);
        TASSIGN(wStage, ubStageBase + k * ioTileBytes);
        TASSIGN(wT, k * accumTileBytes);
        TCVT(wT, wStage, pto::RoundMode::CAST_NONE);
    }
    if (hasBias) {
        IoTile bStage(lanes);
        AccumTile bT(lanes);
        TASSIGN(bStage, ubStageBase + K * ioTileBytes);
        TASSIGN(bT, ubBiasOffset);
        TCVT(bT, bStage, pto::RoundMode::CAST_NONE);
    }
    // The cast TCVTs (V) finish before the input loop's first TMUL/TCVT (also V, in
    // program order) reuses this scratch region -- no extra sync needed.

    // double-buffered input: two load slots with independent handshakes.
    // EVENT_ID3 is reused here (the weight/bias load above already consumed it).
    const event_t IEV[2] = {EVENT_ID0, EVENT_ID3};
    set_flag(PIPE_V, PIPE_MTE2, IEV[0]);  // xin_h[0] initially free
    set_flag(PIPE_V, PIPE_MTE2, IEV[1]);  // xin_h[1] initially free
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);

    // first input row to process (signed): l0==0 with history -> replay K-1 rows.
    const bool zeroPad = (l0 == 0u) && !hasInit;
    int32_t jstart;
    if (l0 == 0u)
        jstart = hasInit ? -(int32_t)(K - 1) : 0;
    else
        jstart = (int32_t)l0 - (int32_t)(K - 1);

    // PROLOGUE: load the first input row (jstart) so iter 0 can prefetch the next.
    // Input row index e: e>=0 -> x[start+e]; e<0 -> conv_states history row (K-1)+e.
    if (jstart < (int32_t)l1) {
        IoTile xin_h0(lanes);
        TASSIGN(xin_h0, ubInputOffset[0]);
        wait_flag(PIPE_V, PIPE_MTE2, IEV[0]);
        if (jstart >= 0) {
            GlobalIoTensor xG(x + (uint64_t)(start + (uint32_t)jstart) * dim + c0, {lanes});
            TLOAD(xin_h0, xG);
        } else {
            const uint32_t hi = (uint32_t)((int32_t)(K - 1) + jstart);
            GlobalIoTensor hG(convStates + ((uint64_t)cacheIdx * stateLen + hi) * dim + c0, {lanes});
            TLOAD(xin_h0, hG);
        }
        set_flag(PIPE_MTE2, PIPE_V, IEV[0]);
    }

    for (int32_t j = jstart; j < (int32_t)l1; ++j) {
        const uint32_t par = (uint32_t)(j - jstart) & 1u;
        IoTile xin_h(lanes);
        AccumTile xin_f(lanes);
        TASSIGN(xin_h, ubInputOffset[par]);
        TASSIGN(xin_f, ubInputFp32);

        // (1) consume current row (loaded by prologue / previous prefetch) in buffer par
        wait_flag(PIPE_MTE2, PIPE_V, IEV[par]);
        TCVT(xin_f, xin_h, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE2, IEV[par]);

        // (2) prefetch next row (x or conv_states history) into the OTHER buffer
        if (j + 1 < (int32_t)l1) {
            const int32_t e = j + 1;
            const uint32_t p1 = par ^ 1u;
            IoTile xin_hn(lanes);
            TASSIGN(xin_hn, ubInputOffset[p1]);
            wait_flag(PIPE_V, PIPE_MTE2, IEV[p1]);
            if (e >= 0) {
                GlobalIoTensor xG(x + (uint64_t)(start + (uint32_t)e) * dim + c0, {lanes});
                TLOAD(xin_hn, xG);
            } else {
                const uint32_t hi = (uint32_t)((int32_t)(K - 1) + e);
                GlobalIoTensor hG(convStates + ((uint64_t)cacheIdx * stateLen + hi) * dim + c0, {lanes});
                TLOAD(xin_hn, hG);
            }
            set_flag(PIPE_MTE2, PIPE_V, IEV[p1]);
        }

        pipe_barrier(PIPE_V);

        const bool startAll = zeroPad && (j == 0);
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t out = j + (int32_t)(K - 1) - (int32_t)k;
            if (out < (int32_t)l0 || out >= (int32_t)l1) continue;
            AccumTile wT(lanes);
            TASSIGN(wT, k * accumTileBytes);
            if (startAll || k == 0) {
                AccumTile acc(lanes);
                TASSIGN(acc, ubAccumRingBase + ((uint32_t)out & (RS - 1u)) * accumTileBytes);
                TMUL(acc, xin_f, wT);
            } else {
                AccumTile t(lanes);
                TASSIGN(t, ubProductBase + (k - 1u) * accumTileBytes);
                TMUL(t, xin_f, wT);
            }
        }
        pipe_barrier(PIPE_V);
        if (!startAll) {
            for (uint32_t k = 1; k < K; ++k) {
                const int32_t out = j + (int32_t)(K - 1) - (int32_t)k;
                if (out < (int32_t)l0 || out >= (int32_t)l1) continue;
                AccumTile acc(lanes);
                AccumTile t(lanes);
                TASSIGN(acc, ubAccumRingBase + ((uint32_t)out & (RS - 1u)) * accumTileBytes);
                TASSIGN(t, ubProductBase + (k - 1u) * accumTileBytes);
                TADD(acc, acc, t);
            }
        }
        pipe_barrier(PIPE_V);

        if (j < (int32_t)l0) continue;  // halo row

        const uint32_t slot = (uint32_t)j & (RS - 1u);
        const uint32_t ob = (uint32_t)j & 1u;
        const event_t oev = (event_t)(1u + ob);
        AccumTile acc(lanes);
        AccumTile tmp(lanes);
        IoTile outT(lanes);
        TASSIGN(acc, ubAccumRingBase + slot * accumTileBytes);
        TASSIGN(tmp, ubProductBase);
        TASSIGN(outT, ubOutputOffset[ob]);

        if (hasBias) {
            AccumTile bT(lanes);
            TASSIGN(bT, ubBiasOffset);
            TADD(acc, acc, bT);
            pipe_barrier(PIPE_V);
        }
        if (activation) {
            applySiluToTile(acc, acc, tmp);
            pipe_barrier(PIPE_V);
        }
        wait_flag(PIPE_MTE3, PIPE_V, oev);
        TCVT(outT, acc, pto::RoundMode::CAST_NONE);
        GlobalIoTensor yG(y + (uint64_t)(start + (uint32_t)j) * dim + c0, {lanes});
        set_flag(PIPE_V, PIPE_MTE3, oev);
        wait_flag(PIPE_V, PIPE_MTE3, oev);
        TSTORE(yG, outT);
        set_flag(PIPE_MTE3, PIPE_V, oev);
    }

    wait_flag(PIPE_V, PIPE_MTE2, IEV[0]);
    wait_flag(PIPE_V, PIPE_MTE2, IEV[1]);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
}

template <typename IoElemType, uint32_t RS, uint32_t MAX_W>
AICORE void runConv(__gm__ IoElemType *x, __gm__ IoElemType *wgt, __gm__ IoElemType *bia, __gm__ IoElemType *convStates,
                    __gm__ int32_t *qsl, __gm__ int32_t *cidx, __gm__ uint8_t *hinit, __gm__ IoElemType *y,
                    uint32_t dim, uint32_t batch, uint32_t inputMode, uint32_t seqLen, uint32_t stateLen, uint32_t K,
                    uint32_t col_w, uint32_t blocksPerSeq, uint32_t lchunks, uint32_t activation, uint32_t hasBias,
                    int32_t padSlot)
{
    set_mask_norm();
    set_vector_mask(-1, -1);
    const uint32_t num_cores = get_block_num();
    const uint32_t core_id = get_block_idx();
    const uint32_t gridSize = batch * blocksPerSeq * lchunks;

    for (uint32_t task = core_id; task < gridSize; task += num_cores) {
        const uint32_t lc = task % lchunks;
        const uint32_t t2 = task / lchunks;
        const uint32_t db = t2 % blocksPerSeq;
        const uint32_t seq = t2 / blocksPerSeq;

        uint32_t start, len;
        if (inputMode == 0u) {
            const int32_t s0 = qsl[seq];
            const int32_t s1 = qsl[seq + 1];
            start = (uint32_t)s0;
            len = (uint32_t)(s1 - s0);
        } else {
            start = seq * seqLen;
            len = seqLen;
        }
        if (len == 0u) continue;
        const int32_t ci = cidx[seq];
        if (ci == padSlot) continue;
        const bool hasInit = hinit[seq] != 0;

        const uint32_t lc_len = (len + lchunks - 1) / lchunks;
        const uint32_t l0 = lc * lc_len;
        if (l0 >= len) continue;
        uint32_t l1 = l0 + lc_len;
        if (l1 > len) l1 = len;

        const uint32_t c0 = db * col_w;
        const uint32_t rem = dim - c0;
        const int32_t lanes = rem > col_w ? (int32_t)col_w : (int32_t)rem;

        convChunk<IoElemType, RS, MAX_W>(x, y, wgt, bia, convStates, dim, stateLen, K, start, len, (uint32_t)ci,
                                         hasInit, c0, lanes, l0, l1, activation, hasBias);
    }
}

// Writeback: convStates[ci, 0:K-1, c0:] = last K-1 rows of xext (x tail / old hist).
template <typename IoElemType, uint32_t RS, uint32_t MAX_W>
AICORE void runWriteback(__gm__ IoElemType *x, __gm__ IoElemType *convStates, __gm__ int32_t *qsl, __gm__ int32_t *cidx,
                         __gm__ uint8_t *hinit, uint32_t dim, uint32_t batch, uint32_t inputMode, uint32_t seqLen,
                         uint32_t stateLen, uint32_t K, uint32_t col_w, uint32_t blocksPerSeq, int32_t padSlot)
{
    using GlobalShape = pto::Shape<1, 1, 1, 1, DYNAMIC>;
    using GlobalStride = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalIoTensor = pto::GlobalTensor<IoElemType, GlobalShape, GlobalStride>;
    using IoTile = Tile<TileType::Vec, IoElemType, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;
    using AccumTile = Tile<TileType::Vec, float, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;
    constexpr uint32_t ioTileBytes = MAX_W * sizeof(IoElemType);
    // compile-time scratch offset past the RS-1 reserved row slots (sized for the
    // worst case K==RS so it never depends on the runtime K).
    constexpr uint32_t SCRATCH_F32 = (RS - 1u) * ioTileBytes;  // fp32 scratch for zeroing

    set_mask_norm();
    set_vector_mask(-1, -1);
    const uint32_t num_cores = get_block_num();
    const uint32_t core_id = get_block_idx();
    const uint32_t gridSize = batch * blocksPerSeq;

    // This core runs several tasks (strided) reusing the same UB tiles, so each
    // task's reload (MTE2) must wait for the previous task's store (MTE3) to finish
    // reading them -- otherwise the store races the reload. Start with UB "free".
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (uint32_t task = core_id; task < gridSize; task += num_cores) {
        const uint32_t db = task % blocksPerSeq;
        const uint32_t seq = task / blocksPerSeq;
        uint32_t start, len;
        if (inputMode == 0u) {
            const int32_t s0 = qsl[seq];
            const int32_t s1 = qsl[seq + 1];
            start = (uint32_t)s0;
            len = (uint32_t)(s1 - s0);
        } else {
            start = seq * seqLen;
            len = seqLen;
        }
        if (len == 0u) continue;
        const int32_t ci = cidx[seq];
        if (ci == padSlot) continue;
        const bool hasInit = hinit[seq] != 0;
        const uint32_t c0 = db * col_w;
        const uint32_t rem = dim - c0;
        const int32_t lanes = rem > col_w ? (int32_t)col_w : (int32_t)rem;

        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);  // previous task's store done -> UB free to reload
        // Phase A (MTE2): load K-1 source rows = xext[len .. len+K-2].
        // xext index e: e>=K-1 -> x[e-(K-1)];  e<K-1 -> history (convStates if hasInit
        // else zero). For the zero case load x[start] (finite) as a placeholder and
        // zero it in phase B (avoids multiplying uninitialised UB).
        for (uint32_t i = 0; i < (K - 1); ++i) {
            const int32_t e = (int32_t)len + (int32_t)i;
            const int32_t xrow = e - (int32_t)(K - 1);
            IoTile row(lanes);
            TASSIGN(row, i * ioTileBytes);
            if (xrow >= 0) {
                GlobalIoTensor sG(x + (uint64_t)(start + (uint32_t)xrow) * dim + c0, {lanes});
                TLOAD(row, sG);
            } else if (hasInit) {
                GlobalIoTensor hG(convStates + ((uint64_t)ci * stateLen + (uint32_t)e) * dim + c0, {lanes});
                TLOAD(row, hG);
            } else {
                GlobalIoTensor sG(x + (uint64_t)start * dim + c0, {lanes});  // placeholder (len>=1)
                TLOAD(row, sG);
            }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // Phase B (V): zero the pure-history rows when there is no initial state.
        // (TMULS has no bf16 overload, so zero via an fp32 round-trip of the finite
        //  placeholder: IoElemType -> fp32 -> *0 -> IoElemType. Works for fp16 and bf16.)
        for (uint32_t i = 0; i < (K - 1); ++i) {
            const int32_t e = (int32_t)len + (int32_t)i;
            const int32_t xrow = e - (int32_t)(K - 1);
            if (xrow < 0 && !hasInit) {
                IoTile row(lanes);
                AccumTile f32(lanes);
                TASSIGN(row, i * ioTileBytes);
                TASSIGN(f32, SCRATCH_F32);
                TCVT(f32, row, pto::RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
                TMULS(f32, f32, 0.0f);
                pipe_barrier(PIPE_V);
                TCVT(row, f32, pto::RoundMode::CAST_NONE);
            }
        }
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // Phase C (MTE3): store to convStates[ci, 0:K-1, c0:].
        for (uint32_t i = 0; i < (K - 1); ++i) {
            IoTile row(lanes);
            TASSIGN(row, i * ioTileBytes);
            GlobalIoTensor dG(convStates + ((uint64_t)ci * stateLen + i) * dim + c0, {lanes});
            TSTORE(dG, row);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);  // store done -> UB free for the next strided task
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);  // drain the final store before the kernel exits
}

}  // namespace cc1d

// Vector-only kernel (the cube/AIC pass gets empty bodies). One conv + writeback
// entry per (RS, dtype): templated on the compile-time ring size RS and tile width
// MAX_W, with the width K passed at runtime; the host launches the rs<RS> variant.
#if defined(__DAV_VEC__)
#define CONV_BODY(T, RS, MW)                                                                                           \
    cc1d::runConv<T, RS, MW>((__gm__ T *)x, (__gm__ T *)wgt, (__gm__ T *)bia, (__gm__ T *)convStates,                  \
                             (__gm__ int32_t *)qsl, (__gm__ int32_t *)cidx, (__gm__ uint8_t *)hinit, (__gm__ T *)y,    \
                             dim, batch, inputMode, seqLen, stateLen, width, col_w, blocksPerSeq, lchunks, activation, \
                             hasBias, padSlot)
#define WB_BODY(T, RS, MW)                                                                                        \
    cc1d::runWriteback<T, RS, MW>((__gm__ T *)x, (__gm__ T *)convStates, (__gm__ int32_t *)qsl,                   \
                                  (__gm__ int32_t *)cidx, (__gm__ uint8_t *)hinit, dim, batch, inputMode, seqLen, \
                                  stateLen, width, col_w, blocksPerSeq, padSlot)
#else  // cube pass: empty bodies; void the params to silence unused warnings.
#define CONV_BODY(T, RS, MW)                                                                                      \
    (void)x, (void)wgt, (void)bia, (void)convStates, (void)qsl, (void)cidx, (void)hinit, (void)y, (void)dim,      \
        (void)batch, (void)inputMode, (void)seqLen, (void)stateLen, (void)width, (void)col_w, (void)blocksPerSeq, \
        (void)lchunks, (void)activation, (void)hasBias, (void)padSlot
#define WB_BODY(T, RS, MW)                                                                                  \
    (void)x, (void)convStates, (void)qsl, (void)cidx, (void)hinit, (void)dim, (void)batch, (void)inputMode, \
        (void)seqLen, (void)stateLen, (void)width, (void)col_w, (void)blocksPerSeq, (void)padSlot
#endif

#define CONV_PARAMS                                                                                               \
    GM_ADDR x, GM_ADDR wgt, GM_ADDR bia, GM_ADDR convStates, GM_ADDR qsl, GM_ADDR cidx, GM_ADDR hinit, GM_ADDR y, \
        uint32_t dim, uint32_t batch, uint32_t inputMode, uint32_t seqLen, uint32_t stateLen, uint32_t width,     \
        uint32_t col_w, uint32_t blocksPerSeq, uint32_t lchunks, uint32_t activation, uint32_t hasBias,           \
        int32_t padSlot
#define WB_PARAMS                                                                                                      \
    GM_ADDR x, GM_ADDR convStates, GM_ADDR qsl, GM_ADDR cidx, GM_ADDR hinit, uint32_t dim, uint32_t batch,             \
        uint32_t inputMode, uint32_t seqLen, uint32_t stateLen, uint32_t width, uint32_t col_w, uint32_t blocksPerSeq, \
        int32_t padSlot

#define DEF_CONV(SUF, T, RS, MW)                                       \
    extern "C" __global__ AICORE void causal_conv1d_##SUF(CONV_PARAMS) \
    {                                                                  \
        CONV_BODY(T, RS, MW);                                          \
    }
#define DEF_WB(SUF, T, RS, MW)                                          \
    extern "C" __global__ AICORE void causal_conv1d_wb_##SUF(WB_PARAMS) \
    {                                                                   \
        WB_BODY(T, RS, MW);                                             \
    }

// Ring sizes the kernel is compiled for -- must match FOR_EACH_RING_SIZE in
// op_host/causal_conv1d.cpp. Each row is (ringSize, maxTileWidth); a larger ring uses
// a smaller tile to fit the 192 KiB UB.
#define FOR_EACH_RING_SIZE(DO) DO(2, 3072) DO(4, 3072) DO(8, 1536) DO(16, 896) DO(32, 384) DO(64, 128)
#define DEFINE_ENTRIES(ringSize, maxTileWidth)                        \
    DEF_CONV(rs##ringSize##_half, half, ringSize, maxTileWidth)       \
    DEF_CONV(rs##ringSize##_bf16, bfloat16_t, ringSize, maxTileWidth) \
    DEF_WB(rs##ringSize##_half, half, ringSize, maxTileWidth)         \
    DEF_WB(rs##ringSize##_bf16, bfloat16_t, ringSize, maxTileWidth)
FOR_EACH_RING_SIZE(DEFINE_ENTRIES)
#undef DEFINE_ENTRIES
#undef FOR_EACH_RING_SIZE
