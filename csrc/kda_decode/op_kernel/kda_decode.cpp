// Vector-only recurrent KDA (Kimi Delta Attention) decode kernel (PTO-ISA).
//
// Upstream source of truth: https://github.com/huawei-csl/megagdn-pto/pull/34.
// This copy differs only in its launch boundary: GM_ADDR arguments for the
// ACLRT codegen, no ffts plumbing (the ACL launch path sets the FFTS base
// itself and this kernel is vector-only, so it never issues a cross-core sync),
// and no out-of-place `state_out` (sglang always updates the gathered pool in
// place).  Keep the recurrence body in sync with upstream.
//
// State is fp32 [slot, head, V, K] -- V-major with row stride K -- which is
// exactly sglang's `temporal_state` pool layout (see mem_cache/memory_pool.py
// and the (V, K)/(K, 1) block pointer in the prefill chunk_delta_h kernel).
//
// Token addressing follows the fused sglang reference: without `cu_seqlens`
// each batch entry owns a contiguous run of `seq_len` tokens, with it every
// sequence's tokens are flattened onto one B=1 axis and sequence `n` spans
// [cu_seqlens[n], cu_seqlens[n + 1]).  `l2norm` enables the reference's
// in-kernel normalization q /= sqrt(sum(q*q)) + 1e-6 (likewise for k).
//
// `state_indices[n] < 0` marks a padded batch lane: no state is read or written
// and the item's output rows are zeroed, which is what makes the kernel safe
// under NPU graph replay at a fixed captured bucket size.
//
// The gating is fused: the kernel takes the raw `A_log`, `a`, `dt_bias` and `b`
// and computes g = -exp(A_log) * softplus(a + dt_bias) / softplus_beta and
// beta = sigmoid(b) itself, in fp32.  q/k/v/out are bf16 -- the model's own
// dtype -- so the launcher is pure marshalling with no elementwise torch work
// in front of the launch, and `g` no longer round-trips through fp16.

#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef KDA_V
#define KDA_V GDN_D
#endif
#ifndef KDA_BV
#define KDA_BV 32
#endif
#ifndef KDA_KERNEL_NAME
#define KDA_KERNEL_NAME launch_kda_decode
#endif
#ifndef AICORE
#define AICORE [aicore]
#endif
// Note the codegen parser does not support arguments of form "type *name", only "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format on

#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C, pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, RV, CV, pto::SLayout::NoneBox, 512, P>;
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor, RV, CV, pto::SLayout::NoneBox, 512>;
#endif

template <int KDim, int VDim, int VTile>
AICORE void kda_decode_kernel(__gm__ bfloat16_t *q_ptr, __gm__ bfloat16_t *k_ptr, __gm__ bfloat16_t *v_ptr,
                              __gm__ float *A_log_ptr, __gm__ float *a_ptr, __gm__ float *dt_bias_ptr,
                              __gm__ float *b_ptr, __gm__ float *state_ptr, __gm__ bfloat16_t *out_ptr,
                              __gm__ int32_t *state_indices, __gm__ int32_t *cu_seqlens, int64_t batch_size,
                              int64_t seq_len, int32_t num_heads, int32_t num_state_slots, float scale, int32_t l2norm,
                              float softplus_beta)
{
    const int32_t cid = get_block_idx();
    const int32_t block_num = get_block_num();

#if defined(__DAV_C220_VEC__)
    static_assert(KDim % 8 == 0, "KDim must be a multiple of 8");
    static_assert(VTile % 8 == 0, "VTile must be a multiple of 8");
    set_mask_norm();
    set_vector_mask(-1, -1);

    constexpr int NumVTiles = (VDim + VTile - 1) / VTile;
    // This build is launched vector-only through ACLRT, where block_dim counts AIV
    // cores directly (the host sizes it from GetCoreNumAiv) and get_subblockid()
    // is always 0 -- there is no mix-mode 1:2 sub-block split here.  Upstream
    // megagdn launches the same body as a mix kernel and uses
    // `worker = cid * 2 + get_subblockid()`; using that mapping here silently
    // drops every odd work item, because `workers` would double while only
    // `block_num` blocks actually run.
    const int worker = cid;
    const int workers = block_num;
    const int64_t total = batch_size * static_cast<int64_t>(num_heads) * NumVTiles;
    const bool use_l2norm = l2norm != 0;

    // TROWSUM's scratch is a half-width binary-tree buffer, not a second copy of
    // the source tile.  For fp32 one repeat spans REPEAT_BYTE/4 == 64 columns and
    // TRowSumOp::FillTmp writes only floor(srcRptPerRow / 2) repeats per row,
    // where srcRptPerRow == KDim / 64; TmpProc then halves in place and never
    // reaches past that.  At KDim == 128 that is a single repeat, so 64 columns
    // is exact and a full [VTile, KDim] tile wastes half of UB -- which is
    // precisely what pushes VTile == 128 past the 184 KiB usable window.
    constexpr int RptCols = 64;  // REPEAT_BYTE(256) / sizeof(float)
    constexpr int FillRpts = (KDim / RptCols) / 2;
    constexpr int TmpCols = FillRpts > 1 ? FillRpts * RptCols : RptCols;

    // [state, work, reduction scratch, q, k, decay, bf16 staging, v, rows, output,
    //  l2norm scratch, gating scratch]
    constexpr int StateAddr = 0;
    constexpr int WorkAddr = StateAddr + VTile * KDim * 4;
    constexpr int TmpAddr = WorkAddr + VTile * KDim * 4;
    constexpr int QAddr = TmpAddr + VTile * TmpCols * 4;
    constexpr int KAddr = QAddr + KDim * 4;
    constexpr int GAddr = KAddr + KDim * 4;
    constexpr int QBfAddr = GAddr + KDim * 4;
    constexpr int KBfAddr = QBfAddr + KDim * 2;
    constexpr int VBfAddr = KBfAddr + KDim * 2;
    constexpr int VAddr = VBfAddr + VTile * 2;
    constexpr int RowAddr = VAddr + VTile * 4;
    // Two out staging halves.  MTE3 is still draining token t-1's store while the
    // vector pipe casts token t, so a single buffer would need a blocking
    // MTE3->V fence on every token purely to protect the reuse.
    constexpr int OutBfAddr = RowAddr + VTile * 4;
    constexpr int OutBfAddr1 = OutBfAddr + VTile * 2;
    // q and k are adjacent fp32 rows, so one [2, KDim] view normalizes both with
    // a single instruction chain instead of two.
    constexpr int SqAddr = OutBfAddr1 + VTile * 2;
    constexpr int NrmTmpAddr = SqAddr + 2 * KDim * 4;
    constexpr int NrmAddr = NrmTmpAddr + 2 * KDim * 4;
    // Fused gating.  `dt_bias` and `coef` depend only on the head, so they are
    // loaded once per work item; `gate_a`/`gate_r`/`gate_t` carry the per-token
    // softplus chain and `sig` the sigmoid denominator for beta.
    constexpr int ARowAddr = NrmAddr + 32;
    constexpr int DtbAddr = ARowAddr + KDim * 4;
    constexpr int GateRAddr = DtbAddr + KDim * 4;
    constexpr int GateTAddr = GateRAddr + KDim * 4;
    constexpr int CoefAddr = GateTAddr + KDim * 4;
    constexpr int SigAddr = CoefAddr + KDim * 4;
    constexpr int ZeroBfAddr = SigAddr + VTile * 4;
    static_assert(KAddr == QAddr + KDim * 4, "q/k must be adjacent for the [2, KDim] l2norm view");
    static_assert(ZeroBfAddr + VTile * 2 <= 184 * 1024, "UB overflow (TMP_UB_OFFSET at 184 KiB)");

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using KStride = Stride<1, 1, 1, KDim, 1>;
    using VStride = Stride<1, 1, 1, VDim, 1>;
    using BfK = GlobalTensor<bfloat16_t, DynShape, KStride>;
    using BfV = GlobalTensor<bfloat16_t, DynShape, VStride>;
    using F32K = GlobalTensor<float, DynShape, KStride>;
    // `a` is [tokens, num_heads * KDim] and dt_bias [num_heads * KDim], both fp32
    // and contiguous, so a head's slice is a plain [1, KDim] row like q/k.
    using F32Row = GlobalTensor<float, DynShape, KStride>;

    // True while an item's state TSTORE is still in flight on EVENT_ID4.
    bool state_pending = false;

    for (int64_t work_id = worker; work_id < total; work_id += workers) {
        const int vt = static_cast<int>(work_id % NumVTiles);
        const int64_t bh = work_id / NumVTiles;
        const int head = static_cast<int>(bh % num_heads);
        const int64_t batch = bh / num_heads;
        const int v0 = vt * VTile;
        const int rows = (v0 + VTile <= VDim) ? VTile : (VDim - v0);

        int64_t bos, tokens;
        if (cu_seqlens == nullptr) {
            bos = batch * seq_len;
            tokens = seq_len;
        } else {
            bos = cu_seqlens[batch];
            tokens = static_cast<int64_t>(cu_seqlens[batch + 1]) - bos;
        }
        if (tokens <= 0) continue;

        DynShape vs;
        vs.shape[3] = 1;
        vs.shape[4] = rows;

        const int slot = state_indices == nullptr ? static_cast<int>(batch) : state_indices[batch];
        if (slot < 0 || slot >= num_state_slots) {
            // Padded lane.  Its output rows are never otherwise written, so zero
            // them here -- that is what lets the launcher hand us an uninitialised
            // `out` instead of paying for a torch.zeros_like pass.
            UbND<float, 1, VTile, DYNAMIC, DYNAMIC> zero_f(1, rows);
            UbND<bfloat16_t, 1, VTile, DYNAMIC, DYNAMIC> zero_bf(1, rows);
            TASSIGN(zero_f, SigAddr);
            TASSIGN(zero_bf, ZeroBfAddr);
            TEXPANDS(zero_f, 0.0f);
            pipe_barrier(PIPE_V);
            TCVT(zero_bf, zero_f, pto::RoundMode::CAST_RINT);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            for (int64_t t = 0; t < tokens; ++t) {
                BfV zero_gm(out_ptr + ((bos + t) * num_heads + head) * VDim + v0, vs);
                TSTORE(zero_gm, zero_bf);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
            continue;
        }

        const int64_t state_off = ((static_cast<int64_t>(slot) * num_heads + head) * VDim + v0) * KDim;
        DynShape state_shape;
        state_shape.shape[3] = rows;
        state_shape.shape[4] = KDim;
        F32K state_gm(state_ptr + state_off, state_shape);
        UbND<float, VTile, KDim, DYNAMIC, DYNAMIC> state(rows, KDim);
        TASSIGN(state, StateAddr);
        // The previous item's state store only *reads* this UB tile, so nothing but
        // the reload has to wait for it.  Fencing MTE3->MTE2 here instead of
        // draining MTE3->S at the end of the item lets that 32 KiB store overlap
        // the rest of the work loop.
        if (state_pending) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
            state_pending = false;
        }
        TLOAD(state, state_gm);

        // Per-head gating constants.  dt_bias is indexed [head, KDim] and A_log is
        // one float per head, so both are loop-invariant across this item's tokens.
        DynShape ks;
        ks.shape[3] = 1;
        ks.shape[4] = KDim;
        UbND<float, 1, KDim> dtb, coef;
        TASSIGN(dtb, DtbAddr);
        TASSIGN(coef, CoefAddr);
        F32Row dtb_gm(dt_bias_ptr + static_cast<int64_t>(head) * KDim, ks);
        TLOAD(dtb, dtb_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // g = -exp(A_log) * softplus(a + dt_bias) / softplus_beta.  Everything
        // except the softplus is per-head, so fold it into one broadcast tile and
        // spend a single TMUL per token instead of an exp and two scalar multiplies.
        TEXPANDS(coef, static_cast<float>(A_log_ptr[head]));
        pipe_barrier(PIPE_V);
        TEXP(coef, coef);
        pipe_barrier(PIPE_V);
        TMULS(coef, coef, -1.0f / softplus_beta);
        pipe_barrier(PIPE_V);

        for (int64_t t = 0; t < tokens; ++t) {
            const int64_t token_head = (bos + t) * num_heads + head;
            const int64_t k_off = token_head * KDim;
            const int64_t v_off = token_head * VDim + v0;
            BfK q_gm(q_ptr + k_off, ks), k_gm(k_ptr + k_off, ks);
            UbND<bfloat16_t, 1, KDim> q_bf, k_bf;
            UbND<float, 1, KDim> gate_a;
            TASSIGN(q_bf, QBfAddr);
            TASSIGN(k_bf, KBfAddr);
            TASSIGN(gate_a, ARowAddr);
            TLOAD(q_bf, q_gm);
            TLOAD(k_bf, k_gm);
            // `a` shares q/k's [token, head, KDim] addressing, just in fp32.
            F32Row a_gm(a_ptr + k_off, ks);
            TLOAD(gate_a, a_gm);
            BfV v_gm(v_ptr + v_off, vs);
            UbND<bfloat16_t, 1, VTile, DYNAMIC, DYNAMIC> v_bf(1, rows);
            TASSIGN(v_bf, VBfAddr);
            TLOAD(v_bf, v_gm);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // v's cast is independent of q/k, so it rides along under the same
            // barrier instead of paying its own.
            UbND<float, 1, KDim> q, k, decay, gate_r, gate_t;
            TASSIGN(q, QAddr);
            TASSIGN(k, KAddr);
            TASSIGN(decay, GAddr);
            TASSIGN(gate_r, GateRAddr);
            TASSIGN(gate_t, GateTAddr);
            UbND<float, 1, VTile, DYNAMIC, DYNAMIC> v_row(1, rows);
            TASSIGN(v_row, VAddr);
            TCVT(q, q_bf, pto::RoundMode::CAST_NONE);
            TCVT(k, k_bf, pto::RoundMode::CAST_NONE);
            TCVT(v_row, v_bf, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            if (use_l2norm) {
                UbND<float, 2, KDim> qk, sq, nrm_tmp;
                TASSIGN(qk, QAddr);
                TASSIGN(sq, SqAddr);
                TASSIGN(nrm_tmp, NrmTmpAddr);
                // ColMajor tiles need a 32-byte-aligned row count, so allocate 8 rows
                // and use the two that hold the q and k norms.  Unary math needs the
                // row-major view of those same two floats.
                UbDN<float, 8, 1, 2, 1> nrm;
                TASSIGN(nrm, NrmAddr);
                UbND<float, 1, 8, 1, 2> nrm_flat;
                TRESHAPE(nrm_flat, nrm);
                TMUL(sq, qk, qk);
                pipe_barrier(PIPE_V);
                TROWSUM(nrm, sq, nrm_tmp);
                pipe_barrier(PIPE_V);
                TSQRT(nrm_flat, nrm_flat);
                pipe_barrier(PIPE_V);
                TADDS(nrm_flat, nrm_flat, 1e-6f);
                pipe_barrier(PIPE_V);
                TROWEXPANDDIV(qk, qk, nrm);
                pipe_barrier(PIPE_V);
            }
            TMULS(q, q, scale);

            // softplus(x) computed as relu(x) + log1p(exp(-|x|)), which is the
            // branch-free form of torch's `beta*x > threshold ? x : log1p(exp(beta*x))/beta`:
            // the exponent is never positive so it cannot overflow, and the two agree
            // to within exp(-threshold), far below fp32's precision at that magnitude.
            // `coef` already carries -exp(A_log)/softplus_beta, so one TMUL finishes g.
            TADD(gate_a, gate_a, dtb);
            pipe_barrier(PIPE_V);
            if (softplus_beta != 1.0f) {
                TMULS(gate_a, gate_a, softplus_beta);
                pipe_barrier(PIPE_V);
            }
            TRELU(gate_r, gate_a);
            TABS(gate_t, gate_a);
            pipe_barrier(PIPE_V);
            TMULS(gate_t, gate_t, -1.0f);
            pipe_barrier(PIPE_V);
            TEXP(gate_t, gate_t);
            pipe_barrier(PIPE_V);
            TADDS(gate_t, gate_t, 1.0f);
            pipe_barrier(PIPE_V);
            TLOG(gate_t, gate_t);
            pipe_barrier(PIPE_V);
            TADD(gate_r, gate_r, gate_t);
            pipe_barrier(PIPE_V);
            TMUL(decay, gate_r, coef);
            pipe_barrier(PIPE_V);
            TEXP(decay, decay);

            // beta = sigmoid(b) folds into the delta as a divide by 1 + exp(-b).
            // Staged here, ahead of the state chain, so only the TDIV sits on the
            // critical path; `b` itself is a scalar GM read, like the old beta was.
            UbND<float, 1, VTile, DYNAMIC, DYNAMIC> sig(1, rows);
            TASSIGN(sig, SigAddr);
            TEXPANDS(sig, -static_cast<float>(b_ptr[token_head]));
            pipe_barrier(PIPE_V);
            TEXP(sig, sig);
            pipe_barrier(PIPE_V);
            TADDS(sig, sig, 1.0f);
            pipe_barrier(PIPE_V);

            UbDN<float, VTile, 1, DYNAMIC, DYNAMIC> delta(rows, 1);
            TRESHAPE(delta, v_row);
            UbND<float, 1, VTile, DYNAMIC, DYNAMIC> delta_flat(1, rows);
            UbND<float, 1, VTile, DYNAMIC, DYNAMIC> row_flat(1, rows);
            TRESHAPE(delta_flat, delta);
            UbND<float, VTile, KDim, DYNAMIC, DYNAMIC> work(rows, KDim);
            UbND<float, VTile, TmpCols, DYNAMIC, DYNAMIC> tmp(rows, TmpCols);
            TASSIGN(work, WorkAddr);
            TASSIGN(tmp, TmpAddr);
            UbDN<float, VTile, 1, DYNAMIC, DYNAMIC> row(rows, 1);
            TASSIGN(row, RowAddr);
            TRESHAPE(row_flat, row);

            TCOLEXPANDMUL(state, state, decay);
            pipe_barrier(PIPE_V);
            TCOLEXPANDMUL(work, state, k);
            pipe_barrier(PIPE_V);
            TROWSUM(row, work, tmp);
            pipe_barrier(PIPE_V);
            TSUB(delta_flat, delta_flat, row_flat);
            pipe_barrier(PIPE_V);
            TDIV(delta_flat, delta_flat, sig);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(work, k);
            pipe_barrier(PIPE_V);
            TROWEXPANDMUL(work, work, delta);
            pipe_barrier(PIPE_V);
            TADD(state, state, work);
            pipe_barrier(PIPE_V);
            TCOLEXPANDMUL(work, state, q);
            pipe_barrier(PIPE_V);
            TROWSUM(row, work, tmp);
            pipe_barrier(PIPE_V);

            // Alternate the staging half so the only thing that must wait for a store
            // is the token that reuses that half, two tokens later.  The cast itself
            // needs no barrier ahead of the store: set_flag(V, MTE3) already orders
            // all prior vector work against MTE3.
            UbND<bfloat16_t, 1, VTile, DYNAMIC, DYNAMIC> out_flat(1, rows);
            if ((t & 1) == 0) {
                TASSIGN(out_flat, OutBfAddr);
                if (t >= 2) wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            } else {
                TASSIGN(out_flat, OutBfAddr1);
                if (t >= 2) wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
            }
            // Named explicitly rather than left to the default: RINT is
            // round-to-nearest-even, which is what torch's .to(bfloat16) does.
            // Only the convert is bf16 -- vconv_f322bf16r, the counterpart of the
            // vconv_bf162f32 on the way in.  All the math above is fp32.
            TCVT(out_flat, row_flat, pto::RoundMode::CAST_RINT);
            UbND<bfloat16_t, 1, VTile, DYNAMIC, DYNAMIC> out_row(1, rows);
            TRESHAPE(out_row, out_flat);
            BfV out_gm(out_ptr + v_off, vs);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(out_gm, out_row);
            if ((t & 1) == 0)
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            else
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        }
        // The last two tokens set a flag nobody waited on; consume them so the
        // event state is clean for the next work item.
        if (tokens >= 2) {
            if (((tokens - 2) & 1) == 0)
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            else
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        }
        if (((tokens - 1) & 1) == 0)
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        else
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(state_gm, state);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        state_pending = true;
    }
    if (state_pending) wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
#endif
}

// Note the codegen parser does not support arguments of form "type *name", only "type* name"
extern "C" __global__ AICORE void KDA_KERNEL_NAME(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR A_log_ptr,
                                                  GM_ADDR a_ptr, GM_ADDR dt_bias_ptr, GM_ADDR b_ptr, GM_ADDR state_ptr,
                                                  GM_ADDR out_ptr, GM_ADDR state_indices_ptr, GM_ADDR cu_seqlens_ptr,
                                                  int64_t num_sequences, int64_t seq_len, int32_t num_heads,
                                                  int32_t num_state_slots, float scale, int32_t l2norm,
                                                  float softplus_beta)
{
    kda_decode_kernel<GDN_D, KDA_V, KDA_BV>(
        reinterpret_cast<__gm__ bfloat16_t *>(q_ptr), reinterpret_cast<__gm__ bfloat16_t *>(k_ptr),
        reinterpret_cast<__gm__ bfloat16_t *>(v_ptr), reinterpret_cast<__gm__ float *>(A_log_ptr),
        reinterpret_cast<__gm__ float *>(a_ptr), reinterpret_cast<__gm__ float *>(dt_bias_ptr),
        reinterpret_cast<__gm__ float *>(b_ptr), reinterpret_cast<__gm__ float *>(state_ptr),
        reinterpret_cast<__gm__ bfloat16_t *>(out_ptr), reinterpret_cast<__gm__ int32_t *>(state_indices_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), num_sequences, seq_len, num_heads, num_state_slots, scale,
        l2norm, softplus_beta);
}
