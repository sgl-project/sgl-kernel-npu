// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one
// launch
//
// Same pipeline as pto_mega_kernel, but scaled_dot_kkt / wy_fast / chunk_h /
// chunk_o use runtime H/Hg dispatch from dynamic_bsnd_groupvalue; cumsum still
// uses H (value heads) like dynamic_bsnd.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
// TODO(anastasios): this should be fixed for A5.
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
// Note the codegen parser does not support arguments of form "type *name", only "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format off
// GDN_MAX_HEADS: compile-time ceiling on the value-head count. num_heads is a
// RUNTIME argument (one .so serves every head count), so the transpose/cumsum
// UB tiles are sized for this worst case; any num_heads <= GDN_MAX_HEADS works
// with the unused columns zero-padded. Must match the host-side _MAX_HEADS
// guard and the GDN_MAX_HEADS in chunk_cumsum.cpp.
#ifndef GDN_MAX_HEADS
#define GDN_MAX_HEADS 64
#endif

#include <runtime/rt_ffts.h>

#include <pto/pto-inst.hpp>
#include <type_traits>

#include "acl/acl.h"
#include "mega_kernel_utils.h"

using namespace pto;
using namespace mega_kernel_utils;

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

template <typename T>
AICORE inline void mega_transpose_TH_to_HT(__gm__ T *src, __gm__ T *dst, int64_t T_len, int32_t H)
{
    // To avoid ambiguity with bisheng intrinsic header's global `enum class
    // Stride`
    using pto::Stride;

#if defined(__DAV_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128;
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t MinTransposeCols = 16;
    constexpr int32_t AlignElems = ((32 / ES) > MinTransposeCols) ? (32 / ES) : MinTransposeCols;
    // H is runtime; size the UB tiles for the worst-case head count. TTRANS
    // always processes the full BLOCK×HP tile (unused cols H..HP are
    // zero-padded), and the store loop below writes only the first H transposed
    // rows.
    constexpr int32_t HP = ((GDN_MAX_HEADS + AlignElems - 1) / AlignElems) * AlignElems;
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * HP * ES;
    constexpr int32_t TMP_UB = DST_UB + HP * BLOCK * ES;

    using UBSrcFull =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, HP, BLOCK, SLayout::NoneBox, 512>;
    using UBDstDyn = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBTmp = Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512>;

    using UBRow = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, 1, BLOCK, SLayout::NoneBox, 512>;
    using UBRowDyn = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;

    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS = Stride<1, 1, 1, DYNAMIC, 1>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;
    GmSrcS src_stride(H);  // runtime row pitch = H elements (skip other heads)

    UBSrcFull ub_src;
    TASSIGN(ub_src, SRC_UB);
    UBDst ub_dst;
    TASSIGN(ub_dst, DST_UB);
    UBTmp ub_tmp;
    TASSIGN(ub_tmp, TMP_UB);

    const int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;

    for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks; bi += static_cast<int64_t>(block_num)) {
        const int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

        if (valid <= 0){
            break;
        }

        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = H;
            GlobalTensor<T, Gm2D, GmSrcS> gm(src + t0 * H, gs, src_stride);
            UBSrcDyn ld(valid, H);
            TASSIGN(ld, SRC_UB);
            TLOAD(ld, gm);
            if (valid != BLOCK || H != HP) TFILLPAD_INPLACE(ub_src, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(ub_dst, ub_src, ub_tmp);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        for (int32_t h = 0; h < H; ++h) {
            Gm1D gs;
            gs.shape[4] = valid;
            GlobalTensor<T, Gm1D, GmS1> gm(dst + h * T_len + t0, gs);
            UBRowDyn st(1, valid);
            TASSIGN(st, DST_UB + h * BLOCK * ES);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        // WAR on ub_src: the flag above only protects DST_UB (store → next
        // TTRANS). Nothing orders MTE2 against V, so hold the next TLOAD until
        // TTRANS has finished reading SRC_UB — both this loop's next iteration
        // and the beta_t call that reuses these same UB offsets.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
#endif
}

template <int32_t H, int32_t C>
AICORE inline void mega_cast_fp32_to_fp16_bsnd(__gm__ float *src, __gm__ half *dst, uint32_t num_matrices,
                                        int64_t total_tokens)
{
    // See mega_transpose_TH_to_HT above — hides the global `enum class Stride`.
    using pto::Stride;

#if defined(__DAV_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));

    using SrcUB = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB =
        Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512>;
    using DynDstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub;
    TASSIGN(src_ub, F32_UB);
    DstUB dst_ub;
    TASSIGN(dst_ub, F16_UB);

    for (uint32_t m = cid; m < num_matrices; m += block_num) {
        uint32_t h = m % static_cast<uint32_t>(H);
        uint32_t chunk_idx = m / static_cast<uint32_t>(H);

        for (int64_t t = 0; t < total_tokens; ++t) {
            int64_t off = t * static_cast<int64_t>(H * C) + static_cast<int64_t>(h * C);

            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
                SrcUB ld;
                TASSIGN(ld, F32_UB);
                TLOAD(ld, gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
                DstUB st;
                TASSIGN(st, F16_UB);
                TSTORE(gm, st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Include original kernel implementations in separate namespaces.
// ===================================================================

#define call_kernel _mk_unused_gv_ck_cumsum
namespace mk_cumsum {
#include "chunk_cumsum.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_gv_ck_kkt
namespace mk_kkt {
#include "scaled_dot_kkt.cpp"
}
#undef call_kernel

namespace mk_solve {
#include "tri_inverse_impl.cpp"
}

#define call_kernel _mk_unused_gv_ck_wy
namespace mk_wy {
#include "wy_fast.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_gv_ck_h
namespace mk_h {
#include "chunk_h.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_gv_ck_o
namespace mk_o {
#include "chunk_o.cpp"
}
#undef call_kernel

AICORE inline void mega_solve_tril(__gm__ half *out, __gm__ half *in, __gm__ half *minus_id, uint32_t matrix_size,
                            uint32_t num_matrices, uint32_t num_bsnd_heads, __gm__ int32_t *cu_seqlens,
                            uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, half, GDN_C, 1, true>(out, in, minus_id, num_matrices, num_bsnd_heads,
                                                                       is_lower, cu_seqlens);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, half, GDN_C, 2, true>(out, in, minus_id, num_matrices, num_bsnd_heads,
                                                                       is_lower, cu_seqlens);
    else
        mk_solve::runKernelTriInvRecUnroll<half, half, GDN_C, 4, true>(out, in, minus_id, num_matrices, num_bsnd_heads,
                                                                       is_lower, cu_seqlens);
}

AICORE inline void mega_kernel_impl(
    GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr,
    GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr,
    GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr,
    GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR h0_ptr, int64_t has_initial_state,
    GM_ADDR kkt_ws_ptr, GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr, GM_ADDR o_ws_qk_ptr,
    GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr, int32_t H, uint32_t num_key_heads, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_matrices)
{

    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;

    if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
        return;
    }

    SyncAllMegaKernel<false>();

    mk_cumsum::cumsum_kernel<C>(reinterpret_cast<__gm__ float *>(g_in_ptr), reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, H);


    SyncAllMegaKernel<false>();

    mega_transpose_TH_to_HT<float>(reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                   reinterpret_cast<__gm__ float *>(g_t_ptr), total_tokens, H);
    mega_transpose_TH_to_HT<half>(reinterpret_cast<__gm__ half *>(beta_ptr),
                                  reinterpret_cast<__gm__ half *>(beta_t_ptr), total_tokens, H);


    SyncAllMegaKernel<false>();

    mk_kkt::kkt_kernel<D, C>(reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(beta_t_ptr),
                             reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_lower_ptr),
                             reinterpret_cast<__gm__ half *>(kkt_ws_ptr), reinterpret_cast<__gm__ half *>(A_ptr),
                             reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
                             static_cast<uint32_t>(H), num_key_heads);

// Drain the kkt handshake: Vec released both workspace slots (flags 2/3) one
// last time after Cube's final iteration, so consume them before the next
// stage. A2: Vec is a separate core → FFTS cross-core flag. A5: both Vec
// sub-blocks share Cube's core → each signals its own intra-block
//     flag (base, base + 16).
#if defined(__DAV_CUBE__)
    pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
    wait_flag_dev(2);
    wait_flag_dev(3);
#else
    WaitBothVecOnA5<PIPE_MTE2>(2);
    WaitBothVecOnA5<PIPE_MTE2>(3);
    pipe_barrier(PIPE_ALL);
#endif
#endif


    SyncAllMegaKernel<false>();

    mega_solve_tril(reinterpret_cast<__gm__ half *>(A_inv_ptr), reinterpret_cast<__gm__ half *>(A_ptr),
                    reinterpret_cast<__gm__ half *>(minus_id_ptr), C, num_matrices, H,
                    reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1);


    SyncAllMegaKernel<false>();

    mk_wy::wy_fast_kernel<D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(v_ptr),
        reinterpret_cast<__gm__ half *>(beta_t_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(A_inv_ptr), reinterpret_cast<__gm__ half *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_a2_ptr), reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len,
        total_tokens, static_cast<uint32_t>(H), num_key_heads);

// Drain the wy_fast handshake: Cube freed the A2/A1 slots (flags 3/4) one last
// time after Vec's final iteration.
// A5: Cube signalled both sub-blocks (base, base + 16); each sub-block waits on
//     its own flag, so a plain intra-block wait is what mirrors the signal.
#if defined(__DAV_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
        wait_flag_dev(3);
        wait_flag_dev(4);
#else
        wait_intra_block(PIPE_MTE3, 3);
        wait_intra_block(PIPE_MTE3, 4);
        pipe_barrier(PIPE_ALL);
#endif
    }
#endif


    SyncAllMegaKernel<false>();

    mk_h::chunk_h_kernel<D, C>(reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(w_ptr),
                               reinterpret_cast<__gm__ half *>(u_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
                               reinterpret_cast<__gm__ half *>(s_ptr), reinterpret_cast<__gm__ half *>(v_new_ptr),
                               reinterpret_cast<__gm__ half *>(fs_ptr), reinterpret_cast<__gm__ half *>(h0_ptr),
                               has_initial_state, 1, reinterpret_cast<__gm__ half *>(h_ws_ptr),
                               reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens,
                               static_cast<uint32_t>(H), num_key_heads);

    SyncAllMegaKernel<false>();

    mk_o::chunk_o_kernel<D, C>(
        reinterpret_cast<__gm__ half *>(q_ptr), reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(v_new_ptr), reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_qk_ptr), reinterpret_cast<__gm__ half *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_gated_ptr), reinterpret_cast<__gm__ half *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens, static_cast<uint32_t>(H),
        num_key_heads);

// Drain the chunk_o handshake: Vec's final "workspace free" (flag 3) is never
// consumed by Cube's loop, so consume it here.
#if defined(__DAV_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
        wait_flag_dev(3);
#else
        WaitBothVecOnA5<PIPE_MTE2>(3);
        pipe_barrier(PIPE_ALL);
#endif
    }
#endif
}

// Note the codegen parser does not support arguments of form "type *name", only "type* name"
extern "C" __global__ AICORE void launch_mega_kernel(
    GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr,
    GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr,
    GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr,
    GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR h0_ptr, int64_t has_initial_state,
    GM_ADDR kkt_ws_ptr, GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr, GM_ADDR o_ws_qk_ptr,
    GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr, uint32_t num_heads, uint32_t num_key_heads, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_matrices)
{

        mega_kernel_impl(q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr,
                            cu_seqlens_ptr, o_ptr, g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr,
                            w_ptr, u_ptr, s_ptr, v_new_ptr, fs_ptr, h0_ptr, has_initial_state, kkt_ws_ptr,
                            wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr,
                            num_heads, num_key_heads, batch_size, seq_len, total_tokens, num_matrices);
    }
