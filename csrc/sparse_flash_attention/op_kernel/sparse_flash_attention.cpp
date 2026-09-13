/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention.cpp
 * \brief Kernel entry for sparse_flash_attention (arch22 / A2+A3).
 *
 * Ported from vllm-ascend csrc/attention/sparse_flash_attention/op_kernel/sparse_flash_attention.cpp.
 * The kernel bodies in arch22/ are upstream's, untouched. What changed is only how an instantiation
 * of them gets selected:
 *
 * Upstream's entry is itself a template, parameterised on
 *     <FLASH_DECODE, PAGE_ATTENTION, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G>
 * and it branches on ORIG_DTYPE_QUERY. Both mechanisms belong to the CANN op packer: it compiles
 * one binary per legal template-argument combination (the ASCENDC_TPL_SEL list in
 * sparse_flash_attention_template_tiling_key.h) and per dtype, injecting ORIG_DTYPE_* as macros, and
 * the tiling key picks the binary.
 *
 * ascendc_library() here builds one binary reached through aclrtlaunch_sparse_flash_attention, so
 * neither is available: the entry has to be a single non-template extern "C" symbol, and the dtype
 * macros do not exist. The selection therefore happens at run time, switching on the dispatch key the
 * host packed into the tiling data. csrc/sparse_attn_sharedkv does the same thing for the same reason.
 *
 * The consequence is that every reachable instantiation is compiled into this one object, so the
 * switch below is the upstream ASCENDC_TPL_SEL list written out. For arch22 that is 16 cases, because
 * SFAType only actually varies on four axes:
 *
 *   dtype        bf16 | fp16          (upstream: separate binaries, ORIG_DTYPE_QUERY)
 *   qLayout      BSND | TND
 *   kvLayout     BSND | TND | PA_BSND (paired with qLayout, so 2 per qLayout, not 3)
 *   templateMode C_TEMPLATE | V_TEMPLATE
 *
 * The other two axes upstream carries are not parameters of the arch22 kernel at all:
 *   - PAGE_ATTENTION is derived inside SFAType as (KV_LAYOUT_T == PA_BSND), so it is redundant.
 *   - IS_SPLIT_G is not passed to SFAType by upstream's own arch22 SFA_OP_IMPL -- only arch35 uses it.
 * Both are still encoded in the dispatch key (the host sets them exactly as upstream's GenTilingKey
 * did) and simply masked off here, so adding arch35 later does not need a key change.
 *
 * Note this file defines exactly ONE kernel function, deliberately. Other ops in this repo
 * (apply_token_bitmask, tri_inv, causal_conv1d_update) instead define one kernel entry per dtype in
 * a single file, which draws "Multiple kernel functions are detected. It is recommended to define
 * only one kernel function per file" from the AscendC frontend -- it emits one aclrtlaunch_<name>
 * stub per kernel function. Folding the dtype into the dispatch key keeps a single entry, so there is
 * one stub, aclrtlaunch_sparse_flash_attention, and no such warning from here.
 *
 * If build time or code size ever becomes a problem, this switch is the place to trim: SGLang's DCP
 * decode path only reaches bf16 x TND x PA_BSND x V_TEMPLATE. Trimming is a behaviour change though --
 * a removed case turns into the "unsupported" fallthrough at run time -- so it should be a deliberate
 * decision, not silent.
 */

#include "kernel_operator.h"

/*
 * arch22/sparse_flash_attention_service_cube_mla.h uses EVENT_ID4..EVENT_ID7 at class scope.
 *
 * ascendc_library compiles each kernel source more than once -- once per core type (aic_obj, aiv_obj)
 * and again for the host launch stub (host_bisheng_obj). The AscendC event ids are declared for the
 * device passes but not for the host one, so the vendored kernel fails there with "use of undeclared
 * identifier 'EVENT_ID4'" even though the device passes are happy. Upstream never meets this because
 * the op packer builds its kernels through a different path.
 *
 * csrc/compressor/op_kernel/compressor.cpp carries this same block for the same reason; this follows
 * it. The values are the event ids themselves, and the #ifndef guards mean the real declarations win
 * wherever they exist -- so the device passes are untouched. These must precede the kernel headers
 * below, which consume them.
 */
#ifndef EVENT_ID0
#define EVENT_ID0 0
#endif
#ifndef EVENT_ID1
#define EVENT_ID1 1
#endif
#ifndef EVENT_ID2
#define EVENT_ID2 2
#endif
#ifndef EVENT_ID3
#define EVENT_ID3 3
#endif
#ifndef EVENT_ID4
#define EVENT_ID4 4
#endif
#ifndef EVENT_ID5
#define EVENT_ID5 5
#endif
#ifndef EVENT_ID6
#define EVENT_ID6 6
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 7
#endif

#include "sparse_flash_attention_template_tiling_key.h"
// From op_host/, which the kernel target has on its include path (see csrc/CMakeLists.txt). Supplies
// the tiling struct and the dispatch-key layout, so host and kernel cannot disagree about either.
#include "sparse_flash_attention_tiling_data.h"

// The arch22 kernel headers name the tiling struct unqualified, as upstream's packer-generated header
// declared it. Must precede the include below.
using optiling::SparseFlashAttentionTilingDataMla;

#include "arch22/sparse_flash_attention_kernel_mla.h"

using namespace AscendC;

// Only the axes the arch22 SFAType actually varies on; see the header comment.
#define SFA_ARCH22_KEY(dtype, qLayout, kvLayout, templateMode)             \
    ((static_cast<uint64_t>(qLayout) << SGL_SFA_KEY_QLAYOUT_SHIFT) |       \
     (static_cast<uint64_t>(kvLayout) << SGL_SFA_KEY_KVLAYOUT_SHIFT) |     \
     (static_cast<uint64_t>(templateMode) << SGL_SFA_KEY_TEMPLATE_SHIFT) | \
     (static_cast<uint64_t>(dtype) << SGL_SFA_KEY_DTYPE_SHIFT))

#define SFA_ARCH22_MASK SFA_ARCH22_KEY(0xFU, 0xFU, 0xFU, 0xFU)

/*
 * Mirrors upstream's arch22 SFA_OP_IMPL. Upstream passes the same dtype three times (query, kv,
 * output), FLASH_DECODE as false, and does not pass IS_SPLIT_G.
 */
#define SFA_LAUNCH(DTYPE, Q_LAYOUT, KV_LAYOUT, TEMPLATE_MODE)                                                       \
    do {                                                                                                            \
        SparseFlashAttentionMla<SFAType<DTYPE, DTYPE, DTYPE, false, Q_LAYOUT, KV_LAYOUT, TEMPLATE_MODE>> op;        \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV, blocktable, queryRope, \
                keyRope, attentionOut, softmaxMax, softmaxSum, user, tilingData, tiling, &tPipe);                   \
        op.Process();                                                                                               \
    } while (0)

// One block per (qLayout, kvLayout) pairing, for a given dtype. The pairings are upstream's: a BSND
// query goes with BSND or PA_BSND kv, a TND query with TND or PA_BSND kv.
#define SFA_LAUNCH_ALL_LAYOUTS(KEY_DTYPE, DTYPE)                                     \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_BSND, SFA_LAYOUT_BSND, C_TEMPLATE):    \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::BSND, SFA_LAYOUT::BSND, C_TEMPLATE);           \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_BSND, SFA_LAYOUT_BSND, V_TEMPLATE):    \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::BSND, SFA_LAYOUT::BSND, V_TEMPLATE);           \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_BSND, SFA_LAYOUT_PA_BSND, C_TEMPLATE): \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::BSND, SFA_LAYOUT::PA_BSND, C_TEMPLATE);        \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_BSND, SFA_LAYOUT_PA_BSND, V_TEMPLATE): \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::BSND, SFA_LAYOUT::PA_BSND, V_TEMPLATE);        \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_TND, SFA_LAYOUT_TND, C_TEMPLATE):      \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::TND, SFA_LAYOUT::TND, C_TEMPLATE);             \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_TND, SFA_LAYOUT_TND, V_TEMPLATE):      \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::TND, SFA_LAYOUT::TND, V_TEMPLATE);             \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND, C_TEMPLATE):  \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, C_TEMPLATE);         \
        break;                                                                       \
    case SFA_ARCH22_KEY(KEY_DTYPE, SFA_LAYOUT_TND, SFA_LAYOUT_PA_BSND, V_TEMPLATE):  \
        SFA_LAUNCH(DTYPE, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, V_TEMPLATE);         \
        break;

/*
 * The parameters are GM_ADDR, not upstream's `__gm__ uint8_t *`, and that is not cosmetic.
 *
 * ascendc_library() generates the launch plumbing (auto_gen_sparse_flash_attention.cpp and
 * aclrtlaunch_triple_chevrons_func.h) by parsing this signature textually. Its parser splits each
 * parameter on whitespace, so `__gm__ uint8_t *query` is read as type `__gm__ uint8_t` with the name
 * `*query`, and the generated wrapper then calls the kernel as `sparse_flash_attention_origin(*query,
 * ...)` while declaring the stub as `void *query` -- which fails to compile on both sides.
 *
 * GM_ADDR is the same type, spelled as a single token the parser handles. Every kernel in this repo
 * built by ascendc_library uses it; the only sources using `__gm__ uint8_t *` are under
 * csrc/attentions/, which the CANN op packer builds through a different codegen path (as does
 * upstream). Do not "restore" upstream's spelling here.
 */
extern "C" __global__ __aicore__ void sparse_flash_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                             GM_ADDR sparseIndices, GM_ADDR blocktable,
                                                             GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKV,
                                                             GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR attentionOut,
                                                             GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR workspace,
                                                             GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    /*
     * Upstream copies the tiling into a local struct with
     *     GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingDataMla, tiling_data_in, tiling);
     * That macro is part of the CANN op-packer's tiling plumbing and does not accept a plain POD
     * struct type here -- it rejects the type name as "does not refer to a value" and never declares
     * the variable. csrc/sparse_attn_sharedkv hit the same wall and resolved it the same way: view the
     * tiling buffer in place as a __gm__ pointer and let the kernel read its scalars from GM, which is
     * why the Init() signatures in arch22/ take a __gm__-qualified tiling pointer.
     */
    const __gm__ SparseFlashAttentionTilingDataMla *__restrict tilingData =
        reinterpret_cast<const __gm__ SparseFlashAttentionTilingDataMla *>(tiling);

    switch (tilingData->dispatchKey & SFA_ARCH22_MASK) {
        SFA_LAUNCH_ALL_LAYOUTS(SGL_SFA_DT_BF16, bfloat16_t)
        SFA_LAUNCH_ALL_LAYOUTS(SGL_SFA_DT_FP16, half)
        default:
            /*
             * Unreachable: the host builds the key from the same enums, and the tiling has already
             * rejected any layout or dtype outside these combinations. There is no way to raise an
             * error from device code, so fall through and leave the outputs untouched rather than
             * running an arbitrary instantiation.
             */
            break;
    }
}
