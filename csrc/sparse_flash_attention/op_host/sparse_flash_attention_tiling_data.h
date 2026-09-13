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
 * \file sparse_flash_attention_tiling_data.h
 * \brief Plain-struct replacement for the upstream BEGIN_TILING_DATA_DEF tiling data.
 *
 * Upstream (vllm-ascend csrc/attention/sparse_flash_attention) declares its tiling data with the
 * CANN op-project macros BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF / REGISTER_TILING_DATA_CLASS
 * from "register/tilingdata_base.h", which are only available inside a CANN custom-op package. This
 * repo builds the op as part of a plain shared library, so the same fields are declared here as
 * POD structs, exactly as csrc/sparse_attn_sharedkv does.
 *
 * Two consequences, both deliberate:
 *   - Field order and types are kept identical to upstream, because the kernel side reads this
 *     struct back out of GM, viewing this exact layout in place. Do not reorder.
 *   - Upstream's generated `set_field(v)` accessors do not exist; the tiling assigns fields
 *     directly (`baseParams.field = v`). Keep that in mind when diffing against upstream.
 *
 * `dispatchKey` is an addition with no upstream counterpart -- see the comment on it below.
 */
#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_TILING_DATA_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_TILING_DATA_H

#include <cstdint>

/*
 * Dispatch-key encoding, shared by the host (SFAMlaTiling::GenTilingKey) and the kernel entry
 * (op_kernel/sparse_flash_attention.cpp). Both include this header so the two cannot drift.
 *
 * The layout values deliberately match upstream's op_kernel/sparse_flash_attention_template_tiling_key.h
 * (SFA_LAYOUT_BSND/TND/PA_BSND) and the SFALayout enum in sparse_flash_attention_tiling.h, so the
 * axes carry the same meaning they do upstream; static_asserts in the launcher pin that down.
 */
#define SGL_SFA_LAYOUT_BSND 0
#define SGL_SFA_LAYOUT_TND 1
#define SGL_SFA_LAYOUT_PA_BSND 2

#define SGL_SFA_TEMPLATE_C 0
#define SGL_SFA_TEMPLATE_V 1

#define SGL_SFA_DT_FP16 0
#define SGL_SFA_DT_BF16 1

#define SGL_SFA_KEY_QLAYOUT_SHIFT 0
#define SGL_SFA_KEY_KVLAYOUT_SHIFT 4
#define SGL_SFA_KEY_TEMPLATE_SHIFT 8
#define SGL_SFA_KEY_SPLITG_SHIFT 12
#define SGL_SFA_KEY_PAGEATTN_SHIFT 13
#define SGL_SFA_KEY_DTYPE_SHIFT 16

#define SGL_SFA_MAKE_KEY(dtype, qLayout, kvLayout, templateMode, isSplitG, pageAttention) \
    ((static_cast<uint64_t>(qLayout) << SGL_SFA_KEY_QLAYOUT_SHIFT) |                      \
     (static_cast<uint64_t>(kvLayout) << SGL_SFA_KEY_KVLAYOUT_SHIFT) |                    \
     (static_cast<uint64_t>(templateMode) << SGL_SFA_KEY_TEMPLATE_SHIFT) |                \
     (static_cast<uint64_t>(isSplitG) << SGL_SFA_KEY_SPLITG_SHIFT) |                      \
     (static_cast<uint64_t>(pageAttention) << SGL_SFA_KEY_PAGEATTN_SHIFT) |               \
     (static_cast<uint64_t>(dtype) << SGL_SFA_KEY_DTYPE_SHIFT))

namespace optiling {

struct SparseFlashAttentionBaseParamsMla {
    uint32_t batchSize = 0;
    uint32_t seqSize = 0;
    uint32_t qSeqSize = 0;
    int64_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    float scaleValue = 1.0F;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t actualLenDimsQ = 0;
    uint32_t actualLenDimsKV = 0;
    uint32_t outputLayout = 0;
    uint32_t sparseMode = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    uint32_t attentionMode = 0;
    uint32_t returnSoftmaxLse = 0;
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0;
    uint32_t isActualLenDimsNull = 0;
    uint32_t isActualLenDimsKVNull = 0;
};

struct SparseFlashAttentionSplitKVParamsMla {
    uint32_t s2 = 0;             // S2切分份数
    uint32_t accumOutSize = 0;   // FD workspace
    uint32_t logSumExpSize = 0;  // FD workspace
};

struct SparseFlashAttentionSingleCoreParamsMla {
    uint32_t usedCoreNum = 0;
};

struct SparseFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize = 0;
    uint32_t bmm2ResUbSize = 0;
};

struct SparseFlashAttentionInnerSplitParams {
    uint32_t mBaseSize = 0;
    uint32_t s2BaseSize = 0;
};

struct SparseFlashAttentionTilingDataMla {
    SparseFlashAttentionBaseParamsMla baseParams{};
    SparseFlashAttentionSplitKVParamsMla splitKVParams{};
    SparseFlashAttentionSingleCoreParamsMla singleCoreParams{};
    SparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize{};
    SparseFlashAttentionInnerSplitParams innerSplitParams{};

    /*
     * Not present upstream.
     *
     * Upstream selects the kernel template instantiation at *compile* time: GenTilingKey() builds a
     * key with GET_TPL_TILING_KEY and hands it to TilingContext::SetTilingKey(), and the CANN op
     * packer compiles one kernel binary per legal template-argument combination declared in
     * op_kernel/sparse_flash_attention_template_tiling_key.h.
     *
     * This repo launches a single kernel binary through aclrtlaunch_sparse_flash_attention, so there
     * is no per-key binary and ge_helper::TilingContext deletes SetTilingKey() outright. The same
     * selection therefore has to happen at *run* time: the host packs the identical set of axes into
     * this field and the kernel entry switches on it, which is exactly how csrc/sparse_attn_sharedkv
     * handles its own dispatchKey.
     *
     * Layout (see SFAMlaTiling::GenDispatchKey and op_kernel/sparse_flash_attention.cpp, which must
     * agree; the static_asserts in op_host/sparse_flash_attention.cpp pin the dtype codes):
     *
     *   bits  0-3   qLayout        SFA_LAYOUT_BSND(0) | SFA_LAYOUT_TND(1)
     *   bits  4-7   kvLayout       SFA_LAYOUT_BSND(0) | SFA_LAYOUT_TND(1) | SFA_LAYOUT_PA_BSND(2)
     *   bits  8-11  templateMode   C_TEMPLATE(0) | V_TEMPLATE(1)
     *   bit  12     isSplitG       gSize > 64
     *   bit  13     pageAttention  kvLayout == PA_BSND
     *   bits 16-19  dtype          SGL_SFA_DT_FP16(0) | SGL_SFA_DT_BF16(1)
     *
     * FLASH_DECODE is not encoded: upstream's own ASCENDC_TPL_SEL list only ever selects 0 for it,
     * and GenTilingKey() passes a literal 0U.
     */
    uint64_t dispatchKey = 0;
};

}  // namespace optiling

#endif  // SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_TILING_DATA_H
