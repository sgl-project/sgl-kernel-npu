/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_tiling_key.h
 * \brief tilingKey 编码，host 侧 DoTiling 与 device 侧 dispatch 共用。
 *
 * 替代上游的 ASCENDC_TPL_ARGS_DECL / GET_TPL_TILING_KEY：sgl-kernel-npu 只有一个
 * kernel 二进制，tilingKey 纯粹用于 device 侧 switch 分发，因此沿用本仓库
 * lightning_indexer 已有的 nibble 编码，并新增 weights-fp32 位。
 *
 *   bits 24-27 : query dtype
 *   bits 20-23 : weights 为 fp32 时置 1 (对应上游 DT_W_FLAG)
 *   bits 16-19 : key dtype
 *   bits 12-15 : output dtype
 *   bits  8-11 : page attention flag
 *   bits  4-7  : query layout
 *   bits  0-3  : key layout
 *
 * dtype 取值必须与 ge_helper.h 的 GE_DATATYPE_TO_KEY() 一致（host 侧有
 * static_assert 守护）。
 */
#ifndef SGL_LI_V2_TILING_KEY_H
#define SGL_LI_V2_TILING_KEY_H

// 与 GE_DATATYPE_TO_KEY() 对齐
#define SGL_LI_V2_DT_FP16 1
#define SGL_LI_V2_DT_INT32 3
#define SGL_LI_V2_DT_BF16 12

// 与 LICommon::LI_LAYOUT 对齐
#define SGL_LI_V2_LAYOUT_BSND 0
#define SGL_LI_V2_LAYOUT_TND 1
#define SGL_LI_V2_LAYOUT_PA_BSND 2

#define SGL_LI_V2_TILING_KEY(DT_Q, DT_K, DT_OUT, W_FP32, PAGE_ATTENTION, LAYOUT_Q, LAYOUT_K)                    \
    ((static_cast<unsigned int>(DT_Q) << 24) | (static_cast<unsigned int>(W_FP32) << 20) |                      \
     (static_cast<unsigned int>(DT_K) << 16) | (static_cast<unsigned int>(DT_OUT) << 12) |                      \
     (static_cast<unsigned int>(PAGE_ATTENTION) << 8) | (static_cast<unsigned int>(LAYOUT_Q) << 4) |            \
     (static_cast<unsigned int>(LAYOUT_K)))

#endif  // SGL_LI_V2_TILING_KEY_H
