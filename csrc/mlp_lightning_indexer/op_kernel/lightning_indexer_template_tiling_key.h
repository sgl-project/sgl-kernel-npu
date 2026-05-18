/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_LI_H_
#define TEMPLATE_TILING_KEY_LI_H_

#include <cstdint>

#define LI_TPL_FP16 1
#define LI_TPL_INT32 3
#define LI_TPL_BF16 27

#define LI_LAYOUT_BSND 0
#define LI_LAYOUT_TND 1
#define LI_LAYOUT_PA_BSND 2

// 本仓只需要 host/kernel 生成完全一致的 tiling key，用于运行时 switch 分派。
// 这里用一个稳定的本地 bit-pack 编码，替代外部 template_argument.h 宏体系。
constexpr inline uint32_t LiGetTplTilingKey(uint32_t dt_q, uint32_t dt_k, uint32_t dt_out,
                                            uint32_t page_attention, uint32_t layout_t,
                                            uint32_t k_layout_t)
{
    return ((dt_q & 0x3FU) << 24) | ((dt_k & 0x3FU) << 18) | ((dt_out & 0x3FU) << 12) |
           ((page_attention & 0x1U) << 11) | ((layout_t & 0xFU) << 7) | ((k_layout_t & 0xFU) << 3);
}

#define GET_TPL_TILING_KEY(DT_Q, DT_K, DT_OUT, PAGE_ATTENTION, LAYOUT_T, K_LAYOUT_T) \
    LiGetTplTilingKey((DT_Q), (DT_K), (DT_OUT), (PAGE_ATTENTION), (LAYOUT_T), (K_LAYOUT_T))

#endif
