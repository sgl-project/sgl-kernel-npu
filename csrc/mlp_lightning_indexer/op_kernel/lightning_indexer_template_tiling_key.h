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

#define LI_TPL_FP16 1
#define LI_TPL_INT32 3
#define LI_TPL_BF16 27

#define LI_LAYOUT_BSND 0
#define LI_LAYOUT_TND 1
#define LI_LAYOUT_PA_BSND 2

// 本仓只需要 host/kernel 生成完全一致的 tiling key，用于运行时 switch 分派。
// 使用纯宏常量表达式，避免 aicore 侧 case 标签触发 host function 调用限制。
#define GET_TPL_TILING_KEY(DT_Q, DT_K, DT_OUT, PAGE_ATTENTION, LAYOUT_T, K_LAYOUT_T) \
    (((((DT_Q) & 0x3FU) << 24) | ((((DT_K) & 0x3FU) << 18)) | ((((DT_OUT) & 0x3FU) << 12)) | \
      ((((PAGE_ATTENTION) & 0x1U) << 11)) | ((((LAYOUT_T) & 0xFU) << 7)) | ((((K_LAYOUT_T) & 0xFU) << 3))))

#endif
