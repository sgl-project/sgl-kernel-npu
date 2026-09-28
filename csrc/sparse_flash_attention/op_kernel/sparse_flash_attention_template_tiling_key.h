/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_template_tiling_key.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H
#define SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H

#include <cstdint>
#define SFA_LAYOUT_BSND 0
#define SFA_LAYOUT_TND 1
#define SFA_LAYOUT_PA_BSND 2

#define ASCENDC_TPL_4_BW 4

#define C_TEMPLATE 0
#define V_TEMPLATE 1

#define GET_TPL_TILING_KEY(flash_decode, page_attention, layout, kv_layout, template_mode, split_g) \
    (static_cast<uint64_t>(flash_decode) | (static_cast<uint64_t>(page_attention) << 1) |           \
     (static_cast<uint64_t>(layout) << 2) | (static_cast<uint64_t>(kv_layout) << 6) |               \
     (static_cast<uint64_t>(template_mode) << 10) | (static_cast<uint64_t>(split_g) << 14))

#endif  // TEMPLATE_TILING_KEY
