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
 * \file COMPRESSOR_template_tiling_key.h
 * \brief
 */

#ifndef COMPRESSOR_TEMPLATE_TILING_KEY_H
#define COMPRESSOR_TEMPLATE_TILING_KEY_H

#include <cstdint>

// bit0 layout; bits1-4 dtype; bits5-6 coff; bits7-8 rotary;
// bits9-10 cache; bits11-12 template (0=NORMAL, 1=EMPTY_X, 2=FULL_LOAD).
#define GET_TPL_TILING_KEY(layout, dtype, coff, rotaryMode, cacheMode, templateId)                              \
    (static_cast<uint64_t>(layout) | (static_cast<uint64_t>(dtype) << 1) | (static_cast<uint64_t>(coff) << 5) | \
     (static_cast<uint64_t>(rotaryMode) << 7) | (static_cast<uint64_t>(cacheMode) << 9) |                       \
     (static_cast<uint64_t>(templateId) << 11))

#endif  // COMPRESSOR_TEMPLATE_TILING_KEY_H
