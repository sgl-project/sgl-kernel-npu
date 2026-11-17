/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// Preserve the existing six fields, including the mode-2-only RoPE contract.
ASCENDC_TPL_ARGS_DECL(
    SglCompressor,
    ASCENDC_TPL_UINT_DECL(X_LAYOUT, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(X_DTYPE, 4, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(COFF, 2, ASCENDC_TPL_UI_LIST, 1, 2),
    ASCENDC_TPL_UINT_DECL(ROTARY_MODE, 2, ASCENDC_TPL_UI_LIST, 2),
    ASCENDC_TPL_UINT_DECL(CACHE_MODE, 2, ASCENDC_TPL_UI_LIST, 1, 2),
    ASCENDC_TPL_UINT_DECL(TEMPLATE_ID, 2, ASCENDC_TPL_UI_LIST, 0, 1, 2));

// Keep all legacy General and FullLoad cases, plus all legal empty keys.
// The Host still selects FullLoad only for short BSH inputs.
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(X_LAYOUT, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(X_DTYPE, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(COFF, ASCENDC_TPL_UI_LIST, 1, 2),
        ASCENDC_TPL_UINT_SEL(ROTARY_MODE, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 1, 2),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_ID, ASCENDC_TPL_UI_LIST, 0, 1, 2),
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::CompressorTilingData)));
