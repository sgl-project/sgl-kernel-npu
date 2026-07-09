/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_key.h
 * \brief causal_conv1d tiling key constants
 */

#ifndef __CUSTOM_CAUSAL_CONV1D_TILING_KEY_H__
#define __CUSTOM_CAUSAL_CONV1D_TILING_KEY_H__

#define CAUSAL_CONV1D_TPL_RUN_MODE_FN 0
#define CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE 1
#define CAUSAL_CONV1D_TPL_WIDTH_RUNTIME 0
#define CAUSAL_CONV1D_TPL_WIDTH_2 1
#define CAUSAL_CONV1D_TPL_WIDTH_3 2
#define CAUSAL_CONV1D_TPL_WIDTH_4 3
#define CAUSAL_CONV1D_TPL_FN_PLAN_INVALID 0
#define CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS 1
#define CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD 2

#endif  // __CUSTOM_CAUSAL_CONV1D_TILING_KEY_H__
