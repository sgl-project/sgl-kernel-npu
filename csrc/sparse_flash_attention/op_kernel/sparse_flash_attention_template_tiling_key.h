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
 * \brief Template-mode and layout constants used by the kernel templates.
 *
 * Upstream's file of this name also carries ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_SEL blocks (from
 * "ascendc/host_api/tiling/template_argument.h") that enumerate the legal template-argument
 * combinations for the CANN op packer, so it can compile one kernel binary per combination and
 * GET_TPL_TILING_KEY can select between them.
 *
 * This build produces a single kernel binary and selects the instantiation at run time from
 * SparseFlashAttentionTilingDataMla::dispatchKey, so there is nothing for the packer to enumerate and
 * the header it needs is not available here. Only the plain constants survive -- the kernel templates
 * compare TEMPLATE_MODE against C_TEMPLATE / V_TEMPLATE, and SFAType defaults to C_TEMPLATE.
 *
 * The upstream combination list is not lost, only moved: it is what the switch in
 * sparse_flash_attention.cpp enumerates. The layout codes are kept identical to upstream's and are
 * tied to the SFALayout enum by static_assert in op_host/sparse_flash_attention.cpp.
 */

#ifndef SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H
#define SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H

#define SFA_LAYOUT_BSND 0
#define SFA_LAYOUT_TND 1
#define SFA_LAYOUT_PA_BSND 2

#define C_TEMPLATE 0
#define V_TEMPLATE 1

#endif  // SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H
