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
 * \file sparse_attn_sharedkv_tiling.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_H

#include <cstring>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include "ge_helper.h"
#include "kv_quant_sparse_attn_sharedkv_tiling_data.h"
#include "kv_quant_sparse_attn_sharedkv_check.h"

namespace optiling {

std::string KvQuantSASLayoutToSerialString(SASLayout layout);

// ---------------算子Tiling类---------------
class KvQuantSparseAttnSharedkvTiling
{
public:
    explicit KvQuantSparseAttnSharedkvTiling(ge_helper::TilingContext *context) : context_(context)
    {
        // ge_helper caches every byte of the tiling record, including padding.
        std::memset(static_cast<void *>(&tilingData_), 0, sizeof(tilingData_));
    }
    ge::graphStatus DoOpTiling(KvQuantSASTilingInfo *tilingInfo);
    const KvQuantSparseAttnSharedkvTilingData &GetTilingData() const
    {
        return tilingData_;
    }
    uint32_t GetBlockDim() const
    {
        return blockDim_;
    }
    uint32_t GetTilingKey() const
    {
        return tilingKey_;
    }

private:
    ge_helper::TilingContext *context_ = nullptr;
    SASTemplateMode perfMode_ = SASTemplateMode::SWA_TEMPLATE_MODE;
    KvQuantSparseAttnSharedkvTilingData tilingData_{};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    KvQuantSASTilingInfo *sasInfo_ = nullptr;
};

}  // namespace optiling
#endif
