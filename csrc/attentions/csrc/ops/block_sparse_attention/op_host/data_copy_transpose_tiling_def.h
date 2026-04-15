/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef DATA_COPY_TRANSPOSE_TILING_DEF_H
#define DATA_COPY_TRANSPOSE_TILING_DEF_H

#pragma once
#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(CopyTransposeTiling)
TILING_DATA_FIELD_DEF(uint32_t, dstShapeB);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeS);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeH);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeB);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeN);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeS);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, originalShapeNLen);
TILING_DATA_FIELD_DEF(uint32_t, shapeSHValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsnValue);
TILING_DATA_FIELD_DEF(uint32_t, invalidParamCopyTransposeTiling);
TILING_DATA_FIELD_DEF(uint32_t, shapeBHValue);
TILING_DATA_FIELD_DEF(uint32_t, paramsAlign);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CopyTransposeTilingOp, CopyTransposeTiling)

} // namespace optiling
#endif  // DATA_COPY_TRANSPOSE_TILING_DEF_H
