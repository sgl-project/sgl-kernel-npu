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

#ifndef DATA_COPY_TRANSPOSE_TILING_H
#define DATA_COPY_TRANSPOSE_TILING_H

#pragma once

#include <vector>
#include <graph/tensor.h>
#include "data_copy_transpose_tiling_def.h"

namespace optiling {

inline void GetDataCopyTransposeTiling(const ge::Shape &dstShape, const ge::Shape &srcShape, const uint32_t typeSize,
                                       optiling::CopyTransposeTiling &tiling)
{
    std::vector<int64_t> dstShapeInfo = dstShape.GetDims();
    std::vector<int64_t> srcShapeInfo = srcShape.GetDims();

    tiling.set_dstShapeB(dstShapeInfo[0]);
    tiling.set_dstShapeN(dstShapeInfo[1]);
    tiling.set_dstShapeS(dstShapeInfo[2]); // 2 is dim2
    tiling.set_dstShapeH(dstShapeInfo[3]); // 3 is dim3
    tiling.set_dstShapeHN(tiling.get_dstShapeH() / tiling.get_dstShapeN());

    tiling.set_srcShapeB(srcShapeInfo[0]);
    tiling.set_srcShapeN(srcShapeInfo[1]);
    tiling.set_srcShapeS(srcShapeInfo[2]); // 2 is dim2
    tiling.set_srcShapeHN(srcShapeInfo[3]); // 3 is dim3
    tiling.set_originalShapeNLen(tiling.get_srcShapeHN() * typeSize);
    tiling.set_shapeSHValue(tiling.get_dstShapeS() * tiling.get_dstShapeH());
    tiling.set_shapeNsValue(tiling.get_dstShapeN() * tiling.get_dstShapeS());
    tiling.set_shapeNsnValue(tiling.get_dstShapeN() * tiling.get_srcShapeS() * tiling.get_srcShapeN());
    tiling.set_shapeBHValue(tiling.get_dstShapeB() * tiling.get_dstShapeH());
}

} // namespace optiling
#endif  // DATA_COPY_TRANSPOSE_TILING__H