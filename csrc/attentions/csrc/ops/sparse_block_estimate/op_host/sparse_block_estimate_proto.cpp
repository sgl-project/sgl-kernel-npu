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

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"


namespace ops {

static ge::graphStatus SparseBlockEstimateInferShape(gert::InferShapeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SparseBlockEstimateInferDtype(gert::InferDataTypeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseBlockEstimate)
    .InferShape(SparseBlockEstimateInferShape)
    .InferDataType(SparseBlockEstimateInferDtype);

} // namespace ops
