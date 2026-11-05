/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE-CANN in the root of the software repository for the full text of the License.
 */

// Host adapter follows the ops-transformer metadata operator's AICPU launcher
// at 39b52ad5e252d98e13df76ed45b6c68b21b98859. AddExample has no upstream
// handwritten ACLNN API; this file supplies the restricted public contract.
#include "aclnn_sgl_add_example.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include <initializer_list>

namespace l0op {
using namespace op;
OP_TYPE_REGISTER(SglAddExample);

static aclnnStatus AddExample(const aclTensor *x1, const aclTensor *x2, const aclTensor *y, aclOpExecutor *executor)
{
    // CANN resolves this OpType through cust_aicpu_kernel.json at runtime.
    static internal::AicpuTaskSpace space("SglAddExample");
    return ADD_TO_LAUNCHER_LIST_AICPU(SglAddExample, OP_ATTR_NAMES({}), OP_INPUT(x1, x2), OP_OUTPUT(y), OP_ATTR());
}
}  // namespace l0op

extern "C" aclnnStatus aclnnSglAddExampleGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *y,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Validate the same-shape, contiguous ND contract before enqueueing device work.
    if (x1 == nullptr || x2 == nullptr || y == nullptr || workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    const auto dtype = x1->GetDataType();
    if ((dtype != op::DataType::DT_FLOAT && dtype != op::DataType::DT_INT32) || x2->GetDataType() != dtype ||
        y->GetDataType() != dtype || x1->GetViewShape() != x2->GetViewShape() ||
        x1->GetViewShape() != y->GetViewShape()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto *tensor : {x1, x2, y}) {
        if (!op::IsContiguous(tensor) || tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (x1->IsEmpty()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    const auto status = l0op::AddExample(x1, x2, y, uniqueExecutor.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSglAddExample(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
