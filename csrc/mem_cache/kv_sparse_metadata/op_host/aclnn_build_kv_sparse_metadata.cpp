/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE-CANN in the root of the software repository for the full text of the License.
 */

// ACLNN launch integration follows examples/add_example.
#include "aclnn_build_kv_sparse_metadata.h"
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
OP_TYPE_REGISTER(BuildKvSparseMetadata);

static aclnnStatus BuildMetadata(
    const aclTensor *di, const aclTensor *hi, const aclIntArray *componentInfo,
    int64_t pageSize, int64_t direction, const aclTensor *src, const aclTensor *dst,
    const aclTensor *lengths, const aclTensor *count, aclOpExecutor *executor)
{
    static internal::AicpuTaskSpace space("BuildKvSparseMetadata");
    return ADD_TO_LAUNCHER_LIST_AICPU(
        BuildKvSparseMetadata, OP_ATTR_NAMES({"component_info", "page_size", "direction"}),
        OP_INPUT(di, hi), OP_OUTPUT(src, dst, lengths, count), OP_ATTR(componentInfo, pageSize, direction));
}
}

extern "C" aclnnStatus aclnnBuildKvSparseMetadataGetWorkspaceSize(
    const aclTensor *di, const aclTensor *hi, const aclIntArray *componentInfo,
    int64_t pageSize, int64_t direction, const aclTensor *src, const aclTensor *dst,
    const aclTensor *lengths, const aclTensor *count, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!di || !hi || !componentInfo || !src || !dst || !lengths || !count || !workspaceSize || !executor) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    if (pageSize <= 0 || (direction != 1 && direction != 2) ||
        di->GetDataType() != op::DataType::DT_INT64 || hi->GetDataType() != op::DataType::DT_INT64 ||
        src->GetDataType() != op::DataType::DT_INT64 || dst->GetDataType() != op::DataType::DT_INT64 ||
        lengths->GetDataType() != op::DataType::DT_INT32 || count->GetDataType() != op::DataType::DT_INT32 ||
        di->GetViewShape() != hi->GetViewShape() || src->GetViewShape() != dst->GetViewShape()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto *tensor : {di, hi, src, dst, lengths, count}) {
        if (!op::IsContiguous(tensor) || tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (!uniqueExecutor.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    const auto status = l0op::BuildMetadata(
        di, hi, componentInfo, pageSize, direction, src, dst, lengths, count, uniqueExecutor.get());
    if (status != ACLNN_SUCCESS) return status;
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnBuildKvSparseMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
