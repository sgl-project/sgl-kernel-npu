/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMegaMoeGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weightScales1Optional,
    const aclTensorList *weightScales2Optional, const aclTensor *xActiveMaskOptional, const aclTensor *scalesOptional,
    int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize, int64_t maxRecvTokenNum,
    int64_t dispatchQuantMode, int64_t dispatchQuantOutType, int64_t combineQuantMode, const char *commAlg,
    int64_t globalBs, int64_t sharedExpertNum, aclTensor *yOut, aclTensor *expertTokenNumsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream);

// Merged-list protocol: the prototype keeps the pre-shared-expert input set;
// shared-expert weights ride as trailing entries of the weight/scale lists
// with the shared_expert_num attribute. Calls that disable shared experts
// therefore convert exactly what release/2606 converted at launch.
aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weightScales1Optional,
    const aclTensorList *weightScales2Optional, const aclTensor *xActiveMaskOptional, const aclTensor *scalesOptional,
    int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize, int64_t maxRecvTokenNum,
    int64_t dispatchQuantMode, int64_t dispatchQuantOutType, int64_t combineQuantMode, const char *commAlg,
    int64_t globalBs, int64_t sharedExpertNum, aclTensor *yOut, aclTensor *expertTokenNumsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_LOGD("aclnn_mega_moe WorkspaceSize start");

    OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkWeights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertTokenNumsOut, return ACLNN_ERR_PARAM_NULLPTR);

    return aclnnInnerMegaMoeGetWorkspaceSize(context, x, topkIds, topkWeights, weight1, weight2, weightScales1Optional,
                                             weightScales2Optional, xActiveMaskOptional, scalesOptional, moeExpertNum,
                                             epWorldSize, cclBufferSize, maxRecvTokenNum, dispatchQuantMode,
                                             dispatchQuantOutType, combineQuantMode, commAlg, globalBs, sharedExpertNum,
                                             yOut, expertTokenNumsOut, workspaceSize, executor);
}

aclnnStatus aclnnMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_mega_moe start");
    return aclnnInnerMegaMoe(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
