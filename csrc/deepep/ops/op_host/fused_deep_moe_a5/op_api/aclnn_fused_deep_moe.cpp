/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: add fused deep moe interface cpp file.
 * Create: 2025-07-21
 * Note:
 * History: 2025-07-21 add fused deep moe interface cpp file.
 */

#if defined(__DAV_C310__)
#include <string.h>
#include "graph/types.h"
#include "aclnn/opdev/platform.h"
#include "aclnnInner_fused_deep_moe.h"
#include "aclnn_fused_deep_moe.h"

namespace {
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0;
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_MTE = 1;
} // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, int32_t sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnFusedDeepMoeGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIds,
    const aclTensorList *gmm1Weight,
    const aclTensorList *gmm1WeightScale,
    const aclTensorList *gmm2Weight,
    const aclTensorList *gmm2WeightScale,
    const aclTensor *expertScales,
    const aclTensor *shareGmm1WeightOptional,
    const aclTensor *shareGmm1WeightScaleOptional,
    const aclTensor *shareGmm2WeightOptional,
    const aclTensor *shareGmm2WeightScaleOptional,
    const aclTensor *expertSmoothScalesOptional,
    const aclTensor *shareSmoothScalesOptional,
    const aclTensor *xActiveMaskOptional,
    char *groupEp,
    int64_t epRankSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t quantMode,
    int64_t globalBs,
    const aclTensor *output,
    const aclTensor *shareOutput,
    const aclTensor *expertTokenNums,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerFusedDeepMoeGetWorkspaceSize(x, expertIds, gmm1Weight, gmm1WeightScale,
        gmm2Weight, gmm2WeightScale, expertScales,
        shareGmm1WeightOptional, shareGmm1WeightScaleOptional,
        shareGmm2WeightOptional, shareGmm2WeightScaleOptional,
        expertSmoothScalesOptional, shareSmoothScalesOptional, xActiveMaskOptional,
        groupEp, epRankSize, epRankId, moeExpertNum, quantMode, globalBs,
        output, shareOutput, expertTokenNums, workspaceSize, executor);
}

aclnnStatus aclnnFusedDeepMoe(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return aclnnInnerFusedDeepMoe(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
#endif  // defined(__DAV_C310__)
