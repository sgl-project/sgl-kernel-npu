/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: add fused deep moe interface header file.
 * Create: 2025-07-21
 * Note:
 * History: 2025-07-21 add fused deep moe interface header file.
 */

#ifndef FUSED_DEEP_MOE
#define FUSED_DEEP_MOE

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnFusedDeepMoeGetWorkspaceSize(
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
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnFusedDeepMoe(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif