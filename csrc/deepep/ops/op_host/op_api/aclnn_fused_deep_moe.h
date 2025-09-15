/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe operator aclnn api header file
 * Author: Wang Qiankun
 * Create: 2025-07-19
 * Note: this file was generated automatically do not change it.
 * History: 2025-07-19 create FusedDeepMoe operator aclnn api header file
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
    const aclTensor *gmm1PermutedWeight,
    const aclTensor *gmm1PermutedWeightScale,
    const aclTensor *gmm2Weight,
    const aclTensor *gmm2WeightScale,
    const aclTensor *expertSmoothScalesOptional,
    const aclTensor *expertScalesOptional,
    char *groupEp,
    int64_t epRankSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t shareExpertNum,
    int64_t shareExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    const aclTensor *output,
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
