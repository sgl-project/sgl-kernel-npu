/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe operator aclnn api implementation file
 * Author: Wang Qiankun
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe operator aclnn api implementation file
 */
#include "aclnn_fused_deep_moe.h"
#include <cstring>
#include "graph/types.h"
#include "aclnn/opdev/platform.h"
#include "aclnnInner_fused_deep_moe.h"

enum class NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnFusedDeepMoeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *gmm1PermutedWeight,
    const aclTensor *gmm1PermutedWeightScale, const aclTensor *gmm2Weight, const aclTensor *gmm2WeightScale,
    const aclTensor *expertSmoothScalesOptional, const aclTensor *expertScalesOptional, char *groupEp,
    int64_t epRankSize, int64_t epRankId, int64_t moeExpertNum, int64_t shareExpertNum, int64_t shareExpertRankNum,
    int64_t quantMode, int64_t globalBs, const aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerFusedDeepMoeGetWorkspaceSize(
        x, expertIds, gmm1PermutedWeight, gmm1PermutedWeightScale, gmm2Weight, gmm2WeightScale,
        expertSmoothScalesOptional, expertScalesOptional, groupEp, epRankSize, epRankId, moeExpertNum, shareExpertNum,
        shareExpertRankNum, quantMode, globalBs, output, workspaceSize, executor);
}

aclnnStatus aclnnFusedDeepMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return aclnnInnerFusedDeepMoe(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
