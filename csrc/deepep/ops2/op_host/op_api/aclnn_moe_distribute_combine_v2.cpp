#include <string.h>
#include <cstdio>

#include "aclnn_moe_distribute_combine_v2.h"
#include "aclnnInner_moe_low_latency_combine_v2.h"
#include "graph/types.h"
#include "aclnn/opdev/platform.h"

#ifndef ACLNN_ERR_INNER_NULLPTR
#define ACLNN_ERR_INNER_NULLPTR (-1)
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeLowLatencyCombineV2GetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCountsOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *activationScaleOptional,
    const aclTensor *weightScaleOptional, const aclTensor *groupListOptional, const aclTensor *expandScalesOptional,
    const aclTensor *sharedExpertXOptional, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    char *commAlg, aclTensor *xOut, const aclTensor *sendCostStats, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return aclnnInnerMoeLowLatencyCombineV2GetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional, xActiveMaskOptional,
        activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional, sharedExpertXOptional,
        nullptr, nullptr, nullptr, nullptr, nullptr, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize,
        tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode,
        groupListType, commAlg, 0, 0, 0, xOut, sendCostStats, workspaceSize, executor);
}

aclnnStatus aclnnMoeLowLatencyCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    if (executor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    return aclnnInnerMoeLowLatencyCombineV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
