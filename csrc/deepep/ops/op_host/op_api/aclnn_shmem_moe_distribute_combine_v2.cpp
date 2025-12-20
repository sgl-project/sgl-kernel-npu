#include "aclnn_shmem_moe_distribute_combine_v2.h"
#include "aclnnInner_shmem_moe_distribute_combine_v2.h"
#include <algorithm>
#include "graph/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerShmemMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCounts,
    const aclTensor *xActiveMask, const aclTensor *activationScale, const aclTensor *weightScale,
    const aclTensor *groupList, const aclTensor *expandScales, const aclTensor *sharedExpertX,
    const aclTensor *elasticInfo, const aclTensor *oriX, const aclTensor *constExpertAlpha1,
    const aclTensor *constExpertAlpha2, const aclTensor *constExpertV, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t extInfo,
    int64_t groupListType, char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    const aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerShmemMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize,
                                                         aclOpExecutor *executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnShmemMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCountsOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *activationScaleOptional,
    const aclTensor *weightScaleOptional, const aclTensor *groupListOptional, const aclTensor *expandScalesOptional,
    const aclTensor *sharedExpertXOptional, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t extInfo,
    int64_t groupListType, char *commAlg, const aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerShmemMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional, xActiveMaskOptional,
        activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional, sharedExpertXOptional,
        nullptr, nullptr, nullptr, nullptr, nullptr, epWorldSize, epRankId, moeExpertNum, tpWorldSize, tpRankId,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, extInfo,
        groupListType, commAlg, 0, 0, 0, xOut, workspaceSize, executor);
}

aclnnStatus aclnnShmemMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    return aclnnInnerShmemMoeDistributeCombineV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
