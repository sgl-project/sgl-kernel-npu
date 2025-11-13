#include "aclnn_moe_distribute_combine_neg_one.h"
#include "aclnnInner_moe_distribute_combine_neg_one.h"
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

extern aclnnStatus aclnnInnerMoeDistributeCombineNegOneGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCounts,
    const aclTensor *xActiveMask, const aclTensor *activationScale, const aclTensor *weightScale,
    const aclTensor *groupList, const aclTensor *expandScales, const aclTensor *sharedExpertX,
    const aclTensor *elasticInfo, const aclTensor *oriX, const aclTensor *constExpertAlpha1,
    const aclTensor *constExpertAlpha2, const aclTensor *constExpertV, char *groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, char *groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum,
    int64_t constExpertNum, const aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDistributeCombineNegOne(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeDistributeCombineNegOneGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
    const aclTensor *epSendCounts, const aclTensor *expertScales, const aclTensor *tpSendCountsOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *activationScaleOptional,
    const aclTensor *weightScaleOptional, const aclTensor *groupListOptional, const aclTensor *expandScalesOptional,
    const aclTensor *sharedExpertXOptional, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    char *commAlg, const aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerMoeDistributeCombineNegOneGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional, xActiveMaskOptional,
        activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional, sharedExpertXOptional,
        nullptr, nullptr, nullptr, nullptr, nullptr, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize,
        tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode,
        groupListType, commAlg, 0, 0, 0, xOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeCombineNegOne(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    return aclnnInnerMoeDistributeCombineNegOne(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
