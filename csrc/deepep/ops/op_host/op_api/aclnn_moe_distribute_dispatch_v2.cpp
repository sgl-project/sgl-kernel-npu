#include "aclnn_moe_distribute_dispatch_v2.h"
#include "aclnnInner_moe_distribute_dispatch_v2.h"
#include <algorithm>
#include "graph/types.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask,
    const aclTensor *elasticInfo, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t shareExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, char *commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, const aclTensor *expandX,
    const aclTensor *dynamicScales, const aclTensor *assist_info_for_combine, const aclTensor *expertTokensNums,
    const aclTensor *epRecvCounts, const aclTensor *tpRecvCounts, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerMoeDistributeDispatchV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, char *commAlg,
    const aclTensor *expandXOut, const aclTensor *dynamicScalesOut, const aclTensor *assistInfoForCombineOut,
    const aclTensor *expertTokenNumsOut, const aclTensor *epRecvCountsOut, const aclTensor *tpRecvCountsOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, nullptr, groupEp, epWorldSize, epRankId, moeExpertNum,
        groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs,
        expertTokenNumsType, commAlg, 0, 0, 0, expandXOut, dynamicScalesOut, assistInfoForCombineOut,
        expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeDispatchV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
