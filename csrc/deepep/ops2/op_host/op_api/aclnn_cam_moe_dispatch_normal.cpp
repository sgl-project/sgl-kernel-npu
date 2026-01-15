#include <string.h>
#include "graph/types.h"
#include "aclnn_cam_moe_dispatch_normal.h"
#include "aclnnInner_cam_moe_dispatch_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCamMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const aclTensor *expert_global_offset,
    const aclTensor *srcrank_in_expert_offset, const aclTensor *r_in_srcrank_offset, char *groupEp, int64_t epWorldSize,
    int64_t epRankId, char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum,
    int64_t quantMode, int64_t realMaxBs, int64_t globalBs, int32_t round, int32_t perRoundTokens,
    const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCamMoeDispatchNormalGetWorkspaceSize(
        x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount, expert_global_offset, srcrank_in_expert_offset,
        r_in_srcrank_offset, groupEp, epWorldSize, epRankId, groupTpOptional, tpWorldSize, tpRankId, moeExpertNum,
        quantMode, realMaxBs, globalBs, round, perRoundTokens, recvX, recvXScales, assistInfoForCombine,
        waitRecvCostStats, workspaceSize, executor);
}

aclnnStatus aclnnCamMoeDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    }
    return aclnnInnerCamMoeDispatchNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
