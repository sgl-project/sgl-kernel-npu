#include <string.h>
#include <cstdio>
#include "graph/types.h"
#include "aclnn_dispatch_normal_a2.h"
#include "aclnn/opdev/platform.h"
#include "aclnnInner_dispatch_normal_a2.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnDispatchNormalA2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask,
    const aclTensor *expertScales, const aclTensor *tokenServerIdx, const aclTensor *tokenServerCnt,
    const aclTensor *epRankTokenCnt, const aclTensor *srcOffsetRankTokenIdx, const aclTensor *dstOffsetRankTokenIdx,
    char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, char *groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t globalBs, int64_t expertTokenNumsType, const aclTensor *recvX, const aclTensor *dynamicScales,
    const aclTensor *expandIdx, const aclTensor *expertTokenNums, const aclTensor *epRecvCount,
    const aclTensor *expandScales, const aclTensor *waitRecvCostStats, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerDispatchNormalA2GetWorkspaceSize(
        x, expertIds, scales, xActiveMask, expertScales, tokenServerIdx, tokenServerCnt, epRankTokenCnt,
        srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, groupEp, epWorldSize, epRankId, moeExpertNum, groupTp,
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs,
        expertTokenNumsType, recvX, dynamicScales, expandIdx, expertTokenNums, epRecvCount, expandScales,
        waitRecvCostStats, workspaceSize, executor);
}

aclnnStatus aclnnDispatchNormalA2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return aclnnInnerDispatchNormalA2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
