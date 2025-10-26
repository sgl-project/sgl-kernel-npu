#ifndef ACLNN_DISPATCH_NORMAL_A2_H_
#define ACLNN_DISPATCH_NORMAL_A2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchNormalA2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask,
    const aclTensor *expertScales, const aclTensor *tokenServerIdx, const aclTensor *tokenServerCnt,
    const aclTensor *epRankTokenCnt, const aclTensor *srcOffsetRankTokenIdx, const aclTensor *dstOffsetRankTokenIdx,
    char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, char *groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t globalBs, int64_t expertTokenNumsType, const aclTensor *recvX, const aclTensor *dynamicScales,
    const aclTensor *expandIdx, const aclTensor *expertTokenNums, const aclTensor *epRecvCount,
    const aclTensor *expandScales, const aclTensor *waitRecvCostStats, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchNormalA2(void *workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
