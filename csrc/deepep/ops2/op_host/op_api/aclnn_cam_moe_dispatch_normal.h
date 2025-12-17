#ifndef ACLNN_CAM_MOE_DISPATCH_NORMAL_H_
#define ACLNN_CAM_MOE_DISPATCH_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnCamMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const aclTensor *expert_global_offset,
    const aclTensor *srcrank_in_expert_offset, const aclTensor *r_in_srcrank_offset, char *groupEp, int64_t epWorldSize,
    int64_t epRankId, char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum,
    int64_t quantMode, int64_t realMaxBs, int64_t globalBs, int32_t round, int32_t perRoundTokens,
    const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnCamMoeDispatchNormal(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor,
                                                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
