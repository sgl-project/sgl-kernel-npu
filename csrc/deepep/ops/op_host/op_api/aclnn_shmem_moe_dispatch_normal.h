#ifndef ACLNN_SHMEM_MOE_DISPATCH_NORMAL_H_
#define ACLNN_SHMEM_MOE_DISPATCH_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendTokenIdx,
    const aclTensor *putOffset, int64_t epWorldSize, int64_t epRankId,
    int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode,
    int64_t globalBs, uint64_t shmemPtr, const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeDispatchNormal(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor,
                                                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
