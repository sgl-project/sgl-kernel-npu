#ifndef ACLNN_SHMEM_MOE_COMBINE_NORMAL_H_
#define ACLNN_SHMEM_MOE_COMBINE_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX, const aclTensor *epRecvCounts, const aclTensor *recvTopkWeights, const aclTensor *topkIdx,
    const aclTensor *sendTokenIdx, uint64_t meta_data_ptr, int64_t epWorldSize, int64_t epRankId, int64_t tpWorldSize,
    int64_t tpRankId, int64_t moeExpertNum, int64_t globalBs, const aclTensor *out, const aclTensor *sendCostStats,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeCombineNormal(void *workspace, uint64_t workspaceSize,
                                                                              aclOpExecutor *executor,
                                                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
