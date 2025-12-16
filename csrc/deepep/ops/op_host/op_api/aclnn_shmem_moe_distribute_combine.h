
/*
 * caution: this file was generated automatically do not change it.
 */

#ifndef ACLNN_SHMEM_MOE_DISTRIBUTE_COMBINE_H_
#define ACLNN_SHMEM_MOE_DISTRIBUTE_COMBINE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnShmemMoeDistributeCombineGetWorkspaceSize
 * parameters :
 * expandX : required
 * expertIds : required
 * expandIdx : required
 * epSendCounts : required
 * expertScales : required
 * tpSendCountsOptional : optional
 * xActiveMaskOptional : optional
 * activationScaleOptional : optional
 * weightScaleOptional : optional
 * groupListOptional : optional
 * expandScalesOptional : optional
 * epWorldSize : required
 * epRankId : required
 * moeExpertNum : required
 * tpWorldSize : optional
 * tpRankId : optional
 * expertShardType : optional
 * sharedExpertNum : optional
 * sharedExpertRankNum : optional
 * globalBs : optional
 * commQuantMode : optional
 * extInfo : required
 * outDtype : optional
 * groupListType : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeDistributeCombineGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *expandIdx, const aclTensor *epSendCounts,
    const aclTensor *expertScales, const aclTensor *tpSendCountsOptional, const aclTensor *xActiveMaskOptional,
    const aclTensor *activationScaleOptional, const aclTensor *weightScaleOptional, const aclTensor *groupListOptional,
    const aclTensor *expandScalesOptional, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t extInfo, int64_t outDtype,
    int64_t groupListType, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnShmemMoeDistributeCombine
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeDistributeCombine(void *workspace,
                                                                                  uint64_t workspaceSize,
                                                                                  aclOpExecutor *executor,
                                                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
