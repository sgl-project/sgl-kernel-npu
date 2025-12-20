
/*
 * caution: this file was generated automatically do not change it.
 */

#ifndef ACLNN_INNER_SHMEM_MOE_DISTRIBUTE_DISPATCH_V2_H_
#define ACLNN_INNER_SHMEM_MOE_DISTRIBUTE_DISPATCH_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnShmemMoeDistributeDispatchV2GetWorkspaceSize
 * parameters :
 * x : required
 * expertIds : required
 * scalesOptional : optional
 * xActiveMaskOptional : optional
 * elasticInfoOptional : optional
 * epWorldSize : required
 * epRankId : required
 * moeExpertNum : required
 * tpWorldSize : optional
 * tpRankId : optional
 * expertShardType : optional
 * sharedExpertNum : optional
 * sharedExpertRankNum : optional
 * quantMode : optional
 * globalBs : optional
 * expertTokenNumsType : optional
 * extInfo : required
 * commAlgOptional : optional
 * zeroExpertNum : optional
 * copyExpertNum : optional
 * constExpertNum : optional
 * expandXOut : required
 * dynamicScalesOut : required
 * assistInfoForCombineOut : required
 * expertTokenNumsOut : required
 * epRecvCountOut : required
 * tpRecvCountOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t extInfo,
    char *commAlgOptional, const aclTensor *expandXOut, const aclTensor *dynamicScalesOut,
    const aclTensor *assistInfoForCombineOut, const aclTensor *expertTokenNumsOut, const aclTensor *epRecvCountOut,
    const aclTensor *tpRecvCountOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnShmemMoeDistributeDispatchV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnShmemMoeDistributeDispatchV2(void *workspace,
                                                                                     uint64_t workspaceSize,
                                                                                     aclOpExecutor *executor,
                                                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
