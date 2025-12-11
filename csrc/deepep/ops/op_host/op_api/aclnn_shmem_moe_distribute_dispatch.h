
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SHMEM_MOE_DISTRIBUTE_DISPATCH_H_
#define ACLNN_SHMEM_MOE_DISTRIBUTE_DISPATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnShmemMoeDistributeDispatchGetWorkspaceSize
 * parameters :
 * x : required
 * expertIds : required
 * scalesOptional : optional
 * xActiveMaskOptional : optional
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
 * expandXOut : required
 * dynamicScalesOut : required
 * expandIdxOut : required
 * expertTokenNumsOut : required
 * epRecvCountOut : required
 * tpRecvCountOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnShmemMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIds,
    const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    int64_t expertTokenNumsType,
    int64_t extInfo,
    const aclTensor *expandXOut,
    const aclTensor *dynamicScalesOut,
    const aclTensor *expandIdxOut,
    const aclTensor *expertTokenNumsOut,
    const aclTensor *epRecvCountOut,
    const aclTensor *tpRecvCountOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnShmemMoeDistributeDispatch
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnShmemMoeDistributeDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
