
#ifndef ACLNN_SHMEM_NOTIFY_DISPATCH_H_
#define ACLNN_SHMEM_NOTIFY_DISPATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnShmemNotifyDispatchGetWorkspaceSize
 * parameters :
 * tokenPerExpertData : required
 * sendCount : required
 * rankSize : required
 * rankId : required
 * localRankSize : required
 * localRankId : required
 * topkNum : required
 * shmemPtr : required
 * recvData : required
 * totalRecvTokens : required
 * maxBs : required
 * recvTokensPerExpert : required
 * putOffset : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnShmemNotifyDispatchGetWorkspaceSize(
    const aclTensor *tokenPerExpertData, int64_t sendCount, int64_t rankSize, int64_t rankId, int64_t localRankSize,
    int64_t localRankId, int64_t topkNum, uint64_t shmemPtr, const aclTensor *recvData, const aclTensor *totalRecvTokens,
    const aclTensor *maxBs, const aclTensor *recvTokensPerExpert, const aclTensor *putOffset, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnShmemNotifyDispatch
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnShmemNotifyDispatch(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
