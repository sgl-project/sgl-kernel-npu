
#ifndef ACLNN_NOTIFY_DISPATCH_H_
#define ACLNN_NOTIFY_DISPATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnNotifyDispatchGetWorkspaceSize
 * parameters :
 * sendData : required
 * tokenPerExpertData : required
 * sendCount : required
 * numTokens : required
 * commGroup : required
 * rankSize : required
 * rankId : required
 * localRankSize : required
 * localRankId : required
 * round : required
 * perRoundTokens : required
 * sendDataOffset : required
 * recvData : required
 * recvCount : required
 * recvOffset : required
 * expertGlobalOffset : required
 * srcrankInExpertOffset : required
 * rInSrcrankOffset : required
 * totalRecvTokens : required
 * maxBs : required
 * recvTokensPerExpert : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount, int32_t numTokens,
    char *commGroup, int32_t rankSize, int32_t rankId, int32_t localRankSize, int32_t localRankId, int32_t round,
    int32_t perRoundTokens, const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensor *recvCount,
    const aclTensor *recvOffset, const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset,
    const aclTensor *rInSrcrankOffset, const aclTensor *totalRecvTokens, const aclTensor *maxBs,
    const aclTensor *recvTokensPerExpert, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnNotifyDispatch
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnNotifyDispatch(void *workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
