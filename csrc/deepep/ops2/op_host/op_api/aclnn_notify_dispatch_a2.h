
#ifndef ACLNN_NOTIFY_DISPATCH_A2_H_
#define ACLNN_NOTIFY_DISPATCH_A2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnNotifyDispatchA2GetWorkspaceSize
 * parameters :
 * sendData : required
 * tokenPerExpertData : required
 * sendCount : required
 * numTokens : required
 * topkNum : required
 * numExperts : required
 * commGroup : required
 * rankSize : required
 * rankId : required
 * localRankSize : required
 * localRankId : required
 * sendDataOffset : required
 * recvData : required
 * tokenServerIdx : required
 * tokenUniquePerServer : required
 * epRankTokenCnt : required
 * localEpTokenCnt : required
 * srcOffsetRankTokenIdx : required
 * dstOffsetRankTokenIdx : required
 * offsetInner : required
 * countOuter : required
 * expandIdx : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnNotifyDispatchA2GetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, const aclTensor *tmpData, int64_t sendCount,
    int64_t numTokens, int64_t topkNum, int64_t numExperts, char *commGroup, int64_t rankSize, int64_t rankId,
    int64_t localRankSize, int64_t localRankId, const aclTensor *sendDataOffset, const aclTensor *recvData,
    const aclTensor *tokenServerIdx, const aclTensor *tokenUniquePerServer, const aclTensor *epRankTokenCnt,
    const aclTensor *localEpTokenCnt, const aclTensor *srcOffsetRankTokenIdx, const aclTensor *dstOffsetRankTokenIdx,
    const aclTensor *offsetInner, const aclTensor *countOuter, const aclTensor *expandIdx,
    const aclTensor *totalRecvTokens, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnNotifyDispatch
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnNotifyDispatchA2(void *workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
