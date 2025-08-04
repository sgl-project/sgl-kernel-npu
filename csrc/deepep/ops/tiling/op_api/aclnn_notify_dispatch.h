
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_NOTIFY_DISPATCH_H_
#define ACLNN_NOTIFY_DISPATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnNotifyDispatchGetWorkspaceSize
 * parameters :
 * sendData : required
 * sendCount : required
 * commGroup : required
 * rankSize : required
 * rankId : required
 * localRankSize : required
 * localRankId : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData,
    int64_t sendCount,
    char *commGroup,
    int64_t rankSize,
    int64_t rankId,
    int64_t localRankSize,
    int64_t localRankId,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnNotifyDispatch
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNotifyDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
