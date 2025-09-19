#ifndef ACLNN_DISPATCH_LAYOUT_A2_H_
#define ACLNN_DISPATCH_LAYOUT_A2_H_

#include <cstdint>
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnDispatchLayoutGetWorkspaceSize
 * topkIdx : required
 * numTokens : required
 * numRanks : required
 * numExperts : required
 * numTopk : required
 * localRankSize : required
 * numTokensPerRank : required
 * numTokensPerExpert : required
 * isTokenInRank : required
 * localTokenServerOffset : required
 * localTokenServerUniqCount : required
 * localTokenServerTotalCount : required
 * localTokenServerNum : required
 * expertRankTokenIdx : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchLayoutA2GetWorkspaceSize(
    const aclTensor *topkIdx, int64_t numTokens, int64_t numRanks, int64_t numExperts, int64_t numTopk,
    int64_t localRankSize, const aclTensor *numTokensPerRank, const aclTensor *numTokensPerExpert,
    const aclTensor *isTokenInRank, const aclTensor *localTokenServerOffset, const aclTensor *localTokenServerUniqCount,
    const aclTensor *localTokenServerTotalCount, const aclTensor *localTokenServerNum,
    const aclTensor *expertRankTokenIdx, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnDispatchLayout
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchLayoutA2(void *workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
