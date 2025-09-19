#include <string.h>
#include "graph/types.h"
#include "aclnn_dispatch_layout_a2.h"
#include "aclnnInner_dispatch_layout_a2.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnDispatchLayoutA2GetWorkspaceSize(const aclTensor *topkIdx, int64_t numTokens, int64_t numRanks,
                                                int64_t numExperts, int64_t numTopk, int64_t localRankSize, const aclTensor *numTokensPerRank,
                                                const aclTensor *numTokensPerExpert, const aclTensor *isTokenInRank,
                                                const aclTensor *localTokenServerOffset, const aclTensor *localTokenServerUniqCount,
                                                const aclTensor *localTokenServerTotalCount, const aclTensor *localTokenServerNum,
                                                const aclTensor *expertRankTokenIdx, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerDispatchLayoutA2GetWorkspaceSize(topkIdx, numTokens, numRanks, numExperts, numTopk, localRankSize,
                                                      numTokensPerRank, numTokensPerExpert, isTokenInRank,
                                                      localTokenServerOffset, localTokenServerUniqCount,
                                                      localTokenServerTotalCount, localTokenServerNum,
                                                      expertRankTokenIdx, workspaceSize, executor);
}

aclnnStatus aclnnDispatchLayoutA2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDispatchLayoutA2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
