#include "graph/types.h"
#include "aclnn_notify_dispatch_a2.h"
#include "aclnnInner_notify_dispatch_a2.h"

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnNotifyDispatchA2GetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, const aclTensor *tmpData, int64_t sendCount,
    int64_t numTokens, int64_t topkNum, int64_t numExperts, char *commGroup, int64_t rankSize, int64_t rankId,
    int64_t localRankSize, int64_t localRankId, const aclTensor *sendDataOffset, const aclTensor *recvData,
    const aclTensor *tokenServerIdx, const aclTensor *tokenUniquePerServer, const aclTensor *epRankTokenCnt,
    const aclTensor *localEpTokenCnt, const aclTensor *srcOffsetRankTokenIdx, const aclTensor *dstOffsetRankTokenIdx,
    const aclTensor *offsetInner, const aclTensor *countOuter, const aclTensor *expandIdx,
    const aclTensor *totalRecvTokens, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerNotifyDispatchA2GetWorkspaceSize(
        sendData, tokenPerExpertData, tmpData, sendCount, numTokens, topkNum, numExperts, commGroup, rankSize, rankId,
        localRankSize, localRankId, sendDataOffset, recvData, tokenServerIdx, tokenUniquePerServer, epRankTokenCnt,
        localEpTokenCnt, srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, offsetInner, countOuter, expandIdx,
        totalRecvTokens, workspaceSize, executor);
}

aclnnStatus aclnnNotifyDispatchA2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);  // A2 需要为 AICPU
    }
    return aclnnInnerNotifyDispatchA2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
