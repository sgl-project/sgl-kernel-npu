#include <string.h>
#include "graph/types.h"
#include "aclnn_notify_dispatch.h"
#include "aclnnInner_notify_dispatch.h"

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

aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount, int32_t numTokens,
    char *commGroup, int32_t rankSize, int32_t rankId, int32_t localRankSize, int32_t localRankId, int32_t round,
    int32_t perRoundTokens, const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensor *recvCount,
    const aclTensor *recvOffset, const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset,
    const aclTensor *rInSrcrankOffset, const aclTensor *totalRecvTokens, const aclTensor *maxBs,
    const aclTensor *recvTokensPerExpert, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerNotifyDispatchGetWorkspaceSize(
        sendData, tokenPerExpertData, sendCount, numTokens, commGroup, rankSize, rankId, localRankSize, localRankId,
        round, perRoundTokens, sendDataOffset, recvData, recvCount, recvOffset, expertGlobalOffset,
        srcrankInExpertOffset, rInSrcrankOffset, totalRecvTokens, maxBs, recvTokensPerExpert, workspaceSize, executor);
}

aclnnStatus aclnnNotifyDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerNotifyDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
