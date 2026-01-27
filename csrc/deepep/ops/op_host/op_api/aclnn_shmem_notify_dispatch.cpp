#include <string.h>
#include "graph/types.h"
#include "aclnn_shmem_notify_dispatch.h"
#include "aclnnInner_shmem_notify_dispatch.h"

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

aclnnStatus aclnnShmemNotifyDispatchGetWorkspaceSize(const aclTensor *tokenPerExpertData, int64_t sendCount,
                                                     int64_t rankSize, int64_t rankId, int64_t localRankSize,
                                                     int64_t localRankId, int64_t topkNum, uint64_t shmemPtr,
                                                     const aclTensor *recvData, const aclTensor *totalRecvTokens,
                                                     const aclTensor *maxBs, const aclTensor *recvTokensPerExpert,
                                                     const aclTensor *putOffset, const aclTensor *balanceMatrix,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerShmemNotifyDispatchGetWorkspaceSize(
        tokenPerExpertData, sendCount, rankSize, rankId, localRankSize, localRankId, topkNum, shmemPtr, recvData,
        totalRecvTokens, maxBs, recvTokensPerExpert, putOffset, balanceMatrix, workspaceSize, executor);
}

aclnnStatus aclnnShmemNotifyDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerShmemNotifyDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
