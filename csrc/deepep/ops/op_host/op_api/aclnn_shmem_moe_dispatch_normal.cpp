#include <string.h>
#include "graph/types.h"
#include "aclnn_shmem_moe_dispatch_normal.h"
#include "aclnnInner_shmem_moe_dispatch_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnShmemMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendTokenIdx, const aclTensor *putOffset,
    int64_t epWorldSize, int64_t epRankId, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum,
    int64_t quantMode, int64_t globalBs, uint64_t shmemPtr, const aclTensor *recvX, const aclTensor *recvXScales,
    const aclTensor *assistInfoForCombine, const aclTensor *waitRecvCostStats, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerShmemMoeDispatchNormalGetWorkspaceSize(
        x, topkIdx, sendTokenIdx, putOffset, epWorldSize, epRankId, tpWorldSize, tpRankId, moeExpertNum, quantMode,
        globalBs, shmemPtr, recvX, recvXScales, assistInfoForCombine, waitRecvCostStats, workspaceSize, executor);
}

aclnnStatus aclnnShmemMoeDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerShmemMoeDispatchNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
