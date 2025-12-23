#include <string.h>
#include "graph/types.h"
#include "aclnn_shmem_moe_combine_normal.h"
#include "aclnnInner_shmem_moe_combine_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnShmemMoeCombineNormalGetWorkspaceSize(const aclTensor *recvX, const aclTensor *epRecvCounts,
                                                       const aclTensor *recvTopkWeights, const aclTensor *topkIdx,
                                                       const aclTensor *sendTokenIdx, uint64_t meta_data_ptr,
                                                       int64_t epWorldSize, int64_t epRankId, int64_t tpWorldSize,
                                                       int64_t tpRankId, int64_t moeExpertNum, int64_t globalBs,
                                                       const aclTensor *out, const aclTensor *sendCostStats,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerShmemMoeCombineNormalGetWorkspaceSize(
        recvX, epRecvCounts, recvTopkWeights, topkIdx, sendTokenIdx, meta_data_ptr, epWorldSize, epRankId, tpWorldSize,
        tpRankId, moeExpertNum, globalBs, out, sendCostStats, workspaceSize, executor);
}

aclnnStatus aclnnShmemMoeCombineNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerShmemMoeCombineNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
