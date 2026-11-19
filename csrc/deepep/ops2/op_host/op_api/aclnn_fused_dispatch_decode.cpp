#include "aclnn_fused_dispatch_decode.h"
#include "aclnnInner_fused_dispatch_decode.h"
enum NnopbaseHcclServerType { NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0 };
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, NnopbaseHcclServerType);
extern "C" {
aclnnStatus aclnnFusedDispatchDecodeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *ids, int32_t sendCount, int32_t numTokens,
    char *commGroup, int32_t rankSize, int32_t rankId, int32_t localRankSize, int32_t localRankId,
    int32_t round, int32_t perRoundTokens, const aclTensor *values, const aclTensor *scales,
    const aclTensor *ends, const aclTensor *source, const aclTensor *counts, const aclTensor *total,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerFusedDispatchDecodeGetWorkspaceSize(x, ids, sendCount, numTokens, commGroup,
        rankSize, rankId, localRankSize, localRankId, round, perRoundTokens,
        values, scales, ends, source, counts, total, workspaceSize, executor);
}
aclnnStatus aclnnFusedDispatchDecode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    return aclnnInnerFusedDispatchDecode(workspace, workspaceSize, executor, stream);
}
}
