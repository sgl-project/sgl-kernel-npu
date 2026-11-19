#pragma once
#include "aclnn/acl_meta.h"
extern "C" {
__attribute__((visibility("default"))) aclnnStatus aclnnFusedDispatchDecodeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *ids, int32_t sendCount, int32_t numTokens,
    char *commGroup, int32_t rankSize, int32_t rankId, int32_t localRankSize, int32_t localRankId,
    int32_t round, int32_t perRoundTokens, const aclTensor *values, const aclTensor *scales,
    const aclTensor *ends, const aclTensor *source, const aclTensor *counts, const aclTensor *total,
    uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnFusedDispatchDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}
