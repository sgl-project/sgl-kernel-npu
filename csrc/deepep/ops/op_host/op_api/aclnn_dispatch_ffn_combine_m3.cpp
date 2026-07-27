/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 1.0.
 */
#include "aclnn_dispatch_ffn_combine_m3.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDispatchFFNCombineM3GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, bool transB, bool weightNz, const aclTensor *out,
    const aclTensor *expertTokenNums, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerDispatchFFNCombineM3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnDispatchFFNCombineM3GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, const aclTensor *out, const aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerDispatchFFNCombineM3GetWorkspaceSize(
        x, weight1, weight2, expertId, scale1, scale2, probs, group, epRankSize, epRankId, maxOutputSize,
        false, true, out, expertTokenNums, workspaceSize, executor);
}

aclnnStatus aclnnDispatchFFNCombineM3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDispatchFFNCombineM3(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
