/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 1.0.
 */
#ifndef ACLNN_DISPATCH_FFN_COMBINE_M3_H
#define ACLNN_DISPATCH_FFN_COMBINE_M3_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineM3GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, const aclTensor *out, const aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineM3(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor,
                                                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DISPATCH_FFN_COMBINE_M3_H
