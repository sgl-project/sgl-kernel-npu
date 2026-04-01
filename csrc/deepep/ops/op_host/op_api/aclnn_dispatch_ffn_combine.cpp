/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_dispatch_ffn_combine.h"
#include <algorithm>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <climits>
#include "../op_host/error_log.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDispatchFFNCombineGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, bool transB, bool weightNz, const aclTensor *out,
    const aclTensor *expertTokenNums, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerDispatchFFNCombine(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnDispatchFFNCombineGetWorkspaceSize(const aclTensor *x, const aclTensor *weight1,
                                                    const aclTensor *weight2, const aclTensor *expertId,
                                                    const aclTensor *scale1, const aclTensor *scale2,
                                                    const aclTensor *probs, const char *group, int64_t epRankSize,
                                                    int64_t epRankId, int64_t maxOutputSize, const aclTensor *out,
                                                    const aclTensor *expertTokenNums, uint64_t *workspaceSize,
                                                    aclOpExecutor **executor)
{
    bool transB = false;
    bool weightNz = true;

    aclnnStatus ret = aclnnInnerDispatchFFNCombineGetWorkspaceSize(
        x, weight1, weight2, expertId, scale1, scale2, probs, group, epRankSize, epRankId, maxOutputSize, transB,
        weightNz, out, expertTokenNums, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnDispatchFFNCombine(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerDispatchFFNCombine(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif
