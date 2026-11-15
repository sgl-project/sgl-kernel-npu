/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE-CANN in the root of the software repository for the full text of the License.
 */

#pragma once

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSglAddExampleGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnSglAddExample(void *workspace, uint64_t workspaceSize,
                                                                      aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
