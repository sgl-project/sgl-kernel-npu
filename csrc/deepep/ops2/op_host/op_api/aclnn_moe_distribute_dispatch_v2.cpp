/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string.h>
#include <cstdio>

#include "aclnn_moe_distribute_dispatch_v2.h"
#include "aclnnInner_moe_distribute_dispatch_v2.h"
#include "aclnn/opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, char *commAlg,
    aclTensor *expandXOut, aclTensor *dynamicScalesOut, aclTensor *assistInfoForCombineOut,
    aclTensor *expertTokenNumsOut, aclTensor *epRecvCountsOut, aclTensor *tpRecvCountsOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, nullptr, groupEp, epWorldSize, epRankId, moeExpertNum, "",
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs,
        expertTokenNumsType, commAlg, 0, 0, 0, expandXOut, dynamicScalesOut, assistInfoForCombineOut,
        expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeDispatchV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
