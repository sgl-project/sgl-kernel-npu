#include <string.h>
#include <cstdio>
#include "aclnn_moe_distribute_combine_a2.h"
#include "aclnnInner_moe_distribute_combine_a2.h"
#include "aclnn/opdev/platform.h"
// #include "aclnn_kernels/common/op_error_check.h"
// #include "opdev/op_log.h"
// #include "opdev/common_types.h"
// #include "opdev/platform.h"

// using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static constexpr size_t HCCL_GROUP_NAME_MAX = 128U;

aclnnStatus aclnnMoeDistributeCombineA2GetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *expandIdx, const aclTensor *epSendCounts,
    const aclTensor *expertScales, const aclTensor *tpSendCounts, const aclTensor *xActiveMask,
    const aclTensor *activationScale, const aclTensor *weightScale, const aclTensor *groupList,
    const aclTensor *expandScales, const aclTensor *offsetInner, const aclTensor *offsetOuter,
    const aclTensor *countOuter, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerMoeDistributeCombineA2GetWorkspaceSize(
        expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts, xActiveMask, activationScale,
        weightScale, groupList, expandScales, offsetInner, offsetOuter, countOuter, groupEp, epWorldSize, epRankId,
        moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs,
        outDtype, commQuantMode, groupListType, x, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeDistributeCombineA2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    aclnnStatus ret = aclnnInnerMoeDistributeCombineA2(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
