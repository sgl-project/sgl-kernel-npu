#include "aclnn_dispatch_ffn_combine_swiglu_oai.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDispatchFFNCombineSwiGluOAIGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, bool transB, bool weightNz, int64_t activationType, float activationAlpha,
    float gateClampMax, float upClampMin, float upClampMax, float upAdd, const aclTensor *out,
    const aclTensor *expertTokenNums, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerDispatchFFNCombineSwiGluOAI(void *workspace, uint64_t workspaceSize,
                                                         aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnDispatchFFNCombineSwiGluOAIGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, int64_t activationType, float activationAlpha, float gateClampMax,
    float upClampMin, float upClampMax, float upAdd, const aclTensor *out, const aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerDispatchFFNCombineSwiGluOAIGetWorkspaceSize(
        x, weight1, weight2, expertId, scale1, scale2, probs, group, epRankSize, epRankId, maxOutputSize, false, true,
        activationType, activationAlpha, gateClampMax, upClampMin, upClampMax, upAdd, out, expertTokenNums,
        workspaceSize, executor);
}

aclnnStatus aclnnDispatchFFNCombineSwiGluOAI(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDispatchFFNCombineSwiGluOAI(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
