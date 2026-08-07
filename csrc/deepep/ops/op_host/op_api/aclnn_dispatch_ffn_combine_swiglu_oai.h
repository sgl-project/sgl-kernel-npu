#ifndef DISPATCH_FFN_COMBINE_SWIGLU_OAI_H
#define DISPATCH_FFN_COMBINE_SWIGLU_OAI_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineSwiGluOAIGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertId,
    const aclTensor *scale1, const aclTensor *scale2, const aclTensor *probs, const char *group, int64_t epRankSize,
    int64_t epRankId, int64_t maxOutputSize, int64_t activationType, float activationAlpha, float gateClampMax,
    float upClampMin, float upClampMax, float upAdd, const aclTensor *out, const aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineSwiGluOAI(void *workspace,
                                                                                    uint64_t workspaceSize,
                                                                                    aclOpExecutor *executor,
                                                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // DISPATCH_FFN_COMBINE_SWIGLU_OAI_H
