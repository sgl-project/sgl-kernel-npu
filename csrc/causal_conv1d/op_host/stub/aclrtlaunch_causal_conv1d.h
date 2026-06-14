#ifndef HEADER_ACLRTLAUNCH_CAUSAL_CONV1D_H
#define HEADER_ACLRTLAUNCH_CAUSAL_CONV1D_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_causal_conv1d(uint32_t numBlocks, aclrtStream stream, void *x, void *weight,
                                              void *convStates, void *bias, void *queryStartLoc, void *cacheIndices,
                                              void *initialStateMode, void *numAcceptedTokens, void *y,
                                              void *workspace, void *tiling);
#endif
