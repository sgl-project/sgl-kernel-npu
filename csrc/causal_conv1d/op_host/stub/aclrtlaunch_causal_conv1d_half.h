#ifndef HEADER_ACLRTLAUNCH_CAUSAL_CONV1D_HALF_H
#define HEADER_ACLRTLAUNCH_CAUSAL_CONV1D_HALF_H

#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_causal_conv1d_half(uint32_t numBlocks, aclrtStream stream, void *x, void *weight,
                                                   void *bias, void *conv_states, void *query_start_loc,
                                                   void *cache_indices, void *has_initial_state, void *y,
                                                   void *workspace, void *tiling);

#endif
