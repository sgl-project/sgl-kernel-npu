#ifndef SGL_KERNEL_NPU_ACLRTLAUNCH_COMPRESSOR_H
#define SGL_KERNEL_NPU_ACLRTLAUNCH_COMPRESSOR_H

#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_compressor(uint32_t numBlocks, aclrtStream stream, void *x, void *wKv, void *wGate,
                                           void *stateCache, void *ape, void *normWeight, void *ropeSin,
                                           void *ropeCos, void *stateBlockTable, void *cuSeqlens, void *seqUsed,
                                           void *startPos, void *cmpKvOut, void *stateCacheOut, void *workspace,
                                           void *tiling);

#endif  // SGL_KERNEL_NPU_ACLRTLAUNCH_COMPRESSOR_H
