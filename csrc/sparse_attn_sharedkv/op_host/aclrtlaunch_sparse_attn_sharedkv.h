#ifndef SGL_KERNEL_NPU_ACLRTLAUNCH_SPARSE_ATTN_SHAREDKV_H
#define SGL_KERNEL_NPU_ACLRTLAUNCH_SPARSE_ATTN_SHAREDKV_H

#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_sparse_attn_sharedkv(uint32_t numBlocks, aclrtStream stream, void *query, void *oriKV,
                                                     void *cmpKV, void *oriSparseIndices, void *cmpSparseIndices,
                                                     void *oriBlockTable, void *cmpBlockTable, void *cuSeqlensQ,
                                                     void *cuSeqlensOriKv, void *cuSeqlensCmpKv, void *seqUsedQ,
                                                     void *seqUsedKV, void *sinks, void *metadata, void *attentionOut,
                                                     void *softmaxLse, void *workspace, void *tiling);

#endif
