#ifndef SHMEM_NOTIFY_DISPATCH_TILING_H
#define SHMEM_NOTIFY_DISPATCH_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct ShmemNotifyDispatchInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t localRankSize;
    uint32_t localRankId;
    uint32_t sendCount;
    uint32_t topkNum;
    uint32_t aivNum;
    uint64_t totalUbSize;
};

struct ShmemNotifyDispatchTilingData {
    ShmemNotifyDispatchInfo notifyDispatchInfo;
    uint64_t shmemPtr;  // shmem symmetric point
};

#endif
