#ifndef SGL_KERNEL_NPU_ASSIGN_CACHE_H
#define SGL_KERNEL_NPU_ASSIGN_CACHE_H

#include "aclrtlaunch_assign_cache_op.h"

namespace sglang {
namespace npu_kernel {

bool RunCustomAssign(at::Tensor &dstTensor, const at::Tensor &srcTensor,
    const at::Tensor &dstStartIdx, const at::Tensor &dstEndIdx,
    const at::Tensor &srcStartIdx, const at::Tensor &srcEndIdx
);
}
}

#endif  //SGL_KERNEL_NPU_ASSIGN_CACHE_H
