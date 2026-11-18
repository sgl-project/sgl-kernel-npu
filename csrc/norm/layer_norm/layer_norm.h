// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#pragma once

#include <aclnn/acl_meta.h>

#include <cstdint>
#include <memory>

namespace sglang::norm {

using AclTensorPtr = std::shared_ptr<aclTensor>;
using AclIntArrayPtr = std::shared_ptr<aclIntArray>;

// ACLNN arguments prepared by the Torch adapter. Descriptors own their ACL
// handles and are captured by the queued launch.
struct LayerNormArgs {
    AclTensorPtr input;
    AclIntArrayPtr normalized_shape;
    AclTensorPtr weight;
    AclTensorPtr bias;
    double eps;
    AclTensorPtr out;
    AclTensorPtr mean;
    AclTensorPtr rstd;
    int64_t impl_mode;
};

// Launch the CANN built-in operator through its resolved ACLNN entry points.
void launch_layer_norm(const LayerNormArgs &args);

}  // namespace sglang::norm
