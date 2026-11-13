// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#pragma once

#include <aclnn/acl_meta.h>

#include <cstdint>
#include <memory>

namespace sglang::attention {

using AclTensorPtr = std::shared_ptr<aclTensor>;
using AclIntArrayPtr = std::shared_ptr<aclIntArray>;

// ACLNN arguments prepared by the Torch adapter. Descriptors own their ACL
// handles and are captured by the queued launch.
struct RainFusionAttentionArgs {
    AclTensorPtr query;
    AclTensorPtr key;
    AclTensorPtr value;
    AclTensorPtr select_idx;
    AclTensorPtr select_num_idx;
    AclIntArrayPtr blockshape;
    AclTensorPtr attn_mask;
    AclIntArrayPtr actual_seq_qlen;
    AclIntArrayPtr actual_seq_kvlen;
    AclTensorPtr block_table;
    AclTensorPtr attention_out;
    AclTensorPtr softmax_lse;
    char *q_input_layout;
    char *kv_input_layout;
    int64_t head_num;
    int64_t mask_type;
    double scale;
    int64_t inner_precise;
    int64_t block_size;
};

// Launch the CANN built-in operator through its resolved ACLNN entry points.
void launch_rain_fusion_attention(const RainFusionAttentionArgs &args);

}  // namespace sglang::attention
