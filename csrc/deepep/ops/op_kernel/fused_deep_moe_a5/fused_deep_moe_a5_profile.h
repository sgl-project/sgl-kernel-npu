/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef FUSED_DEEP_MOE_A5_PROFILE_H
#define FUSED_DEEP_MOE_A5_PROFILE_H

#include "../profiling/kernel/profile_writer_kernel.h"
#include "../profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_payload.h"
#include "../profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_stage.h"

namespace Cam {

using FusedDeepMoeProfileStage = deep_ep::profiling::fused_deep_moe_a5::ProfileStage;

struct FusedDeepMoeProfileWriter : public ProfileWriter {
    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t launchId_, uint32_t coreType_,
                                uint64_t profileBufferBytes_)
    {
        ProfileWriter::Init(profileGM, enable, launchId_, coreType_,
                            static_cast<uint32_t>(deep_ep::profiling::fused_deep_moe_a5::ProfileStage::Count),
                            profileBufferBytes_);
    }

    __aicore__ inline void Record(deep_ep::profiling::fused_deep_moe_a5::ProfileStage stage, uint64_t startCycle,
                                  uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), startCycle, endCycle);
    }

    __aicore__ inline void Record(deep_ep::profiling::fused_deep_moe_a5::ProfileStage stage, uint32_t occurrenceId,
                                  uint64_t startCycle, uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), occurrenceId, startCycle, endCycle);
    }

    __aicore__ inline void Record(deep_ep::profiling::fused_deep_moe_a5::ProfileStage stage, uint64_t startCycle,
                                  uint64_t endCycle, const ProfilePrivatePayloadRaw &payload) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), startCycle, endCycle, payload);
    }

    __aicore__ inline void Record(deep_ep::profiling::fused_deep_moe_a5::ProfileStage stage, uint32_t occurrenceId,
                                  uint64_t startCycle, uint64_t endCycle, const ProfilePrivatePayloadRaw &payload) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), occurrenceId, startCycle, endCycle, payload);
    }
};

}  // namespace Cam

#endif  // FUSED_DEEP_MOE_A5_PROFILE_H
