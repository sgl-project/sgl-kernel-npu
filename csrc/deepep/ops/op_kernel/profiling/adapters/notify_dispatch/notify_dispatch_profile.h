#ifndef NOTIFY_DISPATCH_PROFILE_H
#define NOTIFY_DISPATCH_PROFILE_H

#include "../../kernel/profile_writer_kernel.h"
#include "notify_dispatch_profile_stage.h"

namespace Cam {

using NotifyDispatchProfileStage = deep_ep::profiling::notify_dispatch::ProfileStage;

// notify_dispatch 专属的 ProfileWriter 封装（AIV-only，无 payload）。
struct NotifyDispatchProfileWriter : public ProfileWriter {
    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t launchId_, uint32_t coreType_,
                                uint64_t profileBufferBytes_)
    {
        ProfileWriter::Init(profileGM, enable, launchId_, coreType_,
                            static_cast<uint32_t>(deep_ep::profiling::notify_dispatch::ProfileStage::Count),
                            profileBufferBytes_);
    }

    __aicore__ inline void Record(NotifyDispatchProfileStage stage, uint64_t startCycle, uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), startCycle, endCycle);
    }

    __aicore__ inline void Record(NotifyDispatchProfileStage stage, uint32_t occurrenceId, uint64_t startCycle,
                                  uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), occurrenceId, startCycle, endCycle);
    }
};

}  // namespace Cam

#endif  // NOTIFY_DISPATCH_PROFILE_H