/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_OP_KERNEL_PROFILING_KERNEL_PROFILE_WRITER_KERNEL_H
#define DEEPEP_OP_KERNEL_PROFILING_KERNEL_PROFILE_WRITER_KERNEL_H

#include "../common/profile_protocol_common.h"
#include <kernel_operator.h>

namespace Cam {

struct ProfileWriter {
    __gm__ ProfileRecord *records{nullptr};
    bool enabled{false};
    uint32_t coreIdx{0};
    uint32_t coreType{0};
    uint32_t stageCount{0};
    uint32_t logicalCoreCount{0};
    uint32_t recordsPerLaunch{0};
    uint32_t launchId{0};
    uint64_t profileBufferBytes{0};
    ProfileStageLayout stageLayout{};

    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t launchId_, uint32_t coreType_,
                                uint32_t expectedStageCount, uint64_t profileBufferBytes_)
    {
        enabled = enable;
        if (!enabled || profileGM == nullptr) {
            enabled = false;
            return;
        }
        coreIdx = AscendC::GetBlockIdx();
        coreType = coreType_;
        launchId = launchId_;
        profileBufferBytes = profileBufferBytes_;
        stageCount = expectedStageCount;
        auto *base = reinterpret_cast<__gm__ uint8_t *>(profileGM);
        auto *header = reinterpret_cast<__gm__ ProfileHeader *>(base);
        if (header == nullptr || header->magic != PROFILE_MAGIC || header->version != PROFILE_VERSION ||
            header->cycleToUs == 0 ||
            (UnpackProfileFlags(header->flagsPacked) & static_cast<uint32_t>(PROFILE_FLAG_SESSION_BUFFER)) == 0U) {
            enabled = false;
            return;
        }
        uint64_t launchCountCapacity = UnpackProfileLaunchCapacity(header->launchCountsPacked);
        uint32_t layoutStageCount = UnpackProfileStageCount(header->layoutPacked0);
        uint32_t layoutGroupCountCapacity = UnpackProfileGroupCountCapacity(header->layoutPacked0);
        logicalCoreCount = UnpackProfileLogicalCoreCountCapacity(header->layoutPacked1);
        recordsPerLaunch = UnpackProfileRecordsPerLaunch(header->layoutPacked1);
        auto *stageLayoutGM = reinterpret_cast<__gm__ const ProfileStageLayout *>(base + sizeof(ProfileHeader));
        for (uint32_t i = 0U; i < PROFILE_RESERVED_STAGE_CAPACITY; ++i) {
            stageLayout.occurrenceCount[i] = stageLayoutGM->occurrenceCount[i];
        }
        stageLayout.stageCount = stageLayoutGM->stageCount;
        stageLayout.activeStageCapacity = stageLayoutGM->activeStageCapacity;
        for (uint32_t i = 0U; i < 7U; ++i) {
            stageLayout.reserved[i] = 0U;
        }
        uint32_t layoutActiveStageCapacity = static_cast<uint32_t>(stageLayout.activeStageCapacity);
        uint32_t layoutRecordsPerLaunch = GetProfileRecordsPerLaunch(logicalCoreCount, stageLayout);
        if (launchId >= launchCountCapacity || layoutStageCount == 0U || layoutStageCount != stageCount ||
            layoutStageCount > PROFILE_ACTIVE_STAGE_CAPACITY || layoutStageCount > PROFILE_RESERVED_STAGE_CAPACITY ||
            stageLayout.stageCount != stageCount || layoutActiveStageCapacity != PROFILE_ACTIVE_STAGE_CAPACITY ||
            layoutActiveStageCapacity > PROFILE_RESERVED_STAGE_CAPACITY || layoutGroupCountCapacity == 0U ||
            layoutGroupCountCapacity > PROFILE_MAX_GROUP_COUNT_CAPACITY ||
            logicalCoreCount != PROFILE_LOGICAL_CORE_COUNT_CAPACITY || layoutRecordsPerLaunch != recordsPerLaunch ||
            recordsPerLaunch == 0U) {
            if (coreType == PROFILE_CORE_TYPE_AIC && coreIdx == 0) {
                uint32_t flags = UnpackProfileFlags(header->flagsPacked);
                uint32_t droppedLaunches = UnpackProfileDroppedLaunches(header->flagsPacked) + 1U;
                header->flagsPacked = PackProfileFlags(flags, droppedLaunches);
            }
            enabled = false;
            return;
        }
        uint64_t alignedLaunchBytes =
            (static_cast<uint64_t>(recordsPerLaunch) * sizeof(ProfileRecord) + 63ULL) / 64ULL * 64ULL;
        uint64_t requiredBytes = GetProfileDataOffset() + (static_cast<uint64_t>(launchId) + 1ULL) * alignedLaunchBytes;
        if (requiredBytes > profileBufferBytes) {
            if (coreType == PROFILE_CORE_TYPE_AIC && coreIdx == 0) {
                uint32_t flags = UnpackProfileFlags(header->flagsPacked);
                uint32_t droppedLaunches = UnpackProfileDroppedLaunches(header->flagsPacked) + 1U;
                header->flagsPacked = PackProfileFlags(flags, droppedLaunches);
            }
            enabled = false;
            return;
        }
        uint64_t launchOffset = GetProfileLaunchOffset(GetProfileDataOffset(), recordsPerLaunch,
                                                       static_cast<uint32_t>(sizeof(ProfileRecord)), launchId);
        records = reinterpret_cast<__gm__ ProfileRecord *>(base + launchOffset);
    }

    __aicore__ inline uint64_t Now() const
    {
        return static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    __aicore__ inline void Record(uint32_t stageId, uint64_t startCycle, uint64_t endCycle) const
    {
        Record(stageId, 0U, startCycle, endCycle, MakeEmptyProfilePrivatePayloadRaw());
    }

    __aicore__ inline void Record(uint32_t stageId, uint32_t occurrenceId, uint64_t startCycle, uint64_t endCycle) const
    {
        Record(stageId, occurrenceId, startCycle, endCycle, MakeEmptyProfilePrivatePayloadRaw());
    }

    __aicore__ inline void Record(uint32_t stageId, uint64_t startCycle, uint64_t endCycle,
                                  const ProfilePrivatePayloadRaw &payload) const
    {
        Record(stageId, 0U, startCycle, endCycle, payload);
    }

    __aicore__ inline void Record(uint32_t stageId, uint32_t occurrenceId, uint64_t startCycle, uint64_t endCycle,
                                  const ProfilePrivatePayloadRaw &payload) const
    {
        if (!enabled || stageId >= stageCount) {
            return;
        }
        uint32_t stageOccurrenceCount = GetProfileStageOccurrenceCount(stageLayout, stageId);
        if (occurrenceId >= stageOccurrenceCount) {
            return;
        }
        uint32_t logicalCoreLinear = GetProfileLogicalCoreLinear(coreType, coreIdx);
        if (logicalCoreLinear == UINT32_MAX || logicalCoreLinear >= logicalCoreCount) {
            return;
        }
        uint32_t stageBase = GetProfileStageBaseOffset(stageLayout, stageId);
        uint64_t slot = (static_cast<uint64_t>(stageBase) + static_cast<uint64_t>(occurrenceId)) *
                            static_cast<uint64_t>(logicalCoreCount) +
                        static_cast<uint64_t>(logicalCoreLinear);
        if (slot >= recordsPerLaunch) {
            return;
        }
        records[slot].coreType = coreType;
        records[slot].coreIdx = coreIdx;
        records[slot].stageId = stageId;
        records[slot].occurrenceId = occurrenceId;
        records[slot].launchId = launchId;
        records[slot].startCycle = startCycle;
        records[slot].endCycle = endCycle;
        records[slot].private0 = payload.private0;
        records[slot].private1 = payload.private1;
        records[slot].private2 = payload.private2;
    }
};

}  // namespace Cam

#endif  // DEEPEP_OP_KERNEL_PROFILING_KERNEL_PROFILE_WRITER_KERNEL_H
