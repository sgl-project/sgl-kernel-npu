#ifndef DEEPEP_OPS_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_CORE_H
#define DEEPEP_OPS_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_CORE_H

#ifndef DEEPEP_PROFILE_INLINE
#error "DEEPEP_PROFILE_INLINE must be defined before including profile_protocol_common_core.h"
#endif

namespace Cam {

// Current runtime contract:
//   groupCountCapacity: [1, 64]
//   active stage count: [1, 16]
//   logical cores: 36 AIC + 72 AIV = 108
// Reserved protocol capacity:
//   the metadata table has 16 stage slots.
//   Adding stages within 1..16 only requires extending schema/export names.
// Record ABI:
//   PROFILE_VERSION == 5 extends ProfileRecord with three private u64 payload slots
//   and pads the storage stride to 128B to avoid adjacent records sharing a cache line.
//   private0[7:0]   : valid tag
//   private0[15:8]  : payload format / version
//   private0[63:16] : stage-defined
//   private1/private2: stage-defined
//   If the record ABI changes again, bump PROFILE_VERSION.
constexpr uint64_t PROFILE_MAGIC = 0x46444D5035413031ULL;  // FDMP5A01
constexpr uint64_t PROFILE_VERSION = 5;
constexpr uint64_t PROFILE_CYCLE_TO_US = 1000;
constexpr uint64_t PROFILE_FLAG_SESSION_BUFFER = 0x1ULL;
constexpr uint64_t PROFILE_CORE_TYPE_AIC = 1ULL;
constexpr uint64_t PROFILE_CORE_TYPE_AIV = 2ULL;
constexpr uint32_t PROFILE_AIC_COUNT_CAPACITY = 36U;
constexpr uint32_t PROFILE_AIV_COUNT_CAPACITY = 72U;
constexpr uint32_t PROFILE_LOGICAL_CORE_COUNT_CAPACITY = PROFILE_AIC_COUNT_CAPACITY + PROFILE_AIV_COUNT_CAPACITY;
constexpr uint32_t PROFILE_ACTIVE_STAGE_CAPACITY = 16U;
constexpr uint32_t PROFILE_RESERVED_STAGE_CAPACITY = 16U;
constexpr uint32_t PROFILE_MAX_GROUP_COUNT_CAPACITY = 64U;
constexpr uint32_t PROFILE_PRIVATE_PAYLOAD_WORD_COUNT = 3U;
constexpr uint8_t PROFILE_PRIVATE_DATA_INVALID = 0U;
constexpr uint8_t PROFILE_PRIVATE_DATA_VALID = 1U;

static_assert(PROFILE_ACTIVE_STAGE_CAPACITY <= PROFILE_RESERVED_STAGE_CAPACITY,
              "active stage capacity must fit in the reserved metadata table");

DEEPEP_PROFILE_INLINE constexpr uint64_t PackProfileLaunchCounts(uint32_t launchCountCapacity,
                                                                 uint32_t launchCountCaptured)
{
    return (static_cast<uint64_t>(launchCountCapacity) << 32) | static_cast<uint64_t>(launchCountCaptured);
}

DEEPEP_PROFILE_INLINE constexpr uint64_t PackProfileLayout0(uint16_t stageCount, uint16_t groupCountCapacity,
                                                            uint16_t aicPerGroup, uint16_t aivPerGroup)
{
    return (static_cast<uint64_t>(stageCount) << 48) | (static_cast<uint64_t>(groupCountCapacity) << 32) |
           (static_cast<uint64_t>(aicPerGroup) << 16) | static_cast<uint64_t>(aivPerGroup);
}

DEEPEP_PROFILE_INLINE constexpr uint64_t PackProfileLayout1(uint32_t logicalCoreCountCapacity,
                                                            uint32_t recordsPerLaunch)
{
    return (static_cast<uint64_t>(logicalCoreCountCapacity) << 32) | static_cast<uint64_t>(recordsPerLaunch);
}

DEEPEP_PROFILE_INLINE constexpr uint64_t PackProfileFlags(uint32_t flags, uint32_t droppedLaunches)
{
    return (static_cast<uint64_t>(droppedLaunches) << 32) | static_cast<uint64_t>(flags);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t UnpackProfileLaunchCapacity(uint64_t launchCountsPacked)
{
    return static_cast<uint32_t>(launchCountsPacked >> 32);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t UnpackProfileLaunchCaptured(uint64_t launchCountsPacked)
{
    return static_cast<uint32_t>(launchCountsPacked & 0xFFFFFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint16_t UnpackProfileStageCount(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>((layoutPacked0 >> 48) & 0xFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint16_t UnpackProfileGroupCountCapacity(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>((layoutPacked0 >> 32) & 0xFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint16_t UnpackProfileAicPerGroup(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>((layoutPacked0 >> 16) & 0xFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint16_t UnpackProfileAivPerGroup(uint64_t layoutPacked0)
{
    return static_cast<uint16_t>(layoutPacked0 & 0xFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t UnpackProfileLogicalCoreCountCapacity(uint64_t layoutPacked1)
{
    return static_cast<uint32_t>(layoutPacked1 >> 32);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t UnpackProfileRecordsPerLaunch(uint64_t layoutPacked1)
{
    return static_cast<uint32_t>(layoutPacked1 & 0xFFFFFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t UnpackProfileFlags(uint64_t flagsPacked)
{
    return static_cast<uint32_t>(flagsPacked & 0xFFFFFFFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t UnpackProfileDroppedLaunches(uint64_t flagsPacked)
{
    return static_cast<uint32_t>(flagsPacked >> 32);
}

struct ProfileStageLayout {
    uint16_t occurrenceCount[PROFILE_RESERVED_STAGE_CAPACITY];
    uint16_t stageCount;
    uint16_t activeStageCapacity;
    uint32_t reserved[7];
};

DEEPEP_PROFILE_INLINE constexpr bool SetProfileStageOccurrenceCount(ProfileStageLayout &layout, uint32_t stageId,
                                                                    uint32_t count)
{
    if (stageId >= PROFILE_RESERVED_STAGE_CAPACITY || count > PROFILE_MAX_GROUP_COUNT_CAPACITY) {
        return false;
    }
    layout.occurrenceCount[stageId] = static_cast<uint16_t>(count);
    return true;
}

DEEPEP_PROFILE_INLINE constexpr uint32_t GetProfileStageOccurrenceCount(const ProfileStageLayout &layout,
                                                                        uint32_t stageId)
{
    if (stageId >= PROFILE_RESERVED_STAGE_CAPACITY || stageId >= static_cast<uint32_t>(layout.stageCount)) {
        return 0U;
    }
    return static_cast<uint32_t>(layout.occurrenceCount[stageId]);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t GetProfileStageBaseOffset(const ProfileStageLayout &layout, uint32_t stageId)
{
    if (stageId >= PROFILE_RESERVED_STAGE_CAPACITY || stageId >= static_cast<uint32_t>(layout.stageCount)) {
        return 0U;
    }
    uint32_t base = 0U;
    for (uint32_t i = 0U; i < stageId; ++i) {
        base += static_cast<uint32_t>(layout.occurrenceCount[i]);
    }
    return base;
}

DEEPEP_PROFILE_INLINE constexpr uint32_t GetProfileTotalOccurrences(const ProfileStageLayout &layout)
{
    uint32_t total = 0U;
    uint32_t stageCount = static_cast<uint32_t>(layout.stageCount);
    if (stageCount > PROFILE_RESERVED_STAGE_CAPACITY) {
        return 0U;
    }
    for (uint32_t i = 0U; i < stageCount; ++i) {
        total += static_cast<uint32_t>(layout.occurrenceCount[i]);
    }
    return total;
}

DEEPEP_PROFILE_INLINE constexpr uint32_t GetProfileRecordsPerLaunch(uint32_t logicalCoreCount,
                                                                    const ProfileStageLayout &layout)
{
    return logicalCoreCount * GetProfileTotalOccurrences(layout);
}

DEEPEP_PROFILE_INLINE constexpr uint64_t GetProfileLaunchOffset(uint32_t dataOffset, uint32_t recordsPerLaunch,
                                                                uint32_t recordBytes, uint64_t launchId)
{
    return static_cast<uint64_t>(dataOffset) +
           launchId * static_cast<uint64_t>(recordsPerLaunch) * static_cast<uint64_t>(recordBytes);
}

DEEPEP_PROFILE_INLINE constexpr uint32_t GetProfileLogicalCoreLinear(uint64_t coreType, uint64_t coreIdx)
{
    if (coreType == PROFILE_CORE_TYPE_AIC) {
        return (coreIdx < PROFILE_AIC_COUNT_CAPACITY) ? static_cast<uint32_t>(coreIdx) : UINT32_MAX;
    }
    if (coreType == PROFILE_CORE_TYPE_AIV) {
        return (coreIdx < PROFILE_AIV_COUNT_CAPACITY) ? (PROFILE_AIC_COUNT_CAPACITY + static_cast<uint32_t>(coreIdx))
                                                      : UINT32_MAX;
    }
    return UINT32_MAX;
}

DEEPEP_PROFILE_INLINE constexpr uint64_t PackProfilePrivate0(uint8_t validTag, uint8_t formatId)
{
    return static_cast<uint64_t>(validTag) | (static_cast<uint64_t>(formatId) << 8);
}

DEEPEP_PROFILE_INLINE constexpr uint8_t GetProfilePrivateValidTag(uint64_t private0)
{
    return static_cast<uint8_t>(private0 & 0xFFULL);
}

DEEPEP_PROFILE_INLINE constexpr uint8_t GetProfilePrivateFormatId(uint64_t private0)
{
    return static_cast<uint8_t>((private0 >> 8) & 0xFFULL);
}

union ProfilePrivatePayloadRaw {
    struct {
        uint64_t private0;
        uint64_t private1;
        uint64_t private2;
    };
    uint64_t words[PROFILE_PRIVATE_PAYLOAD_WORD_COUNT];
};

static_assert(sizeof(ProfilePrivatePayloadRaw) == 24, "Unexpected private payload raw size");

DEEPEP_PROFILE_INLINE constexpr ProfilePrivatePayloadRaw MakeProfilePrivatePayloadRaw(uint64_t private0,
                                                                                      uint64_t private1,
                                                                                      uint64_t private2)
{
    return ProfilePrivatePayloadRaw{{private0, private1, private2}};
}

DEEPEP_PROFILE_INLINE constexpr ProfilePrivatePayloadRaw MakeEmptyProfilePrivatePayloadRaw()
{
    return ProfilePrivatePayloadRaw{{0ULL, 0ULL, 0ULL}};
}

struct ProfileHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t cycleToUs;
    uint64_t launchCountsPacked;
    uint64_t layoutPacked0;
    uint64_t layoutPacked1;
    uint64_t stageOccurrencesPacked;  // Reserved for pre-v3 readers; must not be used by v3 code.
    uint64_t flagsPacked;
};

DEEPEP_PROFILE_INLINE constexpr uint32_t GetProfileDataOffset()
{
    constexpr uint32_t raw =
        static_cast<uint32_t>(sizeof(ProfileHeader)) + static_cast<uint32_t>(sizeof(ProfileStageLayout));
    return (raw + 63U) / 64U * 64U;
}

struct alignas(64) ProfileRecord {
    uint64_t coreType;
    uint64_t coreIdx;
    uint64_t stageId;
    uint64_t occurrenceId;
    uint64_t launchId;
    uint64_t startCycle;
    uint64_t endCycle;
    union {
        ProfilePrivatePayloadRaw payload;
        struct {
            uint64_t private0;
            uint64_t private1;
            uint64_t private2;
        };
    };
    uint64_t reserved0;
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint64_t reserved4;
    uint64_t reserved5;
};

DEEPEP_PROFILE_INLINE constexpr ProfilePrivatePayloadRaw GetProfileRecordPayload(const ProfileRecord &record)
{
    return record.payload;
}

static_assert(sizeof(ProfileHeader) == 64, "Unexpected profile header size");
static_assert(sizeof(ProfileStageLayout) == 64, "Unexpected profile stage layout size");
static_assert(sizeof(ProfileRecord) == 128, "Unexpected profile record size");

}  // namespace Cam

#endif  // DEEPEP_OPS_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_CORE_H
