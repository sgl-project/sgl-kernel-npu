/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/core/profile_session.hpp"

#include <cstring>
#include <filesystem>

#include "exception.hpp"
#include "profiling/core/profile_exporter.hpp"
#include "pytorch_npu_helper.hpp"

namespace deep_ep::profiling::session {
namespace {

constexpr uint64_t kMaxBytesPerRank = 128ULL * 1024ULL * 1024ULL;

ManagerSessionState &GetManagerSession()
{
    static ManagerSessionState session;
    return session;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

bool SameLaunchConfig(const ProfileLaunchConfig &lhs, const ProfileLaunchConfig &rhs)
{
    return lhs.groupCountCapacity == rhs.groupCountCapacity &&
           std::memcmp(&lhs.stageLayout, &rhs.stageLayout, sizeof(Cam::ProfileStageLayout)) == 0;
}

Cam::ProfileHeader BuildHeader(uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                               const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    Cam::ProfileHeader header{};
    header.magic = Cam::PROFILE_MAGIC;
    header.version = Cam::PROFILE_VERSION;
    header.cycleToUs = Cam::PROFILE_CYCLE_TO_US;
    header.launchCountsPacked = Cam::PackProfileLaunchCounts(launchCountCapacity, 0U);
    header.layoutPacked0 = Cam::PackProfileLayout0(static_cast<uint16_t>(schema.stageCount),
                                                   static_cast<uint16_t>(groupCountCapacity), 1U, 2U);
    header.stageOccurrencesPacked = 0;
    header.layoutPacked1 =
        Cam::PackProfileLayout1(schema.topology.logicalCoreCount,
                                Cam::GetProfileRecordsPerLaunch(schema.topology.logicalCoreCount, stageLayout));
    header.flagsPacked = Cam::PackProfileFlags(static_cast<uint32_t>(Cam::PROFILE_FLAG_SESSION_BUFFER), 0U);
    return header;
}

void CopyHeaderToDevice(const at::Tensor &profileBuffer, const Cam::ProfileHeader &header,
                        const Cam::ProfileStageLayout &stageLayout)
{
    TORCH_CHECK(
        profileBuffer.defined() &&
            profileBuffer.numel() >= static_cast<int64_t>(sizeof(Cam::ProfileHeader) + sizeof(Cam::ProfileStageLayout)),
        "profile buffer is not large enough for profile metadata.");
    auto headerCpu =
        at::empty({static_cast<int64_t>(sizeof(header))}, at::TensorOptions().device(at::kCPU).dtype(c10::kByte));
    std::memcpy(headerCpu.data_ptr<uint8_t>(), &header, sizeof(header));
    profileBuffer.narrow(0, 0, static_cast<int64_t>(sizeof(header))).copy_(headerCpu);
    auto layoutCpu =
        at::empty({static_cast<int64_t>(sizeof(stageLayout))}, at::TensorOptions().device(at::kCPU).dtype(c10::kByte));
    std::memcpy(layoutCpu.data_ptr<uint8_t>(), &stageLayout, sizeof(stageLayout));
    profileBuffer.narrow(0, static_cast<int64_t>(sizeof(Cam::ProfileHeader)), static_cast<int64_t>(sizeof(stageLayout)))
        .copy_(layoutCpu);
}

OpProfileSession &GetExistingOpSession(const char *opKey)
{
    TORCH_CHECK(opKey != nullptr && opKey[0] != '\0', "profile op key must be non-empty.");
    auto &manager = GetManagerSession();
    auto it = manager.opSessions.find(opKey);
    TORCH_CHECK(it != manager.opSessions.end(), "profile op session not found for op=", opKey);
    return it->second;
}

}  // namespace

void ProfileTimeCalibration::Reset()
{
    valid = false;
    beginDeviceUs = 0.0;
    beginHostUs = 0.0;
    endDeviceUs = 0.0;
    endHostUs = 0.0;
    firstLaunchAlignedTsUs = 0.0;
    sharedLaunchReferenceUs = 0.0;
    launchRebaseUs = 0.0;
    mode = ProfileTimeAlignmentMode::None;
}

void OpProfileSession::Reset()
{
    registration = nullptr;
    launchConfig = ProfileLaunchConfig{};
    profileBuffer = at::Tensor();
    profileBufferBytes = 0;
    perLaunchBytes = 0;
    recordsPerLaunch = 0;
    launchCountCapacity = 0;
    capturedLaunches = 0;
    droppedLaunches = 0;
    initialized = false;
}

void ManagerSessionState::Reset()
{
    active = false;
    numProfileSkipLaunches = 0;
    numProfileActiveLaunches = 0;
    expectedLaunches = 0;
    numRanks = 0;
    profileTraceDir.clear();
    opSessions.clear();
    rankCalibrations.clear();
}

bool IsActive()
{
    return GetManagerSession().active;
}

int64_t GetExpectedLaunches()
{
    return GetManagerSession().expectedLaunches;
}

std::string GetProfileTraceDir()
{
    return GetManagerSession().profileTraceDir;
}

int64_t GetNumProfileSkipLaunches()
{
    return GetManagerSession().numProfileSkipLaunches;
}

int64_t GetNumProfileActiveLaunches()
{
    return GetManagerSession().numProfileActiveLaunches;
}

const ProfileTimeCalibration *GetTimeCalibration(int64_t rank)
{
    auto &manager = GetManagerSession();
    auto it = manager.rankCalibrations.find(rank);
    if (it == manager.rankCalibrations.end()) {
        return nullptr;
    }
    return &it->second;
}

uint32_t GetRecordsPerLaunch(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    return Cam::GetProfileRecordsPerLaunch(schema.topology.logicalCoreCount, stageLayout);
}

uint64_t GetPerLaunchBytes(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    return static_cast<uint64_t>(GetRecordsPerLaunch(stageLayout, schema)) * sizeof(Cam::ProfileRecord);
}

uint64_t GetTotalBytes(uint64_t launchCountCapacity, uint64_t perLaunchBytes)
{
    return Cam::GetProfileDataOffset() + launchCountCapacity * AlignUp(perLaunchBytes, 64ULL);
}

at::Tensor AllocateBuffer(uint64_t totalBytes, uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                          const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    auto profileBuffer = at::zeros({static_cast<int64_t>(totalBytes)}, options.dtype(c10::kByte));
    auto header = BuildHeader(launchCountCapacity, groupCountCapacity, stageLayout, schema);
    header.layoutPacked1 =
        Cam::PackProfileLayout1(schema.topology.logicalCoreCount, GetRecordsPerLaunch(stageLayout, schema));
    CopyHeaderToDevice(profileBuffer, header, stageLayout);
    return profileBuffer;
}

void Begin(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir,
           int64_t numRanks)
{
    TORCH_CHECK(numProfileSkipLaunches >= 0, "num_profile_skip_launches must be non-negative");
    TORCH_CHECK(numProfileActiveLaunches >= 0, "num_profile_active_launches must be non-negative");
    TORCH_CHECK(numRanks > 0, "num_ranks must be positive for profile session.");
    TORCH_CHECK(!GetManagerSession().active, "profile session is already active.");
    int64_t expectedLaunches = numProfileSkipLaunches + numProfileActiveLaunches;
    TORCH_CHECK(expectedLaunches > 0, "profile session needs at least one launch.");
    TORCH_CHECK(static_cast<uint64_t>(expectedLaunches) <= UINT32_MAX,
                "profile session launch count exceeds uint32 capacity.");
    auto &manager = GetManagerSession();
    manager.Reset();
    manager.active = true;
    manager.numProfileSkipLaunches = numProfileSkipLaunches;
    manager.numProfileActiveLaunches = numProfileActiveLaunches;
    manager.expectedLaunches = expectedLaunches;
    manager.numRanks = numRanks;
    manager.profileTraceDir = profileTraceDir;

    if (!profileTraceDir.empty()) {
        for (int64_t rank = 0; rank < numRanks; ++rank) {
            std::filesystem::path anchorPath =
                std::filesystem::path(profileTraceDir) / ("rank" + std::to_string(rank)) / "launch_anchor.txt";
            std::error_code ec;
            std::filesystem::remove(anchorPath, ec);
        }
    }
}

OpProfileSession &EnsureOpSession(const ProfileOpRegistration &registration, const ProfileLaunchConfig &launchConfig)
{
    TORCH_CHECK(registration.opKey != nullptr && registration.opKey[0] != '\0',
                "profile registration opKey must be non-empty.");
    TORCH_CHECK(registration.schemaProvider != nullptr, "profile registration schemaProvider is required.");
    auto &manager = GetManagerSession();
    TORCH_CHECK(manager.active, "profile session is not active.");
    TORCH_CHECK(manager.expectedLaunches > 0, "profile session launch capacity is not set.");
    TORCH_CHECK(launchConfig.groupCountCapacity >= 1U &&
                    launchConfig.groupCountCapacity <= Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY,
                "profile session group count exceeds occurrence capacity: groupCountCapacity=",
                launchConfig.groupCountCapacity, ", valid_range=[1, ", Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY, "]");

    auto [it, inserted] = manager.opSessions.try_emplace(registration.opKey);
    auto &opSession = it->second;
    if (inserted) {
        opSession.registration = &registration;
        opSession.launchConfig = launchConfig;
        opSession.launchCountCapacity = static_cast<uint32_t>(manager.expectedLaunches);
    } else {
        TORCH_CHECK(opSession.registration != nullptr &&
                        std::string(opSession.registration->opKey) == std::string(registration.opKey),
                    "profile op session registration mismatch for op=", registration.opKey);
        TORCH_CHECK(SameLaunchConfig(opSession.launchConfig, launchConfig),
                    "profile launch config changed within one session for op=", registration.opKey);
    }

    if (opSession.initialized) {
        return opSession;
    }

    const auto &schema = registration.schemaProvider();
    opSession.recordsPerLaunch = GetRecordsPerLaunch(launchConfig.stageLayout, schema);
    opSession.perLaunchBytes = GetPerLaunchBytes(launchConfig.stageLayout, schema);
    uint64_t totalBytes = GetTotalBytes(static_cast<uint64_t>(opSession.launchCountCapacity), opSession.perLaunchBytes);
    TORCH_CHECK(totalBytes <= kMaxBytesPerRank, "profile buffer would exceed the per-rank cap: totalBytes=", totalBytes,
                ", cap=", kMaxBytesPerRank, ", op=", registration.opKey);
    opSession.profileBufferBytes = totalBytes;
    opSession.profileBuffer = AllocateBuffer(totalBytes, opSession.launchCountCapacity, launchConfig.groupCountCapacity,
                                             launchConfig.stageLayout, schema);
    opSession.initialized = true;
    return opSession;
}

void IncrementCapturedLaunches(const char *opKey)
{
    ++GetExistingOpSession(opKey).capturedLaunches;
}

void IncrementDroppedLaunches(const char *opKey)
{
    ++GetExistingOpSession(opKey).droppedLaunches;
}

int64_t GetCapturedLaunches(const char *opKey)
{
    return GetExistingOpSession(opKey).capturedLaunches;
}

int64_t GetDroppedLaunches(const char *opKey)
{
    return GetExistingOpSession(opKey).droppedLaunches;
}

uint32_t GetLaunchCountCapacity(const char *opKey)
{
    return GetExistingOpSession(opKey).launchCountCapacity;
}

uint64_t GetProfileBufferBytes(const char *opKey)
{
    return GetExistingOpSession(opKey).profileBufferBytes;
}

const at::Tensor &GetProfileBuffer(const char *opKey)
{
    return GetExistingOpSession(opKey).profileBuffer;
}

void UpdateBeginCalibration(int64_t rank, double deviceUs, double hostUs)
{
    auto &calibration = GetManagerSession().rankCalibrations[rank];
    calibration.beginDeviceUs = deviceUs;
    calibration.beginHostUs = hostUs;
    calibration.endDeviceUs = 0.0;
    calibration.endHostUs = 0.0;
    calibration.mode = ProfileTimeAlignmentMode::OffsetOnly;
    calibration.valid = true;
}

void UpdateEndCalibration(int64_t rank, double deviceUs, double hostUs)
{
    auto &calibration = GetManagerSession().rankCalibrations[rank];
    calibration.endDeviceUs = deviceUs;
    calibration.endHostUs = hostUs;
    if (calibration.valid) {
        calibration.mode = ProfileTimeAlignmentMode::OffsetOnly;
    }
}

void ExportAndReset(int64_t rank)
{
    auto &manager = GetManagerSession();
    if (!manager.active) {
        return;
    }

    std::vector<exporter::ProfileTraceSource> sources;
    sources.reserve(manager.opSessions.size());
    for (auto &[opKey, opSession] : manager.opSessions) {
        if (!opSession.initialized || !opSession.profileBuffer.defined() || opSession.profileBuffer.numel() == 0) {
            continue;
        }
        if (opSession.capturedLaunches != manager.expectedLaunches || opSession.droppedLaunches > 0) {
            TORCH_WARN("profile session op=", opKey, " captured ", opSession.capturedLaunches, " launches, expected ",
                       manager.expectedLaunches, ", dropped extra launches=", opSession.droppedLaunches, ".");
        }
        const auto &schema = opSession.registration->schemaProvider();
        sources.push_back(exporter::ProfileTraceSource{
            &opSession.profileBuffer,
            &schema,
            opSession.registration->launchEventName ? opSession.registration->launchEventName() : nullptr,
            manager.numProfileSkipLaunches,
            opSession.capturedLaunches,
        });
    }

    if (sources.empty()) {
        manager.Reset();
        return;
    }
    const auto *calibration = GetTimeCalibration(rank);
    exporter::ExportAggregatedTrace(sources, rank, manager.profileTraceDir, calibration, manager.numRanks);
    manager.Reset();
}

}  // namespace deep_ep::profiling::session
