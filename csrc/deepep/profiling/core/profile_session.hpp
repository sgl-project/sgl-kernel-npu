#ifndef DEEPEP_PROFILING_CORE_PROFILE_SESSION_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_SESSION_HPP

#include <cstdint>
#include <map>
#include <string>

#include <torch/types.h>

#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::session {

enum class ProfileTimeAlignmentMode {
    None = 0,
    OffsetOnly = 1,
    Linear = 2,
    LaunchRelativeOnly = 3,
};

struct ProfileTimeCalibration {
    bool valid{false};
    double beginDeviceUs{0.0};
    double beginHostUs{0.0};
    double endDeviceUs{0.0};
    double endHostUs{0.0};
    double firstLaunchAlignedTsUs{0.0};
    double sharedLaunchReferenceUs{0.0};
    double launchRebaseUs{0.0};
    ProfileTimeAlignmentMode mode{ProfileTimeAlignmentMode::None};

    void Reset();
};

struct OpProfileSession {
    const ProfileOpRegistration *registration{nullptr};
    ProfileLaunchConfig launchConfig{};
    at::Tensor profileBuffer;
    uint64_t profileBufferBytes{0};
    uint64_t perLaunchBytes{0};
    uint32_t recordsPerLaunch{0};
    uint32_t launchCountCapacity{0};
    int64_t capturedLaunches{0};
    int64_t droppedLaunches{0};
    bool initialized{false};

    void Reset();
};

struct ManagerSessionState {
    bool active{false};
    int64_t numProfileSkipLaunches{0};
    int64_t numProfileActiveLaunches{0};
    int64_t expectedLaunches{0};
    int64_t numRanks{0};
    std::string profileTraceDir;
    std::map<std::string, OpProfileSession> opSessions;
    std::map<int64_t, ProfileTimeCalibration> rankCalibrations;

    void Reset();
};

bool IsActive();
int64_t GetExpectedLaunches();
std::string GetProfileTraceDir();
int64_t GetNumProfileSkipLaunches();
int64_t GetNumProfileActiveLaunches();
const ProfileTimeCalibration *GetTimeCalibration(int64_t rank);

uint32_t GetRecordsPerLaunch(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);
uint64_t GetPerLaunchBytes(const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);
uint64_t GetTotalBytes(uint64_t launchCountCapacity, uint64_t perLaunchBytes);
at::Tensor AllocateBuffer(uint64_t totalBytes, uint32_t launchCountCapacity, uint32_t groupCountCapacity,
                          const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema);

void Begin(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir,
           int64_t numRanks);
OpProfileSession &EnsureOpSession(const ProfileOpRegistration &registration, const ProfileLaunchConfig &launchConfig);
void IncrementCapturedLaunches(const char *opKey);
void IncrementDroppedLaunches(const char *opKey);
int64_t GetCapturedLaunches(const char *opKey);
int64_t GetDroppedLaunches(const char *opKey);
uint32_t GetLaunchCountCapacity(const char *opKey);
uint64_t GetProfileBufferBytes(const char *opKey);
const at::Tensor &GetProfileBuffer(const char *opKey);
void UpdateBeginCalibration(int64_t rank, double deviceUs, double hostUs);
void UpdateEndCalibration(int64_t rank, double deviceUs, double hostUs);
void ExportAndReset(int64_t rank);

}  // namespace deep_ep::profiling::session

#endif  // DEEPEP_PROFILING_CORE_PROFILE_SESSION_HPP
