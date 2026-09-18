#include "profiling/core/profile_runtime.hpp"

#include <chrono>
#include <mutex>

#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::runtime {
namespace {

void WarnProfilingUnsupportedOnce()
{
    static std::once_flag once;
    std::call_once(once, []() {
        TORCH_WARN(
            "DeepEP profiling timestamp alignment is disabled because aclrtEventGetTimestamp is not available in the "
            "current build environment. Profiling collection remains enabled, but device-host and cross-rank time "
            "alignment will fall back to launch-relative export.");
    });
}

bool IsProfilingSupported()
{
#if defined(DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP) && DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP
    return true;
#else
    return false;
#endif
}

struct ProfileTimeAnchor {
    uint64_t deviceSyscnt{0};
    double deviceUs{0.0};
    double hostBeforeUs{0.0};
    double hostAfterUs{0.0};
    double hostUs{0.0};
    bool valid{false};
};

double CaptureHostMonotonicUs()
{
    using Clock = std::chrono::steady_clock;
    const auto now = Clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
}

ProfileTimeAnchor CaptureCurrentStreamAnchor()
{
    ProfileTimeAnchor anchor{};
#if !(defined(DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP) && DEEPEP_HAS_ACLRT_EVENT_GET_TIMESTAMP)
    return anchor;
#else
    aclrtEvent event = nullptr;
    const auto createStatus = aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE);
    if (createStatus != ACL_SUCCESS || event == nullptr) {
        TORCH_WARN("Failed to create ACL timeline event for profiling time alignment, status=", createStatus);
        return anchor;
    }

    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const double hostBeforeUs = CaptureHostMonotonicUs();
    anchor.hostBeforeUs = hostBeforeUs;
    const auto recordStatus = aclrtRecordEvent(event, stream);
    if (recordStatus != ACL_SUCCESS) {
        TORCH_WARN("Failed to record ACL timeline event for profiling time alignment, status=", recordStatus);
        (void)aclrtDestroyEvent(event);
        return anchor;
    }

    const auto syncStatus = aclrtSynchronizeEvent(event);
    const double hostAfterUs = CaptureHostMonotonicUs();
    anchor.hostAfterUs = hostAfterUs;
    if (syncStatus != ACL_SUCCESS) {
        TORCH_WARN("Failed to synchronize ACL timeline event for profiling time alignment, status=", syncStatus);
        (void)aclrtDestroyEvent(event);
        return anchor;
    }

    uint64_t deviceSyscnt = 0;
    const auto tsStatus = aclrtEventGetTimestamp(event, &deviceSyscnt);
    (void)aclrtDestroyEvent(event);
    if (tsStatus != ACL_SUCCESS) {
        TORCH_WARN("Failed to query ACL timeline event timestamp for profiling time alignment, status=", tsStatus);
        return anchor;
    }

    anchor.deviceSyscnt = deviceSyscnt;
    // aclrtEventGetTimestamp() and exported kernel trace timestamps are expected
    // to be on the same device-side time axis here. The runtime event timestamp
    // must therefore stay in its raw device-time unit instead of being divided
    // by PROFILE_CYCLE_TO_US again, otherwise the alignment offset becomes 1000x
    // larger than expected in multi-rank exports.
    anchor.deviceUs = static_cast<double>(deviceSyscnt);
    anchor.hostUs = (hostBeforeUs + hostAfterUs) * 0.5;
    anchor.valid = true;
    return anchor;
#endif
}

}  // namespace

bool IsSessionActive()
{
    return session::IsActive();
}

std::string GetProfileTraceDir()
{
    return session::GetProfileTraceDir();
}

int64_t GetNumProfileSkipLaunches()
{
    return session::GetNumProfileSkipLaunches();
}

int64_t GetNumProfileActiveLaunches()
{
    return session::GetNumProfileActiveLaunches();
}

int64_t GetExpectedLaunches()
{
    return session::GetExpectedLaunches();
}

void BeginSession(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir,
                  int64_t numRanks)
{
    if (!IsProfilingSupported()) {
        WarnProfilingUnsupportedOnce();
    }
    session::Begin(numProfileSkipLaunches, numProfileActiveLaunches, profileTraceDir, numRanks);
}

void EndSession(int64_t rank)
{
    session::ExportAndReset(rank);
}

void CaptureSessionBeginAnchor(int64_t rank)
{
    if (!session::IsActive()) {
        return;
    }
    const auto anchor = CaptureCurrentStreamAnchor();
    if (!anchor.valid) {
        return;
    }
    session::UpdateBeginCalibration(rank, anchor.deviceUs, anchor.hostUs);
}

void CaptureSessionEndAnchor(int64_t rank)
{
    if (!session::IsActive()) {
        return;
    }
    const auto anchor = CaptureCurrentStreamAnchor();
    if (!anchor.valid) {
        return;
    }
    session::UpdateEndCalibration(rank, anchor.deviceUs, anchor.hostUs);
}

ProfileLaunchContext PrepareLaunch(const ProfileOpRegistration &registration, const ProfileLaunchConfig &launchConfig,
                                   bool profileEnable)
{
    const auto &schema = registration.schemaProvider();
    ProfileLaunchContext ctx{};
    ctx.sessionActive = session::IsActive();
    ctx.enabled = profileEnable && ctx.sessionActive;
    ctx.registration = &registration;
    if (!ctx.enabled) {
        return ctx;
    }
    session::EnsureOpSession(registration, launchConfig);
    TORCH_CHECK(session::GetProfileBuffer(registration.opKey).defined() &&
                    session::GetProfileBuffer(registration.opKey).numel() > 0,
                schema.opName ? schema.opName : "profile", " session is active but profile buffer is missing.");
    if (session::GetCapturedLaunches(registration.opKey) >= session::GetExpectedLaunches()) {
        session::IncrementDroppedLaunches(registration.opKey);
        ctx.enabled = false;
        return ctx;
    }
    ctx.launchId = session::GetCapturedLaunches(registration.opKey);
    ctx.profileBuffer = &session::GetProfileBuffer(registration.opKey);
    ctx.profileBufferBytes = static_cast<int64_t>(session::GetProfileBufferBytes(registration.opKey));
    return ctx;
}

void CompleteLaunch(const ProfileLaunchContext &ctx, int64_t rank)
{
    if (!ctx.enabled) {
        return;
    }
    TORCH_CHECK(ctx.registration != nullptr && ctx.registration->schemaProvider != nullptr,
                "profile launch context registration is missing.");
    session::IncrementCapturedLaunches(ctx.registration->opKey);
}

}  // namespace deep_ep::profiling::runtime
