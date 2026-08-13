#include "profiling/core/profile_exporter.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#include "exception.hpp"

namespace deep_ep::profiling::exporter {
namespace {

struct TraceEventRow {
    std::string opName;
    std::string launchEventName;
    std::string stageName;
    std::string stageLabel;
    double ts_us{0.0};
    double dur_us{0.0};
    int64_t launchId{0};
    bool isWarmup{false};
    uint64_t coreType{0};
    uint64_t coreIdx{0};
    uint64_t stageId{0};
    uint64_t occurrenceId{0};
    uint64_t startCycle{0};
    uint64_t endCycle{0};
    Cam::ProfilePrivatePayloadRaw payload = Cam::MakeEmptyProfilePrivatePayloadRaw();
};

struct LaunchTraceBundle {
    std::string opName;
    std::string launchEventName;
    const ProfileSchema *schema{nullptr};
    Cam::ProfileStageLayout stageLayout{};
    uint64_t cycleToUs{Cam::PROFILE_CYCLE_TO_US};
    int64_t launchId{0};
    bool isWarmup{false};
    uint64_t minStartCycle{0};
    uint64_t maxEndCycle{0};
    std::vector<TraceEventRow> rows;
};

static bool IsValidProfileRecord(const Cam::ProfileRecord &record, uint64_t expectedLaunchId,
                                 const Cam::ProfileStageLayout &stageLayout, const ProfileSchema &schema)
{
    if (record.endCycle <= record.startCycle) {
        return false;
    }
    if (record.launchId != expectedLaunchId) {
        return false;
    }
    if (Cam::GetProfileLogicalCoreLinear(record.coreType, record.coreIdx) == UINT32_MAX) {
        return false;
    }
    if (record.stageId >= static_cast<uint64_t>(stageLayout.stageCount)) {
        return false;
    }
    if (record.occurrenceId >=
        Cam::GetProfileStageOccurrenceCount(stageLayout, static_cast<uint32_t>(record.stageId))) {
        return false;
    }
    if (schema.stageName != nullptr) {
        const char *stageName = schema.stageName(record.stageId);
        if (stageName == nullptr || stageName[0] == '\0') {
            return false;
        }
    }
    return true;
}

static std::string JsonEscape(const std::string &value)
{
    std::ostringstream oss;
    oss << '"';
    for (char c : value) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20U) {
                    oss << "\\u";
                    oss << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c)) << std::dec << std::setfill(' ');
                } else {
                    oss << c;
                }
                break;
        }
    }
    oss << '"';
    return oss.str();
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static std::string CoreTypeName(uint64_t coreType)
{
    return (coreType == Cam::PROFILE_CORE_TYPE_AIV) ? "AIV" : "AIC";
}

static std::string DefaultLaunchEventName(const ProfileSchema &schema)
{
    return std::string(schema.opName ? schema.opName : "profile") + "_launch";
}

static double GetRawLaunchTsUs(const LaunchTraceBundle &launch)
{
    return static_cast<double>(launch.minStartCycle) / static_cast<double>(launch.cycleToUs);
}

static bool CollectLaunches(const at::Tensor &profileBuffer, int64_t numProfileSkipLaunches,
                            int64_t launchCountCaptured, const ProfileSchema &schema, const char *launchEventName,
                            std::vector<LaunchTraceBundle> &launches)
{
    if (!profileBuffer.defined() || profileBuffer.numel() == 0) {
        return false;
    }

    auto profileCpu = profileBuffer.to(at::kCPU).contiguous();
    auto *base = profileCpu.data_ptr<uint8_t>();
    auto *header = reinterpret_cast<const Cam::ProfileHeader *>(base);
    if (header == nullptr || header->magic != Cam::PROFILE_MAGIC || header->version != Cam::PROFILE_VERSION ||
        header->cycleToUs == 0) {
        return false;
    }
    auto *stageLayout = reinterpret_cast<const Cam::ProfileStageLayout *>(base + sizeof(Cam::ProfileHeader));

    uint32_t launchCountCapacity = Cam::UnpackProfileLaunchCapacity(header->launchCountsPacked);
    uint32_t stageCount = Cam::UnpackProfileStageCount(header->layoutPacked0);
    uint32_t groupCountCapacity = Cam::UnpackProfileGroupCountCapacity(header->layoutPacked0);
    uint32_t logicalCoreCountCapacity = Cam::UnpackProfileLogicalCoreCountCapacity(header->layoutPacked1);
    uint32_t recordsPerLaunch = Cam::UnpackProfileRecordsPerLaunch(header->layoutPacked1);
    uint32_t expectedRecordsPerLaunch = Cam::GetProfileRecordsPerLaunch(logicalCoreCountCapacity, *stageLayout);
    if (stageCount == 0U || stageCount > Cam::PROFILE_ACTIVE_STAGE_CAPACITY ||
        stageCount > Cam::PROFILE_RESERVED_STAGE_CAPACITY || stageLayout->stageCount != stageCount ||
        stageLayout->activeStageCapacity != Cam::PROFILE_ACTIVE_STAGE_CAPACITY || groupCountCapacity == 0U ||
        groupCountCapacity > Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY ||
        logicalCoreCountCapacity != schema.topology.logicalCoreCount || recordsPerLaunch != expectedRecordsPerLaunch) {
        TORCH_WARN("Unexpected profile layout for op=", schema.opName ? schema.opName : "profile", ", skip export.");
        return false;
    }

    uint64_t perLaunchBytes = static_cast<uint64_t>(recordsPerLaunch) * sizeof(Cam::ProfileRecord);
    uint64_t requiredBytes = Cam::GetProfileDataOffset() + launchCountCapacity * AlignUp(perLaunchBytes, 64ULL);
    if (profileCpu.numel() < static_cast<int64_t>(requiredBytes)) {
        TORCH_WARN("profile buffer is smaller than expected for op=", schema.opName ? schema.opName : "profile",
                   ", skip export.");
        return false;
    }

    uint64_t captured = static_cast<uint64_t>(std::max<int64_t>(0, launchCountCaptured));
    captured = std::min<uint64_t>(captured, launchCountCapacity);
    if (captured == 0) {
        return false;
    }

    std::string opName = schema.opName ? schema.opName : "profile";
    std::string launchName = (launchEventName != nullptr && launchEventName[0] != '\0')
                                 ? std::string(launchEventName)
                                 : DefaultLaunchEventName(schema);

    launches.reserve(launches.size() + static_cast<size_t>(captured));
    for (uint64_t launchId = 0; launchId < captured; ++launchId) {
        auto *launchBase =
            base + Cam::GetProfileLaunchOffset(Cam::GetProfileDataOffset(), recordsPerLaunch,
                                               static_cast<uint32_t>(sizeof(Cam::ProfileRecord)), launchId);
        auto *records = reinterpret_cast<const Cam::ProfileRecord *>(launchBase);
        LaunchTraceBundle bundle;
        bundle.opName = opName;
        bundle.launchEventName = launchName;
        bundle.schema = &schema;
        bundle.stageLayout = *stageLayout;
        bundle.cycleToUs = header->cycleToUs;
        bundle.launchId = static_cast<int64_t>(launchId);
        bundle.isWarmup = static_cast<int64_t>(launchId) < numProfileSkipLaunches;
        bool haveRange = false;
        for (uint64_t i = 0; i < recordsPerLaunch; ++i) {
            const auto &record = records[i];
            if (!IsValidProfileRecord(record, launchId, *stageLayout, schema)) {
                continue;
            }
            TraceEventRow row;
            row.opName = opName;
            row.launchEventName = launchName;
            row.stageName = schema.stageName ? schema.stageName(record.stageId) : "unknown";
            row.stageLabel = schema.stageDisplayName
                                 ? schema.stageDisplayName(record.stageId, record.occurrenceId, *stageLayout)
                                 : std::string("unknown");
            row.ts_us = static_cast<double>(record.startCycle) / static_cast<double>(header->cycleToUs);
            row.dur_us =
                static_cast<double>(record.endCycle - record.startCycle) / static_cast<double>(header->cycleToUs);
            row.launchId = static_cast<int64_t>(record.launchId);
            row.isWarmup = static_cast<int64_t>(record.launchId) < numProfileSkipLaunches;
            row.coreType = record.coreType;
            row.coreIdx = record.coreIdx;
            row.stageId = record.stageId;
            row.occurrenceId = record.occurrenceId;
            row.startCycle = record.startCycle;
            row.endCycle = record.endCycle;
            row.payload = Cam::GetProfileRecordPayload(record);
            bundle.rows.push_back(std::move(row));
            if (!haveRange) {
                bundle.minStartCycle = record.startCycle;
                bundle.maxEndCycle = record.endCycle;
                haveRange = true;
            } else {
                bundle.minStartCycle = std::min(bundle.minStartCycle, record.startCycle);
                bundle.maxEndCycle = std::max(bundle.maxEndCycle, record.endCycle);
            }
        }
        if (!bundle.rows.empty()) {
            std::sort(bundle.rows.begin(), bundle.rows.end(), [](const TraceEventRow &lhs, const TraceEventRow &rhs) {
                if (lhs.ts_us != rhs.ts_us) {
                    return lhs.ts_us < rhs.ts_us;
                }
                if (lhs.coreType != rhs.coreType) {
                    return lhs.coreType < rhs.coreType;
                }
                if (lhs.coreIdx != rhs.coreIdx) {
                    return lhs.coreIdx < rhs.coreIdx;
                }
                if (lhs.stageId != rhs.stageId) {
                    return lhs.stageId < rhs.stageId;
                }
                return lhs.occurrenceId < rhs.occurrenceId;
            });
            launches.push_back(std::move(bundle));
        }
    }

    return !launches.empty();
}

static std::filesystem::path ResolveTracePath(const std::string &profileTraceDir, int64_t rank,
                                              const std::string &fileName)
{
    std::filesystem::path traceRoot =
        std::filesystem::path(profileTraceDir.empty() ? std::filesystem::current_path().string() : profileTraceDir);
    std::filesystem::path traceDir = traceRoot / ("rank" + std::to_string(rank));
    std::error_code ec;
    std::filesystem::create_directories(traceDir, ec);
    if (ec) {
        TORCH_WARN("Failed to create profile trace dir: ", traceDir.string(), ", error=", ec.message());
        return {};
    }
    return traceDir / fileName;
}

static double GetAlignedTsUs(double tsUs, const session::ProfileTimeCalibration *calibration)
{
    if (calibration == nullptr || !calibration->valid ||
        calibration->mode != session::ProfileTimeAlignmentMode::OffsetOnly) {
        return tsUs;
    }
    return tsUs + (calibration->beginHostUs - calibration->beginDeviceUs);
}

static double GetLaunchRebasedTsUs(double alignedTsUs, const session::ProfileTimeCalibration *calibration)
{
    if (calibration == nullptr || !calibration->valid) {
        return alignedTsUs;
    }
    return alignedTsUs + calibration->launchRebaseUs;
}

static const LaunchTraceBundle *SelectLaunchRebaseAnchor(const std::vector<LaunchTraceBundle> &launches)
{
    if (launches.empty()) {
        return nullptr;
    }
    for (const auto &launch : launches) {
        if (!launch.isWarmup) {
            return &launch;
        }
    }
    return &launches.front();
}

static void ApplyLaunchRelativeCalibration(std::vector<LaunchTraceBundle> &launches,
                                           session::ProfileTimeCalibration &effectiveCalibration)
{
    if (launches.empty()) {
        return;
    }
    const auto *anchorLaunch = SelectLaunchRebaseAnchor(launches);
    if (anchorLaunch == nullptr) {
        return;
    }
    const double firstRawLaunchTsUs = GetRawLaunchTsUs(*anchorLaunch);
    effectiveCalibration.valid = true;
    effectiveCalibration.mode = session::ProfileTimeAlignmentMode::LaunchRelativeOnly;
    effectiveCalibration.firstLaunchAlignedTsUs = firstRawLaunchTsUs;
    effectiveCalibration.sharedLaunchReferenceUs = 0.0;
    effectiveCalibration.launchRebaseUs = -firstRawLaunchTsUs;
}

static std::string TimeAlignmentModeName(session::ProfileTimeAlignmentMode mode)
{
    switch (mode) {
        case session::ProfileTimeAlignmentMode::OffsetOnly:
            return "offset_only";
        case session::ProfileTimeAlignmentMode::LaunchRelativeOnly:
            return "launch_relative_only";
        case session::ProfileTimeAlignmentMode::Linear:
            return "linear";
        case session::ProfileTimeAlignmentMode::None:
        default:
            return "none";
    }
}

static std::filesystem::path ResolveLaunchAnchorPath(const std::string &profileTraceDir, int64_t rank)
{
    std::filesystem::path traceRoot =
        std::filesystem::path(profileTraceDir.empty() ? std::filesystem::current_path().string() : profileTraceDir);
    std::filesystem::path traceDir = traceRoot / ("rank" + std::to_string(rank));
    std::error_code ec;
    std::filesystem::create_directories(traceDir, ec);
    if (ec) {
        TORCH_WARN("Failed to create launch anchor dir: ", traceDir.string(), ", error=", ec.message());
        return {};
    }
    return traceDir / "launch_anchor.txt";
}

static void PersistLaunchAnchor(const std::string &profileTraceDir, int64_t rank, double firstLaunchAlignedTsUs)
{
    std::filesystem::path anchorPath = ResolveLaunchAnchorPath(profileTraceDir, rank);
    if (anchorPath.empty()) {
        return;
    }
    std::ofstream ofs(anchorPath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        TORCH_WARN("Failed to open launch anchor file: ", anchorPath.string());
        return;
    }
    ofs << std::fixed << std::setprecision(6) << firstLaunchAlignedTsUs << "\n";
}

static bool LoadLaunchAnchor(const std::string &profileTraceDir, int64_t rank, double &firstLaunchAlignedTsUs)
{
    std::filesystem::path anchorPath = ResolveLaunchAnchorPath(profileTraceDir, rank);
    if (anchorPath.empty() || !std::filesystem::exists(anchorPath)) {
        return false;
    }
    std::ifstream ifs(anchorPath);
    if (!ifs.is_open()) {
        return false;
    }
    ifs >> firstLaunchAlignedTsUs;
    return ifs.good() || ifs.eof();
}

static double ResolveSharedLaunchReferenceUs(const std::string &profileTraceDir, int64_t numRanks,
                                             double localFirstLaunchAlignedTsUs, int64_t rank)
{
    PersistLaunchAnchor(profileTraceDir, rank, localFirstLaunchAlignedTsUs);
    std::vector<double> anchors;
    anchors.reserve(static_cast<size_t>(std::max<int64_t>(numRanks, 1)));
    constexpr int kMaxPollRounds = 50;
    constexpr auto kPollInterval = std::chrono::milliseconds(100);
    for (int poll = 0; poll < kMaxPollRounds; ++poll) {
        anchors.clear();
        for (int64_t otherRank = 0; otherRank < numRanks; ++otherRank) {
            double anchor = 0.0;
            if (LoadLaunchAnchor(profileTraceDir, otherRank, anchor)) {
                anchors.push_back(anchor);
            }
        }
        if (static_cast<int64_t>(anchors.size()) >= numRanks) {
            break;
        }
        std::this_thread::sleep_for(kPollInterval);
    }
    if (anchors.empty()) {
        return localFirstLaunchAlignedTsUs;
    }
    std::sort(anchors.begin(), anchors.end());
    const size_t mid = anchors.size() / 2;
    if ((anchors.size() & 1U) == 0U) {
        return (anchors[mid - 1] + anchors[mid]) * 0.5;
    }
    return anchors[mid];
}

static bool WriteTraceFile(const std::vector<LaunchTraceBundle> &launches, int64_t rank, const std::string &tracePath,
                           const session::ProfileTimeCalibration *calibration)
{
    if (launches.empty()) {
        return false;
    }

    std::ofstream ofs(tracePath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        TORCH_WARN("Failed to open profile trace file: ", tracePath);
        return false;
    }

    ofs << "{\n  \"traceEvents\": [\n";
    bool needComma = false;
    auto emitEvent = [&](const std::string &name, const std::string &cat, const std::string &ph, uint64_t pid,
                         uint64_t tid, double ts, double dur, const std::string &argsJson) {
        if (needComma) {
            ofs << ",\n";
        }
        needComma = true;
        ofs << "    {"
            << "\"name\":" << JsonEscape(name) << ","
            << "\"cat\":" << JsonEscape(cat) << ","
            << "\"ph\":" << JsonEscape(ph) << ","
            << "\"ts\":" << std::fixed << std::setprecision(3) << ts << ","
            << "\"pid\":" << pid << ","
            << "\"tid\":" << tid;
        if (ph == "X") {
            ofs << ",\"dur\":" << std::fixed << std::setprecision(3) << dur;
        }
        if (!argsJson.empty()) {
            ofs << ",\"args\":" << argsJson;
        }
        ofs << "}";
    };

    emitEvent("process_name", "profile", "M", static_cast<uint64_t>(rank), 0, 0.0, 0.0,
              std::string("{\"name\":") + JsonEscape("rank" + std::to_string(rank)) + "}");

    std::set<uint64_t> seenThreads;
    for (size_t sourceIdx = 0; sourceIdx < launches.size(); ++sourceIdx) {
        const auto &bundle = launches[sourceIdx];
        uint64_t launchTid = 1000000ULL + static_cast<uint64_t>(sourceIdx) * 100000ULL +
                             static_cast<uint64_t>(std::max<int64_t>(0, bundle.launchId));
        emitEvent(
            bundle.launchEventName, bundle.opName, "X", static_cast<uint64_t>(rank), launchTid,
            GetLaunchRebasedTsUs(
                GetAlignedTsUs(static_cast<double>(bundle.minStartCycle) / static_cast<double>(bundle.cycleToUs),
                               calibration),
                calibration),
            static_cast<double>(bundle.maxEndCycle - bundle.minStartCycle) / static_cast<double>(bundle.cycleToUs),
            std::string("{\"rank\":") + std::to_string(rank) + ",\"op_name\":" + JsonEscape(bundle.opName) +
                ",\"launch_id\":" + std::to_string(bundle.launchId) +
                ",\"iteration_id\":" + std::to_string(bundle.launchId) +
                ",\"is_warmup\":" + (bundle.isWarmup ? std::string("true") : std::string("false")) + "}");

        for (const auto &row : bundle.rows) {
            uint64_t tid = Cam::GetProfileLogicalCoreLinear(row.coreType, row.coreIdx);
            if (tid == UINT32_MAX) {
                continue;
            }
            if (seenThreads.insert(tid).second) {
                emitEvent("thread_name", "profile", "M", static_cast<uint64_t>(rank), tid, 0.0, 0.0,
                          std::string("{\"name\":") +
                              JsonEscape(CoreTypeName(row.coreType) + "-" + std::to_string(row.coreIdx)) + "}");
            }
        }

        for (const auto &row : bundle.rows) {
            uint64_t tid = Cam::GetProfileLogicalCoreLinear(row.coreType, row.coreIdx);
            if (tid == UINT32_MAX) {
                continue;
            }
            std::ostringstream args;
            args << "{";
            args << "\"rank\":" << rank << ",";
            args << "\"op_name\":" << JsonEscape(row.opName) << ",";
            args << "\"core_type\":" << JsonEscape(CoreTypeName(row.coreType)) << ",";
            args << "\"core_type_raw\":" << row.coreType << ",";
            args << "\"core_idx\":" << row.coreIdx << ",";
            args << "\"stage_id\":" << row.stageId << ",";
            args << "\"stage_name\":" << JsonEscape(row.stageName) << ",";
            args << "\"occurrence_id\":" << row.occurrenceId << ",";
            args << "\"stage_label\":" << JsonEscape(row.stageLabel) << ",";
            args << "\"launch_id\":" << row.launchId << ",";
            args << "\"iteration_id\":" << row.launchId << ",";
            args << "\"is_warmup\":" << (row.isWarmup ? "true" : "false") << ",";
            args << "\"start_cycle\":" << row.startCycle << ",";
            args << "\"end_cycle\":" << row.endCycle;
            if (bundle.schema != nullptr && bundle.schema->privateDataJson != nullptr) {
                Cam::ProfileRecord record{};
                record.coreType = row.coreType;
                record.coreIdx = row.coreIdx;
                record.stageId = row.stageId;
                record.occurrenceId = row.occurrenceId;
                record.launchId = static_cast<uint64_t>(std::max<int64_t>(0, row.launchId));
                record.startCycle = row.startCycle;
                record.endCycle = row.endCycle;
                Cam::SetProfileRecordPayload(record, row.payload);
                std::string privateDataJson =
                    bundle.schema->privateDataJson(row.stageId, row.occurrenceId, record, bundle.stageLayout);
                if (!privateDataJson.empty()) {
                    args << privateDataJson;
                }
            }
            args << "}";
            emitEvent(row.stageLabel, row.opName, "X", static_cast<uint64_t>(rank), tid,
                      GetLaunchRebasedTsUs(GetAlignedTsUs(row.ts_us, calibration), calibration), row.dur_us,
                      args.str());
        }
    }

    ofs << "\n  ]";
    if (calibration != nullptr && calibration->valid) {
        ofs << ",\n  \"deepep_time_alignment\": {";
        ofs << "\"mode\":" << JsonEscape(TimeAlignmentModeName(calibration->mode)) << ",";
        ofs << "\"rank\":" << rank << ",";
        ofs << "\"begin_device_us\":" << std::fixed << std::setprecision(3) << calibration->beginDeviceUs << ",";
        ofs << "\"begin_host_us\":" << std::fixed << std::setprecision(3) << calibration->beginHostUs << ",";
        ofs << "\"end_device_us\":" << std::fixed << std::setprecision(3) << calibration->endDeviceUs << ",";
        ofs << "\"end_host_us\":" << std::fixed << std::setprecision(3) << calibration->endHostUs << ",";
        ofs << "\"offset_us\":" << std::fixed << std::setprecision(3)
            << (calibration->beginHostUs - calibration->beginDeviceUs) << ",";
        ofs << "\"first_launch_aligned_ts_us\":" << std::fixed << std::setprecision(3)
            << calibration->firstLaunchAlignedTsUs << ",";
        ofs << "\"shared_launch_reference_us\":" << std::fixed << std::setprecision(3)
            << calibration->sharedLaunchReferenceUs << ",";
        ofs << "\"launch_rebase_us\":" << std::fixed << std::setprecision(3) << calibration->launchRebaseUs << ",";
        ofs << "\"cross_rank_alignment\":"
            << ((calibration->mode == session::ProfileTimeAlignmentMode::OffsetOnly) ? "true" : "false");
        ofs << "}";
    }
    ofs << "\n}\n";
    return true;
}

}  // namespace

void ExportAggregatedTrace(const std::vector<ProfileTraceSource> &sources, int64_t rank,
                           const std::string &profileTraceDir, const session::ProfileTimeCalibration *calibration,
                           int64_t numRanks)
{
    std::vector<LaunchTraceBundle> launches;
    for (const auto &source : sources) {
        if (source.profileBuffer == nullptr || source.schema == nullptr) {
            continue;
        }
        CollectLaunches(*source.profileBuffer, source.numProfileSkipLaunches, source.launchCountCaptured,
                        *source.schema, source.launchEventName, launches);
    }
    if (launches.empty()) {
        return;
    }

    std::sort(launches.begin(), launches.end(), [](const LaunchTraceBundle &lhs, const LaunchTraceBundle &rhs) {
        if (lhs.minStartCycle != rhs.minStartCycle) {
            return lhs.minStartCycle < rhs.minStartCycle;
        }
        if (lhs.opName != rhs.opName) {
            return lhs.opName < rhs.opName;
        }
        return lhs.launchId < rhs.launchId;
    });

    std::filesystem::path tracePath = ResolveTracePath(profileTraceDir, rank, "trace_view.json");
    if (tracePath.empty()) {
        return;
    }
    session::ProfileTimeCalibration effectiveCalibration =
        (calibration != nullptr) ? *calibration : session::ProfileTimeCalibration{};
    if (!launches.empty()) {
        if (effectiveCalibration.valid && effectiveCalibration.mode == session::ProfileTimeAlignmentMode::OffsetOnly &&
            numRanks > 1) {
            const auto *anchorLaunch = SelectLaunchRebaseAnchor(launches);
            const double firstRawLaunchTsUs = GetRawLaunchTsUs(*anchorLaunch);
            effectiveCalibration.firstLaunchAlignedTsUs = GetAlignedTsUs(firstRawLaunchTsUs, &effectiveCalibration);
            effectiveCalibration.sharedLaunchReferenceUs = ResolveSharedLaunchReferenceUs(
                profileTraceDir, numRanks, effectiveCalibration.firstLaunchAlignedTsUs, rank);
            effectiveCalibration.launchRebaseUs =
                effectiveCalibration.sharedLaunchReferenceUs - effectiveCalibration.firstLaunchAlignedTsUs;
        } else if (!effectiveCalibration.valid) {
            ApplyLaunchRelativeCalibration(launches, effectiveCalibration);
        }
    }
    WriteTraceFile(launches, rank, tracePath.string(), &effectiveCalibration);
}

}  // namespace deep_ep::profiling::exporter
