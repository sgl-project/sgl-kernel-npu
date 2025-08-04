#ifndef COMM_GROUP_H
#define COMM_GROUP_H

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;
constexpr uint32_t PADDING_COUNT = 5;
constexpr uint32_t LINK_TIMEOUT_COUNT = 8;
constexpr uint32_t MULTI_QP_THRESHOLD = 512;
constexpr uint32_t COMM_ID_LEN = 128;
constexpr uint32_t RESERVE_LEN = 16;
constexpr uint32_t ZERO_COPY_PTR_COUNT = 16;
constexpr uint32_t ZERO_COPY_DEVICE_COUNT = 16;

struct HcclSignalInfo {
    uint64_t resId;  // resId, eventid for event, notifyid for notify
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;       // Physical cqId
    uint32_t logicCqids;  // Logical cqId
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;                                  // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;      // RankID in the communication domain
    uint32_t userRankSize;  // Number of Ranks in the communication domain
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;  // Number of Devices in each Module
    uint32_t superPodNum;              // Total number of super Pods in the cluster
    uint32_t devicePhyId;
    uint32_t topoType;  // TopoType
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;                      // Pointer to niclist array
    uint64_t complanRankLength;            // Bytes occupied by complanRank
    uint64_t complanRank;                  // Pointer
    uint64_t bridgeRankNum;                // Number of bridgeRanks
    uint64_t bridgeRank;                   // Pointer
    uint64_t serverAndsuperPodRankLength;  // Bytes occupied by serverAndsuperPodRank
    uint64_t serverAndsuperPodRank;        // Pointer
};

struct HcclOpConfig {
    uint8_t deterministic;  // Deterministic computation switch
    uint8_t retryEnable;    // Whether to retry
    uint8_t highPerfEnable;
    uint8_t padding[PADDING_COUNT];  // Size needs to be 64B aligned, reduce padding when adding parameters
    uint8_t linkTimeOut[LINK_TIMEOUT_COUNT];  // Send timeout duration
    uint64_t notifyWaitTime;  // Timeout duration, same as HCCL_EXEC_TIMEOUT
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false;  // Enable RDMA switch
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = MULTI_QP_THRESHOLD;  // Minimum threshold for data sharing per QP in multi_QP
};

struct HcclMC2WorkSpace {
    uint64_t workSpace;
    uint64_t workSpaceSize;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HDCommunicateParams {
    uint64_t hostAddr{0};
    uint64_t deviceAddr{0};
    uint64_t readCacheAddr{0};
    uint32_t devMemSize{0};
    uint32_t buffLen{0};
    uint32_t flag{0};
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParam {
    // Local resources
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;  // usrrankid
    uint32_t rankSize;        // Total number of Ranks in the communication domain
    uint64_t winSize;  // Size of each win, 0 in static graph, non-zero if there is dynamic graph in the communication domain
    uint64_t localWindowsIn;   // Invalid value if all F
    uint64_t localWindowsOut;  // Invalid value if all F
    char hcomId[COMM_ID_LEN];
    // AI Core identifies remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;   // Start position of HcclRankRelationRes
    uint32_t rWinOffset;  // Size of HcclRemoteRes
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // External configuration parameters
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[RESERVE_LEN];
    uint32_t notifysize;                         // Used in RDMA, 4B for 910B/910_93, 8B for other chips
    uint32_t remoteResNum;                       // Number of valid remoteRes
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];  // Array pointer to HcclRankRelationResV2, index is remoteUserRankId

    // Communication retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    // Used in zero-copy
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[ZERO_COPY_PTR_COUNT];  // Save the input and output memory addresses of each peer in the collection communication
    uint32_t zeroCopyDevicePhyId[ZERO_COPY_DEVICE_COUNT];  // Save the physical card Id corresponding to each rank

    bool utraceStatusFlag;
};

#endif