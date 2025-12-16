/*!
 * \file moe_distribute_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

/* system tick: 50MHz */
#define CAL_US(tick) (((tick) * 2) / 100)

/* performance macro */
// #define USE_256_TO_1__
#ifdef USE_256_TO_1__
#pragma message("use 256 to 1")
#else
#define USE_FOR_OPT__
#define DISPATCH_USE_WRITE_SHUFFLE__
#define USE_TOKEN_COUNT_SPLIT__
#define USE_ONE_CORE_WAIT__

#ifdef USE_ONE_CORE_WAIT__
#pragma message("use one core wait")

//  #define USE_ONE_CORE_GETCUMSUM__
#endif
#ifdef USE_FOR_OPT__
#pragma message("use for optimization")
#define FOR_OPT_MAX_BS__ 64
#define FOR_OPT_MAX_MOE_RANK__ 256
#endif
// #define COMBINE_USE_DYNAMIC_QUANT
#define OPT_RANK_OFFSET 512
#define USE_WRITE_SHUFFLE
#endif

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

struct HcclSignalInfo {
    uint64_t resId;
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
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;  // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;      // RankID
    uint32_t userRankSize;  // Rank Number
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
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
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false;
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512;
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
    // local resource
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;  // usrrankid
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    // aicore detect remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // config parameters
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    // zero-copy
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];
    uint32_t zeroCopyDevicePhyId[16];

    bool utraceStatusFlag;
};

// Transport ÄÚ´æÀàÐÍ
enum class HcclAiRMAMemType : uint32_t {
    LOCAL_INPUT = 0,
    REMOTE_INPUT,

    LOCAL_OUTPUT,
    REMOTE_OUTPUT,

    // ¿ÉÍ¸´«¸ü¶àµÄÄÚ´æ£¬¿ÉÔÚMAX_NUMÖ®Ç°×·¼Ó£¬ÀýÈç£º
    // LOCAL_EXP,
    // REMOTE_EXP,
    MAX_NUM
};

struct HcclAiRMAMemInfo {
    uint32_t memMaxNum{0};         // ×î´óÄÚ´æÊýÁ¿£¬µÈÓÚ HcclAiRMAMemType::MAX_NUM
    uint32_t sizeOfMemDetails{0};  // sizeof(MemDetails)£¬ÓÃÓÚÄÚ´æÐ£ÑéºÍÆ«ÒÆ¼ÆËã
    uint64_t memDetailPtr{0};      // MemDetailsÊý×éÊ×µØÖ·, ¸öÊý: HcclAiRMAMemType::MAX_NUM
    // ¿ÉÍùºó×·¼Ó×Ö¶Î
};

// È«²¿ Transport QP/Mem ÐÅÏ¢
struct HcclAiRMAInfo {
    uint32_t curRankId{0};  // µ±Ç°rankId
    uint32_t rankNum{0};    // rankÊýÁ¿
    uint32_t qpNum{0};      // µ¥¸öTransportµÄQPÊýÁ¿

    uint32_t sizeOfAiRMAWQ{0};   // sizeof(HcclAiRMAWQ)
    uint32_t sizeOfAiRMACQ{0};   // sizeof(HcclAiRMACQ)
    uint32_t sizeOfAiRMAMem{0};  // sizeof(HcclAiRMAMemInfo)

    // HcclAiRMAWQ¶þÎ¬Êý×éÊ×µØÖ·
    // QP¸öÊý: rankNum * qpNum
    // ¼ÆËãÆ«ÒÆ»ñÈ¡SQÖ¸Õë£ºsqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t sqPtr{0};

    // HcclAiRMACQ¶þÎ¬Êý×éÊ×µØÖ·
    // QP¸öÊý: rankNum * qpNum
    // ¼ÆËãÆ«ÒÆ»ñÈ¡SCQÖ¸Õë£ºscqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t scqPtr{0};

    // HcclAiRMAWQ¶þÎ¬Êý×éÊ×µØÖ·
    // QP¸öÊý: rankNum * qpNum
    // ¼ÆËãÆ«ÒÆ»ñÈ¡RQÖ¸Õë£ºrqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t rqPtr{0};

    // HcclAiRMACQ¶þÎ¬Êý×éÊ×µØÖ·
    // QP¸öÊý: rankNum * qpNum
    // ¼ÆËãÆ«ÒÆ»ñÈ¡RCQÖ¸Õë: rcqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t rcqPtr{0};

    // HcclAivMemInfoÒ»Î¬Êý×é
    // ÄÚ´æÐÅÏ¢¸öÊý: rankNum
    // ¼ÆËãÆ«ÒÆ»ñÈ¡ÄÚ´æÐÅÏ¢Ö¸Õë: memPtr + rankId * sizeOfAiRMAMem
    // srcRankId »ñÈ¡×ÔÉíÄÚ´æÐÅÏ¢£¬dstRankId »ñÈ¡ Transport ÄÚ´æÐÅÏ¢
    uint64_t memPtr{0};
    // ¿ÉÍùºó×·¼Ó×Ö¶Î
};

struct CombinedCapability {
    uint64_t dataplaneModeBitmap;
};

struct HcclA2CombineOpParam {
    uint64_t workSpace;                               // Address for communication between client and server,
                                                      // hccl requests and clears
    uint64_t workSpaceSize;                           // Space for communication between client and server
    uint32_t rankId;                                  // id of this rank
    uint32_t rankNum;                                 // num of ranks in this comm group
    uint64_t winSize;                                 // size of each windows memory
    uint64_t windowsIn[AscendC::HCCL_MAX_RANK_NUM];   // windows address for input, windowsIn[rankId] corresponds
                                                      // to the local card address,
                                                      // and others are cross-card mapping addresses.
    uint64_t windowsOut[AscendC::HCCL_MAX_RANK_NUM];  // windows address for output, windowsOut[rankId] corresponds
                                                      // to the local card address,
                                                      // and others are cross-card mapping addresses.
    uint8_t res[8328];
    uint8_t multiFlag;
    __gm__ AscendC::IbVerbsData *data;
    uint64_t dataSize;
    // ×·¼Ó×Ö¶Î
    uint64_t sizeOfAiRMAInfo;  // sizeof(HcclAiRMAInfo)
    uint64_t aiRMAInfo;        // HcclAiRMAInfo* µ¥¸ö½á¹¹ÌåÖ¸Õë

    CombinedCapability *capability;  // address of the communication capability information structure on the Device
    uint64_t capabilitySize;         // size of the communication capability information structure
};

enum class DataplaneMode : uint32_t {
    HOST = 0,
    AICPU = 1,
    AIV = 2,
};

enum class DBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct HcclAiRMAWQ {
    uint32_t wqn{0};
    uint64_t bufAddr{0};
    uint32_t wqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB};  // 0-hw/1-sw
    uint64_t dbAddr{0};
    uint32_t sl{0};
};

struct HcclAiRMACQ {
    uint32_t cqn{0};
    uint64_t bufAddr{0};
    uint32_t cqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB};  // 0-hw/1-sw
    uint64_t dbAddr{0};
};

struct hns_roce_rc_sq_wqe {
    uint32_t byte_4;
    uint32_t msg_len;
    uint32_t immtdata;
    uint32_t byte_16;
    uint32_t byte_20;
    uint32_t rkey;
    uint64_t remoteVA;
};

struct hns_roce_lite_wqe_data_seg {
    uint32_t len;
    uint32_t lkey;
    uint64_t localVA;
};

__aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, uint64_t length)
{
    __gm__ uint8_t *start =
        (__gm__ uint8_t *)((uint64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
        (__gm__ uint8_t *)(((uint64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
}
__aicore__ inline DataplaneMode GetDataplaneMode(GM_ADDR contextGM0)
{
    __gm__ HcclA2CombineOpParam *winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
    CombinedCapability *capability = winContext_->capability;
    uint64_t capabilitySize = winContext_->capabilitySize;
    DataplaneMode dataplaneMode = DataplaneMode::HOST;
    if (capability == 0) {
        dataplaneMode = DataplaneMode::AICPU;
        return dataplaneMode;
    }
    uint64_t dataplaneModeBitmap = capability->dataplaneModeBitmap;
    if ((dataplaneModeBitmap & 0x02) == 0x02) {
        dataplaneMode = DataplaneMode::AICPU;
    }
    if ((dataplaneModeBitmap & 0x04) == 0x04) {
        dataplaneMode = DataplaneMode::AIV;
    }
    return dataplaneMode;
}

#endif  // MOE_DISTRIBUTE_BASE_H
