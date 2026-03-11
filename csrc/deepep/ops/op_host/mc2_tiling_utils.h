#ifndef __MC2_TILING_UTILS_H__
#define __MC2_TILING_UTILS_H__

#include <cstdint>
#include <map>
#include <string>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "hccl/hccl.h"
#include "mc2_hcom_topo_info.h"

class Mc2TilingUtils
{
public:
    static uint64_t GetMaxWindowSize()
    {
        uint16_t defaultWindowSize = 200;
        const char *hcclBuffSize = getenv("DEEPEP_HCCL_BUFFSIZE") == nullptr ? "HCCL_BUFFSIZE" : "DEEPEP_HCCL_BUFFSIZE";
        if (getenv(hcclBuffSize) == nullptr) {
            OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
        } else {
            try {
                std::string envStr(getenv(hcclBuffSize));
                defaultWindowSize = std::stoi(envStr);
            } catch (const std::invalid_argument &ia) {
                OP_LOGE("", "Invalid argument when parsing HCCL_BUFFSIZE: %s", ia.what());
            } catch (const std::out_of_range &oor) {
                OP_LOGE("", "Out of range when parsing HCCL_BUFFSIZE: %s", oor.what());
            }
        }
        const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
        OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
        return maxWindowSize;
    }
};

namespace mc2tiling {
using namespace ge;

constexpr uint32_t COMM_MESH = 0b1U;
constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
constexpr uint32_t COMM_UNDEFINED = 0xFFFFFFFFU;
constexpr uint8_t COMM_ALG_FULL_MESH_HOST = 6;
constexpr uint64_t CHECK_VALUE_ODD = 2;
constexpr uint32_t AIC_NUM_910D = 32;
constexpr uint64_t MC2_TILINGKEY_OFFSET =
    uint64_t(1000000000000000000UL);  // 10^18
constexpr size_t RES_LEN = 64;
constexpr size_t MAX_MSG_NUM = 16;
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;  // 只通信不计算
constexpr char HCCL_DETERMINISTIC[] = "HCCL_DETERMINISTIC";
/**
当前通信API未提供枚举，后续会提供
0：默认值 1：HOST_TS（A2/3支持 A5不支持）2：AICPU_TS（A2/3支持 A5不支持）
3：AIV 4：AIV_ONLY（A2/3支持 A5不支持） 5：CCU_MS（A2/3支持 A5不支持）
6：CCU_SCHED（A2/3支持 A5不支持） 7：AICPU_UB/ROCE（A5不支持）
**/
constexpr uint8_t AIV_ENGINE = 3;
constexpr uint8_t A5_CCU_ENGINE = 5;
constexpr uint8_t Y_INDEX = 3;
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint8_t COMM_VERSION3 = 3;
constexpr double COMM_GROW_RATIO = 1.15;
constexpr uint64_t MTE_STATE_ZONE_SIZE = 1024UL * 1024UL;

constexpr uint64_t LARGE_K = 8192;
constexpr uint64_t LARGE_N = 5120;
constexpr uint64_t SMALL_N_BOUNDARY = 2048;
constexpr uint64_t TINY_M = 512;
constexpr uint64_t SMALL_M = 2048;
constexpr uint64_t MEDIAN_M = 4096;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO1 = 3;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO2 = 1.5;

constexpr uint8_t TIME_LOWER_RATIO = 2;
constexpr double TIME_UPPER_RATIO = 3.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO1 = 1.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO2 = 1.2;
constexpr double CUBE_UTIL_THRESH = 0.85;
constexpr uint32_t AICPU_NUM_BLOCKS_A2 = 6U;

constexpr auto DEFAULT_KEY_FOR_FITTING_MAP = "0_0";

inline std::string GetSocVersion(const gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    return socVersion;
}

// inline ge::graphStatus GetCclBufferSize(const char* groupStr, uint64_t* cclBufferSize, const char* nodeName)
// {
//     HcclComm hcclComm;
//     OP_TILING_CHECK(Mc2Hcom::MC2HcomTopology::CommGetCclBufferSizeByGroup(groupStr, cclBufferSize, &hcclComm)
//         != HCCL_SUCCESS, OP_LOGE(nodeName, "CommGetCclBufferSizeByGroup failed"), return ge::GRAPH_FAILED);
//     if (hcclComm == nullptr) {
//         OP_TILING_CHECK(Mc2Hcom::MC2HcomTopology::CommGetGroupLocalWindowSize(groupStr, cclBufferSize) != HCCL_SUCCESS,
//             OP_LOGE(nodeName, "GetGroupLocalWindowSize from topoInfo failed"), return ge::GRAPH_FAILED);
//         OP_LOGD(nodeName, "Get cclBufferSize by topoInfo");
//     } else {
//         OP_LOGD(nodeName, "Get cclBufferSize from HCCL");
//     }
//     OP_TILING_CHECK(*cclBufferSize == 0,
//             OP_LOGE(nodeName, "Get cclBufferSize failed, cclBufferSize is 0"), return ge::GRAPH_FAILED);
//     return ge::GRAPH_SUCCESS;
// }

inline ge::graphStatus GetEpWinSize(const gert::TilingContext *context, const char *nodeName,
    uint64_t &hcclBufferSizeEp, uint64_t &maxWindowSizeEp, uint32_t attrGroupEpIndex)
{
    auto attrs = context->GetAttrs();
    if (mc2tiling::GetSocVersion(context) == "Ascend910_95") {
        // A5 暂不支持 Hccl CommGetBufSizeCfg 接口，此处暂作规避
        hcclBufferSizeEp = Mc2TilingUtils::GetMaxWindowSize();
        // A5 上前 1MB 作为状态区，剩余空间用作数据区
        maxWindowSizeEp = hcclBufferSizeEp - MTE_STATE_ZONE_SIZE;
    } else {
        // auto groupEpHccl = attrs->GetAttrPointer<char>(static_cast<int>(attrGroupEpIndex));
        // OP_TILING_CHECK(GetCclBufferSize(groupEpHccl, &hcclBufferSizeEp, nodeName) != ge::GRAPH_SUCCESS,
        //     OP_LOGE(nodeName, "Get Ep HcclBufferSizeEP failed, HcclBufferSizeEP is %lu", maxWindowSizeEp),
        //     return ge::GRAPH_FAILED);
        // maxWindowSizeEp = hcclBufferSizeEp;
    }
    return ge::GRAPH_SUCCESS;
}
}
#endif
