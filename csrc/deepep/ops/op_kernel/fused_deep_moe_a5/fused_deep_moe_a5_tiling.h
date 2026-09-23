#ifndef FUSED_DEEP_MOE_TILING_H
#define FUSED_DEEP_MOE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#ifdef DEBUG_SPACE
;
#else
#define ENABLE_REUSE_MEMORY
#endif
namespace Cam {
struct WorkSpaceOffset {
    // MM1/GMM1-Swiglu input
    int64_t shareX1TokenOffset;
    int64_t x1TokenOffset;
    int64_t shareX1ScaleOffset;
    int64_t x1ScaleOffset;
    // MM1/GMM1-Swiglu output
    int64_t shareSwigluOffset;
    int64_t swigluOffset;
    // MM2/GMM2 input
    int64_t shareX2TokenOffset;
    int64_t x2TokenOffset;
    int64_t shareX2ScaleOffset;
    int64_t x2ScaleOffset;

    int64_t shareMm1SwapSpaceOffset;       // 交换空间，用于C->V数据交换
    int64_t shareMm2SwapSpaceOffset;       // 交换空间，用于C->V数据交换
    int64_t gmm1SwapSpaceOffset;           // 交换空间，用于C->V数据交换
    int64_t gmm2SwapSpaceOffset;           // 交换空间，用于C->V数据交换
    int64_t y2TokenOffset;                 // 浅融合使用，已反量化无scale
    int64_t groupListOffset;               // 各专家token数前缀和形式
    int64_t expandIdxOffset;               // dispatch时token在远端索引
    int64_t epSendCountOffset;             // 各专家从各个rank收到的token数
    int64_t routedGroupMetaOffset;         // 稀疏routed路径的group metadata
    int64_t routedActiveGroupCountOffset;  // 稀疏routed路径的active group的数量
    int64_t routedActiveGroupIdsOffset;    // 稀疏routed路径的active group
    int64_t reservedOffset;                // 预留空间
};

struct RoutedGroupMeta {
    uint32_t tokenCount;             // 当前 group 实际收到的 token 数
    uint32_t computeActiveAivCount;  // GMM2 等待的 routed-X2 ready 通知数
    uint8_t padding[24];             // 保持每个 group 独占一个 32B GM block
};
static_assert(sizeof(RoutedGroupMeta) == 32, "RoutedGroupMeta must occupy one GM cache line");

struct FusedDeepMoeInfo {
    uint32_t epRankSize;           // epRankSize
    uint32_t epRankId;             // epRankId
    uint32_t moeExpertNum;         // moe expert number
    uint32_t moeExpertNumPerRank;  // moe expert number per rank
    uint32_t quantMode;            // reserved, from quant_mode attr
    uint32_t activationType;       // 0: SwiGLU (SiLU gate), 1: SiTU
    float beta;                    // SiTU gate soft-saturation bound
    float linearBeta;              // SiTU up soft-saturation bound; 0 disables it
    uint32_t mxActStorageFp4;      // non-zero when gmm weight dtype is FP4; workspace sizing only
    uint32_t profileEnable;        // non-zero when fused kernel stage trace collection is enabled
    uint32_t profileLaunchId;      // launch slot in the session-owned persistent profile buffer
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aicNum;               // aicNum
    uint32_t aivNum;               // aivNum
    uint64_t profileBufferBytes;   // total bytes of the persistent profile buffer
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    uint64_t gmm1HLen;
    uint64_t shareGmm1HLen;     // shared expert gmm1 hidden length
    uint32_t weightLayoutMode;  // 0: ND, 1: FRACTAL_NZ (shared by GMM1/GMM2)
    uint64_t gmm1WeightExpertStrideBytes;
    uint64_t gmm2WeightExpertStrideBytes;
    bool isTensorList;
    bool enableRoutedSparseFastPath;
};

struct FusedDeepMoeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    FusedDeepMoeInfo fusedDeepMoeInfo;
    WorkSpaceOffset workSpaceOffset;
};

constexpr uint32_t GMM1_L1M = 256;
constexpr uint32_t GMM1_L1N = 256;
constexpr uint32_t GMM1_L1K = 256;
constexpr uint32_t GMM1_L0K = 128;
// FP4 stores two logical values per byte, so its L1/L0 K tiles can be
// doubled while keeping the same physical buffer footprint as FP8.
constexpr uint32_t GMM1_L1K_FP4 = 512;
constexpr uint32_t GMM1_L0K_FP4 = 256;
constexpr uint32_t GMM1_EPIM = 64;
constexpr uint32_t GMM1_SWIZZLE_OFFSET = 3;
constexpr uint32_t GMM1_SWIZZLE_DIRECTION = 0;

constexpr uint32_t GMM2_L1M = 256;
constexpr uint32_t GMM2_L1N = 256;
constexpr uint32_t GMM2_L1K = 256;
constexpr uint32_t GMM2_L0K = 128;
constexpr uint32_t GMM2_L1K_FP4 = 512;
constexpr uint32_t GMM2_L0K_FP4 = 256;
constexpr uint32_t GMM2_EPIM = 64;
constexpr uint32_t GMM2_SWIZZLE_OFFSET = 3;
constexpr uint32_t GMM2_SWIZZLE_DIRECTION = 0;

constexpr uint32_t MX_FP4_QUANT_MODE = 4U;
constexpr uint32_t WEIGHT_LAYOUT_ND = 0U;
constexpr uint32_t WEIGHT_LAYOUT_NZ = 1U;
constexpr uint32_t ACTIVATION_SWIGLU = 0U;
constexpr uint32_t ACTIVATION_SITU = 1U;

constexpr uint32_t EXEC_FLAG_DEEP_FUSE = (1U << 0);
constexpr uint32_t EXEC_FLAG_TENSOR_LIST = (1U << 1);
constexpr uint32_t EXEC_FLAG_X_ACTIVE_MASK = (1U << 2);
constexpr uint32_t EXEC_FLAG_SHARED_EXPERT = (1U << 3);
constexpr uint32_t EXEC_FLAG_SMOOTH_QUANT = (1U << 4);

}  // namespace Cam
#endif  // FUSED_DEEP_MOE_TILING_H
