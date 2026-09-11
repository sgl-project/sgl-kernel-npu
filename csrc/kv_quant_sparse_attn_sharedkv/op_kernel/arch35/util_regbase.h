









/*!
 * \file util_regbase.h
 * \brief
 */

#ifndef KV_QUANT_SAS_UTIL_REGBASE_H
#define KV_QUANT_SAS_UTIL_REGBASE_H

#include "util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace regbaseutil {
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int64_t s2LoopEndIdx;           \
    int64_t s2LineStartIdx = 0;     \
    int64_t s2LineEndIdx;           \
    int64_t s2CmpLineEndIdx; \
     \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t attentionOutOffset;     \
    int32_t actualS1Size;       \
    int32_t actualS2Size     \

struct RunParamStr {
    COMMON_RUN_PARAM;

    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;

    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS;
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS;


    int64_t sOuterOffset;
    int64_t cubeSOuterOffset;
    int64_t mOuterOffset;
    int64_t cubeMOuterOffset;


    int64_t softmaxLseOffset;

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx;  \
    int64_t s2EndIdx; \
    int64_t s2LoopCount;  \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0;  \
    int64_t loop = 0; /* for v0 perload loop */ \
    int64_t boIdx = 0;  \
    int64_t n2oIdx = 0;  \
    int64_t goIdx = 0;  \
    int32_t s1RealSize; \
    int32_t halfS1RealSize;  \
    int32_t firstHalfS1RealSize;  \
    int32_t mRealSize; \
    int32_t halfMRealSize; \
    int32_t firstHalfMRealSize; \
    int32_t s2RealSize;  \
    int64_t s2AlignedSize;  \
    int32_t vec2S1BaseSize;  \
    int32_t vec2S1RealSize;  \
    int32_t vec2MBaseSize; \
    int32_t vec2MRealSize; \
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
    int64_t attentionOutOffset; \
    int32_t actualS1Size;  \
    int32_t actualS2Size;  \
    int64_t preTokensPerBatch;  \
    int64_t nextTokensPerBatch;  \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset; \
    int64_t mOuterOffset; \
    bool    isCmp; \

struct RunInfo {
    COMMON_RUN_INFO;


    int64_t softmaxLseOffset;

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
};

#define COMMON_CONST_INFO \
     \
    uint32_t bSize; \
    uint32_t needInit; \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t dSize; /* query d 512 */ \
    int64_t dSizeV; /* key d 512 */ \
    int64_t dSizeVInput; /* key inpue d 640 = rope + nope + scale + pad */ \
    int64_t dSizeNope; /* key nope d 448 */ \
    int64_t dSizeRope; /* key rope d 64 */ \
    int64_t tileSize; /* 64 */ \
    int64_t sparseMode = 3; \
    int64_t gSize;  \
    int64_t n2Size; \
    int64_t s1Size;  \
    int64_t s2Size;  \
     \
    int64_t s1D; \
    int64_t gS1D; \
    int64_t n2GS1D; \
    int64_t s2D; \
    int64_t n2S2D; \
    int64_t s1Dv; \
    int64_t gS1Dv; \
    int64_t n2GS1Dv; \
    int64_t s2Dv; \
    int64_t n2S2Dv; \
    int64_t s1S2; \
    int64_t gS1; \
    int64_t gD; \
    int64_t n2D; \
    int64_t bN2D; \
    int64_t gDv; \
    int64_t n2Dv; \
    int64_t bN2Dv; \
    int64_t n2G; \
    int64_t n2GD; \
    int64_t bN2GD; \
    int64_t n2GDv; \
    int64_t bN2GDv; \
    int64_t gS2; \
    int64_t s1Dr; \
    int64_t gS1Dr; \
    int64_t n2GS1Dr; \
    int64_t s2Dr; \
    int64_t n2S2Dr; \
    int64_t gDr; \
    int64_t n2Dr; \
    int64_t bN2Dr; \
    int64_t n2GDr; \
    int64_t bN2GDr; \
    int32_t s2BaseN2D; \
    int32_t s1BaseN2GD; \
    int64_t s2BaseBN2D; \
    int64_t s1BaseBN2GD; \
    int32_t s1BaseD; \
    int32_t s2BaseD; \
    int64_t s2BaseN2Dv; \
    int64_t s2BaseBN2Dv; \
    int64_t s1BaseN2GDv; \
    int64_t s1BaseBN2GDv; \
    int32_t s1BaseDv; \
    int32_t s2BaseDv; \
     \
    int64_t mm1Ka; \
     \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t subBlockIdx;\

#define INFER_CONST_INFO \
     \
    bool isActualLenDimsNull;  \
    bool isActualLenDimsKVNull;  \
    bool isSoftmaxLseEnable; \
    bool rsvd1; \
    uint32_t sparseBlockCount; \
    uint32_t actualSeqLenSize;  \
    uint32_t actualSeqLenKVSize;  \
    /* service mm1 mm2 pageAttention */ \
    uint32_t oriBlockSize; \
    uint32_t cmpBlockSize; \
    uint32_t paLayoutType; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t sparseBlockSize; \
    bool hasOriSparseIndices; \
    uint32_t oriSparseIndexWidth; \
    uint32_t cmpRatio; \
    float softmaxScale; \
    uint32_t oriKvStride; \
    uint32_t cmpKvStride; \

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    uint32_t bSize;  \
    uint32_t n2Size;  \
    uint32_t gSize;  \
    uint32_t s1Size;  \
    uint32_t s2Size;  \
    uint32_t dSize : 10;  \
    uint32_t dSizeVInput : 12;  \
    uint32_t needInit : 4; \
    uint32_t layoutType : 4;  \
    uint32_t isActualSeqLengthsNull : 1; \
    uint32_t isActualSeqLengthsKVNull : 1; \
    uint32_t sparseBlockCount; \
    float softmaxScale; \
    uint32_t cmpRatio : 9; \
    uint32_t dSizeRope : 11; \
    uint32_t oriMaskMode : 6;\
    uint32_t cmpMaskMode : 6; \
    int32_t oriWinLeft; \
    int32_t oriWinRight; \
    uint32_t tileSize : 8; \
    /* pa params */  \
    uint32_t oriBlockSize : 12; \
    uint32_t cmpBlockSize : 12; \
    uint32_t oriMaxBlockNumPerBatch; \
    uint32_t cmpMaxBlockNumPerBatch; \
    uint32_t oriKvStride; \
    uint32_t cmpKvStride; \
    uint32_t hasOriSparseIndices : 1; \
    uint32_t oriSparseIndexWidth; \


struct ConstInfo{
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
};

/* only support b32 or b64 */
struct CVSharedParams {
    CV_SHARED_PARAMS;
};
}

#endif // KV_QUANT_SAS_UTIL_REGBASE_H
