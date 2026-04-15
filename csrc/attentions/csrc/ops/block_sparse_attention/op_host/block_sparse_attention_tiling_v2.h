/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSEBSA_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSEBSA_V2_H_
#include <cstdint>
#include <vector>
#include <queue>
#include <string>
#include "exe_graph/runtime/tiling_context.h"
#include "data_copy_transpose_tiling_def.h"
#include "data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#include "ops_error.h"

#include "register/op_def_registry.h"
#include "block_sparse_attention_tiling.h"

namespace optiling {

struct BSAShapeInfo {
    uint32_t b = 0;
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t d = 0;
    uint32_t h = 0;
    uint32_t t = 0;
};

std::string GetPfaDataTypeStr(ge::DataType type);
class BlockSparseAttentionTilingV2 {
public:
    platform_ascendc::PlatformAscendC ascendcPlatform;
    explicit BlockSparseAttentionTilingV2(fe::PlatFormInfos* platFormInfo): ascendcPlatform(platFormInfo) {}
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForBSATiling& contextKeyParams,
        uint64_t& tilingKey, uint32_t& blockDimToBeSet, BlockSparseAttentionTilingData& tilingData);
    ge::graphStatus BlockSparseAttentionSetTilingData(gert::TilingContext* context,
        BlockSparseAttentionTilingData& tilingData);
    bool CheckNonEmptyShapeExceptions(const ContextParamsForBSATiling& contextKeyParams,
        const gert::StorageShape* shape, const std::string &sName) const;
#ifndef ASCEND_OPTILING_UT
protected:
    void BlockSparseAttentionInitOutputSplit(int64_t totalSize, BlockSparseAttentionTilingData &tilingData);
    void SetEmptyTensor(ContextParamsForBSATiling& contextKeyParams, uint64_t& tilingKey, uint32_t& blockDimToBeSet,
        BlockSparseAttentionTilingData& tilingData);
    bool CheckIODataType(ContextParamsForBSATiling& contextKeyParams);
    bool SetInputLayout(const char* layout);
    bool GetAndCheckShape(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& shapeInfo,
        const gert::StorageShape* shape, const std::string& sName) const;
    bool GetAndCheckRopeShape(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& shapeInfo,
        BSAShapeInfo& ropeShapeInfo, const gert::StorageShape* shape, const std::string& sName,
        const std::string& rName) const;
    bool SetShape(ContextParamsForBSATiling& contextKeyParams, const gert::StorageShape* shape,
        const std::string inputName, int64_t& b, int64_t& n, int64_t& s, int64_t& d, int64_t& h, int64_t& t) const;
    bool CheckQueryOutParamsConsistency(const ContextParamsForBSATiling& contextKeyParams,
        const gert::StorageShape* queryShape, const gert::StorageShape* outShape) const;
    bool CheckKVDataType(ContextParamsForBSATiling& contextKeyParams);
    bool CheckKeyValueParamsConsistency(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& keyShapeInfo,
        BSAShapeInfo& valueShapeInfo, const gert::StorageShape* keyShape, const gert::StorageShape* valueShape);
    bool SetAndCheckHeadNumRatio(ContextParamsForBSATiling& contextKeyParams,
        BlockSparseAttentionTilingData& tilingData);
    bool CheckInputDimAndHeadNum(ContextParamsForBSATiling& contextKeyParams, const uint32_t nQ, const uint32_t nKV);
    bool CheckPostQuantShape(const ContextParamsForBSATiling& contextKeyParams, uint32_t quantD,
        const gert::StorageShape* quantOffset2Shape, const ge::DataType quantScale2Type, int64_t quantScale2ShapeSize,
        uint32_t h) const;
    bool CheckPostQuantParams(const ContextParamsForBSATiling& contextKeyParams, uint32_t h, uint32_t n) const;
    bool CheckAntiquantParamsShape(ContextParamsForBSATiling& contextKeyParams, const uint32_t n, const uint32_t d,
        const uint32_t h, BlockSparseAttentionTilingData& tilingData);
    bool GetAndCheckPrefixShape(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& prefixShapeInfo, const gert::StorageShape* queryShape,
        BlockSparseAttentionTilingData& tilingData) const;
    bool CheckKeyValuePrefixConsistency(ContextParamsForBSATiling& contextKeyParams,
        const gert::StorageShape* keyShape);
    bool CheckActSharedPrefix(ContextParamsForBSATiling& contextKeyParams, const uint32_t sPrefix, const uint32_t sKV);
    bool CheckPAKeyValueShape(ContextParamsForBSATiling& contextKeyParams, int64_t& keyDim1,
        BSAShapeInfo& queryShapeInfo, const gert::StorageShape* keyShape, const gert::StorageShape* valueShape,
        const size_t keyDim, const int32_t* blockSize, int64_t blockNumValid, int32_t headNumRatio);
    bool CheckPACacheShape(ContextParamsForBSATiling& contextKeyParams, const size_t keyDim, BSAShapeInfo& shapeInfo,
        const gert::StorageShape* shape, const int32_t* blockSize, int64_t blockNumValid, int32_t headNumRatio,
        const std::string& sName);
    bool CheckBlockTableShape(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& queryRopeShapeInfo, const int32_t* blockSize, const gert::StorageShape* blockTableShape,
        BlockSparseAttentionTilingData& tilingData);
    bool CheckMaskShape(ContextParamsForBSATiling& contextKeyParams, const int32_t* sparseMode,
        int64_t& attenMaskBatch, int64_t& attenMaskS1, int64_t& attenMaskS2, bool& checkMask, const uint32_t sQ,
        const uint32_t sK, const uint32_t batchSize, std::string& strMaskShape);
    void SetSparseModeData(ContextParamsForBSATiling& contextKeyParams, const gert::StorageShape* attenMaskShape,
        BlockSparseAttentionTilingData& tilingData, const int32_t* sparseMode, const int64_t* preTokens,
        const int64_t* nextTokens);
    bool CheckMaskShapeCrossSparse(ContextParamsForBSATiling& contextKeyParams, const int32_t* sparseMode,
        const uint32_t sQ, const uint32_t sK, const uint32_t batchSize);
    bool CheckIO(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& valueShapeInfo);
    bool CheckKV(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& keyShapeInfo,
        BSAShapeInfo& valueShapeInfo);
    bool CheckQueryAndKey(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BlockSparseAttentionTilingData& tilingData);
    bool CheckRope(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BSAShapeInfo& queryRopeShapeInfo);
    bool CheckQuant(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BlockSparseAttentionTilingData& tilingData);
    bool CheckPrefix(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BSAShapeInfo& valueShapeInfo, BlockSparseAttentionTilingData& tilingData);
    bool CheckActSeq(const ContextParamsForBSATiling& contextKeyParams, const BSAShapeInfo& queryShapeInfo) const;
    bool CheckActSeqLen(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BlockSparseAttentionTilingData& tilingData);
    bool CheckPATypeAndShape(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& queryRopeShapeInfo, BlockSparseAttentionTilingData& tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForBSATiling& contextKeyParams, uint32_t b, uint32_t n,
        uint32_t s1, uint32_t s2);
    bool CheckInnerPrecise(ContextParamsForBSATiling& contextKeyParams, BlockSparseAttentionTilingData& tilingData);
    bool CheckMaskTypeAndShape(ContextParamsForBSATiling& contextKeyParams, BlockSparseAttentionTilingData& tilingData);
    void SetSparseType(uint32_t qS);
    bool CheckSparseMode(ContextParamsForBSATiling& contextKeyParams, uint32_t qS,
        BlockSparseAttentionTilingData& tilingData);
    bool CheckPACrossover(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BlockSparseAttentionTilingData& tilingData);
    bool CheckMaskCrossover(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BlockSparseAttentionTilingData& tilingData);
    bool CheckTNDLayoutCrossover(ContextParamsForBSATiling& contextKeyParams);
    bool ParseActualSeqLengths(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    bool CheckMultiFeatureCrossover(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
        BlockSparseAttentionTilingData& tilingData);
    void SetTilingDataAttribute(ContextParamsForBSATiling& contextKeyParams,
        BlockSparseAttentionTilingData& tilingData);
    void GetEnableDN(ContextParamsForBSATiling& contextKeyParams, BlockSparseAttentionTilingData& tilingData,
        BSAShapeInfo& queryShapeInfo, BSAShapeInfo& valueShapeInfo, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV);
    void SetTilingData(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& queryRopeShapeInfo, BSAShapeInfo& valueShapeInfo, BlockSparseAttentionTilingData &tilingData);
    void InferTilingMod(const ContextParamsForBSATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen, uint32_t d, uint32_t seqInnerSize,
        int32_t sparseModeVal);
    int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums);
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter,
        int64_t token);
    void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp,
        int64_t& nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV, uint32_t sOuterSize,
        uint32_t sInnerSize, int64_t preTokensLeftUp, int64_t nextTokensLeftUp, bool isAttenMaskUsed);
    void ComputeSplitNBSeq(BlockSparseAttentionTilingData& tilingData, uint32_t batchSize,
        const size_t tilingElementArrayLen, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV, uint32_t sOuterSize, uint32_t sInnerSize, double coreWightTarget,
        uint32_t& curCore);
    void BlockSparseAttentionSplitNBSeq(BlockSparseAttentionTilingData& tilingData,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed);
    void InferSplitCoreMode();
    void InferConstantization();
    bool AdjustCVTilingCVDiff(const ContextParamsForBSATiling& contextKeyParams, uint32_t& sOuterFactor,
        uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor, BlockSparseAttentionTilingData& tilingData,
        const BSAShapeInfo& queryShapeInfo, const BSAShapeInfo& valueShapeInfo);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType);
    bool EnableMTE2BmmPipe(BlockSparseAttentionTilingData& tilingData, matmul_tiling::MatmulApiTiling& bmm,
        TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor);
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData);
    bool BlockSparseAttentionCheckBmm1(BlockSparseAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
        int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, bool allGM = false,
        bool autoBaseMNK = false);
    bool BlockSparseAttentionCheckBmm2(BlockSparseAttentionTilingData& tilingData, TCubeTiling& bmm2TilingData,
        int64_t l1SizeRemain, int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor,
        bool allGM = false, bool autoBaseMNK = false);
    bool BlockSparseAttentionComputeCVDiffParams(BlockSparseAttentionTilingData& tilingData,
        int64_t l1Size, int64_t l0CSize, uint32_t& sOuterFactor, uint32_t &sInnerFactor);
    void GetPreNextTokensLeftUp(BlockSparseAttentionTilingData& tilingData, int64_t actualSeqLength,
        int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    void UpdateTilingKeyMatmulCfg(uint64_t& tilingKey);
    void UpdateTilingKeyMaskCfg(BlockSparseAttentionTilingData& tilingData, uint64_t& tilingKey);
    void UpdateTilingKeyPseCfg(BlockSparseAttentionTilingData& tilingData, uint64_t& tilingKey);
    void UpdateTilingKeyDSizeConst(BlockSparseAttentionTilingData &tilingData, uint64_t& tilingKey);
    void UpdateTilingKeyValueDSizeConst(BlockSparseAttentionTilingData &tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeySInnerConst(BlockSparseAttentionTilingData &tilingData, uint64_t& tilingKey);
    void UpdateTilingKeySOuterConst(BlockSparseAttentionTilingData &tilingData, uint64_t& tilingKey);
    void BlockSparseAttentionInitSoftmaxLseOutputSplit(int64_t totalSize, BlockSparseAttentionTilingData &tilingData);
    void UpdateTilingKeyFlag(ContextParamsForBSATiling& contextKeyParams, uint64_t& tilingKey);
    bool TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey, ContextParamsForBSATiling& contextKeyParams,
        BlockSparseAttentionTilingData &tilingData);
    size_t GetBSAWorkSpaceSize(BlockSparseAttentionTilingData& tilingData);
    ge::graphStatus SetPlatMemoryInfo(ContextParamsForBSATiling& contextKeyParams);
    ge::graphStatus SetAttributeInfo(ContextParamsForBSATiling& contextKeyParams);
    ge::graphStatus CheckTensorInvalid(const ContextParamsForBSATiling& contextKeyParams) const;
    ge::graphStatus CheckSingleAttribute(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, BSAShapeInfo& valueShapeInfo, BSAShapeInfo& queryRopeShapeInfo,
        BlockSparseAttentionTilingData& tilingData);
    ge::graphStatus CheckCrossoverAttribute(ContextParamsForBSATiling& contextKeyParams, BSAShapeInfo& queryShapeInfo,
        BSAShapeInfo& keyShapeInfo, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
        BlockSparseAttentionTilingData& tilingData);
    ge::graphStatus AdjustTilingData(ContextParamsForBSATiling& contextKeyParams,
        BlockSparseAttentionTilingData& tilingData, const BSAShapeInfo& queryShapeInfo,
        const BSAShapeInfo& valueShapeInfo);
    ge::graphStatus ComputeTilingData(ContextParamsForBSATiling& contextKeyParams,
        std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
        BlockSparseAttentionTilingData& tilingData);
    ge::graphStatus ComputeTilingKey(uint64_t& tilingKey, ContextParamsForBSATiling& contextKeyParams,
        uint32_t& blockDimToBeSet, BlockSparseAttentionTilingData& tilingData);
    void SetAttenMaskCompressMode();
    void BSATilingDataconvert(BlockSparseAttentionTilingData& tilingData);
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
protected:
    ContextParamsForBSATiling* contextKeyParamsPtr = nullptr;
    int64_t ubSizeRemain = 1;
    bool isSOuterNoTail = true;
    bool isSInnerNoTail = true;
    bool isDNoTail = true;
    bool enableTensorList = false;
    bool enableLeftPadding = false;
    bool enableActSeqLen = false;
    bool enableActSeqLenKV = false;
    bool enableKVAntiquant = false;
    bool enablePseShift = false;
    bool enableMask = false;
    bool enableQuantBF16 = false;
    bool enableMatmulNorm = false;
    bool enablePA = false;
    bool enableSplitSeqOneN = false;
    bool isDefaultSparseMode = false;
    bool isKVHasPrefix = false;
    bool isBandMode = false;
    bool enableIFAMLA = false;
    bool enableIFA = false;
    bool enableBSAMLA = false;
    bool enableDN = false;
    uint32_t gSize = 1;
    InputLayout inputLayout = InputLayout::BSH;
    ge::DataType inputType{ge::DT_FLOAT16};
    ge::DataType outputType{ge::DT_FLOAT16};
    ge::DataType pseShiftElemType{ge::DT_FLOAT16};
    uint32_t dataTypeSize = FLOAT32SIZE;
    uint32_t outputDataTypeSize = FLOAT32SIZE;
    uint32_t maskElemSize = FLOAT32SIZE;
    int32_t ifaBlockSizeBase = 32;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseShiftElemSize = 0;
    uint32_t pseMaskMaxSize = 0;
    int64_t pseShiftBatch = 0;
    int64_t pseShiftS1 = 0;
    int64_t pseShiftS2 = 0;
    int64_t actSeqLenDims = 0;
    int64_t actSeqLenKVDims = 0;
    int64_t middleActualSeqLengths = 0;
    int64_t actualSharedPrefixLen = 0;
    uint32_t needInit = 0U;
    uint32_t usePseShift = 0;
    // There is no S2 axis for PA. Use the change amount to normalize the S2 length in both PA and non PA scenarios
    uint32_t S2 = 0;
    int32_t blockTableDim2 = 1;
    int32_t paBlockNumSum = 1;
    uint32_t maskTypeByteNum = 0;
    uint32_t softmaxDataTypeSize = FLOAT32SIZE; // BF16 calculates through FP32
    platform_ascendc::SocVersion curShortSocName;
    uint32_t layoutType = 0;
    uint32_t paLayoutType = 0;
    int64_t sparsePreTokens = 0;
    int64_t sparseNextTokens = 0;
    int32_t sparseModeVal = 0;
    SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
    bool isConstantization = false;
    uint32_t splitS2 = 1; // It can only be 0 when the D axis is split
    int32_t innerPrecise = HIGH_PERFORMANCE;
    uint32_t sOuterFactorTiling = 0;
    uint32_t softmaxSInnerFactorTiling = 0;
    uint32_t softmaxSOuterFactorTiling = 0;
    size_t defaultSysWorkspaceSize = 0;
    matmul_tiling::PlatformInfo ascendPlatformInfo;

    bool faRunFlag_ = false;
    uint8_t attenMaskShapeType = 0; // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    uint8_t sparseType = 0;
    FlashAttentionScoreSimplifiedTilingData faTilingAdapter;
#endif
};
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSEBSA_V2_H_
