
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSEBSA_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSEBSA_H_
#include <cstdint>
#include <vector>
#include <queue>
#include <string>
#include "exe_graph/runtime/tiling_context.h"
#include "data_copy_transpose_tiling_def.h"
#include "data_copy_transpose_tiling.h"

#include "tiling/tiling_api.h"

#include "register/op_def_registry.h"
#ifdef ASCENDC_OP_TEST
#define BSA_EXTERN_C extern "C"
#else
#define BSA_EXTERN_C
#endif

#include "ada_block_sparse_attention_tiling_compile_info.h"
#include "ada_block_sparse_attention_tiling_const.h"
#include "ada_block_sparse_attention_tiling_context.h"
#include "ada_block_sparse_attention_tiling_struct.h"

#include <array>

#include "../op_kernel/ada_block_sparse_attention_tiling_data.h"

namespace optiling {

class BufferNum
{
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num;  // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum;  // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class AdaBlockSparseAttentionTiling
{
public:
    explicit AdaBlockSparseAttentionTiling(fe::PlatFormInfos *platFormInfo) : ascendcPlatform(platFormInfo) {}
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForBSATiling &contextKeyParams, uint64_t &tilingKey,
                                                 uint32_t &blockDimToBeSet,
                                                 SglAdaBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus AdaBlockSparseAttentionSetTilingData(gert::TilingContext *context,
                                                         SglAdaBlockSparseAttentionTilingData &tilingData);
    bool CheckNonEmptyShapeExceptions(ContextParamsForBSATiling &contextKeyParams, const gert::StorageShape *shape,
                                      const std::string &sName);
    bool fromBSA_ = true;
    bool CheckBaseApiNonEmptyShapeExceptions(ContextParamsForBSATiling &contextKeyParams,
                                             const gert::StorageShape *shape, const std::string &sName);

protected:
    ge::graphStatus ConvertContextToBSAParams(gert::TilingContext *context, ContextParamsForBSATiling &contextKeyParams,
                                              SglAdaBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus TilingGetTilingKeyAttentionAscendC(uint64_t &tilingKey, ContextParamsForBSATiling &contextKeyParams,
                                                       bool useNewTiling,
                                                       SglAdaBlockSparseAttentionTilingData &tilingData);
    void AdaBlockSparseAttentionSplitNS(ContextParamsForBSATiling &contextKeyParams,
                                        SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t curCoreNum,
                                        std::vector<int64_t> &actualSeqLengths);
    void AdaBlockSparseAttentionSplitNSTable(ContextParamsForBSATiling &contextKeyParams,
                                             SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t curCoreNum,
                                             std::vector<int64_t> &actualSeqLengths,
                                             std::vector<int64_t> &actualSeqLengthsKV, int64_t actualSharedPrefixLen,
                                             bool useBalanceTiling);

    void GetPreNextTokensLeftUp(SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t actualSeqLength,
                                uint32_t actualSeqLengthKV, int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp);
    void SetSplitCoreMode(SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t sOuterFactor);
    void AdaBlockSparseAttentionSplitSeqOneN(SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t curCoreNum,
                                             bool isVectorCore);
    bool EnableMTE2BmmPipe(SglAdaBlockSparseAttentionTilingData &tilingData, matmul_tiling::MatmulApiTiling &bmm,
                           AscendC::tiling::TCubeTiling &bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor);
    void EnableBmmDoubleBuffer(AscendC::tiling::TCubeTiling &bmmTilingData);
    void AdaBlockSparseAttention310PSetBmm1(matmul_tiling::MatmulApiTiling &bmm1);
    void AdaBlockSparseAttention310PSetBmm2(matmul_tiling::MatmulApiTiling &bmm2);
    bool AdaBlockSparseAttentionCheckBmm1(SglAdaBlockSparseAttentionTilingData &tilingData,
                                          AscendC::tiling::TCubeTiling &bmm1TilingData, int64_t l1SizeRemain,
                                          int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor,
                                          bool allGM = false, bool autoBaseMNK = false);
    bool AdaBlockSparseAttentionCheckBmm2(SglAdaBlockSparseAttentionTilingData &tilingData,
                                          AscendC::tiling::TCubeTiling &bmm1TilingData, int64_t l1SizeRemain,
                                          int64_t l0CSize, uint32_t sOuterFactor, uint32_t sInnerFactor,
                                          uint32_t dSplitFactor, bool allGM = false, bool autoBaseMNK = false);
    void AdaBlockSparseAttentionSetTensorSize(SglAdaBlockSparseAttentionTilingData &tilingData,
                                              PromptAttentionSingleCoreTensorSize &tensorSize, uint32_t sOuterFactor,
                                              uint32_t sInnerFactor);
    bool AdaBlockSparseAttentionCheckArgsLegal(SglAdaBlockSparseAttentionTilingData &tilingData, int64_t ubSize,
                                               int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
                                               uint32_t &sOuterFactor, uint32_t sInnerFactor, bool &updateDiv,
                                               uint32_t maskTypeSize, uint32_t dSplitFactor);
    ge::graphStatus AdjustBasicBlock(SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t &sOuterFactor);
    ge::graphStatus AdaBlockSparseAttentionApiTiling(SglAdaBlockSparseAttentionTilingData &tilingData,
                                                     uint32_t typeSize, uint32_t sOuterFactor,
                                                     uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor);
    ge::graphStatus GetRectangleFactor(uint32_t seqSplit, std::queue<uint32_t> &sQueue, int32_t threshold = 16);
    ge::graphStatus SetInputLayout(const char *layout);
    bool GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor, const uint32_t typeByteSize);
    uint32_t CalculateL1SizeUsed(SglAdaBlockSparseAttentionTilingData &tilingData, const uint32_t typeByteSize);
    bool CheckInputDimAndHeadNum(ContextParamsForBSATiling &contextKeyParams, uint32_t nQAttr, uint32_t nKVAttr);
    bool SetTilingHeadNumRatio(ContextParamsForBSATiling &contextKeyParams, const int32_t numQueryHeads,
                               const int32_t numKeyValueHeads, SglAdaBlockSparseAttentionTilingData &tilingData);
    void AdaBlockSparseAttentionInitOutputSplit(uint64_t totalSize, SglAdaBlockSparseAttentionTilingData &tilingData,
                                                uint32_t curCoreNum);
    void AdaBlockSparseAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize,
                                                          SglAdaBlockSparseAttentionTilingData &tilingData);
    void Align(uint32_t &num);
    ge::graphStatus GetBasicShape(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                  const gert::StorageShape *queryShape, const gert::StorageShape *keyShape,
                                  const uint32_t n);
    ge::graphStatus GetBasicShape310P(uint32_t &b, uint32_t &bKV, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape,
                                      const uint32_t n, size_t actualLenDims, size_t actualLenDimsKV);
    ge::graphStatus GetBasicShape910B(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape,
                                      const uint32_t n);
    size_t GetBSAWorkSpaceSize(SglAdaBlockSparseAttentionTilingData &tilingData);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType);
    ge::graphStatus CheckKeyValueParamsConsistency(const ContextParamsForBSATiling &contextKeyParams);
    bool CheckActualSeqLength(ContextParamsForBSATiling &contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
                              const gert::Tensor *actualSeqLenQ, const gert::Tensor *actualSeqLenKV,
                              InputLayout inLayout, SglAdaBlockSparseAttentionTilingData &tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForBSATiling &contextKeyParams, const gert::StorageShape *pseShiftShape,
                                   uint32_t b, uint32_t n, uint32_t s1, uint32_t s2);
    bool CheckPATypeAndShape(ContextParamsForBSATiling &contextKeyParams, const gert::Tensor *actualSeqLenKV, int32_t b,
                             int32_t n, int32_t h, int32_t headNumRatio);
    bool CheckAttenMaskShape(ContextParamsForBSATiling &contextKeyParams, const int32_t *sparseMode,
                             const gert::StorageShape *attenMaskShape, uint32_t sQ, uint32_t sK, uint32_t batchSize);
    bool CheckAntiquantParamsShape(ContextParamsForBSATiling &contextKeyParams,
                                   const gert::StorageShape *antiquantScaleShape,
                                   const gert::StorageShape *antiquantOffsetShape, const uint32_t n, const uint32_t d,
                                   const uint32_t h, SglAdaBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus CheckPostQuantParams(const ContextParamsForBSATiling &contextKeyParams, uint32_t h,
                                         uint32_t n) const;
    ge::graphStatus AdaBlockSparseAttentionCVDiffSetTensorSize(SglAdaBlockSparseAttentionTilingData &tilingData,
                                                               PromptAttentionSingleCoreTensorSize &tensorSize,
                                                               uint32_t sOuterFactor, uint32_t sInnerFactor,
                                                               uint32_t softmaxSOuterFactor);
    bool AdaBlockSparseAttentionComputeCVDiffParams(SglAdaBlockSparseAttentionTilingData &tilingData, int64_t ubSize,
                                                    int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
                                                    uint32_t &sOuterFactor, uint32_t &sInnerFactor,
                                                    uint32_t maskTypeSize, uint32_t &softmaxSOuterFactor);
    bool FindOptimalTilingBasicBLock(SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t &sOuterFactor,
                                     uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor, int64_t ubSize,
                                     uint32_t typeByteSize, uint32_t maskTypeSize);
    bool FindOptimalTilingSouter(SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t &sOuterFactor,
                                 uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor, int64_t ubSize,
                                 uint32_t typeByteSize, uint32_t maskTypeSize);
    void InferTilingMod(const ContextParamsForBSATiling &contextKeyParams, const std::vector<int64_t> &actualSeqLengths,
                        const std::vector<int64_t> &actualSeqLengthsKV, uint32_t actualSeqArrayLen, uint32_t hDivN,
                        uint32_t seqInnerSize, int32_t sparseModeVal);
    ge::graphStatus AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middle_actualSeqLengths, int64_t ubSize,
                                   int64_t l1Size, int64_t l0CSize, uint32_t maskElemSize, uint32_t &sOuterFactor,
                                   uint32_t &sInnerFactor, SglAdaBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t maskElemSize,
                                         uint32_t &sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
                                         SglAdaBlockSparseAttentionTilingData &tilingData);
    bool CheckSparseModeRightDown(ContextParamsForBSATiling &contextKeyParams,
                                  const std::vector<int64_t> &actualSeqLengths,
                                  const std::vector<int64_t> &actualSeqLengthsKV, size_t lenDims);
    ge::graphStatus GetAndCheckEmptyQueryShape(ContextParamsForBSATiling &contextKeyParams,
                                               const gert::StorageShape *queryShape) const;
    void UpdateTilingKeyFlag(ContextParamsForBSATiling &contextKeyParams, uint64_t &tilingKey);
    int64_t AdaBlockSparseAttentionSetMsdUbSize(SglAdaBlockSparseAttentionTilingData &tilingData,
                                                PromptAttentionSingleCoreTensorSize &tensorSize,
                                                int32_t sInnerFactorTmp) const;
    ge::graphStatus CheckIOType(ContextParamsForBSATiling &contextKeyParams,
                                SglAdaBlockSparseAttentionTilingData &tilingData, int32_t &outputDataTypeSize);
    ge::graphStatus CheckD(ContextParamsForBSATiling &contextKeyParams);  // codespell:ignore
    ge::graphStatus CheckDimNums(ContextParamsForBSATiling &contextKeyParams);
    ge::graphStatus CheckMaskType(ContextParamsForBSATiling &contextKeyParams,
                                  SglAdaBlockSparseAttentionTilingData &tilingData, uint32_t &maskElemSize);
    void SetMaskSize(const gert::StorageShape *attenMaskShape, SglAdaBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus CheckShape(ContextParamsForBSATiling &contextKeyParams, const gert::StorageShape *queryShape,
                               const gert::StorageShape *keyShape, const gert::StorageShape *valueShape,
                               const gert::StorageShape *outShape, const gert::StorageShape *pseShiftShape,
                               const gert::StorageShape *attenMaskShape);
    ge::graphStatus CheckBaseAPISupportScenarios(ContextParamsForBSATiling &contextKeyParams);
    size_t GetBSABaseApiWorkSpaceSize(uint32_t &blockDimToBeSet);
    ge::graphStatus TilingGetBaseApiTilingKeyAttentionAscendC(uint64_t &tilingKey,
                                                              ContextParamsForBSATiling &contextKeyParams);
    ge::graphStatus CheckBaseApiRequiredInput(ContextParamsForBSATiling &contextKeyParams);
    ge::graphStatus CheckBaseApiOptionalInput(ContextParamsForBSATiling &contextKeyParams);
    ge::graphStatus CheckBaseApiPse(ContextParamsForBSATiling &contextKeyParams,
                                    const gert::StorageShape *pseShiftShape);
    void SetBaseApiTilingData(ContextParamsForBSATiling &contextKeyParams, std::vector<int64_t> &actualSeqLengths,
                              std::vector<int64_t> &actualSeqLengthsKV);
    void SetBaseApiSeqTilingData(ContextParamsForBSATiling &contextKeyParams, std::vector<int64_t> &actualSeqLengths,
                                 std::vector<int64_t> &actualSeqLengthsKV);

    ge::graphStatus CheckBaseApiMaskBasic(ContextParamsForBSATiling &contextKeyParams,
                                          const gert::StorageShape *pseShiftShape, bool isLongSeq, uint32_t batchSize);
    ge::graphStatus CheckBaseApiMaskVal(ContextParamsForBSATiling &contextKeyParams,
                                        const gert::StorageShape *pseShiftShape,
                                        const std::pair<std::vector<int64_t>, std::string> maskShape);
    ge::graphStatus CheckBaseApiAlibiMask(ContextParamsForBSATiling &contextKeyParams,
                                          const gert::StorageShape *pseShiftShape, uint32_t batchSize,
                                          int32_t maxSeqLen, int32_t maxKvSeqLen, uint32_t kvHead, bool compressHead);
    ge::graphStatus CheckBaseApiNormMask(ContextParamsForBSATiling &contextKeyParams,
                                         const gert::StorageShape *pseShiftShape, int32_t maskType, uint32_t batchSize,
                                         int32_t maxSeqLen, int32_t maxKvSeqLen, bool compressHead);

    ge::graphStatus SetBaseApiPseInfo(ContextParamsForBSATiling &contextKeyParams,
                                      const gert::StorageShape *pseShiftShape);
    void SetBaseApiOtherMaskInfo(ContextParamsForBSATiling &contextKeyParams, const gert::StorageShape *pseShiftShape);
    ge::graphStatus SetBaseApiAlibiMaskInfo(ContextParamsForBSATiling &contextKeyParams,
                                            const gert::StorageShape *pseShiftShape);
    ge::graphStatus AtbSplitBlock(ContextParamsForBSATiling &contextKeyParams);
    uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum);
    bool CalcUBSize();
    void SetDataCopyTransposeTiling();
    void SetSoftMaxTiling();
    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1);
    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                         matmul_tiling::MatmulApiTiling &bmm1, matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    void CalcS1S2BasicBlock(const BufferNum &bufferNum);
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum);
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock);
    bool IsBasicBlockInSoftMax(const ge::Shape &shape);
    void GetBufferNum(BufferNum &bufferNum);
    void SetMultiBatchCoreParams();
    void MatchTemplate(uint32_t valueD);
    void SetTensorSizeParams();
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, BSAMultiCoreParams &multiCoreParams);

    // TND新增
    bool InputLayoutIsTNDLike() const;
    int64_t GetTFromInputShape(const gert::StorageShape *shape) const;
    int64_t GetNFromInputShape(const gert::StorageShape *shape) const;
    int64_t GetTFromOutputShape(const gert::StorageShape *shape) const;
    int64_t GetNFromOutputShape(const gert::StorageShape *shape) const;
    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen);
    void SetMultiCoreParamsTND();
    void SetSparseParamsTND();
    bool InitSparseValidArrayTND(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    bool SetSparseStartIdxTND(const std::vector<int64_t> &sparseValidArray, BSAMultiCoreParams &multiCoreParams);
    int64_t GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, BSAMultiCoreParams &multiCoreParams,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx);
    bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum, int64_t totalSize,
                       const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    ge::graphStatus CheckInputShapeWhenLayoutIsTND(ContextParamsForBSATiling &contextKeyParams);
    ge::graphStatus CheckActSeqWhenLayoutIsTND(ContextParamsForBSATiling &contextKeyParams);

protected:
    ContextParamsForBSATiling *contextKeyParamsPtr = nullptr;
    int64_t ubSizeRemain = 1;
    bool isSOuterNoTail = true;
    bool isSInnerNoTail = true;
    bool isDNoTail = true;
    bool enableKvAntiquant = false;
    bool enableMsd = false;
    bool enableQuantBF16 = false;
    bool enableMatmulNorm = false;
    bool enablePA = false;
    bool isKVHasPrefix = false;
    InputLayout inputLayout = InputLayout::BSH;
    ge::DataType inputType{ge::DT_FLOAT16};
    ge::DataType outputType{ge::DT_FLOAT16};
    ge::DataType intputKeyType{ge::DT_FLOAT16};
    ge::DataType intputValueType{ge::DT_FLOAT16};
    ge::DataType pseShiftElemType{ge::DT_FLOAT16};
    uint32_t dataTypeSize = FLOAT32SIZE;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseShiftElemSize = 0;
    uint32_t pseMaskMaxSize = 0;
    uint32_t pseShiftBatch = 0;
    uint32_t pseShiftS1 = 0;
    uint32_t pseShiftS2 = 0;
    uint32_t usePseShift = 0;
    // In the PA scenario, there is no S2 axis. Use the change amount to
    // normalize the S2 length in both PA and non PA scenarios
    uint32_t tmpS2 = 0;
    int32_t blockTableDim2 = 1;
    int32_t PABlockNumSum = 1;
    uint32_t maskTypeByteNum;
    uint32_t maxQuerySeq = 0;
    int64_t apiTmpSize = 1;
    uint32_t softmaxDataTypeNZ_ = FLOAT32SIZE;
    uint32_t softmaxDataTypeSize = FLOAT32SIZE;  // BF16 calculates through FP32
    platform_ascendc::SocVersion curShortSocName;
    uint32_t dataTypeSize_ = 4;
    uint32_t layoutType = 0;
    uint32_t PAlayoutType = 0;
    platform_ascendc::PlatformAscendC ascendcPlatform;
    TilingMod tilingMod = TilingMod::CVSAME;
    SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
    uint32_t splitD = 0;
    uint32_t splitS2 = 1;  // It can only be 0 when the D axis is split
    uint64_t innerPrecise = HIGH_PERFORMANCE;
    size_t defaultSysWorkspaceSize;
    matmul_tiling::PlatformInfo ascendPlatformInfo;

    int64_t alignedS1 = 0;
    int64_t alignedS2 = 0;
    int64_t alignedD = 0;

    int64_t s1BasicBlock = 0;
    int64_t s2BasicBlock = 0;
    int64_t s1BasicBlockBest = 0;
    int64_t s1VecBasicBlock = 0;
    int64_t dBasicBlock = 0;
    int64_t batchBasic = 1LL;
    int64_t nRatio = 0;

    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t dSize = 0;
    int64_t valueDSize = 0;
    int64_t s1SparseValidSize = 0;
    int64_t s2SparseValidSize = 0;

    int64_t apiMaxUBSize = 0;

    bool atbRunFlag_ = false;
    bool mlaRunFlag_ = false;

    // TND新增
    int64_t realT1Size = 0;
    int64_t realT2Size = 0;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    int64_t accumS1 = 0;
    int64_t accumS2 = 0;
    int64_t bandIndex = 0;

    int64_t bSize = 0;
    int64_t gSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t s1StrideSize = 0;  // query Shape S inner axes, for bmm1
    int64_t s2StrideSize = 0;  // key Shape S inner axes, for bmm1
    int64_t maxS1Val = 0;
    int64_t maxS2Val = 0;

    int64_t h1 = 0;
    int64_t h2 = 0;

    int64_t s2sizeLimitMax = 1024;
    int64_t accumS1BlockNum = 0;

    bool isSameAB = true;
    AdaBlockSparseAttentionBaseApiTilingData baseApiTilingData{};
    MLAGeneralTilingData mlaTilingData{};
};
// end of class AdaBlockSparseAttention
BSA_EXTERN_C ge::graphStatus TilingAdaBlockSparseAttention(gert::TilingContext *context);
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SPARSEBSA_H_
