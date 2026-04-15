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

#ifndef BLOCK_SPARSE_ATTENTION_S1S2_BNS1_X910_BASE_H
#define BLOCK_SPARSE_ATTENTION_S1S2_BNS1_X910_BASE_H
#include <type_traits>
#include "block_sparse_attention_base.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"

#define DEBUG_PRINT(str) do { printf("[DEBUG] Line: %d %s\n", __LINE__, str); } while (0)
#define DUMP_TENSOR(mmResUb, m, n ) \
    do {printf("line %d\n", __LINE__); \
        uint32_t array_ ## __LINE__[] = {static_cast<uint32_t>(m), static_cast<uint32_t>(n)}; \
        AscendC::ShapeInfo shapeInfo_ ## __LINE__(2, array_ ## __LINE__); \
        AscendC::DumpTensor(mmResUb, 2, (m)*(n), shapeInfo_ ## __LINE__); \
    } while (0)


constexpr uint32_t FP32_ONE_BLOCK_SIZE_BSA = 8;
constexpr uint32_t BYTE_BLOCK_BSA = 32;
constexpr uint32_t REPEAT_BLOCK_BYTE_BSA = 256;

using namespace matmul;

#define BSA_InitBuffer(para1, para2)                                                                           \
    do {                                                                                                       \
        pipe->InitBuffer(para1, para2);                                                                        \
    } while (0)
#define BSA_InitQueue(para1, para2, para3)                                                                     \
    do {                                                                                                       \
        pipe->InitBuffer(para1, para2, para3);                                                                 \
    } while (0)

enum class BSALayout {
    BSH = 0,
    BNSD,
};

enum class MatMulType {
    MM_MDL = 0,
    MM_NORM,
    MM_IBSHARE_NORM,
    MM_PA,
    MM_SP,
};

enum class MsdMode {
    MSD_OFF = 0,
    MSD_ON,
};

template <const MatMulType MM_TYPE>
struct GetMatmulConfig {
    static constexpr MatmulConfig mmcfg_value = CFG_MDL;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_MDL> {
    static constexpr MatmulConfig mmcfg_value = CFG_MDL;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_NORM> {
    static constexpr MatmulConfig mmcfg_value = CFG_NORM;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_IBSHARE_NORM> {
    static constexpr MatmulConfig mmcfg_value = CFG_IBSHARE_NORM;
    static constexpr bool ibshare_value = true;
};

template <>
struct GetMatmulConfig<MatMulType::MM_PA> {
    static constexpr MatmulConfig mmcfg_value = GetNormalConfig(false, false, false,
        BatchMode::BATCH_LESS_THAN_L1, false);
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_SP> {
    static constexpr MatmulConfig mmcfg_value = GetNormalConfig(false, false, false,
        BatchMode::BATCH_LESS_THAN_L1, false);
    static constexpr bool ibshare_value = false;
};

template <BSALayout L, typename T, typename U, typename O = T, typename KV_T = T, Mode M = Mode::HighPerformance,
    const MatMulType MM_TYPE_TMP = MatMulType::MM_SP, const bool F = false,  const MsdMode MSD_MODE = MsdMode::MSD_OFF,
    typename...Args>
struct BSAType {
    using inputType = T;
    using maskType = U;
    using outputType = O;
    using kvInputType = KV_T;
    static constexpr BSALayout layout = L;
    static constexpr Mode calcMode = M;
    static constexpr MatMulType MM_TYPE = MM_TYPE_TMP;
    static constexpr MatmulConfig mmCFG = GetMatmulConfig<MM_TYPE>::mmcfg_value;
    static constexpr bool ibShare = GetMatmulConfig<MM_TYPE>::ibshare_value;
    static constexpr bool enablePrefix = F;
    static constexpr MsdMode msdMode = MSD_MODE;
};

constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xC77FE000;
constexpr uint32_t MM2_SINGLE_K_ALIGN_SIZE = 32;
constexpr uint32_t SINGLE_PROCESS_SINNER_BMMTAIL_LIMIT = 32;
constexpr uint32_t BSA_BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BSA_TEMP_BUFFER_SIZE_BYTE = 64 * 1024;

#define TEMPLATE_LAYOUT template<BSALayout layout = BSAT::layout>
#define TYPENAME_BSH_VOID typename std::enable_if<layout == BSALayout::BSH, void>::type
#define TYPENAME_BNSD_VOID typename std::enable_if<layout == BSALayout::BNSD, void>::type
#define TYPENAME_BSH_INT64 typename std::enable_if<layout == BSALayout::BSH, int64_t>::type
#define TYPENAME_BNSD_INT64 typename std::enable_if<layout == BSALayout::BNSD, int64_t>::type

#define TEMPLATE_MASKTYPE template<typename _maskType>
#define TYPENAME_MASKTYPE_BOOL_VOID typename std::enable_if<std::is_same_v<_maskType, bool>, void>::type
#define TYPENAME_MASKTYPE_INT8_VOID typename std::enable_if<std::is_same_v<_maskType, uint8_t>, void>::type
#define TYPENAME_MASKTYPE_HALF_VOID typename std::enable_if<std::is_same_v<_maskType, half>, void>::type

struct BSAComputeParam {
    uint32_t sInnerLoopIdx; // for debug
    bool isFirstInnerIter;
    bool isSecondInnerIter;
    bool isLastInnerIter;
    bool isLastBlockCausalDiag;

    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerBmmTail;
    uint32_t mm1SingleCoreN;
    
    int64_t tensorAOffset;
    int64_t tensorBCoreOffset;
    uint32_t attentionOutOffset;

    uint32_t sOuterOffset;
    uint32_t batchNOffset;
    uint32_t multiSeqOffset;
    uint32_t multiSeqOffsetBSNDOut;
    
    int32_t actualSeqLengthPerBatch = 0;
    int32_t actualSeqLengthKVPerBatch = 0;
    uint32_t sparseBlockCount;
    uint32_t gmPingpong;
    uint32_t pos[8];
};

template <typename BSAT>
class BlockSparseAttentionS1s2Bns1X910Base {
public:
    __aicore__ inline BlockSparseAttentionS1s2Bns1X910Base() {};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                __gm__ uint8_t* value, __gm__ uint8_t* sparseMask,  __gm__ uint8_t* sparseCount,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
                                __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* keySharedPrefix,
                                __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                const BlockSparseAttentionTilingData* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe);
    __aicore__ inline void Process();

    using FT = float;
    using T = typename BSAT::inputType;
    using KV_T = typename BSAT::kvInputType;
    using U = typename BSAT::maskType;
    using O = typename BSAT::outputType;
    using mmBiasType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::mmBiasType;
    using mmOutputTypeTmp = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::mmOutputType;
    using computeType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::softmaxType;
    using pseShiftType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename BlockSparseAttentionTypeTraits<T, BSAT::calcMode>::pseShiftCastType;

    using mmOutputType =
        typename AscendC::Conditional<BSAT::msdMode == MsdMode::MSD_ON, int32_t, mmOutputTypeTmp>::type;

    static constexpr int32_t softmaxTypeByteNum = BYTE_BLOCK_BSA / sizeof(computeType);
    static constexpr FT BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0f;
    constexpr static bool USE_BLOCK_SPARE = BSAT::MM_TYPE == MatMulType::MM_SP;
    static constexpr int32_t BSA_PARAMS_QUEUE_CAPBABILITY = 4;

    template <class SRC_T>
    static __aicore__ void CopyND2NZ(const LocalTensor<SRC_T>& dst, const GlobalTensor<SRC_T>& src, const int row,
        const int col, const int height, const int width, const int gCol, const int ndNum = 1,
        const int srcNdMatrixStride = 0, const int dstNzMatrixStride = 1, const bool kAlignToC0Size = false,
        const int dstNzC0Stride = 0) {  // The minimum range of parameter values is 1.
        int64_t srcOffset = (int64_t)row * (int64_t)gCol + (int64_t)col;
        int32_t alignNum = 16;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;
        if (kAlignToC0Size) {
            if constexpr (IsSameType<SRC_T, int8_t>::value) {
                alignNum = 32;
            } else if constexpr (IsSameType<SRC_T, float>::value) {
                alignNum = 8;
            }
        }
        nd2nzParams.dstNzC0Stride = Ceil(dstNzC0Stride, alignNum) * alignNum;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(dst, src[srcOffset], nd2nzParams);
    }

    template <bool isLayoutBSH>
    static __aicore__ void bmm1CopyB1Sparse(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row,
        int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        //* let kvconcat be torch.tensor([kvblock needcompute])
        //* [TILE]  : row=0, col=0, useK=64, useN=256 // [TILE] 256: row=1, col=0, useK=64, useN=256
        GlobalTensor<uint32_t> bmmLocalInfo;
        bmmLocalInfo.SetGlobalBuffer((__gm__ uint32_t *)dataPtr,  16);  // Align to 8
        uint32_t ii = 0;
        // tilingData
        uint32_t sparseBlockSize = bmmLocalInfo.GetValue(ii++);
        uint32_t kvHeadNum       = bmmLocalInfo.GetValue(ii++);
        uint32_t kvD             = bmmLocalInfo.GetValue(ii++);
        uint32_t baseK           = bmmLocalInfo.GetValue(ii++);
        uint32_t baseN           = bmmLocalInfo.GetValue(ii++);
        uint32_t Kb              = bmmLocalInfo.GetValue(ii++);
       
        // bmm1 row direction corresponds to the k axis and D axis; col direction corresponds to the N axis and S2 axis.
        // The offset of the current useN block in the single block in the S2 direction.
        uint32_t copyXthRowOfKvConcat = col * baseN;
        uint32_t copyFinishRowCnt = 0;
        uint32_t copyRowCnt = 0;
        uint64_t curKvOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        constexpr uint32_t colElementCnt = 32 / sizeof(T);
        // 1. useN <= blockSize : copy part of the blockSize 2. useN > blockSize: Multiple copies of a callback 3.
        // Tail block
        while (copyFinishRowCnt < useN) {
            uint32_t copyingXthBlock = copyXthRowOfKvConcat / sparseBlockSize;
            uint32_t offsetInBlock   = copyXthRowOfKvConcat % sparseBlockSize;
            uint32_t sparseBlockId = bmmLocalInfo.GetValue(8 + copyingXthBlock);
            uint32_t realkvS2Offset  = offsetInBlock + sparseBlockId * sparseBlockSize; // kv s2 offset

            copyRowCnt = sparseBlockSize - offsetInBlock; // tail block
            if (copyFinishRowCnt + copyRowCnt > useN) {  // Copy more than needed
                copyRowCnt = useN - copyFinishRowCnt;
            }
            curKvOffset = isLayoutBSH == 1 ? realkvS2Offset * kvHeadNum * kvD :  realkvS2Offset * kvD;
            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T *)gm);
            LocalTensor<T> dst = bMatrix.template ReinterpretCast<T>();

            baseRowOffsetInSingle = 0;
            baseColOffsetInSingle = row * baseK; // headDim
            CopyND2NZ(dst[copyFinishRowCnt * colElementCnt], src[curKvOffset], baseRowOffsetInSingle,
                baseColOffsetInSingle, copyRowCnt, useK, Kb, 1, 0, 1, true, useN);
 
            copyFinishRowCnt += copyRowCnt;
            copyXthRowOfKvConcat += copyRowCnt;
        }
    }

    template <bool isLayoutBSH>
    static __aicore__ void bmm2CopyB1Sparse(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row,
        int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<uint32_t> bmmLocalInfo;
        bmmLocalInfo.SetGlobalBuffer((__gm__ uint32_t *)dataPtr,  16);
        uint32_t ii = 0;
        uint32_t sparseBlockSize = bmmLocalInfo.GetValue(ii++);
        uint32_t kvHeadNum       = bmmLocalInfo.GetValue(ii++);
        uint32_t kvD             = bmmLocalInfo.GetValue(ii++);
        uint32_t baseK           = bmmLocalInfo.GetValue(ii++);
        uint32_t baseN           = bmmLocalInfo.GetValue(ii++);
        uint32_t Kb              = bmmLocalInfo.GetValue(ii++);
        uint32_t N               = bmmLocalInfo.GetValue(ii++);
        // The offset of the current useN block in the single block in the S2 direction.
        uint32_t copyXthRowOfKvConcat = row * baseK;
        uint32_t copyFinishRowCnt = 0;
        int64_t blockRowOffsetInSingle = 0;
        uint32_t copyRowCnt = 0;
        uint64_t curKvOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        constexpr uint32_t colElementCnt = 32 / sizeof(T);
        // 1. useN <= blockSize : copy part of the blockSize 2. useN > blockSize: Multiple copies of a callback 3.
        // Tail block
        while (copyFinishRowCnt < useK) {
            uint32_t copyingXthBlock = copyXthRowOfKvConcat / sparseBlockSize;
            uint32_t offsetInBlock   = copyXthRowOfKvConcat % sparseBlockSize;
            uint32_t sparseBlockId   = bmmLocalInfo.GetValue(8 + copyingXthBlock);
            uint32_t realkvS2Offset    = offsetInBlock + sparseBlockId * sparseBlockSize;

            copyRowCnt = sparseBlockSize - offsetInBlock;
            if (copyFinishRowCnt + copyRowCnt > useK) {
                copyRowCnt = useK - copyFinishRowCnt;
            }
            curKvOffset = isLayoutBSH == 1 ? realkvS2Offset * kvHeadNum * kvD :  realkvS2Offset * kvD;
         
            GlobalTensor<T> src;  // Pseudo quantization scenarios are also dequantized for fp16, and storage to GM.
            src.SetGlobalBuffer((__gm__ T *)gm);
            LocalTensor<T> dst = bMatrix.template ReinterpretCast<T>();

            baseRowOffsetInSingle = 0;  // The offset of the current base starting point in single.
            baseColOffsetInSingle = col * baseN;
            CopyND2NZ(dst[copyFinishRowCnt * colElementCnt], src[curKvOffset], baseRowOffsetInSingle,
                baseColOffsetInSingle, copyRowCnt, useN, N, 1, 0, 1, true, useK);

            copyFinishRowCnt += copyRowCnt;
            copyXthRowOfKvConcat += copyRowCnt;
        }
    }

    // define matmul
    using MM_IN_T = typename AscendC::Conditional<BSAT::msdMode == MsdMode::MSD_ON, KV_T, T>::type;
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, true, LayoutMode::NONE, BSAT::ibShare>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, mmOutputType>;
    static constexpr bool isBSH = (BSAT::layout == BSALayout::BSH);
    using PACBmm1 = Matmul<a1Type, b1Type, c1Type, bias1Type, BSAT::mmCFG, matmul::MatmulCallBackFunc<nullptr,
        nullptr, bmm1CopyB1Sparse<isBSH> >>;
     // PA doesn't need to carry in large packages temporarily.
    PACBmm1 mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false, LayoutMode::NONE, BSAT::ibShare>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmOutputType>;
    using PACBmm2 = Matmul<a2Type, b2Type, c2Type, bias2Type, BSAT::mmCFG, matmul::MatmulCallBackFunc<nullptr,
        nullptr, bmm2CopyB1Sparse<isBSH> > >;
    PACBmm2 bmm2;

protected:
    const BlockSparseAttentionTilingData* __restrict tilingData;
    TPipe* pipe;

    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECOUT, 1> softmaxOutQueue;
    TBuf<> tmpBuff;
    TBuf<> PABmm1UB;
    TBuf<> PABmm2UB;
    TBuf<> softmaxExpBuf;
    TBuf<> preBmm2Buf;
    TBuf<> causalBuf;
    TBuf<> sparseIdBuf;
    TBuf<> taskBuf;
    TBuf<> loadBuffer;

    event_t bmm1ResCopyInEvent[2];
    event_t bmm2ResCopyInEvent[2];
    event_t bmm1ResCopyOutEvent[2];
    event_t attenOutCopyOut;

    bool copyOutPrevIter = false;
    uint32_t softmaxSouterStepLen = 0;
    bool causal = false;
    uint32_t batchSize = 0;
    bool causalUbReady = false;
    uint32_t sparseBlockSize = 0;

    LocalTensor<uint32_t> bmm1LocalInfo;
    LocalTensor<uint32_t> bmm2LocalInfo;
    LocalTensor<int32_t> blockIdUb;
    LocalTensor<int16_t> taskQid;
    LocalTensor<half> taskCost;
    LocalTensor<int32_t> cores;
    LocalTensor<float> loads;

    LocalTensor<computeType> mmResUb[2];
    LocalTensor<float> softmaxMaxUb;
    LocalTensor<float> softmaxSumUb;
    LocalTensor<computeType> softmaxExpUb;

    __gm__ uint8_t* key_ptr;
    __gm__ uint8_t* value_ptr;
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;

    __gm__ uint32_t* bmm1CBDataPtr[2];
    __gm__ uint32_t* bmm2CBDataPtr[2];

    GlobalTensor<int32_t> sparseBlockCountGm;
    GlobalTensor<T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<int8_t> sparseMaskGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<mmOutputType> workspaceGm;

    GlobalTensor<mmOutputType> bmm1ResGmDb[2];
    GlobalTensor<MM_IN_T> softmaxResGmDb[2];
    GlobalTensor<mmOutputType> bmm2ResGmDb[2];

    GlobalTensor<uint32_t> bmm1CBDataGm[2];
    GlobalTensor<uint32_t> bmm2CBDataGm[2];

    BSAComputeParam bsaParamsQueue[BSA_PARAMS_QUEUE_CAPBABILITY];
    BSAComputeParam *tailParams;
    BSAComputeParam *headParams;
    BSAComputeParam *preHeadParams;
    int32_t headId = 0;
    int32_t tailId = 0;
    int32_t queSize = 0;
    int32_t queSizeLimit = BSA_PARAMS_QUEUE_CAPBABILITY - 2;
    int32_t solvedTaskCount = 0;

    int64_t tmp_block_idx = 0;

    int64_t tensorACoreOffset = 0;
    int64_t tensorBCoreOffset = 0;
    uint32_t s2InCurrentBatch = 0;
    AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;

    uint32_t mm1SingleCoreNPrev = 0;
    uint32_t mm2MStridePrev = 0;
    uint32_t mm2KaStridePrev = 0;

    // tilingdata
    uint32_t singleProcessSOuterSizeWhole = 0;
    uint32_t singleProcessSOuterSizeTail = 0;
    uint32_t mmResUbSize = 0;

    uint32_t softmaxMaxSize = 0;
    uint32_t softmaxSumSize = 0;
    uint32_t softmaxExpSize = 0;
    uint32_t spmTmpSize = 0;
    uint32_t scmTmpSize = 0;
    uint32_t bmm2ResUbSize = 0;

    uint32_t headNumRatio = 0;
    uint32_t headNumSize = 0;
    uint32_t headSize = 0;
    uint32_t sparseMaskS1Size = 0;
    uint32_t sparseMaskS2Size = 0;
    uint32_t selectSpaceUbSize = 0;
    uint32_t maskBmm2ShareSize = 0;
    uint32_t sparseMaskTotalSize = 0;
    uint32_t tmpBuffSize = 0;

    SoftMaxTiling softmaxFlashTilingData;
    SoftMaxTiling softmaxFlashTilingDataNew;
    
    uint32_t MultiHeadQ = 0;
    uint32_t MultiHeadKV = 0;
    int64_t seqListOffset = 0;

    bool isGlobalFirstCompute;

    bool isActualLenDimsNull;
    bool isActualLenDimsKVNull;
    uint32_t isKvContinuous = 0;

    __aicore__ inline void Bmm2UpdateDivNoTail(LocalTensor<computeType>& bmm2ResPreUb,
        LocalTensor<float>& softmaxSumUb);

    __aicore__ inline void UpdateVmul(LocalTensor<computeType>& softmaxExpUb);

    TEMPLATE_LAYOUT
    __aicore__ inline void DataCopyTransposeOutBNSD(LocalTensor<computeType> &bmm2ResUb) {

        uint64_t copySize = this->preHeadParams->singleProcessSOuterSize * this->headSize;

        int64_t attentionOutTokenOffset = 0; // nextTokensOffset * this->headSize;

        struct DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copySize / (BYTE_BLOCK_BSA / sizeof(O));
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        if constexpr (BSAT::calcMode == Mode::HighPrecision ||  IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> FinalResUb = bmm2ResUb.template ReinterpretCast<T>();

            pipe_barrier(PIPE_V);
            Cast(FinalResUb, bmm2ResUb, RoundMode::CAST_ROUND, bmm2ResUbSize);

            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset + attentionOutTokenOffset],
                     FinalResUb[attentionOutTokenOffset], dataCopyParams);
        } else {
            // Before copyOut, synchronous calculation.
            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset + attentionOutTokenOffset],
                     bmm2ResUb[attentionOutTokenOffset], dataCopyParams);
        }
    }

    TEMPLATE_LAYOUT
    __aicore__ inline void DataCopyTransposeOutBSH(LocalTensor<computeType> &bmm2ResUb) {
        TransposeParams transposeParams;
        transposeParams.bIndex = 0;
        transposeParams.nIndex = this->preHeadParams->batchNOffset;
        transposeParams.sIndex = this->preHeadParams->sOuterOffset;
        transposeParams.hNIndex = 0;
        int64_t preTokensOffset = 0;
        int64_t nextTokensOffset = 0;

        CopyTransposeTiling transposeTilingData22 = tilingData->transposeTilingDataRect;
        transposeTilingData22.srcShapeS =
            this->preHeadParams->singleProcessSOuterSize - preTokensOffset - nextTokensOffset;
        transposeTilingData22.invalidParamCopyTransposeTiling = 0;
        transposeParams.sIndex = transposeParams.sIndex + nextTokensOffset;

        int64_t multiSeqOffset = this->preHeadParams->multiSeqOffset;
        if constexpr (BSAT::calcMode == Mode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> FinalResUb = bmm2ResUb.template ReinterpretCast<T>();

            pipe_barrier(PIPE_V);
            Cast(FinalResUb, bmm2ResUb, RoundMode::CAST_ROUND, bmm2ResUbSize);

            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopyTranspose2<O> (attentionOutGm, FinalResUb[nextTokensOffset * this->headSize],
                                   CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                                   transposeTilingData22, multiSeqOffset);
        } else {
            // Before copyOut, synchronous calculation.
            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopyTranspose2<O> (attentionOutGm, bmm2ResUb[nextTokensOffset * this->headSize],
                                   CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                                   transposeTilingData22, multiSeqOffset);
        }
    }

    __aicore__ inline int64_t CalMultiSeqOffset(int sIdx)
    {
        int64_t multiSeqOffset = 0;
        int64_t queryLeftpaddingSize = 0; // this->GetQueryLeftPaddingSize(sIdx);
        if constexpr (BSAT::layout == BSALayout::BNSD) {
            multiSeqOffset = (int64_t)sIdx * this->tilingData->promptAttentionBaseParams.seqSize *
                (int64_t)this->MultiHeadQ + queryLeftpaddingSize * (int64_t)this->headSize; // BNSD
        } else {
            multiSeqOffset = (int64_t)sIdx * this->tilingData->promptAttentionBaseParams.seqSize *
                (int64_t)this->MultiHeadQ + queryLeftpaddingSize * (int64_t)this->MultiHeadQ; // BSH
        }

        if (this->tilingData->promptAttentionBaseParams.isBSNDOut) {
            this->tailParams->multiSeqOffsetBSNDOut = (int64_t)sIdx *
            (int64_t)this->tilingData->promptAttentionBaseParams.seqSize *
            (int64_t)this->MultiHeadQ + queryLeftpaddingSize * (int64_t)this->MultiHeadQ;
        }

        return multiSeqOffset;
    }

    TEMPLATE_LAYOUT
   __aicore__ inline TYPENAME_BSH_VOID LoopSOuterOffsetInit(int64_t seqListOffsetSize, int sIdx) {

        int64_t queryLeftpaddingSize = 0; // GetQueryLeftPaddingSize(sIdx);
        int64_t kvLeftPaddingSize = 0; // GetKVLeftPaddingSize(sIdx);

        tensorACoreOffset = (int64_t)seqListOffsetSize +
                            (int64_t)this->tailParams->sOuterOffset * (int64_t)MultiHeadQ +
                            (int64_t)this->tailParams->batchNOffset * (int64_t)this->headSize;
        int64_t seqInnerOffsetSize;
        if (this->tilingData->promptAttentionBaseParams.isKVHasLeftPadding) {
            seqInnerOffsetSize = ((int64_t)sIdx * (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                                kvLeftPaddingSize) * (int64_t)MultiHeadKV;
        } else if (this->isKvContinuous == 1) {
            // This is the offset required from the GM of KV to the starting address of each batch.
            // Each batch needs to be offset by the length of the entire previous batch.
            seqInnerOffsetSize =
                tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
                (seqListOffsetSize - (queryLeftpaddingSize * (int64_t)MultiHeadQ)) / headNumRatio : (int64_t)sIdx *
                (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize * (int64_t)MultiHeadKV;
        } else {
            // In the KV Tensorist scenario, we can directly set the GM of KV to the start address of the current batch,
            // so the offset is always 0.
            seqInnerOffsetSize = 0;
        }
        tensorBCoreOffset = (int64_t)seqInnerOffsetSize +
                        this->tailParams->batchNOffset / headNumRatio * this->headSize;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_VOID LoopSOuterOffsetInit(int64_t seqListOffsetSize, int sIdx) {
        uint64_t head_stride_q = this->headSize *
                                tilingData->promptAttentionBaseParams.seqSize;
        uint32_t head_stride_kv;
        if (this->isKvContinuous == 1) {
            head_stride_kv = this->headSize *  tilingData->promptAttentionBaseParams.seqInnerSize;
        } else {
            head_stride_kv = this->headSize * s2InCurrentBatch;
        }
        uint32_t seq_stride = this->headSize;

        int64_t queryLeftpaddingSize = 0; // GetQueryLeftPaddingSize(sIdx);
        int64_t kvLeftPaddingSize = 0; // GetKVLeftPaddingSize(sIdx);
        tensorACoreOffset = (int64_t)seqListOffsetSize + \
            (int64_t)this->tailParams->batchNOffset * (int64_t)head_stride_q + \
            (int64_t)this->tailParams->sOuterOffset * (int64_t)seq_stride;
        int64_t seqInnerOffsetSize;
        if (this->tilingData->promptAttentionBaseParams.isKVHasLeftPadding) {
            seqInnerOffsetSize = (int64_t)sIdx * (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize *
                (int64_t)MultiHeadKV + (int64_t)kvLeftPaddingSize * (int64_t)this->headSize;
        } else if (this->isKvContinuous == 1) {
            seqInnerOffsetSize =
                tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
                (seqListOffsetSize - (queryLeftpaddingSize * (int64_t)seq_stride)) / (int64_t)headNumRatio :
                (int64_t)sIdx * (int64_t)head_stride_kv * (int64_t)tilingData->promptAttentionBaseParams.headNumSize /
                (int64_t)headNumRatio;
        } else {
            seqInnerOffsetSize = 0;
        }
        tensorBCoreOffset = (int64_t)seqInnerOffsetSize + \
            (int64_t)this->tailParams->batchNOffset / (int64_t)headNumRatio * (int64_t)head_stride_kv;
        this->tailParams->attentionOutOffset = (int64_t)seqListOffsetSize + \
            (int64_t)this->tailParams->batchNOffset * (int64_t)head_stride_q +
            (int64_t)this->tailParams->sOuterOffset * (int64_t)seq_stride;
    }

    __aicore__ inline void InitTensorSize();

    __aicore__ inline void GetSingleCoreParam(int sIdx);

    __aicore__ inline void InitUb()
    {
        // ! 128 * 4 * 8 * 2   = 8k
        BSA_InitQueue(softmaxOutQueue, 1, 2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize * sizeof(float));
        //! 128 * 4 * 8  = 4k
        BSA_InitBuffer(softmaxExpBuf, tilingData->promptAttentionTensorSizeRect.softmaxExpSize * sizeof(computeType));
        softmaxExpUb = softmaxExpBuf.Get<computeType>();

        auto bmm2ByteSize = tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType);
        // 2: sizeof(computeType)
        tmpBuffSize = sizeof(computeType) == 2 ? BSA_TEMP_BUFFER_SIZE_BYTE : BSA_TEMP_BUFFER_SIZE_BYTE / 2;
        BSA_InitBuffer(tmpBuff, tmpBuffSize);   // ! one 64 one 32
        BSA_InitQueue(Bmm1Queue, 2, 32*1024); // ! 16 * 1024 * 2 * 2 = 64k
        BSA_InitBuffer(preBmm2Buf, bmm2ByteSize);
        BSA_InitBuffer(causalBuf, this->sparseBlockSize * this->sparseBlockSize / 8); // ! 128*128 / 8 = 2k
        BSA_InitBuffer(sparseIdBuf, sizeof(int32_t) * 1024);  // !4k tot 18k
        BSA_InitBuffer(taskBuf, sizeof(int32_t) * 1024);  // !4k tot 22k

        blockIdUb = sparseIdBuf.Get<int32_t>();
        taskQid = taskBuf.Get<int16_t>();
        taskCost = taskQid[1024].template ReinterpretCast<half>(); // 1024?

        BSA_InitBuffer(PABmm1UB, 64);  // dcci refresh 64B
        BSA_InitBuffer(PABmm2UB, 64);  // dcci refresh 64B
        BSA_InitBuffer(loadBuffer, 64 * sizeof(float) * 3);
        loads = loadBuffer.Get<float>();
        cores = loads[64].template ReinterpretCast<int32_t>(); // 64?
    }
};

template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910Base<BSAT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                        __gm__ uint8_t* value, __gm__ uint8_t* sparseMask,  __gm__ uint8_t* sparseCount,
                                        __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                        __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
                                        __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* keySharedPrefix,
                                        __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* workspace,
                                        const BlockSparseAttentionTilingData* __restrict tiling,
                                        __gm__ uint8_t* gmTiling, TPipe* tPipe) {
    tmp_block_idx = GetBlockIdx();

    // init global buffer
    tilingData = tiling;
    key_ptr = key;
    value_ptr = value;

    // For small B*N, perform skip core optimization
    if constexpr (BSAT::MM_TYPE != MatMulType::MM_IBSHARE_NORM) {
        if (tilingData->promptAttentionSingleCoreParams.actualCoreNums <= (GetBlockNum() * GetTaskRation() / 2 + 1)) {
            if (tmp_block_idx & 0x1) {
                tmp_block_idx = (tmp_block_idx + GetBlockNum() * GetTaskRation()) / 2;
            } else {
                tmp_block_idx = tmp_block_idx / 2;
            }
        }
    }
    InitTensorSize();

    pipe = tPipe;
        
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    sparseMaskGm.SetGlobalBuffer((__gm__ int8_t*)sparseMask);
    sparseBlockCountGm.SetGlobalBuffer((__gm__  int32_t*)sparseCount);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    workspaceGm.SetGlobalBuffer((__gm__ mmOutputType*)(workspace));
    currentKey = key_ptr;
    currentValue = value_ptr;
    keyGm.SetGlobalBuffer((__gm__ KV_T*)currentKey);
    valueGm.SetGlobalBuffer((__gm__ KV_T*)currentValue);
    
    if constexpr (USE_BLOCK_SPARE) {
        mm.SetUserDefInfo(reinterpret_cast<uint64_t>(gmTiling));
        bmm2.SetUserDefInfo(reinterpret_cast<uint64_t>(gmTiling));
    }

    isActualLenDimsNull = true;
    isActualLenDimsKVNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths,
        tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsNull = false;
    }
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV,
        tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsKVNull = false;
    }

    InitUb();
    // Use queue prefetching parameters. Enqueue a new calculation parameter each time when calculating the outer tail.
    // The head parameter of the queue is used for calculation. After the calculation, the queue head is dequeued.
    tailId = 0;
    headId = 0;
    queSize = 0;
    tailParams = &bsaParamsQueue[tailId];
    headParams = &bsaParamsQueue[headId];
    preHeadParams = &bsaParamsQueue[headId];
    isGlobalFirstCompute = true;
    mm1SingleCoreNPrev = 0;
    mm2MStridePrev = 0;
    mm2KaStridePrev = 0;
    tailParams->gmPingpong = 0;
}


template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910Base<BSAT>::InitTensorSize() {
    const PromptAttentionSingleCoreTensorSize* tensorSizeTiling = &tilingData->promptAttentionTensorSizeRect;
    this->softmaxFlashTilingData = tilingData->softmaxFlashTilingDataRect;
    this->sparseMaskS2Size = tilingData->promptAttentionBaseParams.sparseMaskS2;
    this->sparseMaskS1Size = tilingData->promptAttentionBaseParams.sparseMaskS1;
    this->sparseBlockSize = tilingData->promptAttentionBaseParams.sparseSize;
    this->headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio; // gqa
    this->headNumSize = tilingData->promptAttentionBaseParams.headNumSize; // qheadnum
    this->headSize = tilingData->promptAttentionBaseParams.headSize; // headdim
    this->isKvContinuous = tilingData->promptAttentionBaseParams.isKvContinuous;
    this->causal = tilingData->promptAttentionBaseParams.causal > 0 ? true : false;
    this->batchSize = tilingData->promptAttentionBaseParams.batchSize;

    mmResUbSize = tensorSizeTiling->mmResUbSize;
    softmaxMaxSize = tensorSizeTiling->softmaxMaxSize;
    softmaxSumSize = tensorSizeTiling->softmaxSumSize;
    softmaxExpSize = tensorSizeTiling->softmaxExpSize;
    spmTmpSize = tensorSizeTiling->spmTmpSize;
    scmTmpSize = tensorSizeTiling->scmTmpSize;
    bmm2ResUbSize = tensorSizeTiling->bmm2ResUbSize;
    this->sparseMaskTotalSize = batchSize * this->headNumSize * this->sparseMaskS2Size * this->sparseMaskS1Size;
}


template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910Base<BSAT>::Bmm2UpdateDivNoTail(
    LocalTensor<computeType>& bmm2ResPreUb, LocalTensor<float>& softmaxSumUb) {
    BSAComputeParam *params = this->preHeadParams;
    int32_t headLoop = (this->headSize + FP32_ONE_BLOCK_SIZE_BSA - 1) / FP32_ONE_BLOCK_SIZE_BSA;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(float);

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = headLoop;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = headLoop;
    int32_t loop = this->headSize / REPEAT_DATA_NUM;
    int32_t remain = this->headSize % REPEAT_DATA_NUM;
    LocalTensor<float> bmm2DivUb;
    if constexpr (!IsSameType<computeType, float>::value) {
        bmm2DivUb = mmResUb[0].template ReinterpretCast<float>(); // need 64kb
        Cast(bmm2DivUb, bmm2ResPreUb, RoundMode::CAST_NONE,  params->singleProcessSOuterSize * this->headSize);
        pipe_barrier(PIPE_V);
    } else {
        bmm2DivUb = bmm2ResPreUb;
    }
    for (int i = 0; i < loop; i++) {
        Div(bmm2DivUb[i * REPEAT_DATA_NUM], bmm2DivUb[i * REPEAT_DATA_NUM], softmaxSumUb, REPEAT_DATA_NUM,
            params->singleProcessSOuterSize, repeatParams);
    }
    if (remain) {
        Div(bmm2DivUb[loop * REPEAT_DATA_NUM], bmm2DivUb[loop * REPEAT_DATA_NUM], softmaxSumUb,
                remain, params->singleProcessSOuterSize, repeatParams);
    }
    if constexpr (!IsSameType<computeType, float>::value) {
        pipe_barrier(PIPE_V);
        Cast(bmm2ResPreUb, bmm2DivUb, RoundMode::CAST_NONE,  params->singleProcessSOuterSize * this->headSize);
    }
}


template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910Base<BSAT>::UpdateVmul(LocalTensor<computeType>& softmaxExpUb) {
    LocalTensor<computeType> bmm2ResPreUb = preBmm2Buf.Get<computeType>(bmm2ResUbSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = (
        this->headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;
    repeatParams.dstRepStride = (
        this->headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;

    //! only support singleProcessSOuterSize <=255, headsize 32B align
    const int32_t numOneRep = 256 / sizeof(computeType);
    int32_t loop = this->headSize / numOneRep;
    int32_t remain =  this->headSize % numOneRep;

    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResPreUb[i * numOneRep], softmaxExpUb, bmm2ResPreUb[i * numOneRep],
            numOneRep, this->headParams->singleProcessSOuterSize, repeatParams);
    }
    if (remain) {
        Mul(bmm2ResPreUb[loop * numOneRep], softmaxExpUb, bmm2ResPreUb[loop * numOneRep],
            remain, this->headParams->singleProcessSOuterSize, repeatParams);
    }
}


template<typename BSAT>
__aicore__ inline void BlockSparseAttentionS1s2Bns1X910Base<BSAT>::GetSingleCoreParam(int sIdx) {
        int64_t actualSeqLengthPerBatch = 0;
        int64_t actualSeqLengthKVPerBatch = 0;
        if (isActualLenDimsNull) {
            actualSeqLengthPerBatch = tilingData->promptAttentionBaseParams.seqSize;
        } else {
            actualSeqLengthPerBatch =
                (tilingData->promptAttentionBaseParams.actualSeqLengthsSize == 1) ?
                actualSeqLengthsGm.GetValue(0) : actualSeqLengthsGm.GetValue(sIdx);
        }
        if (isActualLenDimsKVNull) {
            actualSeqLengthKVPerBatch = tilingData->promptAttentionBaseParams.seqInnerSize;
        } else {
            actualSeqLengthKVPerBatch =
                (tilingData->promptAttentionBaseParams.actualSeqLengthsKVSize == 1) ?
                actualSeqLengthsKVGm.GetValue(0) : actualSeqLengthsKVGm.GetValue(sIdx);
        }

        this->tailParams->singleProcessSInnerSize = tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
        singleProcessSOuterSizeWhole = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
        MultiHeadQ = this->headSize * tilingData->promptAttentionBaseParams.headNumSize; // qH
        MultiHeadKV = MultiHeadQ / headNumRatio; // kvH
        singleProcessSOuterSizeTail =
            (actualSeqLengthPerBatch % singleProcessSOuterSizeWhole != 0) ?
            actualSeqLengthPerBatch % singleProcessSOuterSizeWhole : singleProcessSOuterSizeWhole;

        this->tailParams->actualSeqLengthPerBatch = actualSeqLengthPerBatch;
        this->tailParams->actualSeqLengthKVPerBatch = actualSeqLengthKVPerBatch;
    }


#endif  // PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
