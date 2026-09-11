









/*!
 * \file kv_quant_sparse_attn_sharedkv_scfa_block_cube.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#include "common/offset_calculator.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

template <SAS_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SAS_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class SCFABlockCube {
public:

    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline SCFABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);
    __aicore__ inline void CalcS1Coord(RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm1SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    TPipe *tPipe;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT> queryGm;


    CubeCoordInfo coordInfo[3];
    TEventID mte1ToMte2Id[3];
    TEventID mte2ToMte1Id[3];

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;


    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;
    BuffersPolicy3buff<BufferType::L1> l1KBuffers;



    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, constInfo);
        if constexpr (IS_SPLIT_G) {
            mte1ToMte2Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte1ToMte2Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte1ToMte2Id[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte2ToMte1Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            mte2ToMte1Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            mte2ToMte1Id[2] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitLocalBuffer()
{
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(Q_T);
    l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
    l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);


    l0aBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0bBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0cBufferManager.Init(tPipe, L0C_SHARED_SIZE_256K);

    mmL0ABuffers.Init(l0aBufferManager, BUFFER_SIZE_16K);
    mmL0BBuffers.Init(l0bBufferManager, BUFFER_SIZE_32K);
    mmL0CBuffers.Init(l0cBufferManager, BUFFER_SIZE_128K);
}


TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // SAS_LAYOUT::TND
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            actualSeqQLen, constInfo.actualSeqLenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CalcS1Coord(RunInfo &runInfo,
    ConstInfo &constInfo)
{
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * runInfo.qSNumInOneBlock;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    CalcS1Coord(runInfo, constInfo);

    IterateBmm1SCFA(outputBuf, inputRightBuf, v0ResGm, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    IterateBmm2SCFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;


    if (unlikely(runInfo.s2LoopCount == 0)) {
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();

        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[gmOffset], runInfo.mRealSize, constInfo.dSize,
            constInfo.mm1Ka);

        inputLeftBuf.Set<HardEvent::MTE2_MTE1>();
    } else {
        inputLeftBuf = l1QBuffers.GetPre();

        inputLeftBuf.Set<HardEvent::MTE2_MTE1>();
    }


    inputRightBuf.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);

        LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
        v0ResGm.WaitCrossCore();
        GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
        DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * constInfo.dSize);
        SetFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
        WaitFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
    }

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>();
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>();
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                     static_cast<uint32_t>(constInfo.dSize),   // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        inputLeftBuf.Set<HardEvent::MTE1_MTE2>();
    }

    mm1ResL0C.Set<HardEvent::M_FIX>();
    mm1ResL0C.Wait<HardEvent::M_FIX>();

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams);
    mm1ResL0C.Set<HardEvent::FIX_M>();
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> inputLeftBuf = inputLeftBuffers.Get();
    inputLeftBuf.WaitCrossCore();

    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>();
    MMParam param = {static_cast<uint32_t>(s1BaseSize),
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(),
        param);

    mm2ResL0C.Set<HardEvent::M_FIX>();
    mm2ResL0C.Wait<HardEvent::M_FIX>();
    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    fixpipeParams.mSize = s1BaseSize;
    fixpipeParams.srcStride = Align16Func(s1BaseSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), fixpipeParams);
    mm2ResL0C.Set<HardEvent::FIX_M>();
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class SCFABlockCubeDummy {
public:
    __aicore__ inline SCFABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo) {}
};

template <typename T>
struct CubeBlockTraits;

#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCubeDummy);


#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
