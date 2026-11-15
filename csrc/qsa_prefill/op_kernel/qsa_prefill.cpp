#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#if (__CCE_AICORE__ == 220)
#include "policies.hpp"
#include "block_qk.hpp"
#include "block_pv.hpp"
#include "rescale_o.hpp"
#include "qsa_prefill.hpp"
using namespace NpuArch;
using namespace SasaKernelArch22;
template <class InDtype, class SMDtype>
__aicore__ inline void QsaPrefillInterface(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR selectIdx, GM_ADDR blockTable,
                                           GM_ADDR selectNumIdx, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                           GM_ADDR o, GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = float;
    using ElementOTmp = SMDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    // QK matmul
    using L1TileShapeQK = GemmShape<128, 256, 256>;
    using L0TileShapeQK = GemmShape<32, 256, 64>;
    using DispatchPolicyQK = Gemm::QsaQK<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    // Online softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<Epilogue::LseMode::OUT_ONLY, SMDtype>;
    using MaskType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    // PV matmul
    using L1TileShapePV = GemmShape<32, 256, 256>;
    using L0TileShapePV = GemmShape<32, 256, 64>;
    using DispatchPolicyPV = Gemm::QsaPV<false, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    // Rescale O
    using DispatchPolicyRescaleO = Epilogue::QsaRescaleO<Epilogue::LseMode::OUT_ONLY, SMDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpUpdateType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OTmpUpdateType, LseType>;

    using SasaKernel = QsaPrefillKernel<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO>;

    SasaKernelParamsArch22 params{
        q, k, v, selectIdx, blockTable, selectNumIdx, actualQseqlen, actualKvseqlen, o, softmaxLse, workspace, tiling};
    SasaKernel sasaKernel;
    sasaKernel(params);
}

#endif
extern "C" __global__ __aicore__ void qsa_prefill_runs_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR slots,
                                                              GM_ADDR runs, GM_ADDR counts, GM_ADDR out, GM_ADDR lse,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (__CCE_AICORE__ == 220)
    QsaPrefillInterface<bfloat16_t, float>(q, k, v, slots, runs, counts, nullptr, nullptr, out, lse,
                                           AscendC::GetUserWorkspace(workspace), tiling);
#endif
}
