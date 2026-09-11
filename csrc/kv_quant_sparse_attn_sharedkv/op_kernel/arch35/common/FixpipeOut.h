









/*!
 * \file FixpipeOut.h
 * \brief
 */
#ifndef FIXPIPEOUT_H
#define FIXPIPEOUT_H

constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_GM = {CO2Layout::ROW_MAJOR, false};

struct fixpipeOutParams {
    uint32_t fixpOutMSize;
    uint32_t fixpOutNSize;
};

template<typename mmOutputType, typename computeType, typename l0cType>
__aicore__ inline void FixpipeMmCopyOutToUB(LocalTensor<mmOutputType>& mmResUb, LocalTensor<l0cType>& L0CTensor, const fixpipeOutParams& fixpOutParam)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> L0C2UbFixpParams; // L0C->UB
    L0C2UbFixpParams.nSize = (fixpOutParam.fixpOutNSize + 7) >> 3 << 3;
    L0C2UbFixpParams.srcStride = ((L0C2UbFixpParams.mSize + 15) >> 4) << 4;
    L0C2UbFixpParams.dstStride = (L0C2UbFixpParams.nSize + 15) >> 4 << 4;
    L0C2UbFixpParams.params.srcNdStride = 0;
    L0C2UbFixpParams.params.dstNdStride = 0;
    Fixpipe<mmOutputType, computeType, PFA_CFG_ROW_MAJOR_UB>(mmResUb, L0CTensor, L0C2UbFixpParams);
}

template<typename mmOutputType, typename computeType, typename l0cType>
__aicore__ inline void FixpipeMmCopyOutToGm(GlobalTensor<mmOutputType>& mmResGm,LocalTensor<l0cType>& L0CTensor, const fixpipeOutParams& fixpOutParam)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> L0C2GmFixpParams; // L0C->Gm
    L0C2GmFixpParams.nSize = (fixpOutParam.fixpOutNSize + 7) >> 3 << 3;
    L0C2GmFixpParams.mSize = (fixpOutParam.fixpOutMSize + 1) >> 1 << 1;
    L0C2GmFixpParams.srcStride = ((L0C2GmFixpParams.mSize + 15) >> 4) << 4;
    L0C2GmFixpParams.dualDstCtl = 1;
    L0C2GmFixpParams.params.ndNum = 1;
    L0C2GmFixpParams.params.srcNdStride = 0;
    L0C2GmFixpParams.params.dstNdStride = 0;
    Fixpipe<mmOutputType, computeType, PFA_CFG_ROW_MAJOR_GM>(mmResGm, L0CTensor, L0C2GmFixpParams);
}
#endif
