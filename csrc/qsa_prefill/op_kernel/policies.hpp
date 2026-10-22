#pragma once
#include "../../sparse_attention_score/op_kernel/arch22/kernel_utils.hpp"
namespace NpuArch::Gemm {
template <bool P = false, bool U = false>
struct QsaQK : MmadAtlasA2SFAIQK<P, U> {};
template <bool P = false, bool U = false>
struct QsaPV : MmadAtlasA2SFAIPV<P, U> {};
}  // namespace NpuArch::Gemm
namespace NpuArch::Epilogue {
template <LseMode M, class T>
struct QsaRescaleO : EpilogueAtlasA2RescaleO<M, T> {};
}  // namespace NpuArch::Epilogue
